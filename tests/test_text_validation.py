"""候选验证的两面约束：拒绝字缝／圆角／纹理，同时保留真实微弱笔画和短字。

图片内弱标签测试只验证文字上下文约束；图片定位本身由 test_pictures 覆盖。
"""
import cv2
import numpy as np
import pytest

from skinscan.config import load_config
from skinscan.engine import analyze_image
from skinscan.image import load_frame, write_png
from skinscan.models import walk_issues
from skinscan.pictures import empty_pictures
from skinscan.text import detect_text


def run(tmp_path, image, picture=False, config=None):
    """picture=True 注入全图内部掩码，隔离文字规则，避免图片定位影响用例归因。"""
    path = tmp_path / 'validation.png'
    write_png(path, image)
    frame = load_frame(path, [])
    pictures = empty_pictures(frame)
    if picture:
        pictures.interior[:] = True
    return detect_text(frame, config or load_config(), pictures=pictures)


@pytest.mark.parametrize('background,foreground', [(28, 245), (245, 28)])
def test_readable_glyph_gaps_do_not_become_opposite_polarity_text(tmp_path, background, foreground):
    image = np.full((600, 600, 3), background, np.uint8)
    for y in (160, 260, 360):
        cv2.putText(image, 'GOOD BOOK 008', (35, y), cv2.FONT_HERSHEY_SIMPLEX,
                    1.3, (foreground,)*3, 3, cv2.LINE_AA)
    config = load_config()
    config['text']['detection_mode'] = 'all_low_contrast'
    issues, diagnostics = run(tmp_path, image, config=config)
    assert not issues
    assert diagnostics['candidateValidation'].get('rejected_background_gap_or_container', 0) > 0


def test_navigation_round_corners_do_not_become_text(tmp_path):
    image = np.full((900, 600, 3), 14, np.uint8)
    cv2.rectangle(image, (30, 700), (570, 899), (34,)*3, -1)
    cv2.rectangle(image, (0, 730), (599, 899), (34,)*3, -1)
    cv2.circle(image, (30, 730), 30, (34,)*3, -1)
    cv2.circle(image, (569, 730), 30, (34,)*3, -1)
    issues, _ = run(tmp_path, image)
    assert not issues


@pytest.mark.parametrize('foreground', [34, 35])
def test_same_color_gap_check_preserves_genuinely_faint_strokes(tmp_path, foreground):
    image = np.full((600, 600, 3), 37, np.uint8)
    cv2.putText(image, 'Faint text', (50, 280), cv2.FONT_HERSHEY_SIMPLEX,
                1.3, (foreground,)*3, 3, cv2.LINE_AA)
    issues, _ = run(tmp_path, image)
    assert any(i.type == 'LOW_CONTRAST_TEXT' and i.metrics['contrastRatio'] < 1.05 for i in issues)


def test_photo_texture_does_not_use_color_fallback(tmp_path):
    rng = np.random.default_rng(82)
    image = cv2.GaussianBlur(rng.integers(0, 256, (500, 600, 3), dtype=np.uint8), (5, 5), 1.2)
    image[:90] = [155, 199, 223]
    for x in range(50, 550, 70):
        cv2.ellipse(image, (x, 60), (8, 3), 0, 0, 360, (166, 206, 229), -1)
    issues, diagnostics = run(tmp_path, image, picture=True)
    assert not issues
    assert diagnostics['fallbackIssueCount'] == 0
    assert diagnostics['candidateValidation'].get('pictureUnmeasuredComponentCount', 0) > 0


@pytest.mark.parametrize('content', ['Weak label', '7', '8'])
def test_stable_overlay_text_is_measured_inside_picture(tmp_path, content):
    image = np.full((600, 600, 3), 37, np.uint8)
    cv2.putText(image, content, (70, 280), cv2.FONT_HERSHEY_SIMPLEX,
                1.4, (23,)*3, 3, cv2.LINE_AA)
    issues, _ = run(tmp_path, image, picture=True)
    assert any(i.type == 'LOW_CONTRAST_TEXT' and i.metrics['measurementMethod'] == 'paired_strokes'
               and i.metrics['contrastRatio'] < 1.3 for i in issues)


def test_unknown_strong_text_is_not_given_a_fabricated_ratio(tmp_path):
    image = np.full((600, 600, 3), 37, np.uint8)
    cv2.putText(image, 'Unmeasured text', (40, 280), cv2.FONT_HERSHEY_SIMPLEX,
                1.1, (23,)*3, 3, cv2.LINE_AA)
    config = load_config()
    config['text']['stroke_sampling']['min_width'] = 40
    issues, _ = run(tmp_path, image, config=config)
    assert issues and all(i.type == 'UNREADABLE_TEXT_CANDIDATE' for i in issues)
    assert all('contrastRatio' not in i.metrics and i.requiresReview for i in issues)
    path = tmp_path / 'validation.png'
    result, _ = analyze_image(path, config)
    assert result.status == 'uncertain'


@pytest.mark.parametrize('override', [
    {'same_color_tolerance': 0}, {'surrounding_match_min': 1.1},
    {'max_width_cv': 2}, {'picture_max_width_cv': 0}, {'retry_angle_degrees': 90},
])
def test_invalid_candidate_validation_rejected(tmp_path, override):
    import yaml
    path = tmp_path / 'config.yaml'
    path.write_text(yaml.safe_dump({'text': {'candidate_validation': override}}))
    with pytest.raises(ValueError):
        load_config(path)
