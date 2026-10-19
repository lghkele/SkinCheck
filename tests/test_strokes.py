"""局部配对与范围门控回归：颜色必须来自同一截面，错误背景不能靠平均蒙混通过。

既通过真实候选流水线验证告警，也用已知标签隔离采样几何，区分定位和测量失败。
"""
import cv2
import numpy as np
import pytest

from skinscan.config import load_config
from skinscan.engine import analyze_image
from skinscan.image import contrast_ratio, load_frame, write_png
from skinscan.models import walk_issues
from skinscan.strokes import in_detection_scope, sample_component


def text_issues(result):
    return [i for i in walk_issues(result.issues) if i.type == "LOW_CONTRAST_TEXT"]


@pytest.mark.parametrize("bg,fg,polarity,expected", [
    (37, 23, "blackhat_dark", True),
    (220, 248, "tophat_light", True),
    (245, 181, "blackhat_dark", False),
    (25, 240, "tophat_light", False),
    (245, 181, "tophat_light", False),
    (37, 23, "tophat_light", False),
])
def test_local_background_and_polarity_gate(bg, fg, polarity, expected):
    c = load_config()["text"]
    assert in_detection_scope([fg]*3, [bg]*3, polarity, c) is expected


def test_default_white_card_has_background_but_no_text_warning(screenshot):
    path, _ = screenshot(card=(50, 180, 550, 430, 245), text="Low contrast card", foreground=181)
    result, _ = analyze_image(path)
    assert any(i.type == "BRIGHT_BACKGROUND" for i in result.issues)
    assert not text_issues(result)
    assert result.status == "failed"


def test_all_contrast_mode_retains_gray_on_white_and_paired_measurements(screenshot):
    path, _ = screenshot(card=(50, 180, 550, 430, 245), text="Low contrast card", foreground=181)
    config = load_config()
    config["text"]["detection_mode"] = "all_low_contrast"
    result, _ = analyze_image(path, config)
    found = text_issues(result)
    assert found
    assert any(i.metrics["measurementMethod"] == "paired_strokes" for i in found)
    for issue in found:
        if issue.metrics["measurementMethod"] != "paired_strokes":
            continue
        assert issue.metrics["contrastRatio"] == pytest.approx(1.881, abs=0.08)
        assert issue.metrics["strokePairs"]
        for pair in issue.metrics["strokePairs"]:
            assert pair["contrastRatio"] == pytest.approx(
                contrast_ratio(pair["foregroundRgb"], pair["backgroundRgb"]), abs=0.001)


def test_mixed_foregrounds_do_not_hide_weak_word(screenshot):
    path, image = screenshot(background=245)
    cv2.putText(image, "GOOD", (40, 350), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (30,)*3, 3, cv2.LINE_AA)
    weak_x = 190
    cv2.putText(image, "weak", (weak_x, 350), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (181,)*3, 3, cv2.LINE_AA)
    write_png(path, image)
    config = load_config()
    config["text"]["detection_mode"] = "all_low_contrast"
    result, _ = analyze_image(path, config)
    found = text_issues(result)
    assert any(i.metrics["measurementMethod"] == "paired_strokes" and
               i.bounds[0] >= weak_x-3 and i.metrics["contrastRatio"] < 2 for i in found)
    assert all(i.bounds[0] >= weak_x-3 for i in found)


def test_local_dark_background_not_page_median_controls_detection(screenshot):
    path, image = screenshot(background=245)
    image[:, :245] = 37
    cv2.putText(image, "TEST TEST", (30, 350), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (23,)*3, 3, cv2.LINE_AA)
    write_png(path, image)
    result, _ = analyze_image(path)
    assert result.statistics["medianLuminance"] > 0.9
    found = text_issues(result)
    assert found
    assert any(i.metrics["measurementMethod"] == "paired_strokes" and
               i.metrics["backgroundRgb"] == [37]*3 and i.metrics["contrastRatio"] < 1.3 for i in found)
    assert all(i.bounds[0] < 245 for i in found)


def test_two_different_sides_are_rejected_instead_of_blended(tmp_path):
    image = np.full((70, 70, 3), 37, np.uint8)
    image[:, 34:] = 245
    image[12:58, 30:34] = 23
    labels = np.zeros((70, 70), np.int32)
    labels[12:58, 30:34] = 1
    path = tmp_path / "border.png"
    write_png(path, image)
    frame = load_frame(path, [])
    result = sample_component(frame, labels, (30, 12, 4, 46, 1), "blackhat_dark", load_config()["text"])
    assert result.attempted > 0
    assert result.rejected_background > 0
    assert not result.pairs


@pytest.mark.parametrize("bg,fg,polarity", [(37, 23, "blackhat_dark"), (220, 248, "tophat_light")])
def test_opposing_edges_measure_core_with_antialiasing(tmp_path, bg, fg, polarity):
    image = np.full((90, 90, 3), bg, np.uint8)
    cv2.line(image, (40, 15), (40, 75), (fg,)*3, 3, cv2.LINE_AA)
    # The geometry fixture supplies the candidate mask, while colors come only from PNG pixels.
    labels = (np.abs(image[:, :, 0].astype(int)-bg) >= 2).astype(np.int32)
    ys, xs = np.nonzero(labels)
    path = tmp_path / "stroke.png"
    write_png(path, image)
    frame = load_frame(path, [])
    component = (int(xs.min()), int(ys.min()), int(xs.max()-xs.min()+1), int(ys.max()-ys.min()+1), 1)
    sampled = sample_component(frame, labels, component, polarity, load_config()["text"])
    assert len(sampled.pairs) >= 3
    assert np.median([pair.contrast for pair in sampled.pairs]) == pytest.approx(contrast_ratio([fg]*3, [bg]*3), abs=0.06)
    assert all(pair.background.astype(int).tolist() == [bg]*3 for pair in sampled.pairs)


def test_unknown_stroke_measurement_cannot_be_confirmed(screenshot):
    # 故意使最小宽度不可满足：保留强文字结构未知项，不恢复整行估色对比度。
    path, _ = screenshot(background=37, text="Weak dark text", foreground=23)
    config = load_config()
    config["text"]["stroke_sampling"]["opposite_gradient_cosine"] = 1.0
    config["text"]["stroke_sampling"]["min_width"] = 40.0
    result, _ = analyze_image(path, config)
    assert not text_issues(result)
    found = [i for i in walk_issues(result.issues) if i.type == "UNREADABLE_TEXT_CANDIDATE"]
    assert found and all(i.metrics.get("measurementMethod") == "unmeasured_text_structure" for i in found)
    assert all(i.requiresReview and "contrastRatio" not in i.metrics for i in found)
    assert result.status == "uncertain"


def test_ignored_pixels_cannot_be_used_as_stroke_edges(tmp_path):
    image = np.full((70, 70, 3), 37, np.uint8)
    image[12:58, 30:34] = 23
    labels = np.zeros((70, 70), np.int32)
    labels[12:58, 30:34] = 1
    path = tmp_path / "ignored_edge.png"
    write_png(path, image)
    frame = load_frame(path, [[32, 0, 34, 70]])
    result = sample_component(frame, labels, (30, 12, 4, 46, 1), "blackhat_dark", load_config()["text"])
    assert not result.pairs


def test_page_override_can_expand_dark_background_range(screenshot):
    path, _ = screenshot(background=100, text="Dark gray text", foreground=80)
    config = load_config()
    original, _ = analyze_image(path, config)
    assert not text_issues(original)
    config["pages"]["screen"] = {"text": {"dark_background_max": 0.45}}
    updated, _ = analyze_image(path, config)
    assert any(i.metrics["measurementMethod"] == "paired_strokes" for i in text_issues(updated))


@pytest.mark.parametrize("change", [
    {"detection_mode": "typo"},
    {"dark_background_max": 0.9, "light_background_min": 0.2},
    {"stroke_sampling": {"core_fractions": []}},
    {"stroke_sampling": {"core_fractions": [0.5, float("nan")]}},
    {"stroke_sampling": {"opposite_gradient_cosine": 2.0}},
])
def test_invalid_sampling_config_rejected(tmp_path, change):
    import yaml
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text(yaml.safe_dump({"text": change}))
    with pytest.raises(ValueError):
        load_config(config_path)
