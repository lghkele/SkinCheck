"""引擎级行为回归：亮区／细边、局部文字、忽略／允许配置与模块故障隔离。

涉及嵌套问题时使用 walk_issues；只检查 result.issues 会漏掉背景里的文字子项。
"""
import copy
import json

import cv2
import numpy as np
import pytest

from skinscan.config import DEFAULTS, load_config, page_config
from skinscan.engine import analyze_image
from skinscan.image import contrast_ratio, write_png
from skinscan.models import walk_issues


def issues(result, kind):
    return [issue for issue in walk_issues(result.issues) if issue.type == kind]


def test_wcag_uses_linear_srgb():
    assert contrast_ratio([0, 0, 0], [255, 255, 255]) == pytest.approx(21)
    assert contrast_ratio([23, 23, 23], [37, 37, 40]) == pytest.approx(1.177, abs=0.01)
    assert contrast_ratio([181, 181, 186], [245, 245, 245]) == pytest.approx(1.865, abs=0.03)


def test_white_card(screenshot):
    path, _ = screenshot(card=(50, 180, 550, 430, 245))
    result, frame = analyze_image(path)
    assert (frame.width, frame.height) == (600, 1000)
    found = issues(result, "BRIGHT_BACKGROUND")
    assert found and found[0].severity == "critical"
    assert found[0].bounds == [50, 180, 550, 430]
    assert found[0].metrics["regionLuminance"] == pytest.approx(245/255, abs=0.001)
    assert result.status == "failed"
    assert result.modules["text"]["status"] == "ok"


def test_locally_light_card_independent_of_absolute_threshold(screenshot):
    path, _ = screenshot(background=15, card=(50, 180, 550, 430, 115))
    result, _ = analyze_image(path)
    found = issues(result, "BRIGHT_BACKGROUND")
    assert found and found[0].metrics["brightPixelRatio"] == 0
    assert "local_surrounding_contrast" in found[0].rules


def test_local_contrast_when_page_median_is_light(screenshot):
    path, image = screenshot(background=150)
    image[150:500, 30:570] = 15
    image[230:380, 100:500] = 125  # lower than the page median, lighter than its immediate surroundings
    write_png(path, image)
    result, _ = analyze_image(path)
    assert any(i.bounds == [100, 230, 500, 380] for i in issues(result, "BRIGHT_BACKGROUND"))


def test_whole_light_page(screenshot):
    path, _ = screenshot(background=245, text="NORMAL", foreground=40)
    result, _ = analyze_image(path)
    assert issues(result, "PAGE_NOT_DARK")
    assert result.status == "failed"


@pytest.mark.parametrize("background,foreground", [(37, 23), (245, 181), (220, 248)])
def test_both_text_polarities_and_contrast(screenshot, background, foreground):
    path, _ = screenshot(background=background, text="Dark mode 2026", foreground=foreground)
    config = load_config()
    config["text"]["detection_mode"] = "all_low_contrast"
    result, _ = analyze_image(path, config)
    found = issues(result, "LOW_CONTRAST_TEXT")
    assert found, json.dumps(result.to_dict(), ensure_ascii=False)
    expected = contrast_ratio([foreground]*3, [background]*3)
    assert any(i.metrics["contrastRatio"] == pytest.approx(expected, abs=0.08) for i in found)
    assert any(i.severity == "critical" and i.confidence >= 0.8 for i in found)
    if background == 37:
        assert not issues(result, "BRIGHT_BACKGROUND")
    assert result.status == "failed"


@pytest.mark.parametrize("text", ["7", "OK", "A"])
def test_short_text_is_reviewed_not_lost(screenshot, text):
    path, _ = screenshot(background=37, text=text, foreground=23)
    result, _ = analyze_image(path)
    found = issues(result, "LOW_CONTRAST_TEXT")
    assert found
    assert any("short_text_candidate" in i.rules for i in found)
    assert all(i.requiresReview for i in found)
    assert result.status == "uncertain"


def test_normal_white_text_not_background_or_low_contrast(screenshot):
    path, _ = screenshot(text="Healthy dark UI 2026", foreground=240)
    result, _ = analyze_image(path)
    assert not result.issues, json.dumps(result.to_dict(), ensure_ascii=False)
    assert result.status == "passed"


@pytest.mark.parametrize("closed", [True, False])
def test_single_pixel_ring_and_three_sides(screenshot, closed):
    path, image = screenshot()
    cv2.rectangle(image, (60, 200), (540, 500), (240, 240, 240), 1)
    if not closed:
        image[500, 60:541] = 25
    write_png(path, image)
    result, _ = analyze_image(path)
    found = issues(result, "SUSPICIOUS_LIGHT_EDGE")
    assert found
    assert found[0].metrics["areaRatio"] < DEFAULTS["background"]["min_area_ratio"] * 2
    assert all(i.requiresReview and i.confidence < 0.8 for i in found)


def test_ignored_regions_do_not_influence_statistics_or_candidates(screenshot):
    path, _ = screenshot(card=(0, 0, 600, 430, 245))
    config = load_config()
    config["global"]["ignored_regions"] = [[0, 0, 600, 450]]
    result, _ = analyze_image(path, config)
    assert result.status == "passed"
    assert result.statistics["brightPixelRatio"] == 0
    assert result.statistics["ignoredPixelCount"] == 270000


def test_all_ignored_is_not_a_pass(screenshot):
    path, _ = screenshot()
    config = load_config()
    config["global"]["ignored_regions"] = [[0, 0, 600, 1000]]
    result, _ = analyze_image(path, config)
    assert result.status == "ignored"


def test_color_whitelist_reduces_risk_without_hiding(screenshot):
    path, _ = screenshot(card=(50, 180, 550, 430, 245))
    config = load_config()
    config["global"]["allowed_colors"] = ["#F5F5F5"]
    result, _ = analyze_image(path, config)
    found = issues(result, "BRIGHT_BACKGROUND")
    assert found[0].whitelist == ["allowed_color:#F5F5F5"]
    assert result.status == "uncertain"


def test_text_issue_nested_under_card(screenshot):
    path, image = screenshot(card=(50, 180, 550, 430, 245))
    cv2.putText(image, "Low contrast", (80, 350), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (181, 181, 181), 3, cv2.LINE_AA)
    write_png(path, image)
    config = load_config()
    config["text"]["detection_mode"] = "all_low_contrast"
    result, _ = analyze_image(path, config)
    assert any(i.type == "BRIGHT_BACKGROUND" and any(child.type == "LOW_CONTRAST_TEXT" for child in i.children)
               for i in result.issues)


@pytest.mark.parametrize("broken,remaining", [("detect_background", "text"), ("detect_text", "background")])
def test_failed_chain_does_not_skip_other(screenshot, monkeypatch, broken, remaining):
    path, _ = screenshot(background=37, text="Dark mode", foreground=23, card=(50, 500, 550, 700, 245))
    def fail(*args):
        raise RuntimeError("simulated module failure")
    monkeypatch.setattr("skinscan.engine." + broken, fail)
    result, _ = analyze_image(path)
    assert result.modules[remaining]["status"] == "ok"
    assert result.errors
    assert result.reviewStatus == "needs_review"
    assert result.status != "passed"
    assert result.issues


def test_noise_image_does_not_trigger_high_confidence_failure(screenshot):
    # 只要求噪声不形成高置信度阻断，不要求规则对所有纹理都绝对零候选。
    path, image = screenshot()
    random = np.random.default_rng(42)
    image[150:600, 50:550] = random.integers(0, 256, (450, 500, 3), dtype=np.uint8)
    write_png(path, image)
    result, _ = analyze_image(path)
    assert result.status != "failed"


def test_gradient_text_is_uncertain(screenshot):
    # 渐变使同一行背景不一致；局部能测到的部分仍保留复核，不以整行平均色裁决。
    path, image = screenshot()
    for x in range(600):
        image[:, x] = 25 + x // 4
    cv2.putText(image, "Gradient text", (80, 350), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (70, 70, 70), 3, cv2.LINE_AA)
    write_png(path, image)
    result, _ = analyze_image(path)
    assert any(i.requiresReview for i in issues(result, "LOW_CONTRAST_TEXT"))
    assert result.status != "passed"


def test_cropped_card_is_retained(screenshot):
    path, _ = screenshot(card=(0, 180, 500, 430, 245))
    result, _ = analyze_image(path)
    assert issues(result, "BRIGHT_BACKGROUND")
    assert result.status == "failed"


def test_single_cjk_shaped_glyph_is_retained(screenshot):
    path, image = screenshot(background=37)
    # The visible geometry of the single character 口; content is never recognized.
    cv2.rectangle(image, (100, 300), (130, 330), (23,)*3, 3, cv2.LINE_AA)
    write_png(path, image)
    result, _ = analyze_image(path)
    assert issues(result, "LOW_CONTRAST_TEXT")
    assert result.status == "uncertain"


def test_allowed_color_does_not_hide_text_on_card(screenshot):
    path, image = screenshot(card=(50, 180, 550, 430, 245))
    cv2.putText(image, "Bad contrast", (80, 350), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (181,)*3, 3, cv2.LINE_AA)
    write_png(path, image)
    config = load_config()
    config["global"]["allowed_colors"] = ["#F5F5F5"]
    config["text"]["detection_mode"] = "all_low_contrast"
    result, _ = analyze_image(path, config)
    assert result.status == "failed"
    assert any(i.confidence >= 0.8 and not i.whitelist for i in issues(result, "LOW_CONTRAST_TEXT"))


def test_disabled_page_is_explicitly_ignored(screenshot):
    path, _ = screenshot(background=245)
    config = load_config()
    config["pages"]["screen"] = {"ignore_page": True}
    result, _ = analyze_image(path, config)
    assert result.status == "ignored"
    assert not result.modules


def test_invalid_input_and_metadata_are_not_passes(tmp_path, screenshot):
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"not a png")
    result, _ = analyze_image(bad)
    assert result.status == "error"
    path, _ = screenshot()
    result, _ = analyze_image(path, metadata={"width": 1080})
    assert result.status == "error"


def test_config_page_overrides_and_validation(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("global:\n  ignored_regions: [[0, 0, 600, 20]]\npages:\n  home:\n    ignored_regions: [[0, 900, 600, 1000]]\n    text:\n      warning_contrast: 5.0\n")
    config = load_config(config_path)
    original = copy.deepcopy(config)
    effective = page_config(config, "home")
    assert effective["global"]["ignored_regions"] == [[0, 0, 600, 20], [0, 900, 600, 1000]]
    assert effective["text"]["warning_contrast"] == 5
    assert config == original
    config_path.write_text("background:\n  absolute_bright_threshold: 2\n")
    with pytest.raises(ValueError):
        load_config(config_path)
    config_path.write_text("text:\n  typo: 1\n")
    with pytest.raises(ValueError):
        load_config(config_path)
