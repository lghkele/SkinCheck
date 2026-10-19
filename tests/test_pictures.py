"""图片过滤的正负对照：图片内部应排除，旁侧白卡／弱字／真实细边必须保留。

artwork 是固定种子的纹理夹具，不是真实封面分类样本，不能据此计算生产召回率。
"""
import cv2
import numpy as np
import pytest

from skinscan.config import load_config
from skinscan.engine import analyze_image
from skinscan.image import load_frame, write_png
from skinscan.models import walk_issues
from skinscan.pictures import locate_pictures


def artwork(width=150, height=240, sky=False, grayscale=False):
    """生成独立 RGB 内容，sky／grayscale 分别覆盖平坦亮区和无彩色细节的场景。"""
    rng = np.random.default_rng(271)
    pixels = rng.integers(35, 255, (height, width, 3), dtype=np.uint8)
    pixels = cv2.GaussianBlur(pixels, (3, 3), 0.6)
    # A contiguous bright patch must be considered in the context of the whole artwork.
    if sky:
        pixels[:height//4] = 245
    if grayscale:
        gray = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY)
        pixels = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    return pixels


def save(tmp_path, image, name="screen"):
    path = tmp_path / (name+".png")
    write_png(path, image)
    return path


def page():
    return np.full((900, 600, 3), 25, np.uint8)


def detect(tmp_path, image, config=None):
    path = save(tmp_path, image)
    frame = load_frame(path, (config or load_config())["global"]["ignored_regions"])
    return locate_pictures(frame, config or load_config()), frame


@pytest.mark.parametrize("sky,grayscale", [(False, False), (True, False), (False, True)])
def test_single_picture_can_be_localized_without_repeated_layout(tmp_path, sky, grayscale):
    image = page()
    image[120:360, 60:210] = artwork(sky=sky, grayscale=grayscale)
    pictures, frame = detect(tmp_path, image)
    assert pictures.interior.shape == frame.gray.shape
    assert pictures.interior[135:345, 75:195].mean() > 0.90
    assert not pictures.interior[:, 230:].any()
    assert any(r["classification"] == "picture" and r["superpixelCount"] > 0 for r in pictures.regions)
    assert pictures.diagnostics["status"] == "ok"


def test_white_sky_does_not_generate_confirmed_bright_background(tmp_path):
    image = page()
    image[120:360, 60:210] = artwork(sky=True)
    path = save(tmp_path, image)
    config = load_config()
    before = load_config()
    before["pictures"]["enabled"] = False
    old, _ = analyze_image(path, before)
    new, _ = analyze_image(path, config)
    old_backgrounds = [i for i in walk_issues(old.issues) if i.type == "BRIGHT_BACKGROUND"]
    assert old_backgrounds
    assert not new.errors
    assert new.statistics["pictureInteriorPixelCount"] > 0
    assert not [i for i in walk_issues(new.issues) if i.type == "BRIGHT_BACKGROUND" and
                i.confidence >= 0.8 and not i.requiresReview]


def test_picture_does_not_mask_neighboring_weak_text_or_white_card(tmp_path):
    image = page()
    image[80:320, 40:190] = artwork()
    image[500:760, 30:570] = 245
    image[340:470] = 37
    cv2.putText(image, "Weak nearby text", (230, 400), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (23,)*3, 2, cv2.LINE_AA)
    path = save(tmp_path, image)
    result, _ = analyze_image(path)
    assert not result.errors
    assert result.statistics["pictureInteriorPixelCount"] > 0
    assert any(i.type == "BRIGHT_BACKGROUND" and i.bounds[1] >= 490 for i in walk_issues(result.issues))
    assert any(i.type == "LOW_CONTRAST_TEXT" and 330 <= i.bounds[1] < 470 for i in walk_issues(result.issues))


def test_white_card_with_small_embedded_picture_is_not_wholly_excluded(tmp_path):
    image = page()
    image[100:800, 30:570] = 245
    image[160:400, 70:220] = artwork()
    pictures, _ = detect(tmp_path, image)
    assert pictures.interior[180:380, 90:200].mean() > 0.90
    assert not pictures.interior[450:780, 50:550].any()
    result, _ = analyze_image(save(tmp_path, image))
    assert any(i.type == "BRIGHT_BACKGROUND" and i.severity == "critical" for i in walk_issues(result.issues))


def test_dense_text_and_bright_button_are_not_pictures(tmp_path):
    image = page()
    image[70:700, 25:575] = 245
    for y in range(110, 670, 45):
        cv2.putText(image, "Words 0123 readable", (40, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (50,)*3, 2, cv2.LINE_AA)
    image[750:830, 100:500] = [255, 135, 25]
    cv2.putText(image, "CONTINUE", (180, 800), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255,)*3, 2, cv2.LINE_AA)
    pictures, _ = detect(tmp_path, image)
    assert not pictures.interior.any()


def test_picture_boundary_does_not_erase_external_single_pixel_frame(tmp_path):
    image = page()
    image[120:360, 60:210] = artwork()
    cv2.rectangle(image, (57, 117), (212, 362), (245,)*3, 1)
    pictures, _ = detect(tmp_path, image)
    assert not pictures.interior[117, 57:213].any()
    result, _ = analyze_image(save(tmp_path, image))
    assert any(i.type == "SUSPICIOUS_LIGHT_EDGE" for i in walk_issues(result.issues))


def test_ignored_picture_cannot_supply_a_mask_or_evidence(tmp_path):
    image = page()
    image[120:360, 60:210] = artwork()
    config = load_config()
    config["global"]["ignored_regions"] = [[50, 110, 220, 370]]
    pictures, _ = detect(tmp_path, image, config)
    assert not pictures.interior.any()
    assert not pictures.regions


def test_disabling_picture_localization_retains_original_detection(tmp_path):
    image = page()
    image[120:360, 60:210] = artwork()
    config = load_config()
    config["pictures"]["enabled"] = False
    pictures, _ = detect(tmp_path, image, config)
    assert not pictures.interior.any()
    assert pictures.diagnostics["enabled"] is False


def test_picture_failure_does_not_skip_either_detector(tmp_path, monkeypatch):
    image = page()
    image[100:400, 30:570] = 245
    def fail(*args):
        raise RuntimeError("simulated SLICO failure")
    monkeypatch.setattr("skinscan.engine.locate_pictures", fail)
    result, _ = analyze_image(save(tmp_path, image))
    assert result.modules["pictures"]["status"] == "error"
    assert result.modules["background"]["status"] == result.modules["text"]["status"] == "ok"
    assert result.errors and result.status != "passed"
    assert any(i.type == "BRIGHT_BACKGROUND" for i in walk_issues(result.issues))


@pytest.mark.parametrize("override", [
    {"palette_bin_size": 7}, {"min_boundary_support": 2}, {"slic_passes": 0},
    {"min_area_ratio": 0.8, "max_area_ratio": 0.5}, {"min_segment_support": 0},
])
def test_picture_config_validation(tmp_path, override):
    import yaml
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump({"pictures": override}))
    with pytest.raises(ValueError):
        load_config(path)


def test_excluding_picture_interiors_does_not_manufacture_narrow_edges(tmp_path):
    image = page()
    image[120:360, 60:210] = artwork(sky=True)
    result, _ = analyze_image(save(tmp_path, image))
    assert result.statistics["pictureInteriorPixelCount"] > 0
    assert not [i for i in walk_issues(result.issues) if i.type == "SUSPICIOUS_LIGHT_EDGE"]


def test_picture_dominated_bright_page_uses_remaining_ui_statistics(tmp_path):
    image = page()
    content = artwork(width=540, height=860)
    content[:440] = 245
    image[20:880, 30:570] = content
    path = save(tmp_path, image)
    baseline = load_config()
    baseline["pictures"]["enabled"] = False
    old, _ = analyze_image(path, baseline)
    assert any(i.type == "PAGE_NOT_DARK" for i in walk_issues(old.issues))
    result, _ = analyze_image(path)
    assert not result.errors
    assert result.statistics["brightPixelRatio"] > 0.4
    assert result.statistics["uiBrightPixelRatio"] < 0.15
    assert not any(i.type == "PAGE_NOT_DARK" for i in walk_issues(result.issues))


def test_insufficient_remaining_ui_is_not_a_pass(tmp_path, monkeypatch):
    # 在此隔离图片定位器，仅验证背景链路对“没有足够可评估 UI”的完整性处理。
    from skinscan.pictures import empty_pictures
    def nearly_full_picture(frame, config, debug):
        result = empty_pictures(frame)
        result.interior[:, :590] = True
        result.diagnostics = {"status": "ok"}
        return result
    monkeypatch.setattr("skinscan.engine.locate_pictures", nearly_full_picture)
    result, _ = analyze_image(save(tmp_path, page()))
    assert result.modules["background"]["insufficientUiArea"]
    assert result.status == "uncertain"


def test_repeated_layout_supports_but_does_not_replace_picture_evidence(tmp_path):
    image = page()
    for y in (60, 340, 620):
        image[y:y+240, 60:210] = artwork()
        image[y:y+240, 350:500] = [245, 150, 25]  # equally repeated flat cards
    pictures, _ = detect(tmp_path, image)
    trusted = [r for r in pictures.regions if r["classification"] == "picture"]
    assert len(trusted) == 3
    assert all(r["repeatedNeighbors"] >= 1 for r in trusted)
    assert not pictures.interior[:, 340:].any()


def test_picture_artifact_keeps_full_resolution_and_records_method(tmp_path):
    import json
    from skinscan.cli import main
    image = page()
    image[120:360, 60:210] = artwork()
    path = save(tmp_path, image)
    output = tmp_path / "reports"
    assert main(["scan", str(path), "--output", str(output)]) == 0
    result_path = next(output.glob("*/*/result.json"))
    record = json.loads(result_path.read_text())
    assert record["artifacts"]["pictures"] == "pictures.png"
    assert cv2.imread(str(result_path.with_name("pictures.png"))).shape == image.shape
    assert record["statistics"]["pictureRegions"][0]["method"] == "slico_texture_boundary_layout"
