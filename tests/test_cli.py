"""CLI 工程契约：批次不中断、输入失败隔离、原图保真、历史工件及严格退出码。

从 main 进入而非调用内部 helper，覆盖实际参数解析、调度、序列化和审核边界。
"""
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from skinscan.cli import main
from skinscan.image import write_png


def summary(root):
    return json.loads(next(root.glob("*/summary.json")).read_text())


def test_batch_report_full_resolution_original_and_history(tmp_path, screenshot):
    good, _ = screenshot("good", text="Readable text")
    bad, _ = screenshot("bad", card=(50, 180, 550, 430, 245))
    root = tmp_path / "reports"
    assert main(["scan", str(good), str(bad), "--output", str(root)]) == 0
    report = summary(root)
    assert report["counts"] == {"passed": 1, "failed": 1}
    run = root / report["runId"]
    for record, source in zip(report["results"], (good, bad)):
        directory = run / record["artifactDirectory"]
        assert (directory / "original.png").read_bytes() == source.read_bytes()
        assert cv2.imread(str(directory / "annotated.png")).shape == (1000, 600, 3)
        assert json.loads((directory / "result.json").read_text())["status"] == record["status"]
    assert (run / "index.html").is_file()
    assert main(["scan", str(good), "--output", str(root)]) == 0
    assert len(list(root.glob("*/summary.json"))) == 2


def test_strict_exit_codes(tmp_path, screenshot):
    # 0/1/2 分别覆盖正常、确定风险、无法可靠判断；缺失输入同样属于执行错误 2。
    good, _ = screenshot("normal", text="Readable")
    bad, _ = screenshot("bad", card=(50, 180, 550, 430, 245))
    uncertain, _ = screenshot("short", background=37, text="7", foreground=23)
    for path, expected in ((good, 0), (bad, 1), (uncertain, 2), (tmp_path / "missing.png", 2)):
        assert main(["scan", str(path), "--strict", "--output", str(tmp_path / path.stem / "reports")]) == expected


def test_module_failure_report_only_still_exits_two(tmp_path, screenshot, monkeypatch):
    path, _ = screenshot()
    def fail(*args):
        raise RuntimeError("broken text")
    monkeypatch.setattr("skinscan.engine.detect_text", fail)
    assert main(["scan", str(path), "--output", str(tmp_path / "out")]) == 2
    assert summary(tmp_path / "out")["results"][0]["reviewStatus"] == "needs_review"


def test_manifest_page_override_and_missing_input_continues(tmp_path, screenshot):
    path, _ = screenshot(card=(50, 180, 550, 430, 245))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps([{"imagePath": path.name, "pageId": "home", "theme": "dark"},
                                    {"imagePath": "missing.png", "pageId": "missing"}]))
    config = tmp_path / "config.yaml"
    config.write_text("pages:\n  home:\n    ignored_regions: [[40, 170, 560, 440]]\n    runtime:\n      save_debug_masks: true\n")
    root = tmp_path / "reports"
    assert main(["scan", "--manifest", str(manifest), "--config", str(config), "--output", str(root)]) == 2
    report = summary(root)
    assert report["counts"] == {"passed": 1, "error": 1}
    masks = root / report["runId"] / report["results"][0]["artifactDirectory"] / "debug"
    assert list(masks.glob("*.png"))


def test_human_feedback_does_not_overwrite_machine_result(tmp_path, screenshot):
    path, _ = screenshot(card=(50, 180, 550, 430, 245))
    root = tmp_path / "out"
    main(["scan", str(path), "--output", str(root)])
    result = next(root.glob("*/*/result.json"))
    original = result.read_bytes()
    issue = json.loads(original)["issues"][0]["id"]
    assert main(["review", str(result), "--issue", issue, "--verdict", "confirmed", "--reviewer", "qa", "--note", "Needs fix"]) == 0
    assert main(["review", str(result), "--verdict", "unresolved", "--reviewer", "qa"]) == 0
    feedback = json.loads(result.with_name("review.json").read_text())
    assert feedback["issues"][issue]["verdict"] == "confirmed"
    assert len(feedback["history"]) == 2
    assert result.read_bytes() == original
    assert main(["review", str(result), "--issue", "fake", "--verdict", "dismissed", "--reviewer", "qa"]) == 2


def test_corrupt_sidecar_does_not_skip_other_images(tmp_path, screenshot):
    good, _ = screenshot("good")
    bad, _ = screenshot("bad")
    bad.with_suffix(".json").write_text("broken json")
    root = tmp_path / "out"
    assert main(["scan", str(bad), str(good), "--output", str(root)]) == 2
    assert summary(root)["counts"] == {"error": 1, "passed": 1}


def test_html_escapes_metadata(tmp_path, screenshot):
    path, _ = screenshot()
    path.with_suffix(".json").write_text(json.dumps({"pageId": '<script>alert("bad")</script>'}))
    root = tmp_path / "out"
    assert main(["scan", str(path), "--output", str(root)]) == 0
    content = next(root.glob("*/index.html")).read_text()
    assert "<script>" not in content
    assert "&lt;script&gt;" in content


@pytest.mark.parametrize("kind", ["transparent", "jpeg", "sixteen_bit"])
def test_non_screenshot_inputs_rejected(tmp_path, kind):
    path = tmp_path / "screen.png"
    if kind == "transparent":
        image = np.full((50, 50, 4), 200, np.uint8)
        cv2.imwrite(str(path), image)
    elif kind == "sixteen_bit":
        cv2.imwrite(str(path), np.full((50, 50), 200, np.uint16))
    else:
        ok, data = cv2.imencode(".jpg", np.full((50, 50, 3), 200, np.uint8))
        assert ok
        data.tofile(str(path))
    assert main(["scan", str(path), "--output", str(tmp_path / "out")]) == 2


def test_medium_contrast_does_not_block_strict(tmp_path, screenshot):
    # 中风险提示可保留人工复核，但按当前 CI 策略不阻断；不要把它改成“有告警即失败”。
    path, _ = screenshot(background=30, text="Warning text", foreground=120)
    config = tmp_path / "all_contrast.yaml"
    config.write_text("text:\n  detection_mode: all_low_contrast\n")
    root = tmp_path / "out"
    assert main(["scan", str(path), "--config", str(config), "--strict", "--output", str(root)]) == 0
    assert summary(root)["results"][0]["status"] == "warning"


def test_bounded_candidate_analysis_is_reviewed(tmp_path, screenshot):
    path, _ = screenshot(text="Many words 0123456789")
    config = tmp_path / "config.yaml"
    config.write_text("text:\n  max_candidates: 1\n")
    root = tmp_path / "out"
    assert main(["scan", str(path), "--config", str(config), "--strict", "--output", str(root)]) == 2
    result = summary(root)["results"][0]
    assert result["status"] == "uncertain"
    assert result["modules"]["text"]["candidateLimitReached"]


def test_invalid_yaml_is_clean_cli_error(tmp_path, screenshot):
    path, _ = screenshot()
    config = tmp_path / "broken.yaml"
    config.write_text("text: [unterminated")
    assert main(["scan", str(path), "--config", str(config), "--output", str(tmp_path / "out")]) == 2


def test_directory_scan_handles_uppercase_png(tmp_path, screenshot):
    path, _ = screenshot()
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    path.rename(inputs / "SCREEN.PNG")
    assert main(["scan", str(inputs), "--output", str(tmp_path / "out")]) == 0
