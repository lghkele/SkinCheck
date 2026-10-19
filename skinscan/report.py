"""保存机器结果及原尺寸可视化；报告展示与检测／人工审核判定相互分离。"""
import html
import json
import shutil
from pathlib import Path
from urllib.parse import quote

import cv2
import numpy as np

from .image import write_png
from .models import walk_issues


COLORS = {"critical": (235, 45, 55), "high": (255, 140, 30),
          "medium": (235, 195, 20), "suspicious": (55, 150, 255)}


def write_json(path, value):
    """同目录临时文件写完后替换目标，避免正常替换前暴露半份 JSON。

    禁止 NaN／Infinity 混入工件；这里不提供多写入者锁或断电持久性保证。
    """
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def annotate(frame, issues, high_confidence=0.8):
    """只在 RGB 副本上绘制；需复核项统一蓝色，风险级别仍完整保留在 JSON。"""
    image = frame.rgb.copy()
    for index, issue in enumerate(walk_issues(issues), 1):
        low = issue.requiresReview or issue.confidence < high_confidence
        color = COLORS["suspicious" if low else issue.severity]
        x1, y1, x2, y2 = issue.bounds
        cv2.rectangle(image, (x1, y1), (x2-1, y2-1), color, max(1, frame.width // 500))
        measurement = (f"CR={issue.metrics['contrastRatio']:.2f}" if "contrastRatio" in issue.metrics
                       else f"dY={issue.metrics['luminanceDelta']:.2f}" if "luminanceDelta" in issue.metrics else "")
        # OpenCV's built-in Latin font is portable; full Chinese details are in JSON/HTML.
        label = f"{index} {issue.type} {issue.severity} {measurement} c={issue.confidence:.2f}"
        scale = max(0.35, min(0.8, frame.width / 1500))
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        x = max(0, min(x1, frame.width-tw-4))
        y = min(frame.height-2, max(th+4, y1-3))
        cv2.rectangle(image, (x, y-th-3), (min(frame.width-1, x+tw+4), y+2), color, -1)
        cv2.putText(image, label, (x+2, y-1), cv2.FONT_HERSHEY_SIMPLEX, scale, (10, 15, 20), 1, cv2.LINE_AA)
    return image


def annotate_pictures(frame, regions):
    """绿／橙框展示采用／待复核候选；框不是实际像素掩码，后者见调试工件。"""
    image = frame.rgb.copy()
    for index, region in enumerate(regions, 1):
        x1, y1, x2, y2 = region["bounds"]
        trusted = region["classification"] == "picture"
        color = (50, 210, 140) if trusted else (255, 180, 55)
        cv2.rectangle(image, (x1, y1), (x2-1, y2-1), color, max(1, frame.width//400))
        label = f'{index} {"picture" if trusted else "review"} c={region["confidence"]:.2f}'
        cv2.putText(image, label, (x1, max(16, y1-5)), cv2.FONT_HERSHEY_SIMPLEX,
                    max(0.4, frame.width/1800), color, 1, cv2.LINE_AA)
    return image


def write_artifacts(directory, result, frame, debug=None):
    """写单图工件并返回序列化记录；读取失败、frame=None 时仍保存结果 JSON。

    保留原 PNG 字节；派生标注和调试图保持原尺寸，审核文件由 CLI 单独维护。
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if frame is not None:
        # Copy the source without re-encoding, preserving the original PNG byte for byte.
        shutil.copyfile(result.image, directory / "original.png")
        write_png(directory / "annotated.png", annotate(frame, result.issues,
                  result.effectiveConfig.get("scoring", {}).get("high_confidence", 0.8)))
        if result.statistics.get("pictureRegions"):
            write_png(directory / "pictures.png", annotate_pictures(frame, result.statistics["pictureRegions"]))
    if debug:
        masks = directory / "debug"
        masks.mkdir(exist_ok=True)
        for name, value in debug.items():
            ok, encoded = cv2.imencode(".png", np.asarray(value, np.uint8))
            if not ok:
                raise OSError(f"Cannot encode debug mask {name}")
            encoded.tofile(str(masks / f"{name}.png"))
    record = result.to_dict()
    record["artifacts"] = {"original": "original.png", "annotated": "annotated.png"} if frame is not None else {}
    if frame is not None and result.statistics.get("pictureRegions"):
        record["artifacts"]["pictures"] = "pictures.png"
    write_json(directory / "result.json", record)
    return record


def write_html(directory, summary):
    """生成无需服务的静态报告；截图只通过 CSS 适配显示，不改变处理分辨率。"""
    sections = []
    for record in summary["results"]:
        key = record["artifactDirectory"]
        # 路径按 URL 编码，页面名／消息按 HTML 转义；输入元数据不能成为可执行标记。
        esc = html.escape
        directory_url = quote(key)
        pictures = ""
        if record.get("artifacts"):
            pictures = (f'<div class="pictures"><figure><figcaption>原图</figcaption><a href="{directory_url}/original.png">'
                        f'<img loading="lazy" src="{directory_url}/original.png" alt="原始截图"></a></figure>'
                        f'<figure><figcaption>检测标注</figcaption><a href="{directory_url}/annotated.png">'
                        f'<img loading="lazy" src="{directory_url}/annotated.png" alt="标注截图"></a></figure></div>')
            if record["artifacts"].get("pictures"):
                pictures += (f'<p><a href="{directory_url}/pictures.png">图片定位图：绿色为已采用区域，橙色为待复核候选</a></p>')
        details = esc(json.dumps({"issues": record["issues"], "statistics": record["statistics"],
                                 "modules": record["modules"], "errors": record["errors"]}, ensure_ascii=False, indent=2))
        sections.append(f'<section><h2>{esc(record["pageId"])}</h2><p class="{esc(record["status"])}">'
                        f'{esc(record["status"])} · {esc(record["reviewStatus"])} · {record["durationMs"]} ms</p>'
                        f'<p>{esc(record["image"])}</p>{pictures}'
                        f'<details><summary>异常、白名单、模块状态与检测数据</summary><pre>{details}</pre></details>'
                        f'<p><a href="{directory_url}/result.json">JSON</a> · '
                        f'<a href="{directory_url}/review.json">人工复核记录</a></p></section>')
    document = ('<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
                '<meta name="viewport" content="width=device-width,initial-scale=1">'
                '<title>SkinScan 暗夜模式检测</title><style>'
                'body{font:15px/1.6 system-ui,sans-serif;margin:32px auto;padding:0 20px;max-width:1400px;background:#14171d;color:#e5e7eb}'
                'a{color:#8cc9ff}section{padding:20px;margin:24px 0;background:#20252e;border-radius:12px}'
                'h1,h2{line-height:1.3}.pictures{display:flex;gap:20px;align-items:flex-start}figure{flex:1;min-width:0;margin:0}'
                'img{width:100%;height:auto;display:block}pre{overflow:auto;white-space:pre-wrap;overflow-wrap:anywhere}'
                '.failed,.error{color:#ff8e8e}.uncertain{color:#ffd782}.passed{color:#86e8ad}'
                '@media(max-width:700px){.pictures{display:block}figure{margin-bottom:16px}}</style>'
                '<h1>SkinScan 暗夜模式截图检测</h1><p>完整分辨率 · CPU · 无 OCR。置信度为规则评分；可疑项需人工复核。</p>'
                f'<p>运行：{html.escape(summary["runId"])} · {len(summary["results"])} 张 · '
                f'{html.escape(json.dumps(summary["counts"], ensure_ascii=False))}</p>' + ''.join(sections) + '</html>')
    (Path(directory) / "index.html").write_text(document, encoding="utf-8")
