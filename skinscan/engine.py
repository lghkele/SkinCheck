"""单张截图的编排入口；检测器负责产出证据，本模块负责合并与状态判定。

阅读顺序：analyze_image → 图片定位 → 两条独立检测链路 → 合并／白名单／嵌套
→ _classify。页面状态与人工复核状态独立，执行错误另存 errors。
"""
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .background import detect_background
from .config import load_config, page_config
from .image import load_frame
from .models import DetectionResult, SEVERITY, overlap, walk_issues
from .pictures import empty_pictures, locate_pictures
from .text import detect_text


def _merge_issues(issues, config):
    """同类型且空间／测量相容的候选去重，原地保留最强候选和替代证据。

    多个亮度阈值可能命中同一像素，不能将重复次数当作独立置信度支持。
    若次要候选风险更高，单独保存 severityEvidence，避免丢掉升级依据。
    """
    def duplicate(first, second):
        iou, contained = overlap(first.bounds, second.bounds)
        a = (first.bounds[2]-first.bounds[0]) * (first.bounds[3]-first.bounds[1])
        b = (second.bounds[2]-second.bounds[0]) * (second.bounds[3]-second.bounds[1])
        same_contrast = (first.type == "LOW_CONTRAST_TEXT" and first.severity == second.severity and
                         abs(first.metrics["contrastRatio"]-second.metrics["contrastRatio"]) <= config["text_contrast_merge_tolerance"])
        return iou >= config["merge_iou"] or (contained >= config["containment"] and
                                             (min(a, b)/max(a, b) >= config["merge_iou"] or same_contrast))

    merged = []
    for issue in sorted(issues, key=lambda value: value.confidence, reverse=True):
        same = next((existing for existing in merged if existing.type == issue.type and
                     duplicate(existing, issue)), None)
        if same:
            same.rules = sorted(set(same.rules + issue.rules))
            same.metrics.setdefault("alternativeDetections", []).append({
                "bounds": issue.bounds, "confidence": issue.confidence,
                "requiresReview": issue.requiresReview, "rules": issue.rules,
                "severity": issue.severity, "measurements": dict(issue.metrics),
            })
            if SEVERITY[issue.severity] > SEVERITY[same.severity]:
                same.severity = issue.severity
                same.confidence = min(same.confidence, issue.confidence)
                same.requiresReview |= issue.requiresReview
                same.metrics["severityEvidence"] = {"bounds": issue.bounds, **issue.metrics}
            # The same pixels crossing multiple thresholds are correlated evidence.
        else:
            merged.append(issue)
    return merged


def _nest_issues(issues, config):
    """原地建立背景与内部问题的层级；嵌套只改变组织，不抑制子问题。"""
    backgrounds = [i for i in issues if i.type in {"BRIGHT_BACKGROUND", "PAGE_NOT_DARK"}]
    nested = []
    # Smallest containing background wins, so text belongs to a card within a page.
    for issue in sorted(issues, key=lambda i: (i.bounds[2]-i.bounds[0]) * (i.bounds[3]-i.bounds[1])):
        area = (issue.bounds[2]-issue.bounds[0]) * (issue.bounds[3]-issue.bounds[1])
        parents = [bg for bg in backgrounds if bg is not issue and
                   (bg.bounds[2]-bg.bounds[0]) * (bg.bounds[3]-bg.bounds[1]) > area and
                   overlap(bg.bounds, issue.bounds)[1] >= config["containment"]]
        if parents:
            parent = min(parents, key=lambda i: (i.bounds[2]-i.bounds[0]) * (i.bounds[3]-i.bounds[1]))
            parent.children.append(issue)
        else:
            nested.append(issue)
    return sorted(nested, key=lambda i: (-SEVERITY[i.severity], i.bounds[1], i.bounds[0]))


def _apply_color_whitelist(issues, config):
    """允许色仅降低背景类风险，记录命中并要求复核；不会删除文字问题。"""
    colors = [np.array([int(color[n:n+2], 16) for n in (1, 3, 5)])
              for color in config["global"]["allowed_colors"]]
    filters = config["filter"]
    for issue in issues:
        # A permitted button color must not hide unreadable text on that button.
        if issue.type not in {"BRIGHT_BACKGROUND", "PAGE_NOT_DARK", "SUSPICIOUS_LIGHT_EDGE"}:
            continue
        rgb = issue.metrics.get("backgroundRgb")
        if rgb is None:
            continue
        for color, value in zip(config["global"]["allowed_colors"], colors):
            if np.max(np.abs(np.array(rgb) - value)) <= filters["allowed_color_tolerance"]:
                issue.whitelist.append(f"allowed_color:{color}")
                issue.confidence = round(max(0.1, issue.confidence - filters["allowed_color_penalty"]), 3)
                issue.severity = "high" if issue.severity == "critical" else "medium"
                issue.requiresReview = True
                issue.rules.append("allowed_color_risk_reduction")
                break


def _classify(result, config):
    """遍历所有层级判定页面状态；确定风险优先于分析不完整。

    failed 页面仍可同时 needs_review；CLI 会另外检查 errors，执行故障
    即使伴随确定风险也返回错误码，不能仅凭页面 status 判断运行是否成功。
    """
    issues = [issue for issue in walk_issues(result.issues) if not issue.suppressed]
    hard = [issue for issue in issues if issue.severity in config["fail_severities"] and
            issue.confidence >= config["high_confidence"] and not issue.requiresReview]
    incomplete = any(module["status"] != "ok" for module in result.modules.values())
    if hard:
        result.status = "failed"
    elif incomplete or any((issue.requiresReview and issue.severity != "medium") or (issue.severity in {"critical", "high"} and
                           issue.confidence < config["high_confidence"]) for issue in issues):
        result.status = "uncertain"
    elif issues:
        result.status = "warning"
    else:
        result.status = "passed"
    result.reviewStatus = "needs_review" if issues or incomplete else "not_required"


def analyze_image(path, config=None, metadata=None, debug=None):
    """返回 (DetectionResult, Frame 或 None)，不写报告、不修改输入文件。

    config 为已加载的完整配置；metadata 优先由调用者解析，库入口不自动读 sidecar。
    debug 为可选输出字典，各链路使用自己的键前缀写原尺寸中间图。
    输入／预处理错误使整页 error；单模块错误被隔离，其他链路仍执行。
    """
    started = time.perf_counter()
    metadata = dict(metadata or {})
    page_id = str(metadata.get("pageId") or Path(path).stem)
    result = DetectionResult(page_id, str(Path(path).resolve()), metadata=metadata)
    frame = None
    try:
        effective = page_config(config or load_config(), page_id)
        result.effectiveConfig = effective
        if metadata.get("theme", "dark") != "dark":
            raise ValueError("Detection requires metadata theme=dark")
        frame = load_frame(path, effective["global"]["ignored_regions"])
        result.width, result.height = frame.width, frame.height
        for key, value in (("width", frame.width), ("height", frame.height)):
            if key in metadata and metadata[key] != value:
                raise ValueError(f"Metadata {key}={metadata[key]} differs from PNG {key}={value}")
        result.statistics["ignoredPixelCount"] = frame.width * frame.height - frame.active_area
        result.statistics["ignoredRegions"] = effective["global"]["ignored_regions"]
        if effective["global"]["ignore_page"] or not frame.active_area:
            result.status, result.reviewStatus = "ignored", "not_required"
            result.statistics["whitelistHits"] = ["ignore_page" if frame.active_area else "all_pixels_ignored"]
        else:
            candidates = []
            # A picture failure cannot skip either detector or silently produce a pass.
            pictures = empty_pictures(frame)
            try:
                pictures = locate_pictures(frame, effective, debug)
                result.modules["pictures"] = pictures.diagnostics
                result.statistics["pictureRegions"] = pictures.regions
                result.statistics["pictureInteriorPixelCount"] = int(pictures.interior.sum())
                result.statistics["pictureBoundaryPixelCount"] = int(pictures.boundary.sum())
            except Exception as exc:
                result.modules["pictures"] = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
                result.errors.append(f"pictures: {type(exc).__name__}: {exc}")
            # Independent chains share only read-only image arrays. One failed future cannot skip the other.
            # 图片信息已准备好：背景排除可信内部，文字只加强证据门限，二者不依赖彼此的告警。
            with ThreadPoolExecutor(max_workers=2) as pool:
                jobs = {"background": pool.submit(detect_background, frame, effective, debug, pictures),
                        "text": pool.submit(detect_text, frame, effective, debug, pictures)}
                for module, future in jobs.items():
                    try:
                        output = future.result()
                        candidates.extend(output[0])
                        if module == "background":
                            result.statistics.update(output[1])
                        result.modules[module] = output[-1]
                    except Exception as exc:
                        result.modules[module] = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
                        result.errors.append(f"{module}: {type(exc).__name__}: {exc}")
            candidates = _merge_issues(candidates, effective["scoring"])
            _apply_color_whitelist(candidates, effective)
            result.issues = _nest_issues(candidates, effective["scoring"])
            for issue in walk_issues(result.issues):
                # ID 用于关联人工复核；不含运行时间，但框或类型改变后 ID 也会改变。
                identity = f"{page_id}:{issue.type}:{issue.bounds}"
                issue.id = hashlib.sha256(identity.encode()).hexdigest()[:16]
            _classify(result, effective["scoring"])
    except Exception as exc:
        result.status, result.reviewStatus = "error", "needs_review"
        result.errors.append(f"{type(exc).__name__}: {exc}")
    result.durationMs = round((time.perf_counter()-started)*1000, 2)
    return result, frame
