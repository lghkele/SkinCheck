"""背景独立检测：绝对／相对／局部反差亮块、原始拓扑窄边、剩余 UI 整页规则。

本模块使用编码亮度 Yg，不使用 WCAG 文字对比度。图片可信内部只通过独立
Frame.valid 副本排除，原图和文字链路的有效像素保持不变。
"""
from dataclasses import replace

import cv2
import numpy as np

from .image import odd, ring_samples, texture_metrics
from .models import Issue


def detect_background(frame, config, debug=None, pictures=None):
    """返回 (issues, statistics, diagnostics)；调用前应保证 frame.active_area > 0。

    原图亮暗比例保留作审计，整页规则使用扣除可信图片后的 UI 分母。
    旧 largest*RegionRatio 则来自过滤后的候选、仍除以原有效面积，需与 ui* 区分。
    """
    c, filters = config["background"], config["filter"]
    values = frame.y[frame.valid]
    original_median = float(np.median(values))
    background_valid = frame.valid.copy()
    if pictures is not None:
        background_valid &= ~pictures.interior
    ui_values = frame.y[background_valid]
    ui_area = len(ui_values)
    ui_fraction = ui_area / frame.active_area
    median = float(np.median(ui_values)) if ui_area else original_median
    ui_frame = replace(frame, valid=background_valid)
    # 只替换掩码，共享其他只读数组；外围取样和纹理统计必须采用同一 UI 有效区。
    absolute = c["absolute_bright_threshold"]
    statistics = {
        "luminanceSpace": "gamma_encoded_weighted_sRGB_0_1",
        "medianLuminance": round(original_median, 4),
        "brightPixelRatio": round(float(np.mean(values > absolute)), 4),
        "darkPixelRatio": round(float(np.mean(values < c["page_dark_threshold"])), 4),
        "largestBrightRegionRatio": 0.0,
        "largestLowTextureBrightRegionRatio": 0.0,
        "uiPixelRatio": round(ui_fraction, 4),
        "uiMedianLuminance": round(median, 4),
        "uiBrightPixelRatio": round(float(np.mean(ui_values > absolute)), 4) if ui_area else 0.0,
        "uiDarkPixelRatio": round(float(np.mean(ui_values < c["page_dark_threshold"])), 4) if ui_area else 0.0,
        "uiLargestLowTextureBrightRegionRatio": 0.0,
    }
    thresholds = [("absolute_bright_region", absolute),
                  ("relative_bright_region", median + c["relative_bright_offset"])]
    thresholds += [("local_surrounding_contrast", value) for value in c["local_thresholds"]]
    issues, seen = [], set()
    truncated = False
    padding = odd(frame.width * c["surrounding_width_ratio"])
    close_kernel = np.ones((odd(frame.width * c["morphology_width_ratio"]),) * 2, np.uint8)
    for source, threshold in thresholds:
        if threshold >= 1:
            continue
        # No opening: even a one-pixel ring survives in the raw edge branch.
        raw = ((frame.y > threshold) & background_valid).astype(np.uint8)
        # Keep narrow-edge topology from the original screenshot. Cutting out a
        # picture's interior must not manufacture a new one-pixel white frame.
        edge_raw = ((frame.y > threshold) & frame.valid).astype(np.uint8)
        if debug is not None:
            debug[f"background_{source}_{threshold:.2f}"] = raw * 255
        closed = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, close_kernel)
        closed[~background_valid] = 0
        for edge_branch, mask in ((False, closed), (True, edge_raw)):
            count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
            eligible = [idx for idx in range(1, count)
                        if ((not edge_branch and stats[idx, cv2.CC_STAT_AREA] >= frame.active_area * c["min_area_ratio"])
                            or (edge_branch and max(stats[idx, 2:4]) >= frame.width * c["edge_min_length_ratio"]))]
            eligible.sort(key=lambda idx: int(stats[idx, cv2.CC_STAT_AREA]), reverse=True)
            truncated |= len(eligible) > c["max_candidates"]
            for idx in eligible[:c["max_candidates"]]:
                x, y, w, h, _ = map(int, stats[idx])
                bounds = [x, y, x + w, y + h]
                valid = frame.valid if edge_branch else background_valid
                component = (labels[y:y+h, x:x+w] == idx) & valid[y:y+h, x:x+w]
                original = component & ((edge_raw if edge_branch else raw)[y:y+h, x:x+w] != 0)
                # 闭运算仅负责连接形状；测量颜色和面积仍取原阈值像素，避免填洞抬高证据。
                area = int(original.sum())
                if not area:
                    continue
                region_values = frame.y[y:y+h, x:x+w][original]
                mean = float(np.mean(region_values))
                std = float(np.std(region_values))
                area_ratio = area / frame.active_area
                rectangularity = area / (w * h)
                bright_ratio = float(np.mean(region_values > absolute))
                outer, _ = ring_samples(ui_frame, bounds, padding)
                surrounding = float(np.median(outer)) if len(outer) else median
                delta = mean - surrounding
                texture = texture_metrics(ui_frame, bounds, filters)
                if source == "absolute_bright_region" and not edge_branch:
                    statistics["largestBrightRegionRatio"] = max(statistics["largestBrightRegionRatio"], area_ratio)
                    if std <= c["max_uniform_std"] and texture["textureScore"] < filters["texture_review_threshold"]:
                        statistics["largestLowTextureBrightRegionRatio"] = max(
                            statistics["largestLowTextureBrightRegionRatio"], area_ratio)
                        statistics["uiLargestLowTextureBrightRegionRatio"] = max(
                            statistics["uiLargestLowTextureBrightRegionRatio"], area/max(1, ui_area))
                if edge_branch:
                    if pictures is not None and float(np.mean(pictures.interior[y:y+h, x:x+w][original])) >= 0.90:
                        continue
                    thickness = max(1, int(round(frame.width * c["edge_max_thickness_ratio"])))
                    border = np.ones((h, w), bool)
                    if h > 2 * thickness and w > 2 * thickness:
                        border[thickness:-thickness, thickness:-thickness] = False
                    border_fraction = float(np.mean(border[original]))
                    # 细长线，或大跨度低填充且集中在边带的环／三边结构，都可进入窄边复核。
                    is_edge = (min(w, h) <= thickness or
                               (rectangularity < c["edge_max_rectangularity"] and border_fraction >= c["edge_min_border_fraction"]))
                    if not (is_edge and delta >= c["min_surrounding_delta"]):
                        continue
                    key = ("edge", *bounds)
                    if key in seen:
                        continue
                    seen.add(key)
                    issues.append(Issue("SUSPICIOUS_LIGHT_EDGE", "high", 0.52,
                                        bounds, [source, "continuous_narrow_edge"],
                                        "疑似未适配的浅色窄边或外层背景，需人工确认",
                                        {"regionLuminance": round(mean, 4), "surroundingLuminance": round(surrounding, 4),
                                         "luminanceDelta": round(delta, 4), "areaRatio": round(area_ratio, 6),
                                         "backgroundRgb": np.median(frame.rgb[y:y+h, x:x+w][original], axis=0).astype(int).tolist(),
                                         "continuousLength": max(w, h), "borderFraction": round(border_fraction, 4),
                                         **texture}, requiresReview=True))
                    continue
                # Local contrast remains an independent rule, with no absolute-brightness requirement.
                absolute_hit = (source == "absolute_bright_region" and mean >= c["min_mean_luminance"]
                                and bright_ratio >= c["min_bright_pixel_ratio"])
                local_hit = source != "absolute_bright_region" and mean >= c["min_local_luminance"]
                clipped = x == 0 or y == 0 or x + w == frame.width or y + h == frame.height
                min_rect = c["min_rectangularity"] * (c["clipped_rectangularity_factor"] if clipped else 1)
                if not ((absolute_hit or local_hit) and area_ratio >= c["min_area_ratio"] and
                        delta >= c["min_surrounding_delta"] and rectangularity >= min_rect):
                    continue
                confidence = (0.70 + 0.12 * min(delta / 0.5, 1) +
                              0.10 * max(0, 1 - std / max(c["max_uniform_std"], 1e-6)) +
                              0.05 * rectangularity - filters["texture_penalty"] * texture["textureScore"])
                severity = "high"
                if area_ratio > c["critical_area_ratio"] and bright_ratio > c["critical_bright_ratio"]:
                    severity = "critical"
                issues.append(Issue("BRIGHT_BACKGROUND", severity, round(min(0.98, confidence), 3), bounds,
                                    [source, "surrounding_luminance_delta", "region_geometry"],
                                    "疑似局部背景未适配暗夜模式",
                                    {"areaRatio": round(area_ratio, 6), "regionLuminance": round(mean, 4),
                                     "surroundingLuminance": round(surrounding, 4), "luminanceDelta": round(delta, 4),
                                     "brightPixelRatio": round(bright_ratio, 4), "luminanceStd": round(std, 4),
                                     "rectangularity": round(rectangularity, 4),
                                     "backgroundRgb": np.median(frame.rgb[y:y+h, x:x+w][original], axis=0).astype(int).tolist(),
                                     **texture}, requiresReview=texture["textureScore"] > filters["texture_review_threshold"]))
    if (ui_fraction >= config["pictures"]["min_ui_area_ratio"] and
            statistics["uiBrightPixelRatio"] > c["page_bright_ratio"] and
            statistics["uiDarkPixelRatio"] < c["page_max_dark_ratio"] and
            statistics["uiLargestLowTextureBrightRegionRatio"] > c["page_largest_region_ratio"]):
        ys, xs = np.nonzero(frame.valid)
        issues.append(Issue("PAGE_NOT_DARK", "critical", 0.96,
                            [int(xs.min()), int(ys.min()), int(xs.max()+1), int(ys.max()+1)],
                            ["page_brightness_distribution", "large_uniform_bright_region"],
                            "疑似页面主体未进入暗夜模式", {
                                **statistics, "backgroundRgb": np.median(frame.rgb[frame.valid], axis=0).astype(int).tolist()}))
    if pictures is not None:
        # 不确定图片和可信图片边带不删除；重合较多的候选改走复核，保留观察证据。
        for issue in issues:
            if issue.type == "PAGE_NOT_DARK":
                bright = (frame.y > absolute) & background_valid
                overlap = float(np.mean(pictures.uncertain[bright])) if bright.any() else 0.0
                if overlap >= 0.50:
                    issue.metrics["uncertainPictureBrightPixelFraction"] = round(overlap, 4)
                    issue.rules.append("uncertain_picture_page_coverage")
                    issue.requiresReview = True
                    issue.confidence = min(issue.confidence, 0.60)
                continue
            x1, y1, x2, y2 = issue.bounds
            pixels = (frame.y[y1:y2, x1:x2] >= c["min_local_luminance"]) & background_valid[y1:y2, x1:x2]
            near = pictures.boundary[y1:y2, x1:x2] | pictures.uncertain[y1:y2, x1:x2]
            overlap = float(np.mean(near[pixels])) if pixels.any() else 0.0
            if overlap >= 0.50:
                issue.metrics["pictureBoundaryOrUncertainOverlap"] = round(overlap, 4)
                issue.rules.append("picture_boundary_or_uncertain_content")
                issue.requiresReview = True
                issue.confidence = min(issue.confidence, 0.60)
    insufficient_ui = ui_fraction < config["pictures"]["min_ui_area_ratio"]
    # 没有足够 UI 或候选被截断时，空告警列表也不代表完整分析通过。
    return issues, statistics, {"status": "partial" if truncated or insufficient_ui else "ok",
                                "candidateLimitReached": truncated, "insufficientUiArea": insufficient_ui}
