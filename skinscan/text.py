"""Multi-scale classical stroke detection; never recognizes character content."""
from collections import deque

import cv2
import numpy as np

from .image import luminance, odd
from .models import Issue
from .strokes import in_detection_scope, sample_component
from .textshape import component_shape, line_structure


def _group_components(components, config):
    """Join neighboring, similarly sized glyphs without requiring a minimum glyph count."""
    parent = list(range(len(components)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    active = deque()
    max_h = max((item[3] for item in components), default=1)
    order = sorted(range(len(components)), key=lambda i: components[i][0])
    for i in order:
        x, y, w, h, _ = components[i]
        while active and components[active[0]][0] + components[active[0]][2] < x - max_h * config["line_max_gap_heights"]:
            active.popleft()
        for j in active:
            bx, by, bw, bh, _ = components[j]
            vertical = max(0, min(y+h, by+bh) - max(y, by)) / min(h, bh)
            gap = max(0, x - (bx + bw))
            if (min(h, bh) / max(h, bh) >= config["line_height_similarity"] and
                    vertical >= config["line_vertical_overlap"] and
                    gap <= max(h, bh) * config["line_max_gap_heights"]):
                parent[root(i)] = root(j)
        active.append(i)
    groups = {}
    for i, component in enumerate(components):
        groups.setdefault(root(i), []).append(component)
    return list(groups.values())


def _analyze_line(frame, labels, response, group, polarity, config, pictures=None, diagnostics=None, shapes=None):
    c = config["text"]
    sampling = c["stroke_sampling"]
    measured = {}
    unresolved = []
    structured = line_structure(group, c)
    diagnostics = diagnostics if diagnostics is not None else {}
    def record(key):
        diagnostics[key] = diagnostics.get(key, 0) + 1
    for component in group:
        x, y, w, h, label = component
        on_picture = (pictures is not None and
                      float(pictures.interior[y:y+h, x:x+w][labels[y:y+h, x:x+w] == label].mean()) >= 0.5)
        required = max(sampling["min_pairs"], c["candidate_validation"]["picture_min_pairs"]) if on_picture else sampling["min_pairs"]
        result = sample_component(frame, labels, component, polarity, c)
        if len(result.pairs) < required and (structured or result.attempted-result.rejected_geometry >= sampling["min_pairs"]):
            # Retry bounded directions on the same pixels; never relax color or
            # background checks, and never count duplicate rays as extra support.
            angle = c["candidate_validation"]["retry_angle_degrees"]
            for offset in (-angle, angle):
                retry = sample_component(frame, labels, component, polarity, c, direction_offset=offset)
                record("directionRetryCount")
                if len(retry.pairs) > len(result.pairs):
                    result = retry
        if len(result.pairs) < required:
            if shapes is None or shapes[label]["scopeEligible"]:
                unresolved.append(component)
            record("pictureUnmeasuredComponentCount" if on_picture else "unmeasuredComponentCount")
            continue
        widths = np.asarray([p.width for p in result.pairs])
        if on_picture:
            positions = np.unique(np.asarray([p.position for p in result.pairs]), axis=0)
            span = np.ptp(positions, axis=0)
            if (widths.std()/max(widths.mean(), 1) > c["candidate_validation"]["picture_max_width_cv"] or
                    len(positions) < required or max(span[0]/w, span[1]/h) < 0.30):
                record("pictureIrregularPairCount")
                continue
        eligible = [p for p in result.pairs if in_detection_scope(p.foreground, p.background, polarity, c)]
        low = [p for p in eligible if p.contrast < c["warning_contrast"]]
        if (len(low) >= required and
                len(low)/max(1, len(eligible)) >= sampling["min_low_contrast_fraction"]):
            if on_picture and len({tuple(p.position) for p in low}) < required:
                record("pictureInsufficientLowPairCount")
                continue
            measured[component[4]] = (component, result, eligible, low)
    issues = []
    # Re-group only the locally affected glyphs: healthy characters cannot hide a
    # weak word, nor extend a warning over the full original candidate line.
    for affected in _group_components([value[0] for value in measured.values()], c):
        x1, y1 = min(v[0] for v in affected), min(v[1] for v in affected)
        x2, y2 = max(v[0]+v[2] for v in affected), max(v[1]+v[3] for v in affected)
        records = [measured[component[4]] for component in affected]
        pairs = [p for _, result, _, _ in records for p in result.pairs]
        eligible = [p for _, _, values, _ in records for p in values]
        low = [p for _, _, _, values in records for p in values]
        ratios = np.asarray([p.contrast for p in low])
        representative = min(low, key=lambda p: abs(p.contrast-float(np.median(ratios))))
        ratio = representative.contrast
        heights = np.array([v[3] for v in affected], np.float32)
        height_consistency = max(0, 1-float(heights.std()/max(heights.mean(), 1)))
        widths = np.array([p.width for p in low], np.float32)
        width_consistency = max(0, 1-float(widths.std()/max(widths.mean(), 1)))
        long_line = len(affected) >= c["min_line_components"] and (x2-x1)/(y2-y1) >= c["min_line_aspect"]
        support = len(low)/max(1, len(eligible))
        confidence = 0.64 + 0.10*height_consistency + 0.08*width_consistency + 0.10*support
        if not long_line:
            confidence = min(confidence, 0.68)
        severity = "critical" if ratio < c["critical_contrast"] else "high" if ratio < c["high_risk_contrast"] else "medium"
        rule = "contrast_below_critical" if severity == "critical" else "contrast_below_high" if severity == "high" else "contrast_below_warning"
        # Retain an actual sampled color pair, not a synthetic median foreground
        # combined with a background sampled elsewhere on the line.
        fg, bg = representative.foreground, representative.background
        evidence = sorted(low, key=lambda p: (p.position[0], p.position[1]))
        indices = np.linspace(0, len(evidence)-1, min(len(evidence), sampling["max_evidence_pairs"]), dtype=int)
        issues.append(Issue("LOW_CONTRAST_TEXT", severity, round(min(0.97, confidence), 3),
                            [x1, y1, x2, y2],
                            [f"{polarity}_stroke", "opposing_edge_stroke_pairs", "local_foreground_background_pairs",
                             "local_background_polarity_gate" if c["detection_mode"] == "theme_mismatch" else "all_low_contrast_scope",
                             "text_line_structure" if long_line else "short_text_candidate", rule],
                            "疑似低对比度文字（局部笔画配对测量）",
                            {"foregroundRgb": fg.astype(int).tolist(), "backgroundRgb": bg.astype(int).tolist(),
                             "contrastRatio": round(ratio, 3), "contrastSpace": "linear_sRGB_WCAG",
                             "measurementMethod": "paired_strokes", "detectionMode": c["detection_mode"],
                             "regionLuminance": round(float(luminance(fg)), 4),
                             "backgroundLuminance": round(float(luminance(bg)), 4),
                             "luminanceDelta": round(abs(float(luminance(fg))-float(luminance(bg))), 4),
                             "backgroundStd": round(max(p.background_std for p in low), 4),
                             "componentCount": len(affected), "strokePairCount": len(pairs),
                             "eligiblePairCount": len(eligible), "lowContrastPairCount": len(low),
                             "lowContrastPairFraction": round(support, 4),
                             "contrastQuantiles": {"p10": round(float(np.quantile(ratios, 0.1)), 3),
                                                   "p50": round(float(np.median(ratios)), 3),
                                                   "p90": round(float(np.quantile(ratios, 0.9)), 3)},
                             "strokePairs": [evidence[i].to_dict() for i in indices],
                             "rejectedBackgroundPairs": sum(result.rejected_background for _, result, _, _ in records)},
                            requiresReview=not long_line))
    if not issues and unresolved and line_structure(unresolved, c):
        # Weak fragments are diagnostics, not low-contrast measurements. Preserve
        # strong UI text structure as unknown, with no invented colors or ratio.
        ui = []
        for component in unresolved:
            x, y, w, h, label = component
            if pictures is None or pictures.interior[y:y+h, x:x+w][labels[y:y+h, x:x+w] == label].mean() < 0.5:
                ui.append(component)
        if line_structure(ui, c):
            bounds = [min(v[0] for v in ui), min(v[1] for v in ui),
                      max(v[0]+v[2] for v in ui), max(v[1]+v[3] for v in ui)]
            issues.append(Issue("UNREADABLE_TEXT_CANDIDATE", "suspicious", 0.35, bounds,
                                ["text_line_structure", "stroke_pair_measurement_unavailable"],
                                "存在文字排列结构，但缺少可靠笔画配对，无法判断对比度",
                                {"measurementMethod": "unmeasured_text_structure", "componentCount": len(ui)},
                                requiresReview=True))
    return issues


def detect_text(frame, config, debug=None, pictures=None):
    c = config["text"]
    # Local residual noise controls the response floor, preserving a 14-level dark stroke.
    residual = cv2.absdiff(frame.gray, cv2.medianBlur(frame.gray, 3)).astype(np.float32)
    noise = cv2.boxFilter(residual, -1, (odd(frame.width * c["noise_window_ratio"]),) * 2)
    threshold = np.clip(c["min_response"] + c["noise_multiplier"] * noise,
                        c["min_response"], c["max_response_noise"])
    min_h = max(3, int(frame.width * c["min_height_ratio"]))
    max_h = max(min_h + 1, int(frame.width * c["max_height_ratio"]))
    issues = []
    truncated = False
    count_all = 0
    diagnostics = {}
    responses = {}
    for polarity, operation in (("blackhat_dark", cv2.MORPH_BLACKHAT), ("tophat_light", cv2.MORPH_TOPHAT)):
        response = np.zeros_like(frame.gray)
        for scale in c["morphology_scales"]:
            size = odd(frame.width * c["scale_width_ratios"][scale])
            current = cv2.morphologyEx(frame.gray, operation, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)))
            np.maximum(response, current, out=response)
        responses[polarity] = response
    for polarity, response in responses.items():
        opposite = responses["tophat_light" if polarity == "blackhat_dark" else "blackhat_dark"]
        # Sobel locates weak boundaries independently of the strong-response threshold.
        weak = (frame.gradient > threshold / 255) & (response >= threshold * c["weak_response_fraction"])
        candidate = ((response >= threshold) | weak) & frame.valid
        mask = candidate.astype(np.uint8)
        if debug is not None:
            debug[f"{polarity}_response"] = response
            debug[f"{polarity}_mask"] = mask * 255
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        components = []
        for idx in range(1, count):
            x, y, w, h, area = map(int, stats[idx])
            fill = area / (w * h)
            if (min_h <= h <= max_h and w <= h * c["max_component_aspect"] and
                    area >= c["min_component_area"] and c["min_stroke_fill"] <= fill <= c["max_stroke_fill"]):
                components.append((x, y, w, h, idx))
        count_all += len(components)
        truncated |= len(components) > c["max_candidates"]
        components = sorted(components, key=lambda item: item[2]*item[3], reverse=True)[:c["max_candidates"]]
        accepted = []
        shapes = {}
        for component in components:
            reason, evidence = component_shape(frame, labels, component, response, opposite, polarity, c)
            shapes[component[4]] = evidence
            if reason:
                key = "rejected_" + reason
                diagnostics[key] = diagnostics.get(key, 0) + 1
            else:
                accepted.append(component)
        diagnostics["acceptedComponentCount"] = diagnostics.get("acceptedComponentCount", 0) + len(accepted)
        if debug is not None:
            debug[f"{polarity}_validated_mask"] = np.isin(labels, [v[4] for v in accepted]).astype(np.uint8) * 255
        for group in _group_components(accepted, c):
            issues.extend(_analyze_line(frame, labels, response, group, polarity, config, pictures, diagnostics, shapes))
    return issues, {"status": "partial" if truncated else "ok", "candidateCount": count_all,
                    "candidateLimitReached": truncated, "detectionMode": c["detection_mode"],
                    "measurementMethod": "paired_strokes",
                    "pairedStrokeIssueCount": sum(i.metrics.get("measurementMethod") == "paired_strokes" for i in issues),
                    "fallbackIssueCount": 0,
                    "unmeasuredTextIssueCount": sum(i.type == "UNREADABLE_TEXT_CANDIDATE" for i in issues),
                    "candidateValidation": diagnostics}
