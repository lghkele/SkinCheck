"""文字候选的像素／几何验证；所有组合门限为工程规则，不是字形识别器。

距离变换局部极值只近似中轴宽度，不是完整 SWT 或精确骨架；不能证明候选是字。
"""
import cv2
import numpy as np

from .strokes import in_detection_scope


def component_shape(frame, labels, component, response, opposite, polarity, config):
    """返回 (拒绝原因或 None, 结构证据)，所有局部数组仍直接取自原像素。

    response/opposite 是同尺寸、相反极性的形态学响应。scopeEligible 只决定
    采样失败后是否保留未知项，不能当作实际前景／背景对比度测量。
    """
    x, y, w, h, label = component
    c = config['candidate_validation']
    pad = max(4, int(round(h * 0.5)))
    left, top = max(0, x-pad), max(0, y-pad)
    right, bottom = min(frame.width, x+w+pad), min(frame.height, y+h+pad)
    local = labels[top:bottom, left:right] == label
    valid = frame.valid[top:bottom, left:right]
    rgb = frame.rgb[top:bottom, left:right]
    distance = cv2.distanceTransform(local.astype(np.uint8), cv2.DIST_L2, 3)
    # Local maxima approximate the medial core; include branches without thinning.
    ridge = local & (distance >= cv2.dilate(distance, np.ones((3, 3), np.uint8)) - 1e-5)
    widths = 2 * distance[ridge]
    cv = float(widths.std()/max(widths.mean(), 1)) if len(widths) else 1.0
    strengths = response[top:bottom, left:right][local]
    core = local & (response[top:bottom, left:right] >= np.quantile(strengths, 0.70))
    color = np.median(rgb[core], axis=0)
    # Do not merge a genuinely faint stroke into its background merely because
    # their difference falls below the fixed RGB tolerance.
    tolerance = min(255*c['same_color_tolerance'], max(0.5, float(np.quantile(strengths, 0.70))*0.25))
    matching = (np.max(np.abs(rgb.astype(np.float32)-color), axis=2) <= tolerance) & valid
    outside = valid & ~cv2.dilate(local.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    match_fraction = float(matching[outside].mean()) if outside.any() else 0
    # 周围需广泛同色，再联合“核心同色块连到 ROI 多侧”或“相反极性包围”之一。
    # 不能只凭一个开口或外连就删掉真实字；这里用于识别字缝／容器底色结构。
    ring = cv2.dilate(local.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool) & ~local & valid
    opposite_support = float((opposite[top:bottom, left:right][ring] >= config['min_response']).mean()) if ring.any() else 0
    _, connected = cv2.connectedComponents(matching.astype(np.uint8), 8)
    ids = connected[core & matching]
    connected_to_outer = False
    if len(ids):
        chosen = np.bincount(ids[ids > 0]).argmax() if (ids > 0).any() else 0
        if chosen:
            region = connected == chosen
            connected_to_outer = sum(bool(edge.any()) for edge in (region[0], region[-1], region[:, 0], region[:, -1])) >= 3
    reason = None
    if match_fraction >= c['surrounding_match_min'] and (connected_to_outer or opposite_support >= 0.15):
        reason = 'background_gap_or_container'
    elif cv > c['max_width_cv']:
        reason = 'irregular_stroke_width'
    return reason, {'widthCv': cv, 'surroundingColorMatch': match_fraction,
                    'oppositeStrokeSupport': opposite_support,
                    'scopeEligible': bool(outside.any() and in_detection_scope(
                        color, np.median(rgb[outside], axis=0),
                        polarity, config))}


def line_structure(group, config):
    """较强横向组证据，用于重测／未知分流；不是已测单字和数字的准入条件。"""
    if len(group) < config['min_line_components']:
        return False
    heights = np.array([v[3] for v in group], float)
    x1, x2 = min(v[0] for v in group), max(v[0]+v[2] for v in group)
    y1, y2 = min(v[1] for v in group), max(v[1]+v[3] for v in group)
    return (heights.std()/max(heights.mean(), 1) <= 0.35 and
            (x2-x1)/max(y2-y1, 1) >= config['min_line_aspect'])
