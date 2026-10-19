"""Local stroke/background pairs, using opposing Sobel edges on original pixels.

Rays are vectorized within a glyph. No OCR, resampling or reconstruction of an
unrendered font color is involved. Unstable and one-sided samples are rejected.
"""
from dataclasses import dataclass, field
from typing import List

import cv2
import numpy as np

from .image import contrast_ratio, luminance


def in_detection_scope(foreground, background, polarity, config):
    """先检查笔画明暗方向，再按模式检查局部背景；不使用页面中位数替代背景。"""
    fg, bg = float(luminance(foreground)), float(luminance(background))
    delta = config["min_foreground_delta"]
    if polarity == "blackhat_dark":
        correct = fg <= bg - delta
        matching_background = bg <= config["dark_background_max"]
    else:
        correct = fg >= bg + delta
        matching_background = bg >= config["light_background_min"]
    return correct and (config["detection_mode"] == "all_low_contrast" or matching_background)


@dataclass
class StrokePair:
    """同一截面实测核心与双侧背景，位置／边缘均为原图 [x, y] 坐标。

    width 是射线首次离开候选组件的像素距离，不是恢复出的设计稿字体宽度。
    contrast 来自这一对颜色，不允许与别处的背景重新拼成摘要对。
    """
    foreground: np.ndarray
    background: np.ndarray
    position: List[int]
    edges: List[List[int]]
    width: float
    background_std: float
    core_std: float
    side_delta: float
    contrast: float

    def to_dict(self):
        return {"position": self.position, "edgePositions": self.edges,
                "foregroundRgb": self.foreground.astype(int).tolist(),
                "backgroundRgb": self.background.astype(int).tolist(),
                "contrastRatio": round(self.contrast, 3), "strokeWidth": round(self.width, 2),
                "backgroundStd": round(self.background_std, 4),
                "coreStd": round(self.core_std, 4), "sideLuminanceDelta": round(self.side_delta, 4)}


@dataclass
class SamplingResult:
    """一次方向采样的有效配对与拒绝计数；拒绝计数不覆盖所有失败原因，不能当作概率分布。

    几何、核心、背景按阶段筛选；另有极性失败可能直接跳过，不计入这三个字段。
    """
    pairs: List[StrokePair] = field(default_factory=list)
    attempted: int = 0
    rejected_geometry: int = 0
    rejected_background: int = 0
    rejected_core: int = 0


def sample_component(frame, labels, component, polarity, text_config, direction_offset=0.0):
    """借鉴 SWT 的相反边缘思想，返回单组件某方向偏移下的 SamplingResult。

    component 为 (x,y,w,h,label)，labels 为当前极性全图标签。direction_offset
    单位为度；每次独立测量，不放宽稳定性条件，也不合并其他方向的重复配对。
    """
    x, y, w, h, label = component
    c = text_config["stroke_sampling"]
    max_width = max(3, int(np.ceil(h * c["max_width_height_ratio"])))
    padding = max_width + c["background_samples_per_side"] + 3
    left, top = max(0, x-padding), max(0, y-padding)
    right, bottom = min(frame.width, x+w+padding), min(frame.height, y+h+padding)
    rgb = frame.rgb[top:bottom, left:right]
    ymap = frame.y[top:bottom, left:right]
    local_labels = labels[top:bottom, left:right]
    valid = frame.valid[top:bottom, left:right]
    glyph = (local_labels == label).astype(np.uint8)
    boundary = (glyph != 0) & (cv2.erode(glyph, np.ones((3, 3), np.uint8)) == 0)
    by, bx = np.nonzero(boundary)
    result = SamplingResult()
    if not len(bx):
        return result
    # Uniformly sample the contour, rather than selecting only its highest-contrast part.
    indices = np.linspace(0, len(bx)-1, min(len(bx), c["max_samples_per_component"]), dtype=int)
    bx, by = bx[indices], by[indices]
    gx = cv2.Sobel(ymap, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(ymap, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)
    strengths = magnitude[by, bx]
    usable = strengths > text_config["min_response"] / 255
    bx, by, strengths = bx[usable], by[usable], strengths[usable]
    result.attempted = len(bx)
    if not len(bx):
        return result
    gradients = np.stack((gx[by, bx], gy[by, bx]), axis=1) / strengths[:, None]
    direction = gradients * (-1 if polarity == "blackhat_dark" else 1)
    if direction_offset:
        angle = np.deg2rad(direction_offset)
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        direction = direction @ rotation.T
    origins = np.stack((bx, by), axis=1)

    def points(distances):
        # 向量化数组为 [射线, 距离, xy]；clip 只保护索引，inside 仍必须参与有效性检查。
        positions = np.rint(origins[:, None, :] + direction[:, None, :] * distances[..., None]).astype(int)
        px, py = positions[..., 0], positions[..., 1]
        inside = (px >= 0) & (px < rgb.shape[1]) & (py >= 0) & (py < rgb.shape[0])
        return np.clip(px, 0, rgb.shape[1]-1), np.clip(py, 0, rgb.shape[0]-1), inside

    distances = np.broadcast_to(np.arange(1, max_width+2, dtype=float), (len(bx), max_width+1))
    px, py, inside = points(distances)
    same_glyph = inside & valid[py, px] & (local_labels[py, px] == label)
    exited = ~same_glyph
    has_exit = exited.any(axis=1)
    exit_index = np.argmax(exited, axis=1)
    # 全 False 时 argmax 也会返回 0；后续必须联合 has_exit，不能把它误当一像素笔画。
    widths = exit_index.astype(float) + 1
    # Allow a one-pixel AA offset when testing the opposite edge's gradient.
    edge_distances = widths[:, None] + np.array([-1, 0, 1])[None, :]
    ex, ey, edge_inside = points(edge_distances)
    opposite_magnitude = magnitude[ey, ex]
    dot = (gx[ey, ex] * gradients[:, 0, None] + gy[ey, ex] * gradients[:, 1, None]) / np.maximum(opposite_magnitude, 1e-8)
    usable_edge = (edge_inside & valid[ey, ex] & (dot <= -c["opposite_gradient_cosine"]) &
                   (opposite_magnitude > text_config["min_response"] / 255))
    opposing = usable_edge.any(axis=1)
    opposite_index = np.argmin(np.where(usable_edge, dot, 2.0), axis=1)
    rows = np.arange(len(bx))
    exit_valid = inside[rows, exit_index] & valid[py[rows, exit_index], px[rows, exit_index]]
    geometric = has_exit & exit_valid & opposing & (widths >= c["min_width"]) & (widths <= max_width)
    result.rejected_geometry = int((~geometric).sum())

    core_distances = np.maximum(widths[:, None]-1, 1) * np.asarray(c["core_fractions"])[None, :]
    cx, cy, core_inside = points(core_distances)
    core_pixels = rgb[cy, cx]
    core_y = ymap[cy, cx]
    core_std = np.std(core_y, axis=1)
    core_good = (core_inside & valid[cy, cx] & (local_labels[cy, cx] == label)).all(axis=1)
    core_good &= core_std <= c["max_core_std"]
    core_good &= np.max(np.std(core_pixels.astype(np.float32)/255, axis=1), axis=1) <= c["max_core_std"]
    result.rejected_core = int((geometric & ~core_good).sum())
    # Select a stable rendered core quantile, never infer an original font color from AA.
    quantile = text_config["foreground_quantile"] if polarity == "blackhat_dark" else 1-text_config["foreground_quantile"]
    core_levels = np.quantile(core_y, quantile, axis=1)
    chosen = core_y <= core_levels[:, None] if polarity == "blackhat_dark" else core_y >= core_levels[:, None]

    gap = np.maximum(1, np.ceil(widths * c["background_gap_width_ratio"]))
    offsets = np.arange(c["background_samples_per_side"], dtype=float)[None, :]
    lpx, lpy, li = points(-gap[:, None]-offsets)
    rpx, rpy, ri = points(widths[:, None]+gap[:, None]+offsets)
    left_valid = li & valid[lpy, lpx] & (local_labels[lpy, lpx] == 0)
    # 标签排除只覆盖当前极性的候选；相反极性邻近图案主要由梯度／颜色稳定性拦截。
    right_valid = ri & valid[rpy, rpx] & (local_labels[rpy, rpx] == 0)
    left_valid &= magnitude[lpy, lpx] <= c["background_max_gradient"]
    right_valid &= magnitude[rpy, rpx] <= c["background_max_gradient"]
    for index in np.flatnonzero(geometric & core_good):
        # Require both sides. A button border cannot be treated as the text's background.
        if not left_valid[index].all() or not right_valid[index].all():
            result.rejected_background += 1
            continue
        left_pixels, right_pixels = rgb[lpy[index], lpx[index]], rgb[rpy[index], rpx[index]]
        left_color, right_color = np.median(left_pixels, axis=0), np.median(right_pixels, axis=0)
        side_delta = abs(float(luminance(left_color))-float(luminance(right_color)))
        background_pixels = np.concatenate((left_pixels, right_pixels))
        background_std = float(np.std(luminance(background_pixels)))
        if (side_delta > c["max_side_luminance_delta"] or
                np.max(np.abs(left_color-right_color))/255 > c["max_side_color_delta"] or
                background_std > text_config["max_background_std"] or
                np.max(np.std(background_pixels.astype(np.float32)/255, axis=0)) > text_config["max_background_std"]):
            result.rejected_background += 1
            continue
        foreground = np.rint(np.median(core_pixels[index][chosen[index]], axis=0))
        background = np.rint(np.median(background_pixels, axis=0))
        signed_delta = float(luminance(foreground))-float(luminance(background))
        if ((polarity == "blackhat_dark" and signed_delta > -text_config["min_foreground_delta"]) or
                (polarity == "tophat_light" and signed_delta < text_config["min_foreground_delta"])):
            continue
        center = len(c["core_fractions"]) // 2
        position = [int(cx[index, center]+left), int(cy[index, center]+top)]
        edges = [[int(bx[index]+left), int(by[index]+top)],
                 [int(ex[index, opposite_index[index]]+left), int(ey[index, opposite_index[index]]+top)]]
        result.pairs.append(StrokePair(foreground, background, position, edges, float(widths[index]),
                                       background_std, float(core_std[index]), side_delta,
                                       contrast_ratio(foreground, background)))
    return result
