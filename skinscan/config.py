"""配置的唯一默认值来源：默认值 → 全局 YAML → 页面覆盖 → 完整校验。

背景与极性门控使用编码亮度 Yg；contrast 系列是线性 sRGB 的 WCAG 比值。
像素核／采样距离的换算以各调用处为准，width_ratio 通常相对截图宽度。
参数是工程初值，不是论文统一推荐值；新增键需要同时审查校验与页面覆盖行为。
"""
import copy
from pathlib import Path

import yaml


DEFAULTS = {
    "global": {"ignored_regions": [], "allowed_colors": [], "ignore_page": False},
    "pictures": {
        # 先验证图片范围，再做 ROI 分割；证据不足只保留候选，不自动排除 UI。
        "enabled": True,
        "window_width_ratio": 0.018,
        "palette_bin_size": 8,
        "palette_min_fraction": 0.012,
        "palette_max_colors": 8,
        "palette_background_max": 0.65,
        "background_color_tolerance": 10,
        "flat_color_std": 0.012,
        "flat_max_gradient": 0.03,
        "min_area_ratio": 0.0015,
        "min_width_ratio": 0.045,
        "min_height_ratio": 0.065,
        "max_aspect": 4.0,
        "max_area_ratio": 0.90,
        "min_fill_ratio": 0.55,
        "texture_color_std": 0.04,
        "texture_edge_threshold": 0.08,
        "texture_edge_density": 0.10,
        "min_texture_coverage": 0.30,
        "min_grid_coverage": 0.50,
        "min_boundary_support": 0.65,
        "high_confidence": 0.80,
        "boundary_width_ratio": 0.002,
        "slic_region_width_ratio": 0.012,
        "slic_passes": 4,
        "min_segment_support": 0.50,
        "min_segment_coverage": 0.85,
        "min_ui_area_ratio": 0.08,
        "max_candidates": 80,
    },
    "background": {
        # 绝对亮度、页面中位数偏移和多档局部阈值独立生成候选，最后统一去重。
        "absolute_bright_threshold": 0.80,
        "relative_bright_offset": 0.30,
        "local_thresholds": [0.25, 0.40, 0.55, 0.70],
        "min_mean_luminance": 0.75,
        "min_local_luminance": 0.25,
        "min_area_ratio": 0.002,
        "min_bright_pixel_ratio": 0.70,
        "min_surrounding_delta": 0.25,
        "min_rectangularity": 0.65,
        "max_uniform_std": 0.13,
        "morphology_width_ratio": 0.003,
        "surrounding_width_ratio": 0.015,
        "critical_area_ratio": 0.05,
        "critical_bright_ratio": 0.90,
        "edge_min_length_ratio": 0.12,
        "edge_max_thickness_ratio": 0.006,
        "edge_min_border_fraction": 0.70,
        "edge_max_rectangularity": 0.35,
        "clipped_rectangularity_factor": 0.80,
        "page_bright_ratio": 0.40,
        "page_max_dark_ratio": 0.25,
        "page_dark_threshold": 0.25,
        "page_largest_region_ratio": 0.30,
        "max_candidates": 300,
    },
    "text": {
        "detection_mode": "theme_mismatch",
        "dark_background_max": 0.30,
        "light_background_min": 0.75,
        "critical_contrast": 2.0,
        "high_risk_contrast": 3.0,
        "warning_contrast": 4.5,
        # 这里的组件数用于行支持／未知项分流，不能作为已测单字的最低字数。
        "min_line_components": 3,
        "morphology_scales": ["small", "normal", "large"],
        "scale_width_ratios": {"small": 0.015, "normal": 0.028, "large": 0.05},
        "min_response": 2.0,
        "min_foreground_delta": 0.004,
        "noise_window_ratio": 0.015,
        "weak_response_fraction": 0.50,
        # 旧整行估色路径已删除；此键及下方部分旧参数仅保留以兼容历史配置。
        "fallback_core_quantile": 0.70,
        "noise_multiplier": 2.5,
        "max_response_noise": 14.0,
        "min_height_ratio": 0.006,
        "max_height_ratio": 0.09,
        "max_component_aspect": 2.5,
        "min_component_area": 3,
        "min_stroke_fill": 0.05,
        "max_stroke_fill": 0.90,
        "line_height_similarity": 0.50,
        "line_vertical_overlap": 0.45,
        "line_max_gap_heights": 1.5,
        "max_line_aspect": 60.0,
        "min_line_aspect": 2.0,
        # background_padding/exclusion、max_internal_external_delta、background_delta_fraction、
        # min_core_pixels、core_distance 也属于旧整行路径；max_background_std 仍用于笔画采样。
        "background_padding_ratio": 0.008,
        "background_exclusion_ratio": 0.002,
        "max_background_std": 0.035,
        "max_internal_external_delta": 0.08,
        "background_delta_fraction": 0.50,
        "foreground_quantile": 0.25,
        "min_core_pixels": 4,
        "core_distance": 1.5,
        "candidate_validation": {
            # 图片内要求更强证据，但不把像素从文字 valid 掩码中删除。
            "same_color_tolerance": 0.012,
            "surrounding_match_min": 0.60,
            "max_width_cv": 0.85,
            "picture_min_pairs": 6,
            "picture_max_width_cv": 0.55,
            "retry_angle_degrees": 12.0,
        },
        "stroke_sampling": {
            # 核心、双侧背景稳定性和极性都通过后，才算一个可用配对。
            "max_samples_per_component": 80,
            "min_pairs": 3,
            "max_width_height_ratio": 0.50,
            "min_width": 2.0,
            "opposite_gradient_cosine": 0.50,
            "core_fractions": [0.35, 0.50, 0.65],
            "background_samples_per_side": 3,
            "background_gap_width_ratio": 0.30,
            "background_max_gradient": 0.08,
            "max_side_luminance_delta": 0.035,
            "max_side_color_delta": 0.08,
            "max_core_std": 0.035,
            "min_low_contrast_fraction": 0.40,
            "max_evidence_pairs": 12,
        },
        "max_candidates": 3000,
    },
    "filter": {
        # 纹理是辅助评分，允许色只降背景风险；都不能证明内容在语义上正常。
        "texture_edge_threshold": 0.08,
        "texture_edge_density": 0.22,
        "texture_color_std": 0.20,
        "texture_penalty": 0.28,
        "texture_review_threshold": 0.60,
        "allowed_color_tolerance": 24,
        "allowed_color_penalty": 0.30,
    },
    "scoring": {"merge_iou": 0.50, "containment": 0.85, "text_contrast_merge_tolerance": 0.15,
                "high_confidence": 0.80, "fail_severities": ["critical", "high"]},
    "runtime": {"concurrency": 2, "save_debug_masks": False, "html_report": True},
    "pages": {},
}


def _merge(base, override, path=""):
    """原地递归覆盖已知配置键；pages 延后按完整默认结构校验，列表整体替换。"""
    if not isinstance(override, dict):
        raise ValueError(f"{path or 'config'} must be a mapping")
    for key, value in override.items():
        if key not in base:
            raise ValueError(f"Unknown config key: {path}{key}")
        if key == "pages":
            if not isinstance(value, dict):
                raise ValueError("pages must be a mapping")
            base[key] = copy.deepcopy(value)
        elif isinstance(base[key], dict):
            _merge(base[key], value, f"{path}{key}.")
        else:
            base[key] = copy.deepcopy(value)
    return base


def validate(config):
    """先检查类型及单值范围，再检查跨字段关系；返回原对象，不修正错误值。"""
    def check(reference, actual, path=""):
        for key, default in reference.items():
            value = actual[key]
            name = f"{path}{key}"
            if key == "pages":
                continue
            if isinstance(default, dict):
                check(default, value, name + ".")
            elif isinstance(default, bool):
                if not isinstance(value, bool):
                    raise ValueError(f"{name} must be boolean")
            elif isinstance(default, (int, float)):
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    # Python 的 bool 也是 int，需显式拒绝，避免 true 被当作像素数或比例。
                    raise ValueError(f"{name} must be numeric")
                if not 0 <= value < float("inf"):
                    raise ValueError(f"{name} must be finite and nonnegative")
                if isinstance(default, int) and (not isinstance(value, int) or value < 1):
                    raise ValueError(f"{name} must be a positive integer")
                if ("ratio" in key or "threshold" in key or "delta" in key or
                        key in {"high_confidence", "merge_iou", "containment", "min_mean_luminance",
                                "min_local_luminance", "relative_bright_offset", "max_uniform_std",
                                "max_background_std", "foreground_quantile", "texture_penalty",
                                "allowed_color_penalty", "min_stroke_fill", "max_stroke_fill",
                                "line_height_similarity", "line_vertical_overlap", "background_delta_fraction",
                                "fallback_core_quantile", "weak_response_fraction", "edge_max_rectangularity",
                                "clipped_rectangularity_factor"}) and value > 1:
                    raise ValueError(f"{name} must be within [0, 1]")
            elif isinstance(default, list) and not isinstance(value, list):
                raise ValueError(f"{name} must be a list")
            elif isinstance(default, str) and not isinstance(value, str):
                raise ValueError(f"{name} must be a string")
    check(DEFAULTS, config)
    for region in config["global"]["ignored_regions"]:
        if (not isinstance(region, list) or len(region) != 4 or
                any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in region) or
                region[2] <= region[0] or region[3] <= region[1]):
            raise ValueError("ignored_regions requires nonnegative [left, top, right, bottom]")
    for color in config["global"]["allowed_colors"]:
        if not isinstance(color, str) or len(color) != 7 or color[0] != "#":
            raise ValueError("allowed_colors requires #RRGGBB strings")
        try:
            int(color[1:], 16)
        except ValueError:
            raise ValueError("allowed_colors requires #RRGGBB strings") from None
    t = config["text"]
    p = config["pictures"]
    if p["palette_bin_size"] not in {4, 8, 16, 32}:
        raise ValueError("pictures.palette_bin_size must be 4, 8, 16 or 32")
    if p["background_color_tolerance"] > 255:
        raise ValueError("pictures.background_color_tolerance must be <= 255")
    for key in ("palette_min_fraction", "flat_color_std", "min_fill_ratio",
                "min_texture_coverage", "min_grid_coverage", "min_boundary_support",
                "min_segment_support", "min_segment_coverage", "palette_background_max"):
        if not 0 < p[key] <= 1:
            raise ValueError(f"pictures.{key} must be within (0, 1]")
    if p["max_aspect"] < 1:
        raise ValueError("pictures.max_aspect must be >= 1")
    if p["min_area_ratio"] >= p["max_area_ratio"]:
        raise ValueError("pictures.min_area_ratio must be less than max_area_ratio")
    if t["detection_mode"] not in {"theme_mismatch", "all_low_contrast"}:
        raise ValueError("text.detection_mode must be theme_mismatch or all_low_contrast")
    if not 0 <= t["dark_background_max"] < t["light_background_min"] <= 1:
        raise ValueError("text background limits require 0 <= dark_background_max < light_background_min <= 1")
    sampling = t["stroke_sampling"]
    shape = t["candidate_validation"]
    for key in ("same_color_tolerance", "surrounding_match_min", "max_width_cv", "picture_max_width_cv"):
        if not 0 < shape[key] <= 1:
            raise ValueError(f"text.candidate_validation.{key} must be within (0, 1]")
    if not 0 < shape["retry_angle_degrees"] <= 30:
        raise ValueError("text.candidate_validation.retry_angle_degrees must be within (0, 30]")
    for key in ("opposite_gradient_cosine", "min_low_contrast_fraction"):
        if not 0 < sampling[key] <= 1:
            raise ValueError(f"text.stroke_sampling.{key} must be within (0, 1]")
    fractions = sampling["core_fractions"]
    if (len(fractions) < 2 or any(isinstance(x, bool) or not isinstance(x, (int, float)) or
                                 not 0 < x < 1 for x in fractions)):
        raise ValueError("stroke_sampling.core_fractions needs at least two numbers within (0, 1)")
    if sampling["min_pairs"] > sampling["max_samples_per_component"]:
        raise ValueError("stroke_sampling.min_pairs exceeds max_samples_per_component")
    if sampling["background_samples_per_side"] < 2:
        raise ValueError("stroke_sampling.background_samples_per_side must be at least 2")
    if not 1 < t["critical_contrast"] <= t["high_risk_contrast"] <= t["warning_contrast"]:
        raise ValueError("text contrast thresholds must satisfy 1 < critical <= high <= warning")
    if not t["morphology_scales"] or any(s not in t["scale_width_ratios"] for s in t["morphology_scales"]):
        raise ValueError("morphology_scales must contain small, normal or large")
    if (t["max_response_noise"] < t["min_response"] or t["min_line_aspect"] > t["max_line_aspect"] or
            t["min_height_ratio"] > t["max_height_ratio"] or t["min_stroke_fill"] > t["max_stroke_fill"]):
        raise ValueError("text minimum thresholds must not exceed their maxima")
    if not all(isinstance(x, (int, float)) and not isinstance(x, bool) and 0 < x < 1
               for x in config["background"]["local_thresholds"]):
        raise ValueError("local_thresholds must be numbers between 0 and 1")
    if any(s not in {"critical", "high", "medium", "suspicious"}
           for s in config["scoring"]["fail_severities"]):
        raise ValueError("Unknown fail severity")
    if config["runtime"]["concurrency"] > 32:
        raise ValueError("runtime.concurrency must be <= 32")
    return config


def load_config(path=None):
    """深拷贝默认值加载 YAML；提前校验全部页面，避免并发运行到一半才发现配置错误。"""
    config = copy.deepcopy(DEFAULTS)
    if path:
        try:
            with Path(path).open(encoding="utf-8") as stream:
                value = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid YAML config: {exc}") from exc
        if value is not None:
            _merge(config, value)
    validate(config)
    # Validate every page before starting a batch, including pages not in this run.
    for page in config["pages"]:
        page_config(config, page)
    return config


def page_config(config, page_id):
    """返回独立的页面有效配置；坐标／允许色追加，阈值覆盖，不修改共享全局配置。"""
    effective = copy.deepcopy(config)
    override = copy.deepcopy(effective.pop("pages").get(page_id, {}))
    if not isinstance(override, dict):
        raise ValueError(f"pages.{page_id} must be a mapping")
    # Convenient page coordinate/color keys match the design document.
    for key in ("ignored_regions", "allowed_colors", "ignore_page"):
        if key in override:
            override.setdefault("global", {})[key] = override.pop(key)
    if "pages" in override:
        raise ValueError("Nested pages are not supported")
    # Page coordinates/colors extend globals; thresholds override globals.
    for key in ("ignored_regions", "allowed_colors"):
        if key in override.get("global", {}):
            override["global"][key] = effective["global"][key] + override["global"][key]
    effective["pages"] = {}
    return validate(_merge(effective, override))
