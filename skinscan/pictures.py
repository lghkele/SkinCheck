"""Conservative full-resolution picture localization, without OCR or learned models.

SLICO follows Achanta et al., TPAMI 2012 (10.1109/TPAMI.2012.120).
Palette proposals, texture coverage, layout and confidence are project heuristics.
Only verified interiors affect background detection; text keeps its original mask.
"""
from dataclasses import dataclass, field

import cv2
import numpy as np

from .image import luminance, odd
from .models import overlap


@dataclass
class PictureMap:
    interior: np.ndarray
    boundary: np.ndarray
    uncertain: np.ndarray
    regions: list = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)


def empty_pictures(frame):
    return PictureMap(*(np.zeros(frame.gray.shape, bool) for _ in range(3)))


def _local_features(frame, c):
    size = odd(frame.width * c["window_width_ratio"])
    rgb = frame.rgb.astype(np.float32) / 255
    mean = cv2.boxFilter(rgb, -1, (size, size))
    square = cv2.boxFilter(rgb * rgb, -1, (size, size))
    std = np.sqrt(np.maximum(0, square - mean * mean)).mean(axis=2)
    density = cv2.boxFilter((frame.gradient > c["texture_edge_threshold"]).astype(np.float32),
                           -1, (size, size))
    texture = (std >= c["texture_color_std"]) & (density >= c["texture_edge_density"]) & frame.valid
    return std, texture


def _palette_mask(frame, std, c):
    # Learn common *flat* colors; a multicolored cover is not a background palette.
    flat = frame.valid & (std <= c["flat_color_std"]) & (frame.gradient <= c["flat_max_gradient"])
    size = c["palette_bin_size"]
    bins = 256 // size
    quantized = (frame.rgb // size).astype(np.int32)
    codes = quantized[:, :, 0] + bins * quantized[:, :, 1] + bins * bins * quantized[:, :, 2]
    counts = np.bincount(codes[flat], minlength=bins**3)
    eligible = np.flatnonzero(counts >= frame.active_area * c["palette_min_fraction"])
    selected = sorted(eligible, key=lambda value: int(counts[value]), reverse=True)[:c["palette_max_colors"]]
    colors = []
    for code in selected:
        # Actual median avoids a quantization-bin center moving an exact dark UI color.
        pixels = frame.rgb[flat & (codes == code)]
        color = np.median(pixels, axis=0).astype(np.int16)
        colors.append(color.astype(int).tolist())
    background = np.zeros(frame.gray.shape, bool)
    all_background = np.zeros(frame.gray.shape, bool)
    # In a dark UI, don't discard a poster's flat white sky as another UI palette.
    # Uniform white cards still fail the later texture/spatial coverage checks.
    dark_ui = any(float(luminance(color)) <= c["palette_background_max"] for color in colors)
    signed_rgb = frame.rgb.astype(np.int16)
    for color in colors:
        difference = np.max(np.abs(signed_rgb - np.asarray(color, np.int16)), axis=2)
        matching = difference <= c["background_color_tolerance"]
        all_background |= matching
        if not dark_ui or float(luminance(color)) <= c["palette_background_max"]:
            background |= matching
    return background, all_background, colors


def _longest_run(values):
    indices = np.flatnonzero(values)
    if not len(indices):
        return None
    groups = np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)
    group = max(groups, key=len)
    return int(group[0]), int(group[-1] + 1)


def _trim_appendage(mask, bounds):
    # A small floating button touching a poster must not expand the poster to its top.
    x1, y1, x2, y2 = bounds
    local = mask[y1:y2, x1:x2] != 0
    rows = _longest_run(local.mean(axis=1) >= 0.45)
    if rows and rows[1] - rows[0] >= local.shape[0] * 0.50:
        y1, y2 = y1 + rows[0], y1 + rows[1]
    local = mask[y1:y2, x1:x2] != 0
    columns = _longest_run(local.mean(axis=0) >= 0.45)
    if columns and columns[1] - columns[0] >= local.shape[1] * 0.65:
        x1, x2 = x1 + columns[0], x1 + columns[1]
    return [x1, y1, x2, y2]


def _boundary_support(frame, bounds, tolerance):
    x1, y1, x2, y2 = bounds
    rgb = frame.rgb.astype(np.int16, copy=False)
    sides = []
    # Compare points a few original pixels inside/outside each side. No resampling.
    for inside_x, outside_x in ((x1+2, x1-3), (x2-3, x2+2)):
        if 0 <= outside_x < frame.width and 0 <= inside_x < frame.width:
            valid = frame.valid[y1:y2, inside_x] & frame.valid[y1:y2, outside_x]
            delta = np.max(np.abs(rgb[y1:y2, inside_x]-rgb[y1:y2, outside_x]), axis=1)
            if valid.any():
                sides.append(float(np.mean(delta[valid] > tolerance)))
    for inside_y, outside_y in ((y1+2, y1-3), (y2-3, y2+2)):
        if 0 <= outside_y < frame.height and 0 <= inside_y < frame.height:
            valid = frame.valid[inside_y, x1:x2] & frame.valid[outside_y, x1:x2]
            delta = np.max(np.abs(rgb[inside_y, x1:x2]-rgb[outside_y, x1:x2]), axis=1)
            if valid.any():
                sides.append(float(np.mean(delta[valid] > tolerance)))
    # Cropped pictures can have one missing side, but need at least two measured sides.
    return float(np.mean(sides)) if len(sides) >= 2 else 0.0


def _measure_candidate(frame, texture, proposal, bounds, c):
    x1, y1, x2, y2 = bounds
    h, w = y2-y1, x2-x1
    if (w < max(8, frame.width*c["min_width_ratio"]) or h < max(8, frame.width*c["min_height_ratio"]) or
            max(w/h, h/w) > c["max_aspect"] or
            not frame.active_area*c["min_area_ratio"] <= w*h <= frame.active_area*c["max_area_ratio"]):
        return None
    valid = frame.valid[y1:y2, x1:x2]
    if float(valid.mean()) < 0.98:
        return None  # Ignored content cannot supply evidence to suppress neighboring UI.
    fill = float(np.mean(proposal[y1:y2, x1:x2] != 0))
    local = texture[y1:y2, x1:x2]
    coverage = float(np.mean(local))
    grids = [float(tile.mean()) for row in np.array_split(local, 4, axis=0)
             for tile in np.array_split(row, 4, axis=1) if tile.size]
    grid = float(np.mean(np.asarray(grids) >= 0.20))
    # Flat cards with lines of text should not become pictures merely from their edges.
    quantized = (frame.rgb[y1:y2, x1:x2] // 16).astype(np.int32)
    codes = quantized[:, :, 0] + 16*quantized[:, :, 1] + 256*quantized[:, :, 2]
    dominant = float(np.bincount(codes[valid], minlength=4096).max() / valid.sum())
    if (coverage < c["min_texture_coverage"] or grid < c["min_grid_coverage"] or dominant > 0.80):
        return None
    boundary = _boundary_support(frame, bounds, c["background_color_tolerance"])
    score = min(0.98, 0.40 + 0.16*min(1, coverage/0.60) + 0.12*grid + 0.18*boundary + 0.07*fill)
    return {"bounds": bounds, "confidence": round(score, 3), "textureCoverage": round(coverage, 4),
            "gridCoverage": round(grid, 4), "foregroundFill": round(fill, 4),
            "dominantColorFraction": round(dominant, 4), "boundarySupport": round(boundary, 4),
            "repeatedNeighbors": 0, "method": "slico_texture_boundary_layout"}


def _layout_support(regions):
    for region in regions:
        a = region["bounds"]
        w, h = a[2]-a[0], a[3]-a[1]
        for other in regions:
            if region is other:
                continue
            b = other["bounds"]
            bw, bh = b[2]-b[0], b[3]-b[1]
            similar = min(w, bw)/max(w, bw) >= 0.80 and min(h, bh)/max(h, bh) >= 0.80
            column = abs(a[0]-b[0]) <= max(w, bw)*0.15 and 0.8*h <= abs(a[1]-b[1]) <= 3*h
            row = abs(a[1]-b[1]) <= max(h, bh)*0.15 and 0.8*w <= abs(a[0]-b[0]) <= 6*w
            if similar and (column or row):
                region["repeatedNeighbors"] += 1
        if region["repeatedNeighbors"]:
            region["confidence"] = round(min(0.98, region["confidence"] + 0.04), 3)


def _slico_interior(frame, region, c, debug):
    if not hasattr(cv2, "ximgproc") or not hasattr(cv2.ximgproc, "createSuperpixelSLIC"):
        raise RuntimeError("Picture localization requires opencv-contrib-python-headless (SLICO)")
    x1, y1, x2, y2 = region["bounds"]
    size = max(5, int(round(frame.width*c["slic_region_width_ratio"])))
    left, top = max(0, x1-size*2), max(0, y1-size*2)
    right, bottom = min(frame.width, x2+size*2), min(frame.height, y2+size*2)
    rgb = frame.rgb[top:bottom, left:right].copy()
    valid = frame.valid[top:bottom, left:right]
    if not valid.all():
        rgb[~valid] = np.median(rgb[valid], axis=0)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    slic = cv2.ximgproc.createSuperpixelSLIC(lab, algorithm=cv2.ximgproc.SLICO, region_size=size)
    slic.iterate(c["slic_passes"])
    slic.enforceLabelConnectivity(25)
    labels = slic.getLabels()
    count = slic.getNumberOfSuperpixels()
    rectangle = np.zeros(labels.shape, bool)
    rectangle[y1-top:y2-top, x1-left:x2-left] = True
    sizes = np.bincount(labels.ravel(), minlength=count)
    inside = np.bincount(labels[rectangle & valid], minlength=count)
    selected = inside / np.maximum(1, sizes) >= c["min_segment_support"]
    supported = selected[labels] & rectangle & valid
    inset = max(1, int(round(frame.width*c["boundary_width_ratio"])))
    core = np.zeros(labels.shape, bool)
    if y2-y1 > inset*2 and x2-x1 > inset*2:
        core[y1-top+inset:y2-top-inset, x1-left+inset:x2-left-inset] = True
    interior = supported & core
    region["superpixelCount"] = int(count)
    region["segmentCoverage"] = round(float(supported[rectangle].mean()), 4)
    region["interiorPixelCount"] = int(interior.sum())
    if debug is not None:
        debug.setdefault("pictures_slico_boundaries", np.zeros(frame.gray.shape, np.uint8))[
            top:bottom, left:right] |= slic.getLabelContourMask(False)
    return (left, top, right, bottom), interior, supported & ~interior


def locate_pictures(frame, config, debug=None):
    c = config["pictures"]
    output = empty_pictures(frame)
    output.diagnostics = {"status": "ok", "enabled": c["enabled"], "candidateCount": 0,
                          "candidateLimitReached": False, "measurementMethod": "slico_texture_boundary_layout"}
    if not c["enabled"] or not frame.active_area:
        return output
    std, texture = _local_features(frame, c)
    background, all_background, colors = _palette_mask(frame, std, c)
    # Candidate generation is conservative if the screenshot has no learned flat UI colors.
    kernel = np.ones((3, 3), np.uint8)
    output.diagnostics["paletteColors"] = colors
    regions = []
    background_models = [all_background]
    if not np.array_equal(background, all_background):
        background_models.append(background)
    proposal_debug = np.zeros(frame.gray.shape, np.uint8) if debug is not None else None
    for model in background_models:
        proposal = ((~model) & frame.valid).astype(np.uint8)
        proposal = cv2.morphologyEx(proposal, cv2.MORPH_CLOSE, kernel)
        proposal = cv2.morphologyEx(proposal, cv2.MORPH_OPEN, kernel)
        proposal[~frame.valid] = 0
        if proposal_debug is not None:
            proposal_debug |= proposal*255
        count, labels, stats, _ = cv2.connectedComponentsWithStats(proposal, 8)
        eligible = [i for i in range(1, count)
                    if stats[i, 2] >= frame.width*c["min_width_ratio"] and
                    stats[i, 3] >= frame.width*c["min_height_ratio"] and
                    stats[i, 4] >= frame.active_area*c["min_area_ratio"]]
        eligible.sort(key=lambda i: int(stats[i, 4]), reverse=True)
        remaining = max(0, c["max_candidates"]-output.diagnostics["candidateCount"])
        output.diagnostics["candidateCount"] += len(eligible)
        if len(eligible) > remaining:
            output.diagnostics.update(status="partial", candidateLimitReached=True)
        for idx in eligible[:remaining]:
            x, y, w, h, _ = map(int, stats[idx])
            trimmed = _trim_appendage((labels[y:y+h, x:x+w] == idx).astype(np.uint8), [0, 0, w, h])
            trimmed = [trimmed[0]+x, trimmed[1]+y, trimmed[2]+x, trimmed[3]+y]
            options = [_measure_candidate(frame, texture, proposal, bounds, c)
                       for bounds in ([x, y, x+w, y+h], trimmed)]
            options = [region for region in options if region is not None]
            if options:
                regions.append(max(options, key=lambda region: region["confidence"]))
    # Prefer a well-supported complete picture over a nested detail/sky fragment.
    deduplicated = []
    for region in sorted(regions, key=lambda r: (
            r["confidence"] >= c["high_confidence"] and r["boundarySupport"] >= c["min_boundary_support"] and
            r["foregroundFill"] >= c["min_fill_ratio"],
            (r["bounds"][2]-r["bounds"][0]) * (r["bounds"][3]-r["bounds"][1])), reverse=True):
        if not any(overlap(region["bounds"], other["bounds"])[0] >= 0.65 or
                   overlap(region["bounds"], other["bounds"])[1] >= 0.95 for other in deduplicated):
            deduplicated.append(region)
    regions = deduplicated
    _layout_support(regions)
    for region in regions:
        x1, y1, x2, y2 = region["bounds"]
        trusted = (region["confidence"] >= c["high_confidence"] and
                   region["boundarySupport"] >= c["min_boundary_support"] and
                   region["foregroundFill"] >= c["min_fill_ratio"] and bool(colors))
        region["classification"] = "picture" if trusted else "uncertain_picture"
        if trusted:
            roi, interior, boundary = _slico_interior(frame, region, c, debug)
            left, top, right, bottom = roi
            if region["segmentCoverage"] >= c["min_segment_coverage"]:
                output.interior[top:bottom, left:right] |= interior
                output.boundary[top:bottom, left:right] |= boundary
            else:
                region["classification"] = "uncertain_picture"
                region["interiorPixelCount"] = 0
                output.uncertain[y1:y2, x1:x2] |= frame.valid[y1:y2, x1:x2]
        else:
            output.uncertain[y1:y2, x1:x2] |= frame.valid[y1:y2, x1:x2]
        output.regions.append(region)
    # A boundary of one candidate is not UI evidence inside another verified picture.
    output.boundary &= ~output.interior
    output.uncertain &= ~output.interior
    output.diagnostics["pictureCount"] = sum(r["classification"] == "picture" for r in output.regions)
    output.diagnostics["uncertainPictureCount"] = sum(r["classification"] != "picture" for r in output.regions)
    if debug is not None:
        debug["pictures_texture"] = texture.astype(np.uint8)*255
        debug["pictures_proposals"] = proposal_debug
        debug["pictures_interiors"] = output.interior.astype(np.uint8)*255
        debug["pictures_boundaries"] = output.boundary.astype(np.uint8)*255
    return output
