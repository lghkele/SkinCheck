from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


def odd(value, minimum=3):
    return max(minimum, int(round(value)) | 1)


def luminance(rgb):
    """Gamma-encoded weighted RGB, normalized to [0, 1]; background screening only."""
    return np.asarray(rgb, dtype=np.float32) @ np.array([0.2126, 0.7152, 0.0722], np.float32) / 255


def relative_luminance(rgb):
    channels = np.asarray(rgb, dtype=np.float64) / 255
    linear = np.where(channels <= 0.04045, channels / 12.92, ((channels + 0.055) / 1.055) ** 2.4)
    return float(linear @ [0.2126, 0.7152, 0.0722])


def contrast_ratio(first, second):
    a, b = sorted((relative_luminance(first), relative_luminance(second)))
    return (b + 0.05) / (a + 0.05)


@dataclass
class Frame:
    rgb: np.ndarray
    y: np.ndarray
    gray: np.ndarray
    gradient: np.ndarray
    valid: np.ndarray

    @property
    def width(self):
        return self.gray.shape[1]

    @property
    def height(self):
        return self.gray.shape[0]

    @property
    def active_area(self):
        return int(np.count_nonzero(self.valid))


def load_frame(path, ignored_regions):
    path = Path(path)
    # imdecode supports Unicode file paths on all platforms.
    data = np.fromfile(str(path), dtype=np.uint8)
    if data[:8].tobytes() != b"\x89PNG\r\n\x1a\n":
        raise ValueError("Only lossless PNG screenshots are accepted")
    decoded = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if decoded is None or decoded.dtype != np.uint8:
        raise ValueError("Expected an 8-bit PNG screenshot")
    if decoded.ndim == 2:
        rgb = cv2.cvtColor(decoded, cv2.COLOR_GRAY2RGB)
    elif decoded.shape[2] == 4:
        if np.any(decoded[:, :, 3] != 255):
            raise ValueError("Transparent PNG has no unambiguous rendered background; use an opaque screenshot")
        rgb = cv2.cvtColor(decoded, cv2.COLOR_BGRA2RGB)
    else:
        rgb = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
    y = luminance(rgb)
    gray = np.rint(y * 255).astype(np.uint8)
    gx = cv2.Sobel(y, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(y, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(gx, gy)
    valid = np.ones(gray.shape, dtype=bool)
    h, w = gray.shape
    for left, top, right, bottom in ignored_regions:
        valid[min(top, h):min(bottom, h), min(left, w):min(right, w)] = False
    return Frame(rgb, y, gray, gradient, valid)


def write_png(path, rgb):
    ok, data = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise OSError("Could not encode PNG")
    data.tofile(str(path))


def ring_samples(frame, bounds, padding, exclusion=1):
    x1, y1, x2, y2 = bounds
    left, top = max(0, x1 - padding), max(0, y1 - padding)
    right, bottom = min(frame.width, x2 + padding), min(frame.height, y2 + padding)
    ring = frame.valid[top:bottom, left:right].copy()
    ring[max(0, y1 - exclusion - top):min(bottom - top, y2 + exclusion - top),
         max(0, x1 - exclusion - left):min(right - left, x2 + exclusion - left)] = False
    return frame.y[top:bottom, left:right][ring], frame.rgb[top:bottom, left:right][ring]


def texture_metrics(frame, bounds, config):
    x1, y1, x2, y2 = bounds
    valid = frame.valid[y1:y2, x1:x2]
    pixels = frame.rgb[y1:y2, x1:x2][valid]
    if len(pixels) == 0:
        return {"colorStd": 0.0, "edgeDensity": 0.0, "textureScore": 0.0}
    color_std = float(np.mean(np.std(pixels.astype(np.float32) / 255, axis=0)))
    density = float(np.mean(frame.gradient[y1:y2, x1:x2][valid] > config["texture_edge_threshold"]))
    score = min(1.0, (color_std / max(config["texture_color_std"], 1e-6)) *
                (density / max(config["texture_edge_density"], 1e-6)))
    return {"colorStd": round(color_std, 4), "edgeDensity": round(density, 4),
            "textureScore": round(score, 4)}
