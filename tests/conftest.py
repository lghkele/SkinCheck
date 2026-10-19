"""共享合成夹具：直接生成原生 600×1000 RGB 像素，不从大图缩放。

这些用例检验规则行为，不代表业务截图准确率；资源较大的验收样本在 examples。
"""
import cv2
import numpy as np
import pytest

from skinscan.image import write_png


@pytest.fixture(autouse=True, scope="session")
def cpu_threads():
    """限制 OpenCV 内部线程，避免测试线程池叠加导致不必要的资源竞争。"""
    cv2.setNumThreads(1)


@pytest.fixture
def screenshot(tmp_path):
    """返回构造函数，产出 (PNG 路径, 可继续绘制的 RGB 数组)，源文件不自动重写。"""
    def create(name="screen", background=25, text=None, foreground=230, card=None):
        image = np.full((1000, 600, 3), background, np.uint8)
        if card:
            x1, y1, x2, y2, color = card
            image[y1:y2, x1:x2] = color
        if text:
            cv2.putText(image, text, (80, 350), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                        (foreground,) * 3, 3, cv2.LINE_AA)
        path = tmp_path / f"{name}.png"
        write_png(path, image)
        return path, image
    return create
