import cv2
import numpy as np
import pytest

from skinscan.image import write_png


@pytest.fixture(autouse=True, scope="session")
def cpu_threads():
    cv2.setNumThreads(1)


@pytest.fixture
def screenshot(tmp_path):
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
