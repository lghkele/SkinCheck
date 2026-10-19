"""Create lossless 1080x2400 synthetic acceptance inputs (no screenshot resizing)."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="examples/screenshots")
    args = parser.parse_args()
    directory = Path(args.output)
    directory.mkdir(parents=True, exist_ok=True)

    def page(color=25):
        return np.full((2400, 1080, 3), color, np.uint8)

    def text(image, value, y, color, scale=1.8):
        cv2.putText(image, value, (90, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (color,)*3, 4, cv2.LINE_AA)

    samples = {}
    normal = page()
    text(normal, "Healthy dark UI", 180, 240)
    text(normal, "All systems ready", 500, 220)
    samples["normal_dark"] = normal

    white_card = page()
    white_card[400:1100, 70:1010] = 245
    text(white_card, "Low contrast card", 650, 181)
    samples["white_card"] = white_card

    dark_text = page(37)
    text(dark_text, "Dark text on dark", 800, 23)
    text(dark_text, "Balance 2026", 1000, 23)
    samples["dark_text"] = dark_text

    light_text = page()
    light_text[400:1100, 70:1010] = 220
    text(light_text, "Light text on light", 650, 248)
    samples["light_text"] = light_text

    gray_card = page(15)
    gray_card[400:1100, 70:1010] = 120
    samples["local_gray_card"] = gray_card

    ring = page()
    cv2.rectangle(ring, (70, 500), (1010, 1300), (245,)*3, 1)
    samples["one_pixel_ring"] = ring

    partial = ring.copy()
    partial[1300, 70:1011] = 25
    samples["three_sided_edge"] = partial

    short = page(37)
    text(short, "7", 700, 23)
    text(short, "OK", 1100, 23)
    samples["short_text"] = short

    light_page = page(245)
    text(light_page, "Theme did not change", 800, 30)
    samples["whole_light_page"] = light_page

    mixed = page(245)
    text(mixed, "GOOD", 800, 30)
    good_width = cv2.getTextSize("GOOD", cv2.FONT_HERSHEY_SIMPLEX, 1.8, 4)[0][0]
    cv2.putText(mixed, "weak", (90+good_width+15, 800), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (181,)*3, 4, cv2.LINE_AA)
    samples["mixed_foreground"] = mixed

    split_background = page(245)
    split_background[:, :500] = 37
    text(split_background, "SAME TEXT", 800, 23)
    samples["split_background"] = split_background

    image_content = page()
    rng = np.random.default_rng(42)
    image_content[400:1400, 70:1010] = rng.integers(0, 256, (1000, 940, 3), dtype=np.uint8)
    samples["complex_image"] = image_content

    def cover(width, height, sky=False):
        pixels = rng.integers(35, 255, (height, width, 3), dtype=np.uint8)
        pixels = cv2.GaussianBlur(pixels, (3, 3), 0.6)
        if sky:
            pixels[:height//4] = 245
        return pixels

    bright_cover = page()
    bright_cover[400:1400, 150:850] = cover(700, 1000, sky=True)
    samples["cover_bright_sky"] = bright_cover

    covers_and_card = page()
    for y in (180, 630, 1080):
        covers_and_card[y:y+350, 90:330] = cover(240, 350)
        cv2.putText(covers_and_card, "Book title", (390, y+100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (220,)*3, 3, cv2.LINE_AA)
    covers_and_card[1700:2150, 70:1010] = 245
    samples["covers_and_white_card"] = covers_and_card

    cover_and_text = page(37)
    cover_and_text[300:900, 70:470] = cover(400, 600)
    text(cover_and_text, "Weak nearby text", 1200, 23)
    samples["cover_and_weak_text"] = cover_and_text

    manifest = []
    for name, image in samples.items():
        ok, encoded = cv2.imencode(".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        if not ok:
            raise RuntimeError("PNG encoding failed")
        encoded.tofile(str(directory / f"{name}.png"))
        manifest.append({"pageId": name, "caseId": f"synthetic_{name}", "theme": "dark",
                         "width": 1080, "height": 2400, "imagePath": f"{name}.png",
                         "device": "synthetic_fixture"})
    (directory / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(directory.resolve())


if __name__ == "__main__":
    main()
