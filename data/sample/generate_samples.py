"""
data/sample/generate_samples.py
Programmatically generates synthetic satellite-like demo images.
Run once to create:
  pre_event.png  — pre-flood scene
  post_event.png — post-flood scene (visually different)
  sar_like.png   — SAR-like synthetic image
"""

import os
import numpy as np
import cv2

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def generate_pre_event(size: int = 512) -> np.ndarray:
    """
    Simulate a pre-flood optical satellite scene.
    Contains: buildings (grey), roads (dark lines), fields (green), terrain (tan).
    """
    img = np.zeros((size, size, 3), dtype=np.uint8)

    # Background: open terrain (sandy tan)
    img[:, :] = [180, 200, 140]   # BGR: pale green-tan

    # Agricultural fields (green patches)
    for fy, fx, fh, fw in [
        (20, 20, 100, 120),
        (20, 300, 80, 100),
        (420, 20, 80, 140),
        (400, 320, 100, 160),
    ]:
        color = [int(40 + np.random.randint(20)), int(140 + np.random.randint(30)), int(60 + np.random.randint(20))]
        img[fy:fy+fh, fx:fx+fw] = color
        # Texture: vary slightly per row
        for row in range(fy, fy+fh, 8):
            img[row, fx:fx+fw] = np.clip(np.array(color) - 15, 0, 255)

    # Built-up areas (grey rectangular blocks)
    buildings = [
        (150, 120, 60, 80),   # (y, x, h, w)
        (220, 130, 40, 50),
        (160, 280, 70, 90),
        (240, 290, 50, 60),
        (300, 150, 55, 70),
        (320, 350, 45, 55),
        (140, 370, 60, 65),
    ]
    for by, bx, bh, bw in buildings:
        base = 160 + np.random.randint(30)
        img[by:by+bh, bx:bx+bw] = [base, base, base]
        # Rooftop variation
        for r in range(by, by+bh, 12):
            roof_shade = base - 20
            img[r:r+5, bx:bx+bw] = np.clip([roof_shade]*3, 0, 255)

    # Roads (dark lines)
    cv2.line(img, (0, 200), (size, 200), (60, 60, 60), 8)        # horizontal
    cv2.line(img, (0, 350), (size, 350), (60, 60, 60), 6)
    cv2.line(img, (250, 0), (250, size), (60, 60, 60), 8)         # vertical
    cv2.line(img, (400, 0), (400, size), (60, 60, 60), 6)

    # Small river (dark blue-grey strip)
    pts = np.array([[60, 330], [120, 340], [180, 345], [250, 350]], np.int32)
    cv2.polylines(img, [pts], False, (110, 90, 60), 12)

    # Add subtle noise
    noise = np.random.randint(-8, 8, img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return img


def generate_post_event(pre: np.ndarray, size: int = 512) -> np.ndarray:
    """
    Simulate a post-flood scene by introducing water-like regions
    over some built-up areas and terrain.
    """
    post = pre.copy()

    # Water color: dark blue-grey (flood)
    WATER = [139, 90, 43]   # BGR ≈ muddy water blue

    # Flood patches (overlay water in specific regions)
    flood_regions = [
        (150, 120, 90, 130),   # over first building cluster
        (280, 120, 80, 120),   # road area
        (300, 250, 70, 100),   # mixed terrain
        (200, 200, 60, 80),    # junction
    ]
    for fy, fx, fh, fw in flood_regions:
        # Blend water with existing content
        region = post[fy:fy+fh, fx:fx+fw].astype(np.float32)
        water_arr = np.full_like(region, WATER, dtype=np.float32)
        blended = cv2.addWeighted(region, 0.15, water_arr, 0.85, 0)
        post[fy:fy+fh, fx:fx+fw] = blended.astype(np.uint8)

    # Additional water spread from river
    river_flood = [
        (330, 0, 50, 200),
        (310, 180, 60, 100),
    ]
    for fy, fx, fh, fw in river_flood:
        region = post[fy:fy+fh, fx:fx+fw].astype(np.float32)
        water_arr = np.full_like(region, [130, 85, 40], dtype=np.float32)
        blended = cv2.addWeighted(region, 0.20, water_arr, 0.80, 0)
        post[fy:fy+fh, fx:fx+fw] = blended.astype(np.uint8)

    # Slight overcast haze effect (reduce contrast/brightness slightly)
    post = cv2.addWeighted(post, 0.90, np.full_like(post, 200), 0.10, 0)

    # Add noise
    noise = np.random.randint(-6, 6, post.shape, dtype=np.int16)
    post = np.clip(post.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return post


def generate_sar_like(size: int = 512) -> np.ndarray:
    """
    Create a synthetic SAR-like image.
    SAR characteristics: high contrast, speckle, water = dark, urban = bright.
    Explicitly labeled SAR-like synthetic observation.
    """
    img = np.zeros((size, size, 3), dtype=np.uint8)

    # Background: medium speckled grey
    base = np.random.randint(50, 90, (size, size), dtype=np.uint8)
    img[:, :, 0] = base
    img[:, :, 1] = base
    img[:, :, 2] = base

    # Bright urban returns (buildings scatter intensely)
    buildings = [
        (150, 120, 60, 80),
        (220, 130, 40, 50),
        (160, 280, 70, 90),
        (240, 290, 50, 60),
        (300, 150, 55, 70),
    ]
    for by, bx, bh, bw in buildings:
        bright = np.random.randint(180, 240, (bh, bw), dtype=np.uint8)
        img[by:by+bh, bx:bx+bw, 0] = bright
        img[by:by+bh, bx:bx+bw, 1] = bright
        img[by:by+bh, bx:bx+bw, 2] = bright

    # Dark water returns (smooth water absorbs radar)
    water_regions = [
        (150, 120, 90, 130),
        (330, 0, 50, 200),
        (60, 330, 12, 60),
    ]
    for wy, wx, wh, ww in water_regions:
        dark = np.random.randint(5, 25, (wh, ww), dtype=np.uint8)
        img[wy:wy+wh, wx:wx+ww, 0] = dark
        img[wy:wy+wh, wx:wx+ww, 1] = dark
        img[wy:wy+wh, wx:wx+ww, 2] = dark

    # Simulate speckle (multiplicative noise)
    speckle = np.random.exponential(1.0, (size, size))
    speckle = np.clip(speckle * 128, 0, 255).astype(np.uint8)
    gray_ch = img[:, :, 0].astype(np.float32)
    gray_ch = np.clip(gray_ch * (speckle / 128.0), 0, 255)
    gray_ch = gray_ch.astype(np.uint8)
    img[:, :, 0] = gray_ch
    img[:, :, 1] = gray_ch
    img[:, :, 2] = gray_ch

    return img


def ensure_samples():
    """Generate sample images if they don't exist."""
    np.random.seed(42)   # deterministic

    pre_path = os.path.join(OUT_DIR, "pre_event.png")
    post_path = os.path.join(OUT_DIR, "post_event.png")
    sar_path = os.path.join(OUT_DIR, "sar_like.png")

    if not os.path.exists(pre_path):
        pre = generate_pre_event()
        cv2.imwrite(pre_path, pre)

    if not os.path.exists(post_path):
        pre = cv2.imread(pre_path)
        if pre is None:
            pre = generate_pre_event()
        post = generate_post_event(pre)
        cv2.imwrite(post_path, post)

    if not os.path.exists(sar_path):
        sar = generate_sar_like()
        cv2.imwrite(sar_path, sar)


if __name__ == "__main__":
    ensure_samples()
    print("Sample images generated in", OUT_DIR)
