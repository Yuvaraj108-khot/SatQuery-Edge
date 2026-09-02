"""
data/sample/generate_samples.py
Programmatically generates synthetic satellite-like demo images & real GeoTIFF files.
Creates:
  pre_event.png / pre_event.tif   — pre-flood scene (4-band Optical RGB+NIR, EPSG:4326)
  post_event.png / post_event.tif — post-flood scene (4-band Optical RGB+NIR, EPSG:4326)
  sar_like.png / sar_sentinel1.tif — Sentinel-1 SAR scene (2-band VV+VH, EPSG:4326)
"""

import os
import numpy as np
import cv2
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Center near Guwahati / Assam (26.1445°N, 91.7362°E)
DEMO_BOUNDS = (91.70, 26.12, 91.78, 26.18)  # (min_lon, min_lat, max_lon, max_lat)


def generate_pre_event(size: int = 512) -> np.ndarray:
    """
    Simulate a pre-flood optical satellite scene (BGR).
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
        for row in range(fy, fy+fh, 8):
            img[row, fx:fx+fw] = np.clip(np.array(color) - 15, 0, 255)

    # Built-up areas (grey rectangular blocks)
    buildings = [
        (150, 120, 60, 80),
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
        for r in range(by, by+bh, 12):
            roof_shade = base - 20
            img[r:r+5, bx:bx+bw] = np.clip([roof_shade]*3, 0, 255)

    # Roads (dark lines)
    cv2.line(img, (0, 200), (size, 200), (60, 60, 60), 8)
    cv2.line(img, (0, 350), (size, 350), (60, 60, 60), 6)
    cv2.line(img, (250, 0), (250, size), (60, 60, 60), 8)
    cv2.line(img, (400, 0), (400, size), (60, 60, 60), 6)

    # Small river
    pts = np.array([[60, 330], [120, 340], [180, 345], [250, 350]], np.int32)
    cv2.polylines(img, [pts], False, (110, 90, 60), 12)

    noise = np.random.randint(-8, 8, img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img


def generate_post_event(pre: np.ndarray, size: int = 512) -> np.ndarray:
    """
    Simulate a post-flood scene by introducing water-like regions
    over built-up areas and terrain.
    """
    post = pre.copy()
    WATER = [139, 90, 43]   # BGR ≈ muddy water blue

    flood_regions = [
        (150, 120, 90, 130),
        (280, 120, 80, 120),
        (300, 250, 70, 100),
        (200, 200, 60, 80),
    ]
    for fy, fx, fh, fw in flood_regions:
        region = post[fy:fy+fh, fx:fx+fw].astype(np.float32)
        water_arr = np.full_like(region, WATER, dtype=np.float32)
        blended = cv2.addWeighted(region, 0.15, water_arr, 0.85, 0)
        post[fy:fy+fh, fx:fx+fw] = blended.astype(np.uint8)

    river_flood = [
        (330, 0, 50, 200),
        (310, 180, 60, 100),
    ]
    for fy, fx, fh, fw in river_flood:
        region = post[fy:fy+fh, fx:fx+fw].astype(np.float32)
        water_arr = np.full_like(region, [130, 85, 40], dtype=np.float32)
        blended = cv2.addWeighted(region, 0.20, water_arr, 0.80, 0)
        post[fy:fy+fh, fx:fx+fw] = blended.astype(np.uint8)

    post = cv2.addWeighted(post, 0.90, np.full_like(post, 200), 0.10, 0)
    noise = np.random.randint(-6, 6, post.shape, dtype=np.int16)
    post = np.clip(post.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return post


def generate_sar_like(size: int = 512) -> np.ndarray:
    """
    Create a synthetic SAR-like image (speckled, water=dark, urban=bright).
    """
    img = np.zeros((size, size, 3), dtype=np.uint8)
    base = np.random.randint(50, 90, (size, size), dtype=np.uint8)
    img[:, :, 0] = base
    img[:, :, 1] = base
    img[:, :, 2] = base

    buildings = [
        (150, 120, 60, 80),
        (220, 130, 40, 50),
        (160, 280, 70, 90),
        (240, 290, 50, 60),
        (300, 150, 55, 70),
    ]
    for by, bx, bh, bw in buildings:
        bright = np.random.randint(180, 240, (bh, bw), dtype=np.uint8)
        img[by:by+bh, bx:bx+bw] = np.stack([bright]*3, axis=-1)

    water_regions = [
        (150, 120, 90, 130),
        (330, 0, 50, 200),
        (60, 330, 12, 60),
    ]
    for wy, wx, wh, ww in water_regions:
        dark = np.random.randint(5, 25, (wh, ww), dtype=np.uint8)
        img[wy:wy+wh, wx:wx+ww] = np.stack([dark]*3, axis=-1)

    speckle = np.random.exponential(1.0, (size, size))
    speckle = np.clip(speckle * 128, 0, 255).astype(np.uint8)
    gray_ch = img[:, :, 0].astype(np.float32)
    gray_ch = np.clip(gray_ch * (speckle / 128.0), 0, 255).astype(np.uint8)
    img[:, :, 0] = gray_ch
    img[:, :, 1] = gray_ch
    img[:, :, 2] = gray_ch
    return img


def write_geotiff(filename: str, bgr_arr: np.ndarray, is_sar: bool = False):
    """Write a multi-band GeoTIFF file with EPSG:4326 geotransform."""
    h, w = bgr_arr.shape[:2]
    min_lon, min_lat, max_lon, max_lat = DEMO_BOUNDS
    transform = from_bounds(min_lon, min_lat, max_lon, max_lat, w, h)

    if not is_sar:
        # Sentinel-2 style: Band 1: Blue, Band 2: Green, Band 3: Red, Band 4: NIR
        blue = bgr_arr[:, :, 0].astype(np.uint16) * 40
        green = bgr_arr[:, :, 1].astype(np.uint16) * 40
        red = bgr_arr[:, :, 2].astype(np.uint16) * 40
        # NIR high over vegetation (green), low over water (blue)
        nir = (green.astype(np.float32) * 1.5 - blue.astype(np.float32) * 0.8)
        nir = np.clip(nir, 100, 10000).astype(np.uint16)

        profile = {
            'driver': 'GTiff',
            'dtype': 'uint16',
            'nodata': 0,
            'width': w,
            'height': h,
            'count': 4,
            'crs': CRS.from_epsg(4326),
            'transform': transform,
        }
        with rasterio.open(filename, 'w', **profile) as dst:
            dst.write(blue, 1)
            dst.set_band_description(1, 'B02_Blue')
            dst.write(green, 2)
            dst.set_band_description(2, 'B03_Green')
            dst.write(red, 3)
            dst.set_band_description(3, 'B04_Red')
            dst.write(nir, 4)
            dst.set_band_description(4, 'B08_NIR')
    else:
        # Sentinel-1 style: Band 1: VV (dB), Band 2: VH (dB)
        gray = bgr_arr[:, :, 0].astype(np.float32) / 255.0
        vv_db = -25.0 + gray * 20.0  # -25 dB to -5 dB
        vh_db = -32.0 + gray * 18.0  # -32 dB to -14 dB

        profile = {
            'driver': 'GTiff',
            'dtype': 'float32',
            'nodata': -9999.0,
            'width': w,
            'height': h,
            'count': 2,
            'crs': CRS.from_epsg(4326),
            'transform': transform,
        }
        with rasterio.open(filename, 'w', **profile) as dst:
            dst.write(vv_db, 1)
            dst.set_band_description(1, 'VV_dB')
            dst.write(vh_db, 2)
            dst.set_band_description(2, 'VH_dB')


def ensure_samples():
    """Generate sample images & GeoTIFFs if they don't exist."""
    np.random.seed(42)

    pre_path = os.path.join(OUT_DIR, "pre_event.png")
    post_path = os.path.join(OUT_DIR, "post_event.png")
    sar_path = os.path.join(OUT_DIR, "sar_like.png")

    pre_tif = os.path.join(OUT_DIR, "pre_event.tif")
    post_tif = os.path.join(OUT_DIR, "post_event.tif")
    sar_tif = os.path.join(OUT_DIR, "sar_sentinel1.tif")

    pre = generate_pre_event()
    if not os.path.exists(pre_path):
        cv2.imwrite(pre_path, pre)
    if not os.path.exists(pre_tif):
        write_geotiff(pre_tif, pre, is_sar=False)

    post = generate_post_event(pre)
    if not os.path.exists(post_path):
        cv2.imwrite(post_path, post)
    if not os.path.exists(post_tif):
        write_geotiff(post_tif, post, is_sar=False)

    sar = generate_sar_like()
    if not os.path.exists(sar_path):
        cv2.imwrite(sar_path, sar)
    if not os.path.exists(sar_tif):
        write_geotiff(sar_tif, sar, is_sar=True)


if __name__ == "__main__":
    ensure_samples()
    print("Sample images and real GeoTIFFs generated in", OUT_DIR)
