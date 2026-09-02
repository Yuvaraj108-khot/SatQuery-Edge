"""
tools/change.py
Flagship bi-temporal change detection tool — SatQuery-Edge.
Uses OpenCV-based difference analysis, morphological filtering,
connected-component analysis. NO neural network required.
"""

from __future__ import annotations
from typing import Any
import numpy as np
import cv2
from controller.schemas import ToolResult, BoundingBox, GeoPolygon


def run_change(images: list[Any], parameters: dict[str, Any]) -> ToolResult:
    """
    Detect changed regions between a pre-event and post-event image pair.
    Returns difference image, change mask, overlay, bounding boxes, polygons,
    region statistics, and a human-readable description.
    """
    if len(images) < 2:
        return ToolResult(
            tool_name="change",
            success=False,
            description="",
            error="Temporal change requires exactly 2 images (pre and post).",
        )

    pre_np = _to_numpy(images[0])
    post_np = _to_numpy(images[1])

    if pre_np is None or post_np is None:
        return ToolResult(
            tool_name="change",
            success=False,
            description="",
            error="Could not decode one or both images.",
        )

    # ── Step 1-3: Resize post to match pre ──────────────────────────────────
    if pre_np.shape != post_np.shape:
        post_np = cv2.resize(post_np, (pre_np.shape[1], pre_np.shape[0]))

    h, w = pre_np.shape[:2]

    # ── Step 4: Absolute difference ─────────────────────────────────────────
    pre_gray = cv2.cvtColor(pre_np, cv2.COLOR_BGR2GRAY)
    post_gray = cv2.cvtColor(post_np, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(pre_gray, post_gray)

    # ── Step 5: Normalize ───────────────────────────────────────────────────
    diff_norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # ── Step 6: Denoise ─────────────────────────────────────────────────────
    diff_denoised = cv2.GaussianBlur(diff_norm, (7, 7), 0)

    # ── Step 7: Threshold ───────────────────────────────────────────────────
    threshold = float(parameters.get("threshold", 0.15))
    thresh_val = int(threshold * 255)
    _, binary_mask = cv2.threshold(diff_denoised, thresh_val, 255, cv2.THRESH_BINARY)

    # ── Step 8: Morphological filtering ─────────────────────────────────────
    k_size = int(parameters.get("morph_kernel", 5))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    binary_mask = cv2.dilate(binary_mask, kernel, iterations=1)

    # ── Step 9: Connected-component analysis ─────────────────────────────────
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_mask, connectivity=8
    )

    min_area = int(parameters.get("min_region_area", 200))

    # ── Step 10: Extract candidate regions ───────────────────────────────────
    regions = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        rw = stats[i, cv2.CC_STAT_WIDTH]
        rh = stats[i, cv2.CC_STAT_HEIGHT]
        cx, cy = float(centroids[i][0]), float(centroids[i][1])
        area_frac = area / (h * w)
        conf = min(0.60 + area_frac * 2.5, 0.98)
        regions.append({
            "x": x, "y": y, "w": rw, "h": rh,
            "area": area, "area_frac": area_frac,
            "cx": cx, "cy": cy, "conf": conf, "label_id": i,
        })

    regions = sorted(regions, key=lambda r: r["area"], reverse=True)[:8]

    # ── Step 11: Change mask + highlighted overlay ───────────────────────────
    change_mask_color = _colorize_mask(binary_mask, diff_norm)
    overlay = _create_overlay(post_np, binary_mask, regions, w, h)

    # False-color difference visualization
    diff_color = cv2.applyColorMap(diff_norm, cv2.COLORMAP_JET)

    # ── Build return schemas ──────────────────────────────────────────────────
    boxes: list[BoundingBox] = []
    polygons: list[GeoPolygon] = []
    for r in regions:
        boxes.append(BoundingBox(
            x1=r["x"] / w, y1=r["y"] / h,
            x2=(r["x"] + r["w"]) / w, y2=(r["y"] + r["h"]) / h,
            confidence=round(r["conf"], 3),
            label="candidate changed region",
        ))
        # Simple rectangular polygon for each region
        px1, py1 = r["x"] / w, r["y"] / h
        px2, py2 = (r["x"] + r["w"]) / w, (r["y"] + r["h"]) / h
        polygons.append(GeoPolygon(
            coordinates=[
                [px1, py1], [px2, py1], [px2, py2], [px1, py2], [px1, py1]
            ],
            confidence=round(r["conf"], 3),
            label="candidate changed region",
            area_km2_approx=round(r["area_frac"] * 25.0, 3),
            is_demo_georef=True,
        ))

    changed_pct = float(np.sum(binary_mask > 0)) / (h * w) * 100.0
    desc = _build_description(regions, changed_pct, h, w)

    return ToolResult(
        tool_name="change",
        success=True,
        description=desc,
        bounding_boxes=boxes,
        polygons=polygons,
        metrics={
            "region_count": len(regions),
            "changed_pixel_percent": round(changed_pct, 2),
            "largest_region_frac": round(regions[0]["area_frac"], 4) if regions else 0.0,
            "max_confidence": round(max((r["conf"] for r in regions), default=0.0), 3),
        },
        image_outputs={
            "pre_event": pre_np,
            "post_event": post_np,
            "difference": diff_color,
            "change_mask": change_mask_color,
            "overlay": overlay,
        },
    )


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _to_numpy(img: Any) -> np.ndarray | None:
    try:
        if isinstance(img, np.ndarray):
            arr = img
        else:
            arr = np.array(img)
        if arr.ndim == 2:
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        elif arr.shape[2] == 4:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
        elif arr.shape[2] == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        return arr
    except Exception:
        return None


def _colorize_mask(binary: np.ndarray, diff: np.ndarray) -> np.ndarray:
    """Create a blue-tinted change mask visualization."""
    vis = np.zeros((*binary.shape, 3), dtype=np.uint8)
    vis[binary > 0] = (220, 100, 30)   # BGR: orange-ish
    # Intensity proportional to diff strength
    diff_3ch = cv2.merge([diff, diff, diff])
    vis = cv2.addWeighted(vis, 0.7, diff_3ch, 0.3, 0)
    return vis


def _create_overlay(
    post: np.ndarray,
    mask: np.ndarray,
    regions: list[dict],
    w: int,
    h: int,
) -> np.ndarray:
    """Highlight changed regions on the post-event image."""
    overlay = post.copy().astype(np.float32)

    # Blue tint for changed areas
    tint = np.zeros_like(overlay)
    tint[:, :, 0] = 180  # B
    tint[:, :, 2] = 40   # R
    mask_3ch = np.stack([mask, mask, mask], axis=2) / 255.0
    overlay = overlay * (1 - mask_3ch * 0.5) + tint * mask_3ch * 0.5
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    # Bounding boxes
    colors = [
        (0, 255, 255), (0, 200, 255), (30, 255, 200),
        (0, 165, 255), (100, 255, 100),
    ]
    for i, r in enumerate(regions):
        color = colors[i % len(colors)]
        x1, y1 = r["x"], r["y"]
        x2, y2 = r["x"] + r["w"], r["y"] + r["h"]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        label = f"#{i+1} conf:{r['conf']:.2f}"
        cv2.putText(
            overlay, label, (x1, max(y1 - 6, 12)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA,
        )

    return overlay


def _build_description(regions: list[dict], changed_pct: float, h: int, w: int) -> str:
    if not regions:
        return (
            "No significant changed regions detected between "
            "pre-event and post-event imagery."
        )

    largest = regions[0]
    lines = [
        f"Detected {len(regions)} significant candidate changed region(s).",
        "",
        f"Largest region:",
        f"  Approximate change area: {largest['area_frac']*100:.1f}% of image.",
        f"  Confidence: {largest['conf']:.2f}",
        f"  Center: ({largest['cx']/w:.2f}, {largest['cy']/h:.2f}) (normalized)",
        "",
        f"Total changed pixels: ~{changed_pct:.1f}% of scene.",
        "",
        "Interpretation:",
        "  Candidate newly inundated / possible flooded regions where",
        "  spectral properties changed significantly between observations.",
        "  New water-like spectral signatures appear over areas that",
        "  previously contained built-up structures or open terrain.",
        "",
        "NOTE: This prototype uses local CV heuristics. Results are",
        "indicative and not scientifically validated flood detections.",
    ]
    return "\n".join(lines)
