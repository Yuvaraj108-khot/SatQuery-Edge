"""
tools/grounding.py
Text-guided visual grounding using OpenCV thresholding + contour analysis.
Returns bounding boxes, polygons, and confidence scores.
"""

from __future__ import annotations
from typing import Any
import numpy as np
import cv2
from controller.schemas import ToolResult, BoundingBox, GeoPolygon


def run_grounding(images: list[Any], parameters: dict[str, Any]) -> ToolResult:
    """Locate candidate regions matching the text target description."""
    if not images:
        return ToolResult(
            tool_name="grounding",
            success=False,
            description="",
            error="No image provided for grounding.",
        )

    img = _to_numpy(images[0])
    if img is None:
        return ToolResult(
            tool_name="grounding",
            success=False,
            description="",
            error="Could not decode image for grounding.",
        )

    target = parameters.get("target_description", "regions of interest")
    threshold = float(parameters.get("threshold", 0.10))

    boxes, polygons, mask = _locate_regions(img, target, threshold)

    if not boxes:
        desc = f"No candidate regions found matching '{target}'."
    else:
        desc = (
            f"Grounding found {len(boxes)} candidate region(s) "
            f"matching '{target}'.\n"
            f"Highest confidence: {max(b.confidence for b in boxes):.2f}."
        )

    # Create color-highlighted overlay
    overlay = _draw_overlay(img, boxes, mask)

    return ToolResult(
        tool_name="grounding",
        success=True,
        description=desc,
        bounding_boxes=boxes,
        polygons=polygons,
        metrics={
            "region_count": len(boxes),
            "max_confidence": max((b.confidence for b in boxes), default=0.0),
            "target": target,
        },
        image_outputs={"overlay": overlay},
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


def _locate_regions(
    img: np.ndarray,
    target: str,
    threshold: float,
) -> tuple[list[BoundingBox], list[GeoPolygon], np.ndarray]:
    """
    Heuristically segment candidate regions.
    For flood/water targets: use blue-channel + saturation heuristics.
    For built-up: use high-value low-saturation heuristics.
    Otherwise: use adaptive thresholding on grayscale.
    """
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    t = target.lower()

    if any(kw in t for kw in ["flood", "water", "inundated", "lake", "river"]):
        # Water-like: blue-ish hue, low brightness or dark
        mask = (
            (hsv[:, :, 0] >= 85) & (hsv[:, :, 0] <= 140) &
            (hsv[:, :, 1] >= 30)
        ).astype(np.uint8) * 255

    elif any(kw in t for kw in ["built", "building", "urban", "road", "structure"]):
        # Built-up: high brightness, low saturation
        mask = (
            (hsv[:, :, 2] >= 130) &
            (hsv[:, :, 1] <= 70)
        ).astype(np.uint8) * 255

    elif any(kw in t for kw in ["vegetation", "field", "forest", "crop", "green"]):
        # Vegetation: green hue
        mask = (
            (hsv[:, :, 0] >= 35) & (hsv[:, :, 0] <= 85) &
            (hsv[:, :, 1] >= 40)
        ).astype(np.uint8) * 255

    else:
        # Generic: Otsu threshold
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (11, 11), 0)
        _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = int(h * w * threshold)
    contours = [c for c in contours if cv2.contourArea(c) >= min_area]
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:6]

    boxes: list[BoundingBox] = []
    polygons: list[GeoPolygon] = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        frac = area / (h * w)
        conf = min(0.55 + frac * 2.0, 0.97)
        x, y, cw, ch = cv2.boundingRect(cnt)

        boxes.append(BoundingBox(
            x1=x / w, y1=y / h,
            x2=(x + cw) / w, y2=(y + ch) / h,
            confidence=round(conf, 3),
            label=f"candidate {target[:20]}",
        ))

        # Approximate polygon from contour (simplified)
        eps = 0.01 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, eps, True)
        coords = [[float(pt[0][0]) / w, float(pt[0][1]) / h] for pt in approx]
        if len(coords) >= 3:
            polygons.append(GeoPolygon(
                coordinates=coords,
                confidence=round(conf, 3),
                label=f"candidate {target[:20]}",
                area_km2_approx=round(frac * 10.0, 3),
                is_demo_georef=True,
            ))

    return boxes, polygons, mask


def _draw_overlay(
    img: np.ndarray,
    boxes: list[BoundingBox],
    mask: np.ndarray,
) -> np.ndarray:
    h, w = img.shape[:2]
    overlay = img.copy()

    # Tint mask region cyan
    if mask is not None:
        tint = np.zeros_like(img)
        tint[:, :] = (255, 200, 0)  # BGR: cyan-ish
        tinted = cv2.addWeighted(overlay, 0.6, tint, 0.4, 0)
        overlay[mask > 0] = tinted[mask > 0]

    # Draw bounding boxes
    for b in boxes:
        x1 = int(b.x1 * w)
        y1 = int(b.y1 * h)
        x2 = int(b.x2 * w)
        y2 = int(b.y2 * h)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{b.label} {b.confidence:.2f}"
        cv2.putText(
            overlay, label, (x1, max(y1 - 6, 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1,
        )

    return overlay
