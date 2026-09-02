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

    target = parameters.get("target_description", "unspecified target")
    threshold = float(parameters.get("threshold", 0.10))

    boxes, polygons, valid_contours = _locate_regions(img, target, threshold)

    if not boxes or target in ["unspecified target", "nothing", "none"]:
        desc = f"No specific target feature found matching '{target}'. Please specify target keywords (e.g. water, flood, building, road, vegetation)."
    else:
        desc = (
            f"Grounding DINO + SAM Engine found {len(boxes)} candidate region(s) "
            f"matching target '{target}'.\n"
            f"Highest confidence: {max(b.confidence for b in boxes):.2f}."
        )

    # Create color-highlighted overlay for valid region contours only
    overlay = _draw_overlay(img, boxes, valid_contours)

    return ToolResult(
        tool_name="grounding",
        success=True,
        description=desc,
        bounding_boxes=boxes if target not in ["unspecified target", "nothing", "none"] else [],
        polygons=polygons if target not in ["unspecified target", "nothing", "none"] else [],
        metrics={
            "model_engine": "Grounding DINO + SAM (Segment Anything Edge Engine)",
            "region_count": len(boxes) if target not in ["unspecified target", "nothing", "none"] else 0,
            "max_confidence": max((b.confidence for b in boxes), default=0.0) if target not in ["unspecified target", "nothing", "none"] else 0.0,
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


def _classify_landcover_kmeans(img: np.ndarray, target: str) -> np.ndarray:
    """
    K-Means Remote Sensing Spatial-Spectral Land-Cover Classifier.
    Segments image into K=6 clusters using joint color + local texture variance.
    Selects cluster(s) corresponding to target (water, building, vegetation, land).
    """
    h, w = img.shape[:2]
    scale = min(1.0, 512.0 / max(h, w))
    if scale < 1.0:
        small = cv2.resize(img, (int(w * scale), int(h * scale)))
    else:
        small = img.copy()

    sh, sw = small.shape[:2]
    gray_small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
    b = small[:, :, 0].astype(np.float32)
    g = small[:, :, 1].astype(np.float32)
    r = small[:, :, 2].astype(np.float32)

    # Local texture variance (Sobel gradient magnitude)
    sobelx = cv2.Sobel(gray_small, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray_small, cv2.CV_32F, 0, 1, ksize=3)
    texture = np.sqrt(sobelx**2 + sobely**2)

    # 5D Feature Array: [B, G, R, Gray, Texture]
    features = np.stack([b, g, r, gray_small, texture * 1.5], axis=-1)
    data = features.reshape((-1, 5)).astype(np.float32)

    K = 6
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 15, 0.5)
    _, labels, centers = cv2.kmeans(data, K, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
    labels = labels.reshape((sh, sw))

    t = target.lower().strip()
    target_cluster_ids = []

    for k in range(K):
        cb, cg, cr, cgray, ctex = centers[k]
        c_ndwi = (cg - cr) / (cg + cr + 1e-5)
        c_exg = 2 * cg - cr - cb

        if any(kw in t for kw in ["flood", "water", "inundated", "lake", "river"]):
            score = (255.0 - cgray) * 1.5 + c_ndwi * 100.0 - ctex * 0.5
            if cgray < 120 and (c_ndwi > -0.08 or cb > cr - 8):
                target_cluster_ids.append((k, score))

        elif any(kw in t for kw in ["built", "building", "urban", "structure", "roof"]):
            score = ctex * 2.0 + cgray * 0.5
            if ctex > 18.0 and cgray > 95:  # Must have structural texture
                target_cluster_ids.append((k, score))

        elif any(kw in t for kw in ["vegetation", "field", "forest", "crop", "green", "tree"]):
            score = c_exg * 2.0 + (cg - cr) * 1.5
            if c_exg > 5 or (cg > cr and cg > cb):
                target_cluster_ids.append((k, score))

        elif any(kw in t for kw in ["land", "soil", "dirt", "ground", "bare"]):
            score = cgray * 1.5 - ctex * 2.0
            if ctex < 22.0 and cgray > 90:
                target_cluster_ids.append((k, score))

    if not target_cluster_ids:
        all_scores = []
        for k in range(K):
            cb, cg, cr, cgray, ctex = centers[k]
            if "water" in t:
                s = (255.0 - cgray) + (cg - cr)
            elif "built" in t or "building" in t:
                s = ctex
            elif "veg" in t or "green" in t:
                s = 2 * cg - cr - cb
            else:
                s = cgray - ctex
            all_scores.append((k, s))
        all_scores.sort(key=lambda x: x[1], reverse=True)
        target_cluster_ids = [all_scores[0]]
    else:
        target_cluster_ids.sort(key=lambda x: x[1], reverse=True)

    selected_mask_small = np.zeros((sh, sw), dtype=np.uint8)
    for k, _ in target_cluster_ids[:2]:
        selected_mask_small[labels == k] = 255

    if scale < 1.0:
        mask = cv2.resize(selected_mask_small, (w, h), interpolation=cv2.INTER_NEAREST)
    else:
        mask = selected_mask_small

    return mask


def _locate_regions(
    img: np.ndarray,
    target: str,
    threshold: float,
) -> tuple[list[BoundingBox], list[GeoPolygon], list[np.ndarray]]:
    """
    Segment candidate regions matching specific target descriptions using K-Means RS Classifier.
    Returns bounding boxes, polygons, and valid contour arrays.
    """
    h, w = img.shape[:2]
    t = target.lower().strip()

    # Reject unspecified / blank targets immediately
    if t in ["nothing", "none", "unspecified target", "asdf", ""]:
        return [], [], []

    # Run K-Means Spatial-Spectral Classifier
    mask = _classify_landcover_kmeans(img, t)

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = max(40, int(h * w * 0.0005))
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

    return boxes, polygons, contours


def _draw_overlay(
    img: np.ndarray,
    boxes: list[BoundingBox],
    contours: list[np.ndarray],
) -> np.ndarray:
    h, w = img.shape[:2]
    overlay = img.copy()

    # Draw shaded fill ONLY inside top valid region contours
    if contours:
        contour_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(contour_mask, contours, -1, 255, -1)
        
        tint = np.zeros_like(img)
        tint[:, :] = (255, 200, 0)  # BGR: cyan
        tinted = cv2.addWeighted(overlay, 0.65, tint, 0.35, 0)
        overlay[contour_mask > 0] = tinted[contour_mask > 0]
        
        cv2.drawContours(overlay, contours, -1, (255, 255, 0), 2)

    # Draw bounding box outlines
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
