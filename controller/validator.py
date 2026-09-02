"""
controller/validator.py
Pre-processing validation for SatQuery-Edge.
Checks image count, quality, and pair compatibility.
"""

from __future__ import annotations
from typing import Any
import numpy as np
import cv2
from controller.schemas import ValidationResult


def validate_inputs(
    images: list[Any],
    required_count: int,
) -> ValidationResult:
    """
    Validate images for the selected intent.
    Returns ValidationResult with PASS or FAIL + reasons.
    """
    reasons: list[str] = []
    warnings: list[str] = []
    quality_scores: dict[str, float] = {}
    passed = True

    # ── Check image count ────────────────────────────────────────────────────
    actual = len(images)
    if actual < required_count:
        passed = False
        reasons.append(
            f"Task requires {required_count} image(s), but {actual} provided. "
            "Please upload the missing image(s)."
        )
        return ValidationResult(
            passed=False, reasons=reasons, warnings=warnings,
            image_quality_scores=quality_scores,
        )

    reasons.append(f"{actual} image(s) detected — count OK.")

    # ── Per-image quality checks ──────────────────────────────────────────────
    for idx, img in enumerate(images[:required_count]):
        label = f"image_{idx+1}"
        try:
            arr = _to_numpy(img)
        except Exception:
            passed = False
            reasons.append(f"Image {idx+1}: Could not decode. Unreadable or unsupported format.")
            quality_scores[label] = 0.0
            continue

        if arr is None:
            passed = False
            reasons.append(f"Image {idx+1}: Decode returned None.")
            quality_scores[label] = 0.0
            continue

        h, w = arr.shape[:2]
        if w < 16 or h < 16:
            passed = False
            reasons.append(f"Image {idx+1}: Dimensions too small ({w}×{h}).")
            quality_scores[label] = 0.0
            continue

        gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        mean_val = float(np.mean(gray))
        std_val = float(np.std(gray))
        variance = float(np.var(gray))

        # Black image
        if mean_val < 5.0:
            passed = False
            reasons.append(f"Image {idx+1}: Appears completely black (mean={mean_val:.1f}).")
            quality_scores[label] = 0.1
            continue

        # White image
        if mean_val > 250.0:
            passed = False
            reasons.append(f"Image {idx+1}: Appears completely white (mean={mean_val:.1f}).")
            quality_scores[label] = 0.1
            continue

        # Near-zero variance
        if variance < 10.0:
            passed = False
            reasons.append(
                f"Image {idx+1}: Near-zero variance ({variance:.1f}). "
                "Image may be a solid fill."
            )
            quality_scores[label] = 0.2
            continue

        # Low contrast warning
        if std_val < 20.0:
            warnings.append(
                f"Image {idx+1}: Low contrast (std={std_val:.1f}). "
                "Results may be unreliable."
            )

        # Quality score: normalized from 0–1
        q = min(std_val / 60.0, 1.0) * 0.5 + min(variance / 3600.0, 1.0) * 0.5
        quality_scores[label] = round(q, 3)
        reasons.append(f"Image {idx+1}: Quality OK (mean={mean_val:.0f}, std={std_val:.1f}).")

    # ── Pair validation ───────────────────────────────────────────────────────
    if required_count >= 2 and len(images) >= 2 and passed:
        arr0 = _to_numpy(images[0])
        arr1 = _to_numpy(images[1])
        if arr0 is not None and arr1 is not None:
            h0, w0 = arr0.shape[:2]
            h1, w1 = arr1.shape[:2]
            if (w0, h0) == (w1, h1):
                reasons.append("Image pair: identical dimensions — perfect alignment.")
            else:
                ar0 = w0 / h0
                ar1 = w1 / h1
                ratio_diff = abs(ar0 - ar1)
                if ratio_diff > 0.3:
                    warnings.append(
                        f"Image pair: aspect ratios differ significantly "
                        f"({ar0:.2f} vs {ar1:.2f}). "
                        "Change detection may be less accurate. Images will be resized."
                    )
                else:
                    reasons.append(
                        f"Image pair: dimensions differ ({w0}×{h0} vs {w1}×{h1}). "
                        "Post-event will be resized to match pre-event."
                    )

    return ValidationResult(
        passed=passed,
        reasons=reasons,
        warnings=warnings,
        image_quality_scores=quality_scores,
    )


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
