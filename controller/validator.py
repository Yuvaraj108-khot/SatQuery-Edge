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
        ch = arr.shape[2] if arr.ndim == 3 else 1

        if w < 16 or h < 16:
            passed = False
            reasons.append(f"Image {idx+1}: Dimensions too small ({w}×{h}).")
            quality_scores[label] = 0.0
            continue

        # Convert to single-channel gray for intensity stats
        if arr.ndim == 3 and arr.shape[2] >= 3:
            gray = cv2.cvtColor(arr[:, :, :3], cv2.COLOR_BGR2GRAY)
        elif arr.ndim == 3:
            gray = arr[:, :, 0]
        else:
            gray = arr

        mean_val = float(np.mean(gray))
        std_val = float(np.std(gray))
        variance = float(np.var(gray))

        # Check nodata / zero pixels
        zero_pct = float(np.sum(gray == 0)) / (w * h) * 100.0
        if zero_pct > 60.0:
            warnings.append(f"Image {idx+1}: High zero/nodata pixel percentage ({zero_pct:.1f}%).")

        # Black image check
        if mean_val < 2.0 and zero_pct > 90.0:
            passed = False
            reasons.append(f"Image {idx+1}: Appears completely black / empty (mean={mean_val:.1f}).")
            quality_scores[label] = 0.1
            continue

        # White / saturated check
        if mean_val > 252.0:
            passed = False
            reasons.append(f"Image {idx+1}: Appears completely white / saturated (mean={mean_val:.1f}).")
            quality_scores[label] = 0.1
            continue

        # Near-zero variance check
        if variance < 5.0 and zero_pct < 50.0:
            passed = False
            reasons.append(
                f"Image {idx+1}: Extremely low variance ({variance:.1f}). "
                "Image may be a solid uniform fill."
            )
            quality_scores[label] = 0.2
            continue

        # Low contrast warning
        if std_val < 15.0:
            warnings.append(
                f"Image {idx+1}: Low dynamic contrast (std={std_val:.1f}). "
                "Feature extraction precision may be reduced."
            )

        # Quality score: normalized from 0–1
        q = min(std_val / 60.0, 1.0) * 0.4 + min(variance / 3600.0, 1.0) * 0.4 + (1.0 - min(zero_pct / 100.0, 1.0)) * 0.2
        quality_scores[label] = round(q, 3)
        reasons.append(f"Image {idx+1}: Quality OK ({w}×{h}, {ch} channel(s), mean={mean_val:.0f}, std={std_val:.1f}).")

    # ── Pair validation & spatial CRS checks ──────────────────────────────────
    if required_count >= 2 and len(images) >= 2 and passed:
        arr0 = _to_numpy(images[0])
        arr1 = _to_numpy(images[1])
        if arr0 is not None and arr1 is not None:
            h0, w0 = arr0.shape[:2]
            h1, w1 = arr1.shape[:2]
            if (w0, h0) == (w1, h1):
                reasons.append("Image pair: identical dimensions — perfect pixel grid alignment.")
            else:
                ar0 = w0 / h0
                ar1 = w1 / h1
                ratio_diff = abs(ar0 - ar1)
                if ratio_diff > 0.3:
                    warnings.append(
                        f"Image pair: aspect ratios differ significantly "
                        f"({ar0:.2f} vs {ar1:.2f}). "
                        "Sub-sampling will re-align geometry."
                    )
                else:
                    reasons.append(
                        f"Image pair: dimensions differ ({w0}×{h0} vs {w1}×{h1}). "
                        "Post-event image will be resampled to match pre-event reference."
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
