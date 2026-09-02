"""
evidence/fusion.py
Evidence fusion module — SatQuery-Edge.
Combines specialist tool outputs into five sub-dimensions:
  M — Specialist-model agreement
  S — Sensor reliability
  P — Spatial consistency
  T — Temporal consistency
  I — Input-quality validity
"""

from __future__ import annotations
from typing import Any
from controller.schemas import (
    ToolResult, ValidationResult, TaskGraph,
    EvidenceComponent, EvidenceScore,
)


def fuse_evidence(
    task_graph: TaskGraph,
    tool_results: list[ToolResult],
    validation: ValidationResult,
) -> EvidenceScore:
    """
    Combine all specialist outputs into a structured evidence score.
    Not a simple average — each component reflects different quality axes.
    """
    intent = task_graph.intent

    M = _compute_model_agreement(tool_results, intent)
    S = _compute_sensor_reliability(tool_results, intent)
    P = _compute_spatial_consistency(tool_results)
    T = _compute_temporal_consistency(tool_results, intent)
    I = _compute_input_quality(validation)

    components = [
        EvidenceComponent(name="Model Agreement",        code="M", value=round(M, 3),
                          weight=0.30, description="Agreement across specialist tool outputs"),
        EvidenceComponent(name="Sensor Reliability",     code="S", value=round(S, 3),
                          weight=0.20, description="Estimated reliability of input sensor type"),
        EvidenceComponent(name="Spatial Consistency",    code="P", value=round(P, 3),
                          weight=0.20, description="Overlap and agreement of detected spatial regions"),
        EvidenceComponent(name="Temporal Consistency",   code="T", value=round(T, 3),
                          weight=0.15, description="Consistency of detected change across temporal pair"),
        EvidenceComponent(name="Input Quality",          code="I", value=round(I, 3),
                          weight=0.15, description="Image quality and validation gate result"),
    ]

    from evidence.score import compute_score
    final = compute_score(components)
    abstain, reason = _check_abstention(final, components, tool_results)

    return EvidenceScore(
        score=round(final, 1),
        components=components,
        abstain=abstain,
        abstain_reason=reason,
    )

# ─── Abstention check ─────────────────────────────────────────────────────────

def _check_abstention(
    score: float,
    components: list[EvidenceComponent],
    tool_results: list[ToolResult],
) -> tuple[bool, str]:
    """
    Decide whether the system should abstain from drawing a conclusion.
    Returns (abstain, reason).
    """
    from evidence.score import should_abstain
    return should_abstain(score, components)


# ─── Sub-dimension computations ───────────────────────────────────────────────

def _compute_model_agreement(results: list[ToolResult], intent: str) -> float:
    """
    Agreement between relevant specialist outputs.
    Measured as fraction of successful tools and confidence consistency.
    """
    if not results:
        return 0.3

    successful = [r for r in results if r.success]
    if not successful:
        return 0.2

    success_rate = len(successful) / len(results)

    # Collect max confidences from each tool
    confidences: list[float] = []
    for r in successful:
        boxes = r.bounding_boxes
        if boxes:
            confidences.append(max(b.confidence for b in boxes))
        elif r.metrics.get("max_confidence"):
            confidences.append(float(r.metrics["max_confidence"]))
        elif r.metrics.get("region_count", 0) > 0:
            confidences.append(0.75)

    if not confidences:
        base = 0.5
    else:
        avg_conf = sum(confidences) / len(confidences)
        # Agreement = how similar are the confidence values?
        if len(confidences) > 1:
            spread = max(confidences) - min(confidences)
            agreement = 1.0 - spread
        else:
            agreement = avg_conf
        base = 0.5 * avg_conf + 0.5 * agreement

    return float(min(success_rate * 0.3 + base * 0.7, 1.0))


def _compute_sensor_reliability(results: list[ToolResult], intent: str) -> float:
    """
    Deterministic sensor reliability based on intent type.
    Optical imagery: 0.85–0.92 (established, high reliability for optical).
    SAR-like synthetic: 0.80 (lower for demo).
    Two-image temporal: bonus for having two observations.
    """
    base_reliabilities = {
        "bi_temporal_change":    0.88,
        "optical_sar_fusion":    0.82,
        "text_guided_grounding": 0.86,
        "single_image_vqa":      0.84,
    }
    base = base_reliabilities.get(intent, 0.82)

    # Small boost if change tool found regions
    for r in results:
        if r.tool_name == "change" and r.success:
            rc = r.metrics.get("region_count", 0)
            if rc >= 3:
                base = min(base + 0.04, 0.96)
            elif rc >= 1:
                base = min(base + 0.02, 0.96)

    return float(base)


def _compute_spatial_consistency(results: list[ToolResult]) -> float:
    """
    How well do detected regions from different tools overlap?
    Compare boxes from change vs grounding where available.
    """
    all_box_sets: list[list[Any]] = []
    for r in results:
        if r.success and r.bounding_boxes:
            all_box_sets.append(r.bounding_boxes)

    if not all_box_sets:
        return 0.65

    if len(all_box_sets) == 1:
        # Single tool — score by confidence uniformity
        boxes = all_box_sets[0]
        if not boxes:
            return 0.60
        avg_conf = sum(b.confidence for b in boxes) / len(boxes)
        return float(min(avg_conf + 0.05, 0.98))

    # Multiple tools: check centroid proximity
    def centroid(boxes: list[Any]) -> tuple[float, float]:
        cx = sum((b.x1 + b.x2) / 2 for b in boxes) / len(boxes)
        cy = sum((b.y1 + b.y2) / 2 for b in boxes) / len(boxes)
        return cx, cy

    centroids = [centroid(bs) for bs in all_box_sets]
    distances: list[float] = []
    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            d = ((centroids[i][0] - centroids[j][0])**2 +
                 (centroids[i][1] - centroids[j][1])**2) ** 0.5
            distances.append(d)

    avg_dist = sum(distances) / len(distances) if distances else 0.5
    # Close centroids → high consistency
    consistency = max(0.50, 1.0 - avg_dist * 2.0)
    return float(min(consistency, 0.98))


def _compute_temporal_consistency(results: list[ToolResult], intent: str) -> float:
    """
    For temporal change: measure consistency based on change statistics.
    For non-temporal intents, return a default moderate-high value.
    """
    if intent != "bi_temporal_change":
        return 0.82

    for r in results:
        if r.tool_name == "change" and r.success:
            changed_pct = float(r.metrics.get("changed_pixel_percent", 0.0))
            region_count = int(r.metrics.get("region_count", 0))

            # Changed 5–25%: highly plausible flooding signature
            if 5.0 <= changed_pct <= 30.0:
                plausibility = 0.90
            elif changed_pct < 5.0:
                plausibility = 0.65
            else:
                plausibility = 0.78

            region_bonus = min(region_count * 0.02, 0.08)
            return float(min(plausibility + region_bonus, 0.98))

    return 0.72


def _compute_input_quality(validation: ValidationResult) -> float:
    """Input quality score based on the validation gate."""
    if not validation.passed:
        # Fail → low but non-zero score
        return 0.30

    # Warnings reduce score slightly
    warning_penalty = len(validation.warnings) * 0.03
    quality_scores = list(validation.image_quality_scores.values())
    if quality_scores:
        avg_quality = sum(quality_scores) / len(quality_scores)
    else:
        avg_quality = 0.90

    return float(min(max(avg_quality - warning_penalty, 0.40), 1.0))
