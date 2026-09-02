"""
evidence/score.py
Evidence Score computation for SatQuery-Edge.

Formula:
  E = 100 × (w_m*M + w_s*S + w_p*P + w_t*T + w_i*I)
            / (w_m + w_s + w_p + w_t + w_i)

Default weights:
  w_m = 0.30   (Model Agreement)
  w_s = 0.20   (Sensor Reliability)
  w_p = 0.20   (Spatial Consistency)
  w_t = 0.15   (Temporal Consistency)
  w_i = 0.15   (Input Quality)
"""

from __future__ import annotations
from controller.schemas import EvidenceComponent, EvidenceScore

ABSTENTION_THRESHOLD = 45.0   # Score below this → abstain


def compute_score(components: list[EvidenceComponent]) -> float:
    """
    Apply the weighted evidence formula and return a score in [0, 100].
    """
    if not components:
        return 0.0

    numerator = sum(c.value * c.weight for c in components)
    denominator = sum(c.weight for c in components)
    if denominator == 0:
        return 0.0

    score = 100.0 * numerator / denominator
    return round(min(max(score, 0.0), 100.0), 2)


def should_abstain(score: float, components: list[EvidenceComponent]) -> tuple[bool, str]:
    """
    Determine whether the system should abstain from making a conclusion.
    Returns (abstain: bool, reason: str).
    """
    if score < ABSTENTION_THRESHOLD:
        return True, (
            f"Evidence Score ({score:.1f}) is below the abstention threshold "
            f"({ABSTENTION_THRESHOLD}). Insufficient evidence to reliably conclude."
        )

    # Check for strong disagreement (any component < 0.30)
    low_components = [c for c in components if c.value < 0.30]
    if low_components:
        names = ", ".join(c.name for c in low_components)
        return True, (
            f"Strong specialist disagreement detected in: {names}. "
            "Cannot reliably conclude."
        )

    return False, ""
