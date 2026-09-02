"""
controller/router.py
Deterministic keyword-based intent router for SatQuery-Edge.
Returns a strict TaskGraph Pydantic object.
"""

from __future__ import annotations
from .schemas import TaskGraph

# ─── Keyword Tables ───────────────────────────────────────────────────────────

_VQA_KEYWORDS = {
    "what", "describe", "is there", "how many", "identify", "analysis",
    "classification", "classify", "label", "detect", "count",
}

_GROUNDING_KEYWORDS = {
    "where", "locate", "find", "show location", "highlight", "mark",
    "point to", "show me where", "bounding box",
}

_CHANGE_KEYWORDS = {
    "change", "changed", "before", "after", "newly", "flooded", "flood",
    "damage", "temporal", "difference", "delta", "progression",
    "post event", "pre event", "comparison", "compare",
}

_OPTICAL_SAR_KEYWORDS = {
    "sar", "optical", "radar", "fusion", "combine", "multi sensor",
    "multisensor", "synthetic aperture", "backscatter", "dual sensor",
}


def _score(query_lower: str, keywords: set[str]) -> int:
    return sum(1 for kw in keywords if kw in query_lower)


def route_query(query: str, num_images: int = 1) -> TaskGraph:
    """
    Deterministically route a natural-language query to an intent + tool set.
    
    Priority order (highest → lowest):
      optical_sar_fusion > bi_temporal_change > text_guided_grounding > single_image_vqa
    """
    q = query.lower().strip()

    scores = {
        "optical_sar_fusion":   _score(q, _OPTICAL_SAR_KEYWORDS),
        "bi_temporal_change":   _score(q, _CHANGE_KEYWORDS),
        "text_guided_grounding": _score(q, _GROUNDING_KEYWORDS),
        "single_image_vqa":     _score(q, _VQA_KEYWORDS),
    }

    # Enforce image-count constraints
    if num_images < 2:
        scores["optical_sar_fusion"] = 0
        scores["bi_temporal_change"] = 0

    best_intent = max(scores, key=lambda k: scores[k])

    # If all scores are zero, fall back based on image count
    if scores[best_intent] == 0:
        best_intent = "bi_temporal_change" if num_images >= 2 else "single_image_vqa"

    # Build tool list and parameters per intent
    if best_intent == "bi_temporal_change":
        return TaskGraph(
            intent="bi_temporal_change",
            selected_tools=["change", "grounding", "geospatial"],
            parameters={
                "threshold": 0.15,
                "min_region_area": 200,
                "morph_kernel": 5,
                "target_description": _extract_target(q),
            },
            query=query,
            num_images_required=2,
        )

    elif best_intent == "optical_sar_fusion":
        return TaskGraph(
            intent="optical_sar_fusion",
            selected_tools=["optical_sar", "grounding", "geospatial"],
            parameters={
                "optical_weight": 0.6,
                "sar_weight": 0.4,
                "target_description": _extract_target(q),
            },
            query=query,
            num_images_required=2,
        )

    elif best_intent == "text_guided_grounding":
        return TaskGraph(
            intent="text_guided_grounding",
            selected_tools=["grounding", "geospatial"],
            parameters={
                "target_description": _extract_target(q),
                "threshold": 0.10,
            },
            query=query,
            num_images_required=1,
        )

    else:  # single_image_vqa
        return TaskGraph(
            intent="single_image_vqa",
            selected_tools=["vqa", "grounding", "geospatial"],
            parameters={
                "target_description": _extract_target(q),
            },
            query=query,
            num_images_required=1,
        )


def _extract_target(query_lower: str) -> str:
    """Heuristically extract a noun phrase target from the query."""
    targets = [
        "flooded built-up areas", "flood", "water", "built-up", "roads",
        "vegetation", "damage", "change", "region",
    ]
    for t in targets:
        if t in query_lower:
            return t
    return "regions of interest"
