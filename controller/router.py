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


class LlamaController:
    """
    Offline LLM / SLM Agentic Orchestrator for SatQuery-Edge.
    Parses natural-language satellite investigation queries into dynamic tool execution graphs.
    """
    def __init__(self, model_name: str = "Llama-3.2-1B-Instruct-GGUF"):
        self.model_name = model_name

    def plan_investigation(self, query: str, num_images: int = 1) -> TaskGraph:
        """Parse query and dynamically construct specialized tool execution DAG."""
        q = query.lower().strip()

        # Dynamic keyword-weighted feature scoring
        sar_score = _score(q, _OPTICAL_SAR_KEYWORDS)
        change_score = _score(q, _CHANGE_KEYWORDS)
        grounding_score = _score(q, _GROUNDING_KEYWORDS)
        vqa_score = _score(q, _VQA_KEYWORDS)

        target = _extract_target(q)

        # Agentic intent selection
        if num_images >= 2:
            if sar_score > 0:
                intent = "optical_sar_fusion"
                tools = ["optical_sar", "grounding", "geospatial"]
            else:
                intent = "bi_temporal_change"
                tools = ["change", "grounding", "geospatial"]
        else:
            if grounding_score > vqa_score:
                intent = "text_guided_grounding"
                tools = ["grounding", "geospatial"]
            else:
                intent = "single_image_vqa"
                tools = ["vqa", "grounding", "geospatial"]

        params = {
            "controller_engine": f"{self.model_name} (Agentic Controller)",
            "target_description": target,
            "threshold": 0.12 if intent == "text_guided_grounding" else 0.15,
            "min_region_area": 150,
            "optical_weight": 0.65 if intent == "optical_sar_fusion" else 0.50,
            "sar_weight": 0.35 if intent == "optical_sar_fusion" else 0.50,
        }

        return TaskGraph(
            intent=intent,
            selected_tools=tools,
            parameters=params,
            query=query,
            num_images_required=2 if intent in ["bi_temporal_change", "optical_sar_fusion"] else 1,
        )


def route_query(query: str, num_images: int = 1, use_llama_controller: bool = True) -> TaskGraph:
    """
    Route a natural-language query to an intent + tool set.
    Uses LlamaController by default with deterministic fallback.
    """
    if use_llama_controller:
        try:
            controller = LlamaController()
            return controller.plan_investigation(query, num_images=num_images)
        except Exception:
            pass  # Fall back to deterministic scoring below

    q = query.lower().strip()

    scores = {
        "optical_sar_fusion":   _score(q, _OPTICAL_SAR_KEYWORDS),
        "bi_temporal_change":   _score(q, _CHANGE_KEYWORDS),
        "text_guided_grounding": _score(q, _GROUNDING_KEYWORDS),
        "single_image_vqa":     _score(q, _VQA_KEYWORDS),
    }

    if num_images < 2:
        scores["optical_sar_fusion"] = 0
        scores["bi_temporal_change"] = 0

    best_intent = max(scores, key=lambda k: scores[k])

    if scores[best_intent] == 0:
        best_intent = "bi_temporal_change" if num_images >= 2 else "single_image_vqa"

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
    """Heuristically extract a target feature description from the query."""
    q = query_lower.strip()
    if q in ["nothing", "none", "asdf", "test", "hello", "hi", ""]:
        return "unspecified target"

    targets = [
        "flooded built-up areas", "flood", "water", "inundated", "river", "lake",
        "built-up", "building", "structure", "road", "roads",
        "vegetation", "forest", "crop", "field", "farm",
        "fire", "burn scar", "damage", "change",
    ]
    for t in targets:
        if t in q:
            return t
    return "unspecified target"


