"""
controller/schemas.py
Pydantic v2 strict schemas for SatQuery-Edge pipeline.
"""

from __future__ import annotations
from typing import Any, Optional
from pydantic import BaseModel, Field, field_validator
import datetime


# ─── Image / Observation ──────────────────────────────────────────────────────

class ImageObservation(BaseModel):
    """Represents a loaded satellite or optical image."""
    image_id: str
    filename: str
    width: int
    height: int
    channels: int
    file_size_bytes: int
    sensor: str = "Unknown / Demo Optical"
    acquisition_date: str = "Unknown / Demo Date"
    crs: str = "Approximate Demo CRS"
    format: str = "PNG"


class ValidationResult(BaseModel):
    """Result of pre-processing validation."""
    passed: bool
    reasons: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    image_quality_scores: dict[str, float] = Field(default_factory=dict)

    @property
    def status(self) -> str:
        return "PASS" if self.passed else "FAIL"


# ─── Spatial Primitives ───────────────────────────────────────────────────────

class BoundingBox(BaseModel):
    """Normalized bounding box (0–1 per dimension)."""
    x1: float = Field(ge=0.0, le=1.0)
    y1: float = Field(ge=0.0, le=1.0)
    x2: float = Field(ge=0.0, le=1.0)
    y2: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    label: str = "region"


class GeoPolygon(BaseModel):
    """GeoJSON-like polygon with approximate coordinates."""
    coordinates: list[list[float]]   # list of [lon, lat] pairs
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    label: str = "candidate region"
    area_km2_approx: float = 0.0
    is_demo_georef: bool = True


# ─── Tool I/O ─────────────────────────────────────────────────────────────────

class ToolRequest(BaseModel):
    """Input specification for a specialist tool."""
    tool_name: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    image_ids: list[str] = Field(default_factory=list)


class ToolResult(BaseModel):
    """Output from a specialist tool."""
    tool_name: str
    success: bool
    description: str = ""
    bounding_boxes: list[BoundingBox] = Field(default_factory=list)
    polygons: list[GeoPolygon] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    image_outputs: dict[str, Any] = Field(default_factory=dict)   # key → numpy array or PIL image
    raw_data: dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None

    model_config = {"arbitrary_types_allowed": True}


# ─── Task Graph ───────────────────────────────────────────────────────────────

class TaskGraph(BaseModel):
    """The controller's plan for executing an investigation."""
    intent: str
    selected_tools: list[str]
    parameters: dict[str, Any] = Field(default_factory=dict)
    query: str = ""
    num_images_required: int = 1


# ─── Evidence ─────────────────────────────────────────────────────────────────

class EvidenceComponent(BaseModel):
    """A single sub-dimension of the evidence score."""
    name: str
    code: str            # M, S, P, T, I
    value: float = Field(ge=0.0, le=1.0)
    weight: float = Field(ge=0.0, le=1.0)
    description: str = ""


class EvidenceScore(BaseModel):
    """Final fused evidence score."""
    score: float = Field(ge=0.0, le=100.0)
    components: list[EvidenceComponent]
    abstain: bool = False
    abstain_reason: str = ""


# ─── Investigation Result ─────────────────────────────────────────────────────

class InvestigationResult(BaseModel):
    """Complete result of a SatQuery-Edge investigation."""
    investigation_id: str
    timestamp: str
    query: str
    task_graph: TaskGraph
    validation: ValidationResult
    tool_results: list[ToolResult]
    evidence: EvidenceScore
    final_answer: str
    geo_polygons: list[GeoPolygon] = Field(default_factory=list)
    execution_trace: list["TraceEvent"] = Field(default_factory=list)

    model_config = {"arbitrary_types_allowed": True}


# ─── Trace ────────────────────────────────────────────────────────────────────

class TraceEvent(BaseModel):
    """A single step in the observable execution trace."""
    timestamp: str
    stage: str
    message: str
    detail: Optional[str] = None

    @classmethod
    def now(cls, stage: str, message: str, detail: Optional[str] = None) -> "TraceEvent":
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        return cls(timestamp=ts, stage=stage, message=message, detail=detail)
