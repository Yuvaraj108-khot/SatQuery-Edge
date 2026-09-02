"""
controller/registry.py
Central tool registry for SatQuery-Edge.
All specialist tools must be registered here.
The controller exclusively routes execution through this registry.
"""

from __future__ import annotations
from typing import Callable, Any
from controller.schemas import ToolRequest, ToolResult

# Import specialist tools
from tools.vqa import run_vqa
from tools.grounding import run_grounding
from tools.change import run_change
from tools.optical_sar import run_optical_sar
from tools.geospatial import run_geospatial

# ─── Registry Map ─────────────────────────────────────────────────────────────

TOOLS: dict[str, Callable[..., ToolResult]] = {
    "vqa":        run_vqa,
    "grounding":  run_grounding,
    "change":     run_change,
    "optical_sar": run_optical_sar,
    "geospatial": run_geospatial,
}


def execute_tool(request: ToolRequest, images: list[Any]) -> ToolResult:
    """
    Execute a registered specialist tool by name.
    Raises ValueError if the tool is not registered.
    No arbitrary tool execution is permitted.
    """
    tool_name = request.tool_name
    if tool_name not in TOOLS:
        return ToolResult(
            tool_name=tool_name,
            success=False,
            description="",
            error=f"Tool '{tool_name}' is not registered in the registry.",
        )
    try:
        fn = TOOLS[tool_name]
        return fn(images=images, parameters=request.parameters)
    except Exception as exc:
        return ToolResult(
            tool_name=tool_name,
            success=False,
            description="",
            error=f"Tool execution error: {exc}",
        )


def list_registered_tools() -> list[str]:
    return list(TOOLS.keys())
