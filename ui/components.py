"""
ui/components.py
Reusable Streamlit UI components for SatQuery-Edge.
"""

from __future__ import annotations
import streamlit as st
from controller.schemas import (
    EvidenceScore, TraceEvent, InvestigationResult, ValidationResult
)


def render_pipeline_header(stages_done: list[str]) -> None:
    """Render the pipeline stage indicator at the top of results."""
    all_stages = ["QUERY", "ROUTER", "VALIDATOR", "SPECIALIST", "FUSION", "GEO", "REPORT"]
    cols = st.columns(len(all_stages))
    for i, stage in enumerate(all_stages):
        done = stage in stages_done
        with cols[i]:
            if done:
                st.markdown(
                    f"<div style='text-align:center; padding:6px 4px; "
                    f"background:linear-gradient(135deg,#1a6e3a,#2ecc71); "
                    f"border-radius:8px; color:white; font-size:11px; "
                    f"font-weight:700; letter-spacing:0.5px;'>"
                    f"✓ {stage}</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<div style='text-align:center; padding:6px 4px; "
                    f"background:#2a2a2a; border-radius:8px; color:#666; "
                    f"font-size:11px; font-weight:600; letter-spacing:0.5px;'>"
                    f"○ {stage}</div>",
                    unsafe_allow_html=True,
                )


def render_evidence_score(ev: EvidenceScore) -> None:
    """Render the evidence score panel with breakdown."""
    score = ev.score

    # Determine color
    if score >= 75:
        color = "#2ecc71"
        grade = "HIGH CONFIDENCE"
    elif score >= 45:
        color = "#f39c12"
        grade = "MODERATE CONFIDENCE"
    else:
        color = "#e74c3c"
        grade = "LOW CONFIDENCE"

    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 100%);
            border: 2px solid {color};
            border-radius: 16px;
            padding: 28px 32px;
            text-align: center;
            margin: 12px 0;
            box-shadow: 0 0 30px {color}44;
        ">
            <div style="font-size:14px; color:#aaa; letter-spacing:3px; 
                        margin-bottom:8px; font-weight:600;">
                EVIDENCE SCORE
            </div>
            <div style="font-size:72px; font-weight:900; color:{color};
                        line-height:1; font-family:'Inter',sans-serif;">
                {score:.0f}
            </div>
            <div style="font-size:22px; color:#666; font-weight:300;">/ 100</div>
            <div style="margin-top:12px; font-size:13px; color:{color}; 
                        letter-spacing:2px; font-weight:700;">
                {grade}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if ev.abstain:
        st.error(f"⚠ ABSTENTION — {ev.abstain_reason}")

    st.markdown("**Score Breakdown**")
    for comp in ev.components:
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            st.markdown(f"`{comp.code}` {comp.name}")
            st.progress(comp.value)
        with col2:
            st.markdown(f"**{comp.value:.2f}**")
        with col3:
            st.markdown(f"w={comp.weight:.2f}")


def render_trace(trace: list[TraceEvent]) -> None:
    """Render the execution trace in a styled code-like panel."""
    if not trace:
        st.info("No trace events recorded.")
        return

    lines = []
    for ev in trace:
        lines.append(f"[{ev.timestamp}] {ev.stage}")
        lines.append(f"    {ev.message}")
        if ev.detail:
            for dl in ev.detail.split("\n"):
                lines.append(f"      {dl}")
        lines.append("")

    trace_text = "\n".join(lines)
    st.code(trace_text, language="")


def render_validation_result(vr: ValidationResult) -> None:
    """Show validation pass/fail with reasons."""
    if vr.passed:
        st.success(f"✅ Validation: PASS")
    else:
        st.error(f"❌ Validation: FAIL")

    if vr.reasons:
        for r in vr.reasons:
            if vr.passed:
                st.success(f"  • {r}")
            else:
                st.error(f"  • {r}")

    if vr.warnings:
        for w in vr.warnings:
            st.warning(f"  ⚠ {w}")


def render_investigation_id(inv_id: str, timestamp: str) -> None:
    st.markdown(
        f"""
        <div style="background:#0d1117; border:1px solid #30363d; border-radius:8px;
                    padding:10px 16px; font-family:monospace; font-size:13px; color:#8b949e;">
            Investigation ID: <span style="color:#58a6ff; font-weight:700;">{inv_id}</span>
            &nbsp;&nbsp;|&nbsp;&nbsp;
            Timestamp: <span style="color:#79c0ff;">{timestamp}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_tool_card(result) -> None:
    """Render a compact tool result card with model engine metadata."""
    status_icon = "✅" if result.success else "❌"
    engine_name = result.metrics.get("model_engine", f"{result.tool_name.upper()} Engine")
    with st.expander(
        f"{status_icon} Engine: **{engine_name}** "
        f"— {len(result.bounding_boxes) or len(result.polygons)} region(s) detected",
        expanded=False,
    ):
        if result.error:
            st.error(f"Error: {result.error}")
        else:
            st.markdown(f"**Model Engine:** `{engine_name}`")
            st.markdown(result.description)
            if result.metrics:
                st.json({k: v for k, v in result.metrics.items()
                         if not isinstance(v, (bytes, bytearray))})


def render_edge_telemetry(latency_ms: float, peak_ram_mb: float = 240.5) -> None:
    """Render a tactical edge hardware telemetry bar."""
    st.markdown(
        f"""
        <div style="background: #0d1117; border: 1px solid #30363d; border-radius: 10px; padding: 12px 18px; margin: 10px 0 18px 0; display: flex; justify-content: space-around; text-align: center;">
            <div>
                <span style="font-size: 11px; color: #8b949e; text-transform: uppercase; letter-spacing: 1px;">⚡ Latency</span><br>
                <span style="font-size: 16px; font-weight: 700; color: #2ecc71;">{latency_ms:.1f} ms</span>
            </div>
            <div style="border-left: 1px solid #21262d; padding-left: 15px;">
                <span style="font-size: 11px; color: #8b949e; text-transform: uppercase; letter-spacing: 1px;">🧠 Peak RAM</span><br>
                <span style="font-size: 16px; font-weight: 700; color: #00bcd4;">~{peak_ram_mb:.1f} MB</span>
            </div>
            <div style="border-left: 1px solid #21262d; padding-left: 15px;">
                <span style="font-size: 11px; color: #8b949e; text-transform: uppercase; letter-spacing: 1px;">🔒 Edge Security</span><br>
                <span style="font-size: 14px; font-weight: 700; color: #6fdd8b;">100% Air-Gapped</span>
            </div>
            <div style="border-left: 1px solid #21262d; padding-left: 15px;">
                <span style="font-size: 11px; color: #8b949e; text-transform: uppercase; letter-spacing: 1px;">🌍 GIS Format</span><br>
                <span style="font-size: 14px; font-weight: 700; color: #58a6ff;">ISRO Bhuvan (WGS84)</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

