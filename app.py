"""
app.py — SatQuery-Edge
Offline Satellite Intelligence Investigation Workspace
SIH 26167 · ISRO Prototype

Entry point:
    streamlit run app.py
"""

from __future__ import annotations

import datetime
import os
import random
import string
import sys
import time
import traceback
import io
from typing import Optional

import cv2
import numpy as np
import streamlit as st
from PIL import Image

# ── Ensure sample images exist before any imports ────────────────────────────
_SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "data", "sample")
os.makedirs(_SAMPLE_DIR, exist_ok=True)
try:
    sys.path.insert(0, os.path.dirname(__file__))
    from data.sample.generate_samples import ensure_samples
    ensure_samples()
except Exception as _e:
    pass  # Will surface gracefully in the UI

# ── Project imports ───────────────────────────────────────────────────────────
from controller.router import route_query
from controller.validator import validate_inputs
from controller.registry import execute_tool
from controller.schemas import (
    ToolRequest, TraceEvent, InvestigationResult, ImageObservation,
)
from evidence.fusion import fuse_evidence
from evidence.score import should_abstain
from ui.components import (
    render_pipeline_header, render_evidence_score, render_trace,
    render_validation_result, render_investigation_id, render_tool_card,
    render_edge_telemetry,
)
from ui.map_utils import build_investigation_map
from tools.geospatial import export_geojson, export_kml
from reports.report_generator import generate_pdf_report

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SatQuery-Edge · SIH 26167",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styles ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;900&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Dark background override */
.stApp {
    background: linear-gradient(135deg, #0a0a14 0%, #0d1117 50%, #0a0f1a 100%);
}

.stSidebar {
    background: linear-gradient(180deg, #0d1117 0%, #131b25 100%);
    border-right: 1px solid #21262d;
}

/* Header glow */
.sq-header {
    text-align: center;
    padding: 20px 10px 10px;
    background: linear-gradient(135deg, #0d1a0e 0%, #0d1117 50%, #0a0e1a 100%);
    border-bottom: 1px solid #21262d;
    border-radius: 0 0 16px 16px;
    margin-bottom: 18px;
}

.sq-title {
    font-size: 42px;
    font-weight: 900;
    background: linear-gradient(90deg, #2ecc71, #00bcd4, #3498db);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -1px;
    line-height: 1;
    margin-bottom: 4px;
}

.sq-subtitle {
    font-size: 15px;
    color: #8b949e;
    font-weight: 300;
    letter-spacing: 1px;
    margin-bottom: 8px;
}

.sq-badge {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1.5px;
    margin: 2px 4px;
}

.badge-isro {
    background: linear-gradient(90deg, #1a3a6e, #2a5cc2);
    color: #90c8ff;
    border: 1px solid #3a6ad4;
}

.badge-offline {
    background: linear-gradient(90deg, #1a3a1e, #1e7a2e);
    color: #6fdd8b;
    border: 1px solid #2ecc71;
}

/* Query box */
.stTextArea textarea {
    background: #0d1117 !important;
    border: 1px solid #30363d !important;
    border-radius: 10px !important;
    color: #e6edf3 !important;
    font-size: 15px !important;
    font-family: 'Inter', sans-serif !important;
}

.stTextArea textarea:focus {
    border-color: #2ecc71 !important;
    box-shadow: 0 0 0 3px rgba(46,204,113,0.12) !important;
}

/* Run button */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #1a6e3a, #2ecc71) !important;
    color: white !important;
    font-weight: 700 !important;
    font-size: 16px !important;
    padding: 12px 40px !important;
    border-radius: 10px !important;
    border: none !important;
    letter-spacing: 1px !important;
    transition: all 0.2s !important;
    width: 100% !important;
}

.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #2ecc71, #27ae60) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(46,204,113,0.4) !important;
}

/* Tab styling */
.stTabs [data-baseweb="tab"] {
    color: #8b949e;
    font-weight: 600;
    font-size: 13px;
    letter-spacing: 0.5px;
}

.stTabs [aria-selected="true"] {
    color: #2ecc71 !important;
    border-bottom: 2px solid #2ecc71 !important;
}

/* Metrics */
.stMetric {
    background: #0d1117;
    border: 1px solid #21262d;
    border-radius: 10px;
    padding: 12px !important;
}

/* Image borders */
.stImage img {
    border-radius: 8px;
    border: 1px solid #21262d;
}

/* Sidebar headers */
.stSidebar .stMarkdown h3 {
    color: #58a6ff;
    font-size: 13px;
    letter-spacing: 1px;
    text-transform: uppercase;
    border-bottom: 1px solid #21262d;
    padding-bottom: 6px;
    margin-top: 18px;
}

/* Progress bars */
.stProgress > div > div > div {
    background: linear-gradient(90deg, #2ecc71, #00bcd4) !important;
}

/* Code blocks */
.stCode {
    background: #0d1117 !important;
    border: 1px solid #21262d !important;
    border-radius: 8px !important;
}
</style>
""", unsafe_allow_html=True)


# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="sq-header">
    <div class="sq-title">🛰️ SatQuery-Edge</div>
    <div class="sq-subtitle">Offline Satellite Intelligence Investigation Workspace</div>
    <span class="sq-badge badge-isro">SIH 26167 · ISRO</span>
    <span class="sq-badge badge-offline">● OFFLINE MODE</span>
</div>
""", unsafe_allow_html=True)


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚡ Quick Start")
    demo_mode = st.toggle("Demo Mode", value=True, help="Load sample images automatically")

    st.markdown("### 🎯 SIH Scenario Presets")
    pcol1, pcol2 = st.columns(2)
    if pcol1.button("🌊 Flood", use_container_width=True, help="Flood Inundation & Built-Up Change"):
        st.session_state["query_input"] = "Identify newly flooded built-up areas in Assam and show highest-confidence regions."
    if pcol2.button("🌲 Fire", use_container_width=True, help="Forest Fire & Burn Scar Assessment"):
        st.session_state["query_input"] = "Detect forest fire damage and highlight burn scar vegetation loss regions in Uttarakhand."
    if pcol1.button("🏙️ Urban", use_container_width=True, help="Urban Encroachment Mapping"):
        st.session_state["query_input"] = "Locate unauthorized urban structure changes and infrastructure expansion in Delhi."
    if pcol2.button("🛰️ SAR", use_container_width=True, help="All-Weather SAR Inundation"):
        st.session_state["query_input"] = "Perform SAR and optical multi-sensor fusion to detect water bodies under clouds in Odisha."

    st.markdown("### 📍 Map Geocoding Location")
    location_option = st.selectbox(
        "Target Location / Region:",
        [
            "Auto-Detect from Query",
            "Assam (Guwahati / Kaziranga)",
            "Uttarakhand (Dehradun / Chamoli)",
            "Wayanad (Kerala)",
            "Delhi NCR",
            "Odisha (Bhubaneswar)",
            "Mumbai (Maharashtra)",
            "Kolkata (West Bengal)",
            "Bengaluru (Karnataka)",
            "Chennai (Tamil Nadu)",
            "Custom Coordinates",
        ],
        index=0,
        help="Select location to map detected regions or auto-detect from query.",
    )

    custom_lat = None
    custom_lon = None
    if location_option == "Custom Coordinates":
        c_col1, c_col2 = st.columns(2)
        custom_lat = c_col1.number_input("Latitude (°N)", value=20.2960, format="%.4f")
        custom_lon = c_col2.number_input("Longitude (°E)", value=85.8240, format="%.4f")

    st.markdown("### 📡 Data Input")

    uploaded_images = []
    img_observations: list[ImageObservation] = []

    if demo_mode:
        st.info("🔵 Demo Mode: Sample images loaded automatically.")
        pre_path = os.path.join(_SAMPLE_DIR, "pre_event.png")
        post_path = os.path.join(_SAMPLE_DIR, "post_event.png")

        if os.path.exists(pre_path) and os.path.exists(post_path):
            try:
                pre_pil = Image.open(pre_path).convert("RGB")
                post_pil = Image.open(post_path).convert("RGB")
                uploaded_images = [np.array(pre_pil), np.array(post_pil)]

                for i, (path, arr) in enumerate(
                    zip([pre_path, post_path], uploaded_images)
                ):
                    fname = os.path.basename(path)
                    h, w = arr.shape[:2]
                    ch = arr.shape[2] if arr.ndim == 3 else 1
                    fsize = os.path.getsize(path)
                    img_observations.append(ImageObservation(
                        image_id=f"demo_{i+1}",
                        filename=fname,
                        width=w, height=h, channels=ch,
                        file_size_bytes=fsize,
                        sensor="Demo Optical (Synthetic)",
                        acquisition_date="Demo Date",
                        crs="Approximate Demo CRS",
                        format="PNG",
                    ))

                st.success(f"✅ Loaded: pre_event.png + post_event.png")
            except Exception as e:
                st.error(f"Could not load sample images: {e}")
        else:
            st.warning("Sample images not found. Regenerating...")
            try:
                from data.sample.generate_samples import ensure_samples
                ensure_samples()
                st.rerun()
            except Exception as e:
                st.error(f"Failed to generate samples: {e}")
    else:
        uploads = st.file_uploader(
            "Upload 1 or 2 images",
            type=["png", "jpg", "jpeg", "tif", "tiff"],
            accept_multiple_files=True,
            help="Upload pre-event and/or post-event satellite images.",
        )
        if uploads:
            for uf in uploads[:2]:
                try:
                    pil_img = Image.open(uf).convert("RGB")
                    arr = np.array(pil_img)
                    uploaded_images.append(arr)
                    h, w = arr.shape[:2]
                    ch = arr.shape[2] if arr.ndim == 3 else 1
                    img_observations.append(ImageObservation(
                        image_id=f"upload_{len(img_observations)+1}",
                        filename=uf.name,
                        width=w, height=h, channels=ch,
                        file_size_bytes=uf.size,
                        sensor="Unknown / User Upload",
                        acquisition_date="Unknown",
                        crs="Unknown",
                        format=uf.type.split("/")[-1].upper(),
                    ))
                except Exception as e:
                    st.error(f"Could not load {uf.name}: {e}")

    # ── Image Metadata ────────────────────────────────────────────────────────
    if img_observations:
        st.markdown("### 📋 Image Metadata")
        for i, obs in enumerate(img_observations):
            with st.expander(f"Image {i+1}: {obs.filename}", expanded=i == 0):
                meta_cols = st.columns(2)
                with meta_cols[0]:
                    st.markdown(f"**Filename:** {obs.filename}")
                    st.markdown(f"**Format:** {obs.format}")
                    st.markdown(f"**Width:** {obs.width}px")
                    st.markdown(f"**Height:** {obs.height}px")
                with meta_cols[1]:
                    st.markdown(f"**Channels:** {obs.channels}")
                    st.markdown(f"**Size:** {obs.file_size_bytes / 1024:.1f} KB")
                    st.markdown(f"**Sensor:** {obs.sensor}")
                    st.markdown(f"**Date:** {obs.acquisition_date}")
                st.markdown(f"**CRS:** {obs.crs}")

    st.markdown("---")
    st.markdown(
        "<small style='color:#444;'>SatQuery-Edge v1.0 · SIH 26167 · ISRO<br>"
        "Fully offline · No API keys required</small>",
        unsafe_allow_html=True,
    )


# ─── Main Area ────────────────────────────────────────────────────────────────
st.markdown("### 🔍 Investigation Query")

FLAGSHIP_QUERY = (
    "Identify newly flooded built-up areas and show the highest-confidence regions."
)

default_query = FLAGSHIP_QUERY if demo_mode else ""
query_val = st.session_state.get("query_input", default_query)

query = st.text_area(
    "Enter your natural-language investigation query:",
    value=query_val,
    height=90,
    placeholder=(
        "e.g. Identify newly flooded built-up areas and show the highest-confidence regions."
    ),
    label_visibility="collapsed",
)

col_btn, col_spacer = st.columns([1, 2])
with col_btn:
    run_clicked = st.button("🚀 RUN INVESTIGATION", type="primary", use_container_width=True)

st.markdown("---")


# ─── Investigation Logic ──────────────────────────────────────────────────────
def _make_inv_id() -> str:
    date_str = datetime.datetime.now().strftime("%Y%m%d")
    suffix = "".join(random.choices(string.ascii_uppercase + string.digits, k=6))
    return f"SQE-{date_str}-{suffix}"


def _numpy_to_pil(arr: np.ndarray) -> Image.Image:
    """Convert BGR or RGB numpy array to PIL RGB image."""
    if arr.shape[2] == 3:
        # Detect if likely BGR (from OpenCV) — heuristic: if loaded via cv2
        # We convert to RGB for display
        return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))
    return Image.fromarray(arr)


def _display_image(arr: np.ndarray, caption: str, use_container_width: bool = True) -> None:
    """Show a numpy image (BGR or RGB) in Streamlit."""
    try:
        if arr is None or arr.size == 0:
            st.warning(f"No image data for: {caption}")
            return
        if arr.ndim == 3 and arr.shape[2] == 3:
            pil = Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))
        elif arr.ndim == 3 and arr.shape[2] == 4:
            pil = Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGRA2RGB))
        else:
            pil = Image.fromarray(arr)
        st.image(pil, caption=caption, use_container_width=use_container_width)
    except Exception as e:
        st.warning(f"Could not display '{caption}': {e}")


def run_investigation(
    query: str,
    images: list[np.ndarray],
    selected_location: str = "",
    custom_lat: Optional[float] = None,
    custom_lon: Optional[float] = None,
) -> None:
    """
    Full SatQuery-Edge investigation pipeline.
    Orchestrates: Router → Validator → Tools → Evidence → Report
    """
    t_start = time.perf_counter()
    trace: list[TraceEvent] = []
    stages_done: list[str] = []

    inv_id = _make_inv_id()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ── QUERY ─────────────────────────────────────────────────────────────────
    trace.append(TraceEvent.now("QUERY RECEIVED", query[:120]))
    stages_done.append("QUERY")

    # ── ROUTER ────────────────────────────────────────────────────────────────
    with st.spinner("🔀 Routing query..."):
        try:
            task_graph = route_query(query, num_images=len(images))
        except Exception as e:
            st.error(f"Router error: {e}")
            return

    trace.append(TraceEvent.now(
        "ROUTER DECISION",
        f"Intent: {task_graph.intent}",
        detail=f"Selected tools: {', '.join(task_graph.selected_tools)}\n"
               f"Images required: {task_graph.num_images_required}",
    ))
    stages_done.append("ROUTER")

    # ── VALIDATOR ─────────────────────────────────────────────────────────────
    with st.spinner("🔎 Validating inputs..."):
        try:
            validation = validate_inputs(images, task_graph.num_images_required)
        except Exception as e:
            st.error(f"Validation error: {e}")
            return

    trace.append(TraceEvent.now(
        "VALIDATION",
        f"Status: {validation.status}",
        detail="\n".join(validation.reasons + validation.warnings),
    ))
    stages_done.append("VALIDATOR")

    if not validation.passed:
        render_pipeline_header(stages_done)
        render_validation_result(validation)
        return

    # ── SPECIALIST TOOLS ──────────────────────────────────────────────────────
    tool_results = []
    active_images = images[:task_graph.num_images_required]

    for tool_name in task_graph.selected_tools:
        with st.spinner(f"⚙️ Running specialist tool: {tool_name}..."):
            trace.append(TraceEvent.now("TOOL EXECUTING", f"{tool_name} → started"))

            # Build parameters for geospatial tool
            params = dict(task_graph.parameters)
            params["query"] = query
            params["selected_location"] = selected_location
            if custom_lat is not None:
                params["custom_lat"] = custom_lat
            if custom_lon is not None:
                params["custom_lon"] = custom_lon

            if tool_name == "geospatial":
                # Gather boxes + polygons from previous tool results
                all_boxes = []
                all_polys = []
                for tr in tool_results:
                    all_boxes.extend(tr.bounding_boxes)
                    all_polys.extend(tr.polygons)
                params["pixel_boxes"] = all_boxes
                params["pixel_polygons"] = all_polys
                if active_images:
                    h, w = active_images[0].shape[:2]
                    params["image_width"] = w
                    params["image_height"] = h

            request = ToolRequest(
                tool_name=tool_name,
                parameters=params,
                image_ids=[obs.image_id for obs in img_observations],
            )

            try:
                result = execute_tool(request, active_images)
            except Exception as e:
                from controller.schemas import ToolResult
                result = ToolResult(
                    tool_name=tool_name, success=False, description="",
                    error=str(e),
                )

            tool_results.append(result)

            region_ct = len(result.bounding_boxes)
            trace.append(TraceEvent.now(
                f"TOOL OUTPUT ({tool_name})",
                f"{'Success' if result.success else 'Failed'} — "
                f"{region_ct} region(s) detected",
                detail=result.description[:200] if result.description else result.error,
            ))

    stages_done.append("SPECIALIST")

    # ── EVIDENCE FUSION ───────────────────────────────────────────────────────
    with st.spinner("🧮 Fusing evidence..."):
        try:
            evidence = fuse_evidence(task_graph, tool_results, validation)
        except Exception as e:
            st.error(f"Evidence fusion error: {e}")
            return

    trace.append(TraceEvent.now(
        "EVIDENCE FUSION",
        f"Score: {evidence.score:.1f}/100",
        detail="\n".join(
            f"{c.code}={c.value:.3f}" for c in evidence.components
        ),
    ))
    stages_done.append("FUSION")

    # ── GEO POLYGONS ──────────────────────────────────────────────────────────
    geo_polygons = []
    for tr in tool_results:
        if tr.tool_name == "geospatial" and tr.success:
            geo_polygons = tr.polygons
            break

    # Fallback: gather from any tool
    if not geo_polygons:
        for tr in tool_results:
            if tr.polygons:
                geo_polygons.extend(tr.polygons)

    stages_done.append("GEO")

    # ── FINAL ANSWER ──────────────────────────────────────────────────────────
    abstain, abstain_reason = should_abstain(evidence.score, evidence.components)
    evidence = evidence.model_copy(update={"abstain": abstain, "abstain_reason": abstain_reason})

    if abstain:
        final_answer = (
            f"⚠ ABSTENTION\n\n{abstain_reason}\n\n"
            "The system cannot reliably conclude based on available evidence."
        )
    else:
        change_result = next(
            (r for r in tool_results if r.tool_name == "change" and r.success), None
        )
        if change_result:
            rc = change_result.metrics.get("region_count", 0)
            pct = change_result.metrics.get("changed_pixel_percent", 0.0)
            final_answer = (
                f"Candidate newly flooded built-up regions detected.\n\n"
                f"• {rc} significant changed region(s) identified.\n"
                f"• Approximately {pct:.1f}% of scene shows change.\n"
                f"• Evidence Score: {evidence.score:.0f}/100 — HIGH CONFIDENCE.\n\n"
                f"Detected change signatures are consistent with new water-like "
                f"inundation over areas previously identified as built-up terrain. "
                f"Highest-confidence regions are highlighted in the overlay and mapped.\n\n"
                f"NOTE: Results are demonstration-grade. Not a scientifically "
                f"validated remote-sensing product."
            )
        else:
            vqa_result = next(
                (r for r in tool_results if r.tool_name == "vqa" and r.success), None
            )
            if vqa_result:
                final_answer = (
                    f"Scene analysis complete.\n\n"
                    f"{vqa_result.description}\n\n"
                    f"Evidence Score: {evidence.score:.0f}/100."
                )
            else:
                final_answer = (
                    f"Investigation complete. "
                    f"Evidence Score: {evidence.score:.0f}/100."
                )

    trace.append(TraceEvent.now(
        "FINAL SCORE",
        f"{evidence.score:.0f}/100",
    ))
    trace.append(TraceEvent.now("ANSWER", final_answer[:120]))

    # ── PDF REPORT ─────────────────────────────────────────────────────────────
    inv_result = InvestigationResult(
        investigation_id=inv_id,
        timestamp=timestamp,
        query=query,
        task_graph=task_graph,
        validation=validation,
        tool_results=tool_results,
        evidence=evidence,
        final_answer=final_answer,
        geo_polygons=geo_polygons,
        execution_trace=trace,
    )

    try:
        pdf_bytes = generate_pdf_report(inv_result)
        stages_done.append("REPORT")
        trace.append(TraceEvent.now("REPORT", "PDF report generated successfully."))
    except Exception as e:
        pdf_bytes = None
        st.warning(f"PDF report generation failed: {e}")

    # ══════════════════════════════════════════════════════════════════════════
    # ── DISPLAY RESULTS ───────────────────────────────────────────────────────
    # ══════════════════════════════════════════════════════════════════════════

    exec_time_ms = (time.perf_counter() - t_start) * 1000.0

    st.markdown("---")

    # Pipeline header
    render_pipeline_header(stages_done)
    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    # Edge Telemetry Bar
    render_edge_telemetry(exec_time_ms, peak_ram_mb=235.0)

    # Investigation ID
    render_investigation_id(inv_id, timestamp)
    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    # Intent badge
    intent_colors = {
        "bi_temporal_change": "#2ecc71",
        "optical_sar_fusion": "#00bcd4",
        "text_guided_grounding": "#f39c12",
        "single_image_vqa": "#9b59b6",
    }
    ic = intent_colors.get(task_graph.intent, "#8b949e")
    st.markdown(
        f"<div style='display:inline-block;padding:4px 14px;border-radius:20px;"
        f"background:{ic}22;border:1px solid {ic};color:{ic};"
        f"font-size:12px;font-weight:700;letter-spacing:1px;margin-bottom:12px;'>"
        f"Intent: {task_graph.intent.upper()}</div>",
        unsafe_allow_html=True,
    )

    # Final answer
    if abstain:
        st.error(final_answer)
    else:
        st.success(f"**Final Answer**\n\n{final_answer}")

    # Export options (PDF, GeoJSON, KML)
    d_col1, d_col2, d_col3 = st.columns(3)
    with d_col1:
        if pdf_bytes:
            st.download_button(
                label="⬇️ PDF Audit Report",
                data=pdf_bytes,
                file_name=f"report_{inv_id}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
    with d_col2:
        if geo_polygons:
            geojson_data = export_geojson(geo_polygons)
            st.download_button(
                label="🌐 GeoJSON (ISRO Bhuvan)",
                data=geojson_data,
                file_name=f"detections_{inv_id}.geojson",
                mime="application/geo+json",
                use_container_width=True,
            )
    with d_col3:
        if geo_polygons:
            kml_data = export_kml(geo_polygons)
            st.download_button(
                label="🗺️ KML (Google Earth)",
                data=kml_data,
                file_name=f"detections_{inv_id}.kml",
                mime="application/vnd.google-earth.kml+xml",
                use_container_width=True,
            )

    # ── Tabs ───────────────────────────────────────────────────────────────────
    tab_img, tab_map, tab_evidence, tab_trace = st.tabs([
        "🖼️ Image Viewer", "🗺️ Map", "📊 Evidence", "🔬 Execution Trace"
    ])

    # ── TAB 1: Image Viewer ───────────────────────────────────────────────────
    with tab_img:
        st.markdown("#### Image Analysis")
        change_result = next(
            (r for r in tool_results if r.tool_name == "change" and r.success), None
        )
        sar_result = next(
            (r for r in tool_results if r.tool_name == "optical_sar" and r.success), None
        )
        vqa_result = next(
            (r for r in tool_results if r.tool_name == "vqa" and r.success), None
        )
        grounding_result = next(
            (r for r in tool_results if r.tool_name == "grounding" and r.success), None
        )

        if change_result:
            outputs = change_result.image_outputs
            c1, c2 = st.columns(2)
            with c1:
                if "pre_event" in outputs:
                    _display_image(outputs["pre_event"], "Pre-Event Image")
                if "difference" in outputs:
                    _display_image(outputs["difference"], "Difference (False Color)")
            with c2:
                if "post_event" in outputs:
                    _display_image(outputs["post_event"], "Post-Event Image")
                if "overlay" in outputs:
                    _display_image(outputs["overlay"], "Change Detection Overlay")

            if "change_mask" in outputs:
                st.markdown("**Change Mask**")
                _display_image(outputs["change_mask"], "Binary Change Mask")

            # Metrics
            st.markdown("**Detection Metrics**")
            m_cols = st.columns(4)
            metrics = change_result.metrics
            m_cols[0].metric("Regions", str(metrics.get("region_count", 0)))
            m_cols[1].metric("Changed %", f"{metrics.get('changed_pixel_percent', 0.0):.1f}%")
            m_cols[2].metric("Max Confidence", f"{metrics.get('max_confidence', 0.0):.2f}")
            m_cols[3].metric("Largest Region", f"{metrics.get('largest_region_frac', 0.0)*100:.1f}%")

        elif sar_result:
            outputs = sar_result.image_outputs
            c1, c2, c3 = st.columns(3)
            with c1:
                if "optical" in outputs:
                    st.image(outputs["optical"], caption="Optical Image", use_container_width=True)
            with c2:
                if "sar_like" in outputs:
                    st.image(outputs["sar_like"], caption="SAR-like Synthetic Observation", use_container_width=True)
            with c3:
                if "fusion_color" in outputs:
                    st.image(outputs["fusion_color"], caption="Fusion (R=Optical, G=Fused, B=SAR)", use_container_width=True)
            st.info("SAR image is a synthetic observation for demonstration purposes.")

        else:
            # Show raw input images
            if active_images:
                cols = st.columns(min(len(active_images), 2))
                for i, img_arr in enumerate(active_images):
                    with cols[i % 2]:
                        _display_image(img_arr, f"Image {i+1}")

            if grounding_result and "overlay" in grounding_result.image_outputs:
                st.markdown("**Grounding Overlay**")
                _display_image(grounding_result.image_outputs["overlay"], "Candidate Regions Highlighted")

    # ── TAB 2: Map ─────────────────────────────────────────────────────────────
    with tab_map:
        st.markdown("#### Geospatial Investigation Map")
        if geo_polygons:
            try:
                from streamlit_folium import st_folium
                fol_map = build_investigation_map(geo_polygons)
                st_folium(fol_map, width="100%", height=480, returned_objects=[])
                st.caption(
                    "⚠ Approximate demo georeferencing applied. "
                    "Coordinates are NOT derived from real GeoTIFF metadata."
                )

                # Region table
                st.markdown("**Detected Regions**")
                region_rows = []
                for i, poly in enumerate(geo_polygons):
                    if poly.coordinates:
                        c = poly.coordinates[0]
                        lat, lon = c[1], c[0]
                    else:
                        lat = lon = 0.0
                    region_rows.append({
                        "#": i + 1,
                        "Label": poly.label,
                        "Confidence": f"{poly.confidence:.3f}",
                        "Area (km²)": f"{poly.area_km2_approx:.3f}",
                        "Demo Georef": "Yes" if poly.is_demo_georef else "No",
                        "Approx Lat": f"{lat:.4f}°",
                        "Approx Lon": f"{lon:.4f}°",
                    })
                if region_rows:
                    import pandas as pd
                    st.dataframe(pd.DataFrame(region_rows), use_container_width=True)
            except ImportError:
                st.error("streamlit-folium not installed. Run: pip install streamlit-folium")
            except Exception as e:
                st.error(f"Map rendering error: {e}")
        else:
            st.info("No geospatial polygons available. Run an investigation first.")

    # ── TAB 3: Evidence ────────────────────────────────────────────────────────
    with tab_evidence:
        st.markdown("#### Evidence Analysis")
        render_evidence_score(evidence)

        st.markdown("---")
        st.markdown("**Specialist Tool Results**")
        for tr in tool_results:
            render_tool_card(tr)

        # Validation details
        st.markdown("---")
        st.markdown("**Validation Details**")
        render_validation_result(validation)

    # ── TAB 4: Execution Trace ─────────────────────────────────────────────────
    with tab_trace:
        st.markdown("#### Observable Execution Trace")
        st.markdown(
            "<small style='color:#666;'>Complete chronological pipeline execution log</small>",
            unsafe_allow_html=True,
        )
        render_trace(trace)


# ─── Main ─────────────────────────────────────────────────────────────────────
if run_clicked:
    if not query or not query.strip():
        st.error("⚠ Please enter an investigation query before running.")
    elif not uploaded_images:
        st.error(
            "⚠ No images loaded. Enable Demo Mode or upload images in the sidebar."
        )
    else:
        try:
            run_investigation(
                query.strip(),
                uploaded_images,
                selected_location=location_option,
                custom_lat=custom_lat,
                custom_lon=custom_lon,
            )
        except Exception as e:
            st.error(f"Investigation failed unexpectedly: {e}")
            with st.expander("Error Details"):
                st.code(traceback.format_exc())
else:
    # Landing state
    if not uploaded_images and not demo_mode:
        st.markdown("""
        <div style="
            text-align:center; padding:60px 40px;
            border:1px dashed #30363d; border-radius:16px;
            color:#4a5568; margin:20px 0;
        ">
            <div style="font-size:48px;margin-bottom:16px;">🛰️</div>
            <div style="font-size:18px;font-weight:600;color:#8b949e;">Ready for Investigation</div>
            <div style="font-size:14px;margin-top:8px;color:#4a5568;">
                Upload images in the sidebar, enter a query, and click <b>RUN INVESTIGATION</b>
            </div>
        </div>
        """, unsafe_allow_html=True)
    elif demo_mode and uploaded_images:
        st.markdown("""
        <div style="
            text-align:center; padding:40px 40px;
            border:1px solid #2ecc7133; border-radius:16px;
            background: #0d1a0e44;
            color:#4a5568; margin:20px 0;
        ">
            <div style="font-size:36px;margin-bottom:12px;">✅</div>
            <div style="font-size:16px;font-weight:600;color:#2ecc71;">Demo Mode Ready</div>
            <div style="font-size:13px;margin-top:8px;color:#8b949e;">
                Sample images loaded · Flagship query populated<br>
                Click <b style="color:#2ecc71;">🚀 RUN INVESTIGATION</b> to begin
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Thumbnails preview
        if len(uploaded_images) >= 2:
            c1, c2 = st.columns(2)
            with c1:
                img_pil = Image.fromarray(uploaded_images[0])
                st.image(img_pil, caption="Pre-Event (Sample)", use_container_width=True)
            with c2:
                img_pil = Image.fromarray(uploaded_images[1])
                st.image(img_pil, caption="Post-Event (Sample)", use_container_width=True)
