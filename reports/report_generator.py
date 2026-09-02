"""
reports/report_generator.py
PDF report generator for SatQuery-Edge using ReportLab.
"""

from __future__ import annotations
import io
import os
import datetime
from typing import Optional
import numpy as np

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether, Image as RLImage,
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

from controller.schemas import InvestigationResult


def generate_pdf_report(result: InvestigationResult) -> bytes:
    """
    Generate a PDF investigation report and return as bytes.
    Saves a copy to the reports/ directory.
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        rightMargin=20 * mm,
        leftMargin=20 * mm,
        topMargin=20 * mm,
        bottomMargin=20 * mm,
        title=f"SatQuery-Edge Report — {result.investigation_id}",
    )

    story = _build_story(result)
    doc.build(story)

    pdf_bytes = buf.getvalue()

    # Save to reports/ directory
    reports_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports"
    )
    os.makedirs(reports_dir, exist_ok=True)
    filename = f"report_{result.investigation_id}.pdf"
    with open(os.path.join(reports_dir, filename), "wb") as f:
        f.write(pdf_bytes)

    return pdf_bytes


def _arr_to_rl_image(arr: np.ndarray, width_mm: float = 50.0, height_mm: float = 50.0) -> Optional[RLImage]:
    """Convert numpy array image to ReportLab RLImage flowable."""
    try:
        import cv2
        from PIL import Image as PILImage
        if arr is None or not isinstance(arr, np.ndarray) or arr.size == 0:
            return None
        if arr.ndim == 3 and arr.shape[2] == 3:
            rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        elif arr.ndim == 3 and arr.shape[2] == 4:
            rgb = cv2.cvtColor(arr, cv2.COLOR_BGRA2RGB)
        else:
            rgb = arr
        pil_img = PILImage.fromarray(rgb)
        img_buf = io.BytesIO()
        pil_img.save(img_buf, format="PNG")
        img_buf.seek(0)
        return RLImage(img_buf, width=width_mm * mm, height=height_mm * mm)
    except Exception:
        return None


def _build_story(result: InvestigationResult) -> list:
    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "SQETitle",
        parent=styles["Title"],
        fontSize=22,
        textColor=colors.HexColor("#2ecc71"),
        spaceAfter=4,
        alignment=TA_CENTER,
    )
    subtitle_style = ParagraphStyle(
        "SQESubtitle",
        parent=styles["Normal"],
        fontSize=11,
        textColor=colors.HexColor("#aaaaaa"),
        spaceAfter=2,
        alignment=TA_CENTER,
    )
    section_style = ParagraphStyle(
        "SQESection",
        parent=styles["Heading2"],
        fontSize=13,
        textColor=colors.HexColor("#58a6ff"),
        spaceBefore=14,
        spaceAfter=6,
        borderPad=3,
    )
    body_style = ParagraphStyle(
        "SQEBody",
        parent=styles["Normal"],
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#333333"),
    )
    mono_style = ParagraphStyle(
        "SQEMono",
        parent=styles["Code"],
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#1a1a1a"),
        backColor=colors.HexColor("#f5f5f5"),
        borderPad=4,
    )
    notice_style = ParagraphStyle(
        "SQENotice",
        parent=styles["Normal"],
        fontSize=9,
        textColor=colors.HexColor("#555555"),
        backColor=colors.HexColor("#fff8dc"),
        borderPad=6,
        leading=12,
    )

    story = []

    # ── Header ──────────────────────────────────────────────────────────────
    story.append(Paragraph("SatQuery-Edge", title_style))
    story.append(Paragraph("Offline Satellite Intelligence Investigation Report", subtitle_style))
    story.append(Paragraph("SIH 26167 · ISRO · Prototype · Offline Mode", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#2ecc71")))
    story.append(Spacer(1, 8))

    # ── Meta ─────────────────────────────────────────────────────────────────
    meta_data = [
        ["Investigation ID", result.investigation_id],
        ["Timestamp", result.timestamp],
        ["Intent", result.task_graph.intent],
        ["Tools Used", ", ".join(result.task_graph.selected_tools)],
    ]
    meta_table = Table(meta_data, colWidths=[50 * mm, 120 * mm])
    meta_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#e8f5e9")),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#1a5e37")),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, colors.HexColor("#fafafa")]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 10))

    # ── Query ─────────────────────────────────────────────────────────────────
    story.append(Paragraph("Query", section_style))
    story.append(Paragraph(result.query, body_style))
    story.append(Spacer(1, 8))

    # ── Final Answer ──────────────────────────────────────────────────────────
    story.append(Paragraph("Final Answer", section_style))
    answer_para = Paragraph(result.final_answer.replace("\n", "<br/>"), body_style)
    story.append(answer_para)
    story.append(Spacer(1, 8))

    # ── Evidence Score ────────────────────────────────────────────────────────
    story.append(Paragraph("Evidence Score", section_style))
    ev = result.evidence
    score_data = [["Component", "Code", "Value", "Weight"]]
    for comp in ev.components:
        score_data.append([comp.name, comp.code, f"{comp.value:.3f}", f"{comp.weight:.2f}"])
    score_data.append(["", "", "", ""])
    score_data.append(["FINAL SCORE", "", f"{ev.score:.1f} / 100", ""])

    score_table = Table(score_data, colWidths=[70 * mm, 20 * mm, 35 * mm, 25 * mm])
    score_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a5e37")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("ROWBACKGROUNDS", (0, 1), (-1, -3),
         [colors.white, colors.HexColor("#f0f8f0")]),
        ("BACKGROUND", (0, -1), (-1, -1), colors.HexColor("#e8f5e9")),
        ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
        ("TEXTCOLOR", (0, -1), (-1, -1), colors.HexColor("#1a5e37")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(score_table)

    if ev.abstain:
        story.append(Spacer(1, 6))
        story.append(Paragraph(
            f"⚠ ABSTENTION: {ev.abstain_reason}",
            ParagraphStyle("abs", parent=body_style,
                           textColor=colors.HexColor("#cc0000"),
                           backColor=colors.HexColor("#fff0f0"), borderPad=4),
        ))

    story.append(Spacer(1, 10))

    # ── Validation ────────────────────────────────────────────────────────────
    story.append(Paragraph("Validation", section_style))
    vr = result.validation
    val_status = "PASS ✓" if vr.passed else "FAIL ✗"
    story.append(Paragraph(f"Status: <b>{val_status}</b>", body_style))
    for reason in vr.reasons:
        story.append(Paragraph(f"  • {reason}", body_style))
    for warn in vr.warnings:
        story.append(Paragraph(f"  ⚠ {warn}", body_style))
    story.append(Spacer(1, 8))

    # ── Detected Regions ──────────────────────────────────────────────────────
    all_polygons = result.geo_polygons
    if all_polygons:
        story.append(Paragraph("Detected Regions & Approximate Coordinates", section_style))
        region_data = [["#", "Label", "Confidence", "Area (km²)", "Demo Georef"]]
        for i, poly in enumerate(all_polygons):
            region_data.append([
                str(i + 1),
                poly.label[:30],
                f"{poly.confidence:.2f}",
                f"{poly.area_km2_approx:.3f}",
                "Yes" if poly.is_demo_georef else "No",
            ])
        region_table = Table(
            region_data,
            colWidths=[10 * mm, 65 * mm, 25 * mm, 25 * mm, 20 * mm]
        )
        region_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a5e37")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f0f8f0")]),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))
        story.append(region_table)
        story.append(Spacer(1, 8))

    # ── Key Visual Evidence ────────────────────────────────────────────────────
    vis_images = []
    vis_captions = []
    for tr in result.tool_results:
        outputs = tr.image_outputs
        if "pre_event" in outputs and "post_event" in outputs:
            img1 = _arr_to_rl_image(outputs["pre_event"])
            img2 = _arr_to_rl_image(outputs["post_event"])
            img3 = _arr_to_rl_image(outputs.get("overlay", outputs.get("difference")))
            if img1 and img2:
                row = [img1, img2]
                cap = ["Pre-Event Image", "Post-Event Image"]
                if img3:
                    row.append(img3)
                    cap.append("Change Overlay")
                vis_images.append(row)
                vis_captions.append(cap)
                break
        elif "optical" in outputs and "sar_like" in outputs:
            img1 = _arr_to_rl_image(outputs["optical"])
            img2 = _arr_to_rl_image(outputs["sar_like"])
            img3 = _arr_to_rl_image(outputs.get("fusion_color"))
            if img1 and img2:
                row = [img1, img2]
                cap = ["Optical Image", "SAR-like Image"]
                if img3:
                    row.append(img3)
                    cap.append("Fusion Color")
                vis_images.append(row)
                vis_captions.append(cap)
                break

    if vis_images:
        story.append(Paragraph("Key Visual Evidence", section_style))
        row = vis_images[0]
        caps = vis_captions[0]
        col_w = 165.0 / len(row)
        tbl_data = [
            row,
            [Paragraph(f"<b>{c}</b>", ParagraphStyle("cap", parent=body_style, alignment=1, fontSize=8)) for c in caps],
        ]
        img_table = Table(tbl_data, colWidths=[col_w * mm] * len(row))
        img_table.setStyle(TableStyle([
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(img_table)
        story.append(Spacer(1, 8))

    # ── Tool Results Summary ───────────────────────────────────────────────────
    story.append(Paragraph("Specialist Tool Outputs", section_style))
    for tr in result.tool_results:
        status = "SUCCESS" if tr.success else "FAILED"
        story.append(Paragraph(
            f"<b>{tr.tool_name.upper()}</b> — {status} — "
            f"{len(tr.bounding_boxes)} region(s)",
            body_style,
        ))
        if tr.description:
            desc_lines = tr.description[:400].replace("\n", "<br/>")
            story.append(Paragraph(desc_lines, mono_style))
        story.append(Spacer(1, 4))

    # ── Execution Trace ────────────────────────────────────────────────────────
    story.append(Paragraph("Execution Trace", section_style))
    trace_lines = []
    for ev_t in result.execution_trace:
        trace_lines.append(f"[{ev_t.timestamp}] {ev_t.stage}: {ev_t.message}")
        if ev_t.detail:
            for dl in ev_t.detail.split("\n")[:3]:
                trace_lines.append(f"    {dl}")
    trace_text = "\n".join(trace_lines)
    story.append(Paragraph(trace_text.replace("\n", "<br/>"), mono_style))
    story.append(Spacer(1, 10))

    # ── Tool Versions ─────────────────────────────────────────────────────────
    story.append(Paragraph("Tool Versions", section_style))
    try:
        import cv2 as _cv2
        import numpy as _np
        import streamlit as _st
        cv_ver = _cv2.__version__
        np_ver = _np.__version__
        st_ver = _st.__version__
    except Exception:
        cv_ver = np_ver = st_ver = "unknown"

    versions = [
        ["Component", "Version"],
        ["SatQuery-Edge Prototype", "1.0.0-SIH26167"],
        ["Python OpenCV", cv_ver],
        ["NumPy", np_ver],
        ["Streamlit", st_ver],
        ["Report Engine", "ReportLab PDF"],
        ["Mode", "OFFLINE — No external APIs"],
    ]
    ver_table = Table(versions, colWidths=[70 * mm, 80 * mm])
    ver_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a5e37")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f5f5")]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(ver_table)
    story.append(Spacer(1, 10))

    # ── Disclaimer ────────────────────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#cccccc")))
    story.append(Spacer(1, 6))
    disclaimer = (
        "NOTE: This prototype uses lightweight local computer vision and synthetic/demo "
        "geospatial observations where applicable. Results are intended for demonstration "
        "purposes only and are NOT scientifically validated remote-sensing products. "
        "No cloud APIs, external inference services, or internet connectivity is used. "
        "All processing is fully offline. — SIH 26167 · ISRO · SatQuery-Edge"
    )
    story.append(Paragraph(disclaimer, notice_style))

    return story
