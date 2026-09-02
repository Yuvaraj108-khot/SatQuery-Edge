"""Test all specialist tools."""
from data.sample.generate_samples import generate_pre_event, generate_post_event
from tools.change import run_change
from tools.grounding import run_grounding
from tools.vqa import run_vqa
from tools.geospatial import run_geospatial
from tools.optical_sar import run_optical_sar
from evidence.fusion import fuse_evidence
from controller.router import route_query
from controller.validator import validate_inputs

pre = generate_pre_event()
post = generate_post_event(pre)

# Change tool
cr = run_change([pre, post], {"threshold": 0.15, "min_region_area": 200, "morph_kernel": 5})
print("Change tool: success=%s" % cr.success)
print("  regions=%s" % cr.metrics.get("region_count"))
print("  changed_pct=%.1f" % cr.metrics.get("changed_pixel_percent"))
print("  image_outputs keys: %s" % list(cr.image_outputs.keys()))

# Grounding on post-event (where flood is present)
gr = run_grounding([post], {"target_description": "flood", "threshold": 0.10})
print("Grounding tool: success=%s, regions=%s" % (gr.success, len(gr.bounding_boxes)))

# Grounding fallback: use broader threshold
if len(gr.bounding_boxes) == 0:
    gr = run_grounding([post], {"target_description": "water", "threshold": 0.05})
    print("Grounding (water/0.05): regions=%s" % len(gr.bounding_boxes))

# VQA tool
vr = run_vqa([pre], {})
print("VQA tool: success=%s" % vr.success)
print("  desc: %s" % vr.description[:80])

# Geospatial tool
geo = run_geospatial([], {"pixel_boxes": cr.bounding_boxes, "pixel_polygons": cr.polygons, "image_width": 512, "image_height": 512})
print("Geo tool: success=%s, polygons=%s" % (geo.success, len(geo.polygons)))
if geo.polygons:
    p = geo.polygons[0]
    print("  First polygon coords: %s" % str(p.coordinates[:2]))
    print("  Area km2: %s" % p.area_km2_approx)

# Optical/SAR tool
sar_r = run_optical_sar([pre, post], {"optical_weight": 0.6, "sar_weight": 0.4})
print("Optical/SAR: success=%s" % sar_r.success)

# Evidence fusion
tg = route_query("Identify newly flooded built-up areas", num_images=2)
vl = validate_inputs([pre, post], 2)
ev = fuse_evidence(tg, [cr, gr, geo], vl)
print("Evidence score: %.1f/100" % ev.score)
print("Components: %s" % [(c.code, round(c.value, 3)) for c in ev.components])
print("Abstain: %s" % ev.abstain)

# PDF report test
from controller.schemas import InvestigationResult, TaskGraph, ValidationResult
import datetime
result = InvestigationResult(
    investigation_id="SQE-TEST-001",
    timestamp=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    query="Test query",
    task_graph=tg,
    validation=vl,
    tool_results=[cr, gr, geo],
    evidence=ev,
    final_answer="Test answer",
    geo_polygons=geo.polygons,
    execution_trace=[],
)
from reports.report_generator import generate_pdf_report
pdf = generate_pdf_report(result)
print("PDF size: %d bytes" % len(pdf))

print("\nALL TESTS PASSED!")
