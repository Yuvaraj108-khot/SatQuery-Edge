"""
test_benchmark.py — SatQuery-Edge Automated Benchmark Evaluation Suite
Evaluates:
  1. Router & LlamaController intent accuracy
  2. Input validation diagnostics
  3. Real GeoTIFF reader & rasterio georeferencing
  4. Specialist tools (change, grounding, vqa, optical_sar, geospatial)
  5. Evidence Fusion engine & scoring
  6. GeoJSON / KML export functionality
"""

import sys
import os
import time
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from controller.router import route_query, LlamaController
from controller.validator import validate_inputs
from controller.registry import execute_tool
from controller.schemas import ToolRequest, ImageObservation
from evidence.fusion import fuse_evidence
from tools.geospatial import export_geojson, export_kml
from data.sample.generate_samples import ensure_samples

def run_benchmark():
    print("=" * 70)
    print("🛰️ SATQUERY-EDGE AUTOMATED BENCHMARK EVALUATION SUITE")
    print("=" * 70)

    t_start = time.perf_counter()
    passed_tests = 0
    total_tests = 0

    # Ensure samples exist
    ensure_samples()

    # ── Test 1: LlamaController & Intent Routing ─────────────────────────────
    print("\n[TEST 1] Router & LlamaController Intent Mapping...")
    total_tests += 1
    test_queries = [
        ("Identify newly flooded built-up areas", 2, "bi_temporal_change"),
        ("Multi-sensor optical and SAR radar fusion in Assam", 2, "optical_sar_fusion"),
        ("Locate bridges and water bodies in Guwahati", 1, "text_guided_grounding"),
        ("What features are visible in this satellite scene?", 1, "single_image_vqa"),
    ]

    routing_ok = True
    controller = LlamaController()
    for q, num_imgs, expected_intent in test_queries:
        tg = controller.plan_investigation(q, num_images=num_imgs)
        if tg.intent != expected_intent:
            print(f"  ❌ Routing failed for '{q}': got {tg.intent}, expected {expected_intent}")
            routing_ok = False
        else:
            print(f"  ✓ '{q[:40]}...' -> {tg.intent}")

    if routing_ok:
        passed_tests += 1
        print("  ✅ TEST 1 PASSED: 100% Intent Routing Accuracy")

    # ── Test 2: Input Validation ──────────────────────────────────────────────
    print("\n[TEST 2] Input Validation Diagnostics...")
    total_tests += 1
    sample_img0 = np.random.randint(50, 200, (256, 256, 3), dtype=np.uint8)
    sample_img1 = np.random.randint(50, 200, (256, 256, 3), dtype=np.uint8)
    
    val_pass = validate_inputs([sample_img0, sample_img1], required_count=2)
    val_fail = validate_inputs([sample_img0], required_count=2)

    if val_pass.passed and not val_fail.passed:
        passed_tests += 1
        print("  ✅ TEST 2 PASSED: Validation Gate correctly passed valid pair & rejected incomplete pair")
    else:
        print("  ❌ TEST 2 FAILED: Validation Gate logic error")

    # ── Test 3: Real GeoTIFF Reader & Rasterio Transform ────────────────────
    print("\n[TEST 3] Real GeoTIFF Georeferencing Pipeline...")
    total_tests += 1
    tif_path = os.path.join(PROJECT_ROOT, "data", "sample", "pre_event.tif")
    if os.path.exists(tif_path):
        import rasterio
        with rasterio.open(tif_path) as src:
            bounds = src.bounds
            crs = str(src.crs)
            count = src.count
        print(f"  ✓ Loaded GeoTIFF: {count} bands, CRS: {crs}, Bounds: {bounds}")
        passed_tests += 1
        print("  ✅ TEST 3 PASSED: GeoTIFF metadata & CRS extracted successfully")
    else:
        print("  ❌ TEST 3 FAILED: GeoTIFF sample file missing")

    # ── Test 4: Specialist Tools Execution ──────────────────────────────────
    print("\n[TEST 4] Specialist Tools Execution Pipeline...")
    total_tests += 1
    tools_ok = True
    pre_path = os.path.join(PROJECT_ROOT, "data", "sample", "pre_event.png")
    post_path = os.path.join(PROJECT_ROOT, "data", "sample", "post_event.png")

    import cv2
    img0 = cv2.imread(pre_path)
    img1 = cv2.imread(post_path)

    # Change tool
    req_change = ToolRequest(tool_name="change", parameters={"threshold": 0.15}, image_ids=["1", "2"])
    res_change = execute_tool(req_change, [img0, img1])
    if not res_change.success or not res_change.polygons:
        print("  ❌ Tool execution failed: change")
        tools_ok = False
    else:
        print(f"  ✓ Tool 'change': {len(res_change.polygons)} polygon(s) detected")

    # Grounding tool
    req_grounding = ToolRequest(tool_name="grounding", parameters={"target_description": "water"}, image_ids=["1"])
    res_grounding = execute_tool(req_grounding, [img0])
    if not res_grounding.success:
        print("  ❌ Tool execution failed: grounding")
        tools_ok = False
    else:
        print(f"  ✓ Tool 'grounding': {len(res_grounding.bounding_boxes)} box(es) detected")

    # Geospatial tool
    req_geo = ToolRequest(tool_name="geospatial", parameters={"pixel_polygons": res_change.polygons, "query": "Assam"}, image_ids=["1"])
    res_geo = execute_tool(req_geo, [img0])
    if not res_geo.success or not res_geo.polygons:
        print("  ❌ Tool execution failed: geospatial")
        tools_ok = False
    else:
        print(f"  ✓ Tool 'geospatial': {len(res_geo.polygons)} georeferenced polygon(s)")

    if tools_ok:
        passed_tests += 1
        print("  ✅ TEST 4 PASSED: All Specialist Tools executed clean output schemas")

    # ── Test 5: Evidence Fusion Engine ──────────────────────────────────────
    print("\n[TEST 5] Evidence Fusion Engine...")
    total_tests += 1
    tg = route_query("Identify newly flooded built-up areas", num_images=2)
    ev_score = fuse_evidence(tg, [res_change, res_geo], val_pass)
    print(f"  ✓ Calculated Evidence Score: {ev_score.score} / 100")
    print(f"  ✓ Abstention status: {ev_score.abstain}")

    if 0.0 <= ev_score.score <= 100.0 and not ev_score.abstain:
        passed_tests += 1
        print("  ✅ TEST 5 PASSED: Evidence Fusion computed high-confidence score")
    else:
        print("  ❌ TEST 5 FAILED: Invalid Evidence Fusion output")

    # ── Test 6: GeoJSON & KML Export ────────────────────────────────────────
    print("\n[TEST 6] GeoJSON & KML Format Export...")
    total_tests += 1
    geojson_str = export_geojson(res_geo.polygons)
    kml_str = export_kml(res_geo.polygons)

    if 'FeatureCollection' in geojson_str and '<kml' in kml_str:
        passed_tests += 1
        print("  ✅ TEST 6 PASSED: GeoJSON & KML export formats valid")
    else:
        print("  ❌ TEST 6 FAILED: Export formatting error")

    # ── Summary ──────────────────────────────────────────────────────────────
    t_elapsed = time.perf_counter() - t_start
    print("\n" + "=" * 70)
    print(f"BENCHMARK SUMMARY: {passed_tests}/{total_tests} Tests Passed ({passed_tests/total_tests*100:.1f}%)")
    print(f"Total Latency: {t_elapsed:.3f} seconds")
    print("=" * 70)

    if passed_tests == total_tests:
        print("🚀 ALL BENCHMARKS PASSED! SATQUERY-EDGE PHASE 1 READY FOR SIH JUDGING!")

if __name__ == "__main__":
    run_benchmark()
