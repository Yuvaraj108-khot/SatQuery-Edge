"""
tools/geospatial.py
Geospatial reasoner — SatQuery-Edge.
Converts pixel bounding boxes / polygons into approximate lat/lon coordinates.
Uses rasterio transform if a GeoTIFF is provided; otherwise uses
deterministic demo georeferencing centered on a representative Indian location.
Clearly labels all demo coordinates as approximate.
"""

from __future__ import annotations
from typing import Any, Optional
import numpy as np
from controller.schemas import ToolResult, BoundingBox, GeoPolygon

# Demo center: Bhubaneswar, Odisha, India (prone to cyclone flooding)
_DEMO_CENTER_LAT = 20.296
_DEMO_CENTER_LON = 85.824
_DEMO_PIXEL_TO_DEG = 0.0002   # ~22m per pixel at equator


def run_geospatial(images: list[Any], parameters: dict[str, Any]) -> ToolResult:
    """
    Convert pixel-space bounding boxes / polygons to approximate geographic
    coordinates. Uses rasterio transform if available; demo coords otherwise.
    """
    pixel_boxes: list[BoundingBox] = parameters.get("pixel_boxes", [])
    pixel_polygons: list[GeoPolygon] = parameters.get("pixel_polygons", [])
    image_width: int = parameters.get("image_width", 512)
    image_height: int = parameters.get("image_height", 512)
    geo_transform: Optional[Any] = parameters.get("geo_transform", None)
    crs_str: str = parameters.get("crs", "Demo CRS")

    is_demo = geo_transform is None

    geo_polygons: list[GeoPolygon] = []

    if geo_transform is not None:
        # Real GeoTIFF transform via rasterio
        try:
            geo_polygons = _convert_with_rasterio(
                pixel_polygons, pixel_boxes,
                image_width, image_height, geo_transform,
            )
        except Exception as e:
            # Fallback to demo
            is_demo = True
            geo_polygons = _convert_demo(
                pixel_polygons, pixel_boxes,
                image_width, image_height,
            )
    else:
        geo_polygons = _convert_demo(
            pixel_polygons, pixel_boxes,
            image_width, image_height,
        )

    total_area = sum(p.area_km2_approx for p in geo_polygons)

    if is_demo:
        georef_note = (
            "⚠ Approximate demo georeferencing applied.\n"
            f"Demo center: ({_DEMO_CENTER_LAT:.3f}°N, {_DEMO_CENTER_LON:.3f}°E)\n"
            "Coordinates are NOT derived from real GeoTIFF metadata."
        )
    else:
        georef_note = f"Real georeferencing applied. CRS: {crs_str}"

    desc = (
        f"Geospatial conversion complete.\n"
        f"Converted {len(geo_polygons)} region(s) to geographic coordinates.\n"
        f"Total approximate area: {total_area:.2f} km².\n\n"
        f"{georef_note}"
    )

    return ToolResult(
        tool_name="geospatial",
        success=True,
        description=desc,
        polygons=geo_polygons,
        metrics={
            "region_count": len(geo_polygons),
            "total_area_km2": round(total_area, 3),
            "is_demo_georef": is_demo,
            "center_lat": _DEMO_CENTER_LAT,
            "center_lon": _DEMO_CENTER_LON,
        },
        image_outputs={},
    )


# ─── Conversion helpers ───────────────────────────────────────────────────────

def _convert_demo(
    pixel_polygons: list[GeoPolygon],
    pixel_boxes: list[BoundingBox],
    image_width: int,
    image_height: int,
) -> list[GeoPolygon]:
    """
    Assign approximate geo-coordinates using a demo affine transform
    centered on Bhubaneswar, India.
    """
    results: list[GeoPolygon] = []

    # Use polygons if available, else derive from bounding boxes
    sources = pixel_polygons if pixel_polygons else _boxes_to_polygons(pixel_boxes)

    for poly in sources:
        geo_coords = []
        for px_norm, py_norm in poly.coordinates:
            # px_norm, py_norm in [0, 1] (normalized pixel space)
            lon = _DEMO_CENTER_LON + (px_norm - 0.5) * image_width * _DEMO_PIXEL_TO_DEG
            lat = _DEMO_CENTER_LAT - (py_norm - 0.5) * image_height * _DEMO_PIXEL_TO_DEG
            geo_coords.append([round(lon, 6), round(lat, 6)])

        # Approximate area from bounding box of polygon
        lons = [c[0] for c in geo_coords]
        lats = [c[1] for c in geo_coords]
        dlon = max(lons) - min(lons)
        dlat = max(lats) - min(lats)
        area_km2 = round(dlon * 111.0 * dlat * 111.0, 3)

        results.append(GeoPolygon(
            coordinates=geo_coords,
            confidence=poly.confidence,
            label=poly.label,
            area_km2_approx=area_km2,
            is_demo_georef=True,
        ))

    return results


def _convert_with_rasterio(
    pixel_polygons: list[GeoPolygon],
    pixel_boxes: list[BoundingBox],
    image_width: int,
    image_height: int,
    transform: Any,
) -> list[GeoPolygon]:
    """Convert using rasterio affine transform."""
    results: list[GeoPolygon] = []
    sources = pixel_polygons if pixel_polygons else _boxes_to_polygons(pixel_boxes)

    for poly in sources:
        geo_coords = []
        for px_norm, py_norm in poly.coordinates:
            col = px_norm * image_width
            row = py_norm * image_height
            lon, lat = transform * (col, row)
            geo_coords.append([round(float(lon), 6), round(float(lat), 6)])

        lons = [c[0] for c in geo_coords]
        lats = [c[1] for c in geo_coords]
        dlon = max(lons) - min(lons)
        dlat = max(lats) - min(lats)
        area_km2 = round(dlon * 111.0 * dlat * 111.0, 3)

        results.append(GeoPolygon(
            coordinates=geo_coords,
            confidence=poly.confidence,
            label=poly.label,
            area_km2_approx=area_km2,
            is_demo_georef=False,
        ))

    return results


def _boxes_to_polygons(boxes: list[BoundingBox]) -> list[GeoPolygon]:
    """Convert bounding boxes to rectangular normalized polygons."""
    polys: list[GeoPolygon] = []
    for b in boxes:
        polys.append(GeoPolygon(
            coordinates=[
                [b.x1, b.y1], [b.x2, b.y1],
                [b.x2, b.y2], [b.x1, b.y2],
                [b.x1, b.y1],
            ],
            confidence=b.confidence,
            label=b.label,
            area_km2_approx=0.0,
            is_demo_georef=True,
        ))
    return polys
