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

# Default demo center (Bhubaneswar, Odisha, India) & affine scale (~22m/px)
_DEMO_CENTER_LAT = 20.296
_DEMO_CENTER_LON = 85.824
_DEMO_PIXEL_TO_DEG = 0.0002

# Offline Geocoding Database for Indian Locations
LOCATION_DATABASE = {
    "assam": (26.2006, 92.9376, "Assam, India"),
    "guwahati": (26.1445, 91.7362, "Guwahati, Assam"),
    "kaziranga": (26.5775, 93.1711, "Kaziranga, Assam"),
    "silchar": (24.8333, 92.7789, "Silchar, Assam"),
    "uttarakhand": (30.0668, 79.0193, "Uttarakhand, India"),
    "dehradun": (30.3165, 78.0322, "Dehradun, Uttarakhand"),
    "chamoli": (30.4124, 79.3304, "Chamoli, Uttarakhand"),
    "joshimath": (30.5574, 79.5664, "Joshimath, Uttarakhand"),
    "delhi": (28.6139, 77.2090, "Delhi NCR, India"),
    "wayanad": (11.6854, 76.1320, "Wayanad, Kerala"),
    "kerala": (10.8505, 76.2711, "Kerala, India"),
    "kochi": (9.9312, 76.2673, "Kochi, Kerala"),
    "odisha": (20.9517, 85.0985, "Odisha, India"),
    "bhubaneswar": (20.2960, 85.8240, "Bhubaneswar, Odisha"),
    "puri": (19.8135, 85.8312, "Puri, Odisha"),
    "cuttack": (20.4625, 85.8828, "Cuttack, Odisha"),
    "mumbai": (19.0760, 72.8777, "Mumbai, Maharashtra"),
    "maharashtra": (19.7515, 75.7139, "Maharashtra, India"),
    "kolkata": (22.5726, 88.3639, "Kolkata, West Bengal"),
    "bengaluru": (12.9716, 77.5946, "Bengaluru, Karnataka"),
    "chennai": (13.0827, 80.2707, "Chennai, Tamil Nadu"),
    "hyderabad": (17.3850, 78.4867, "Hyderabad, Telangana"),
    "shimla": (31.1048, 77.1734, "Shimla, Himachal Pradesh"),
    "srinagar": (34.0837, 74.7973, "Srinagar, Jammu & Kashmir"),
    "ladakh": (34.1526, 77.5771, "Ladakh, India"),
    "gujarat": (22.2587, 71.1924, "Gujarat, India"),
    "surat": (21.1702, 72.8311, "Surat, Gujarat"),
    "goa": (15.2993, 74.1240, "Goa, India"),
}


def resolve_location_coords(parameters: dict[str, Any]) -> tuple[float, float, str]:
    """Dynamically determine center Lat/Lon from parameters, dropdown, or query text."""
    # 1. Custom Lat/Lon overrides
    if parameters.get("custom_lat") is not None and parameters.get("custom_lon") is not None:
        try:
            clat = float(parameters["custom_lat"])
            clon = float(parameters["custom_lon"])
            return clat, clon, f"Custom Coordinates ({clat:.4f}°N, {clon:.4f}°E)"
        except (ValueError, TypeError):
            pass

    # 2. Selected Location in Sidebar dropdown
    sel = str(parameters.get("selected_location", "")).lower().strip()
    for key, (lat, lon, name) in LOCATION_DATABASE.items():
        if key in sel:
            return lat, lon, name

    # 3. Query Text NLP Search
    q = str(parameters.get("query", "")).lower()
    for key, (lat, lon, name) in LOCATION_DATABASE.items():
        if key in q:
            return lat, lon, name

    # 4. Default Fallback
    return _DEMO_CENTER_LAT, _DEMO_CENTER_LON, "Bhubaneswar, Odisha (Default Demo)"


def run_geospatial(images: list[Any], parameters: dict[str, Any]) -> ToolResult:
    """
    Convert pixel-space bounding boxes / polygons to geographic coordinates.
    Uses rasterio transform if available; dynamic georeferencing otherwise.
    """
    pixel_boxes: list[BoundingBox] = parameters.get("pixel_boxes", [])
    pixel_polygons: list[GeoPolygon] = parameters.get("pixel_polygons", [])
    image_width: int = parameters.get("image_width", 512)
    image_height: int = parameters.get("image_height", 512)
    geo_transform: Optional[Any] = parameters.get("geo_transform", None)
    crs_str: str = parameters.get("crs", "Demo CRS")

    center_lat, center_lon, loc_name = resolve_location_coords(parameters)
    is_demo = geo_transform is None
    geo_polygons: list[GeoPolygon] = []

    if geo_transform is not None:
        try:
            geo_polygons = _convert_with_rasterio(
                pixel_polygons, pixel_boxes,
                image_width, image_height, geo_transform,
            )
        except Exception:
            is_demo = True
            geo_polygons = _convert_demo(
                pixel_polygons, pixel_boxes,
                image_width, image_height,
                center_lat=center_lat, center_lon=center_lon,
            )
    else:
        geo_polygons = _convert_demo(
            pixel_polygons, pixel_boxes,
            image_width, image_height,
            center_lat=center_lat, center_lon=center_lon,
        )

    total_area = sum(p.area_km2_approx for p in geo_polygons)

    if is_demo:
        georef_note = (
            f"📍 Location Georeferenced to: {loc_name}\n"
            f"Center Coords: ({center_lat:.4f}°N, {center_lon:.4f}°E)\n"
            "Approximate pixel-to-geographic affine scale applied."
        )
    else:
        georef_note = f"Real GeoTIFF georeferencing applied. CRS: {crs_str}"

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
            "model_engine": "GeoPandas + Rasterio CRS Georeferencing Engine",
            "location_name": loc_name,
            "region_count": len(geo_polygons),
            "total_area_km2": round(total_area, 3),
            "is_demo_georef": is_demo,
            "center_lat": center_lat,
            "center_lon": center_lon,
        },
        image_outputs={},
    )


# ─── Conversion helpers ───────────────────────────────────────────────────────

def _convert_demo(
    pixel_polygons: list[GeoPolygon],
    pixel_boxes: list[BoundingBox],
    image_width: int,
    image_height: int,
    center_lat: float = _DEMO_CENTER_LAT,
    center_lon: float = _DEMO_CENTER_LON,
) -> list[GeoPolygon]:
    """Assign geo-coordinates using dynamic center transform."""
    results: list[GeoPolygon] = []
    sources = pixel_polygons if pixel_polygons else _boxes_to_polygons(pixel_boxes)

    for poly in sources:
        geo_coords = []
        for px_norm, py_norm in poly.coordinates:
            lon = center_lon + (px_norm - 0.5) * image_width * _DEMO_PIXEL_TO_DEG
            lat = center_lat - (py_norm - 0.5) * image_height * _DEMO_PIXEL_TO_DEG
            geo_coords.append([round(lon, 6), round(lat, 6)])

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


def export_geojson(polygons: list[GeoPolygon]) -> str:
    """Export GeoPolygon list as valid GeoJSON FeatureCollection string."""
    import json
    features = []
    for i, poly in enumerate(polygons):
        if not poly.coordinates:
            continue
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [poly.coordinates]
            },
            "properties": {
                "id": f"region_{i+1}",
                "label": poly.label,
                "confidence": poly.confidence,
                "area_km2_approx": poly.area_km2_approx,
                "is_demo_georef": poly.is_demo_georef,
            }
        })
    fc = {
        "type": "FeatureCollection",
        "name": "SatQuery_Edge_Detections",
        "crs": {
            "type": "name",
            "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}
        },
        "features": features,
    }
    return json.dumps(fc, indent=2)


def export_kml(polygons: list[GeoPolygon]) -> str:
    """Export GeoPolygon list as KML string for Google Earth / ISRO Bhuvan."""
    kml_lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<kml xmlns="http://www.opengis.net/kml/2.2">',
        '  <Document>',
        '    <name>SatQuery-Edge Detections</name>',
        '    <description>Satellite Intelligence Analysis Results - SIH 26167 (ISRO)</description>',
        '    <Style id="detectPoly">',
        '      <LineStyle><color>ff00aaff</color><width>3</width></LineStyle>',
        '      <PolyStyle><color>7f00aaff</color></PolyStyle>',
        '    </Style>',
    ]
    for i, poly in enumerate(polygons):
        if not poly.coordinates:
            continue
        coord_str = " ".join(f"{c[0]},{c[1]},0" for c in poly.coordinates)
        kml_lines.extend([
            '    <Placemark>',
            f'      <name>Region #{i+1} ({poly.label})</name>',
            f'      <description>Confidence: {poly.confidence:.2f} | Area: {poly.area_km2_approx:.2f} km²</description>',
            '      <styleUrl>#detectPoly</styleUrl>',
            '      <Polygon>',
            '        <outerBoundaryIs>',
            '          <LinearRing>',
            f'            <coordinates>{coord_str}</coordinates>',
            '          </LinearRing>',
            '        </outerBoundaryIs>',
            '      </Polygon>',
            '    </Placemark>',
        ])
    kml_lines.extend([
        '  </Document>',
        '</kml>',
    ])
    return "\n".join(kml_lines)

