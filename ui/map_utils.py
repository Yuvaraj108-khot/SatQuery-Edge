"""
ui/map_utils.py
Folium map rendering utilities for SatQuery-Edge.
Converts GeoPolygon objects to Folium map layers.
"""

from __future__ import annotations
from typing import Optional
import folium
from controller.schemas import GeoPolygon

# Default center for demo investigations
_DEFAULT_CENTER = [20.296, 85.824]   # Bhubaneswar, Odisha, India
_DEFAULT_ZOOM = 13


def build_investigation_map(
    geo_polygons: list[GeoPolygon],
    center: Optional[list[float]] = None,
    zoom: int = _DEFAULT_ZOOM,
) -> folium.Map:
    """
    Build a Folium map with all detected candidate regions overlaid as polygons.
    """
    if center is None:
        if geo_polygons:
            # Compute centroid of first polygon
            coords = geo_polygons[0].coordinates
            if coords:
                lons = [c[0] for c in coords]
                lats = [c[1] for c in coords]
                center = [sum(lats) / len(lats), sum(lons) / len(lons)]
            else:
                center = _DEFAULT_CENTER
        else:
            center = _DEFAULT_CENTER

    m = folium.Map(
        location=center,
        zoom_start=zoom,
        tiles="CartoDB dark_matter",
        prefer_canvas=True,
    )

    # Confidence color scale
    def _conf_color(conf: float) -> str:
        if conf >= 0.85:
            return "#ff6b35"
        elif conf >= 0.70:
            return "#f7c59f"
        elif conf >= 0.55:
            return "#efefd0"
        else:
            return "#80ced7"

    for i, poly in enumerate(geo_polygons):
        if not poly.coordinates or len(poly.coordinates) < 3:
            continue

        # Folium expects [lat, lon]
        folium_coords = [[c[1], c[0]] for c in poly.coordinates]
        color = _conf_color(poly.confidence)

        popup_html = f"""
        <div style="font-family:monospace; font-size:12px; min-width:200px;">
            <b style="color:{color};">Region #{i+1}</b><br>
            Label: {poly.label}<br>
            Confidence: <b>{poly.confidence:.2f}</b><br>
            Area (approx): {poly.area_km2_approx:.2f} km²<br>
            {"<i>⚠ Demo georef</i>" if poly.is_demo_georef else "Real GeoTIFF"}
        </div>
        """

        folium.Polygon(
            locations=folium_coords,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.25,
            weight=2.5,
            popup=folium.Popup(popup_html, max_width=280),
            tooltip=f"Region #{i+1} | conf: {poly.confidence:.2f}",
        ).add_to(m)

        # Centroid marker
        if len(folium_coords) > 1:
            lat_c = sum(p[0] for p in folium_coords) / len(folium_coords)
            lon_c = sum(p[1] for p in folium_coords) / len(folium_coords)
            folium.CircleMarker(
                location=[lat_c, lon_c],
                radius=6,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.8,
                popup=f"Centroid: {lat_c:.4f}°N, {lon_c:.4f}°E",
            ).add_to(m)

    # Demo notice if any polygon uses demo georef
    if any(p.is_demo_georef for p in geo_polygons):
        folium.Marker(
            location=[center[0] - 0.02, center[1] - 0.02],
            popup="⚠ Approximate demo georeferencing applied",
            icon=folium.Icon(color="orange", icon="info-sign"),
        ).add_to(m)

    return m
