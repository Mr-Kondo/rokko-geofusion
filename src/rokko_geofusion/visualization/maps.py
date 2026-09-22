"""Interactive 2D maps for the notebook.

Display code only: nothing here computes anything. Geometries arrive in the
projected CRS and are converted to WGS84 for the web map, because that is what
Leaflet expects -- the conversion is explicit, never assumed.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from rokko_geofusion.config import Config
from rokko_geofusion.crs import RoiGeometry

logger = logging.getLogger(__name__)

LAYER_STYLES: dict[str, dict[str, Any]] = {
    "building": {"color": "#e15759", "weight": 1, "fillOpacity": 0.45},
    "road": {"color": "#f2c744", "weight": 2, "fillOpacity": 0.0},
    "water": {"color": "#4e79a7", "weight": 1, "fillOpacity": 0.5},
    "landuse": {"color": "#59a14f", "weight": 1, "fillOpacity": 0.25},
    "railway": {"color": "#b07aa1", "weight": 2, "fillOpacity": 0.0},
}
#: Columns kept in map popups; anything else would bloat the HTML.
POPUP_FIELDS = ("osm_id", "name", "building", "highway", "landuse", "height_m")


def roi_map(
    config: Config,
    roi: RoiGeometry,
    *,
    layers: dict[str, Any] | None = None,
    zoom_start: int = 15,
    max_features_per_layer: int = 3000,
):
    """A folium map of the ROI with the downloaded GIS layers on top."""
    import folium

    lon, lat = roi.center_geographic
    fmap = folium.Map(
        location=[lat, lon],
        zoom_start=zoom_start,
        tiles=None,
        control_scale=True,
    )
    folium.TileLayer(
        tiles=config.visualization.basemap,
        attr=config.visualization.basemap_attribution,
        name="GSI base map",
        overlay=False,
    ).add_to(fmap)
    folium.TileLayer(
        tiles=(f"{config.imagery.base_url}/{config.imagery.dataset}"
               "/{z}/{x}/{y}." + config.imagery.extension),
        attr=config.imagery.attribution,
        name=f"GSI {config.imagery.dataset}",
        overlay=False,
        max_zoom=18,
    ).add_to(fmap)

    min_lon, min_lat, max_lon, max_lat = roi.bounds_geographic
    folium.Rectangle(
        bounds=[[min_lat, min_lon], [max_lat, max_lon]],
        color="#111111",
        weight=2,
        fill=False,
        tooltip=f"ROI {roi.key}",
    ).add_to(fmap)

    for name, frame in (layers or {}).items():
        if frame is None or len(frame) == 0:
            continue
        subset = frame
        if len(frame) > max_features_per_layer:
            logger.warning(
                "layer %s has %d features; showing the first %d on the map "
                "(analysis still uses all of them)",
                name, len(frame), max_features_per_layer,
            )
            subset = frame.iloc[:max_features_per_layer]
        columns = [c for c in POPUP_FIELDS if c in subset.columns]
        geojson = subset[[*columns, "geometry"]].to_crs(config.crs.geographic)
        style = LAYER_STYLES.get(name, {"color": "#888888", "weight": 1})
        folium.GeoJson(
            geojson.to_json(),
            name=f"{name} ({len(frame)})",
            style_function=lambda _feature, style=style: style,
            tooltip=folium.GeoJsonTooltip(fields=columns) if columns else None,
        ).add_to(fmap)

    folium.LayerControl(collapsed=False).add_to(fmap)
    fmap.fit_bounds([[min_lat, min_lon], [max_lat, max_lon]])
    return fmap


def raster_overlay_map(
    config: Config,
    roi: RoiGeometry,
    raster_path: Path | str,
    *,
    name: str = "raster",
    colormap: str = "viridis",
    opacity: float = 0.7,
    max_side_px: int = 1024,
):
    """Overlay a single-band raster on a folium map (display downsampled)."""
    import folium
    import numpy as np
    import rasterio
    from matplotlib import cm
    from matplotlib.colors import Normalize
    from rasterio.enums import Resampling
    from rasterio.warp import transform_bounds

    with rasterio.open(raster_path) as dataset:
        scale = max(1, max(dataset.height, dataset.width) // max_side_px)
        array = dataset.read(
            1,
            out_shape=(dataset.height // scale, dataset.width // scale),
            resampling=Resampling.average,
        )
        west, south, east, north = transform_bounds(
            dataset.crs, config.crs.geographic, *dataset.bounds
        )

    finite = np.isfinite(array)
    if not finite.any():
        raise ValueError(f"{raster_path} contains no finite values to display")
    normalise = Normalize(vmin=float(np.nanpercentile(array[finite], 2)),
                          vmax=float(np.nanpercentile(array[finite], 98)))
    rgba = cm.get_cmap(colormap)(normalise(array))
    rgba[..., 3] = np.where(finite, opacity, 0.0)

    lon, lat = roi.center_geographic
    fmap = folium.Map(location=[lat, lon], zoom_start=15, tiles=None)
    folium.TileLayer(
        tiles=config.visualization.basemap,
        attr=config.visualization.basemap_attribution,
        name="GSI base map",
    ).add_to(fmap)
    folium.raster_layers.ImageOverlay(
        image=rgba,
        bounds=[[south, west], [north, east]],
        name=name,
        opacity=1.0,
    ).add_to(fmap)
    folium.LayerControl(collapsed=False).add_to(fmap)
    fmap.fit_bounds([[south, west], [north, east]])
    return fmap
