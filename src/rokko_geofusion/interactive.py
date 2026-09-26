"""The interactive GeoAI entry points used from the notebook.

Two operations, deliberately separated because they cost very different things:

``analyze_roi``
    a *new* area: acquire, process and analyse it end to end. Minutes.
``summarize_subregion``
    a sub-box of the area already processed: statistics straight from the
    existing rasters, with no downloads. Milliseconds -- this is what a map
    selection should call.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from rokko_geofusion.config import Config, load_config
from rokko_geofusion.crs import Bounds, RoiGeometry, roi_from_config
from rokko_geofusion.exceptions import UnsupportedError
from rokko_geofusion.pipeline import PipelineResult, run_pipeline
from rokko_geofusion.report import build_payload
from rokko_geofusion.utils.cli import DEFAULT_CONFIG
from rokko_geofusion.utils.metadata import read_json

logger = logging.getLogger(__name__)


@dataclass
class RoiAnalysis:
    """Everything produced for one ROI."""

    config: Config
    roi: RoiGeometry
    payload: dict[str, Any]
    pipeline: PipelineResult | None = None
    products: dict[str, Path] = field(default_factory=dict)

    @property
    def failed_stages(self) -> list[str]:
        return self.pipeline.failed if self.pipeline else []

    def summary(self) -> dict[str, Any]:
        """Compact dictionary for printing in a notebook cell."""
        terrain = self.payload.get("terrain") or {}
        elevation = terrain.get("elevation_m") or {}
        slope = terrain.get("slope_deg") or {}
        fusion = self.payload.get("fusion") or {}
        return {
            "roi": self.roi.key,
            "area_km2": round(self.roi.area_m2 / 1e6, 3),
            "elevation_min_m": elevation.get("min"),
            "elevation_mean_m": elevation.get("mean"),
            "elevation_max_m": elevation.get("max"),
            "slope_mean_deg": slope.get("mean"),
            "object_height": terrain.get("object_height_m"),
            "fused_class_fractions": fusion.get("class_fractions"),
            "missing": list(self.payload.get("missing", {})),
            "failed_stages": self.failed_stages,
        }

    def figures(self) -> list[Path]:
        return sorted(self.config.paths.figures.glob("*.png"))


def _roi_overrides(
    *,
    bounds: Sequence[float] | None,
    center: Sequence[float] | None,
    radius_m: float | None,
    name: str | None,
) -> list[str]:
    overrides: list[str] = []
    if bounds is not None:
        if len(bounds) != 4:
            raise ValueError("bounds must be [min_lon, min_lat, max_lon, max_lat]")
        overrides.append("roi.center=null")
        overrides.append("roi.bbox=[" + ",".join(f"{value:.6f}" for value in bounds) + "]")
    elif center is not None:
        if len(center) != 2:
            raise ValueError("center must be (lat, lon)")
        overrides.append("roi.bbox=null")
        overrides.append(f"roi.center={{lat: {center[0]:.6f}, lon: {center[1]:.6f}}}")
        if radius_m is not None:
            overrides.append(f"roi.radius_m={float(radius_m)}")
    elif radius_m is not None:
        overrides.append(f"roi.radius_m={float(radius_m)}")
    if name:
        overrides.append(f"roi.name={name}")
    return overrides


def analyze_roi(
    bounds: Sequence[float] | None = None,
    *,
    center: Sequence[float] | None = None,
    radius_m: float | None = None,
    name: str | None = None,
    config_path: Path | str = DEFAULT_CONFIG,
    stages: Sequence[str] | None = None,
    overrides: Sequence[str] | None = None,
    overwrite: bool = False,
    run: bool = True,
) -> RoiAnalysis:
    """Analyse an area end to end.

    ``bounds`` is ``[min_lon, min_lat, max_lon, max_lat]``; alternatively give
    ``center=(lat, lon)`` with ``radius_m``. Everything else comes from the
    configuration file, so one call analyses a new place with the same,
    reproducible settings.
    """
    all_overrides = _roi_overrides(bounds=bounds, center=center, radius_m=radius_m, name=name)
    all_overrides += list(overrides or [])

    config = load_config(config_path, overrides=all_overrides)
    roi = roi_from_config(config)
    config.paths.ensure()
    logger.info("analyse %s", roi)

    pipeline_result: PipelineResult | None = None
    if run:
        pipeline_result = run_pipeline(
            stages,
            config_path=config.source_path or config_path,
            overrides=all_overrides,
            overwrite=overwrite,
            root=config.root,
        )
        if pipeline_result.failed:
            logger.warning("stages failed: %s", ", ".join(pipeline_result.failed))

    payload = build_payload(config, roi)
    products = {
        "dem": config.paths.interim / "dem.tif",
        "orthophoto": config.paths.interim / "orthophoto.tif",
        "pointcloud": config.paths.pointcloud / f"cloud.{config.output.pointcloud_format}",
        "fused_class": config.paths.raster / "fused_class.tif",
        "fusion_table": config.paths.vector / "fusion_cells.parquet",
    }
    # The Markdown report is only a product when the run that wrote it succeeded;
    # the JSON status says so, the file's existence alone does not.
    report_status = config.paths.reports / "geoai_report.json"
    if report_status.is_file() and read_json(report_status).get("status") == "ok":
        products["report"] = config.paths.reports / "geoai_report.md"
    return RoiAnalysis(
        config=config,
        roi=roi,
        payload=payload,
        pipeline=pipeline_result,
        products={key: path for key, path in products.items() if path.is_file()},
    )


def _projected_bounds(
    config: Config, roi: RoiGeometry,
    bounds: Sequence[float], geographic: bool
) -> Bounds:
    if len(bounds) != 4:
        raise ValueError("bounds must have four values")
    if not geographic:
        return (float(bounds[0]), float(bounds[1]), float(bounds[2]), float(bounds[3]))
    return roi.crs.transform_bounds(
        (float(bounds[0]), float(bounds[1]), float(bounds[2]), float(bounds[3])),
        config.crs.geographic, config.crs.projected,
    )


def summarize_subregion(
    config: Config,
    bounds: Sequence[float],
    *,
    geographic: bool = True,
) -> dict[str, Any]:
    """Statistics for a sub-box of the already-processed ROI.

    Reads only the window that overlaps ``bounds`` from each raster, so this
    stays fast no matter how large the processed area is.
    """
    import rasterio
    from rasterio.windows import from_bounds as window_from_bounds

    from rokko_geofusion.crs import GridSpec
    from rokko_geofusion.io.raster import grid_from_raster
    from rokko_geofusion.terrain.analysis import terrain_statistics

    roi = roi_from_config(config)
    selection = _projected_bounds(config, roi, bounds, geographic)

    elevation_path = config.paths.raster / "elevation.tif"
    if not elevation_path.is_file():
        raise UnsupportedError(
            "no terrain products for this ROI yet; run "
            "`python scripts/run_pipeline.py --stage terrain` first"
        )

    def window_read(path: Path) -> tuple[np.ndarray | None, Any]:
        if not path.is_file():
            return None, None
        with rasterio.open(path) as dataset:
            window = window_from_bounds(*selection, transform=dataset.transform)
            window = window.round_offsets().round_lengths()
            if window.width < 1 or window.height < 1:
                return None, None
            array = dataset.read(1, window=window).astype(np.float32)
            if dataset.nodata is not None and not np.isnan(dataset.nodata):
                array = np.where(array == dataset.nodata, np.nan, array)
            return array, dataset.window_transform(window)

    elevation, transform = window_read(elevation_path)
    if elevation is None or elevation.size == 0:
        raise UnsupportedError(
            f"the requested box {tuple(round(v, 5) for v in bounds)} does not overlap "
            f"the processed ROI {roi.key}"
        )

    base_grid = grid_from_raster(elevation_path)
    grid = GridSpec(
        bounds=(transform.c, transform.f - elevation.shape[0] * base_grid.resolution_m,
                transform.c + elevation.shape[1] * base_grid.resolution_m, transform.f),
        resolution_m=base_grid.resolution_m,
        crs=base_grid.crs,
    )

    slope, _ = window_read(config.paths.raster / "slope.tif")
    aspect, _ = window_read(config.paths.raster / "aspect.tif")
    relief, _ = window_read(config.paths.raster / "relief.tif")
    object_height, _ = window_read(config.paths.raster / "ndsm.tif")

    unavailable: dict[str, str] = {}
    if object_height is None:
        unavailable["object_height"] = "no DSM is configured for this ROI"

    statistics = terrain_statistics(
        grid=grid,
        elevation=elevation,
        slope_deg=slope if slope is not None else np.full_like(elevation, np.nan),
        aspect_deg=aspect if aspect is not None else np.full_like(elevation, np.nan),
        relief_m=relief,
        object_height=object_height,
        unavailable=unavailable,
    )

    fused_path = config.paths.raster / "fused_class.tif"
    if fused_path.is_file():
        with rasterio.open(fused_path) as dataset:
            window = window_from_bounds(*selection, transform=dataset.transform)
            window = window.round_offsets().round_lengths()
            classes = dataset.read(1, window=window)
        if classes.size:
            counts = np.bincount(classes.ravel(), minlength=len(config.fusion.classes))
            statistics["land_cover"] = {
                name: float(count / classes.size)
                for name, count in zip(config.fusion.classes, counts, strict=True)
            }

    statistics["selection"] = {
        "requested_bounds": [float(v) for v in bounds],
        "crs": config.crs.geographic if geographic else config.crs.projected,
        "projected_bounds": [float(v) for v in selection],
    }
    return statistics


def roi_selector(config: Config, *, zoom: int = 14):
    """An ipyleaflet map with a rectangle tool; returns ``(map, get_bounds)``.

    Colab's widget support is not always reliable, so this is a convenience,
    not the interface: ``analyze_roi(bounds=[...])`` does the same job from a
    plain Python call.
    """
    try:
        from ipyleaflet import DrawControl, Map, TileLayer, basemaps
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise UnsupportedError(
            "interactive selection needs ipyleaflet: pip install -e '.[viz]'. "
            "Use analyze_roi(bounds=[min_lon, min_lat, max_lon, max_lat]) instead."
        ) from exc

    roi = roi_from_config(config)
    lon, lat = roi.center_geographic
    leaflet_map = Map(center=(lat, lon), zoom=zoom, basemap=basemaps.OpenStreetMap.Mapnik,
                      scroll_wheel_zoom=True)
    leaflet_map.add_layer(TileLayer(
        url=(f"{config.imagery.base_url}/{config.imagery.dataset}"
             "/{z}/{x}/{y}." + config.imagery.extension),
        attribution=config.imagery.attribution, name=config.imagery.dataset,
    ))

    draw_control = DrawControl(
        rectangle={"shapeOptions": {"color": "#e15759", "fillOpacity": 0.1}},
        polygon={}, circle={}, circlemarker={}, polyline={}, marker={},
    )
    selected: dict[str, list[float]] = {}

    def handle(_self, action, geo_json):  # noqa: ANN001 - ipyleaflet callback
        if action != "created":
            return
        coordinates = geo_json["geometry"]["coordinates"][0]
        longitudes = [point[0] for point in coordinates]
        latitudes = [point[1] for point in coordinates]
        selected["bounds"] = [min(longitudes), min(latitudes),
                              max(longitudes), max(latitudes)]
        logger.info("selected bounds: %s",
                    [round(value, 5) for value in selected["bounds"]])

    draw_control.on_draw(handle)
    leaflet_map.add_control(draw_control)

    def get_bounds() -> list[float] | None:
        return selected.get("bounds")

    return leaflet_map, get_bounds
