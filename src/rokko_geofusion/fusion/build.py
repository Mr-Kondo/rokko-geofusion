"""Assemble the multimodal feature table and the fused class product.

For every cell of a common grid this produces one row carrying

    x, y, z, r, g, b, elevation, object_height, slope, aspect, relief,
    image_class, image_confidence, GIS attributes, fused_class, rule

which is the feature vector the project brief asks for. The table is written
tile by tile straight into a Parquet file, so the working set is one tile and
not the whole ROI.

Missing modalities are represented as NaN / "unavailable", never as zero: a
cell with no DSM has ``object_height = NaN`` and the height-dependent fusion
rules record themselves as skipped.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from rokko_geofusion.config import Config
from rokko_geofusion.crs import Bounds, GridSpec, RoiGeometry
from rokko_geofusion.exceptions import UnsupportedError
from rokko_geofusion.fusion.rules import FusionInputs, apply_rules, class_fractions
from rokko_geofusion.imagery.sampler import sample_raster, sample_rgb
from rokko_geofusion.io.raster import write_grid_raster
from rokko_geofusion.lidar.pointcloud import cell_centers_in, iter_tiles
from rokko_geofusion.utils.metadata import build_metadata, write_sidecar

logger = logging.getLogger(__name__)

#: Column order of the fusion table.
FEATURE_COLUMNS = (
    "x", "y", "z",
    "red", "green", "blue",
    "elevation", "object_height", "slope", "aspect", "relief",
    "image_class", "image_confidence",
    "in_building", "on_road", "in_water", "building_height_osm",
    "fused_class", "rule",
)


@dataclass
class FusionProduct:
    table_path: Path
    raster_path: Path
    grid: GridSpec
    class_names: list[str]
    n_cells: int
    statistics: dict[str, Any] = field(default_factory=dict)
    unavailable: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    pointcloud_path: Path | None = None


def _optional(path: Path) -> Path | None:
    return path if path.is_file() else None


def _rasterize_gis(config: Config, grid: GridSpec) -> dict[str, np.ndarray]:
    """Burn the GIS layers onto the fusion grid once, up front.

    A boolean mask for the whole ROI is a few megabytes at 1 m, which is far
    cheaper than running a spatial join per tile.
    """
    from rasterio.features import rasterize

    from rokko_geofusion.gis.osm import load_layer

    masks: dict[str, np.ndarray] = {}
    specifications = [
        ("in_building", "building", 0.0),
        ("on_road", "road", 4.0),      # OSM roads are centrelines; give them width
        ("in_water", "water", 2.0),
    ]
    for name, layer, buffer_m in specifications:
        try:
            frame = load_layer(config, layer)
        except FileNotFoundError:
            logger.warning("GIS layer %s is not available; %s will be unknown", layer, name)
            continue
        geometries = [
            (geometry.buffer(buffer_m) if buffer_m else geometry, 1)
            for geometry in frame.geometry
            if geometry is not None and not geometry.is_empty
        ]
        masks[name] = (
            rasterize(geometries, out_shape=grid.shape, transform=grid.transform,
                      fill=0, dtype="uint8").astype(bool)
            if geometries else np.zeros(grid.shape, dtype=bool)
        )
        logger.info("GIS mask %-12s covers %.1f%% of the ROI", name, 100 * masks[name].mean())

    # Mapped building heights (an attribute, not a measurement).
    try:
        buildings = load_layer(config, "building")
    except FileNotFoundError:
        return masks
    if "height_m" in buildings.columns:
        with_height = buildings[buildings["height_m"].notna()]
        if len(with_height):
            masks["building_height_osm"] = rasterize(
                ((geometry, float(height)) for geometry, height
                 in zip(with_height.geometry, with_height["height_m"], strict=True)
                 if geometry is not None and not geometry.is_empty),
                out_shape=grid.shape, transform=grid.transform,
                fill=np.nan, dtype="float32",
            )
            logger.info("mapped building heights available for %d/%d footprints",
                        len(with_height), len(buildings))
    return masks


def _mask_values(mask: np.ndarray | None, grid: GridSpec,
                 x: np.ndarray, y: np.ndarray) -> np.ndarray | None:
    """Look up a full-ROI raster mask at the given cell centres."""
    if mask is None:
        return None
    cols = np.floor((x - grid.bounds[0]) / grid.resolution_m).astype(np.int64)
    rows = np.floor((grid.bounds[3] - y) / grid.resolution_m).astype(np.int64)
    # Cell centres are inside the grid by construction; the clip is here
    # because a negative index would wrap silently to the far edge instead of
    # failing, which would be a very quiet spatial bug.
    np.clip(cols, 0, grid.width - 1, out=cols)
    np.clip(rows, 0, grid.height - 1, out=rows)
    return mask[rows, cols]


def build_fusion(
    config: Config,
    roi: RoiGeometry,
    *,
    overwrite: bool = False,
) -> FusionProduct:
    """Build the fused feature table, class raster and (optionally) point cloud."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    settings = config.fusion
    if not settings.enabled:
        raise UnsupportedError("fusion.enabled is false")

    grid = roi.grid(settings.cell_size_m)
    interim, raster_dir = config.paths.interim, config.paths.raster

    sources = {
        "dem": _optional(interim / "dem.tif"),
        "orthophoto": _optional(interim / "orthophoto.tif"),
        "slope": _optional(raster_dir / "slope.tif"),
        "aspect": _optional(raster_dir / "aspect.tif"),
        "relief": _optional(raster_dir / "relief.tif"),
        "ndsm": _optional(raster_dir / "ndsm.tif"),
        "segmentation_class": _optional(raster_dir / "segmentation_class.tif"),
        "segmentation_confidence": _optional(raster_dir / "segmentation_confidence.tif"),
    }
    if sources["dem"] is None:
        raise UnsupportedError("fusion needs a DEM; run scripts/download_lidar.py first")

    unavailable: dict[str, str] = {}
    if sources["ndsm"] is None:
        unavailable["object_height"] = (
            "no DSM is configured, so height above ground is unknown; the "
            "height-dependent fusion rules are skipped"
        )
    if sources["segmentation_class"] is None:
        unavailable["image_class"] = (
            "no segmentation raster; run scripts/segment_imagery.py to add image "
            "semantics to the fusion"
        )
    for name, reason in unavailable.items():
        logger.warning("%s unavailable: %s", name, reason)

    gis_masks = _rasterize_gis(config, grid) if settings.use_gis_evidence else {}

    table_path = config.paths.vector / "fusion_cells.parquet"
    raster_path = raster_dir / "fused_class.tif"
    class_names = list(settings.classes)
    image_classes = list(config.segmentation.classes)

    fused_raster = np.zeros(grid.shape, dtype=np.uint8)
    provenance_total: dict[str, int] = {}
    skipped_conditions: list[str] = []
    writer: pq.ParquetWriter | None = None
    n_rows = 0
    cloud_writer = None

    if settings.write_pointcloud:
        from rokko_geofusion.lidar.pointcloud import PointWriter

        cloud_path = config.paths.pointcloud / "cloud_fused.laz"
        cloud_writer = PointWriter(cloud_path, "laz", grid.crs, with_rgb=True)
    else:
        cloud_path = None

    table_path.parent.mkdir(parents=True, exist_ok=True)
    tiles = list(iter_tiles(grid, settings.chunk_size_m))
    logger.info("fusing %d cells at %.2f m over %d tiles",
                grid.width * grid.height, grid.resolution_m, len(tiles))

    try:
        for index, bounds in enumerate(tiles, start=1):
            frame, fused, rule, provenance = _fuse_tile(
                config, grid, bounds, sources, gis_masks, image_classes
            )
            if frame is None:
                continue

            for name, count in provenance["rule_counts"].items():
                provenance_total[name] = provenance_total.get(name, 0) + count
            for condition in provenance["skipped_conditions"]:
                if condition not in skipped_conditions:
                    skipped_conditions.append(condition)

            cols = np.floor((frame["x"].to_numpy() - grid.bounds[0])
                            / grid.resolution_m).astype(np.int64)
            rows = np.floor((grid.bounds[3] - frame["y"].to_numpy())
                            / grid.resolution_m).astype(np.int64)
            inside = (rows >= 0) & (rows < grid.height) & (cols >= 0) & (cols < grid.width)
            fused_raster[rows[inside], cols[inside]] = fused[inside]

            table = pa.Table.from_pandas(frame[list(FEATURE_COLUMNS)], preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(table_path, table.schema, compression="zstd")
            writer.write_table(table)
            n_rows += len(frame)

            if cloud_writer is not None:
                cloud_writer.write(
                    frame[["x", "y", "z"]].to_numpy(dtype=np.float64),
                    frame[["red", "green", "blue"]].to_numpy(dtype=np.uint8),
                    classification=fused,
                )
            if index % 10 == 0 or index == len(tiles):
                logger.info("fusion: tile %d/%d (%s rows)", index, len(tiles), f"{n_rows:,}")
    finally:
        if writer is not None:
            writer.close()
        if cloud_writer is not None:
            cloud_writer.close()

    if n_rows == 0:
        raise UnsupportedError(f"fusion produced no rows for ROI {roi.key}")

    for condition in skipped_conditions:
        logger.warning("fusion rule condition NOT applied: %s", condition)

    statistics, metadata = _summarise_fusion(
        config, roi, grid,
        fused_raster=fused_raster,
        n_rows=n_rows,
        sources=sources,
        rule_counts=provenance_total,
        skipped_conditions=skipped_conditions,
        unavailable=unavailable,
    )
    fractions = statistics["class_fractions"]
    write_grid_raster(raster_path, fused_raster, grid, nodata=None,
                      compress=config.output.compress,
                      band_descriptions=["fused_class"], metadata=metadata)
    write_sidecar(table_path, metadata)
    logger.info("fused classes: %s",
                ", ".join(f"{name}={value:.1%}" for name, value in fractions.items() if value))

    return FusionProduct(
        table_path=table_path,
        raster_path=raster_path,
        grid=grid,
        class_names=class_names,
        n_cells=n_rows,
        statistics=statistics,
        unavailable=unavailable,
        metadata=metadata,
        pointcloud_path=cloud_path,
    )


def _summarise_fusion(
    config: Config,
    roi: RoiGeometry,
    grid: GridSpec,
    *,
    fused_raster: np.ndarray,
    n_rows: int,
    sources: dict[str, Path | None],
    rule_counts: dict[str, int],
    skipped_conditions: list[str],
    unavailable: dict[str, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Statistics and metadata for a completed fusion pass."""
    settings = config.fusion
    class_names = list(settings.classes)
    fractions = class_fractions(fused_raster.ravel(), class_names)
    statistics = {
        "cells": int(n_rows),
        "cell_size_m": grid.resolution_m,
        "area_m2": float(n_rows * grid.resolution_m**2),
        "class_fractions": fractions,
        "rule_counts": rule_counts,
        "skipped_conditions": skipped_conditions,
        "modalities": {
            name: ("available" if path else "unavailable")
            for name, path in sources.items()
        },
    }
    metadata = build_metadata(
        kind="fusion",
        source="fusion:" + "+".join(sorted(k for k, v in sources.items() if v)),
        crs=grid.crs,
        resolution_m=grid.resolution_m,
        roi=roi.to_dict(),
        acquisition={name: str(path) for name, path in sources.items() if path},
        processing={
            "grid": grid.to_dict(),
            "chunk_size_m": settings.chunk_size_m,
            "rules": "see rokko_geofusion.fusion.rules.apply_rules",
            "thresholds": settings.thresholds.model_dump(),
            "use_gis_evidence": settings.use_gis_evidence,
            "skipped_conditions": skipped_conditions,
        },
        config_fingerprint=config.fingerprint(),
        class_names=class_names,
        class_fractions=fractions,
        rule_counts=rule_counts,
        unavailable=unavailable,
        notes=(
            "Rule-based fusion of image semantics, terrain and mapped geometry. "
            "Each cell records the rule that assigned it."
        ),
    )
    return statistics, metadata


def _fuse_tile(
    config: Config,
    grid: GridSpec,
    bounds: Bounds,
    sources: dict[str, Path | None],
    gis_masks: dict[str, np.ndarray],
    image_classes: list[str],
):
    """Sample every modality for one tile and apply the fusion rules."""
    import pandas as pd

    x, y = cell_centers_in(grid, bounds)
    if x.size == 0:
        return None, None, None, None

    crs = grid.crs

    def sample_at(points_x, points_y, name, method="bilinear"):
        """Sample one source at the given coordinates; NaN when it is absent."""
        path = sources.get(name)
        if path is None:
            return np.full(points_x.size, np.nan, np.float32)
        return sample_raster(path, points_x, points_y, expected_crs=crs, bands=[1],
                             method=method)[0].astype(np.float32)

    # Cells without ground elevation carry no usable feature vector, so they
    # are dropped before every other modality is sampled.
    elevation = sample_at(x, y, "dem")
    valid = np.isfinite(elevation)
    if not valid.any():
        return None, None, None, None
    x, y, elevation = x[valid], y[valid], elevation[valid]

    rgb = (sample_rgb(sources["orthophoto"], x, y, expected_crs=crs, method="nearest")
           if sources["orthophoto"] else np.zeros((x.size, 3), np.uint8))

    object_height = sample_at(x, y, "ndsm")
    slope = sample_at(x, y, "slope")
    aspect = sample_at(x, y, "aspect", "nearest")
    relief = sample_at(x, y, "relief")
    image_class = sample_at(x, y, "segmentation_class", "nearest")
    image_confidence = sample_at(x, y, "segmentation_confidence")

    image_class_index = np.where(np.isfinite(image_class), image_class, -1).astype(np.int16)
    image_confidence = np.where(np.isfinite(image_confidence), image_confidence, 0.0)

    in_building = _mask_values(gis_masks.get("in_building"), grid, x, y)
    on_road = _mask_values(gis_masks.get("on_road"), grid, x, y)
    in_water = _mask_values(gis_masks.get("in_water"), grid, x, y)
    building_height = _mask_values(gis_masks.get("building_height_osm"), grid, x, y)

    fused, rule, provenance = apply_rules(
        FusionInputs(
            image_class=image_class_index,
            image_confidence=image_confidence,
            object_height=object_height if sources["ndsm"] else None,
            in_building=in_building,
            on_road=on_road,
            in_water=in_water,
        ),
        config=config.fusion,
        image_classes=image_classes,
    )

    frame = pd.DataFrame(
        {
            "x": x.astype(np.float64),
            "y": y.astype(np.float64),
            "z": elevation.astype(np.float32),
            "red": rgb[:, 0],
            "green": rgb[:, 1],
            "blue": rgb[:, 2],
            "elevation": elevation.astype(np.float32),
            "object_height": object_height,
            "slope": slope,
            "aspect": aspect,
            "relief": relief,
            "image_class": image_class_index,
            "image_confidence": image_confidence.astype(np.float32),
            "in_building": (in_building if in_building is not None
                            else np.zeros(x.size, bool)),
            "on_road": on_road if on_road is not None else np.zeros(x.size, bool),
            "in_water": in_water if in_water is not None else np.zeros(x.size, bool),
            "building_height_osm": (building_height if building_height is not None
                                    else np.full(x.size, np.nan, np.float32)),
            "fused_class": fused,
            "rule": rule,
        }
    )
    return frame, fused, rule, provenance
