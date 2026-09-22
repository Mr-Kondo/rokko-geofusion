"""Terrain derivatives and ROI statistics.

Everything here is computed in the projected CRS, where one unit is one metre,
so slopes and areas are meaningful. Slope and aspect use Horn's 3x3 operator
(the same one ESRI and GDAL use), which is well behaved on the noisy, resampled
grids we get from web tiles.

``nDSM`` (normalised DSM, i.e. height above ground) requires **both** a DSM and
a DEM. When no DSM is configured the nDSM and every object-height statistic is
reported as explicitly unavailable -- never as zero.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from rokko_geofusion.config import Config
from rokko_geofusion.crs import GridSpec, RoiGeometry
from rokko_geofusion.exceptions import AlignmentError
from rokko_geofusion.io.raster import (
    coverage_fraction,
    grid_from_raster,
    read_raster,
    write_grid_raster,
)
from rokko_geofusion.utils.metadata import build_metadata

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Derivatives
# ---------------------------------------------------------------------------
def _horn_gradients(elevation: np.ndarray, cell_size: float) -> tuple[np.ndarray, np.ndarray]:
    """Horn's 3x3 gradient estimate.

    Returns ``(dz_dx, dz_dy)`` where ``dz_dy`` is measured along *increasing
    row index*, i.e. southwards on a north-up grid. Edges use replicated
    values, so the outermost ring is an approximation.
    """
    if elevation.ndim != 2:
        raise ValueError(f"expected a 2-D elevation array, got {elevation.shape}")
    padded = np.pad(elevation.astype(np.float64), 1, mode="edge")
    a, b, c = padded[0:-2, 0:-2], padded[0:-2, 1:-1], padded[0:-2, 2:]
    d, f = padded[1:-1, 0:-2], padded[1:-1, 2:]
    g, h, i = padded[2:, 0:-2], padded[2:, 1:-1], padded[2:, 2:]

    dz_dx = ((c + 2 * f + i) - (a + 2 * d + g)) / (8 * cell_size)
    dz_dy = ((g + 2 * h + i) - (a + 2 * b + c)) / (8 * cell_size)
    return dz_dx, dz_dy


def slope(elevation: np.ndarray, cell_size: float, *, units: str = "degrees") -> np.ndarray:
    """Steepest slope per cell."""
    dz_dx, dz_dy = _horn_gradients(elevation, cell_size)
    rise_run = np.hypot(dz_dx, dz_dy)
    if units == "percent":
        result = rise_run * 100.0
    elif units == "degrees":
        result = np.degrees(np.arctan(rise_run))
    else:
        raise ValueError(f"unknown slope units {units!r}; use 'degrees' or 'percent'")
    return np.where(np.isfinite(elevation), result, np.nan).astype(np.float32)


def aspect(elevation: np.ndarray, cell_size: float) -> np.ndarray:
    """Down-slope compass direction in degrees (0 = north, 90 = east).

    Flat cells (zero gradient) are returned as NaN rather than an arbitrary
    direction, so they can be excluded from circular statistics.
    """
    dz_dx, dz_dy = _horn_gradients(elevation, cell_size)
    # dz_dy grows southwards; the northward derivative is its negative.
    dz_dy_north = -dz_dy
    # Down-slope direction vector is (-dz_dx, -dz_dy_north); converting that to
    # a compass bearing gives atan2(east, north) of the descent direction.
    bearing = np.degrees(np.arctan2(-dz_dx, -dz_dy_north))
    bearing = np.mod(bearing, 360.0)
    flat = (np.abs(dz_dx) < 1e-12) & (np.abs(dz_dy) < 1e-12)
    bearing = np.where(flat | ~np.isfinite(elevation), np.nan, bearing)
    return bearing.astype(np.float32)


def relief(elevation: np.ndarray, window_cells: int = 5) -> np.ndarray:
    """Local relief: max - min inside a square window."""
    from scipy.ndimage import maximum_filter, minimum_filter

    if window_cells < 3 or window_cells % 2 == 0:
        raise ValueError(f"window_cells must be an odd number >= 3, got {window_cells}")
    filled = np.where(np.isfinite(elevation), elevation, np.nan)
    # NaN-aware min/max: replace with +-inf so gaps do not poison the window.
    high = maximum_filter(np.nan_to_num(filled, nan=-np.inf), size=window_cells)
    low = minimum_filter(np.nan_to_num(filled, nan=np.inf), size=window_cells)
    result = high - low
    result[~np.isfinite(result)] = np.nan
    return np.where(np.isfinite(elevation), result, np.nan).astype(np.float32)


def hillshade(
    elevation: np.ndarray,
    cell_size: float,
    *,
    azimuth_deg: float = 315.0,
    altitude_deg: float = 45.0,
    z_factor: float = 1.0,
) -> np.ndarray:
    """Shaded relief in 0..255 (display and VLM input only, never a statistic)."""
    dz_dx, dz_dy = _horn_gradients(elevation, cell_size)
    dz_dx *= z_factor
    dz_dy *= z_factor
    slope_rad = np.arctan(np.hypot(dz_dx, dz_dy))
    aspect_rad = np.arctan2(dz_dy, -dz_dx)
    zenith = np.radians(90.0 - altitude_deg)
    azimuth = np.radians(360.0 - azimuth_deg + 90.0)
    shaded = (
        np.cos(zenith) * np.cos(slope_rad)
        + np.sin(zenith) * np.sin(slope_rad) * np.cos(azimuth - aspect_rad)
    )
    shaded = np.clip(shaded, 0.0, 1.0) * 255.0
    return np.where(np.isfinite(elevation), shaded, np.nan).astype(np.float32)


def normalised_dsm(
    dsm: np.ndarray,
    dem: np.ndarray,
    *,
    clip_min: float | None = 0.0,
    clip_max: float | None = None,
) -> np.ndarray:
    """nDSM = DSM - DEM, i.e. height above ground.

    Negative values (DSM below DEM) indicate a mismatch between the two
    products and are clipped to ``clip_min``; the caller should report how many
    cells were affected, because a large fraction means the inputs are not
    really co-registered (validation item V3).
    """
    if dsm.shape != dem.shape:
        raise AlignmentError(
            f"DSM shape {dsm.shape} does not match DEM shape {dem.shape}; "
            "both must be on the canonical ROI grid"
        )
    height = dsm.astype(np.float32) - dem.astype(np.float32)
    if clip_min is not None:
        height = np.where(height < clip_min, clip_min, height)
    if clip_max is not None:
        height = np.where(height > clip_max, clip_max, height)
    return np.where(np.isfinite(dsm) & np.isfinite(dem), height, np.nan).astype(np.float32)


def point_density(x: np.ndarray, y: np.ndarray, grid: GridSpec) -> np.ndarray:
    """Points per cell on ``grid`` (counts, not per-m2)."""
    min_x, min_y, max_x, max_y = grid.bounds
    cols = np.floor((x - min_x) / grid.resolution_m).astype(np.int64)
    rows = np.floor((max_y - y) / grid.resolution_m).astype(np.int64)
    valid = (rows >= 0) & (rows < grid.height) & (cols >= 0) & (cols < grid.width)
    counts = np.zeros(grid.shape, dtype=np.float32)
    np.add.at(counts, (rows[valid], cols[valid]), 1.0)
    return counts


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def _describe(values: np.ndarray, percentiles: tuple[float, ...] = (5, 25, 50, 75, 95)
              ) -> dict[str, Any] | None:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    summary: dict[str, Any] = {
        "count": int(finite.size),
        "min": float(finite.min()),
        "mean": float(finite.mean()),
        "max": float(finite.max()),
        "std": float(finite.std()),
    }
    for percentile, value in zip(percentiles, np.percentile(finite, percentiles), strict=True):
        summary[f"p{int(percentile)}"] = float(value)
    return summary


def _circular_mean_deg(values: np.ndarray) -> float | None:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    radians = np.radians(finite)
    return float(np.degrees(np.arctan2(np.sin(radians).mean(), np.cos(radians).mean())) % 360.0)


#: Eight compass sectors used for the aspect histogram.
ASPECT_SECTORS = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")


def aspect_distribution(values: np.ndarray) -> dict[str, float]:
    """Fraction of finite cells facing each of eight compass sectors."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {name: 0.0 for name in ASPECT_SECTORS}
    sector = np.floor(((finite + 22.5) % 360.0) / 45.0).astype(int)
    counts = np.bincount(sector, minlength=8)
    return {name: float(count / finite.size) for name, count in zip(ASPECT_SECTORS, counts,
                                                                    strict=True)}


@dataclass
class TerrainProducts:
    """Paths and arrays produced by :func:`build_terrain`."""

    grid: GridSpec
    paths: dict[str, Path] = field(default_factory=dict)
    statistics: dict[str, Any] = field(default_factory=dict)
    unavailable: dict[str, str] = field(default_factory=dict)


def terrain_statistics(
    *,
    grid: GridSpec,
    elevation: np.ndarray,
    slope_deg: np.ndarray,
    aspect_deg: np.ndarray,
    relief_m: np.ndarray | None = None,
    object_height: np.ndarray | None = None,
    density: np.ndarray | None = None,
    mask: np.ndarray | None = None,
    unavailable: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Summarise a (sub-)region. ``mask`` selects cells; ``None`` means all."""

    def select(array: np.ndarray | None) -> np.ndarray | None:
        if array is None:
            return None
        return array[mask] if mask is not None else array

    n_cells = int(mask.sum()) if mask is not None else int(elevation.size)
    cell_area = grid.resolution_m**2
    statistics: dict[str, Any] = {
        "crs": grid.crs,
        "resolution_m": grid.resolution_m,
        "cells": n_cells,
        "area_m2": float(n_cells * cell_area),
        "elevation": _describe(select(elevation)),
        "slope": _describe(select(slope_deg)),
        "aspect": {
            "circular_mean_deg": _circular_mean_deg(select(aspect_deg)),
            "sectors": aspect_distribution(select(aspect_deg)),
        },
        "coverage": coverage_fraction(select(elevation)),
    }
    if relief_m is not None:
        statistics["relief"] = _describe(select(relief_m))
    if object_height is not None:
        statistics["object_height"] = _describe(select(object_height))
    else:
        statistics["object_height"] = None
    if density is not None:
        selected = select(density)
        statistics["point_density"] = {
            "per_cell": _describe(selected),
            "per_m2": (float(np.nansum(selected) / (n_cells * cell_area))
                       if n_cells else None),
        }
    if unavailable:
        statistics["unavailable"] = dict(unavailable)
    return statistics


# ---------------------------------------------------------------------------
# Pipeline stage
# ---------------------------------------------------------------------------
def build_terrain(
    config: Config,
    roi: RoiGeometry,
    *,
    dem_path: Path,
    dsm_path: Path | None = None,
    pointcloud_path: Path | None = None,
    overwrite: bool = False,
) -> TerrainProducts:
    """Compute every terrain derivative and write them to ``outputs/raster``."""
    settings = config.terrain
    grid = grid_from_raster(dem_path)
    dem, _ = read_raster(dem_path, band=1)
    logger.info("terrain on a %dx%d grid at %.2f m (%s)",
                grid.width, grid.height, grid.resolution_m, grid.crs)

    products = TerrainProducts(grid=grid)
    out_dir = config.paths.raster

    def emit(name: str, array: np.ndarray, description: str, extra: dict[str, Any] | None = None):
        path = out_dir / f"{name}.tif"
        metadata = build_metadata(
            kind=f"terrain_{name}",
            source=f"derived:{dem_path.name}" + (f"+{dsm_path.name}" if dsm_path else ""),
            crs=grid.crs,
            resolution_m=grid.resolution_m,
            roi=roi.to_dict(),
            processing={"method": description, "grid": grid.to_dict(), **(extra or {})},
            config_fingerprint=config.fingerprint(),
            coverage=coverage_fraction(array),
        )
        write_grid_raster(path, array, grid, nodata=float("nan"),
                          compress=config.output.compress,
                          band_descriptions=[name], metadata=metadata)
        products.paths[name] = path

    slope_deg = slope(dem, grid.resolution_m, units=settings.slope_units)
    aspect_deg = aspect(dem, grid.resolution_m)
    relief_m = relief(dem, settings.relief_window_cells)
    shade = hillshade(dem, grid.resolution_m)

    emit("elevation", dem.astype(np.float32), "copy of the DEM on the analysis grid")
    emit("slope", slope_deg, f"Horn 3x3, {settings.slope_units}")
    emit("aspect", aspect_deg, "Horn 3x3, compass degrees (0=N, 90=E), flat=NaN")
    emit("relief", relief_m, f"max-min over {settings.relief_window_cells}x"
                             f"{settings.relief_window_cells} cells")
    emit("hillshade", shade, "azimuth 315 deg, altitude 45 deg (display only)")

    object_height: np.ndarray | None = None
    if dsm_path and Path(dsm_path).is_file():
        dsm_grid = grid_from_raster(dsm_path)
        if not dsm_grid.matches(grid):
            raise AlignmentError(
                "DSM and DEM are not on the same grid -- V2 would fail:\n"
                f"  DEM: {grid.to_dict()}\n  DSM: {dsm_grid.to_dict()}"
            )
        dsm, _ = read_raster(dsm_path, band=1)
        raw_difference = dsm.astype(np.float32) - dem.astype(np.float32)
        finite = np.isfinite(raw_difference)
        negative_fraction = (float((raw_difference[finite] < -0.5).mean())
                             if finite.any() else 0.0)
        if negative_fraction > 0.05:
            logger.warning(
                "V3 warning: %.1f%% of cells have DSM below DEM by more than 0.5 m; "
                "the two products may not be co-registered",
                100 * negative_fraction,
            )
        object_height = normalised_dsm(
            dsm, dem, clip_min=settings.ndsm_clip_min_m, clip_max=settings.ndsm_clip_max_m
        )
        emit("ndsm", object_height, "DSM - DEM, clipped",
             {"clip_min_m": settings.ndsm_clip_min_m,
              "clip_max_m": settings.ndsm_clip_max_m,
              "negative_fraction_before_clip": negative_fraction})
        products.statistics["ndsm_negative_fraction"] = negative_fraction
    else:
        reason = (
            "no DSM is configured (lidar.dsm.provider='none'), so height above "
            "ground cannot be measured. See README 'Data sources'."
        )
        products.unavailable["ndsm"] = reason
        products.unavailable["object_height"] = reason
        logger.warning("nDSM unavailable: %s", reason)

    density: np.ndarray | None = None
    if pointcloud_path and Path(pointcloud_path).is_file():
        from rokko_geofusion.lidar.pointcloud import read_point_cloud

        xyz, _ = read_point_cloud(pointcloud_path)
        density = point_density(xyz[:, 0], xyz[:, 1], grid)
        emit("point_density", density, "points per cell of the analysis grid")

    products.statistics.update(
        terrain_statistics(
            grid=grid,
            elevation=dem,
            slope_deg=slope_deg,
            aspect_deg=aspect_deg,
            relief_m=relief_m,
            object_height=object_height,
            density=density,
            unavailable=products.unavailable,
        )
    )
    products.statistics["roi"] = roi.to_dict()
    products.statistics["sources"] = {
        "dem": str(dem_path),
        "dsm": str(dsm_path) if dsm_path and Path(dsm_path).is_file() else None,
        "pointcloud": str(pointcloud_path) if density is not None else None,
    }
    return products
