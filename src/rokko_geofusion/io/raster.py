"""GeoTIFF read/write and explicit reprojection onto the canonical ROI grid.

Rasters produced here always carry a CRS and are always written on the grid
returned by :meth:`RoiGeometry.grid`, so DEM, DSM, imagery, segmentation and
fusion products are pixel-aligned by construction (V2).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.warp import reproject
from rasterio.windows import from_bounds as window_from_bounds

from rokko_geofusion.crs import Bounds, GridSpec
from rokko_geofusion.exceptions import AlignmentError, CrsError
from rokko_geofusion.utils.metadata import write_sidecar

logger = logging.getLogger(__name__)

RESAMPLING = {
    "nearest": Resampling.nearest,
    "bilinear": Resampling.bilinear,
    "cubic": Resampling.cubic,
    "average": Resampling.average,
    "mode": Resampling.mode,
    "max": Resampling.max,
    "min": Resampling.min,
}


def resampling_from_name(name: str) -> Resampling:
    try:
        return RESAMPLING[name]
    except KeyError as exc:
        raise ValueError(
            f"unknown resampling {name!r}; choose from {sorted(RESAMPLING)}"
        ) from exc


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------
def write_geotiff(
    path: Path | str,
    array: np.ndarray,
    *,
    transform: Affine,
    crs: str,
    nodata: float | int | None = None,
    compress: str = "deflate",
    band_descriptions: Sequence[str] | None = None,
    metadata: dict[str, Any] | None = None,
    tiled: bool = True,
) -> Path:
    """Write a 2-D or ``(bands, rows, cols)`` array as a GeoTIFF (+ sidecar)."""
    if array.ndim == 2:
        array = array[np.newaxis, ...]
    if array.ndim != 3:
        raise ValueError(f"expected a 2-D or 3-D array, got shape {array.shape}")
    if not crs:
        raise CrsError("refusing to write a raster without a CRS")

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count, height, width = array.shape

    profile: dict[str, Any] = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": count,
        "dtype": array.dtype,
        "crs": crs,
        "transform": transform,
        "compress": compress,
    }
    if nodata is not None:
        profile["nodata"] = nodata
    if tiled and height >= 256 and width >= 256:
        profile.update({"tiled": True, "blockxsize": 256, "blockysize": 256})

    with rasterio.open(out_path, "w", **profile) as dataset:
        dataset.write(array)
        if band_descriptions:
            for index, description in enumerate(band_descriptions[:count], start=1):
                dataset.set_band_description(index, description)
        if metadata:
            dataset.update_tags(**{k: str(v) for k, v in metadata.items()})

    logger.info("wrote raster %s  (%d band(s), %dx%d, %s)",
                out_path, count, width, height, array.dtype)
    if metadata is not None:
        write_sidecar(out_path, metadata)
    return out_path


def write_grid_raster(
    path: Path | str,
    array: np.ndarray,
    grid: GridSpec,
    *,
    nodata: float | int | None = None,
    compress: str = "deflate",
    band_descriptions: Sequence[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Write an array that is already on ``grid``."""
    expected = grid.shape
    actual = array.shape[-2:]
    if actual != expected:
        raise AlignmentError(
            f"array shape {actual} does not match grid shape {expected} "
            f"(grid: {grid.to_dict()})"
        )
    return write_geotiff(
        path,
        array,
        transform=grid.transform,
        crs=grid.crs,
        nodata=nodata,
        compress=compress,
        band_descriptions=band_descriptions,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------
def read_raster(
    path: Path | str,
    *,
    band: int | None = None,
    masked_to_nan: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Read a whole raster. Returns ``(array, profile)``.

    For float rasters, nodata is converted to NaN so downstream statistics do
    not silently include sentinel values such as -9999.
    """
    with rasterio.open(path) as dataset:
        array = dataset.read(band) if band else dataset.read()
        profile = dict(dataset.profile)
        profile["descriptions"] = dataset.descriptions
        profile["tags"] = dataset.tags()
        nodata = dataset.nodata

    if masked_to_nan and nodata is not None and np.issubdtype(array.dtype, np.floating):
        array = np.where(array == nodata, np.nan, array)
    return array, profile


def read_window(path: Path | str, bounds: Bounds) -> tuple[np.ndarray, Affine]:
    """Read only the part of a raster overlapping ``bounds`` (windowed IO)."""
    with rasterio.open(path) as dataset:
        window = window_from_bounds(*bounds, transform=dataset.transform)
        window = window.round_offsets().round_lengths()
        array = dataset.read(window=window, boundless=True, fill_value=dataset.nodata or 0)
        transform = dataset.window_transform(window)
    return array, transform


def grid_from_raster(path: Path | str) -> GridSpec:
    """Recover the :class:`GridSpec` a raster was written on."""
    with rasterio.open(path) as dataset:
        transform = dataset.transform
        if abs(transform.b) > 1e-12 or abs(transform.d) > 1e-12:
            raise AlignmentError(f"{path} is rotated; only north-up grids are supported")
        resolution_x, resolution_y = abs(transform.a), abs(transform.e)
        if abs(resolution_x - resolution_y) > 1e-6:
            raise AlignmentError(
                f"{path} has non-square pixels ({resolution_x} x {resolution_y})"
            )
        if dataset.crs is None:
            raise CrsError(f"{path} has no CRS; refusing to guess one")
        bounds = dataset.bounds
        return GridSpec(
            bounds=(bounds.left, bounds.bottom, bounds.right, bounds.top),
            resolution_m=resolution_x,
            crs=dataset.crs.to_string(),
        )


def assert_aligned(*paths: Path | str, tol: float = 1e-6) -> GridSpec:
    """Raise :class:`AlignmentError` unless every raster shares one grid (V2)."""
    if not paths:
        raise ValueError("assert_aligned needs at least one raster")
    grids = [(Path(p), grid_from_raster(p)) for p in paths]
    reference_path, reference = grids[0]
    for path, grid in grids[1:]:
        if not grid.matches(reference, tol=tol):
            raise AlignmentError(
                "rasters are not on a common grid:\n"
                f"  {reference_path}: {reference.to_dict()}\n"
                f"  {path}: {grid.to_dict()}"
            )
    return reference


# ---------------------------------------------------------------------------
# Reprojection
# ---------------------------------------------------------------------------
def reproject_array_to_grid(
    source: np.ndarray,
    *,
    src_transform: Affine,
    src_crs: str,
    grid: GridSpec,
    resampling: str | Resampling = "bilinear",
    src_nodata: float | None = None,
    dst_nodata: float | None = None,
    dtype: str | np.dtype | None = None,
) -> np.ndarray:
    """Reproject an in-memory array onto the canonical ROI grid.

    This is the only place where pixels change CRS; no other module may
    approximate a reprojection by scaling indices.
    """
    if source.ndim == 2:
        source = source[np.newaxis, ...]
    count = source.shape[0]
    out_dtype = np.dtype(dtype) if dtype is not None else source.dtype
    if np.issubdtype(out_dtype, np.floating) and dst_nodata is None:
        dst_nodata = float("nan")

    destination = np.full((count, grid.height, grid.width),
                          dst_nodata if dst_nodata is not None else 0, dtype=out_dtype)
    method = resampling if isinstance(resampling, Resampling) else resampling_from_name(resampling)

    reproject(
        source=source,
        destination=destination,
        src_transform=src_transform,
        src_crs=src_crs,
        src_nodata=src_nodata,
        dst_transform=grid.transform,
        dst_crs=grid.crs,
        dst_nodata=dst_nodata,
        resampling=method,
        num_threads=2,
    )
    logger.debug(
        "reprojected %s %s -> %s on %s grid (%s)",
        source.shape, src_crs, grid.crs, grid.shape, method.name,
    )
    return destination[0] if count == 1 else destination


def coverage_fraction(array: np.ndarray) -> float:
    """Fraction of finite (non-NaN) cells -- reported in metadata."""
    finite = np.isfinite(array)
    return float(finite.sum() / finite.size) if finite.size else 0.0
