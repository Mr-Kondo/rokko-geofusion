"""Sample raster values at arbitrary projected coordinates.

This implements the chain the project brief requires:

    point XYZ -> XY coordinate -> orthophoto pixel -> RGB

Sampling is windowed: only the part of the raster covering the requested
points is read, so colourising a tile of a large cloud never loads the whole
orthophoto. The caller is responsible for passing coordinates that are already
in the raster's CRS -- the sampler refuses to guess (it compares the CRS it was
given with the file's own and raises on a mismatch).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import numpy as np
import rasterio
from rasterio.windows import Window

from rokko_geofusion.exceptions import CrsError

logger = logging.getLogger(__name__)

SampleMethod = Literal["nearest", "bilinear"]


def _window_for_points(
    dataset: rasterio.DatasetReader, x: np.ndarray, y: np.ndarray, pad: int
) -> Window | None:
    rows, cols = rasterio.transform.rowcol(
        dataset.transform, x, y, op=lambda v: np.floor(v).astype(np.int64)
    )
    rows = np.asarray(rows)
    cols = np.asarray(cols)
    inside = (rows >= 0) & (rows < dataset.height) & (cols >= 0) & (cols < dataset.width)
    if not inside.any():
        return None
    row_min = max(int(rows[inside].min()) - pad, 0)
    row_max = min(int(rows[inside].max()) + pad + 1, dataset.height)
    col_min = max(int(cols[inside].min()) - pad, 0)
    col_max = min(int(cols[inside].max()) + pad + 1, dataset.width)
    return Window(col_min, row_min, col_max - col_min, row_max - row_min)


def sample_raster(
    path: Path | str,
    x: np.ndarray,
    y: np.ndarray,
    *,
    expected_crs: str | None = None,
    bands: list[int] | None = None,
    method: SampleMethod = "nearest",
    fill_value: float = np.nan,
) -> np.ndarray:
    """Sample ``path`` at projected coordinates ``(x, y)``.

    Returns ``(n_bands, n_points)``. Points outside the raster, or on nodata,
    receive ``fill_value``.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.shape != y.shape:
        raise ValueError(f"x and y must have the same length, got {x.shape} and {y.shape}")

    with rasterio.open(path) as dataset:
        if expected_crs is not None:
            if dataset.crs is None:
                raise CrsError(f"{path} has no CRS; refusing to sample it blindly")
            if not dataset.crs.equals(rasterio.crs.CRS.from_user_input(expected_crs)):
                raise CrsError(
                    f"CRS mismatch while sampling {path}: points are in "
                    f"{expected_crs}, raster is in {dataset.crs.to_string()}. "
                    "Reproject explicitly before sampling."
                )
        band_list = bands or list(range(1, dataset.count + 1))
        n_bands = len(band_list)
        output = np.full((n_bands, x.size), fill_value, dtype=np.float64)

        pad = 1 if method == "bilinear" else 0
        window = _window_for_points(dataset, x, y, pad)
        if window is None:
            logger.debug("no sample points fall inside %s", path)
            return output

        block = dataset.read(band_list, window=window).astype(np.float64)
        nodata = dataset.nodata
        if nodata is not None and not np.isnan(nodata):
            block = np.where(block == nodata, np.nan, block)
        transform = dataset.window_transform(window)
        height, width = block.shape[1], block.shape[2]

    # Continuous pixel coordinates within the window, cell-centre referenced.
    # The affine inverse is written out so this works with any `affine`
    # version (the `transform * (x, y)` operator is deprecated).
    a, b, c, d, e, f = transform.a, transform.b, transform.c, transform.d, transform.e, transform.f
    determinant = a * e - b * d
    if determinant == 0:
        raise ValueError(f"degenerate raster transform for {path}: {transform}")
    dx = x - c
    dy = y - f
    col_f = (e * dx - b * dy) / determinant - 0.5
    row_f = (-d * dx + a * dy) / determinant - 0.5

    if method == "nearest":
        cols = np.rint(col_f).astype(np.int64)
        rows = np.rint(row_f).astype(np.int64)
        valid = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
        if valid.any():
            output[:, valid] = block[:, rows[valid], cols[valid]]
        return output

    # Bilinear: four neighbours, weighted, skipping NaN contributions.
    col0 = np.floor(col_f).astype(np.int64)
    row0 = np.floor(row_f).astype(np.int64)
    fx = col_f - col0
    fy = row_f - row0
    accumulator = np.zeros((n_bands, x.size), dtype=np.float64)
    weights = np.zeros(x.size, dtype=np.float64)
    for d_row, d_col, weight in (
        (0, 0, (1 - fx) * (1 - fy)),
        (0, 1, fx * (1 - fy)),
        (1, 0, (1 - fx) * fy),
        (1, 1, fx * fy),
    ):
        rows = row0 + d_row
        cols = col0 + d_col
        valid = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width) & (weight > 0)
        if not valid.any():
            continue
        values = block[:, rows[valid], cols[valid]]
        finite = np.isfinite(values).all(axis=0)
        idx = np.where(valid)[0][finite]
        if idx.size == 0:
            continue
        accumulator[:, idx] += values[:, finite] * weight[idx]
        weights[idx] += weight[idx]

    good = weights > 0
    output[:, good] = accumulator[:, good] / weights[good]
    return output


def sample_rgb(
    path: Path | str,
    x: np.ndarray,
    y: np.ndarray,
    *,
    expected_crs: str | None = None,
    method: SampleMethod = "nearest",
) -> np.ndarray:
    """Sample an RGB raster; returns ``(n_points, 3)`` uint8 (black where absent)."""
    samples = sample_raster(
        path, x, y, expected_crs=expected_crs, bands=[1, 2, 3], method=method, fill_value=0.0
    )
    rgb = np.nan_to_num(samples, nan=0.0)
    return np.clip(np.rint(rgb), 0, 255).astype(np.uint8).T
