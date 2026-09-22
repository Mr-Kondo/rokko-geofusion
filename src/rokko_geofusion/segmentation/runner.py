"""Tiled semantic segmentation of the orthophoto.

The image is processed in overlapping tiles and the per-class probabilities are
blended with a tapered window, so tile seams do not show up as straight lines in
the class raster. Tile size and batch size come from the detected
:class:`~rokko_geofusion.environment.ResourceProfile`, and an out-of-memory
failure walks down the documented ladder (batch -> tile -> CPU) instead of
crashing.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import from_bounds as window_from_bounds

from rokko_geofusion.config import Config
from rokko_geofusion.crs import Bounds, GridSpec, RoiGeometry
from rokko_geofusion.environment import ResourceProfile
from rokko_geofusion.exceptions import ResourceError, UnsupportedError
from rokko_geofusion.io.raster import write_grid_raster
from rokko_geofusion.segmentation.model import SegmentationModel, load_segmenter
from rokko_geofusion.utils.metadata import build_metadata

logger = logging.getLogger(__name__)


@dataclass
class SegmentationProduct:
    class_path: Path
    confidence_path: Path
    grid: GridSpec
    class_names: list[str]
    statistics: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


def _tile_origins(size: int, tile_px: int, overlap_px: int) -> list[int]:
    """Start indices covering ``size`` with ``tile_px`` windows.

    The final tile is pulled back so every tile has the same shape (batching
    needs that); edge tiles therefore overlap their neighbour a little more.
    """
    if tile_px >= size:
        return [0]
    step = max(1, tile_px - overlap_px)
    origins = list(range(0, size - tile_px + 1, step))
    if origins[-1] != size - tile_px:
        origins.append(size - tile_px)
    return origins


def _blend_window(tile_px: int, taper_px: int, floor: float = 0.05) -> np.ndarray:
    """Separable cosine taper with a floor, so no pixel ends up unweighted."""
    ramp = np.ones(tile_px, dtype=np.float32)
    taper = int(min(taper_px, tile_px // 2))
    if taper > 0:
        edge = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, taper + 2)[1:-1]))
        ramp[:taper] = edge
        ramp[-taper:] = edge[::-1]
    # The floor applies to the finished 2-D window: clipping the 1-D ramp
    # instead would leave corner pixels at floor**2.
    return np.clip(np.outer(ramp, ramp), floor, 1.0)


def _iter_tile_batches(
    imagery_path: Path,
    grid: GridSpec,
    tile_px: int,
    overlap_px: int,
    batch_size: int,
) -> Iterator[tuple[np.ndarray, list[tuple[int, int]]]]:
    """Yield ``(batch (B,3,tile,tile) uint8, [(row, col), ...])`` over the grid."""
    rows = _tile_origins(grid.height, tile_px, overlap_px)
    cols = _tile_origins(grid.width, tile_px, overlap_px)
    positions = [(row, col) for row in rows for col in cols]

    with rasterio.open(imagery_path) as dataset:
        batch: list[np.ndarray] = []
        coordinates: list[tuple[int, int]] = []
        for row, col in positions:
            bounds: Bounds = (
                grid.bounds[0] + col * grid.resolution_m,
                grid.bounds[3] - (row + tile_px) * grid.resolution_m,
                grid.bounds[0] + (col + tile_px) * grid.resolution_m,
                grid.bounds[3] - row * grid.resolution_m,
            )
            window = window_from_bounds(*bounds, transform=dataset.transform)
            tile = dataset.read(
                [1, 2, 3],
                window=window,
                out_shape=(3, tile_px, tile_px),
                resampling=Resampling.average,
                boundless=True,
                fill_value=0,
            ).astype(np.uint8)
            batch.append(tile)
            coordinates.append((row, col))
            if len(batch) == batch_size:
                yield np.stack(batch), coordinates
                batch, coordinates = [], []
        if batch:
            yield np.stack(batch), coordinates


def _run_pass(
    model: SegmentationModel,
    imagery_path: Path,
    grid: GridSpec,
    *,
    tile_px: int,
    overlap_px: int,
    batch_size: int,
    n_classes: int,
) -> tuple[np.ndarray, np.ndarray]:
    """One full tiled inference pass; returns ``(probabilities, weights)``."""
    accumulator = np.zeros((n_classes, grid.height, grid.width), dtype=np.float32)
    weights = np.zeros((grid.height, grid.width), dtype=np.float32)
    window = _blend_window(tile_px, overlap_px)

    n_tiles = len(_tile_origins(grid.height, tile_px, overlap_px)) * len(
        _tile_origins(grid.width, tile_px, overlap_px)
    )
    done = 0
    for batch, coordinates in _iter_tile_batches(
        imagery_path, grid, tile_px, overlap_px, batch_size
    ):
        probabilities = model.predict_proba(batch)
        for (row, col), tile_probabilities in zip(coordinates, probabilities, strict=True):
            height = min(tile_px, grid.height - row)
            width = min(tile_px, grid.width - col)
            weight = window[:height, :width]
            accumulator[:, row:row + height, col:col + width] += (
                tile_probabilities[:, :height, :width] * weight
            )
            weights[row:row + height, col:col + width] += weight
        done += len(coordinates)
        if done % (batch_size * 10) < batch_size or done == n_tiles:
            logger.info("segmentation: %d/%d tiles", done, n_tiles)
    return accumulator, weights


def segment_imagery(
    config: Config,
    roi: RoiGeometry,
    imagery_path: Path,
    *,
    profile: ResourceProfile,
    model: SegmentationModel | None = None,
    overwrite: bool = False,
) -> SegmentationProduct:
    """Segment the orthophoto and write class + confidence rasters."""
    settings = config.segmentation
    if not settings.enabled:
        raise UnsupportedError("segmentation.enabled is false")
    if not Path(imagery_path).is_file():
        raise UnsupportedError(
            f"no orthophoto at {imagery_path}; run scripts/download_imagery.py first"
        )

    grid = roi.grid(settings.output_resolution_m)
    class_path = config.paths.raster / "segmentation_class.tif"
    confidence_path = config.paths.raster / "segmentation_confidence.tif"
    class_names = list(settings.classes)

    if class_path.is_file() and confidence_path.is_file() and not overwrite:
        from rokko_geofusion.utils.metadata import read_sidecar

        metadata = read_sidecar(class_path)
        if metadata and metadata.get("config_fingerprint") == config.fingerprint():
            logger.info("reusing existing segmentation (%s)", class_path)
            return SegmentationProduct(
                class_path=class_path,
                confidence_path=confidence_path,
                grid=grid,
                class_names=class_names,
                # Same shape as a fresh run: a reused product must be
                # indistinguishable downstream.
                statistics={
                    "class_fractions": metadata.get("class_fractions", {}),
                    "mean_confidence": metadata.get("mean_confidence", {}),
                },
                metadata=metadata,
            )

    model = model or load_segmenter(config, profile.device)
    tile_px = min(profile.seg_tile_px or settings.tile_px, settings.tile_px)
    batch_size = max(1, profile.seg_batch_size or settings.batch_size)
    overlap_px = min(settings.overlap_px, max(0, tile_px // 4))
    current = profile

    logger.info(
        "segmenting %s on a %dx%d grid at %.2f m (model GSD), tiles %dpx, batch %d, device %s",
        Path(imagery_path).name, grid.width, grid.height, grid.resolution_m,
        tile_px, batch_size, current.device,
    )

    attempts = 0
    while True:
        try:
            accumulator, weights = _run_pass(
                model, Path(imagery_path), grid,
                tile_px=tile_px, overlap_px=overlap_px,
                batch_size=batch_size, n_classes=len(class_names),
            )
            break
        except ResourceError as exc:
            attempts += 1
            if not settings.auto_downscale or attempts > 5:
                raise
            previous = (current.device, tile_px, batch_size)
            current = current.downscale(min_tile_px=settings.min_tile_px)
            tile_px = min(tile_px, current.seg_tile_px)
            batch_size = max(1, current.seg_batch_size)
            overlap_px = min(overlap_px, max(0, tile_px // 4))
            if (current.device, tile_px, batch_size) == previous:
                raise
            logger.warning(
                "%s -- retrying with device=%s tile=%dpx batch=%d (attempt %d)",
                exc, current.device, tile_px, batch_size, attempts,
            )
            if current.device != previous[0]:
                model = load_segmenter(config, current.device)

    weights = np.maximum(weights, 1e-6)
    probabilities = accumulator / weights[np.newaxis, :, :]
    class_index = probabilities.argmax(axis=0).astype(np.uint8)
    confidence = probabilities.max(axis=0).astype(np.float32)

    counts = np.bincount(class_index.ravel(), minlength=len(class_names))
    fractions = {name: float(count / class_index.size)
                 for name, count in zip(class_names, counts, strict=True)}
    mean_confidence = {
        name: (float(confidence[class_index == index].mean())
               if counts[index] else None)
        for index, name in enumerate(class_names)
    }

    description = model.describe()
    metadata = build_metadata(
        kind="segmentation",
        source=f"segmentation:{description.get('model_id', settings.provider)}",
        crs=grid.crs,
        resolution_m=grid.resolution_m,
        roi=roi.to_dict(),
        acquisition={"imagery": str(imagery_path)},
        processing={
            "grid": grid.to_dict(),
            "tile_px": tile_px,
            "overlap_px": overlap_px,
            "batch_size": batch_size,
            "device": current.device,
            "downscale_steps": current.downscale_steps,
            "blending": "separable cosine taper, floor 0.05",
        },
        config_fingerprint=config.fingerprint(),
        model=description,
        class_names=class_names,
        class_fractions=fractions,
        mean_confidence=mean_confidence,
        notes=description.get("domain_note"),
    )

    write_grid_raster(class_path, class_index, grid, nodata=None,
                      compress=config.output.compress,
                      band_descriptions=["class_index"], metadata=metadata)
    write_grid_raster(confidence_path, confidence, grid, nodata=float("nan"),
                      compress=config.output.compress,
                      band_descriptions=["confidence"], metadata=metadata)

    logger.info("class fractions: %s",
                ", ".join(f"{name}={value:.1%}" for name, value in fractions.items()))
    return SegmentationProduct(
        class_path=class_path,
        confidence_path=confidence_path,
        grid=grid,
        class_names=class_names,
        statistics={"class_fractions": fractions, "mean_confidence": mean_confidence},
        metadata=metadata,
    )
