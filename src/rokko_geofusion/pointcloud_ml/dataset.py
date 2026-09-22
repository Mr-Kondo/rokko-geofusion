"""Tiling, voxelisation and sampling for point cloud learning.

The brief's required flow is implemented literally:

    ROI -> spatial tile -> voxelisation -> sampling -> mini batch -> model

A whole ROI is never handed to the network. The fusion table is the source of
truth, because it already carries every modality on one grid, so the four
feature sets (XYZ / +RGB / +terrain / +semantics) are exactly the same points
with more columns -- which is what makes the comparison meaningful.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rokko_geofusion.config import Config
from rokko_geofusion.exceptions import UnsupportedError
from rokko_geofusion.lidar.pointcloud import voxel_downsample

logger = logging.getLogger(__name__)

#: Which columns each feature set needs from the fusion table.
FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "xyz": ("x", "y", "z"),
    "xyz_rgb": ("x", "y", "z", "red", "green", "blue"),
    "xyz_rgb_terrain": ("x", "y", "z", "red", "green", "blue",
                        "slope", "aspect", "relief", "object_height"),
    "xyz_rgb_terrain_semantic": ("x", "y", "z", "red", "green", "blue",
                                 "slope", "aspect", "relief", "object_height",
                                 "image_class", "image_confidence",
                                 "in_building", "on_road"),
}

#: Normalisation constants, documented rather than magic: slopes are degrees,
#: relief and object height are metres, colour is 8-bit.
_SLOPE_SCALE = 90.0
_RELIEF_SCALE = 20.0
_HEIGHT_SCALE = 30.0
_Z_SCALE = 50.0


def feature_dimension(feature_set: str, n_image_classes: int) -> int:
    """Number of channels a feature set produces after encoding."""
    if feature_set == "xyz":
        return 3
    if feature_set == "xyz_rgb":
        return 6
    if feature_set == "xyz_rgb_terrain":
        # xyz + rgb + slope + sin/cos(aspect) + relief + height + height_known
        return 6 + 6
    if feature_set == "xyz_rgb_terrain_semantic":
        return 6 + 6 + n_image_classes + 2
    raise UnsupportedError(f"unknown feature set {feature_set!r}")


@dataclass(frozen=True)
class TileSample:
    """One tile, ready for the network."""

    tile_id: int
    origin: tuple[float, float]
    features: dict[str, np.ndarray]     # feature set -> (num_points, channels)
    fused_class_histogram: np.ndarray   # for evaluation only, never an input
    n_points_available: int


def _encode(
    columns: dict[str, np.ndarray],
    feature_set: str,
    *,
    origin: tuple[float, float],
    tile_size_m: float,
    n_image_classes: int,
) -> np.ndarray:
    """Normalise raw columns into the network's input channels."""
    half = tile_size_m / 2.0
    x = (columns["x"] - (origin[0] + half)) / half
    y = (columns["y"] - (origin[1] + half)) / half
    z_raw = columns["z"]
    z = (z_raw - np.nanmin(z_raw)) / _Z_SCALE
    parts: list[np.ndarray] = [x, y, z]

    if feature_set == "xyz":
        return np.stack(parts, axis=1).astype(np.float32)

    parts += [columns["red"] / 255.0, columns["green"] / 255.0, columns["blue"] / 255.0]
    if feature_set == "xyz_rgb":
        return np.stack(parts, axis=1).astype(np.float32)

    slope = np.nan_to_num(columns["slope"], nan=0.0) / _SLOPE_SCALE
    aspect = np.radians(np.nan_to_num(columns["aspect"], nan=0.0))
    relief = np.nan_to_num(columns["relief"], nan=0.0) / _RELIEF_SCALE
    height = columns["object_height"]
    height_known = np.isfinite(height).astype(np.float32)
    height_value = np.nan_to_num(height, nan=0.0) / _HEIGHT_SCALE
    parts += [slope, np.sin(aspect), np.cos(aspect), relief, height_value, height_known]
    if feature_set == "xyz_rgb_terrain":
        return np.stack(parts, axis=1).astype(np.float32)

    image_class = columns["image_class"].astype(np.int64)
    one_hot = np.zeros((image_class.size, n_image_classes), np.float32)
    valid = (image_class >= 0) & (image_class < n_image_classes)
    one_hot[np.arange(image_class.size)[valid], image_class[valid]] = 1.0
    confidence = np.nan_to_num(columns["image_confidence"], nan=0.0)
    one_hot *= confidence[:, None]
    stacked = np.stack(parts, axis=1).astype(np.float32)
    extras = np.stack(
        [columns["in_building"].astype(np.float32), columns["on_road"].astype(np.float32)],
        axis=1,
    )
    return np.concatenate([stacked, one_hot, extras], axis=1).astype(np.float32)


class FusionTileDataset:
    """Spatial tiles sampled from the fusion table.

    Tiles are built once and held as index arrays; the point attributes stay in
    one contiguous array, so nothing is duplicated per feature set.
    """

    def __init__(
        self,
        config: Config,
        table_path: Path | str,
        *,
        feature_sets: Sequence[str] | None = None,
        num_points: int | None = None,
        tile_size_m: float | None = None,
        seed: int = 0,
    ) -> None:
        import pandas as pd

        settings = config.pointcloud_ml
        self.config = config
        self.feature_sets = list(feature_sets or settings.feature_sets)
        unknown = set(self.feature_sets) - set(FEATURE_SETS)
        if unknown:
            raise UnsupportedError(f"unknown feature set(s): {sorted(unknown)}")
        self.num_points = int(num_points or settings.num_points)
        self.tile_size_m = float(tile_size_m or settings.tile_size_m)
        self.n_image_classes = len(config.segmentation.classes)
        self.rng = np.random.default_rng(seed)

        needed = sorted({column for name in self.feature_sets for column in FEATURE_SETS[name]}
                        | {"fused_class"})
        table_path = Path(table_path)
        if not table_path.is_file():
            raise UnsupportedError(
                f"no fusion table at {table_path}; run scripts/fuse_modalities.py first"
            )
        frame = pd.read_parquet(table_path, columns=needed)
        logger.info("loaded %s fusion rows (%d columns) for point cloud ML",
                    f"{len(frame):,}", len(needed))

        self.columns = {name: frame[name].to_numpy() for name in needed}
        self.fused_class = self.columns["fused_class"].astype(np.int64)
        #: Voxel-downsampled point indices per tile. The voxel grid does not
        #: change between epochs, so computing it once instead of on every
        #: access is the difference between minutes and tens of minutes.
        self._voxel_cache: dict[int, np.ndarray] = {}
        self._build_tiles()

    # -- tiling --------------------------------------------------------------
    def _build_tiles(self) -> None:
        x, y = self.columns["x"], self.columns["y"]
        min_x, min_y = float(x.min()), float(y.min())
        tile_x = np.floor((x - min_x) / self.tile_size_m).astype(np.int64)
        tile_y = np.floor((y - min_y) / self.tile_size_m).astype(np.int64)
        n_x = int(tile_x.max()) + 1
        key = tile_y * n_x + tile_x

        order = np.argsort(key, kind="stable")
        sorted_key = key[order]
        boundaries = np.flatnonzero(np.diff(sorted_key)) + 1
        groups = np.split(order, boundaries)
        unique_keys = sorted_key[np.concatenate([[0], boundaries])] if len(order) else []

        minimum = max(32, self.num_points // 16)
        self.tiles: list[np.ndarray] = []
        self.tile_origins: list[tuple[float, float]] = []
        for tile_key, indices in zip(unique_keys, groups, strict=True):
            if indices.size < minimum:
                continue
            self.tiles.append(indices)
            self.tile_origins.append(
                (min_x + (tile_key % n_x) * self.tile_size_m,
                 min_y + (tile_key // n_x) * self.tile_size_m)
            )
        logger.info("built %d tiles of %.0f m (>= %d points each) from %s points",
                    len(self.tiles), self.tile_size_m, minimum,
                    f"{len(x):,}")
        if not self.tiles:
            raise UnsupportedError(
                "no tile has enough points; lower pointcloud_ml.num_points or "
                "raise pointcloud_ml.tile_size_m"
            )

    # -- sampling ------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.tiles)

    def _voxel_indices(self, index: int) -> np.ndarray:
        """Point indices of one tile after voxel downsampling (cached)."""
        cached = self._voxel_cache.get(index)
        if cached is not None:
            return cached
        indices = self.tiles[index]
        xyz = np.stack(
            [self.columns["x"][indices], self.columns["y"][indices],
             self.columns["z"][indices]], axis=1
        )
        _, kept = voxel_downsample(
            xyz, voxel_size_m=float(self.config.processing.voxel_size_m),
            attributes=indices[:, None],
        )
        selected = kept[:, 0] if kept is not None else indices
        self._voxel_cache[index] = selected
        return selected

    def sample(self, index: int, *, rng: np.random.Generator | None = None) -> TileSample:
        """Voxel-downsample a tile, then sample a fixed number of points."""
        rng = rng or self.rng
        indices = self.tiles[index]
        selected = self._voxel_indices(index)
        available = int(selected.size)

        if available >= self.num_points:
            chosen = rng.choice(selected, size=self.num_points, replace=False)
        else:
            chosen = rng.choice(selected, size=self.num_points, replace=True)

        columns = {name: values[chosen] for name, values in self.columns.items()}
        features = {
            name: _encode(columns, name, origin=self.tile_origins[index],
                          tile_size_m=self.tile_size_m,
                          n_image_classes=self.n_image_classes)
            for name in self.feature_sets
        }
        histogram = np.bincount(
            self.fused_class[indices], minlength=len(self.config.fusion.classes)
        ).astype(np.float32)
        return TileSample(
            tile_id=index,
            origin=self.tile_origins[index],
            features=features,
            fused_class_histogram=histogram / max(histogram.sum(), 1.0),
            n_points_available=available,
        )

    def batch(
        self,
        indices: Sequence[int],
        feature_set: str,
        *,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """``(B, num_points, channels)`` for one feature set."""
        return np.stack([self.sample(i, rng=rng).features[feature_set] for i in indices])

    def dominant_classes(self) -> np.ndarray:
        """Majority fused class per tile -- evaluation only, never an input."""
        return np.array(
            [int(np.argmax(np.bincount(self.fused_class[indices],
                                       minlength=len(self.config.fusion.classes))))
             for indices in self.tiles],
            dtype=np.int64,
        )

    def describe(self) -> dict[str, Any]:
        return {
            "source": "fusion_cells.parquet",
            "tiles": len(self.tiles),
            "tile_size_m": self.tile_size_m,
            "num_points": self.num_points,
            "voxel_size_m": self.config.processing.voxel_size_m,
            "feature_sets": {
                name: feature_dimension(name, self.n_image_classes)
                for name in self.feature_sets
            },
        }
