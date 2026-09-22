"""Derive, colourise, store and downsample the point cloud.

The cloud is built tile by tile: for each tile of the ROI we generate the cell
centres of the target grid, sample Z from the surface raster and RGB from the
orthophoto, and stream the result straight into the output file. Nothing
larger than one tile is ever held in memory, so the same code path works for a
200 m test ROI and for a city-scale run.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rokko_geofusion.config import Config
from rokko_geofusion.crs import Bounds, GridSpec, RoiGeometry
from rokko_geofusion.exceptions import UnsupportedError
from rokko_geofusion.imagery.sampler import sample_raster, sample_rgb
from rokko_geofusion.io.raster import grid_from_raster
from rokko_geofusion.utils.metadata import build_metadata, write_sidecar

logger = logging.getLogger(__name__)

#: Column order used by every in-memory point array in this project.
XYZ_COLUMNS = ("x", "y", "z")
RGB_COLUMNS = ("red", "green", "blue")


@dataclass(frozen=True)
class PointCloudProduct:
    path: Path
    format: str
    n_points: int
    bounds: Bounds
    z_range: tuple[float, float]
    surface: str
    has_rgb: bool
    is_true_lidar: bool
    metadata: dict[str, Any]


# ---------------------------------------------------------------------------
# Tiling
# ---------------------------------------------------------------------------
def iter_tiles(grid: GridSpec, tile_size_m: float) -> Iterator[Bounds]:
    """Yield tile bounds covering ``grid``, snapped to its own resolution."""
    step = max(tile_size_m, grid.resolution_m)
    min_x, min_y, max_x, max_y = grid.bounds
    y = min_y
    while y < max_y:
        x = min_x
        top = min(y + step, max_y)
        while x < max_x:
            right = min(x + step, max_x)
            yield (x, y, right, top)
            x = right
        y = top


def cell_centers_in(grid: GridSpec, bounds: Bounds) -> tuple[np.ndarray, np.ndarray]:
    """Cell-centre coordinates of ``grid`` that fall inside ``bounds``."""
    xs, ys = grid.cell_centers()
    min_x, min_y, max_x, max_y = bounds
    x_sel = xs[(xs >= min_x) & (xs < max_x)]
    y_sel = ys[(ys >= min_y) & (ys < max_y)]
    if x_sel.size == 0 or y_sel.size == 0:
        return np.empty(0), np.empty(0)
    mesh_x, mesh_y = np.meshgrid(x_sel, y_sel)
    return mesh_x.ravel(), mesh_y.ravel()


# ---------------------------------------------------------------------------
# Downsampling
# ---------------------------------------------------------------------------
def voxel_downsample(
    xyz: np.ndarray, *, voxel_size_m: float, attributes: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray | None]:
    """Keep one point per voxel (the first one encountered).

    Deterministic: points are bucketed by integer voxel index and the lowest
    original index wins, so re-running yields an identical cloud (V6).
    """
    if voxel_size_m <= 0:
        raise ValueError("voxel_size_m must be positive")
    if xyz.size == 0:
        return xyz, attributes

    keys = np.floor(xyz / voxel_size_m).astype(np.int64)
    _, first_index = np.unique(keys, axis=0, return_index=True)
    first_index.sort()
    return xyz[first_index], (attributes[first_index] if attributes is not None else None)


def subsample_for_display(
    xyz: np.ndarray,
    attributes: np.ndarray | None,
    max_points: int,
    *,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Deterministically thin a cloud for browser display.

    Analysis always uses the full cloud on disk; only what is shipped to a
    widget goes through here.
    """
    n = xyz.shape[0]
    if n <= max_points:
        return xyz, attributes
    rng = np.random.default_rng(seed)
    index = np.sort(rng.choice(n, size=max_points, replace=False))
    return xyz[index], (attributes[index] if attributes is not None else None)


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------
class PointWriter:
    """Streaming writer shared by the supported output formats."""

    def __init__(self, path: Path, fmt: str, crs: str, with_rgb: bool) -> None:
        self.path = path
        self.format = fmt
        self.crs = crs
        self.with_rgb = with_rgb
        self.count = 0
        self._chunks_xyz: list[np.ndarray] = []
        self._chunks_rgb: list[np.ndarray] = []
        self._chunks_class: list[np.ndarray] = []
        self._las_writer = None
        if fmt in {"laz", "las"}:
            self._open_las()

    def _open_las(self) -> None:
        try:
            import laspy
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise UnsupportedError(
                "writing LAS/LAZ needs laspy: pip install -e '.[pointcloud]', "
                "or set output.pointcloud_format to 'parquet'"
            ) from exc

        header = laspy.LasHeader(point_format=2 if self.with_rgb else 0, version="1.4")
        # 1 mm precision is far finer than any of our sources; offsets keep the
        # 32-bit integer storage well within range.
        header.scales = np.array([0.001, 0.001, 0.001])
        header.offsets = np.array([0.0, 0.0, 0.0])
        try:
            from pyproj import CRS as PyprojCRS

            header.add_crs(PyprojCRS.from_user_input(self.crs))
        except Exception as exc:  # noqa: BLE001 - CRS VLR is best-effort
            logger.warning("could not embed CRS %s in the LAS header: %s", self.crs, exc)
        self._las_writer = laspy.open(self.path, mode="w", header=header)

    def write(
        self,
        xyz: np.ndarray,
        rgb: np.ndarray | None = None,
        classification: np.ndarray | None = None,
    ) -> None:
        """Append points. ``classification`` becomes the LAS classification byte."""
        if xyz.size == 0:
            return
        if self._las_writer is not None:
            import laspy

            record = laspy.ScaleAwarePointRecord.zeros(
                xyz.shape[0],
                point_format=self._las_writer.header.point_format,
                scales=self._las_writer.header.scales,
                offsets=self._las_writer.header.offsets,
            )
            record.x = xyz[:, 0]
            record.y = xyz[:, 1]
            record.z = xyz[:, 2]
            if self.with_rgb and rgb is not None:
                # LAS stores colour as 16-bit channels.
                record.red = rgb[:, 0].astype(np.uint16) * 257
                record.green = rgb[:, 1].astype(np.uint16) * 257
                record.blue = rgb[:, 2].astype(np.uint16) * 257
            if classification is not None:
                # Point formats 0-5 store the class in 5 bits (0-31).
                record.classification = np.clip(classification, 0, 31).astype(np.uint8)
            self._las_writer.write_points(record)
        else:
            self._chunks_xyz.append(xyz.astype(np.float32))
            if rgb is not None:
                self._chunks_rgb.append(rgb)
            if classification is not None:
                self._chunks_class.append(np.asarray(classification, dtype=np.uint8))
        self.count += xyz.shape[0]

    def close(self) -> None:
        if self._las_writer is not None:
            self._las_writer.close()
            return
        xyz = (np.concatenate(self._chunks_xyz) if self._chunks_xyz
               else np.empty((0, 3), np.float32))
        rgb = (np.concatenate(self._chunks_rgb) if self._chunks_rgb
               else np.empty((0, 3), np.uint8))
        classification = (np.concatenate(self._chunks_class) if self._chunks_class
                          else np.empty(0, np.uint8))
        if self.format == "npz":
            np.savez_compressed(self.path, xyz=xyz, rgb=rgb,
                                classification=classification, crs=np.array([self.crs]))
        elif self.format == "parquet":
            import pandas as pd

            frame = pd.DataFrame(
                {"x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2]}
                | ({"red": rgb[:, 0], "green": rgb[:, 1], "blue": rgb[:, 2]}
                   if self.with_rgb and rgb.size else {})
                | ({"classification": classification} if classification.size else {})
            )
            frame.to_parquet(self.path, index=False)
        else:  # pragma: no cover - pydantic restricts the literal
            raise UnsupportedError(f"unknown point cloud format {self.format!r}")


def _output_suffix(fmt: str) -> str:
    return {"laz": ".laz", "las": ".las", "parquet": ".parquet", "npz": ".npz"}[fmt]


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------
def read_point_cloud(
    path: Path | str, *, max_points: int | None = None, seed: int = 0
) -> tuple[np.ndarray, np.ndarray | None]:
    """Read a cloud written by :func:`build_point_cloud`.

    ``max_points`` thins the result deterministically -- use it for display,
    never for analysis.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".laz", ".las"}:
        import laspy

        with laspy.open(path) as reader:
            las = reader.read()
        xyz = np.column_stack([las.x, las.y, las.z]).astype(np.float64)
        rgb = None
        if "red" in las.point_format.dimension_names:
            rgb = np.column_stack(
                [las.red // 257, las.green // 257, las.blue // 257]
            ).astype(np.uint8)
    elif suffix == ".npz":
        payload = np.load(path)
        xyz = payload["xyz"].astype(np.float64)
        rgb = payload["rgb"] if "rgb" in payload and payload["rgb"].size else None
    elif suffix == ".parquet":
        import pandas as pd

        frame = pd.read_parquet(path)
        xyz = frame[list(XYZ_COLUMNS)].to_numpy(dtype=np.float64)
        rgb = (frame[list(RGB_COLUMNS)].to_numpy(dtype=np.uint8)
               if set(RGB_COLUMNS).issubset(frame.columns) else None)
    else:
        raise UnsupportedError(f"unsupported point cloud file type: {path.name}")

    if max_points is not None:
        xyz, rgb = subsample_for_display(xyz, rgb, max_points, seed=seed)
    return xyz, rgb


# ---------------------------------------------------------------------------
# Building
# ---------------------------------------------------------------------------
def read_point_cloud_classification(path: Path | str) -> np.ndarray | None:
    """Read the per-point class of a fused cloud, when it carries one."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".laz", ".las"}:
        import laspy

        with laspy.open(path) as reader:
            las = reader.read()
        return np.asarray(las.classification, dtype=np.uint8)
    if suffix == ".npz":
        payload = np.load(path)
        values = payload.get("classification")
        return values if values is not None and values.size else None
    if suffix == ".parquet":
        import pandas as pd

        frame = pd.read_parquet(path)
        if "classification" not in frame.columns:
            return None
        return frame["classification"].to_numpy(dtype=np.uint8)
    raise UnsupportedError(f"unsupported point cloud file type: {path.name}")


def choose_surface(
    config: Config, dem_path: Path | None, dsm_path: Path | None
) -> tuple[Path, str]:
    """Pick the raster that provides Z, honouring ``pointcloud.surface``."""
    preference = config.pointcloud.surface
    if preference == "dsm":
        if not (dsm_path and Path(dsm_path).is_file()):
            raise UnsupportedError(
                "pointcloud.surface='dsm' but no DSM exists. Configure lidar.dsm "
                "(see README 'Data sources') or use surface='auto'."
            )
        return Path(dsm_path), "dsm"
    if preference == "dem":
        if not (dem_path and Path(dem_path).is_file()):
            raise UnsupportedError("pointcloud.surface='dem' but no DEM exists")
        return Path(dem_path), "dem"
    if dsm_path and Path(dsm_path).is_file():
        return Path(dsm_path), "dsm"
    if dem_path and Path(dem_path).is_file():
        logger.warning(
            "no DSM available; the cloud describes the GROUND surface (DEM). "
            "Object heights are not measurable from it."
        )
        return Path(dem_path), "dem"
    raise UnsupportedError("neither a DSM nor a DEM is available for this ROI")


def build_point_cloud(
    config: Config,
    roi: RoiGeometry,
    *,
    dem_path: Path | None = None,
    dsm_path: Path | None = None,
    imagery_path: Path | None = None,
    overwrite: bool = False,
) -> PointCloudProduct:
    """Generate ``<outputs>/pointcloud/cloud.<ext>`` for the ROI."""
    settings = config.pointcloud
    surface_path, surface_kind = choose_surface(config, dem_path, dsm_path)
    surface_grid = grid_from_raster(surface_path)

    resolution = settings.resolution_m or surface_grid.resolution_m
    if resolution < surface_grid.resolution_m * 0.99:
        logger.warning(
            "pointcloud.resolution_m=%.2f m is finer than the %s grid (%.2f m); "
            "the extra points are interpolated, not measured",
            resolution, surface_kind.upper(), surface_grid.resolution_m,
        )
    grid = roi.grid(resolution)
    expected_points = grid.width * grid.height
    if expected_points > settings.max_points:
        raise UnsupportedError(
            f"{expected_points:,} points would be generated at {resolution} m, above "
            f"pointcloud.max_points ({settings.max_points:,}). Coarsen "
            "pointcloud.resolution_m or shrink the ROI."
        )

    colorize = bool(settings.colorize and imagery_path and Path(imagery_path).is_file())
    if settings.colorize and not colorize:
        logger.warning("colourisation requested but no orthophoto is available")

    fmt = config.output.pointcloud_format
    out_path = config.paths.pointcloud / f"cloud{_output_suffix(fmt)}"
    if out_path.is_file() and not overwrite:
        from rokko_geofusion.utils.metadata import read_sidecar

        metadata = read_sidecar(out_path)
        if metadata and metadata.get("config_fingerprint") == config.fingerprint():
            logger.info("reusing existing point cloud %s", out_path)
            return PointCloudProduct(
                path=out_path,
                format=fmt,
                n_points=int(metadata.get("n_points", 0)),
                bounds=tuple(metadata.get("bounds", grid.bounds)),  # type: ignore[arg-type]
                z_range=tuple(metadata.get("z_range", (float("nan"),) * 2)),  # type: ignore
                surface=str(metadata.get("surface", surface_kind)),
                has_rgb=bool(metadata.get("has_rgb", False)),
                is_true_lidar=bool(metadata.get("is_true_lidar", False)),
                metadata=metadata,
            )

    logger.info(
        "building point cloud: surface=%s grid=%dx%d @ %.2f m (<= %s points), rgb=%s, format=%s",
        surface_kind, grid.width, grid.height, resolution, f"{expected_points:,}",
        colorize, fmt,
    )

    writer = PointWriter(out_path, fmt, grid.crs, with_rgb=colorize)
    z_min, z_max = float("inf"), float("-inf")
    tiles = list(iter_tiles(grid, settings.chunk_size_m))
    dropped = 0

    try:
        for index, bounds in enumerate(tiles, start=1):
            x, y = cell_centers_in(grid, bounds)
            if x.size == 0:
                continue
            z = sample_raster(
                surface_path, x, y, expected_crs=grid.crs, bands=[1], method="bilinear"
            )[0]
            valid = np.isfinite(z)
            dropped += int((~valid).sum())
            if not valid.any():
                continue
            x, y, z = x[valid], y[valid], z[valid]

            rgb = None
            if colorize:
                rgb = sample_rgb(
                    imagery_path, x, y,
                    expected_crs=grid.crs, method=settings.rgb_sampling,
                )
            writer.write(np.column_stack([x, y, z]), rgb)
            z_min = min(z_min, float(z.min()))
            z_max = max(z_max, float(z.max()))
            if index % 10 == 0 or index == len(tiles):
                logger.debug("tile %d/%d: %s points so far", index, len(tiles),
                             f"{writer.count:,}")
    finally:
        writer.close()

    if writer.count == 0:
        raise UnsupportedError(
            f"no valid points generated for ROI {roi.key}; the {surface_kind.upper()} "
            "may be empty over this area"
        )

    metadata = build_metadata(
        kind="pointcloud",
        source=f"derived:{surface_kind}+orthophoto" if colorize else f"derived:{surface_kind}",
        crs=grid.crs,
        resolution_m=resolution,
        roi=roi.to_dict(),
        acquisition={
            "surface_raster": str(surface_path),
            "imagery_raster": str(imagery_path) if colorize else None,
        },
        processing={
            "grid": grid.to_dict(),
            "chunk_size_m": settings.chunk_size_m,
            "z_sampling": "bilinear",
            "rgb_sampling": settings.rgb_sampling if colorize else None,
            "cells_without_elevation": dropped,
        },
        config_fingerprint=config.fingerprint(),
        n_points=writer.count,
        bounds=list(grid.bounds),
        z_range=[z_min, z_max],
        surface=surface_kind,
        has_rgb=colorize,
        is_true_lidar=False,
        notes=(
            "Gridded cloud derived from a raster elevation product; one point per "
            "grid cell. NOT raw LiDAR returns."
        ),
    )
    write_sidecar(out_path, metadata)

    logger.info("point cloud -> %s  (%s points, z %.2f..%.2f m, rgb=%s)",
                out_path, f"{writer.count:,}", z_min, z_max, colorize)
    return PointCloudProduct(
        path=out_path,
        format=fmt,
        n_points=writer.count,
        bounds=grid.bounds,
        z_range=(z_min, z_max),
        surface=surface_kind,
        has_rgb=colorize,
        is_true_lidar=False,
        metadata=metadata,
    )
