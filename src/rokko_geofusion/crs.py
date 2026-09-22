"""Coordinate reference systems and ROI geometry.

Every spatial product in this project is produced on a grid that is derived
here, from the configured projected CRS and a snapped origin. Two rasters
generated for the same ROI at the same resolution are therefore pixel-aligned
by construction (validation item V2), and re-running an ROI reproduces exactly
the same grid (V6).

The three CRS roles are kept distinct at all times:

``geographic``
    how the user specifies the ROI and how web maps display it (EPSG:4326).
``tile``
    the CRS the XYZ tile pyramids are published on (EPSG:3857).
``projected``
    the metric CRS all analysis happens in (EPSG:6673 for Hyogo by default).
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
from pyproj import CRS, Transformer
from pyproj.exceptions import CRSError as PyprojCrsError

from rokko_geofusion.config import CrsConfig, RoiConfig
from rokko_geofusion.exceptions import CrsError

logger = logging.getLogger(__name__)

Bounds = tuple[float, float, float, float]  # (min_x, min_y, max_x, max_y)


@lru_cache(maxsize=64)
def _parse_crs(spec: str) -> CRS:
    try:
        return CRS.from_user_input(spec)
    except (PyprojCrsError, ValueError, TypeError) as exc:
        raise CrsError(f"cannot interpret CRS specification {spec!r}: {exc}") from exc


@lru_cache(maxsize=256)
def _transformer(src: str, dst: str) -> Transformer:
    """Cached, always-xy transformer between two CRS specifications."""
    return Transformer.from_crs(_parse_crs(src), _parse_crs(dst), always_xy=True)


class CrsManager:
    """Explicit transformations between the configured CRS roles."""

    def __init__(self, config: CrsConfig) -> None:
        self.config = config
        self.geographic = config.geographic
        self.projected = config.projected
        self.tile = config.tile
        self.output = config.effective_output
        for role, spec in (
            ("geographic", self.geographic),
            ("projected", self.projected),
            ("tile", self.tile),
            ("output", self.output),
        ):
            crs = _parse_crs(spec)
            if role == "projected" and not crs.is_projected:
                raise CrsError(
                    f"crs.projected must be a projected (metric) CRS; {spec!r} is not. "
                    "Analysis distances, slopes and areas would be meaningless."
                )
            if role == "geographic" and not crs.is_geographic:
                raise CrsError(f"crs.geographic must be a geographic CRS; {spec!r} is not.")
        logger.debug(
            "CRS roles: geographic=%s projected=%s tile=%s output=%s",
            self.geographic, self.projected, self.tile, self.output,
        )

    # -- transformation ------------------------------------------------------
    def transformer(self, src: str, dst: str) -> Transformer:
        return _transformer(src, dst)

    def transform(self, x: Any, y: Any, src: str, dst: str) -> tuple[Any, Any]:
        """Transform scalars or arrays from ``src`` to ``dst``.

        Input order is always (x, y) == (easting/longitude, northing/latitude),
        regardless of the axis order declared by the CRS.
        """
        if src == dst:
            return x, y
        return self.transformer(src, dst).transform(x, y)

    def transform_points(self, points: np.ndarray, src: str, dst: str) -> np.ndarray:
        """Transform an ``(N, 2)`` or ``(N, 3)`` array; Z is passed through."""
        array = np.asarray(points)
        if array.ndim != 2 or array.shape[1] not in (2, 3):
            raise CrsError(f"expected an (N, 2) or (N, 3) array, got shape {array.shape}")
        if src == dst:
            return array.copy()
        x, y = self.transform(array[:, 0], array[:, 1], src, dst)
        out = array.astype(np.float64, copy=True)
        out[:, 0] = x
        out[:, 1] = y
        return out

    def transform_bounds(self, bounds: Bounds, src: str, dst: str, *, densify: int = 51) -> Bounds:
        """Transform a bounding box, densifying edges so the result contains it."""
        if src == dst:
            return bounds
        return self.transformer(src, dst).transform_bounds(*bounds, densify_pts=densify)

    # -- convenience ---------------------------------------------------------
    def geographic_to_projected(self, lon: Any, lat: Any) -> tuple[Any, Any]:
        return self.transform(lon, lat, self.geographic, self.projected)

    def projected_to_geographic(self, x: Any, y: Any) -> tuple[Any, Any]:
        return self.transform(x, y, self.projected, self.geographic)

    def describe(self) -> dict[str, str]:
        return {
            "geographic": self.geographic,
            "projected": self.projected,
            "tile": self.tile,
            "output": self.output,
        }


# ---------------------------------------------------------------------------
# ROI
# ---------------------------------------------------------------------------
def snap_bounds(bounds: Bounds, resolution: float) -> Bounds:
    """Expand ``bounds`` outwards to whole multiples of ``resolution``.

    Snapping to the CRS origin is what makes independently produced rasters
    (DEM today, DSM tomorrow) share one grid.
    """
    if resolution <= 0:
        raise ValueError("resolution must be positive")
    min_x, min_y, max_x, max_y = bounds
    return (
        math.floor(min_x / resolution) * resolution,
        math.floor(min_y / resolution) * resolution,
        math.ceil(max_x / resolution) * resolution,
        math.ceil(max_y / resolution) * resolution,
    )


@dataclass(frozen=True)
class GridSpec:
    """A raster grid in the projected CRS: snapped bounds + resolution."""

    bounds: Bounds
    resolution_m: float
    crs: str

    @property
    def width(self) -> int:
        return int(round((self.bounds[2] - self.bounds[0]) / self.resolution_m))

    @property
    def height(self) -> int:
        return int(round((self.bounds[3] - self.bounds[1]) / self.resolution_m))

    @property
    def shape(self) -> tuple[int, int]:
        """``(rows, cols)`` -- numpy order."""
        return (self.height, self.width)

    @property
    def transform(self):  # -> affine.Affine
        """North-up affine transform (rasterio convention)."""
        from rasterio.transform import from_origin

        return from_origin(self.bounds[0], self.bounds[3], self.resolution_m, self.resolution_m)

    def cell_centers(self) -> tuple[np.ndarray, np.ndarray]:
        """1-D arrays of cell-centre x (west->east) and y (north->south)."""
        half = self.resolution_m / 2.0
        xs = self.bounds[0] + half + np.arange(self.width) * self.resolution_m
        ys = self.bounds[3] - half - np.arange(self.height) * self.resolution_m
        return xs, ys

    def to_dict(self) -> dict[str, Any]:
        return {
            "crs": self.crs,
            "bounds": list(self.bounds),
            "resolution_m": self.resolution_m,
            "width": self.width,
            "height": self.height,
        }

    def matches(self, other: GridSpec, *, tol: float = 1e-6) -> bool:
        return (
            self.crs == other.crs
            and abs(self.resolution_m - other.resolution_m) < tol
            and all(abs(a - b) < tol for a, b in zip(self.bounds, other.bounds, strict=True))
        )


@dataclass(frozen=True)
class RoiGeometry:
    """The ROI expressed in every CRS the pipeline needs."""

    key: str
    name: str
    crs: CrsManager
    bounds_projected: Bounds
    bounds_geographic: Bounds
    center_geographic: tuple[float, float]
    center_projected: tuple[float, float]

    # -- construction --------------------------------------------------------
    @classmethod
    def from_config(cls, roi: RoiConfig, crs: CrsManager) -> RoiGeometry:
        if roi.center is not None:
            cx, cy = crs.geographic_to_projected(roi.center.lon, roi.center.lat)
            radius = float(roi.radius_m)
            bounds_projected: Bounds = (cx - radius, cy - radius, cx + radius, cy + radius)
            bounds_geographic = crs.transform_bounds(
                bounds_projected, crs.projected, crs.geographic
            )
            center_geographic = (roi.center.lon, roi.center.lat)
        else:
            assert roi.bbox is not None
            bounds_geographic = (roi.bbox[0], roi.bbox[1], roi.bbox[2], roi.bbox[3])
            bounds_projected = crs.transform_bounds(
                bounds_geographic, crs.geographic, crs.projected
            )
            center_geographic = (
                (bounds_geographic[0] + bounds_geographic[2]) / 2.0,
                (bounds_geographic[1] + bounds_geographic[3]) / 2.0,
            )
            cx, cy = crs.geographic_to_projected(*center_geographic)

        return cls(
            key=roi.key,
            name=roi.name,
            crs=crs,
            bounds_projected=tuple(float(v) for v in bounds_projected),  # type: ignore[arg-type]
            bounds_geographic=tuple(float(v) for v in bounds_geographic),  # type: ignore[arg-type]
            center_geographic=(float(center_geographic[0]), float(center_geographic[1])),
            center_projected=(float(cx), float(cy)),
        )

    # -- derived -------------------------------------------------------------
    @property
    def width_m(self) -> float:
        return self.bounds_projected[2] - self.bounds_projected[0]

    @property
    def height_m(self) -> float:
        return self.bounds_projected[3] - self.bounds_projected[1]

    @property
    def area_m2(self) -> float:
        return self.width_m * self.height_m

    @property
    def polygon(self):  # -> shapely.geometry.Polygon
        from shapely.geometry import box

        return box(*self.bounds_projected)

    @property
    def polygon_geographic(self):
        from shapely.geometry import box

        return box(*self.bounds_geographic)

    def bounds_in(self, crs_spec: str) -> Bounds:
        """ROI bounds in an arbitrary CRS (e.g. the tile CRS)."""
        return self.crs.transform_bounds(self.bounds_projected, self.crs.projected, crs_spec)

    def buffered(self, metres: float) -> RoiGeometry:
        """A copy grown by ``metres`` on every side (edge-effect padding)."""
        min_x, min_y, max_x, max_y = self.bounds_projected
        grown: Bounds = (min_x - metres, min_y - metres, max_x + metres, max_y + metres)
        return RoiGeometry(
            key=self.key,
            name=self.name,
            crs=self.crs,
            bounds_projected=grown,
            bounds_geographic=self.crs.transform_bounds(
                grown, self.crs.projected, self.crs.geographic
            ),
            center_geographic=self.center_geographic,
            center_projected=self.center_projected,
        )

    def grid(self, resolution_m: float) -> GridSpec:
        """The canonical analysis grid for this ROI at ``resolution_m``."""
        return GridSpec(
            bounds=snap_bounds(self.bounds_projected, resolution_m),
            resolution_m=float(resolution_m),
            crs=self.crs.projected,
        )

    def contains_projected(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        min_x, min_y, max_x, max_y = self.bounds_projected
        return (x >= min_x) & (x <= max_x) & (y >= min_y) & (y <= max_y)

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "name": self.name,
            "crs": self.crs.describe(),
            "center_geographic": {"lon": self.center_geographic[0],
                                  "lat": self.center_geographic[1]},
            "center_projected": {"x": self.center_projected[0], "y": self.center_projected[1]},
            "bounds_geographic": list(self.bounds_geographic),
            "bounds_projected": list(self.bounds_projected),
            "width_m": self.width_m,
            "height_m": self.height_m,
            "area_m2": self.area_m2,
        }

    def __str__(self) -> str:
        return (
            f"ROI {self.key}: {self.width_m:.0f}x{self.height_m:.0f} m in {self.crs.projected}, "
            f"geographic bbox {tuple(round(v, 5) for v in self.bounds_geographic)}"
        )


def roi_from_config(config: Any) -> RoiGeometry:
    """Convenience: build the ROI geometry straight from a :class:`Config`."""
    return RoiGeometry.from_config(config.roi, CrsManager(config.crs))


def densify_bounds(bounds: Bounds, steps: int = 20) -> Iterable[tuple[float, float]]:
    """Yield points along the outline of ``bounds`` (for CRS round-trip tests)."""
    min_x, min_y, max_x, max_y = bounds
    for index in range(steps + 1):
        t = index / steps
        yield (min_x + t * (max_x - min_x), min_y)
        yield (min_x + t * (max_x - min_x), max_y)
        yield (min_x, min_y + t * (max_y - min_y))
        yield (max_x, min_y + t * (max_y - min_y))
