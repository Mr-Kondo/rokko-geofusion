"""Web-Mercator XYZ tile arithmetic.

GSI (and virtually every other tile server) publishes pyramids on the
Web-Mercator grid. We mosaic tiles in that CRS and reproject once, explicitly,
into the configured projected CRS -- never by ad-hoc scaling of pixel indices.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterator
from dataclasses import dataclass

logger = logging.getLogger(__name__)

#: Half the circumference of the Web-Mercator square, in metres.
MERCATOR_HALF_EXTENT = 20037508.342789244
TILE_PX = 256
#: Latitude beyond which Web-Mercator is undefined.
MAX_MERCATOR_LAT = 85.05112878

Bounds = tuple[float, float, float, float]


def tile_span_m(zoom: int) -> float:
    """Side length of one tile in Web-Mercator metres."""
    return 2.0 * MERCATOR_HALF_EXTENT / (2**zoom)


def tile_pixel_size_m(zoom: int) -> float:
    """Web-Mercator metres per pixel (NOT ground distance; see :func:`ground_resolution_m`)."""
    return tile_span_m(zoom) / TILE_PX


def ground_resolution_m(zoom: int, latitude: float) -> float:
    """True ground sample distance at ``latitude``, in metres per pixel."""
    return tile_pixel_size_m(zoom) * math.cos(math.radians(latitude))


def lonlat_to_tile(lon: float, lat: float, zoom: int) -> tuple[int, int]:
    """Tile indices containing a geographic coordinate."""
    lat = max(min(lat, MAX_MERCATOR_LAT), -MAX_MERCATOR_LAT)
    n = 2**zoom
    x = int((lon + 180.0) / 360.0 * n)
    lat_rad = math.radians(lat)
    y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return min(max(x, 0), n - 1), min(max(y, 0), n - 1)


def tile_to_lonlat(x: int, y: int, zoom: int) -> tuple[float, float]:
    """North-west corner of a tile, in geographic coordinates."""
    n = 2**zoom
    lon = x / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * y / n))))
    return lon, lat


def tile_bounds_mercator(x: int, y: int, zoom: int) -> Bounds:
    """Tile extent in Web-Mercator metres ``(min_x, min_y, max_x, max_y)``."""
    span = tile_span_m(zoom)
    min_x = -MERCATOR_HALF_EXTENT + x * span
    max_y = MERCATOR_HALF_EXTENT - y * span
    return (min_x, max_y - span, min_x + span, max_y)


@dataclass(frozen=True)
class TileRange:
    """An inclusive rectangle of tile indices at one zoom level."""

    zoom: int
    x_min: int
    y_min: int
    x_max: int
    y_max: int

    @property
    def n_x(self) -> int:
        return self.x_max - self.x_min + 1

    @property
    def n_y(self) -> int:
        return self.y_max - self.y_min + 1

    @property
    def count(self) -> int:
        return self.n_x * self.n_y

    @property
    def pixel_width(self) -> int:
        return self.n_x * TILE_PX

    @property
    def pixel_height(self) -> int:
        return self.n_y * TILE_PX

    def bounds_mercator(self) -> Bounds:
        """Extent of the whole mosaic in Web-Mercator metres."""
        left, _, _, top = tile_bounds_mercator(self.x_min, self.y_min, self.zoom)
        _, bottom, right, _ = tile_bounds_mercator(self.x_max, self.y_max, self.zoom)
        return (left, bottom, right, top)

    def transform(self):  # -> affine.Affine
        from rasterio.transform import from_origin

        left, _, _, top = self.bounds_mercator()
        pixel = tile_pixel_size_m(self.zoom)
        return from_origin(left, top, pixel, pixel)

    def __iter__(self) -> Iterator[tuple[int, int]]:
        for y in range(self.y_min, self.y_max + 1):
            for x in range(self.x_min, self.x_max + 1):
                yield x, y

    def offset_of(self, x: int, y: int) -> tuple[int, int]:
        """Pixel offset ``(row, col)`` of a tile's top-left corner in the mosaic."""
        return ((y - self.y_min) * TILE_PX, (x - self.x_min) * TILE_PX)

    def to_dict(self) -> dict[str, int]:
        return {
            "zoom": self.zoom,
            "x_min": self.x_min,
            "y_min": self.y_min,
            "x_max": self.x_max,
            "y_max": self.y_max,
            "count": self.count,
        }


def tile_range_for_geographic_bounds(bounds: Bounds, zoom: int) -> TileRange:
    """Smallest tile rectangle covering a geographic bbox."""
    min_lon, min_lat, max_lon, max_lat = bounds
    x_min, y_min = lonlat_to_tile(min_lon, max_lat, zoom)  # NW corner
    x_max, y_max = lonlat_to_tile(max_lon, min_lat, zoom)  # SE corner
    return TileRange(zoom=zoom, x_min=min(x_min, x_max), y_min=min(y_min, y_max),
                     x_max=max(x_min, x_max), y_max=max(y_min, y_max))


def tile_url(base_url: str, dataset: str, x: int, y: int, zoom: int, extension: str) -> str:
    """Build a ``{base}/{dataset}/{z}/{x}/{y}.{ext}`` tile URL."""
    return f"{base_url.rstrip('/')}/{dataset}/{zoom}/{x}/{y}.{extension.lstrip('.')}"
