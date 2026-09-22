"""Web-Mercator tile arithmetic."""

from __future__ import annotations

import math

import pytest

from rokko_geofusion.io.tiles import (
    MERCATOR_HALF_EXTENT,
    TILE_PX,
    ground_resolution_m,
    lonlat_to_tile,
    tile_bounds_mercator,
    tile_range_for_geographic_bounds,
    tile_span_m,
    tile_to_lonlat,
    tile_url,
)

KOBE = (135.2348, 34.7284)


def test_known_tile_index_for_the_default_roi():
    """Verified against the live GSI endpoint during development."""
    assert lonlat_to_tile(*KOBE, 15) == (28693, 13009)
    assert lonlat_to_tile(*KOBE, 18) == (229546, 104075)


def test_tile_contains_the_point_that_produced_it():
    for zoom in (10, 14, 15, 18):
        x, y = lonlat_to_tile(*KOBE, zoom)
        west, north = tile_to_lonlat(x, y, zoom)
        east, south = tile_to_lonlat(x + 1, y + 1, zoom)
        assert west <= KOBE[0] < east
        assert south < KOBE[1] <= north


def test_tile_bounds_tile_the_plane_without_gaps():
    zoom = 12
    a = tile_bounds_mercator(100, 200, zoom)
    right = tile_bounds_mercator(101, 200, zoom)
    below = tile_bounds_mercator(100, 201, zoom)
    assert a[2] == pytest.approx(right[0])
    assert a[1] == pytest.approx(below[3])
    assert a[2] - a[0] == pytest.approx(tile_span_m(zoom))


def test_mercator_grid_origin():
    top_left = tile_bounds_mercator(0, 0, 0)
    assert top_left[0] == pytest.approx(-MERCATOR_HALF_EXTENT)
    assert top_left[3] == pytest.approx(MERCATOR_HALF_EXTENT)


def test_ground_resolution_matches_the_documented_gsd():
    # z18 at 35N is the ~0.5 m imagery the config targets.
    assert ground_resolution_m(18, 34.7284) == pytest.approx(0.49, abs=0.01)
    assert ground_resolution_m(15, 34.7284) == pytest.approx(3.93, abs=0.02)
    # Halving the zoom step doubles the pixel size.
    assert ground_resolution_m(17, 0.0) == pytest.approx(2 * ground_resolution_m(18, 0.0))


def test_resolution_shrinks_with_latitude():
    assert ground_resolution_m(18, 60.0) < ground_resolution_m(18, 0.0)


def test_tile_range_covers_the_bbox():
    bounds = (135.22379, 34.7193, 135.24582, 34.73749)
    tile_range = tile_range_for_geographic_bounds(bounds, 18)
    assert tile_range.count == tile_range.n_x * tile_range.n_y
    assert tile_range.pixel_width == tile_range.n_x * TILE_PX
    mosaic = tile_range.bounds_mercator()
    # Every bbox corner must fall inside the mosaic extent.
    import pyproj

    to_mercator = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    for lon in (bounds[0], bounds[2]):
        for lat in (bounds[1], bounds[3]):
            x, y = to_mercator.transform(lon, lat)
            assert mosaic[0] <= x <= mosaic[2]
            assert mosaic[1] <= y <= mosaic[3]


def test_tile_range_iteration_is_row_major_and_complete():
    tile_range = tile_range_for_geographic_bounds((135.22, 34.71, 135.25, 34.74), 15)
    seen = list(tile_range)
    assert len(seen) == tile_range.count
    assert len(set(seen)) == tile_range.count
    assert seen[0] == (tile_range.x_min, tile_range.y_min)
    assert seen[-1] == (tile_range.x_max, tile_range.y_max)


def test_offsets_place_tiles_side_by_side():
    tile_range = tile_range_for_geographic_bounds((135.22, 34.71, 135.25, 34.74), 15)
    row0, col0 = tile_range.offset_of(tile_range.x_min, tile_range.y_min)
    assert (row0, col0) == (0, 0)
    _, col1 = tile_range.offset_of(tile_range.x_min + 1, tile_range.y_min)
    assert col1 == TILE_PX


def test_transform_matches_the_mosaic_bounds():
    tile_range = tile_range_for_geographic_bounds((135.22, 34.71, 135.25, 34.74), 15)
    transform = tile_range.transform()
    left, bottom, right, top = tile_range.bounds_mercator()
    assert transform.c == pytest.approx(left)
    assert transform.f == pytest.approx(top)
    x, y = transform * (tile_range.pixel_width, tile_range.pixel_height)
    assert x == pytest.approx(right)
    assert y == pytest.approx(bottom)


def test_tile_url_shape():
    url = tile_url("https://example.org/xyz/", "dem5a", 1, 2, 15, "txt")
    assert url == "https://example.org/xyz/dem5a/15/1/2.txt"


def test_latitude_is_clamped_to_the_mercator_limit():
    x, y = lonlat_to_tile(0.0, 89.9, 5)
    assert 0 <= y < 2**5
    assert not math.isnan(y)
