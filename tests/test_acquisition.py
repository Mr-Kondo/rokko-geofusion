"""Parsing and assembly of downloaded elevation / imagery / GIS payloads.

These tests are offline: they exercise the parsing and geometry code against
synthetic payloads shaped like the real ones. Live-network checks live in
``tests/test_network.py`` behind the ``network`` marker.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from rokko_geofusion.exceptions import UnsupportedError
from rokko_geofusion.gis.osm import (
    LAYER_QUERIES,
    _elements_to_records,
    build_query,
    estimate_building_heights,
)
from rokko_geofusion.imagery.orthophoto import decode_tile
from rokko_geofusion.io.tiles import TILE_PX
from rokko_geofusion.lidar.elevation import parse_gsi_elevation_tile, read_xyz_grid


# --- GSI elevation tiles ----------------------------------------------------
def _fake_elevation_tile(value: float = 100.0, nodata_cells: int = 0) -> bytes:
    rows = []
    remaining = nodata_cells
    for row in range(TILE_PX):
        cells = []
        for col in range(TILE_PX):
            if remaining > 0:
                cells.append("e")
                remaining -= 1
            else:
                cells.append(f"{value + row * 0.01 + col * 0.001:.3f}")
        rows.append(",".join(cells))
    return ("\n".join(rows)).encode("utf-8")


def test_parse_elevation_tile_shape_and_values():
    array = parse_gsi_elevation_tile(_fake_elevation_tile(250.0))
    assert array.shape == (TILE_PX, TILE_PX)
    assert array.dtype == np.float32
    assert array[0, 0] == pytest.approx(250.0)
    assert np.isfinite(array).all()


def test_parse_elevation_tile_maps_nodata_to_nan():
    array = parse_gsi_elevation_tile(_fake_elevation_tile(nodata_cells=10))
    assert np.isnan(array[0, :10]).all()
    assert np.isfinite(array[0, 10])
    assert int(np.isnan(array).sum()) == 10


def test_parse_elevation_tile_rejects_the_wrong_shape():
    with pytest.raises(UnsupportedError, match="format may have changed"):
        parse_gsi_elevation_tile(b"1,2,3\n4,5,6")


def test_parse_elevation_tile_honours_a_custom_nodata_token():
    payload = _fake_elevation_tile().replace(b"100.000", b"NA", 1)
    array = parse_gsi_elevation_tile(payload, nodata_token="NA")
    assert np.isnan(array[0, 0])


# --- imagery tiles ----------------------------------------------------------
def _fake_image_tile(size: int = TILE_PX, colour=(10, 120, 200)) -> bytes:
    import io

    from PIL import Image

    image = Image.new("RGB", (size, size), colour)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_decode_tile_returns_channels_first_rgb():
    array = decode_tile(_fake_image_tile())
    assert array.shape == (3, TILE_PX, TILE_PX)
    assert array.dtype == np.uint8
    assert tuple(array[:, 0, 0]) == (10, 120, 200)


def test_decode_tile_converts_greyscale():
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.new("L", (TILE_PX, TILE_PX), 77).save(buffer, format="PNG")
    array = decode_tile(buffer.getvalue())
    assert array.shape == (3, TILE_PX, TILE_PX)
    assert tuple(array[:, 5, 5]) == (77, 77, 77)


def test_decode_tile_rejects_unexpected_size():
    with pytest.raises(UnsupportedError):
        decode_tile(_fake_image_tile(size=64))


# --- regular-grid XYZ -------------------------------------------------------
def test_read_xyz_grid_reconstructs_the_grid(tmp_path):
    spacing = 2.0
    lines = []
    for row in range(5):
        for col in range(4):
            x = 1000.0 + col * spacing
            y = 2000.0 + row * spacing
            lines.append(f"{x},{y},{row * 10 + col}")
    path = tmp_path / "grid.xyz"
    path.write_text("\n".join(lines), encoding="utf-8")

    array, transform, info = read_xyz_grid(path, crs="EPSG:6673")
    assert info["grid_spacing_m"] == pytest.approx(spacing)
    assert info["rows"] == 20
    assert array.shape == (5, 4)
    # Row 0 of the array is the northernmost line, i.e. the last input row.
    assert array[0, 0] == pytest.approx(40.0)
    assert array[-1, 0] == pytest.approx(0.0)
    assert transform.a == pytest.approx(spacing)
    assert transform.e == pytest.approx(-spacing)


def test_read_xyz_grid_handles_whitespace_and_chunks(tmp_path):
    path = tmp_path / "grid.txt"
    path.write_text("# comment\n0 0 1\n1 0 2\n0 1 3\n1 1 4\n", encoding="utf-8")
    array, _, info = read_xyz_grid(path, crs="EPSG:6673", chunk_rows=2)
    assert array.shape == (2, 2)
    assert info["rows"] == 4
    assert array[1, 0] == pytest.approx(1.0)


def test_read_xyz_grid_rejects_a_single_point(tmp_path):
    path = tmp_path / "one.xyz"
    path.write_text("0,0,5\n", encoding="utf-8")
    with pytest.raises(UnsupportedError, match="regular grid spacing"):
        read_xyz_grid(path, crs="EPSG:6673")


# --- Overpass ---------------------------------------------------------------
def test_build_query_uses_overpass_bbox_order():
    query = build_query("building", (135.22, 34.71, 135.25, 34.74), 180)
    assert "(34.71,135.22,34.74,135.25)" in query  # S,W,N,E
    assert query.startswith("[out:json][timeout:180];")
    assert query.rstrip().endswith("out geom;")


def test_build_query_rejects_unknown_layers():
    with pytest.raises(UnsupportedError):
        build_query("volcanoes", (0, 0, 1, 1), 60)


def test_every_configured_layer_has_a_query():
    from rokko_geofusion.config import OsmConfig

    for layer in OsmConfig().layers:
        assert layer in LAYER_QUERIES


def test_closed_way_with_area_tag_becomes_a_polygon():
    element = {
        "type": "way",
        "id": 1,
        "tags": {"building": "yes", "name": "A"},
        "geometry": [
            {"lon": 0.0, "lat": 0.0},
            {"lon": 0.001, "lat": 0.0},
            {"lon": 0.001, "lat": 0.001},
            {"lon": 0.0, "lat": 0.0},
        ],
    }
    records = _elements_to_records([element])
    assert len(records) == 1
    assert records[0]["geometry"].geom_type == "Polygon"
    assert records[0]["osm_id"] == "way/1"
    assert records[0]["name"] == "A"
    assert json.loads(records[0]["tags"])["building"] == "yes"


def test_open_way_becomes_a_linestring():
    element = {
        "type": "way",
        "id": 2,
        "tags": {"highway": "residential"},
        "geometry": [{"lon": 0.0, "lat": 0.0}, {"lon": 0.001, "lat": 0.001}],
    }
    records = _elements_to_records([element])
    assert records[0]["geometry"].geom_type == "LineString"
    assert records[0]["highway"] == "residential"


def test_closed_way_without_area_tag_stays_a_line():
    ring = [
        {"lon": 0.0, "lat": 0.0},
        {"lon": 0.001, "lat": 0.0},
        {"lon": 0.001, "lat": 0.001},
        {"lon": 0.0, "lat": 0.0},
    ]
    element = {"type": "way", "id": 3, "tags": {"highway": "service"}, "geometry": ring}
    assert _elements_to_records([element])[0]["geometry"].geom_type == "LineString"


def test_relation_outer_members_are_polygonised():
    element = {
        "type": "relation",
        "id": 9,
        "tags": {"building": "yes"},
        "members": [
            {
                "role": "outer",
                "geometry": [
                    {"lon": 0.0, "lat": 0.0},
                    {"lon": 0.002, "lat": 0.0},
                    {"lon": 0.002, "lat": 0.002},
                    {"lon": 0.0, "lat": 0.002},
                    {"lon": 0.0, "lat": 0.0},
                ],
            },
            {"role": "inner", "geometry": [{"lon": 0.0005, "lat": 0.0005}]},
        ],
    }
    records = _elements_to_records([element])
    assert len(records) == 1
    assert records[0]["geometry"].geom_type in {"Polygon", "MultiPolygon"}


def test_degenerate_geometry_is_skipped():
    assert _elements_to_records([{"type": "way", "id": 4, "geometry": []}]) == []
    assert _elements_to_records([{"type": "node", "id": 5}]) == []


def test_building_heights_prefer_explicit_tags_over_levels(config):
    import geopandas as gpd
    from shapely.geometry import Point

    frame = gpd.GeoDataFrame(
        {
            "height": ["12.5", None, None, "bogus"],
            "building_levels": ["3", "4", None, "2"],
            "geometry": [Point(0, 0)] * 4,
        },
        crs=config.crs.projected,
    )
    result = estimate_building_heights(frame, config)
    assert result.loc[0, "height_m"] == pytest.approx(12.5)
    assert result.loc[0, "height_source"] == "height"
    assert result.loc[1, "height_m"] == pytest.approx(4 * config.gis.metres_per_level)
    assert result.loc[1, "height_source"] == "building:levels"
    assert np.isnan(result.loc[2, "height_m"])
    # "bogus" has no numeric part, so the levels tag is used instead.
    assert result.loc[3, "height_m"] == pytest.approx(2 * config.gis.metres_per_level)


def test_building_heights_on_an_empty_frame(config):
    import geopandas as gpd

    frame = gpd.GeoDataFrame({"geometry": []}, crs=config.crs.projected)
    result = estimate_building_heights(frame, config)
    assert "height_m" in result.columns
