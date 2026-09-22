"""Live checks against the real upstream services.

These are canaries for CLAUDE.md rule 7: if GSI or Overpass change their URL
scheme or payload format, these tests fail loudly instead of the pipeline
silently reporting "no data for this ROI".

Run the offline suite with ``pytest -m "not network"``.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from rokko_geofusion.config import HttpConfig, OsmConfig
from rokko_geofusion.gis.osm import build_query
from rokko_geofusion.imagery.orthophoto import decode_tile
from rokko_geofusion.io.http import HttpClient
from rokko_geofusion.io.tiles import TILE_PX, lonlat_to_tile, tile_url
from rokko_geofusion.lidar.elevation import parse_gsi_elevation_tile

pytestmark = pytest.mark.network

KOBE_LON, KOBE_LAT = 135.2348, 34.7284
GSI_BASE = "https://cyberjapandata.gsi.go.jp/xyz"


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    cache = tmp_path_factory.mktemp("net-cache")
    return HttpClient(HttpConfig(timeout_s=30.0, max_retries=2, backoff_s=1.0), cache_dir=cache)


def test_gsi_dem5a_tile_is_still_a_256x256_csv_grid(client):
    x, y = lonlat_to_tile(KOBE_LON, KOBE_LAT, 15)
    payload = client.request(
        tile_url(GSI_BASE, "dem5a", x, y, 15, "txt"), expect_content_type="text/plain"
    )
    array = parse_gsi_elevation_tile(payload)
    assert array.shape == (TILE_PX, TILE_PX)
    finite = array[np.isfinite(array)]
    assert finite.size > 0
    # Rokko: sea level to ~930 m. Anything outside this means the payload
    # changed units or meaning.
    assert 0.0 < float(finite.min()) and float(finite.max()) < 1000.0


def test_gsi_dem10b_fallback_tile_is_reachable(client):
    x, y = lonlat_to_tile(KOBE_LON, KOBE_LAT, 14)
    payload = client.request(
        tile_url(GSI_BASE, "dem", x, y, 14, "txt"), expect_content_type="text/plain"
    )
    assert parse_gsi_elevation_tile(payload).shape == (TILE_PX, TILE_PX)


@pytest.mark.parametrize("dataset", ["seamlessphoto", "ort"])
def test_gsi_imagery_tiles_still_decode(client, dataset):
    x, y = lonlat_to_tile(KOBE_LON, KOBE_LAT, 17)
    payload = client.request(
        tile_url(GSI_BASE, dataset, x, y, 17, "jpg"), expect_content_type="image/"
    )
    tile = decode_tile(payload)
    assert tile.shape == (3, TILE_PX, TILE_PX)
    assert tile.std() > 1.0, "tile is a flat colour; coverage or the URL may have changed"


def test_missing_tile_is_reported_as_missing_not_as_an_error(client):
    """Zoom 15 DEM tiles do not exist over open ocean far from Japan."""
    x, y = lonlat_to_tile(-150.0, 0.0, 15)
    assert client.request(
        tile_url(GSI_BASE, "dem5a", x, y, 15, "txt"),
        expect_content_type="text/plain",
        allow_missing=True,
    ) is None


def test_overpass_still_accepts_the_query_format(client):
    osm = OsmConfig()
    query = build_query("building", (135.2338, 34.7274, 135.2358, 34.7294), 25)
    payload = None
    errors = []
    for endpoint in (osm.endpoint, *osm.mirrors):
        try:
            payload = client.request(
                endpoint,
                method="POST",
                data=f"data={query}",
                expect_content_type="application/json",
                backoff_s=5.0,
            )
            break
        except Exception as exc:  # noqa: BLE001 - try the next mirror
            errors.append(f"{endpoint}: {exc}")
    if payload is None:
        pytest.skip(f"no Overpass endpoint reachable: {errors}")

    document = json.loads(payload)
    assert "elements" in document
    assert any(element.get("type") == "way" for element in document["elements"])
    # `out geom;` must still inline coordinates, otherwise our parser breaks.
    way = next(e for e in document["elements"] if e.get("type") == "way")
    assert "geometry" in way and "lat" in way["geometry"][0]
