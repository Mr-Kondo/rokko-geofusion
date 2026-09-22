"""OpenStreetMap vector layers for the ROI, via the Overpass API.

Notes on the upstream service (verified, not assumed):

* Overpass answers ``POST /api/interpreter`` with the query in the ``data``
  form field, and rejects requests without a descriptive ``User-Agent``
  (HTTP 406). Both are handled by :class:`~rokko_geofusion.io.http.HttpClient`.
* Public instances rate-limit to a couple of slots, so queries are issued one
  layer at a time and every response is cached on disk.
* ``out geom;`` returns inline coordinates for ways, which avoids a second
  round trip to resolve node references.

Relation handling is best-effort: outer members are polygonised and inner
rings are not subtracted. Multipolygon buildings with courtyards are therefore
slightly over-sized; this is recorded in the layer metadata rather than hidden.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rokko_geofusion.config import Config
from rokko_geofusion.crs import RoiGeometry
from rokko_geofusion.exceptions import DataSourceError, UnsupportedError
from rokko_geofusion.io.http import HttpClient
from rokko_geofusion.utils.metadata import build_metadata, write_sidecar

logger = logging.getLogger(__name__)

#: Overpass QL body per layer. ``{bbox}`` is substituted with "S,W,N,E".
LAYER_QUERIES: dict[str, str] = {
    "building": """
        way["building"]({bbox});
        relation["building"]({bbox});
    """,
    "road": """
        way["highway"]({bbox});
    """,
    "water": """
        way["natural"="water"]({bbox});
        way["waterway"]({bbox});
        relation["natural"="water"]({bbox});
    """,
    "landuse": """
        way["landuse"]({bbox});
        way["leisure"]({bbox});
        relation["landuse"]({bbox});
    """,
    "railway": """
        way["railway"]({bbox});
    """,
}

#: Tags promoted to their own column; everything else is kept in ``tags``.
PROMOTED_TAGS = (
    "name",
    "building",
    "building:levels",
    "height",
    "highway",
    "surface",
    "landuse",
    "leisure",
    "natural",
    "waterway",
    "railway",
    "amenity",
)

#: Tags that make a closed way an area rather than a ring-shaped line.
_AREA_TAGS = ("building", "landuse", "leisure", "natural", "amenity", "area")


@dataclass(frozen=True)
class VectorLayer:
    name: str
    path: Path
    feature_count: int
    geometry_types: tuple[str, ...]
    crs: str
    metadata: dict[str, Any]


def build_query(layer: str, bounds_geographic: tuple[float, float, float, float],
                timeout_s: float) -> str:
    """Compose the Overpass QL for one layer over a geographic bbox."""
    if layer not in LAYER_QUERIES:
        raise UnsupportedError(
            f"no Overpass query defined for layer {layer!r}; "
            f"known layers: {sorted(LAYER_QUERIES)}"
        )
    min_lon, min_lat, max_lon, max_lat = bounds_geographic
    bbox = f"{min_lat},{min_lon},{max_lat},{max_lon}"  # Overpass order: S,W,N,E
    body = LAYER_QUERIES[layer].format(bbox=bbox).strip()
    return f"[out:json][timeout:{int(timeout_s)}];\n({body}\n);\nout geom;"


def _is_area(tags: dict[str, str], closed: bool) -> bool:
    if not closed:
        return False
    if tags.get("area") == "no":
        return False
    return any(key in tags for key in _AREA_TAGS)


def _way_geometry(element: dict[str, Any]):
    from shapely.geometry import LineString, Polygon

    coordinates = [(node["lon"], node["lat"]) for node in element.get("geometry", [])]
    if len(coordinates) < 2:
        return None
    closed = len(coordinates) >= 4 and coordinates[0] == coordinates[-1]
    if _is_area(element.get("tags", {}), closed):
        try:
            polygon = Polygon(coordinates)
            return polygon if polygon.is_valid else polygon.buffer(0)
        except (ValueError, TypeError):
            return None
    return LineString(coordinates)


def _relation_geometry(element: dict[str, Any]):
    """Best-effort multipolygon: polygonise the outer members."""
    from shapely.geometry import LineString, MultiPolygon, Polygon
    from shapely.ops import polygonize, unary_union

    lines: list[LineString] = []
    for member in element.get("members", []):
        if member.get("role") not in ("outer", "", None):
            continue
        coordinates = [(node["lon"], node["lat"]) for node in member.get("geometry", [])]
        if len(coordinates) >= 2:
            lines.append(LineString(coordinates))
    if not lines:
        return None
    polygons = list(polygonize(unary_union(lines)))
    if not polygons:
        return None
    if len(polygons) == 1:
        return polygons[0]
    return MultiPolygon([p for p in polygons if isinstance(p, Polygon)])


def _elements_to_records(elements: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    skipped = 0
    for element in elements:
        kind = element.get("type")
        if kind == "way":
            geometry = _way_geometry(element)
        elif kind == "relation":
            geometry = _relation_geometry(element)
        else:
            continue
        if geometry is None or geometry.is_empty:
            skipped += 1
            continue

        tags = element.get("tags", {}) or {}
        record: dict[str, Any] = {
            "osm_id": f"{kind}/{element.get('id')}",
            "osm_type": kind,
            "geometry": geometry,
        }
        for key in PROMOTED_TAGS:
            record[key.replace(":", "_")] = tags.get(key)
        record["tags"] = json.dumps(tags, ensure_ascii=False)
        records.append(record)
    if skipped:
        logger.debug("skipped %d element(s) with unusable geometry", skipped)
    return records


def fetch_layer(
    config: Config,
    roi: RoiGeometry,
    layer: str,
    *,
    client: HttpClient,
    overwrite: bool = False,
):
    """Download one OSM layer and return it as a GeoDataFrame in the projected CRS."""
    import geopandas as gpd

    osm = config.gis.osm
    query = build_query(layer, roi.bounds_geographic, osm.timeout_s)
    endpoints = [osm.endpoint, *osm.mirrors]

    payload: bytes | None = None
    last_error: Exception | None = None
    for endpoint in endpoints:
        try:
            payload = client.request(
                endpoint,
                method="POST",
                data=f"data={query}",
                expect_content_type="application/json",
                force_refresh=overwrite,
                backoff_s=osm.backoff_s,
                context={"layer": layer, "roi": roi.key},
            )
            break
        except DataSourceError as exc:
            last_error = exc
            logger.warning("Overpass endpoint %s failed for layer %s: %s",
                           endpoint, layer, exc)
    if payload is None:
        raise DataSourceError(
            f"all Overpass endpoints failed for layer {layer!r}: {last_error}"
        )

    try:
        document = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DataSourceError(
            f"Overpass returned a non-JSON body for layer {layer!r}; the API may have "
            f"changed: {exc}"
        ) from exc

    records = _elements_to_records(document.get("elements", []))
    if not records:
        logger.warning("layer %s: Overpass returned no usable features for this ROI", layer)
        return gpd.GeoDataFrame(
            {"osm_id": [], "geometry": []}, geometry="geometry", crs=config.crs.geographic
        ).to_crs(config.crs.projected)

    frame = gpd.GeoDataFrame(records, geometry="geometry", crs=config.crs.geographic)
    frame = frame.to_crs(config.crs.projected)

    # Clip to the exact ROI square; Overpass answers a geographic bbox, which
    # is slightly larger than the projected analysis square.
    clipped = gpd.clip(frame, roi.polygon)
    clipped = clipped[~clipped.geometry.is_empty & clipped.geometry.notna()]
    logger.info("layer %-9s %5d features (%d before clipping to the ROI)",
                layer, len(clipped), len(frame))
    return clipped.reset_index(drop=True)


def estimate_building_heights(frame, config: Config):
    """Add a ``height_m`` column derived from OSM tags where available.

    This is an *attribute-derived* height, not a measured one; the column
    ``height_source`` records which tag it came from so that fusion never
    confuses it with a LiDAR-measured object height.
    """
    import numpy as np
    import pandas as pd

    if frame.empty:
        frame["height_m"] = []
        frame["height_source"] = []
        return frame

    heights = pd.Series(np.nan, index=frame.index, dtype="float64")
    sources = pd.Series(None, index=frame.index, dtype="object")

    for key in config.gis.building_height_keys:
        column = key.replace(":", "_")
        if column not in frame.columns:
            continue
        raw = pd.to_numeric(
            frame[column].astype("string").str.extract(r"([0-9]+\.?[0-9]*)")[0],
            errors="coerce",
        )
        if key.endswith("levels"):
            raw = raw * config.gis.metres_per_level
        fill = heights.isna() & raw.notna()
        heights[fill] = raw[fill]
        sources[fill] = key

    frame = frame.copy()
    frame["height_m"] = heights
    frame["height_source"] = sources
    known = int(heights.notna().sum())
    if len(frame):
        logger.info("building heights from OSM tags: %d/%d features (%.1f%%)",
                    known, len(frame), 100 * known / len(frame))
    return frame


def fetch_gis_layers(
    config: Config,
    roi: RoiGeometry,
    *,
    client: HttpClient,
    overwrite: bool = False,
) -> tuple[dict[str, VectorLayer], dict[str, str]]:
    """Download every configured layer; returns ``(layers, failures)``."""
    if "osm" not in config.gis.providers:
        raise UnsupportedError(
            f"only the 'osm' GIS provider is implemented; configured: "
            f"{config.gis.providers}"
        )

    out_dir = config.paths.interim
    gpkg_path = out_dir / "gis.gpkg"
    layers: dict[str, VectorLayer] = {}
    failures: dict[str, str] = {}

    for layer_name in config.gis.osm.layers:
        parquet_path = out_dir / f"gis_{layer_name}.parquet"
        if parquet_path.is_file() and not overwrite:
            import geopandas as gpd

            existing = gpd.read_parquet(parquet_path)
            logger.info("reusing existing layer %s (%d features)", layer_name, len(existing))
            layers[layer_name] = VectorLayer(
                name=layer_name,
                path=parquet_path,
                feature_count=len(existing),
                geometry_types=tuple(sorted(existing.geom_type.dropna().unique())),
                crs=str(existing.crs),
                metadata={"status": "cached"},
            )
            continue

        try:
            frame = fetch_layer(config, roi, layer_name, client=client, overwrite=overwrite)
        except DataSourceError as exc:
            # One unreachable layer must not discard the layers that worked --
            # but it is recorded as a failure, never as "no features here".
            logger.error("layer %s could not be downloaded: %s", layer_name, exc)
            failures[layer_name] = str(exc)
            continue
        if layer_name == "building":
            frame = estimate_building_heights(frame, config)

        metadata = build_metadata(
            kind=f"gis_{layer_name}",
            source=f"osm:overpass:{config.gis.osm.endpoint}",
            crs=config.crs.projected,
            roi=roi.to_dict(),
            acquisition={
                "endpoint": config.gis.osm.endpoint,
                "attribution": config.gis.osm.attribution,
                "query": build_query(layer_name, roi.bounds_geographic,
                                     config.gis.osm.timeout_s),
            },
            processing={
                "clipped_to_roi": True,
                "relation_handling": "outer members polygonised; inner rings not subtracted",
            },
            config_fingerprint=config.fingerprint(),
            feature_count=len(frame),
        )

        frame.to_parquet(parquet_path, index=False)
        write_sidecar(parquet_path, metadata)
        if not frame.empty:
            frame.to_file(gpkg_path, layer=layer_name, driver="GPKG")

        layers[layer_name] = VectorLayer(
            name=layer_name,
            path=parquet_path,
            feature_count=len(frame),
            geometry_types=tuple(sorted(frame.geom_type.dropna().unique())),
            crs=str(frame.crs),
            metadata=metadata,
        )
        logger.info("wrote %s (%d features)", parquet_path, len(frame))

    if failures:
        logger.error(
            "%d of %d GIS layer(s) failed: %s",
            len(failures), len(config.gis.osm.layers), ", ".join(sorted(failures)),
        )
    return layers, failures


def load_layer(config: Config, layer: str):
    """Read a previously downloaded layer."""
    import geopandas as gpd

    path = config.paths.interim / f"gis_{layer}.parquet"
    if not path.is_file():
        raise FileNotFoundError(
            f"GIS layer {layer!r} has not been downloaded yet ({path}). "
            "Run scripts/download_gis.py first."
        )
    return gpd.read_parquet(path)
