"""DEM / DSM acquisition, normalised onto the canonical ROI grid.

Providers
---------
``gsi_tile``
    GSI elevation tiles: 256x256 CSV grids of orthometric heights on the
    Web-Mercator XYZ pyramid. ``dem5a``/``dem5b``/``dem5c`` are the 5 m mesh
    (zoom 15) and ``dem`` is DEM10B (10 m mesh, zoom 14). Missing tiles inside
    the primary dataset are filled from the configured fallback datasets, and
    the resulting coverage fraction is recorded in the metadata sidecar.
``local_raster`` / ``local_xyz``
    A file the user supplied (GeoTIFF, or a regular-grid XYZ text file). XYZ
    files are read in chunks so a multi-gigabyte file never has to fit in RAM.
``ckan``
    A resource URL discovered through a CKAN catalogue and pinned in the
    config (see :mod:`rokko_geofusion.io.ckan`).
``none``
    Raises :class:`ConfigurationRequiredError`. This is the default for DSM:
    no verifiable public DSM URL exists for the default ROI, and inventing one
    would be worse than failing loudly.

**Important:** GSI elevation tiles are a *regular gridded product*, not raw
LiDAR returns. Every product written here records ``is_true_lidar`` so that
downstream reports never present a resampled grid as a LiDAR point cloud.
"""

from __future__ import annotations

import io as _io
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from rokko_geofusion.config import Config, ElevationSourceConfig
from rokko_geofusion.crs import GridSpec, RoiGeometry
from rokko_geofusion.exceptions import (
    ConfigurationRequiredError,
    DataUnavailableError,
    UnsupportedError,
)
from rokko_geofusion.io import ckan as ckan_io
from rokko_geofusion.io.http import HttpClient
from rokko_geofusion.io.raster import (
    coverage_fraction,
    read_raster,
    reproject_array_to_grid,
    write_grid_raster,
)
from rokko_geofusion.io.tiles import (
    TILE_PX,
    TileRange,
    tile_range_for_geographic_bounds,
    tile_url,
)
from rokko_geofusion.utils.logging import log_failure_context
from rokko_geofusion.utils.metadata import build_metadata

logger = logging.getLogger(__name__)

ElevationKind = Literal["dem", "dsm"]

#: Documented native zoom level of each GSI elevation product.
GSI_ELEVATION_ZOOM: dict[str, int] = {"dem5a": 15, "dem5b": 15, "dem5c": 15, "dem": 14}
#: Nominal mesh spacing, for the metadata sidecar.
GSI_ELEVATION_MESH_M: dict[str, float] = {"dem5a": 5.0, "dem5b": 5.0, "dem5c": 5.0, "dem": 10.0}


@dataclass(frozen=True)
class ElevationProduct:
    """A DEM or DSM raster on the canonical ROI grid."""

    kind: ElevationKind
    path: Path
    grid: GridSpec
    source: str
    is_true_lidar: bool
    coverage: float
    metadata: dict[str, Any]

    def read(self) -> np.ndarray:
        array, _ = read_raster(self.path, band=1)
        return array


# ---------------------------------------------------------------------------
# GSI elevation tiles
# ---------------------------------------------------------------------------
def parse_gsi_elevation_tile(payload: bytes, *, nodata_token: str = "e") -> np.ndarray:
    """Parse one 256x256 CSV elevation tile into a float32 array with NaN gaps."""
    array = np.genfromtxt(
        _io.BytesIO(payload),
        delimiter=",",
        dtype=np.float32,
        missing_values=nodata_token,
        filling_values=np.nan,
    )
    if array.shape != (TILE_PX, TILE_PX):
        raise UnsupportedError(
            f"expected a {TILE_PX}x{TILE_PX} elevation tile, got {array.shape}. "
            "The upstream tile format may have changed."
        )
    return array


def _mosaic_gsi_dataset(
    client: HttpClient,
    *,
    base_url: str,
    dataset: str,
    tile_range: TileRange,
    nodata_token: str,
    max_tiles: int,
    tile_crs: str,
) -> tuple[np.ndarray, Any, dict[str, Any]]:
    """Download one dataset's tiles and assemble them into a Web-Mercator mosaic."""
    if tile_range.count > max_tiles:
        raise UnsupportedError(
            f"{tile_range.count} tiles requested for {dataset} at zoom {tile_range.zoom}, "
            f"above the configured limit of {max_tiles}. Reduce roi.radius_m or the zoom."
        )

    urls = {
        (x, y): tile_url(base_url, dataset, x, y, tile_range.zoom, "txt")
        for x, y in tile_range
    }
    payloads = client.get_many(
        list(urls.values()),
        expect_content_type="text/plain",
        allow_missing=True,
        desc=f"{dataset} z{tile_range.zoom}",
    )

    mosaic = np.full((tile_range.pixel_height, tile_range.pixel_width), np.nan, np.float32)
    present = 0
    for (x, y), url in urls.items():
        payload = payloads.get(url)
        if not payload:
            continue
        tile = parse_gsi_elevation_tile(payload, nodata_token=nodata_token)
        row, col = tile_range.offset_of(x, y)
        mosaic[row:row + TILE_PX, col:col + TILE_PX] = tile
        present += 1

    stats = {
        "dataset": dataset,
        "zoom": tile_range.zoom,
        "tiles_requested": tile_range.count,
        "tiles_returned": present,
        "tile_crs": tile_crs,
    }
    logger.info("%s: %d/%d tiles returned data", dataset, present, tile_range.count)
    return mosaic, tile_range.transform(), stats


def _fetch_gsi_elevation(
    config: Config,
    roi: RoiGeometry,
    source: ElevationSourceConfig,
    grid: GridSpec,
    client: HttpClient,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fetch the primary dataset and fill its gaps from the fallbacks."""
    gsi = source.gsi
    datasets = [gsi.dataset, *[d for d in gsi.fallback_datasets if d != gsi.dataset]]
    # Pad the download by two native cells so bilinear resampling has support
    # at the ROI edge instead of producing a NaN fringe.
    padded = roi.buffered(3 * config.lidar.resolution_m)

    elevation = np.full(grid.shape, np.nan, np.float32)
    per_dataset: list[dict[str, Any]] = []

    for dataset in datasets:
        zoom = gsi.zoom or GSI_ELEVATION_ZOOM.get(dataset)
        if zoom is None:
            raise ConfigurationRequiredError(
                "lidar.<kind>.gsi.zoom",
                f"no documented default zoom for GSI dataset {dataset!r}",
            )
        tile_range = tile_range_for_geographic_bounds(padded.bounds_geographic, zoom)
        mosaic, transform, stats = _mosaic_gsi_dataset(
            client,
            base_url=gsi.base_url,
            dataset=dataset,
            tile_range=tile_range,
            nodata_token=gsi.nodata_token,
            max_tiles=config.lidar.max_tiles,
            tile_crs=config.crs.tile,
        )
        if stats["tiles_returned"] == 0:
            logger.warning("dataset %s returned no tiles for this ROI", dataset)
            per_dataset.append({**stats, "contributed_cells": 0})
            continue

        regridded = reproject_array_to_grid(
            mosaic,
            src_transform=transform,
            src_crs=config.crs.tile,
            grid=grid,
            resampling=config.lidar.resampling,
            src_nodata=float("nan"),
            dst_nodata=float("nan"),
            dtype=np.float32,
        )
        gaps = ~np.isfinite(elevation)
        filled = gaps & np.isfinite(regridded)
        elevation[filled] = regridded[filled]
        stats["mesh_m"] = GSI_ELEVATION_MESH_M.get(dataset)
        stats["contributed_cells"] = int(filled.sum())
        per_dataset.append(stats)
        logger.info("%s contributed %d cells; coverage now %.1f%%",
                    dataset, stats["contributed_cells"], 100 * coverage_fraction(elevation))
        if np.isfinite(elevation).all():
            break

    coverage = coverage_fraction(elevation)
    if coverage == 0.0:
        log_failure_context(
            logger,
            what="GSI elevation download produced no data",
            target=", ".join(datasets),
            roi=roi.key,
            crs=f"{config.crs.tile} -> {grid.crs}",
            cause="the ROI may be outside coverage, or the tile URL scheme changed",
        )
        raise DataUnavailableError(
            f"no GSI elevation data for ROI {roi.key} from datasets {datasets}"
        )

    return elevation, {
        "provider": "gsi_tile",
        "datasets": per_dataset,
        "attribution": gsi.attribution,
        "base_url": gsi.base_url,
    }


# ---------------------------------------------------------------------------
# Local files
# ---------------------------------------------------------------------------
def _load_local_raster(
    config: Config,
    roi: RoiGeometry,
    source: ElevationSourceConfig,
    grid: GridSpec,
    key_prefix: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    if not source.local.path:
        raise ConfigurationRequiredError(
            f"{key_prefix}.local.path",
            "provider is 'local_raster' but no file was given",
            "Put the file under data/raw/ and set the path in your config.",
        )
    path = Path(source.local.path)
    if not path.is_absolute():
        path = config.root / path
    if not path.is_file():
        raise ConfigurationRequiredError(
            f"{key_prefix}.local.path",
            f"file not found: {path}",
            "Raw inputs are never downloaded implicitly; place the file yourself.",
        )

    import rasterio

    with rasterio.open(path) as dataset:
        if dataset.crs is None and not source.local.crs:
            raise ConfigurationRequiredError(
                f"{key_prefix}.local.crs",
                f"{path} carries no CRS and none was configured",
                "Refusing to guess the CRS of an elevation raster.",
            )
        src_crs = dataset.crs.to_string() if dataset.crs else source.local.crs
        array = dataset.read(1).astype(np.float32)
        if dataset.nodata is not None:
            array = np.where(array == dataset.nodata, np.nan, array)
        transform = dataset.transform

    regridded = reproject_array_to_grid(
        array,
        src_transform=transform,
        src_crs=src_crs,
        grid=grid,
        resampling=config.lidar.resampling,
        src_nodata=float("nan"),
        dst_nodata=float("nan"),
        dtype=np.float32,
    )
    return regridded, {"provider": "local_raster", "file": str(path), "source_crs": src_crs}


def read_xyz_grid(
    path: Path,
    *,
    crs: str,
    chunk_rows: int = 2_000_000,
    delimiter: str | None = None,
) -> tuple[np.ndarray, Any, dict[str, Any]]:
    """Read a regular-grid XYZ text file in chunks into a raster.

    Two passes: the first derives the extent and grid spacing, the second fills
    the array. Nothing larger than ``chunk_rows`` rows is ever in memory, so a
    multi-gigabyte national-grid file is fine.
    """
    import pandas as pd
    from rasterio.transform import from_origin

    read_kwargs: dict[str, Any] = {
        "header": None,
        "usecols": [0, 1, 2],
        "names": ["x", "y", "z"],
        "dtype": np.float64,
        "chunksize": chunk_rows,
        "comment": "#",
    }
    if delimiter is None:
        read_kwargs["sep"] = r"[,\s]+"
        read_kwargs["engine"] = "python"
    else:
        read_kwargs["sep"] = delimiter

    min_x = min_y = float("inf")
    max_x = max_y = float("-inf")
    spacings_x: set[float] = set()
    spacings_y: set[float] = set()
    total_rows = 0

    for chunk in pd.read_csv(path, **read_kwargs):
        min_x, max_x = min(min_x, chunk.x.min()), max(max_x, chunk.x.max())
        min_y, max_y = min(min_y, chunk.y.min()), max(max_y, chunk.y.max())
        total_rows += len(chunk)
        if len(spacings_x) < 4:
            unique_x = np.unique(np.round(chunk.x.to_numpy()[:50_000], 6))
            unique_y = np.unique(np.round(chunk.y.to_numpy()[:50_000], 6))
            if unique_x.size > 1:
                spacings_x.add(float(np.min(np.diff(unique_x))))
            if unique_y.size > 1:
                spacings_y.add(float(np.min(np.diff(unique_y))))

    if total_rows == 0:
        raise DataUnavailableError(f"{path} contains no rows")
    spacing = min([*spacings_x, *spacings_y]) if (spacings_x or spacings_y) else None
    if not spacing or spacing <= 0:
        raise UnsupportedError(
            f"could not infer a regular grid spacing from {path}; "
            "set lidar.<kind>.local.grid_spacing_m explicitly"
        )

    width = int(round((max_x - min_x) / spacing)) + 1
    height = int(round((max_y - min_y) / spacing)) + 1
    array = np.full((height, width), np.nan, np.float32)

    for chunk in pd.read_csv(path, **read_kwargs):
        cols = np.rint((chunk.x.to_numpy() - min_x) / spacing).astype(np.int64)
        rows = np.rint((max_y - chunk.y.to_numpy()) / spacing).astype(np.int64)
        valid = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
        array[rows[valid], cols[valid]] = chunk.z.to_numpy()[valid].astype(np.float32)

    transform = from_origin(min_x - spacing / 2, max_y + spacing / 2, spacing, spacing)
    acquisition = {
        "rows": total_rows,
        "grid_spacing_m": spacing,
        "width": width,
        "height": height,
        "source_crs": crs,
        "is_regular_grid_xyz": True,
    }
    logger.info("read %s rows from %s onto a %dx%d grid at %.3f m spacing",
                f"{total_rows:,}", path.name, width, height, spacing)
    return array, transform, acquisition


def _load_local_xyz(
    config: Config,
    roi: RoiGeometry,
    source: ElevationSourceConfig,
    grid: GridSpec,
    key_prefix: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    if not source.local.path:
        raise ConfigurationRequiredError(
            f"{key_prefix}.local.path", "provider is 'local_xyz' but no file was given"
        )
    if not source.local.crs:
        raise ConfigurationRequiredError(
            f"{key_prefix}.local.crs",
            "plain XYZ text files carry no CRS",
            "Refusing to guess; state the CRS of the file explicitly.",
        )
    path = Path(source.local.path)
    if not path.is_absolute():
        path = config.root / path
    if not path.is_file():
        raise ConfigurationRequiredError(f"{key_prefix}.local.path", f"file not found: {path}")

    array, transform, acquisition = read_xyz_grid(path, crs=source.local.crs)
    regridded = reproject_array_to_grid(
        array,
        src_transform=transform,
        src_crs=source.local.crs,
        grid=grid,
        resampling=config.lidar.resampling,
        src_nodata=float("nan"),
        dst_nodata=float("nan"),
        dtype=np.float32,
    )
    return regridded, {"provider": "local_xyz", "file": str(path), **acquisition}


def _load_ckan(
    config: Config,
    roi: RoiGeometry,
    source: ElevationSourceConfig,
    grid: GridSpec,
    client: HttpClient,
    key_prefix: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    if not (source.ckan.resource_url or source.ckan.dataset_id):
        raise ConfigurationRequiredError(
            f"{key_prefix}.ckan.resource_url",
            "provider is 'ckan' but no resource was pinned",
            "Run `python scripts/download_lidar.py --discover '<query>'` to search "
            "the catalogue, then paste the resource URL into the config.",
        )
    url = ckan_io.resolve_resource_url(
        client,
        source.ckan.catalog_url,
        dataset_id=source.ckan.dataset_id,
        resource_url=source.ckan.resource_url,
        format_filter=source.ckan.format,
    )
    payload = client.request(url)
    if payload is None:
        raise DataUnavailableError(f"CKAN resource {url} returned no data")

    cache_path = config.paths.raw / "ckan" / Path(url).name
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(payload)
    logger.info("downloaded CKAN resource to %s (%.1f MB)", cache_path, len(payload) / 1e6)

    suffix = cache_path.suffix.lower()
    patched = source.model_copy(deep=True)
    patched.local.path = str(cache_path)
    patched.local.crs = source.local.crs
    if suffix in {".tif", ".tiff", ".vrt", ".img"}:
        array, acquisition = _load_local_raster(config, roi, patched, grid, key_prefix)
    elif suffix in {".xyz", ".txt", ".csv", ".asc"}:
        array, acquisition = _load_local_xyz(config, roi, patched, grid, key_prefix)
    else:
        raise UnsupportedError(
            f"CKAN resource {cache_path.name} has an unsupported extension {suffix!r}. "
            "Convert it to GeoTIFF or regular-grid XYZ and use a local_* provider."
        )
    return array, {**acquisition, "provider": "ckan", "resource_url": url}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def fetch_elevation(
    config: Config,
    roi: RoiGeometry,
    kind: ElevationKind,
    *,
    client: HttpClient,
    overwrite: bool = False,
) -> ElevationProduct:
    """Produce ``<interim>/<kind>.tif`` on the canonical ROI grid."""
    source: ElevationSourceConfig = getattr(config.lidar, kind)
    key_prefix = f"lidar.{kind}"
    grid = roi.grid(config.lidar.resolution_m)
    out_path = config.paths.interim / f"{kind}.tif"

    if out_path.is_file() and not overwrite:
        from rokko_geofusion.io.raster import grid_from_raster
        from rokko_geofusion.utils.metadata import read_sidecar

        existing = grid_from_raster(out_path)
        if existing.matches(grid):
            metadata = read_sidecar(out_path) or {}
            logger.info("reusing existing %s (%s); pass --overwrite to refetch",
                        kind.upper(), out_path)
            return ElevationProduct(
                kind=kind,
                path=out_path,
                grid=existing,
                source=str(metadata.get("source", "cached")),
                is_true_lidar=bool(metadata.get("is_true_lidar", False)),
                coverage=float(metadata.get("coverage", float("nan"))),
                metadata=metadata,
            )
        logger.warning("existing %s is on a different grid; regenerating", out_path)

    if source.provider == "none":
        raise ConfigurationRequiredError(
            f"{key_prefix}.provider",
            f"no {kind.upper()} source is configured",
            "No verifiable public DSM URL exists for the default ROI, so the "
            "pipeline refuses to invent one. Set provider to local_raster, "
            "local_xyz or ckan (see README 'Data sources').",
        )

    logger.info("acquiring %s via provider '%s' on a %dx%d grid at %.2f m",
                kind.upper(), source.provider, grid.width, grid.height, grid.resolution_m)

    if source.provider == "gsi_tile":
        array, acquisition = _fetch_gsi_elevation(config, roi, source, grid, client)
    elif source.provider == "local_raster":
        array, acquisition = _load_local_raster(config, roi, source, grid, key_prefix)
    elif source.provider == "local_xyz":
        array, acquisition = _load_local_xyz(config, roi, source, grid, key_prefix)
    elif source.provider == "ckan":
        array, acquisition = _load_ckan(config, roi, source, grid, client, key_prefix)
    else:  # pragma: no cover - pydantic restricts the literal
        raise UnsupportedError(f"unknown elevation provider {source.provider!r}")

    coverage = coverage_fraction(array)
    is_true_lidar = bool(config.lidar.is_true_lidar) and source.provider not in {"gsi_tile"}
    source_label = f"{source.provider}:{acquisition.get('datasets', acquisition.get('file', ''))}"
    if source.provider == "gsi_tile":
        source_label = "gsi_tile:" + "+".join(d["dataset"] for d in acquisition["datasets"])

    metadata = build_metadata(
        kind=kind,
        source=source_label,
        crs=grid.crs,
        resolution_m=grid.resolution_m,
        roi=roi.to_dict(),
        acquisition=acquisition,
        processing={
            "resampling": config.lidar.resampling,
            "tile_crs": config.crs.tile,
            "grid": grid.to_dict(),
            "nodata": "nan",
        },
        config_fingerprint=config.fingerprint(),
        coverage=coverage,
        is_true_lidar=is_true_lidar,
        notes=(
            "Regular gridded elevation product resampled from web tiles; "
            "NOT raw LiDAR returns." if source.provider == "gsi_tile" else None
        ),
    )

    write_grid_raster(
        out_path,
        array.astype(np.float32),
        grid,
        nodata=float("nan"),  # NaN is the explicit nodata marker for float rasters
        compress=config.output.compress,
        band_descriptions=[f"{kind}_m"],
        metadata=metadata,
    )
    logger.info("%s coverage: %.2f%% of %d cells", kind.upper(), 100 * coverage, array.size)

    return ElevationProduct(
        kind=kind,
        path=out_path,
        grid=grid,
        source=source_label,
        is_true_lidar=is_true_lidar,
        coverage=coverage,
        metadata=metadata,
    )
