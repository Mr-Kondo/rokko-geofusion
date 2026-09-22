"""Aerial / orthophoto acquisition onto the canonical ROI grid.

Tiles are decoded, mosaicked in the tile CRS and reprojected once into the
projected CRS. The result is a georeferenced RGB GeoTIFF that shares its
lattice with the DEM/DSM products, which is what makes "sample the orthophoto
at a point's XY" a well-defined operation later on (V1, V5).
"""

from __future__ import annotations

import io as _io
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rokko_geofusion.config import Config
from rokko_geofusion.crs import GridSpec, RoiGeometry
from rokko_geofusion.exceptions import (
    ConfigurationRequiredError,
    DataUnavailableError,
    UnsupportedError,
)
from rokko_geofusion.io.http import HttpClient
from rokko_geofusion.io.raster import (
    grid_from_raster,
    read_raster,
    reproject_array_to_grid,
    write_grid_raster,
)
from rokko_geofusion.io.tiles import (
    TILE_PX,
    TileRange,
    ground_resolution_m,
    tile_range_for_geographic_bounds,
    tile_url,
)
from rokko_geofusion.utils.logging import log_failure_context
from rokko_geofusion.utils.metadata import build_metadata, read_sidecar

logger = logging.getLogger(__name__)

#: What each GSI imagery product actually is -- recorded in the sidecar so a
#: report never claims a seamless mosaic is a single-date orthophoto.
GSI_IMAGERY_DESCRIPTION = {
    "seamlessphoto": (
        "GSI seamless photo mosaic: a nationwide composite assembled from "
        "aerial photography of mixed acquisition dates."
    ),
    "ort": (
        "GSI orthorectified aerial photographs; coverage and acquisition year "
        "vary by tile and some tiles are greyscale."
    ),
}


@dataclass(frozen=True)
class ImageryProduct:
    path: Path
    grid: GridSpec
    source: str
    coverage: float
    metadata: dict[str, Any]

    def read(self) -> np.ndarray:
        """``(3, rows, cols)`` uint8 RGB."""
        array, _ = read_raster(self.path)
        return array


def decode_tile(payload: bytes) -> np.ndarray:
    """Decode one image tile into ``(3, 256, 256)`` uint8 RGB."""
    from PIL import Image

    with Image.open(_io.BytesIO(payload)) as image:
        rgb = image.convert("RGB")
        array = np.asarray(rgb, dtype=np.uint8)
    if array.shape[:2] != (TILE_PX, TILE_PX):
        raise UnsupportedError(
            f"expected a {TILE_PX}x{TILE_PX} image tile, got {array.shape[:2]}. "
            "The upstream tile format may have changed."
        )
    return np.transpose(array, (2, 0, 1))


def _mosaic_tiles(
    client: HttpClient,
    *,
    base_url: str,
    dataset: str,
    extension: str,
    tile_range: TileRange,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return ``(rgb_mosaic, valid_mask, tiles_returned)`` in the tile CRS."""
    urls = {
        (x, y): tile_url(base_url, dataset, x, y, tile_range.zoom, extension)
        for x, y in tile_range
    }
    expect = "image/" if extension in {"jpg", "jpeg", "png"} else None
    payloads = client.get_many(
        list(urls.values()),
        expect_content_type=expect,
        allow_missing=True,
        desc=f"{dataset} z{tile_range.zoom}",
    )

    mosaic = np.zeros((3, tile_range.pixel_height, tile_range.pixel_width), np.uint8)
    valid = np.zeros((tile_range.pixel_height, tile_range.pixel_width), bool)
    returned = 0
    for (x, y), url in urls.items():
        payload = payloads.get(url)
        if not payload:
            continue
        tile = decode_tile(payload)
        row, col = tile_range.offset_of(x, y)
        mosaic[:, row:row + TILE_PX, col:col + TILE_PX] = tile
        valid[row:row + TILE_PX, col:col + TILE_PX] = True
        returned += 1
    return mosaic, valid, returned


def _load_local_imagery(
    config: Config, source_path: str, grid: GridSpec
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    import rasterio

    path = Path(source_path)
    if not path.is_absolute():
        path = config.root / path
    if not path.is_file():
        raise ConfigurationRequiredError("imagery.local.path", f"file not found: {path}")

    with rasterio.open(path) as dataset:
        if dataset.crs is None and not config.imagery.local.crs:
            raise ConfigurationRequiredError(
                "imagery.local.crs", f"{path} carries no CRS and none was configured"
            )
        src_crs = dataset.crs.to_string() if dataset.crs else config.imagery.local.crs
        bands = min(dataset.count, 3)
        array = dataset.read(list(range(1, bands + 1))).astype(np.uint8)
        if bands == 1:
            array = np.repeat(array, 3, axis=0)
        transform = dataset.transform

    rgb = reproject_array_to_grid(
        array, src_transform=transform, src_crs=src_crs, grid=grid,
        resampling="bilinear", dst_nodata=0, dtype=np.uint8,
    )
    valid = rgb.any(axis=0)
    return rgb, valid, {"provider": "local_raster", "file": str(path), "source_crs": src_crs}


def fetch_imagery(
    config: Config,
    roi: RoiGeometry,
    *,
    client: HttpClient,
    overwrite: bool = False,
) -> ImageryProduct:
    """Produce ``<interim>/orthophoto.tif`` on the ROI grid."""
    imagery = config.imagery
    grid = roi.grid(imagery.resolution_m)
    out_path = config.paths.interim / "orthophoto.tif"

    if out_path.is_file() and not overwrite:
        existing = grid_from_raster(out_path)
        if existing.matches(grid):
            metadata = read_sidecar(out_path) or {}
            logger.info("reusing existing orthophoto (%s); pass --overwrite to refetch", out_path)
            return ImageryProduct(
                path=out_path,
                grid=existing,
                source=str(metadata.get("source", "cached")),
                coverage=float(metadata.get("coverage", float("nan"))),
                metadata=metadata,
            )
        logger.warning("existing %s is on a different grid; regenerating", out_path)

    if imagery.provider == "none":
        raise ConfigurationRequiredError(
            "imagery.provider", "no imagery source is configured"
        )

    native_gsd = ground_resolution_m(imagery.zoom, roi.center_geographic[1])
    if grid.resolution_m < native_gsd * 0.8:
        logger.warning(
            "imagery.resolution_m=%.2f m is finer than the native GSD at z%d (%.2f m); "
            "the extra pixels are interpolation, not detail",
            grid.resolution_m, imagery.zoom, native_gsd,
        )

    logger.info(
        "acquiring imagery via '%s:%s' at z%d (native GSD %.2f m) on a %dx%d grid at %.2f m",
        imagery.provider, imagery.dataset, imagery.zoom, native_gsd,
        grid.width, grid.height, grid.resolution_m,
    )

    if imagery.provider == "local_raster":
        rgb, valid_mask, info = _load_local_imagery(config, imagery.local.path or "", grid)
    elif imagery.provider == "gsi_tile":
        padded = roi.buffered(3 * imagery.resolution_m)
        tile_range = tile_range_for_geographic_bounds(padded.bounds_geographic, imagery.zoom)
        if tile_range.count > imagery.max_tiles:
            raise UnsupportedError(
                f"{tile_range.count} imagery tiles requested, above the configured "
                f"limit of {imagery.max_tiles}. Lower imagery.zoom or roi.radius_m."
            )
        mosaic, valid, returned = _mosaic_tiles(
            client,
            base_url=imagery.base_url,
            dataset=imagery.dataset,
            extension=imagery.extension,
            tile_range=tile_range,
        )
        if returned == 0:
            log_failure_context(
                logger,
                what="imagery download produced no tiles",
                target=f"{imagery.dataset} z{imagery.zoom}",
                roi=roi.key,
                crs=config.crs.tile,
                cause="the ROI may be outside coverage, or the tile URL scheme changed",
            )
            raise DataUnavailableError(
                f"no imagery tiles for ROI {roi.key} from {imagery.dataset}"
            )
        logger.info("%s: %d/%d tiles returned data", imagery.dataset, returned,
                    tile_range.count)

        transform = tile_range.transform()
        rgb = reproject_array_to_grid(
            mosaic, src_transform=transform, src_crs=config.crs.tile, grid=grid,
            resampling="bilinear", dst_nodata=0, dtype=np.uint8,
        )
        valid_mask = reproject_array_to_grid(
            valid.astype(np.uint8), src_transform=transform, src_crs=config.crs.tile,
            grid=grid, resampling="nearest", dst_nodata=0, dtype=np.uint8,
        ).astype(bool)
        info = {
            "provider": "gsi_tile",
            "dataset": imagery.dataset,
            "zoom": imagery.zoom,
            "tiles_requested": tile_range.count,
            "tiles_returned": returned,
            "tile_range": tile_range.to_dict(),
            "native_gsd_m": round(native_gsd, 4),
            "description": GSI_IMAGERY_DESCRIPTION.get(imagery.dataset, ""),
            "attribution": imagery.attribution,
        }
    else:  # pragma: no cover
        raise UnsupportedError(f"unknown imagery provider {imagery.provider!r}")

    coverage = float(valid_mask.mean())
    metadata = build_metadata(
        kind="orthophoto",
        source=f"{imagery.provider}:{imagery.dataset}",
        crs=grid.crs,
        resolution_m=grid.resolution_m,
        roi=roi.to_dict(),
        acquisition=info,
        processing={
            "resampling": "bilinear",
            "tile_crs": config.crs.tile,
            "grid": grid.to_dict(),
        },
        config_fingerprint=config.fingerprint(),
        coverage=coverage,
        notes=GSI_IMAGERY_DESCRIPTION.get(imagery.dataset),
    )

    write_grid_raster(
        out_path,
        rgb,
        grid,
        nodata=0,
        compress=config.output.compress,
        band_descriptions=["red", "green", "blue"],
        metadata=metadata,
    )
    logger.info("imagery coverage: %.2f%%", 100 * coverage)
    return ImageryProduct(
        path=out_path,
        grid=grid,
        source=f"{imagery.provider}:{imagery.dataset}",
        coverage=coverage,
        metadata=metadata,
    )
