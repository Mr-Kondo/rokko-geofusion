#!/usr/bin/env python
"""Compute terrain derivatives (slope, aspect, relief, hillshade, nDSM) and stats.

    python scripts/build_terrain.py --config configs/rokko.yaml

Writes GeoTIFFs to outputs/<roi>/raster/ and a statistics JSON to
outputs/<roi>/metrics/terrain.json. Object-height statistics require a DSM;
without one they are reported as explicitly unavailable.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rokko_geofusion.crs import roi_from_config  # noqa: E402
from rokko_geofusion.terrain.analysis import build_terrain  # noqa: E402
from rokko_geofusion.utils.cli import build_parser, init_stage  # noqa: E402
from rokko_geofusion.utils.metadata import write_json  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = build_parser(__doc__.splitlines()[0])
    parser.add_argument(
        "--no-density",
        action="store_true",
        help="skip the point-density raster (avoids reading the point cloud)",
    )
    args = parser.parse_args(argv)
    ctx = init_stage(args, stage_name="build_terrain")
    config, logger = ctx.config, ctx.logger

    roi = roi_from_config(config)
    dem_path = config.paths.interim / "dem.tif"
    dsm_path = config.paths.interim / "dsm.tif"
    cloud_path = config.paths.pointcloud / f"cloud.{config.output.pointcloud_format}"

    if not dem_path.is_file():
        logger.error("no DEM at %s. Run scripts/download_lidar.py first.", dem_path)
        return 1

    products = build_terrain(
        config,
        roi,
        dem_path=dem_path,
        dsm_path=dsm_path if dsm_path.is_file() else None,
        pointcloud_path=None if args.no_density else cloud_path,
        overwrite=ctx.overwrite,
    )

    write_json(config.paths.metrics / "terrain.json", products.statistics)
    for name, path in products.paths.items():
        logger.info("%-14s -> %s", name, path)
    for name, reason in products.unavailable.items():
        logger.warning("%-14s UNAVAILABLE: %s", name, reason)

    elevation = products.statistics.get("elevation") or {}
    slope_stats = products.statistics.get("slope") or {}
    logger.info(
        "elevation %.1f..%.1f m (mean %.1f) | slope mean %.1f deg, max %.1f deg | area %.2f km2",
        elevation.get("min", float("nan")),
        elevation.get("max", float("nan")),
        elevation.get("mean", float("nan")),
        slope_stats.get("mean", float("nan")),
        slope_stats.get("max", float("nan")),
        products.statistics.get("area_m2", 0.0) / 1e6,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
