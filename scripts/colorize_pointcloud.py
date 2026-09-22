#!/usr/bin/env python
"""Build the RGB point cloud from the elevation raster and the orthophoto.

    python scripts/colorize_pointcloud.py --config configs/rokko.yaml
    python scripts/colorize_pointcloud.py --config configs/rokko.yaml \
        --set pointcloud.resolution_m=1.0
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rokko_geofusion.crs import roi_from_config  # noqa: E402
from rokko_geofusion.lidar.pointcloud import build_point_cloud  # noqa: E402
from rokko_geofusion.utils.cli import build_parser, init_stage  # noqa: E402
from rokko_geofusion.utils.metadata import write_json  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = build_parser(__doc__.splitlines()[0])
    args = parser.parse_args(argv)
    ctx = init_stage(args, stage_name="colorize_pointcloud")
    config, logger = ctx.config, ctx.logger

    roi = roi_from_config(config)
    interim = config.paths.interim
    dem_path = interim / "dem.tif"
    dsm_path = interim / "dsm.tif"
    imagery_path = interim / "orthophoto.tif"

    if not dem_path.is_file() and not dsm_path.is_file():
        logger.error(
            "no elevation raster found in %s. Run scripts/download_lidar.py first.", interim
        )
        return 1

    product = build_point_cloud(
        config,
        roi,
        dem_path=dem_path if dem_path.is_file() else None,
        dsm_path=dsm_path if dsm_path.is_file() else None,
        imagery_path=imagery_path if imagery_path.is_file() else None,
        overwrite=ctx.overwrite,
    )
    write_json(
        config.paths.metrics / "pointcloud.json",
        {
            "path": str(product.path),
            "format": product.format,
            "n_points": product.n_points,
            "surface": product.surface,
            "has_rgb": product.has_rgb,
            "is_true_lidar": product.is_true_lidar,
            "z_range_m": list(product.z_range),
            "bounds": list(product.bounds),
        },
    )
    logger.info("point cloud -> %s (%s points)", product.path, f"{product.n_points:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
