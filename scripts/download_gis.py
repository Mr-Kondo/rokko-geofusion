#!/usr/bin/env python
"""Download the configured OSM vector layers for the ROI.

    python scripts/download_gis.py --config configs/rokko.yaml
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rokko_geofusion.crs import roi_from_config  # noqa: E402
from rokko_geofusion.gis.osm import fetch_gis_layers  # noqa: E402
from rokko_geofusion.io.http import client_from_config  # noqa: E402
from rokko_geofusion.utils.cli import build_parser, init_stage  # noqa: E402
from rokko_geofusion.utils.metadata import write_json  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = build_parser(__doc__.splitlines()[0])
    args = parser.parse_args(argv)
    ctx = init_stage(args, stage_name="download_gis")
    config, logger = ctx.config, ctx.logger

    roi = roi_from_config(config)
    logger.info("%s", roi)

    with client_from_config(config, cache_subdir="gis") as client:
        layers, failures = fetch_gis_layers(
            config, roi, client=client, overwrite=ctx.overwrite
        )
        write_json(
            config.paths.metrics / "download_gis.json",
            {
                "roi": roi.to_dict(),
                "layers": {
                    name: {
                        "path": str(layer.path),
                        "features": layer.feature_count,
                        "geometry_types": list(layer.geometry_types),
                        "crs": layer.crs,
                    }
                    for name, layer in layers.items()
                },
                "failures": failures,
                "http_cache": client.stats,
            },
        )

    for name, layer in layers.items():
        logger.info("%-9s %5d features  %s", name, layer.feature_count,
                    ", ".join(layer.geometry_types))
    for name, reason in failures.items():
        logger.error("%-9s FAILED: %s", name, reason)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
