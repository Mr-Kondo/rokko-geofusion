#!/usr/bin/env python
"""Acquire DEM (and DSM, when configured) for the ROI.

    python scripts/download_lidar.py --config configs/rokko.yaml
    python scripts/download_lidar.py --config configs/rokko.yaml --kind dsm
    python scripts/download_lidar.py --discover "3次元点群"      # search CKAN

The DSM provider is deliberately ``none`` by default: no verifiable public DSM
download URL exists for the default ROI. ``--discover`` searches a CKAN
catalogue so you can pin a resource URL yourself instead of the pipeline
guessing one.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rokko_geofusion.crs import roi_from_config  # noqa: E402
from rokko_geofusion.exceptions import (  # noqa: E402
    ConfigurationRequiredError,
    GeoFusionError,
)
from rokko_geofusion.io import ckan as ckan_io  # noqa: E402
from rokko_geofusion.io.http import client_from_config  # noqa: E402
from rokko_geofusion.lidar.elevation import fetch_elevation  # noqa: E402
from rokko_geofusion.utils.cli import build_parser, init_stage  # noqa: E402
from rokko_geofusion.utils.metadata import write_json  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = build_parser(__doc__.splitlines()[0])
    parser.add_argument(
        "--kind",
        choices=["dem", "dsm", "both"],
        default="both",
        help="which elevation product to acquire",
    )
    parser.add_argument(
        "--discover",
        metavar="QUERY",
        default=None,
        help="search the configured CKAN catalogue and exit (no download)",
    )
    args = parser.parse_args(argv)
    ctx = init_stage(args, stage_name="download_lidar")
    config, logger = ctx.config, ctx.logger

    with client_from_config(config, cache_subdir="elevation") as client:
        if args.discover:
            catalog = config.lidar.dsm.ckan.catalog_url
            logger.info("searching %s for %r", catalog, args.discover)
            datasets = ckan_io.search_datasets(client, catalog, args.discover, rows=10)
            if not datasets:
                logger.warning("no datasets matched %r", args.discover)
                return 0
            for dataset in datasets:
                print(f"\n{dataset}")
            print(
                "\nPin the resource you want in your config:\n"
                "  lidar.dsm.provider: ckan\n"
                "  lidar.dsm.ckan.resource_url: <url from above>\n"
            )
            return 0

        roi = roi_from_config(config)
        logger.info("%s", roi)

        kinds = ["dem", "dsm"] if args.kind == "both" else [args.kind]
        summary: dict[str, object] = {"roi": roi.to_dict(), "products": {}}
        failures = 0

        for kind in kinds:
            try:
                product = fetch_elevation(
                    config, roi, kind, client=client, overwrite=ctx.overwrite
                )
            except ConfigurationRequiredError as exc:
                # Expected, documented state -- not a crash.
                logger.warning("%s unavailable: %s", kind.upper(), exc)
                summary["products"][kind] = {"status": "unavailable", "reason": str(exc)}
                continue
            except GeoFusionError as exc:
                logger.error("%s acquisition failed: %s", kind.upper(), exc)
                summary["products"][kind] = {"status": "failed", "reason": str(exc)}
                failures += 1
                continue

            summary["products"][kind] = {
                "status": "ok",
                "path": str(product.path),
                "source": product.source,
                "coverage": product.coverage,
                "is_true_lidar": product.is_true_lidar,
                "grid": product.grid.to_dict(),
            }
            logger.info("%s -> %s (coverage %.1f%%)",
                        kind.upper(), product.path, 100 * product.coverage)

        summary["http_cache"] = client.stats
        write_json(config.paths.metrics / "download_lidar.json", summary)

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
