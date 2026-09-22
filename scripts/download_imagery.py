#!/usr/bin/env python
"""Acquire the orthophoto / aerial imagery mosaic for the ROI.

    python scripts/download_imagery.py --config configs/rokko.yaml
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rokko_geofusion.crs import roi_from_config  # noqa: E402
from rokko_geofusion.imagery.orthophoto import fetch_imagery  # noqa: E402
from rokko_geofusion.io.http import client_from_config  # noqa: E402
from rokko_geofusion.utils.cli import build_parser, init_stage  # noqa: E402
from rokko_geofusion.utils.metadata import write_json  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = build_parser(__doc__.splitlines()[0])
    args = parser.parse_args(argv)
    ctx = init_stage(args, stage_name="download_imagery")
    config, logger = ctx.config, ctx.logger

    roi = roi_from_config(config)
    logger.info("%s", roi)

    with client_from_config(config, cache_subdir="imagery") as client:
        product = fetch_imagery(config, roi, client=client, overwrite=ctx.overwrite)
        write_json(
            config.paths.metrics / "download_imagery.json",
            {
                "roi": roi.to_dict(),
                "path": str(product.path),
                "source": product.source,
                "coverage": product.coverage,
                "grid": product.grid.to_dict(),
                "http_cache": client.stats,
            },
        )
    logger.info("imagery -> %s", product.path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
