#!/usr/bin/env python
"""Collect every computed statistic for the ROI into one structured document.

    python scripts/generate_report_data.py --config configs/rokko.yaml

Writes outputs/<roi>/reports/analysis_payload.json, which is the exact input
the LLM stage receives. Sections whose upstream stage has not run are reported
in a `missing` block rather than being omitted silently.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rokko_geofusion.crs import roi_from_config  # noqa: E402
from rokko_geofusion.report import write_payload  # noqa: E402
from rokko_geofusion.utils.cli import build_parser, init_stage  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = build_parser(__doc__.splitlines()[0])
    args = parser.parse_args(argv)
    ctx = init_stage(args, stage_name="generate_report_data")
    config, logger = ctx.config, ctx.logger

    roi = roi_from_config(config)
    path, payload = write_payload(config, roi)

    present = [key for key in ("terrain", "image_semantics", "fusion", "pointcloud",
                               "pointcloud_ml", "validation") if key in payload]
    logger.info("payload -> %s", path)
    logger.info("sections present: %s", ", ".join(present) or "none")
    for name, reason in payload.get("missing", {}).items():
        logger.warning("missing section %-16s %s", name, reason)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
