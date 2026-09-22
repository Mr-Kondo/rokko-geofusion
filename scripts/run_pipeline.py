#!/usr/bin/env python
"""Run one stage, several stages, or the whole pipeline.

    python scripts/run_pipeline.py --config configs/rokko.yaml --stage all
    python scripts/run_pipeline.py --config configs/rokko.yaml --stage fusion
    python scripts/run_pipeline.py --config configs/rokko.yaml \
        --stage lidar imagery gis --overwrite

Each stage is the same scripts/*.py entry point you would run by hand, so
running a stage alone and running it as part of `--stage all` do exactly the
same thing. A failing stage is reported and the run continues unless
--stop-on-error is given; the exit code is non-zero if any stage failed.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rokko_geofusion.pipeline import (  # noqa: E402
    DEFAULT_ORDER,
    STAGE_SCRIPTS,
    run_pipeline,
)
from rokko_geofusion.utils.cli import build_parser, init_stage  # noqa: E402
from rokko_geofusion.utils.metadata import write_json  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = build_parser(__doc__.splitlines()[0])
    parser.add_argument(
        "--stage", nargs="+", default=["all"],
        help=f"stages to run: all, or any of {', '.join(STAGE_SCRIPTS)}",
    )
    parser.add_argument("--stop-on-error", action="store_true",
                        help="stop at the first failing stage")
    parser.add_argument("--list-stages", action="store_true",
                        help="print the default stage order and exit")
    args = parser.parse_args(argv)

    if args.list_stages:
        print("default order for --stage all:")
        for index, stage in enumerate(DEFAULT_ORDER, start=1):
            print(f"  {index:2d}. {stage:<14} ({STAGE_SCRIPTS[stage]})")
        extra = sorted(set(STAGE_SCRIPTS) - set(DEFAULT_ORDER))
        if extra:
            print("also available: " + ", ".join(extra))
        return 0

    ctx = init_stage(args, stage_name="run_pipeline")
    config, logger = ctx.config, ctx.logger

    result = run_pipeline(
        args.stage,
        config_path=config.source_path or args.config,
        overrides=getattr(args, "overrides", None),
        overwrite=ctx.overwrite,
        stop_on_error=args.stop_on_error,
        log_level=args.log_level,
        root=config.root,
    )

    write_json(config.paths.metrics / "pipeline.json", result.to_dict())
    logger.info("%-16s %-8s %8s", "stage", "status", "seconds")
    for outcome in result.stages:
        logger.info("%-16s %-8s %8.1f", outcome.stage, outcome.status, outcome.seconds)
        if outcome.error:
            logger.error("  %s: %s", outcome.stage, outcome.error)
    total = sum(outcome.seconds for outcome in result.stages)
    logger.info("total %.1f s, %d stage(s), %d failed",
                total, len(result.stages), len(result.failed))
    return 1 if result.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
