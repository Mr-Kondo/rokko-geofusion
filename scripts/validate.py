#!/usr/bin/env python
"""Run the spatial validation checks (V1-V6) and write the results.

    python scripts/validate.py --config configs/rokko.yaml
    python scripts/validate.py --config configs/rokko.yaml --check seg

Exit code is non-zero if any check fails. Checks that cannot run (for example
anything needing a DSM when none is configured) are reported as unavailable,
never as a pass.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rokko_geofusion.crs import roi_from_config  # noqa: E402
from rokko_geofusion.utils.cli import build_parser, init_stage  # noqa: E402
from rokko_geofusion.utils.metadata import write_json  # noqa: E402
from rokko_geofusion.validation import run_all, summarise, to_markdown  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = build_parser(__doc__.splitlines()[0])
    parser.add_argument("--check", default=None,
                        help="run only this check (V1..V6 or SEG)")
    args = parser.parse_args(argv)
    ctx = init_stage(args, stage_name="validate")
    config, logger = ctx.config, ctx.logger

    roi = roi_from_config(config)
    results = run_all(config, roi, only=args.check)
    summary = summarise(results)
    summary["roi"] = roi.to_dict()
    summary["config_fingerprint"] = config.fingerprint()

    write_json(config.paths.metrics / "validation.json", summary)
    markdown = config.paths.metrics / "validation.md"
    markdown.write_text(
        f"# Validation - {roi.key}\n\n"
        f"config fingerprint `{config.fingerprint()}`\n\n"
        + to_markdown(results) + "\n",
        encoding="utf-8",
    )
    logger.info("wrote %s", markdown)

    counts = ", ".join(f"{status}={count}" for status, count in sorted(summary["counts"].items()))
    logger.info("validation summary: %s", counts)
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
