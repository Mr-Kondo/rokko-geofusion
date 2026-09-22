#!/usr/bin/env python
"""Render the ROI views and ask a vision model to interpret them.

    export ANTHROPIC_API_KEY=...
    python scripts/run_vlm.py --config configs/rokko.yaml
    python scripts/run_vlm.py --config configs/rokko.yaml --render-only

The model receives rendered views only -- never the raw cloud or rasters -- and
is instructed to produce no numbers. Every quantity in the final report comes
from Python.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rokko_geofusion.crs import roi_from_config  # noqa: E402
from rokko_geofusion.exceptions import ConfigurationRequiredError  # noqa: E402
from rokko_geofusion.utils.cli import build_parser, init_stage  # noqa: E402
from rokko_geofusion.utils.metadata import write_json  # noqa: E402
from rokko_geofusion.vlm.adapter import load_vlm  # noqa: E402
from rokko_geofusion.vlm.inputs import render_vlm_inputs  # noqa: E402

DEFAULT_QUESTION = (
    "Describe this area: its terrain, what covers the ground, and how the built "
    "environment is arranged. Point out spatial relationships between the "
    "landforms and what is built on them."
)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser(__doc__.splitlines()[0])
    parser.add_argument("--render-only", action="store_true",
                        help="render the input views and exit without calling a model")
    parser.add_argument("--question", default=DEFAULT_QUESTION,
                        help="what to ask the vision model")
    args = parser.parse_args(argv)
    ctx = init_stage(args, stage_name="run_vlm")
    config, logger = ctx.config, ctx.logger

    roi = roi_from_config(config)
    images = render_vlm_inputs(config, roi, overwrite=ctx.overwrite)
    if not images:
        logger.error("no views could be rendered; run the earlier stages first")
        return 1
    for image in images:
        logger.info("view %-20s %s", image.name, image.path)

    if args.render_only:
        logger.info("--render-only: skipping the model call")
        return 0

    try:
        adapter = load_vlm(config)
    except ConfigurationRequiredError as exc:
        logger.warning("VLM not run: %s", exc)
        write_json(
            config.paths.reports / "vlm_analysis.json",
            {"status": "unavailable", "reason": str(exc),
             "images": [image.to_dict() for image in images]},
        )
        return 0

    analysis = adapter.analyze(images, question=args.question)
    payload = analysis.to_dict()
    payload["status"] = "ok" if analysis.parsed else "unparsed"
    payload["question"] = args.question
    write_json(config.paths.reports / "vlm_analysis.json", payload)

    if analysis.parsed:
        logger.info("terrain: %s", analysis.terrain_description[:200])
        logger.info("land cover: %s", analysis.land_cover_description[:200])
        for item in analysis.uncertainties:
            logger.info("uncertainty: %s", item)
    else:
        logger.warning("the response was not valid JSON; the raw text was kept")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
