#!/usr/bin/env python
"""Integrate the measurements and the visual analysis into one GeoAI report.

    python scripts/run_llm.py --config configs/rokko.yaml

By default the same local open-weights model as the VLM stage writes the
report; set llm.provider to anthropic or openai to use a hosted model.

The report separates Measured (computed here), Observed (seen by the vision
model), Inferred (the language model's reasoning) and Uncertain. The language
model is given the numbers; it never produces them.
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
from rokko_geofusion.llm.report import run_llm_analysis  # noqa: E402
from rokko_geofusion.report import write_payload  # noqa: E402
from rokko_geofusion.utils.cli import build_parser, init_stage  # noqa: E402
from rokko_geofusion.utils.metadata import read_json, write_json  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = build_parser(__doc__.splitlines()[0])
    parser.add_argument("--prompt-only", action="store_true",
                        help="write the exact prompt and exit without calling a model")
    args = parser.parse_args(argv)
    ctx = init_stage(args, stage_name="run_llm")
    config, logger = ctx.config, ctx.logger

    result_path = config.paths.reports / "geoai_report.json"
    markdown_path = config.paths.reports / "geoai_report.md"
    if not args.prompt_only:
        # Replace the previous report before anything can fail, so that no
        # outcome -- not even an unanticipated exception -- leaves an earlier
        # run's report standing as this run's.
        write_json(result_path, {"status": "incomplete",
                                 "reason": "the LLM run started but did not finish"})
        markdown_path.unlink(missing_ok=True)

    roi = roi_from_config(config)
    payload_path, payload = write_payload(config, roi)
    logger.info("measurements -> %s", payload_path)

    vlm_path = config.paths.reports / "vlm_analysis.json"
    vlm_analysis = None
    if vlm_path.is_file():
        candidate = read_json(vlm_path)
        if candidate.get("status") == "ok":
            vlm_analysis = {
                key: candidate.get(key)
                for key in ("terrain_description", "land_cover_description",
                            "built_environment_description", "notable_patterns",
                            "uncertainties")
            }
        else:
            logger.warning("visual analysis is %s; continuing without it",
                           candidate.get("status"))
    else:
        logger.warning("no visual analysis found; run scripts/run_vlm.py first")

    if args.prompt_only:
        from rokko_geofusion.llm.report import SYSTEM_PROMPT, build_prompt

        prompt_path = config.paths.reports / "llm_prompt.txt"
        prompt_path.write_text(
            SYSTEM_PROMPT + "\n\n---\n\n"
            + build_prompt(payload, vlm_analysis, language=config.llm.language),
            encoding="utf-8",
        )
        logger.info("prompt -> %s", prompt_path)
        return 0

    try:
        report = run_llm_analysis(config, payload, vlm_analysis)
    except ConfigurationRequiredError as exc:
        logger.warning("LLM not run: %s", exc)
        write_json(result_path, {"status": "unavailable", "reason": str(exc)})
        return 0
    except GeoFusionError as exc:
        logger.error("LLM failed: %s", exc)
        write_json(result_path, {"status": "failed", "reason": str(exc)})
        return 1

    result = report.to_dict()
    result["status"] = "ok" if report.parsed else "unparsed"
    result["used_visual_analysis"] = vlm_analysis is not None
    write_json(result_path, result)

    if not report.parsed:
        # A Markdown file of empty sections would read like a finished report.
        logger.warning("the response was not valid JSON; kept the raw text in %s and "
                       "wrote no Markdown report", result_path)
        return 0

    markdown_path.write_text(report.to_markdown(roi.key), encoding="utf-8")
    logger.info("report -> %s", markdown_path)
    logger.info("summary: %s", report.summary[:300])
    logger.info("measured=%d observed=%d inferred=%d uncertain=%d",
                len(report.measured), len(report.observed),
                len(report.inferred), len(report.uncertain))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
