"""Integrated GeoAI analysis: Python measurements + VLM observation + LLM reasoning.

The output keeps four kinds of statement strictly apart, because conflating
them is how a geospatial report becomes confidently wrong:

``measured``
    numbers computed by this pipeline, quoted back with their units.
``observed``
    what the vision model saw in the rendered views (qualitative only).
``inferred``
    the language model's own reasoning, explicitly marked as reasoning.
``uncertain``
    limitations, missing modalities and things that need checking.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from typing import Any

from rokko_geofusion.config import Config
from rokko_geofusion.llm.adapter import LlmAdapter, load_llm
from rokko_geofusion.vlm.adapter import parse_json_response

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a geospatial analyst writing the integrated analysis of one area.

You receive two things:
1. MEASUREMENTS: numbers computed by a geospatial pipeline (elevations, slopes, \
areas, class fractions, validation results). These are the only numbers that \
exist. Quote them; never recompute, round differently, extrapolate or invent.
2. VISUAL ANALYSIS: qualitative observations from a vision model that looked at \
rendered views of the same area. It was forbidden from producing numbers.

Keep four kinds of statement strictly apart and never promote one to another:
- "measured": restate the pipeline's numbers with units and what they describe.
- "observed": what the vision model reported seeing. Qualitative only.
- "inferred": YOUR reasoning connecting the above. Every item must be \
recognisable as an inference, not a fact.
- "uncertain": limitations, missing data, and anything that would change the \
conclusion if checked. If a modality is unavailable, say what cannot be \
concluded because of it.

Rules:
- If the measurements do not support a statement, do not make it.
- Never state a quantity that is absent from MEASUREMENTS.
- Mention explicitly which analyses could not be performed and why.
- Answer with a single JSON object and nothing else."""

RESPONSE_SCHEMA = {
    "summary": "string: two or three sentences describing the area",
    "measured": ["string: a pipeline number, with units and what it describes"],
    "observed": ["string: a qualitative observation from the visual analysis"],
    "inferred": ["string: your own reasoning, recognisable as an inference"],
    "uncertain": ["string: a limitation, gap or thing to verify"],
    "recommended_checks": ["string: a concrete next step that would reduce uncertainty"],
}


@dataclass
class GeoAiReport:
    summary: str = ""
    measured: list[str] = field(default_factory=list)
    observed: list[str] = field(default_factory=list)
    inferred: list[str] = field(default_factory=list)
    uncertain: list[str] = field(default_factory=list)
    recommended_checks: list[str] = field(default_factory=list)
    raw_text: str = ""
    parsed: bool = True
    provider: str = ""
    model: str = ""
    language: str = "ja"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_markdown(self, roi_key: str) -> str:
        sections = [
            ("Measured (computed by the pipeline)", self.measured),
            ("Observed (visual interpretation)", self.observed),
            ("Inferred (model reasoning)", self.inferred),
            ("Uncertain / limitations", self.uncertain),
            ("Recommended checks", self.recommended_checks),
        ]
        lines = [f"# GeoAI analysis - {roi_key}", ""]
        if self.summary:
            lines += [self.summary, ""]
        for title, items in sections:
            lines.append(f"## {title}")
            lines += [f"- {item}" for item in items] or ["- (none reported)"]
            lines.append("")
        lines.append(f"_Generated with {self.provider}:{self.model}. "
                     "Numbers come from the pipeline, not from the language model._")
        return "\n".join(lines)


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    return [str(value)]


def build_prompt(
    payload: dict[str, Any],
    vlm_analysis: dict[str, Any] | None,
    *,
    language: str = "ja",
) -> str:
    """Compose the user message: measurements, visual analysis, response schema."""
    language_name = {"ja": "Japanese", "en": "English"}.get(language, language)
    visual = (
        json.dumps(vlm_analysis, indent=2, ensure_ascii=False)
        if vlm_analysis
        else '"No visual analysis is available for this ROI."'
    )
    return (
        "MEASUREMENTS (the only numbers that exist):\n"
        f"{json.dumps(payload, indent=2, ensure_ascii=False)}\n\n"
        "VISUAL ANALYSIS (qualitative, produced by a vision model):\n"
        f"{visual}\n\n"
        f"Write the analysis in {language_name}. Reply with a single JSON object "
        "using exactly these keys:\n"
        f"{json.dumps(RESPONSE_SCHEMA, indent=2, ensure_ascii=False)}"
    )


def run_llm_analysis(
    config: Config,
    payload: dict[str, Any],
    vlm_analysis: dict[str, Any] | None = None,
    *,
    adapter: LlmAdapter | None = None,
) -> GeoAiReport:
    """Ask the language model for the integrated analysis."""
    adapter = adapter or load_llm(config)
    prompt = build_prompt(payload, vlm_analysis, language=config.llm.language)
    text = adapter.complete(SYSTEM_PROMPT, prompt)

    parsed_payload, reason = parse_json_response(text)
    if parsed_payload is None:
        logger.warning("LLM response could not be parsed (%s); keeping the raw text", reason)
        return GeoAiReport(
            raw_text=text, parsed=False, provider=adapter.provider, model=adapter.model,
            language=config.llm.language,
            uncertain=[f"structured parsing failed: {reason}"],
        )
    return GeoAiReport(
        summary=str(parsed_payload.get("summary", "")),
        measured=_as_list(parsed_payload.get("measured")),
        observed=_as_list(parsed_payload.get("observed")),
        inferred=_as_list(parsed_payload.get("inferred")),
        uncertain=_as_list(parsed_payload.get("uncertain")),
        recommended_checks=_as_list(parsed_payload.get("recommended_checks")),
        raw_text=text,
        parsed=True,
        provider=adapter.provider,
        model=adapter.model,
        language=config.llm.language,
    )
