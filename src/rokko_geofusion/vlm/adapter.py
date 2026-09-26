"""Vision-language model adapters.

The VLM's job is **visual interpretation only**. It never computes elevations,
areas, slopes or ratios: those come from Python and are passed to the LLM stage
separately. The prompt says so explicitly, and the response schema has no
numeric fields.

Providers are swappable: ``local`` runs an open-weights model in-process (see
:mod:`rokko_geofusion.local_model`), ``anthropic`` / ``openai`` call an API. A
provider that is not configured raises :class:`ConfigurationRequiredError` -- it
never returns invented text.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from rokko_geofusion.config import Config
from rokko_geofusion.exceptions import (
    ConfigurationRequiredError,
    DataSourceError,
    UnsupportedError,
)
from rokko_geofusion.vlm.inputs import VlmImage

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a remote-sensing analyst looking at rendered views of one geographic \
area. Your task is VISUAL INTERPRETATION ONLY.

Rules you must follow:
- Describe only what is visible in the images.
- Do NOT state elevations, slopes, areas, distances, counts or percentages. \
Every number is computed separately in Python and any number you invent would \
corrupt the analysis. Use qualitative language instead ("steep", "a large \
share of the area", "considerably higher than the surroundings").
- Distinguish what you can see from what you infer. Put inferences in \
`notable_patterns` and anything you are unsure about in `uncertainties`.
- If an image is ambiguous or an artefact is visible, say so rather than \
guessing.
- Answer with a single JSON object and nothing else."""

RESPONSE_SCHEMA = {
    "terrain_description": "string: the shape of the land, relief, drainage, orientation",
    "land_cover_description": "string: what covers the ground and how it is arranged",
    "built_environment_description": "string: settlement pattern, road structure, building types",
    "notable_patterns": ["string: spatial relationships or structures worth pointing out"],
    "uncertainties": ["string: what is unclear, ambiguous or possibly an artefact"],
}


@dataclass
class VlmAnalysis:
    """Structured result of one VLM call."""

    terrain_description: str = ""
    land_cover_description: str = ""
    built_environment_description: str = ""
    notable_patterns: list[str] = field(default_factory=list)
    uncertainties: list[str] = field(default_factory=list)
    raw_text: str = ""
    parsed: bool = True
    provider: str = ""
    model: str = ""
    images: list[dict[str, Any]] = field(default_factory=list)
    #: Device, dtype and why this model was chosen (local provider only).
    runtime: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@runtime_checkable
class VlmAdapter(Protocol):
    """Contract every vision backend satisfies."""

    provider: str
    model: str

    def analyze(self, images: Sequence[VlmImage], *, question: str) -> VlmAnalysis: ...


def _read_api_key(env_var: str, provider: str) -> str:
    key = os.environ.get(env_var, "").strip()
    if not key:
        raise ConfigurationRequiredError(
            f"environment variable {env_var}",
            f"the {provider} provider needs an API key",
            f"Set {env_var} before running this stage, or set vlm.provider/llm.provider "
            "to 'none' to skip the AI stages.",
        )
    return key


def _encode_image(path: Path) -> tuple[str, str]:
    media_type = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return media_type, base64.standard_b64encode(path.read_bytes()).decode("ascii")


def parse_json_response(text: str) -> tuple[dict[str, Any] | None, str]:
    """Extract the JSON object from a model response.

    Returns ``(payload, reason)``; ``payload`` is ``None`` when the response
    could not be parsed, in which case the raw text is preserved by the caller
    rather than being replaced with a plausible-looking default.
    """
    # Reasoning checkpoints emit <think>...</think> first; braces inside it
    # would otherwise be mistaken for the answer.
    stripped = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, re.DOTALL)
    if fenced:
        stripped = fenced.group(1)
    else:
        start, end = stripped.find("{"), stripped.rfind("}")
        if start != -1 and end > start:
            stripped = stripped[start:end + 1]
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError as exc:
        return None, f"response was not valid JSON: {exc}"
    if not isinstance(payload, dict):
        return None, f"response JSON was a {type(payload).__name__}, expected an object"
    return payload, ""


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    return [str(value)]


def build_user_prompt(images: Sequence[VlmImage], question: str) -> str:
    listing = "\n".join(f"{index}. {image.name}: {image.caption}"
                        for index, image in enumerate(images, start=1))
    return (
        f"{question}\n\n"
        f"You are given {len(images)} rendered views of the SAME area, in this order:\n"
        f"{listing}\n\n"
        "Reply with a single JSON object using exactly these keys:\n"
        f"{json.dumps(RESPONSE_SCHEMA, indent=2, ensure_ascii=False)}"
    )


class AnthropicVlm:
    """Anthropic messages API with image content blocks."""

    provider = "anthropic"

    def __init__(self, config: Config) -> None:
        settings = config.vlm
        self.model = settings.model
        self.max_output_tokens = settings.max_output_tokens
        self.temperature = settings.temperature
        self._api_key = _read_api_key(settings.api_key_env, self.provider)
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise UnsupportedError(
                "the anthropic provider needs the SDK: pip install -e '.[ai]'"
            ) from exc
        self._client = anthropic.Anthropic(api_key=self._api_key)

    def analyze(self, images: Sequence[VlmImage], *, question: str) -> VlmAnalysis:
        if not images:
            raise UnsupportedError("no rendered views were produced for the VLM")
        content: list[dict[str, Any]] = []
        for image in images:
            media_type, data = _encode_image(image.path)
            content.append({"type": "text", "text": f"{image.name}: {image.caption}"})
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": media_type, "data": data},
            })
        content.append({"type": "text", "text": build_user_prompt(images, question)})

        logger.info("VLM request: %s, %d image(s)", self.model, len(images))
        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=self.max_output_tokens,
                temperature=self.temperature,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": content}],
            )
        except Exception as exc:  # noqa: BLE001 - surface provider errors verbatim
            raise DataSourceError(f"Anthropic VLM request failed: {exc}") from exc

        text = "".join(block.text for block in response.content if block.type == "text")
        return _to_analysis(text, provider=self.provider, model=self.model, images=images)


class OpenAiVlm:
    """OpenAI chat completions with image_url content blocks."""

    provider = "openai"

    def __init__(self, config: Config) -> None:
        settings = config.vlm
        self.model = settings.model
        self.max_output_tokens = settings.max_output_tokens
        self.temperature = settings.temperature
        self._api_key = _read_api_key(settings.api_key_env, self.provider)
        try:
            import openai
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise UnsupportedError(
                "the openai provider needs the SDK: pip install openai"
            ) from exc
        self._client = openai.OpenAI(api_key=self._api_key)

    def analyze(self, images: Sequence[VlmImage], *, question: str) -> VlmAnalysis:
        if not images:
            raise UnsupportedError("no rendered views were produced for the VLM")
        content: list[dict[str, Any]] = []
        for image in images:
            media_type, data = _encode_image(image.path)
            content.append({"type": "text", "text": f"{image.name}: {image.caption}"})
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{media_type};base64,{data}"},
            })
        content.append({"type": "text", "text": build_user_prompt(images, question)})

        logger.info("VLM request: %s, %d image(s)", self.model, len(images))
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                max_tokens=self.max_output_tokens,
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": content},
                ],
            )
        except Exception as exc:  # noqa: BLE001
            raise DataSourceError(f"OpenAI VLM request failed: {exc}") from exc

        text = response.choices[0].message.content or ""
        return _to_analysis(text, provider=self.provider, model=self.model, images=images)


def image_for_model(path: Path, max_side_px: int):
    """Load a rendered view as RGB, shrunk so its longest side is ``max_side_px``."""
    from PIL import Image

    with Image.open(path) as source:
        image = source.convert("RGB")
    image.thumbnail((max_side_px, max_side_px), Image.Resampling.LANCZOS)
    return image


class LocalVlm:
    """An open-weights vision-language model running on this machine."""

    provider = "local"

    def __init__(self, config: Config, *, generator: Any | None = None) -> None:
        from rokko_geofusion.environment import detect_environment
        from rokko_geofusion.local_model import LocalGenerator, choose_local_model

        settings = config.vlm
        self.max_output_tokens = settings.max_output_tokens
        self.temperature = settings.temperature
        self.image_max_side_px = config.local_models.image_max_side_px
        if generator is None:
            choice = choose_local_model(settings.model, config,
                                        detect_environment(probe_packages=False))
            generator = LocalGenerator(choice)
        self._generator = generator
        self.model = generator.choice.model_id

    def analyze(self, images: Sequence[VlmImage], *, question: str) -> VlmAnalysis:
        if not images:
            raise UnsupportedError("no rendered views were produced for the VLM")
        content: list[dict[str, Any]] = []
        for image in images:
            content.append({"type": "text", "text": f"{image.name}: {image.caption}"})
            content.append({"type": "image",
                            "image": image_for_model(image.path, self.image_max_side_px)})
        content.append({"type": "text", "text": build_user_prompt(images, question)})

        logger.info("VLM (local): %s, %d image(s) at <= %d px",
                    self.model, len(images), self.image_max_side_px)
        text = self._generator.generate(
            system=SYSTEM_PROMPT, content=content,
            max_new_tokens=self.max_output_tokens, temperature=self.temperature,
        )
        return _to_analysis(text, provider=self.provider, model=self.model, images=images,
                            runtime=self._generator.describe())


def _to_analysis(text: str, *, provider: str, model: str, images: Sequence[VlmImage],
                 runtime: dict[str, Any] | None = None) -> VlmAnalysis:
    payload, reason = parse_json_response(text)
    if payload is None:
        logger.warning("VLM response could not be parsed (%s); keeping the raw text", reason)
        return VlmAnalysis(raw_text=text, parsed=False, provider=provider, model=model,
                           images=[image.to_dict() for image in images],
                           uncertainties=[f"structured parsing failed: {reason}"],
                           runtime=runtime or {})
    return VlmAnalysis(
        terrain_description=str(payload.get("terrain_description", "")),
        land_cover_description=str(payload.get("land_cover_description", "")),
        built_environment_description=str(payload.get("built_environment_description", "")),
        notable_patterns=_as_list(payload.get("notable_patterns")),
        uncertainties=_as_list(payload.get("uncertainties")),
        raw_text=text,
        parsed=True,
        provider=provider,
        model=model,
        images=[image.to_dict() for image in images],
        runtime=runtime or {},
    )


def require_api_model(section: str, provider: str, model: str) -> None:
    """``auto`` only means something for the local provider."""
    if model == "auto":
        raise ConfigurationRequiredError(
            f"{section}.model",
            f"'auto' picks a local checkpoint, but {section}.provider is {provider!r}",
            f"Name the {provider} model explicitly, e.g. {section}.model: claude-opus-5.",
        )


def load_vlm(config: Config) -> VlmAdapter:
    """Instantiate the configured vision backend."""
    provider = config.vlm.provider
    if not config.vlm.enabled or provider == "none":
        raise ConfigurationRequiredError(
            "vlm.provider",
            "no vision model is configured",
            "Set vlm.provider to 'local' (runs on this machine's GPU), or to "
            "'anthropic' / 'openai' with the API key named by vlm.api_key_env.",
        )
    if provider == "local":
        return LocalVlm(config)
    require_api_model("vlm", provider, config.vlm.model)
    if provider == "anthropic":
        return AnthropicVlm(config)
    if provider == "openai":
        return OpenAiVlm(config)
    raise UnsupportedError(f"unknown VLM provider {provider!r}")
