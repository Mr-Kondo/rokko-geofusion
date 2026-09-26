"""Text LLM adapters.

The LLM integrates what Python measured and what the VLM observed. It is told
explicitly to keep the four kinds of statement apart -- Measured, Observed,
Inferred, Uncertain -- and never to invent a number.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol, runtime_checkable

from rokko_geofusion.config import Config
from rokko_geofusion.exceptions import (
    ConfigurationRequiredError,
    DataSourceError,
    UnsupportedError,
)
from rokko_geofusion.vlm.adapter import _read_api_key, require_api_model

logger = logging.getLogger(__name__)


@runtime_checkable
class LlmAdapter(Protocol):
    provider: str
    model: str

    def complete(self, system: str, user: str) -> str: ...


class AnthropicLlm:
    provider = "anthropic"

    def __init__(self, config: Config) -> None:
        settings = config.llm
        self.model = settings.model
        self.max_output_tokens = settings.max_output_tokens
        self.temperature = settings.temperature
        api_key = _read_api_key(settings.api_key_env, self.provider)
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise UnsupportedError(
                "the anthropic provider needs the SDK: pip install -e '.[ai]'"
            ) from exc
        self._client = anthropic.Anthropic(api_key=api_key)

    def complete(self, system: str, user: str) -> str:
        logger.info("LLM request: %s (%d chars of context)", self.model, len(user))
        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=self.max_output_tokens,
                temperature=self.temperature,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
        except Exception as exc:  # noqa: BLE001 - surface provider errors verbatim
            raise DataSourceError(f"Anthropic LLM request failed: {exc}") from exc
        return "".join(block.text for block in response.content if block.type == "text")


class OpenAiLlm:
    provider = "openai"

    def __init__(self, config: Config) -> None:
        settings = config.llm
        self.model = settings.model
        self.max_output_tokens = settings.max_output_tokens
        self.temperature = settings.temperature
        api_key = _read_api_key(settings.api_key_env, self.provider)
        try:
            import openai
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise UnsupportedError(
                "the openai provider needs the SDK: pip install openai"
            ) from exc
        self._client = openai.OpenAI(api_key=api_key)

    def complete(self, system: str, user: str) -> str:
        logger.info("LLM request: %s (%d chars of context)", self.model, len(user))
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                max_tokens=self.max_output_tokens,
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
        except Exception as exc:  # noqa: BLE001
            raise DataSourceError(f"OpenAI LLM request failed: {exc}") from exc
        return response.choices[0].message.content or ""


class LocalLlm:
    """An open-weights model running on this machine, used for text only.

    With ``llm.model: auto`` this is the same vision-language checkpoint the
    VLM stage uses; a text-only Hugging Face id works too.
    """

    provider = "local"

    def __init__(self, config: Config, *, generator: Any | None = None) -> None:
        from rokko_geofusion.environment import detect_environment
        from rokko_geofusion.local_model import LocalGenerator, choose_local_model

        settings = config.llm
        self.max_output_tokens = settings.max_output_tokens
        self.temperature = settings.temperature
        if generator is None:
            choice = choose_local_model(settings.model, config,
                                        detect_environment(probe_packages=False))
            generator = LocalGenerator(choice)
        self._generator = generator
        self.model = generator.choice.model_id

    def complete(self, system: str, user: str) -> str:
        logger.info("LLM (local): %s (%d chars of context)", self.model, len(user))
        return self._generator.generate(
            system=system, content=[{"type": "text", "text": user}],
            max_new_tokens=self.max_output_tokens, temperature=self.temperature,
        )

    def describe(self) -> dict[str, Any]:
        return self._generator.describe()


def load_llm(config: Config) -> LlmAdapter:
    provider = config.llm.provider
    if not config.llm.enabled or provider == "none":
        raise ConfigurationRequiredError(
            "llm.provider",
            "no language model is configured",
            "Set llm.provider to 'local' (runs on this machine's GPU), or to "
            "'anthropic' / 'openai' with the API key named by llm.api_key_env.",
        )
    if provider == "local":
        return LocalLlm(config)
    require_api_model("llm", provider, config.llm.model)
    if provider == "anthropic":
        return AnthropicLlm(config)
    if provider == "openai":
        return OpenAiLlm(config)
    raise UnsupportedError(f"unknown LLM provider {provider!r}")
