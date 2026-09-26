"""Open-weights vision-language models run in-process with transformers.

Nothing leaves the machine and no API key is needed. One checkpoint serves both
AI stages: the VLM gives it rendered views plus text, the LLM gives it text only.
A text-only checkpoint also works for the LLM stage.

The model is picked from ``local_models.auto_tiers`` by the accelerator memory
actually detected, never assumed: the same config gives an 8B model on a 40 GB
A100 and a 4B model on a 16 GB T4.

Decoding is greedy unless a temperature is configured -- deliberately not the
sampling the checkpoint ships with -- so re-running an ROI on the same device
and dtype reproduces the same text (V6). A different dtype changes the text.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any

from rokko_geofusion.config import Config
from rokko_geofusion.environment import EnvironmentInfo, select_device
from rokko_geofusion.exceptions import DataSourceError, ResourceError, UnsupportedError

logger = logging.getLogger(__name__)

#: Apple silicon shares memory with the OS and every other process.
_UNIFIED_MEMORY_SHARE = 0.5


@dataclass(frozen=True)
class LocalModelChoice:
    """Which checkpoint runs where, and why -- recorded with every output."""

    model_id: str
    device: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def accelerator_memory_gb(device: str, env: EnvironmentInfo) -> float:
    """Memory the model can use on ``device``; 0 on CPU (smallest tier)."""
    if device == "cuda" and env.primary_gpu is not None:
        return env.primary_gpu.total_vram_gb
    if device == "mps":
        return (env.ram_total_gb or 0.0) * _UNIFIED_MEMORY_SHARE
    return 0.0


def choose_local_model(requested: str, config: Config, env: EnvironmentInfo) -> LocalModelChoice:
    """Resolve ``auto`` to a tier, or keep an explicitly configured model id."""
    device = select_device(config.runtime.device, env)
    if requested != "auto":
        return LocalModelChoice(requested, device, "configured explicitly")

    memory = accelerator_memory_gb(device, env)
    tiers = config.local_models.auto_tiers
    tier = next((t for t in tiers if memory >= t.min_memory_gb), tiers[-1])
    where = f"{memory:.1f} GB on {device}" if device != "cpu" else "CPU (no accelerator)"
    if device == "cpu":
        logger.warning("no GPU detected: the local model will run on the CPU and be slow")
    return LocalModelChoice(
        tier.model_id, device, f"auto: {where} meets the {tier.min_memory_gb:g} GB tier"
    )


def _torch_dtype(device: str):
    import torch

    if device == "cuda":
        # T4 (sm_75) has no bf16; bf16 is the checkpoints' native dtype elsewhere.
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.bfloat16


def _is_out_of_memory(error: BaseException) -> bool:
    import torch

    # CUDA raises torch.OutOfMemoryError; MPS raises a RuntimeError that says so.
    return isinstance(error, torch.OutOfMemoryError) or "out of memory" in str(error).lower()


class LocalGenerator:
    """Loads one checkpoint and turns chat messages into text."""

    def __init__(self, choice: LocalModelChoice) -> None:
        try:
            import torch
            from transformers import (
                AutoConfig,
                AutoModelForCausalLM,
                AutoModelForImageTextToText,
                AutoProcessor,
                AutoTokenizer,
            )
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise UnsupportedError(
                "the local provider needs torch + transformers: pip install -e '.[ml]'"
            ) from exc

        self._torch = torch
        self.choice = choice
        dtype = _torch_dtype(choice.device)
        self.dtype = str(dtype).removeprefix("torch.")
        logger.info("loading %s on %s (%s) -- %s", choice.model_id, choice.device,
                    self.dtype, choice.reason)
        try:
            self.multimodal = hasattr(AutoConfig.from_pretrained(choice.model_id),
                                      "vision_config")
            model_class = AutoModelForImageTextToText if self.multimodal else AutoModelForCausalLM
            loader = AutoProcessor if self.multimodal else AutoTokenizer
            self._processor = loader.from_pretrained(choice.model_id)
            self._model = model_class.from_pretrained(
                choice.model_id, dtype=dtype, device_map=choice.device
            ).eval()
        except OSError as exc:
            raise DataSourceError(
                f"could not load local model {choice.model_id!r}: {exc}. Check the id on "
                "Hugging Face and that the runtime can reach it."
            ) from exc
        except (RuntimeError, torch.OutOfMemoryError) as exc:
            if _is_out_of_memory(exc):
                raise ResourceError(self._memory_advice(exc)) from exc
            raise

    def _memory_advice(self, error: BaseException) -> str:
        return (
            f"{self.choice.model_id} does not fit on {self.choice.device} ({error}). "
            "Set vlm.model / llm.model to a smaller checkpoint, e.g. "
            "Qwen/Qwen3-VL-2B-Instruct, or lower local_models.image_max_side_px."
        )

    def generate(
        self,
        *,
        system: str,
        content: list[dict[str, Any]],
        max_new_tokens: int,
        temperature: float = 0.0,
    ) -> str:
        """``content`` holds ``{"type": "text"|"image", ...}`` parts, in order."""
        torch = self._torch
        messages = self._messages(system, content)
        sampling = temperature > 0
        options: dict[str, Any] = {"max_new_tokens": max_new_tokens, "do_sample": sampling}
        if sampling:
            options["temperature"] = temperature
        else:
            # Clear the checkpoint's shipped sampling settings so greedy decoding
            # is really greedy (and transformers does not warn about them).
            options.update(temperature=None, top_p=None, top_k=None)

        try:
            # Moving the image tensors to the device can exhaust memory just as
            # generation can, so both are inside the OOM handling.
            inputs = self._processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt",
            ).to(self._model.device)
            with torch.inference_mode():
                output = self._model.generate(**inputs, **options)
        except (RuntimeError, torch.OutOfMemoryError) as exc:
            if _is_out_of_memory(exc):
                raise ResourceError(self._memory_advice(exc)) from exc
            raise
        new_tokens = output[:, inputs["input_ids"].shape[1]:]
        return self._processor.batch_decode(new_tokens, skip_special_tokens=True)[0]

    def _messages(self, system: str, content: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if self.multimodal:
            return [
                {"role": "system", "content": [{"type": "text", "text": system}]},
                {"role": "user", "content": content},
            ]
        if any(part["type"] != "text" for part in content):
            raise UnsupportedError(
                f"{self.choice.model_id} is a text-only model and cannot read images; "
                "use a vision-language checkpoint for the VLM stage"
            )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": "\n\n".join(part["text"] for part in content)},
        ]

    def describe(self) -> dict[str, Any]:
        return {**self.choice.to_dict(), "dtype": self.dtype, "multimodal": self.multimodal}
