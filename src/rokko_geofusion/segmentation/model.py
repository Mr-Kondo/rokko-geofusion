"""Segmentation model adapters.

Adapters are swappable: anything implementing :class:`SegmentationModel` can be
plugged in. The shipped adapter wraps a HuggingFace semantic-segmentation
checkpoint and folds its vocabulary into the project classes.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

import numpy as np

from rokko_geofusion.config import Config
from rokko_geofusion.exceptions import ResourceError, UnsupportedError
from rokko_geofusion.segmentation.classes import build_class_mapping, mapping_summary

logger = logging.getLogger(__name__)


@runtime_checkable
class SegmentationModel(Protocol):
    """Minimal contract every segmentation backend must satisfy."""

    class_names: Sequence[str]
    device: str

    def predict_proba(self, batch: np.ndarray) -> np.ndarray:
        """``(B, 3, H, W) uint8`` -> ``(B, C, H, W) float32`` probabilities."""

    def describe(self) -> dict[str, Any]:
        """Provenance for the metadata sidecar."""


class HuggingFaceSegmenter:
    """Wraps a HF ``AutoModelForSemanticSegmentation`` checkpoint.

    The checkpoint's own vocabulary is grouped into the project classes by
    label name (see :mod:`rokko_geofusion.segmentation.classes`), so the
    probability of "project class = vegetation" is the summed probability of
    every source label that means vegetation.
    """

    def __init__(
        self,
        model_id: str,
        class_names: Sequence[str],
        *,
        device: str = "cpu",
        label_overrides: Mapping[int, str] | None = None,
    ) -> None:
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise UnsupportedError(
                "semantic segmentation needs torch + transformers: "
                "pip install -e '.[ml]'"
            ) from exc

        self._torch = torch
        self.model_id = model_id
        self.class_names = list(class_names)
        self.device = device

        logger.info("loading segmentation checkpoint %s on %s", model_id, device)
        self._processor = AutoImageProcessor.from_pretrained(model_id)
        self._model = AutoModelForSemanticSegmentation.from_pretrained(model_id)
        self._model.eval().to(device)

        declared = {int(key): str(value) for key, value in self._model.config.id2label.items()}
        id2label = dict(declared)
        self.label_overrides = dict(label_overrides or {})
        if self.label_overrides:
            unknown = set(self.label_overrides) - set(declared)
            if unknown:
                raise UnsupportedError(
                    f"segmentation.label_overrides refers to class indices {sorted(unknown)} "
                    f"that {model_id} does not have (it declares {sorted(declared)})"
                )
            id2label.update({int(k): str(v) for k, v in self.label_overrides.items()})
            logger.warning(
                "overriding the checkpoint's declared labels for %s: %s -> %s. "
                "Do this only with measured evidence (see README 'Segmentation').",
                model_id,
                {k: declared[k] for k in sorted(self.label_overrides)},
                {k: id2label[k] for k in sorted(self.label_overrides)},
            )
        self._declared_labels = declared
        self._mapping = build_class_mapping(id2label, self.class_names)
        self._mapping_summary = mapping_summary(id2label, self._mapping, self.class_names)
        self._group_index = torch.tensor(
            [self._mapping[i] for i in sorted(id2label)], dtype=torch.long, device=device
        )
        self._n_source = len(id2label)
        logger.info("mapped %d source labels onto %d project classes",
                    self._n_source, len(self.class_names))

    # -- inference -----------------------------------------------------------
    def predict_proba(self, batch: np.ndarray) -> np.ndarray:
        torch = self._torch
        if batch.ndim != 4 or batch.shape[1] != 3:
            raise ValueError(f"expected (B, 3, H, W) uint8, got {batch.shape}")
        height, width = batch.shape[2], batch.shape[3]

        images = [np.transpose(tile, (1, 2, 0)) for tile in batch]
        inputs = self._processor(images=images, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        try:
            with torch.no_grad():
                logits = self._model(**inputs).logits
                logits = torch.nn.functional.interpolate(
                    logits, size=(height, width), mode="bilinear", align_corners=False
                )
                probabilities = torch.softmax(logits, dim=1)
                grouped = torch.zeros(
                    (probabilities.shape[0], len(self.class_names), height, width),
                    dtype=probabilities.dtype,
                    device=probabilities.device,
                )
                index = self._group_index.view(1, -1, 1, 1).expand_as(probabilities)
                grouped.scatter_add_(1, index, probabilities)
        except RuntimeError as exc:
            if _is_out_of_memory(exc):
                raise ResourceError(f"out of accelerator memory: {exc}") from exc
            raise
        return grouped.float().cpu().numpy()

    def describe(self) -> dict[str, Any]:
        description: dict[str, Any] = {
            "backend": "huggingface",
            "model_id": self.model_id,
            "device": self.device,
            "source_classes": self._n_source,
            "project_classes": list(self.class_names),
            "label_mapping": self._mapping_summary,
            "domain_note": (
                "Pretrained semantic segmentation applied to an orthophoto. No "
                "accuracy figure is claimed for this ROI; treat the classes as "
                "indicative and cross-check them against independent geometry "
                "(scripts/validate.py --check segmentation)."
            ),
        }
        if self.label_overrides:
            description["declared_labels"] = self._declared_labels
            description["label_overrides"] = self.label_overrides
            description["label_override_note"] = (
                "The checkpoint's published id2label did not match the class "
                "indices its head learned; the corrected mapping is configured in "
                "segmentation.label_overrides and was established by scoring each "
                "candidate against independent OSM geometry."
            )
        return description


def _is_out_of_memory(error: BaseException) -> bool:
    text = str(error).lower()
    return "out of memory" in text or "cuda oom" in text or "mps backend out of memory" in text


def load_segmenter(config: Config, device: str) -> SegmentationModel:
    """Instantiate the configured segmentation backend."""
    provider = config.segmentation.provider
    if provider == "none":
        raise UnsupportedError("segmentation.provider is 'none'")
    if provider == "huggingface":
        return HuggingFaceSegmenter(
            config.segmentation.model_id,
            config.segmentation.classes,
            device=device,
            label_overrides=config.segmentation.label_overrides,
        )
    raise UnsupportedError(f"unknown segmentation provider {provider!r}")
