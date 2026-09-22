"""Rule-based fusion of image semantics, object height and mapped geometry.

The rules are deliberately explicit and auditable: every cell records *which*
rule fired (``rule`` column), so a fused class can always be traced back to the
evidence that produced it. Thresholds come from the configuration.

When no DSM exists, height-dependent conditions cannot be evaluated. They are
then **skipped and recorded as skipped** -- not silently treated as satisfied
and not treated as failed.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from rokko_geofusion.config import FusionConfig

logger = logging.getLogger(__name__)

#: Name of the rule that fired, per fused class, for the audit trail.
RULE_NAMES = (
    "none",
    "image_water",
    "image_building_tall",
    "image_building_no_height",
    "gis_building",
    "image_vegetation_tall",
    "image_vegetation_low",
    "image_vegetation_no_height",
    "image_road_low",
    "image_road_no_height",
    "gis_road",
    "image_bare_soil",
    "image_other_low_confidence",
)


@dataclass(frozen=True)
class FusionInputs:
    """Per-cell evidence arrays; all 1-D and the same length."""

    image_class: np.ndarray          # index into the segmentation vocabulary
    image_confidence: np.ndarray
    object_height: np.ndarray | None  # metres above ground, or None when no DSM
    in_building: np.ndarray | None    # mapped building footprint
    on_road: np.ndarray | None        # mapped road corridor
    in_water: np.ndarray | None       # mapped water body

    def size(self) -> int:
        return int(self.image_class.size)


def apply_rules(
    inputs: FusionInputs,
    *,
    config: FusionConfig,
    image_classes: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Return ``(fused_class_index, rule_index, provenance)``.

    Rules are applied in priority order; a cell keeps the first class assigned
    to it, so earlier rules dominate.
    """
    thresholds = config.thresholds
    fused_names = list(config.classes)
    n = inputs.size()

    fused = np.zeros(n, dtype=np.uint8)          # 0 == unknown
    rule = np.zeros(n, dtype=np.uint8)           # 0 == none
    assigned = np.zeros(n, dtype=bool)

    def class_index(name: str) -> int:
        return fused_names.index(name)

    def rule_index(name: str) -> int:
        return RULE_NAMES.index(name)

    def image_is(name: str) -> np.ndarray:
        if name not in image_classes:
            return np.zeros(n, dtype=bool)
        return inputs.image_class == image_classes.index(name)

    confident = inputs.image_confidence >= thresholds.min_image_confidence
    have_height = inputs.object_height is not None
    height = inputs.object_height if have_height else None
    height_known = (np.isfinite(height) if have_height else np.zeros(n, dtype=bool))

    def assign(mask: np.ndarray, class_name: str, rule_name: str) -> int:
        nonlocal fused, rule, assigned
        target = mask & ~assigned
        count = int(target.sum())
        if count:
            fused[target] = class_index(class_name)
            rule[target] = rule_index(rule_name)
            assigned |= target
        return count

    counts: dict[str, int] = {}
    skipped: list[str] = []

    # 1. Water: image evidence first, then mapped water bodies.
    counts["image_water"] = assign(image_is("water") & confident, "water", "image_water")
    if config.use_gis_evidence and inputs.in_water is not None:
        counts["gis_water"] = assign(inputs.in_water, "water", "image_water")

    # 2. Buildings: "image says building AND it stands above the ground".
    building_image = image_is("building") & confident
    if have_height:
        tall = height_known & (height >= thresholds.building_min_height_m)
        counts["image_building_tall"] = assign(
            building_image & tall, "building", "image_building_tall"
        )
        counts["image_building_no_height"] = assign(
            building_image & ~height_known, "building", "image_building_no_height"
        )
    else:
        skipped.append(
            f"building height >= {thresholds.building_min_height_m} m "
            "(no DSM, so object height is unknown)"
        )
        counts["image_building_no_height"] = assign(
            building_image, "building", "image_building_no_height"
        )
    if config.use_gis_evidence and inputs.in_building is not None:
        counts["gis_building"] = assign(inputs.in_building, "building", "gis_building")

    # 3. Vegetation: split by height when a DSM exists.
    vegetation_image = image_is("vegetation") & confident
    if have_height:
        tall = height_known & (height >= thresholds.tall_vegetation_min_height_m)
        counts["image_vegetation_tall"] = assign(
            vegetation_image & tall, "tall_vegetation", "image_vegetation_tall"
        )
        counts["image_vegetation_low"] = assign(
            vegetation_image & height_known, "low_vegetation", "image_vegetation_low"
        )
        counts["image_vegetation_no_height"] = assign(
            vegetation_image, "low_vegetation", "image_vegetation_no_height"
        )
    else:
        skipped.append(
            f"vegetation height >= {thresholds.tall_vegetation_min_height_m} m "
            "(no DSM, so tall and low vegetation cannot be separated)"
        )
        counts["image_vegetation_no_height"] = assign(
            vegetation_image, "low_vegetation", "image_vegetation_no_height"
        )

    # 4. Roads: image evidence, then mapped road corridors.
    road_image = image_is("road") & confident
    if have_height:
        low = height_known & (height <= thresholds.road_max_height_m)
        counts["image_road_low"] = assign(road_image & low, "road", "image_road_low")
        counts["image_road_no_height"] = assign(
            road_image & ~height_known, "road", "image_road_no_height"
        )
    else:
        skipped.append(
            f"road height <= {thresholds.road_max_height_m} m "
            "(no DSM, so bridges and overpasses cannot be distinguished)"
        )
        counts["image_road_no_height"] = assign(road_image, "road", "image_road_no_height")
    if config.use_gis_evidence and inputs.on_road is not None:
        counts["gis_road"] = assign(inputs.on_road, "road", "gis_road")

    # 5. Bare soil, then everything the image saw but we do not trust.
    counts["image_bare_soil"] = assign(
        image_is("bare_soil") & confident, "bare_soil", "image_bare_soil"
    )
    counts["image_other_low_confidence"] = assign(
        ~assigned & (inputs.image_confidence > 0), "other", "image_other_low_confidence"
    )

    provenance = {
        "rule_counts": {name: value for name, value in counts.items() if value},
        "skipped_conditions": skipped,
        "thresholds": thresholds.model_dump(),
        "used_gis_evidence": bool(config.use_gis_evidence),
        "height_available": have_height,
        "unassigned": int((~assigned).sum()),
    }
    return fused, rule, provenance


def class_fractions(fused: np.ndarray, class_names: Sequence[str]) -> dict[str, float]:
    counts = np.bincount(fused, minlength=len(class_names))
    total = max(int(fused.size), 1)
    return {name: float(count / total) for name, count in zip(class_names, counts, strict=True)}
