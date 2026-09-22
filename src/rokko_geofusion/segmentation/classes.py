"""Mapping a general-purpose segmentation vocabulary onto the project classes.

Labels are matched **by name**, never by hard-coded class index, so the same
code serves a remote-sensing checkpoint (LoveDA: background / building / road /
water / barren / forest / agriculture) and a general-purpose one (ADE20K: 150
ground-level classes). Swapping `segmentation.model_id` is enough.

The groups below therefore list the vocabulary of every checkpoint we support.
A label that matches nothing falls into "other", and a project class with no
counterpart in the loaded vocabulary logs a warning instead of silently never
being predicted.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Mapping, Sequence

logger = logging.getLogger(__name__)

#: Project class -> source-vocabulary label names that should fall into it.
#: Anything unmatched becomes "other".
DEFAULT_LABEL_GROUPS: dict[str, tuple[str, ...]] = {
    "building": (
        "building", "edifice", "house", "skyscraper", "hovel", "hut", "shack",
        "tower", "booth", "kiosk", "tent", "shelter",
    ),
    "road": (
        "road", "route", "path", "sidewalk", "pavement", "runway", "bridge",
        "viaduct", "crosswalk", "street",
    ),
    "vegetation": (
        # LoveDA
        "forest", "agriculture", "agricultural", "farmland", "woodland",
        # ADE20K
        "tree", "grass", "plant", "flora", "flower", "field", "palm", "bush",
        "hedge", "shrub",
    ),
    "bare_soil": (
        # LoveDA
        "barren", "bare", "bare land", "barren land",
        # ADE20K
        "earth", "ground", "sand", "dirt track", "land", "soil", "hill",
        "mountain", "mount", "rock", "stone", "gravel",
    ),
    "water": (
        "water", "sea", "river", "lake", "pool", "swimming pool", "waterfall",
        "falls", "pond",
    ),
}

_SPLIT = re.compile(r"[;,/]")


def normalise_label(label: str) -> set[str]:
    """Split an ADE20K-style compound label into its synonyms."""
    return {part.strip().lower() for part in _SPLIT.split(label) if part.strip()}


def build_class_mapping(
    id2label: Mapping[int, str],
    project_classes: Sequence[str],
    *,
    groups: Mapping[str, Iterable[str]] | None = None,
) -> dict[int, int]:
    """Map every source class id onto a project class index.

    ``project_classes[0]`` must be the catch-all class (``other``).
    """
    if not project_classes:
        raise ValueError("project_classes must not be empty")
    groups = groups or DEFAULT_LABEL_GROUPS
    other_index = 0

    wanted: dict[str, int] = {}
    for name, synonyms in groups.items():
        if name not in project_classes:
            logger.debug("group %s is not among the configured classes; ignored", name)
            continue
        index = project_classes.index(name)
        for synonym in synonyms:
            wanted[synonym.lower()] = index

    mapping: dict[int, int] = {}
    matched_per_class: dict[int, list[str]] = {}
    for source_id, label in id2label.items():
        target = other_index
        for synonym in normalise_label(str(label)):
            if synonym in wanted:
                target = wanted[synonym]
                break
        mapping[int(source_id)] = target
        matched_per_class.setdefault(target, []).append(str(label))

    for index, name in enumerate(project_classes):
        labels = matched_per_class.get(index, [])
        if index != other_index and not labels:
            logger.warning(
                "project class %r has no counterpart in the model vocabulary; it "
                "will never be predicted", name,
            )
        else:
            logger.debug("class %-12s <- %d source label(s)", name, len(labels))
    return mapping


def mapping_summary(
    id2label: Mapping[int, str],
    mapping: Mapping[int, int],
    project_classes: Sequence[str],
) -> dict[str, list[str]]:
    """Human-readable record of the mapping, stored in the output metadata."""
    summary: dict[str, list[str]] = {name: [] for name in project_classes}
    for source_id, target in mapping.items():
        if 0 <= target < len(project_classes):
            summary[project_classes[target]].append(str(id2label[source_id]))
    return {name: sorted(labels) for name, labels in summary.items()}
