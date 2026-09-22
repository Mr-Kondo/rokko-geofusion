"""Metadata sidecars for every intermediate and output artefact.

Any file we write gets a ``<name>.meta.json`` neighbour recording CRS, source,
resolution, acquisition info and processing parameters. Without this, a
GeoTIFF in ``data/interim`` is an anonymous grid of numbers six months later.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SIDECAR_SUFFIX = ".meta.json"
SCHEMA_VERSION = 1


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (set, tuple)):
        return list(value)
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return str(value)


def sidecar_path(target: Path | str) -> Path:
    return Path(str(target) + SIDECAR_SUFFIX)


def build_metadata(
    *,
    kind: str,
    source: str,
    crs: str | None = None,
    resolution_m: float | None = None,
    roi: dict[str, Any] | None = None,
    acquisition: dict[str, Any] | None = None,
    processing: dict[str, Any] | None = None,
    config_fingerprint: str | None = None,
    notes: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """Assemble the standard metadata dictionary."""
    from rokko_geofusion import __version__

    metadata: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": kind,
        "source": source,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "producer": f"rokko-geofusion/{__version__}",
    }
    if crs is not None:
        metadata["crs"] = crs
    if resolution_m is not None:
        metadata["resolution_m"] = resolution_m
    if roi is not None:
        metadata["roi"] = roi
    if acquisition is not None:
        metadata["acquisition"] = acquisition
    if processing is not None:
        metadata["processing"] = processing
    if config_fingerprint is not None:
        metadata["config_fingerprint"] = config_fingerprint
    if notes:
        metadata["notes"] = notes
    metadata.update(extra)
    return metadata


def write_sidecar(target: Path | str, metadata: dict[str, Any]) -> Path:
    """Write ``metadata`` next to ``target`` and return the sidecar path."""
    path = sidecar_path(target)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    logger.debug("wrote metadata sidecar %s", path)
    return path


def read_sidecar(target: Path | str) -> dict[str, Any] | None:
    """Read the sidecar for ``target``; ``None`` when it does not exist."""
    path = sidecar_path(target)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        logger.warning("metadata sidecar %s is not valid JSON: %s", path, exc)
        return None


def write_json(target: Path | str, payload: Any, *, indent: int = 2) -> Path:
    """Write any JSON artefact (metrics, reports) with consistent formatting."""
    path = Path(target)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=indent, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    logger.info("wrote %s", path)
    return path


def read_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))
