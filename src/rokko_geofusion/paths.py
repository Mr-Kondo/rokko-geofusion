"""Filesystem layout.

Every artefact of a run lives under a deterministic, ROI-scoped directory so
that two different ROIs never overwrite each other and re-running the same ROI
reproduces the same paths (validation item V6).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

_ROOT_MARKERS = ("pyproject.toml", ".git")


def find_project_root(start: Path | str | None = None) -> Path:
    """Walk upwards from ``start`` looking for a repository root marker.

    Falls back to the current working directory, which is what happens in a
    Colab cell that ``cd``-ed into the cloned repository.
    """
    current = Path(start or Path.cwd()).resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if any((candidate / marker).exists() for marker in _ROOT_MARKERS):
            return candidate
    logger.debug("no project root marker found above %s; using cwd", current)
    return Path.cwd().resolve()


def resolve_path(path: Path | str, root: Path) -> Path:
    """Resolve ``path`` against ``root`` unless it is already absolute."""
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else (root / candidate).resolve()


@dataclass(frozen=True)
class Paths:
    """Concrete directories for one (config, ROI) pair."""

    root: Path
    data_root: Path
    output_root: Path
    roi_key: str

    # --- raw / interim / processed -----------------------------------------
    @property
    def raw(self) -> Path:
        return self.data_root / "raw"

    @property
    def interim(self) -> Path:
        return self.data_root / "interim" / self.roi_key

    @property
    def processed(self) -> Path:
        return self.data_root / "processed" / self.roi_key

    @property
    def cache(self) -> Path:
        """Shared, ROI-independent HTTP/tile cache (raw bytes keyed by URL)."""
        return self.data_root / "raw" / "_cache"

    # --- outputs ------------------------------------------------------------
    @property
    def out_root(self) -> Path:
        return self.output_root / self.roi_key

    @property
    def pointcloud(self) -> Path:
        return self.out_root / "pointcloud"

    @property
    def raster(self) -> Path:
        return self.out_root / "raster"

    @property
    def vector(self) -> Path:
        return self.out_root / "vector"

    @property
    def figures(self) -> Path:
        return self.out_root / "figures"

    @property
    def metrics(self) -> Path:
        return self.out_root / "metrics"

    @property
    def reports(self) -> Path:
        return self.out_root / "reports"

    @property
    def logs(self) -> Path:
        return self.out_root / "logs"

    def all_dirs(self) -> tuple[Path, ...]:
        return (
            self.raw,
            self.interim,
            self.processed,
            self.cache,
            self.pointcloud,
            self.raster,
            self.vector,
            self.figures,
            self.metrics,
            self.reports,
            self.logs,
        )

    def ensure(self) -> Paths:
        """Create every directory. Safe to call repeatedly."""
        for directory in self.all_dirs():
            directory.mkdir(parents=True, exist_ok=True)
        return self
