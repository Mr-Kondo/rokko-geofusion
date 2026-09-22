"""Stage orchestration.

Each stage is the very same ``scripts/*.py`` entry point a user would run by
hand, loaded and called in-process. Nothing is reimplemented here, so
``--stage all`` and ``--stage fusion`` cannot drift apart from each other or
from the single-stage debugging path.
"""

from __future__ import annotations

import importlib.util
import logging
import sys
import time
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from rokko_geofusion.exceptions import GeoFusionError, UnsupportedError
from rokko_geofusion.paths import find_project_root

logger = logging.getLogger(__name__)

#: Stage name -> script file, in dependency order.
STAGE_SCRIPTS: dict[str, str] = {
    "environment": "check_environment.py",
    "lidar": "download_lidar.py",
    "imagery": "download_imagery.py",
    "gis": "download_gis.py",
    "pointcloud": "colorize_pointcloud.py",
    "terrain": "build_terrain.py",
    "segmentation": "segment_imagery.py",
    "fusion": "fuse_modalities.py",
    "pointcloud_ml": "run_pointcloud_ml.py",
    "vlm": "run_vlm.py",
    "llm": "run_llm.py",
    "validate": "validate.py",
    "report": "generate_report_data.py",
}

#: The order ``--stage all`` runs. ``report`` comes last so it sees everything.
DEFAULT_ORDER: tuple[str, ...] = (
    "lidar", "imagery", "gis", "pointcloud", "terrain", "segmentation",
    "fusion", "pointcloud_ml", "vlm", "llm", "validate", "report",
)

#: Stages that need no config file (they take their own arguments).
_NO_CONFIG_STAGES = frozenset({"environment"})


@dataclass
class StageOutcome:
    stage: str
    status: str            # "ok" | "failed" | "skipped"
    exit_code: int | None
    seconds: float
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PipelineResult:
    stages: list[StageOutcome] = field(default_factory=list)

    @property
    def failed(self) -> list[str]:
        return [outcome.stage for outcome in self.stages if outcome.status == "failed"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "stages": [outcome.to_dict() for outcome in self.stages],
            "failed": self.failed,
        }


def scripts_dir(root: Path | None = None) -> Path:
    return (root or find_project_root()) / "scripts"


def _load_script(name: str, root: Path | None = None):
    """Import a ``scripts/*.py`` file as a module, once per process."""
    if name not in STAGE_SCRIPTS:
        raise UnsupportedError(
            f"unknown stage {name!r}; known stages: {', '.join(STAGE_SCRIPTS)}"
        )
    module_name = f"rokko_geofusion_stage_{name}"
    if module_name in sys.modules:
        return sys.modules[module_name]

    path = scripts_dir(root) / STAGE_SCRIPTS[name]
    if not path.is_file():
        raise UnsupportedError(f"stage script not found: {path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:  # pragma: no cover - import machinery
        raise UnsupportedError(f"could not load stage script {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def resolve_stages(requested: Sequence[str] | None) -> list[str]:
    """Expand ``["all"]`` and validate the requested stage names."""
    if not requested or list(requested) == ["all"]:
        return list(DEFAULT_ORDER)
    resolved: list[str] = []
    for name in requested:
        if name == "all":
            resolved.extend(stage for stage in DEFAULT_ORDER if stage not in resolved)
            continue
        if name not in STAGE_SCRIPTS:
            raise UnsupportedError(
                f"unknown stage {name!r}; known stages: {', '.join(STAGE_SCRIPTS)}"
            )
        if name not in resolved:
            resolved.append(name)
    return resolved


def run_stage(
    stage: str,
    *,
    config_path: Path | str,
    overrides: Iterable[str] | None = None,
    extra_args: Sequence[str] | None = None,
    overwrite: bool = False,
    log_level: str | None = None,
    root: Path | None = None,
) -> StageOutcome:
    """Run one stage in-process and report how it went."""
    module = _load_script(stage, root)
    argv: list[str] = []
    if stage not in _NO_CONFIG_STAGES:
        argv += ["--config", str(config_path)]
        for override in overrides or []:
            argv += ["--set", override]
        if overwrite:
            argv.append("--overwrite")
        if log_level:
            argv += ["--log-level", log_level]
    else:
        argv += ["--config", str(config_path)]
    argv += list(extra_args or [])

    logger.info("=== stage %s ===", stage)
    started = time.perf_counter()
    try:
        exit_code = int(module.main(argv) or 0)
    except GeoFusionError as exc:
        seconds = time.perf_counter() - started
        logger.error("stage %s failed: %s", stage, exc)
        return StageOutcome(stage, "failed", None, seconds, str(exc))
    except Exception as exc:  # noqa: BLE001 - a stage must not kill the pipeline
        seconds = time.perf_counter() - started
        logger.exception("stage %s raised an unexpected error", stage)
        return StageOutcome(stage, "failed", None, seconds, f"{type(exc).__name__}: {exc}")

    seconds = time.perf_counter() - started
    status = "ok" if exit_code == 0 else "failed"
    logger.info("stage %s %s in %.1f s (exit %d)", stage, status, seconds, exit_code)
    return StageOutcome(stage, status, exit_code, seconds)


def run_pipeline(
    stages: Sequence[str] | None,
    *,
    config_path: Path | str,
    overrides: Iterable[str] | None = None,
    overwrite: bool = False,
    stop_on_error: bool = False,
    log_level: str | None = None,
    root: Path | None = None,
) -> PipelineResult:
    """Run several stages in order."""
    result = PipelineResult()
    overrides = list(overrides or [])
    for stage in resolve_stages(stages):
        outcome = run_stage(
            stage, config_path=config_path, overrides=overrides,
            overwrite=overwrite, log_level=log_level, root=root,
        )
        result.stages.append(outcome)
        if outcome.status == "failed" and stop_on_error:
            logger.error("stopping: stage %s failed and --stop-on-error is set", stage)
            break
    return result
