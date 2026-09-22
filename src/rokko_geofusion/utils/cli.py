"""Shared argparse plumbing so every script in ``scripts/`` behaves the same."""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

from rokko_geofusion.config import Config, load_config
from rokko_geofusion.environment import ResourceProfile, make_resource_profile
from rokko_geofusion.paths import Paths, find_project_root
from rokko_geofusion.utils.logging import setup_logging
from rokko_geofusion.utils.seed import set_global_seed

DEFAULT_CONFIG = "configs/rokko.yaml"


def build_parser(description: str) -> argparse.ArgumentParser:
    """Base parser with the options every stage script accepts."""
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help="path to the YAML configuration file",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        metavar="KEY=VALUE",
        help="override a config value, e.g. --set roi.radius_m=500 (repeatable)",
    )
    parser.add_argument(
        "--log-level",
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="logging verbosity (default: runtime.log_level from the config)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="regenerate artefacts even when they already exist",
    )
    return parser


@dataclass(frozen=True)
class StageContext:
    """Everything a stage script needs after start-up."""

    config: Config
    paths: Paths
    profile: ResourceProfile
    logger: logging.Logger
    args: argparse.Namespace

    @property
    def overwrite(self) -> bool:
        return bool(getattr(self.args, "overwrite", False)) or self.config.output.overwrite


def init_stage(
    args: argparse.Namespace,
    *,
    stage_name: str,
    need_profile: bool = False,
) -> StageContext:
    """Load config, configure logging, seed RNGs and create directories."""
    config_path = Path(args.config)
    if not config_path.is_absolute():
        # Allow running scripts from anywhere inside the repository.
        candidate = find_project_root() / config_path
        if candidate.is_file():
            config_path = candidate

    config = load_config(config_path, overrides=getattr(args, "overrides", None))
    level = args.log_level or config.runtime.log_level
    paths = config.paths.ensure()
    logger = setup_logging(level, log_file=paths.logs / f"{stage_name}.log", force=True)
    logger = logging.getLogger(f"rokko_geofusion.{stage_name}")

    set_global_seed(config.project.seed)
    profile = make_resource_profile(config) if need_profile else _null_profile()

    logger.info("stage=%s config=%s fingerprint=%s", stage_name, config_path,
                config.fingerprint())
    logger.info("roi=%s output=%s", config.roi.key, paths.out_root)
    return StageContext(config=config, paths=paths, profile=profile, logger=logger, args=args)


def _null_profile() -> ResourceProfile:
    """Placeholder profile for stages that never touch the GPU."""
    return ResourceProfile(
        device="cpu",
        tier="not-probed",
        seg_tile_px=0,
        seg_batch_size=0,
        pc_num_points=0,
        pc_batch_size=0,
        max_points_in_memory=0,
        http_max_workers=8,
        display_points=0,
    )
