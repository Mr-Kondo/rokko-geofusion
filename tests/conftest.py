"""Shared pytest fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "rokko.yaml"


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def default_config_path() -> Path:
    return DEFAULT_CONFIG_PATH


@pytest.fixture()
def config(tmp_path: Path):
    """The shipped configuration, redirected to a temporary data/output root."""
    from rokko_geofusion.config import load_config

    cfg = load_config(
        DEFAULT_CONFIG_PATH,
        overrides=[
            f"project.data_root={tmp_path / 'data'}",
            f"project.output_root={tmp_path / 'outputs'}",
        ],
    )
    return cfg
