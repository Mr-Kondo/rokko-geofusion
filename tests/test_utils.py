"""Logging, seeding and metadata sidecars."""

from __future__ import annotations

import json
import random

import numpy as np

from rokko_geofusion.utils.logging import log_failure_context, setup_logging
from rokko_geofusion.utils.metadata import (
    build_metadata,
    read_json,
    read_sidecar,
    sidecar_path,
    write_json,
    write_sidecar,
)
from rokko_geofusion.utils.seed import set_global_seed


def test_seed_makes_rng_reproducible():
    set_global_seed(123)
    a = (random.random(), np.random.rand(3).tolist())
    set_global_seed(123)
    b = (random.random(), np.random.rand(3).tolist())
    assert a == b


def test_sidecar_roundtrip(tmp_path):
    target = tmp_path / "dem.tif"
    target.write_bytes(b"not-a-real-tif")
    metadata = build_metadata(
        kind="dem",
        source="gsi:dem5a",
        crs="EPSG:6673",
        resolution_m=5.0,
        roi={"key": "test_roi"},
        processing={"resampling": "bilinear"},
        config_fingerprint="deadbeef",
    )
    path = write_sidecar(target, metadata)
    assert path == sidecar_path(target)
    assert path.name == "dem.tif.meta.json"

    loaded = read_sidecar(target)
    assert loaded is not None
    assert loaded["kind"] == "dem"
    assert loaded["crs"] == "EPSG:6673"
    assert loaded["resolution_m"] == 5.0
    assert loaded["processing"]["resampling"] == "bilinear"
    assert loaded["schema_version"] == 1
    assert loaded["producer"].startswith("rokko-geofusion/")
    assert "created_utc" in loaded


def test_read_sidecar_missing_returns_none(tmp_path):
    assert read_sidecar(tmp_path / "absent.tif") is None


def test_write_json_handles_paths_and_numpy(tmp_path):
    target = tmp_path / "metrics" / "stats.json"
    write_json(target, {"path": tmp_path, "values": [1, 2, 3]})
    payload = read_json(target)
    assert payload["values"] == [1, 2, 3]
    assert isinstance(payload["path"], str)


def test_log_failure_context_emits_all_fields(capsys):
    # `setup_logging` installs its own handlers on the root logger (and removes
    # pytest's), so the output is asserted on stderr rather than via caplog.
    logger = setup_logging("DEBUG", force=True).getChild("test")
    log_failure_context(
        logger,
        what="downloading DEM tile",
        target="z15/28689/13006",
        roi="rokkodai",
        crs="EPSG:3857",
        cause="upstream tile scheme changed",
    )
    text = capsys.readouterr().err
    assert "FAILED: downloading DEM tile" in text
    assert "28689" in text
    assert "EPSG:3857" in text
    assert "upstream tile scheme changed" in text


def test_metadata_json_is_utf8_readable(tmp_path):
    target = tmp_path / "ort.tif"
    write_sidecar(target, build_metadata(kind="imagery", source="国土地理院"))
    raw = sidecar_path(target).read_text(encoding="utf-8")
    assert "国土地理院" in raw
    assert json.loads(raw)["source"] == "国土地理院"
