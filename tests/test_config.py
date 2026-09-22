"""Configuration loading, overriding and validation."""

from __future__ import annotations

import pytest

from rokko_geofusion.config import Config, apply_overrides, load_config
from rokko_geofusion.exceptions import ConfigError


def test_default_config_loads(default_config_path):
    cfg = load_config(default_config_path)
    assert cfg.project.name == "rokko_geofusion"
    assert cfg.roi.center is not None
    assert cfg.roi.center.lat == pytest.approx(34.7284)
    assert cfg.roi.center.lon == pytest.approx(135.2348)
    assert cfg.roi.radius_m == pytest.approx(1000.0)
    assert cfg.crs.geographic == "EPSG:4326"
    assert cfg.crs.projected == "EPSG:6673"


def test_no_epsg_codes_are_hardcoded_in_processing_modules(repo_root):
    """EPSG codes must come from config, not from literals in processing code.

    Docstrings and comments may mention them (they are documentation);
    ``config.py`` declares the defaults. Anything else is a hard-coded CRS.
    """
    import ast
    import io
    import tokenize

    offenders = []
    for path in sorted((repo_root / "src").rglob("*.py")):
        if path.name == "config.py":
            continue
        source = path.read_text(encoding="utf-8")

        docstring_lines: set[int] = set()
        tree = ast.parse(source, str(path))
        for node in ast.walk(tree):
            if not isinstance(
                node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
            ):
                continue
            body = getattr(node, "body", None)
            if not body:
                continue
            first = body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) \
                    and isinstance(first.value.value, str):
                docstring_lines.update(range(first.lineno, (first.end_lineno or first.lineno) + 1))

        for token in tokenize.generate_tokens(io.StringIO(source).readline):
            if token.type != tokenize.STRING or "EPSG:" not in token.string:
                continue
            if token.start[0] in docstring_lines:
                continue
            offenders.append(
                f"{path.relative_to(repo_root)}:{token.start[0]}: {token.string.strip()}"
            )
    assert not offenders, "hard-coded EPSG codes outside config.py:\n" + "\n".join(offenders)


def test_roi_key_is_deterministic_and_safe(default_config_path):
    a = load_config(default_config_path).roi.key
    b = load_config(default_config_path).roi.key
    assert a == b == "rokkodai_34.72840_135.23480_r1000"
    assert "/" not in a and " " not in a


def test_roi_key_changes_with_roi(default_config_path):
    base = load_config(default_config_path)
    moved = load_config(default_config_path, overrides=["roi.radius_m=500"])
    assert base.roi.key != moved.roi.key


def test_override_applies_nested_value(default_config_path):
    cfg = load_config(
        default_config_path,
        overrides=["roi.radius_m=250", "segmentation.enabled=false", "imagery.zoom=17"],
    )
    assert cfg.roi.radius_m == 250
    assert cfg.segmentation.enabled is False
    assert cfg.imagery.zoom == 17


def test_override_requires_equals():
    with pytest.raises(ConfigError):
        apply_overrides({}, ["roi.radius_m"])


def test_unknown_key_is_rejected(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "roi:\n  center: {lat: 34.7, lon: 135.2}\n  radius_m: 100\nnot_a_section: 1\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError):
        load_config(bad)


def test_missing_file_raises(tmp_path):
    with pytest.raises(ConfigError):
        load_config(tmp_path / "nope.yaml")


def test_roi_requires_center_or_bbox(tmp_path):
    bad = tmp_path / "roi.yaml"
    bad.write_text("roi:\n  name: x\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(bad)


def test_bbox_roi_is_accepted(tmp_path):
    good = tmp_path / "roi.yaml"
    good.write_text(
        "roi:\n  name: bb\n  bbox: [135.20, 34.71, 135.26, 34.75]\n", encoding="utf-8"
    )
    cfg = load_config(good)
    assert cfg.roi.bbox == [135.20, 34.71, 135.26, 34.75]
    assert cfg.roi.key.startswith("bb_bbox_")


def test_degenerate_bbox_is_rejected(tmp_path):
    bad = tmp_path / "roi.yaml"
    bad.write_text("roi:\n  bbox: [135.26, 34.71, 135.20, 34.75]\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(bad)


def test_env_expansion(tmp_path, monkeypatch):
    monkeypatch.setenv("RGF_TEST_MODEL", "some-model-id")
    path = tmp_path / "env.yaml"
    path.write_text(
        "roi:\n  center: {lat: 34.7, lon: 135.2}\n"
        "llm:\n  model: ${RGF_TEST_MODEL}\n"
        "vlm:\n  model: ${RGF_UNSET_VAR:-fallback-model}\n",
        encoding="utf-8",
    )
    cfg = load_config(path)
    assert cfg.llm.model == "some-model-id"
    assert cfg.vlm.model == "fallback-model"


def test_fingerprint_is_stable_and_sensitive(default_config_path):
    a = load_config(default_config_path)
    b = load_config(default_config_path)
    c = load_config(default_config_path, overrides=["roi.radius_m=999"])
    assert a.fingerprint() == b.fingerprint()
    assert a.fingerprint() != c.fingerprint()


def test_roundtrip_through_yaml(default_config_path, tmp_path):
    cfg = load_config(default_config_path)
    out = cfg.save(tmp_path / "dump.yaml")
    again = load_config(out)
    assert again.fingerprint() == cfg.fingerprint()


def test_paths_are_roi_scoped(config):
    paths = config.paths
    assert paths.roi_key == config.roi.key
    assert paths.out_root.name == config.roi.key
    assert paths.interim.name == config.roi.key
    paths.ensure()
    for directory in paths.all_dirs():
        assert directory.is_dir()


def test_effective_output_crs_defaults_to_projected(default_config_path):
    cfg = load_config(default_config_path)
    assert cfg.crs.effective_output == cfg.crs.projected
    cfg2 = load_config(default_config_path, overrides=["crs.output=EPSG:4326"])
    assert cfg2.crs.effective_output == "EPSG:4326"


def test_config_model_is_constructible_without_yaml():
    cfg = Config(roi={"center": {"lat": 34.7284, "lon": 135.2348}, "radius_m": 300})
    assert cfg.lidar.dem.provider == "gsi_tile"
    assert cfg.lidar.dsm.provider == "none"
