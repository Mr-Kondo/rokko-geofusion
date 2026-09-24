"""Stage orchestration, the interactive API, and the notebook discipline rule."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from rokko_geofusion.exceptions import UnsupportedError
from rokko_geofusion.interactive import _roi_overrides, summarize_subregion
from rokko_geofusion.pipeline import (
    DEFAULT_ORDER,
    STAGE_SCRIPTS,
    PipelineResult,
    StageOutcome,
    resolve_stages,
    run_stage,
    scripts_dir,
)


# --- stage registry ---------------------------------------------------------
def test_every_registered_stage_script_exists(repo_root):
    for stage, filename in STAGE_SCRIPTS.items():
        path = scripts_dir(repo_root) / filename
        assert path.is_file(), f"stage {stage} points at a missing script: {path}"


def test_every_stage_script_is_registered(repo_root):
    """A new script must be wired into the pipeline, not left orphaned."""
    on_disk = {path.name for path in (repo_root / "scripts").glob("*.py")}
    registered = set(STAGE_SCRIPTS.values())
    assert on_disk - registered - {"run_pipeline.py"} == set()


def test_default_order_covers_the_processing_stages():
    assert set(DEFAULT_ORDER) <= set(STAGE_SCRIPTS)
    # `environment` is a diagnostic, not part of a run.
    assert "environment" not in DEFAULT_ORDER
    # Dependencies must come before their consumers.
    order = list(DEFAULT_ORDER)
    assert order.index("lidar") < order.index("pointcloud") < order.index("fusion")
    assert order.index("terrain") < order.index("fusion")
    assert order.index("segmentation") < order.index("fusion")
    assert order.index("fusion") < order.index("pointcloud_ml")
    assert order.index("vlm") < order.index("llm")
    assert order[-1] == "report"


def test_resolve_stages_expands_all():
    assert resolve_stages(None) == list(DEFAULT_ORDER)
    assert resolve_stages(["all"]) == list(DEFAULT_ORDER)


def test_resolve_stages_keeps_order_and_deduplicates():
    assert resolve_stages(["terrain", "lidar", "terrain"]) == ["terrain", "lidar"]


def test_resolve_stages_rejects_unknown_names():
    with pytest.raises(UnsupportedError, match="unknown stage"):
        resolve_stages(["teleport"])


def test_run_stage_reports_a_failure_without_raising(tmp_path, repo_root):
    outcome = run_stage("terrain", config_path=tmp_path / "missing.yaml", root=repo_root)
    assert outcome.status == "failed"
    assert outcome.error and "config file not found" in outcome.error
    assert outcome.seconds >= 0


def test_pipeline_result_collects_failures():
    result = PipelineResult(stages=[
        StageOutcome("lidar", "ok", 0, 1.0),
        StageOutcome("fusion", "failed", 1, 2.0, "boom"),
    ])
    assert result.failed == ["fusion"]
    assert result.to_dict()["stages"][1]["error"] == "boom"


# --- interactive ------------------------------------------------------------
def test_roi_overrides_for_a_bbox():
    overrides = _roi_overrides(bounds=[135.2, 34.7, 135.25, 34.74], center=None,
                               radius_m=None, name="test")
    assert "roi.center=null" in overrides
    assert any(item.startswith("roi.bbox=[135.200000,34.700000") for item in overrides)
    assert "roi.name=test" in overrides


def test_roi_overrides_for_a_centre():
    overrides = _roi_overrides(bounds=None, center=(34.7284, 135.2348), radius_m=500,
                               name=None)
    assert "roi.bbox=null" in overrides
    assert any("lat: 34.728400" in item for item in overrides)
    assert "roi.radius_m=500.0" in overrides


def test_roi_overrides_are_loadable(default_config_path):
    from rokko_geofusion.config import load_config

    overrides = _roi_overrides(bounds=[135.2, 34.7, 135.25, 34.74], center=None,
                               radius_m=None, name="bbox_roi")
    config = load_config(default_config_path, overrides=overrides)
    assert config.roi.center is None
    assert config.roi.bbox == [135.2, 34.7, 135.25, 34.74]
    assert config.roi.name == "bbox_roi"


def test_roi_overrides_validate_their_input():
    with pytest.raises(ValueError):
        _roi_overrides(bounds=[1, 2, 3], center=None, radius_m=None, name=None)
    with pytest.raises(ValueError):
        _roi_overrides(bounds=None, center=(1,), radius_m=None, name=None)


def test_summarize_subregion_without_products_is_refused(config):
    config.roi.name = "nothing"
    config.paths.ensure()
    with pytest.raises(UnsupportedError, match="run_pipeline"):
        summarize_subregion(config, [135.2, 34.7, 135.21, 34.71])


def test_summarize_subregion_reads_only_the_window(config):
    import numpy as np

    from rokko_geofusion.crs import CrsManager, RoiGeometry
    from rokko_geofusion.io.raster import write_grid_raster

    config.roi.name = "sub"
    config.roi.radius_m = 200.0
    config.paths.ensure()
    roi = RoiGeometry.from_config(config.roi, CrsManager(config.crs),
                                  grid_snap_m=config.crs.grid_snap_m)
    grid = roi.grid(config.lidar.resolution_m)
    rows, _ = np.mgrid[0:grid.height, 0:grid.width]
    write_grid_raster(config.paths.raster / "elevation.tif",
                      (100.0 + rows).astype(np.float32), grid, nodata=float("nan"))
    write_grid_raster(config.paths.raster / "slope.tif",
                      np.full(grid.shape, 7.0, np.float32), grid, nodata=float("nan"))
    write_grid_raster(config.paths.raster / "aspect.tif",
                      np.full(grid.shape, 180.0, np.float32), grid, nodata=float("nan"))

    min_x, min_y, max_x, max_y = roi.bounds_projected
    quarter = (min_x, min_y, min_x + (max_x - min_x) / 4, min_y + (max_y - min_y) / 4)
    statistics = summarize_subregion(config, quarter, geographic=False)

    assert statistics["cells"] < grid.width * grid.height / 4 * 1.3
    assert statistics["slope"]["mean"] == pytest.approx(7.0)
    assert "object_height" in statistics["unavailable"]
    assert statistics["selection"]["crs"] == config.crs.projected


def test_summarize_subregion_rejects_a_disjoint_box(config):
    import numpy as np

    from rokko_geofusion.crs import CrsManager, RoiGeometry
    from rokko_geofusion.io.raster import write_grid_raster

    config.roi.name = "disjoint"
    config.roi.radius_m = 100.0
    config.paths.ensure()
    roi = RoiGeometry.from_config(config.roi, CrsManager(config.crs),
                                  grid_snap_m=config.crs.grid_snap_m)
    grid = roi.grid(config.lidar.resolution_m)
    write_grid_raster(config.paths.raster / "elevation.tif",
                      np.zeros(grid.shape, np.float32), grid, nodata=float("nan"))
    with pytest.raises(UnsupportedError, match="does not overlap"):
        summarize_subregion(config, [120.0, 20.0, 120.1, 20.1])


# --- notebook discipline ----------------------------------------------------
def _notebook(repo_root: Path) -> dict:
    path = repo_root / "notebooks" / "geofusion_demo.ipynb"
    assert path.is_file(), "the demo notebook has not been generated"
    return json.loads(path.read_text(encoding="utf-8"))


def test_notebook_is_valid_and_has_no_stored_output(repo_root):
    notebook = _notebook(repo_root)
    assert notebook["nbformat"] == 4
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == []
            assert cell["execution_count"] is None


def test_notebook_cells_carry_the_ids_nbformat_4_5_requires(repo_root):
    notebook = _notebook(repo_root)
    assert (notebook["nbformat"], notebook["nbformat_minor"]) >= (4, 5)
    ids = [cell.get("id") for cell in notebook["cells"]]
    assert all(ids), "every cell needs an id under nbformat 4.5"
    assert len(set(ids)) == len(ids), "cell ids must be unique"
    assert all(re.fullmatch(r"[A-Za-z0-9_-]{1,64}", cell_id) for cell_id in ids)


def test_notebook_defines_no_functions_or_classes(repo_root):
    """CLAUDE.md rule 1: processing logic never lives in a notebook."""
    offenders = []
    for index, cell in enumerate(_notebook(repo_root)["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        for pattern in (r"^\s*def\s+\w+", r"^\s*class\s+\w+"):
            if re.search(pattern, source, re.MULTILINE):
                offenders.append(index)
    assert not offenders, f"cells {offenders} define functions or classes"


def test_notebook_only_calls_scripts_or_the_package(repo_root):
    notebook = _notebook(repo_root)
    sources = ["".join(cell["source"]) for cell in notebook["cells"]
               if cell["cell_type"] == "code"]
    joined = "\n".join(sources)
    assert "scripts/run_pipeline.py" in joined
    assert "from rokko_geofusion" in joined
    # No raw download or CRS arithmetic in the notebook.
    for forbidden in ("requests.get", "Transformer.from_crs", "rasterio.open(",
                      "overpass", "cyberjapandata"):
        assert forbidden not in joined, f"notebook performs {forbidden} itself"


def test_notebook_covers_every_pipeline_stage(repo_root):
    joined = "\n".join("".join(cell["source"]) for cell in _notebook(repo_root)["cells"])
    for stage in DEFAULT_ORDER:
        assert stage in joined, f"the notebook never mentions the {stage} stage"


def test_first_cell_makes_the_package_importable_without_a_restart(repo_root):
    """Regression: in Colab, cell 03 failed with ModuleNotFoundError.

    Cell 01 runs `pip install -e` inside the already-running kernel. An editable
    install is registered through a .pth file, and Python reads .pth files only
    at start-up, so the kernel that performed the install cannot import it.

    `python -S` reproduces that state (no site-packages, no .pth processing), and
    the notebooks/ folder is where Jupyter starts a local kernel.
    """
    import subprocess
    import sys

    notebook = _notebook(repo_root)
    first_cell = next(cell for cell in notebook["cells"] if cell["cell_type"] == "code")
    probe = (
        "".join(first_cell["source"])
        + "\nimport os\nimport rokko_geofusion\nprint('CWD=' + os.getcwd())\n"
    )
    completed = subprocess.run(
        [sys.executable, "-S", "-c", probe],
        cwd=repo_root / "notebooks",
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    # Every later cell uses paths relative to the repository root.
    assert f"CWD={repo_root}" in completed.stdout


def test_notebook_generator_is_deterministic(repo_root, tmp_path, monkeypatch):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "build_notebook", repo_root / "tools" / "build_notebook.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    first = json.dumps(module.build(), indent=1, ensure_ascii=False)
    second = json.dumps(module.build(), indent=1, ensure_ascii=False)
    assert first == second
    on_disk = (repo_root / "notebooks" / "geofusion_demo.ipynb").read_text(encoding="utf-8")
    assert on_disk.strip() == first.strip(), (
        "notebooks/geofusion_demo.ipynb is out of date; run tools/build_notebook.py"
    )
