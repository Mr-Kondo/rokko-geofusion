"""The V1-V6 spatial validation checks."""

from __future__ import annotations

import json

import numpy as np
import pytest

from rokko_geofusion.crs import CrsManager, RoiGeometry
from rokko_geofusion.io.raster import write_grid_raster
from rokko_geofusion.validation import (
    best_shift,
    check_v2_grid_alignment,
    check_v3_dsm_above_dem,
    check_v4_buildings_are_tall,
    check_v5_cloud_matches_imagery,
    check_v6_reproducibility,
    mask_agreement,
    run_all,
    summarise,
    to_markdown,
)


@pytest.fixture()
def scene(config):
    config.roi.name = "val"
    config.roi.radius_m = 50.0
    config.paths.ensure()
    roi = RoiGeometry.from_config(config.roi, CrsManager(config.crs),
                                  grid_snap_m=config.crs.grid_snap_m)
    return config, roi


# --- helpers ----------------------------------------------------------------
def test_mask_agreement_on_identical_masks():
    mask = np.zeros((10, 10), bool)
    mask[2:6, 2:6] = True
    values = mask_agreement(mask, mask)
    assert values["iou"] == pytest.approx(1.0)
    assert values["recall"] == pytest.approx(1.0)
    assert values["precision"] == pytest.approx(1.0)


def test_mask_agreement_on_disjoint_masks():
    a = np.zeros((10, 10), bool)
    a[0:3, 0:3] = True
    b = np.zeros((10, 10), bool)
    b[7:10, 7:10] = True
    values = mask_agreement(a, b)
    assert values["iou"] == 0.0
    assert values["recall"] == 0.0


def test_mask_agreement_with_empty_prediction():
    values = mask_agreement(np.zeros((4, 4), bool), np.ones((4, 4), bool))
    assert values["precision"] == 0.0
    assert values["recall"] == 0.0


def test_best_shift_finds_zero_for_aligned_masks():
    mask = np.zeros((32, 32), bool)
    mask[8:16, 8:16] = True
    assert best_shift(mask, mask, 4)[:2] == (0, 0)


@pytest.mark.parametrize(("d_row", "d_col"), [(2, 0), (0, -3), (-2, 2)])
def test_best_shift_recovers_a_known_offset(d_row, d_col):
    reference = np.zeros((40, 40), bool)
    reference[12:24, 12:24] = True
    shifted = np.roll(np.roll(reference, -d_row, axis=0), -d_col, axis=1)
    assert best_shift(shifted, reference, 5)[:2] == (d_row, d_col)


# --- checks -----------------------------------------------------------------
def test_v2_needs_at_least_two_rasters(scene):
    config, roi = scene
    assert check_v2_grid_alignment(config, roi).status == "unavailable"


def test_v2_passes_for_nested_grids(scene):
    config, roi = scene
    for name, resolution in (("dem.tif", 5.0), ("orthophoto.tif", 0.5)):
        grid = roi.grid(resolution)
        array = np.zeros((3, grid.height, grid.width), np.uint8) if "ortho" in name \
            else np.zeros(grid.shape, np.float32)
        write_grid_raster(config.paths.interim / name, array, grid, nodata=None)
    result = check_v2_grid_alignment(config, roi)
    assert result.status == "pass"
    assert len(result.measurements["grids"]) == 2


def test_v2_fails_on_a_mismatched_crs(scene):
    config, roi = scene
    grid = roi.grid(5.0)
    write_grid_raster(config.paths.interim / "dem.tif", np.zeros(grid.shape, np.float32),
                      grid, nodata=None)
    from dataclasses import replace

    foreign = replace(grid, crs="EPSG:3857")
    write_grid_raster(config.paths.interim / "dsm.tif", np.zeros(foreign.shape, np.float32),
                      foreign, nodata=None)
    result = check_v2_grid_alignment(config, roi)
    assert result.status == "fail"
    assert "CRS" in result.detail


def test_v3_unavailable_without_a_dsm(scene):
    config, roi = scene
    result = check_v3_dsm_above_dem(config, roi)
    assert result.status == "unavailable"
    assert "no DSM is configured" in result.detail


def test_v3_passes_when_the_surface_is_above_the_ground(scene):
    config, roi = scene
    grid = roi.grid(5.0)
    dem = np.full(grid.shape, 100.0, np.float32)
    write_grid_raster(config.paths.interim / "dem.tif", dem, grid, nodata=float("nan"))
    write_grid_raster(config.paths.interim / "dsm.tif", dem + 4.0, grid, nodata=float("nan"))
    result = check_v3_dsm_above_dem(config, roi)
    assert result.status == "pass"
    assert result.measurements["median_difference_m"] == pytest.approx(4.0)


def test_v3_fails_when_the_surface_sits_below_the_ground(scene):
    config, roi = scene
    grid = roi.grid(5.0)
    dem = np.full(grid.shape, 100.0, np.float32)
    dsm = dem.copy()
    dsm[: grid.height // 2] -= 5.0      # half the cells clearly below ground
    write_grid_raster(config.paths.interim / "dem.tif", dem, grid, nodata=float("nan"))
    write_grid_raster(config.paths.interim / "dsm.tif", dsm, grid, nodata=float("nan"))
    result = check_v3_dsm_above_dem(config, roi)
    assert result.status == "fail"
    assert result.measurements["negative_fraction"] > 0.4


def test_v4_unavailable_without_an_ndsm(scene):
    config, roi = scene
    assert check_v4_buildings_are_tall(config, roi).status == "unavailable"


def test_v5_unavailable_without_a_cloud(scene):
    config, roi = scene
    assert check_v5_cloud_matches_imagery(config, roi).status == "unavailable"


def test_v5_passes_for_a_freshly_built_cloud(scene):
    config, roi = scene
    grid = roi.grid(config.lidar.resolution_m)
    dem = np.full(grid.shape, 120.0, np.float32)
    dem_path = write_grid_raster(config.paths.interim / "dem.tif", dem, grid,
                                 nodata=float("nan"))
    rgb_grid = roi.grid(config.imagery.resolution_m)
    rng = np.random.default_rng(0)
    rgb = rng.integers(20, 240, size=(3, rgb_grid.height, rgb_grid.width), dtype=np.uint8)
    ortho_path = write_grid_raster(config.paths.interim / "orthophoto.tif", rgb, rgb_grid,
                                   nodata=None)

    from rokko_geofusion.lidar.pointcloud import build_point_cloud

    build_point_cloud(config, roi, dem_path=dem_path, imagery_path=ortho_path)
    result = check_v5_cloud_matches_imagery(config, roi)
    assert result.status == "pass"
    assert result.measurements["unsampled_points"] == 0
    assert result.measurements["red_correlation"] > 0.95


def test_v6_is_deterministic(scene):
    config, roi = scene
    grid = roi.grid(config.lidar.resolution_m)
    rng = np.random.default_rng(1)
    dem = rng.normal(100.0, 5.0, size=grid.shape).astype(np.float32)
    write_grid_raster(config.paths.interim / "dem.tif", dem, grid, nodata=float("nan"))
    result = check_v6_reproducibility(config, roi)
    assert result.status == "pass"
    assert result.measurements["stable"] is True
    assert result.measurements["config_fingerprint"] == config.fingerprint()
    assert len(result.measurements["digest"]) == 16
    assert result.measurements["digest"] in result.detail
    # What was hashed is kept, so two machines can be compared number by number.
    json.dumps(result.measurements["digest_inputs"])
    assert set(result.measurements["digest_inputs"]["moments"]) == {
        "elevation_m", "slope", "local_relief_m"}


def test_v6_fails_when_recomputation_is_not_bit_identical(scene, monkeypatch):
    import rokko_geofusion.terrain.analysis as terrain

    config, roi = scene
    grid = roi.grid(config.lidar.resolution_m)
    write_grid_raster(config.paths.interim / "dem.tif",
                      np.full(grid.shape, 100.0, np.float32), grid, nodata=float("nan"))
    real = terrain.slope
    noise = np.random.default_rng(2)
    monkeypatch.setattr(terrain, "slope",
                        lambda *a, **k: real(*a, **k) + noise.normal(0, 1e-3, size=grid.shape))
    assert check_v6_reproducibility(config, roi).status == "fail"


def _gsi_like_dem(shape, seed=3):
    """Terrain quantised at 1 cm like the GSI source heights."""
    rng = np.random.default_rng(seed)
    rows, cols = np.mgrid[0:shape[0], 0:shape[1]]
    surface = 150.0 + 0.8 * rows + 0.3 * cols + rng.normal(0, 2.0, size=shape)
    return np.round(surface, 2).astype(np.float32)


def test_results_digest_is_immune_to_last_bit_differences(scene):
    """Regression: the digest hashed raw float bytes, so the macOS and Linux
    results of the same run never matched. A different machine's resampling or
    maths library shows up as last-bit noise; the digest must not see it."""
    from rokko_geofusion.validation import results_digest

    config, roi = scene
    dem = _gsi_like_dem(roi.grid(config.lidar.resolution_m).shape)
    reference, _ = results_digest(config, roi, dem)
    for seed in range(10):
        rng = np.random.default_rng(seed)
        towards = np.where(rng.random(dem.shape) < 0.5, -np.inf, np.inf).astype(np.float32)
        noisy = dem.copy()
        for _ in range(4):
            noisy = np.nextafter(noisy, towards)
        assert results_digest(config, roi, noisy)[0] == reference


def test_results_digest_detects_real_changes(scene):
    from rokko_geofusion.validation import results_digest

    config, roi = scene
    dem = _gsi_like_dem(roi.grid(config.lidar.resolution_m).shape)
    reference, _ = results_digest(config, roi, dem)
    assert results_digest(config, roi, dem + np.float32(0.01))[0] != reference
    changed = dem.copy()
    changed[: changed.shape[0] // 3] += 2.0
    assert results_digest(config, roi, changed)[0] != reference
    config.terrain.relief_window_cells = 7
    assert results_digest(config, roi, dem)[0] != reference


def test_results_digest_only_matches_runs_of_the_same_configuration(scene):
    """Equal terrain under a different configuration is not the same result."""
    from rokko_geofusion.validation import results_digest

    config, roi = scene
    dem = _gsi_like_dem(roi.grid(config.lidar.resolution_m).shape)
    reference, _ = results_digest(config, roi, dem)
    config.fusion.thresholds.building_min_height_m += 0.5   # does not touch terrain
    assert results_digest(config, roi, dem)[0] != reference


def test_results_digest_does_not_depend_on_the_sign_of_zero(scene):
    from rokko_geofusion.validation import results_digest

    config, roi = scene
    # A mean of -0.0001 m rounds to -0.0, which JSON would write as "-0.0" on
    # one machine and "0.0" on another whose mean came out at +0.0001.
    shape = roi.grid(config.lidar.resolution_m).shape
    below = np.full(shape, -0.0001, np.float32)
    above = np.full(shape, 0.0001, np.float32)
    digest_below, inputs = results_digest(config, roi, below)
    assert "-0.0" not in json.dumps(inputs)
    assert digest_below == results_digest(config, roi, above)[0]


def test_run_all_reports_every_check(scene):
    config, roi = scene
    results = run_all(config, roi)
    assert {r.id for r in results} == {"V1", "V2", "V3", "V4", "V5", "V6", "SEG"}
    # Nothing exists yet, so nothing may claim to have passed.
    assert all(r.status in {"unavailable", "fail"} for r in results)


def test_run_all_can_select_one_check(scene):
    config, roi = scene
    results = run_all(config, roi, only="V6")
    assert [r.id for r in results] == ["V6"]


def test_summary_and_markdown(scene):
    config, roi = scene
    results = run_all(config, roi)
    summary = summarise(results)
    assert sum(summary["counts"].values()) == len(results)
    markdown = to_markdown(results)
    assert markdown.startswith("| check |")
    assert "V1" in markdown
