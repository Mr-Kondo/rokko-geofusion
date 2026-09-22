"""Segmentation: vocabulary mapping, tiling, blending and the OOM ladder."""

from __future__ import annotations

import numpy as np
import pytest

from rokko_geofusion.crs import CrsManager, RoiGeometry
from rokko_geofusion.exceptions import ResourceError, UnsupportedError
from rokko_geofusion.io.raster import grid_from_raster, write_grid_raster
from rokko_geofusion.segmentation.classes import (
    build_class_mapping,
    mapping_summary,
    normalise_label,
)
from rokko_geofusion.segmentation.runner import (
    _blend_window,
    _tile_origins,
    segment_imagery,
)

PROJECT_CLASSES = ["other", "building", "road", "vegetation", "bare_soil", "water"]

LOVEDA = {0: "ignore", 1: "background", 2: "building", 3: "road", 4: "water",
          5: "barren", 6: "forest"}
ADE_LIKE = {0: "wall", 1: "building;edifice", 2: "sky", 3: "floor;flooring",
            4: "tree", 6: "road;route", 9: "grass", 13: "earth;ground",
            21: "water", 46: "sand"}


# --- vocabulary -------------------------------------------------------------
def test_normalise_label_splits_synonyms():
    assert normalise_label("building;edifice") == {"building", "edifice"}
    assert normalise_label("earth, ground") == {"earth", "ground"}
    assert normalise_label("Water") == {"water"}


def test_loveda_vocabulary_maps_onto_project_classes():
    mapping = build_class_mapping(LOVEDA, PROJECT_CLASSES)
    assert mapping[2] == PROJECT_CLASSES.index("building")
    assert mapping[3] == PROJECT_CLASSES.index("road")
    assert mapping[4] == PROJECT_CLASSES.index("water")
    assert mapping[5] == PROJECT_CLASSES.index("bare_soil")
    assert mapping[6] == PROJECT_CLASSES.index("vegetation")
    # "ignore" and "background" have no counterpart -> the catch-all class.
    assert mapping[0] == 0
    assert mapping[1] == 0


def test_ade_vocabulary_maps_onto_project_classes():
    mapping = build_class_mapping(ADE_LIKE, PROJECT_CLASSES)
    assert mapping[1] == PROJECT_CLASSES.index("building")
    assert mapping[6] == PROJECT_CLASSES.index("road")
    assert mapping[4] == mapping[9] == PROJECT_CLASSES.index("vegetation")
    assert mapping[13] == mapping[46] == PROJECT_CLASSES.index("bare_soil")
    assert mapping[21] == PROJECT_CLASSES.index("water")
    assert mapping[2] == 0  # sky -> other


def test_mapping_is_by_name_not_by_index():
    """Reordering the checkpoint's ids must not change the semantics."""
    shuffled = {100 + index: label for index, label in LOVEDA.items()}
    base = build_class_mapping(LOVEDA, PROJECT_CLASSES)
    other = build_class_mapping(shuffled, PROJECT_CLASSES)
    for index, label in LOVEDA.items():
        assert base[index] == other[100 + index], label


def test_missing_project_class_is_warned_not_silent(caplog):
    with caplog.at_level("WARNING"):
        build_class_mapping({0: "building"}, PROJECT_CLASSES)
    assert "no counterpart" in caplog.text


def test_mapping_summary_lists_source_labels():
    mapping = build_class_mapping(LOVEDA, PROJECT_CLASSES)
    summary = mapping_summary(LOVEDA, mapping, PROJECT_CLASSES)
    assert summary["building"] == ["building"]
    assert summary["vegetation"] == ["forest"]
    assert set(summary["other"]) == {"ignore", "background"}


def test_empty_project_classes_are_rejected():
    with pytest.raises(ValueError):
        build_class_mapping(LOVEDA, [])


# --- tiling and blending ----------------------------------------------------
def test_tile_origins_cover_the_axis():
    origins = _tile_origins(1000, 512, 64)
    assert origins[0] == 0
    assert origins[-1] == 1000 - 512  # last tile pulled back, never past the edge
    assert all(o + 512 <= 1000 for o in origins)
    covered = np.zeros(1000, bool)
    for origin in origins:
        covered[origin:origin + 512] = True
    assert covered.all()


def test_tile_origins_when_the_tile_is_larger_than_the_image():
    assert _tile_origins(300, 512, 64) == [0]


def test_blend_window_tapers_but_never_reaches_zero():
    window = _blend_window(64, 16)
    assert window.shape == (64, 64)
    assert window.max() == pytest.approx(1.0)
    assert window.min() >= 0.05
    assert window[0, 0] < window[32, 32]
    np.testing.assert_allclose(window, window.T, atol=1e-6)      # separable
    np.testing.assert_allclose(window, window[::-1, ::-1], atol=1e-6)  # symmetric


def test_blend_window_without_taper():
    np.testing.assert_allclose(_blend_window(16, 0), 1.0)


# --- runner -----------------------------------------------------------------
class _StripeModel:
    """Deterministic stand-in: left half 'building', right half 'vegetation'."""

    def __init__(self, class_names, *, fail_if_batch_over: int | None = None):
        self.class_names = list(class_names)
        self.device = "cpu"
        self.calls = 0
        self.attempted_batches: list[int] = []
        self.completed_batches: list[int] = []
        self._fail_if_batch_over = fail_if_batch_over

    def predict_proba(self, batch: np.ndarray) -> np.ndarray:
        self.calls += 1
        self.attempted_batches.append(batch.shape[0])
        if self._fail_if_batch_over is not None and batch.shape[0] > self._fail_if_batch_over:
            raise ResourceError("simulated out of accelerator memory")
        self.completed_batches.append(batch.shape[0])
        n, _, height, width = batch.shape
        probabilities = np.zeros((n, len(self.class_names), height, width), np.float32)
        building = self.class_names.index("building")
        vegetation = self.class_names.index("vegetation")
        # Decide from the pixel content so tiles are position-independent.
        bright = batch[:, 0] > 127
        probabilities[:, building] = np.where(bright, 0.9, 0.1)
        probabilities[:, vegetation] = np.where(bright, 0.1, 0.9)
        return probabilities

    def describe(self):
        return {"backend": "test", "model_id": "stripe", "device": self.device}


@pytest.fixture()
def ortho_scene(config):
    config.roi.name = "seg"
    config.roi.radius_m = 60.0
    config.segmentation.output_resolution_m = 1.0
    config.segmentation.tile_px = 64
    config.segmentation.overlap_px = 16
    config.segmentation.batch_size = 2
    config.paths.ensure()
    roi = RoiGeometry.from_config(config.roi, CrsManager(config.crs),
                                  grid_snap_m=config.crs.grid_snap_m)
    grid = roi.grid(config.imagery.resolution_m)
    rgb = np.zeros((3, grid.height, grid.width), np.uint8)
    rgb[:, :, : grid.width // 2] = 255        # west half bright
    path = write_grid_raster(config.paths.interim / "orthophoto.tif", rgb, grid, nodata=None)
    return config, roi, path


def test_segment_writes_aligned_class_and_confidence_rasters(ortho_scene):
    from rokko_geofusion.environment import make_resource_profile

    config, roi, path = ortho_scene
    model = _StripeModel(config.segmentation.classes)
    product = segment_imagery(config, roi, path,
                              profile=make_resource_profile(config), model=model)

    assert product.class_path.is_file() and product.confidence_path.is_file()
    expected = roi.grid(config.segmentation.output_resolution_m)
    assert grid_from_raster(product.class_path).matches(expected)
    # Class and confidence rasters must share the grid with everything else.
    assert grid_from_raster(product.confidence_path).matches(expected)

    from rokko_geofusion.io.raster import read_raster

    class_index, _ = read_raster(product.class_path, band=1)
    names = config.segmentation.classes
    west = class_index[:, : class_index.shape[1] // 2 - 2]
    east = class_index[:, class_index.shape[1] // 2 + 2:]
    assert (west == names.index("building")).mean() > 0.95
    assert (east == names.index("vegetation")).mean() > 0.95


def test_confidence_is_a_probability(ortho_scene):
    from rokko_geofusion.environment import make_resource_profile
    from rokko_geofusion.io.raster import read_raster

    config, roi, path = ortho_scene
    product = segment_imagery(config, roi, path,
                              profile=make_resource_profile(config),
                              model=_StripeModel(config.segmentation.classes))
    confidence, _ = read_raster(product.confidence_path, band=1)
    assert 0.0 <= np.nanmin(confidence) and np.nanmax(confidence) <= 1.0
    assert np.nanmean(confidence) > 0.5


def test_class_fractions_are_recorded_and_sum_to_one(ortho_scene):
    from rokko_geofusion.environment import make_resource_profile

    config, roi, path = ortho_scene
    product = segment_imagery(config, roi, path,
                              profile=make_resource_profile(config),
                              model=_StripeModel(config.segmentation.classes))
    fractions = product.statistics["class_fractions"]
    assert sum(fractions.values()) == pytest.approx(1.0)
    assert fractions["building"] > 0.4
    assert fractions["vegetation"] > 0.4


def test_out_of_memory_walks_down_the_ladder(ortho_scene):
    from rokko_geofusion.environment import make_resource_profile

    config, roi, path = ortho_scene
    config.segmentation.batch_size = 4
    profile = make_resource_profile(config)
    model = _StripeModel(config.segmentation.classes, fail_if_batch_over=1)
    product = segment_imagery(config, roi, path, profile=profile, model=model)
    # The first attempt is too large and raises; every completed batch after the
    # retry fits within the model's limit.
    assert max(model.attempted_batches) > 1
    assert max(model.completed_batches) == 1
    assert product.metadata["processing"]["batch_size"] == 1
    assert product.metadata["processing"]["downscale_steps"] >= 1


def test_out_of_memory_is_raised_when_downscaling_is_disabled(ortho_scene):
    from rokko_geofusion.environment import make_resource_profile

    config, roi, path = ortho_scene
    config.segmentation.auto_downscale = False
    config.segmentation.batch_size = 4
    model = _StripeModel(config.segmentation.classes, fail_if_batch_over=1)
    with pytest.raises(ResourceError):
        segment_imagery(config, roi, path,
                        profile=make_resource_profile(config), model=model)


def test_disabled_segmentation_is_refused(ortho_scene):
    from rokko_geofusion.environment import make_resource_profile

    config, roi, path = ortho_scene
    config.segmentation.enabled = False
    with pytest.raises(UnsupportedError):
        segment_imagery(config, roi, path, profile=make_resource_profile(config),
                        model=_StripeModel(config.segmentation.classes))


def test_missing_orthophoto_is_refused(ortho_scene, tmp_path):
    from rokko_geofusion.environment import make_resource_profile

    config, roi, _ = ortho_scene
    with pytest.raises(UnsupportedError, match="no orthophoto"):
        segment_imagery(config, roi, tmp_path / "absent.tif",
                        profile=make_resource_profile(config),
                        model=_StripeModel(config.segmentation.classes))


def test_rerun_is_reproducible(ortho_scene):
    from rokko_geofusion.environment import make_resource_profile
    from rokko_geofusion.io.raster import read_raster

    config, roi, path = ortho_scene
    profile = make_resource_profile(config)
    first = segment_imagery(config, roi, path, profile=profile,
                            model=_StripeModel(config.segmentation.classes))
    a, _ = read_raster(first.class_path, band=1)
    second = segment_imagery(config, roi, path, profile=profile,
                             model=_StripeModel(config.segmentation.classes), overwrite=True)
    b, _ = read_raster(second.class_path, band=1)
    np.testing.assert_array_equal(a, b)
