"""Point cloud ML: tiling, feature encoding, augmentation and the comparison."""

from __future__ import annotations

import numpy as np
import pytest

from rokko_geofusion.exceptions import UnsupportedError
from rokko_geofusion.pointcloud_ml.dataset import (
    FEATURE_SETS,
    FusionTileDataset,
    feature_dimension,
)

torch = pytest.importorskip("torch")

from rokko_geofusion.pointcloud_ml.pointnet import (  # noqa: E402
    augment,
    build_pointnet,
    nt_xent_loss,
)


@pytest.fixture()
def fusion_table(config, tmp_path):
    """A synthetic fusion table: two halves with different colour and class."""
    import pandas as pd

    config.pointcloud_ml.num_points = 64
    config.pointcloud_ml.tile_size_m = 20.0
    config.pointcloud_ml.batch_size = 4
    config.pointcloud_ml.epochs = 1
    config.pointcloud_ml.n_clusters = 2
    config.processing.voxel_size_m = 1.0

    rng = np.random.default_rng(0)
    xs, ys = np.meshgrid(np.arange(0.5, 80.5, 1.0), np.arange(0.5, 80.5, 1.0))
    x = xs.ravel() + 1000.0
    y = ys.ravel() + 2000.0
    west = x < 1040.0
    frame = pd.DataFrame(
        {
            "x": x,
            "y": y,
            "z": np.where(west, 100.0, 130.0) + rng.normal(0, 0.2, x.size),
            "red": np.where(west, 200, 40).astype(np.uint8),
            "green": np.full(x.size, 90, np.uint8),
            "blue": np.where(west, 40, 200).astype(np.uint8),
            "slope": np.where(west, 5.0, 25.0).astype(np.float32),
            "aspect": np.full(x.size, 180.0, np.float32),
            "relief": np.full(x.size, 2.0, np.float32),
            "object_height": np.full(x.size, np.nan, np.float32),
            "image_class": np.where(west, 1, 3).astype(np.int16),
            "image_confidence": np.full(x.size, 0.8, np.float32),
            "in_building": west,
            "on_road": ~west,
            "fused_class": np.where(west, 1, 4).astype(np.uint8),
        }
    )
    path = tmp_path / "fusion_cells.parquet"
    frame.to_parquet(path, index=False)
    return config, path


# --- feature encoding -------------------------------------------------------
@pytest.mark.parametrize("feature_set", list(FEATURE_SETS))
def test_feature_dimension_matches_the_encoded_width(fusion_table, feature_set):
    config, path = fusion_table
    dataset = FusionTileDataset(config, path, feature_sets=[feature_set])
    sample = dataset.sample(0)
    array = sample.features[feature_set]
    assert array.shape == (config.pointcloud_ml.num_points,
                           feature_dimension(feature_set, len(config.segmentation.classes)))
    assert array.dtype == np.float32
    assert np.isfinite(array).all(), "NaNs must be imputed before the network sees them"


def test_xyz_is_normalised_into_the_unit_tile(fusion_table):
    config, path = fusion_table
    dataset = FusionTileDataset(config, path, feature_sets=["xyz"])
    array = dataset.sample(0).features["xyz"]
    assert np.abs(array[:, :2]).max() <= 1.0 + 1e-6
    assert array[:, 2].min() >= 0.0


def test_colour_channels_are_scaled_to_unit_range(fusion_table):
    config, path = fusion_table
    dataset = FusionTileDataset(config, path, feature_sets=["xyz_rgb"])
    array = dataset.sample(0).features["xyz_rgb"]
    assert 0.0 <= array[:, 3:6].min() and array[:, 3:6].max() <= 1.0


def test_missing_object_height_is_flagged_not_faked(fusion_table):
    config, path = fusion_table
    dataset = FusionTileDataset(config, path, feature_sets=["xyz_rgb_terrain"])
    array = dataset.sample(0).features["xyz_rgb_terrain"]
    # Last two terrain channels are (height_value, height_known).
    assert np.allclose(array[:, -1], 0.0), "height_known must be 0 when no DSM exists"
    assert np.allclose(array[:, -2], 0.0)


def test_semantic_channels_are_confidence_weighted_one_hot(fusion_table):
    config, path = fusion_table
    dataset = FusionTileDataset(config, path, feature_sets=["xyz_rgb_terrain_semantic"])
    array = dataset.sample(0).features["xyz_rgb_terrain_semantic"]
    n_classes = len(config.segmentation.classes)
    one_hot = array[:, 12:12 + n_classes]
    assert np.allclose(one_hot.sum(axis=1), 0.8, atol=1e-5)  # confidence weighted
    assert (one_hot > 0).sum(axis=1).max() == 1


def test_unknown_feature_set_is_refused(fusion_table):
    config, path = fusion_table
    with pytest.raises(UnsupportedError):
        FusionTileDataset(config, path, feature_sets=["xyz_lidar_magic"])
    with pytest.raises(UnsupportedError):
        feature_dimension("nope", 6)


def test_missing_fusion_table_is_refused(config, tmp_path):
    with pytest.raises(UnsupportedError, match="fuse_modalities"):
        FusionTileDataset(config, tmp_path / "absent.parquet")


# --- tiling and sampling ----------------------------------------------------
def test_tiles_partition_the_area(fusion_table):
    config, path = fusion_table
    dataset = FusionTileDataset(config, path, feature_sets=["xyz"])
    # 80 m of data in 20 m tiles.
    assert len(dataset) == 16
    total = sum(indices.size for indices in dataset.tiles)
    assert total == 80 * 80


def test_sampling_returns_the_requested_point_count(fusion_table):
    config, path = fusion_table
    dataset = FusionTileDataset(config, path, feature_sets=["xyz"])
    for index in (0, len(dataset) - 1):
        assert dataset.sample(index).features["xyz"].shape[0] == config.pointcloud_ml.num_points


def test_voxel_downsampling_reduces_the_pool(fusion_table):
    config, path = fusion_table
    config.processing.voxel_size_m = 5.0
    dataset = FusionTileDataset(config, path, feature_sets=["xyz"])
    sample = dataset.sample(0)
    # Voxels are 3-D: a 20 m tile holds 4x4 columns, times however many 5 m
    # height bands the (slightly noisy) surface spans -- far fewer than the
    # 400 raw cells, which is the point of the step.
    raw_points = dataset.tiles[0].size
    assert raw_points == 400
    assert sample.n_points_available < raw_points // 4
    assert sample.n_points_available <= 16 * 3
    # Sampling still returns a fixed count, resampling the survivors.
    assert sample.features["xyz"].shape[0] == config.pointcloud_ml.num_points


def test_batch_shape(fusion_table):
    config, path = fusion_table
    dataset = FusionTileDataset(config, path, feature_sets=["xyz_rgb"])
    batch = dataset.batch([0, 1, 2], "xyz_rgb")
    assert batch.shape == (3, config.pointcloud_ml.num_points, 6)


def test_dominant_class_is_evaluation_only(fusion_table):
    config, path = fusion_table
    dataset = FusionTileDataset(config, path, feature_sets=["xyz"])
    classes = dataset.dominant_classes()
    assert classes.shape == (len(dataset),)
    assert set(classes.tolist()) <= {1, 4}


def test_describe_reports_the_pipeline(fusion_table):
    config, path = fusion_table
    dataset = FusionTileDataset(config, path, feature_sets=["xyz", "xyz_rgb"])
    described = dataset.describe()
    assert described["tiles"] == len(dataset)
    assert described["feature_sets"] == {"xyz": 3, "xyz_rgb": 6}


# --- model ------------------------------------------------------------------
def test_encoder_is_permutation_invariant():
    model = build_pointnet(3, embedding_dim=32).eval()
    points = torch.randn(2, 128, 3)
    shuffled = points[:, torch.randperm(128), :]
    with torch.no_grad():
        a = model(points, project=False)
        b = model(shuffled, project=False)
    torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-4)


def test_encoder_output_shapes():
    model = build_pointnet(6, embedding_dim=64, projection_dim=16).eval()
    with torch.no_grad():
        embedding, projection = model(torch.randn(4, 64, 6))
    assert embedding.shape == (4, 64)
    assert projection.shape == (4, 16)


def test_nt_xent_prefers_matching_views():
    torch.manual_seed(0)
    a = torch.nn.functional.normalize(torch.randn(8, 16), dim=1)
    matched = nt_xent_loss(a, a.clone())
    mismatched = nt_xent_loss(a, torch.nn.functional.normalize(torch.randn(8, 16), dim=1))
    assert matched < mismatched


def test_augment_preserves_shape_and_colour_range():
    rng = np.random.default_rng(0)
    points = np.concatenate(
        [rng.uniform(-1, 1, size=(64, 3)), rng.uniform(0, 1, size=(64, 3))], axis=1
    ).astype(np.float32)
    augmented = augment(points, rng)
    assert augmented.shape == points.shape
    assert 0.0 <= augmented[:, 3:6].min() and augmented[:, 3:6].max() <= 1.0
    assert not np.allclose(augmented[:, :3], points[:, :3])


def test_augment_rotation_preserves_radius():
    rng = np.random.default_rng(1)
    points = rng.uniform(-1, 1, size=(32, 3)).astype(np.float32)
    rotated = augment(points, rng, jitter=0.0, dropout=0.0)
    before = np.sort(np.hypot(points[:, 0], points[:, 1]))
    after = np.sort(np.hypot(rotated[:, 0], rotated[:, 1]))
    np.testing.assert_allclose(before, after, atol=1e-5)


# --- end to end -------------------------------------------------------------
def test_comparison_runs_and_flags_the_leaky_feature_set(fusion_table):
    from rokko_geofusion.environment import make_resource_profile
    from rokko_geofusion.pointcloud_ml.train import run_comparison

    config, path = fusion_table
    outcome = run_comparison(
        config,
        profile=make_resource_profile(config),
        table_path=path,
        feature_sets=["xyz", "xyz_rgb_terrain_semantic"],
    )
    assert [r["feature_set"] for r in outcome["results"]] == [
        "xyz", "xyz_rgb_terrain_semantic"
    ]
    for result in outcome["results"]:
        assert np.isfinite(result["final_loss"])
        assert len(result["loss_curve"]) == config.pointcloud_ml.epochs
        assert outcome["embeddings"][result["feature_set"]].shape == (
            len(outcome["tile_origins"]), config.pointcloud_ml.embedding_dim
        )
    leaky = outcome["results"][1]
    assert leaky["caveat"] and "inflated by construction" in leaky["caveat"]
    assert outcome["results"][0]["caveat"] is None
    assert "No supervised accuracy" in outcome["evaluation_note"]


def test_comparison_is_refused_when_disabled(fusion_table):
    from rokko_geofusion.environment import make_resource_profile
    from rokko_geofusion.pointcloud_ml.train import run_comparison

    config, path = fusion_table
    config.pointcloud_ml.enabled = False
    with pytest.raises(UnsupportedError):
        run_comparison(config, profile=make_resource_profile(config), table_path=path)
