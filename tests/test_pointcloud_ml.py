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


def test_feature_dropout_blanks_whole_non_geometry_channels():
    """Without this, rotation-invariant channels make the contrastive task trivial."""
    rng = np.random.default_rng(0)
    points = np.ones((16, 10), np.float32)
    points[:, :3] = rng.uniform(-1, 1, size=(16, 3))
    views = [augment(points, rng, jitter=0.0, dropout=0.0, rotate=False,
                     feature_dropout=1.0) for _ in range(3)]
    for view in views:
        assert np.allclose(view[:, 3:], 0.0), "every extra channel should be blanked"
    kept = augment(points, rng, jitter=0.0, dropout=0.0, rotate=False,
                   colour_jitter=0.0, feature_dropout=0.0)
    assert np.allclose(kept[:, 3:], 1.0)


def test_feature_dropout_leaves_geometry_alone():
    rng = np.random.default_rng(1)
    points = np.ones((8, 8), np.float32)
    points[:, :3] = 0.5
    view = augment(points, rng, jitter=0.0, dropout=0.0, rotate=False,
                   feature_dropout=1.0)
    assert np.allclose(view[:, :3], 0.5)


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


# --- the resource profile really drives point sampling ------------------------------
def _profile(**overrides):
    from rokko_geofusion.environment import ResourceProfile

    base = dict(device="cpu", tier="test", seg_tile_px=512, seg_batch_size=1,
                pc_num_points=32, pc_batch_size=2, max_points_in_memory=1_000_000,
                http_max_workers=1, display_points=1000)
    base.update(overrides)
    return ResourceProfile(**base)


def test_training_samples_the_profiles_point_count(fusion_table, monkeypatch):
    """Regression: the dataset always sampled the config's count (4096 in the
    default config), so a CPU runtime trained on 4x more points than its profile
    allows and the Colab CPU run took 3.2 hours."""
    from rokko_geofusion.pointcloud_ml.train import train_feature_set

    config, path = fusion_table                   # config asks for 64 points
    dataset = FusionTileDataset(config, path, feature_sets=["xyz"])
    seen: list[tuple[int, ...]] = []
    original = FusionTileDataset.batch

    def spy(self, *args, **kwargs):
        array = original(self, *args, **kwargs)
        seen.append(array.shape)
        return array

    monkeypatch.setattr(FusionTileDataset, "batch", spy)
    result, _ = train_feature_set(config, dataset, "xyz", profile=_profile(pc_num_points=32))

    assert {shape[1] for shape in seen} == {32}   # training and embedding passes
    assert result.num_points == 32
    assert result.batch_size == 2


def test_the_tile_set_does_not_depend_on_the_sample_count(fusion_table, tmp_path):
    """A CPU run and a GPU run must compare the same tiles.

    The fixture's tiles are all full (400 points), so an 80-point edge strip is
    added: it passes a threshold derived from 16 sampled points (32) but not one
    derived from 2048 (128), which is exactly where the old code diverged.
    """
    import pandas as pd

    config, path = fusion_table
    frame = pd.read_parquet(path)
    edge = frame[frame["x"] < frame["x"].min() + 4].copy()      # 4 columns x 80 rows
    edge["x"] = edge["x"] + 80.0                                # beyond the last tile
    extended = tmp_path / "extended.parquet"
    pd.concat([frame, edge], ignore_index=True).to_parquet(extended, index=False)

    few = FusionTileDataset(config, extended, feature_sets=["xyz"], num_points=16)
    many = FusionTileDataset(config, extended, feature_sets=["xyz"], num_points=2048)
    assert any(len(indices) == 80 for indices in few.tiles)     # the edge tiles exist
    assert few.tile_origins == many.tile_origins


def test_the_ladder_steps_over_rungs_that_only_touch_segmentation():
    from rokko_geofusion.pointcloud_ml.train import smaller_pointcloud_profile

    # Batch already 1: the next shared rung only halves seg_tile_px, which
    # changes nothing here. It used to end the ladder at that point.
    start = _profile(device="cuda", seg_batch_size=1, seg_tile_px=1024,
                     pc_batch_size=1, pc_num_points=4096)
    following = smaller_pointcloud_profile(start)
    assert following is not None
    assert following.pc_num_points == 2048
    assert following.device == "cuda"


def test_the_ladder_ends_on_the_cpu_and_then_reports_exhaustion():
    from rokko_geofusion.pointcloud_ml.train import smaller_pointcloud_profile

    rungs = []
    current = _profile(device="cuda", seg_batch_size=4, pc_batch_size=8, pc_num_points=4096)
    while (current := smaller_pointcloud_profile(current)) is not None:
        rungs.append((current.device, current.pc_batch_size, current.pc_num_points))
        assert len(rungs) < 50, "the ladder must terminate"
    assert rungs[-1][0] == "cpu"
    assert (rungs[0][1], rungs[0][2]) == (4, 4096)          # batch shrinks first
    assert any(points < 4096 for _, _, points in rungs)     # then the point count


def test_out_of_memory_reaches_the_point_count_rung(fusion_table, monkeypatch):
    """Regression: after the batch reached 1 the retry loop gave up instead of
    lowering the point count or falling back to the CPU."""
    import rokko_geofusion.pointcloud_ml.train as train
    from rokko_geofusion.exceptions import ResourceError

    config, path = fusion_table
    attempts: list[tuple[str, int, int]] = []
    real = train.train_feature_set

    def picky(config, dataset, feature_set, *, profile, seed=0):
        attempts.append((profile.device, profile.pc_batch_size, profile.pc_num_points))
        if profile.pc_num_points > 2048:
            raise ResourceError("simulated out of memory")
        return real(config, dataset, feature_set, profile=profile, seed=seed)

    monkeypatch.setattr(train, "train_feature_set", picky)
    start = _profile(device="cpu", seg_batch_size=2, seg_tile_px=512,
                     pc_batch_size=2, pc_num_points=4096)
    outcome = train.run_comparison(config, profile=start, table_path=path,
                                   feature_sets=["xyz"])
    assert outcome["results"][0]["num_points"] == 2048
    assert attempts[-1][2] == 2048
    assert len(attempts) > 2          # it walked past the batch-only rungs
