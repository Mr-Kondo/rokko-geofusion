"""Point cloud generation, tiling, colourisation and downsampling."""

from __future__ import annotations

import numpy as np
import pytest

from rokko_geofusion.crs import CrsManager, GridSpec, RoiGeometry
from rokko_geofusion.exceptions import UnsupportedError
from rokko_geofusion.io.raster import read_raster, write_grid_raster
from rokko_geofusion.lidar.pointcloud import (
    build_point_cloud,
    cell_centers_in,
    choose_surface,
    iter_tiles,
    read_point_cloud,
    subsample_for_display,
    voxel_downsample,
)


# --- tiling -----------------------------------------------------------------
def test_tiles_cover_the_grid_exactly_once():
    grid = GridSpec((0.0, 0.0, 1000.0, 1000.0), 5.0, "EPSG:6673")
    tiles = list(iter_tiles(grid, 300.0))
    area = sum((t[2] - t[0]) * (t[3] - t[1]) for t in tiles)
    assert area == pytest.approx(1000.0 * 1000.0)
    assert min(t[0] for t in tiles) == 0.0
    assert max(t[2] for t in tiles) == 1000.0


def test_cell_centers_partition_the_grid():
    grid = GridSpec((0.0, 0.0, 100.0, 100.0), 5.0, "EPSG:6673")
    collected = []
    for bounds in iter_tiles(grid, 30.0):
        x, y = cell_centers_in(grid, bounds)
        collected.append(np.column_stack([x, y]))
    points = np.vstack(collected)
    assert len(points) == grid.width * grid.height
    assert len(np.unique(points, axis=0)) == len(points)


def test_tile_size_smaller_than_a_cell_is_clamped():
    grid = GridSpec((0.0, 0.0, 50.0, 50.0), 10.0, "EPSG:6673")
    tiles = list(iter_tiles(grid, 1.0))
    assert all((t[2] - t[0]) >= 10.0 for t in tiles)


# --- downsampling -----------------------------------------------------------
def test_voxel_downsample_reduces_and_is_deterministic():
    rng = np.random.default_rng(0)
    xyz = rng.uniform(0, 10, size=(5000, 3))
    a, _ = voxel_downsample(xyz, voxel_size_m=1.0)
    b, _ = voxel_downsample(xyz, voxel_size_m=1.0)
    assert len(a) < len(xyz)
    np.testing.assert_array_equal(a, b)
    # One point per 1 m voxel, at most.
    assert len(a) <= 10 * 10 * 10


def test_voxel_downsample_keeps_attributes_aligned():
    xyz = np.array([[0.1, 0.1, 0.1], [0.2, 0.2, 0.2], [5.0, 5.0, 5.0]])
    attributes = np.array([[10], [20], [30]])
    kept, kept_attributes = voxel_downsample(xyz, voxel_size_m=1.0, attributes=attributes)
    assert len(kept) == 2
    assert kept_attributes.tolist() == [[10], [30]]


def test_voxel_downsample_rejects_bad_size():
    with pytest.raises(ValueError):
        voxel_downsample(np.zeros((3, 3)), voxel_size_m=0.0)


def test_subsample_for_display_is_deterministic_and_capped():
    xyz = np.arange(300).reshape(100, 3).astype(float)
    a, _ = subsample_for_display(xyz, None, 25, seed=7)
    b, _ = subsample_for_display(xyz, None, 25, seed=7)
    assert len(a) == 25
    np.testing.assert_array_equal(a, b)
    # Different seed -> different sample, same size.
    c, _ = subsample_for_display(xyz, None, 25, seed=8)
    assert len(c) == 25


def test_subsample_below_the_cap_is_a_no_op():
    xyz = np.zeros((10, 3))
    out, attributes = subsample_for_display(xyz, np.ones((10, 3)), 100)
    assert len(out) == 10
    assert attributes is not None


# --- building ---------------------------------------------------------------
@pytest.fixture()
def tiny_scene(config, tmp_path):
    """A 100 m ROI with a synthetic tilted DEM and a two-colour orthophoto."""
    config.roi.name = "tiny"
    config.roi.radius_m = 50.0
    config.pointcloud.chunk_size_m = 40.0
    config.paths.ensure()

    roi = RoiGeometry.from_config(
        config.roi, CrsManager(config.crs), grid_snap_m=config.crs.grid_snap_m
    )
    dem_grid = roi.grid(config.lidar.resolution_m)
    rows, cols = np.mgrid[0:dem_grid.height, 0:dem_grid.width]
    dem = (100.0 + cols * 0.5).astype(np.float32)
    dem_path = write_grid_raster(config.paths.interim / "dem.tif", dem, dem_grid,
                                 nodata=float("nan"))

    rgb_grid = roi.grid(config.imagery.resolution_m)
    rgb = np.zeros((3, rgb_grid.height, rgb_grid.width), np.uint8)
    rgb[0, :, : rgb_grid.width // 2] = 255     # west half red
    rgb[2, :, rgb_grid.width // 2:] = 255      # east half blue
    ortho_path = write_grid_raster(config.paths.interim / "orthophoto.tif", rgb, rgb_grid,
                                   nodata=None)
    return config, roi, dem_path, ortho_path, dem_grid


def test_build_point_cloud_produces_one_point_per_cell(tiny_scene):
    config, roi, dem_path, ortho_path, dem_grid = tiny_scene
    product = build_point_cloud(config, roi, dem_path=dem_path, imagery_path=ortho_path)
    assert product.n_points == dem_grid.width * dem_grid.height
    assert product.surface == "dem"
    assert product.has_rgb is True
    assert product.is_true_lidar is False


def test_built_cloud_roundtrips_with_correct_geometry_and_colour(tiny_scene):
    config, roi, dem_path, ortho_path, dem_grid = tiny_scene
    product = build_point_cloud(config, roi, dem_path=dem_path, imagery_path=ortho_path)
    xyz, rgb = read_point_cloud(product.path)

    assert len(xyz) == product.n_points
    assert rgb is not None and rgb.dtype == np.uint8
    # Geometry: exactly the grid's cell centres.
    xs, ys = dem_grid.cell_centers()
    np.testing.assert_allclose(np.unique(np.round(xyz[:, 0], 3)), np.sort(xs), atol=1e-3)
    np.testing.assert_allclose(np.unique(np.round(xyz[:, 1], 3)), np.sort(np.unique(ys)),
                               atol=1e-3)
    # Z follows the synthetic west-to-east ramp.
    west = xyz[:, 0] < np.median(xyz[:, 0])
    assert xyz[west, 2].mean() < xyz[~west, 2].mean()
    # Colour: the west half is red, the east half blue. No point is unsampled.
    assert int((rgb.sum(axis=1) == 0).sum()) == 0
    assert rgb[west][:, 0].mean() > 200
    assert rgb[~west][:, 2].mean() > 200


def test_no_edge_point_is_left_uncoloured(tiny_scene):
    """Regression: nested grids must cover every coarse cell centre (V5)."""
    config, roi, dem_path, ortho_path, _ = tiny_scene
    product = build_point_cloud(config, roi, dem_path=dem_path, imagery_path=ortho_path)
    xyz, rgb = read_point_cloud(product.path)
    on_edge = (
        (xyz[:, 0] == xyz[:, 0].max())
        | (xyz[:, 1] == xyz[:, 1].min())
        | (xyz[:, 0] == xyz[:, 0].min())
        | (xyz[:, 1] == xyz[:, 1].max())
    )
    assert on_edge.any()
    assert int((rgb[on_edge].sum(axis=1) == 0).sum()) == 0


def test_nodata_cells_do_not_become_points(tiny_scene):
    config, roi, dem_path, ortho_path, dem_grid = tiny_scene
    array, _ = read_raster(dem_path, band=1)
    array[0, :] = np.nan
    write_grid_raster(dem_path, array, dem_grid, nodata=float("nan"))
    product = build_point_cloud(config, roi, dem_path=dem_path, imagery_path=ortho_path,
                                overwrite=True)
    assert product.n_points == dem_grid.width * (dem_grid.height - 1)
    assert product.metadata["processing"]["cells_without_elevation"] == dem_grid.width


@pytest.mark.parametrize("fmt", ["parquet", "npz", "laz"])
def test_every_output_format_roundtrips(tiny_scene, fmt):
    config, roi, dem_path, ortho_path, _ = tiny_scene
    config.output.pointcloud_format = fmt
    product = build_point_cloud(config, roi, dem_path=dem_path, imagery_path=ortho_path)
    xyz, rgb = read_point_cloud(product.path)
    assert len(xyz) == product.n_points
    assert rgb is not None and len(rgb) == product.n_points


def test_cloud_without_colour(tiny_scene):
    config, roi, dem_path, _, _ = tiny_scene
    config.pointcloud.colorize = False
    product = build_point_cloud(config, roi, dem_path=dem_path)
    assert product.has_rgb is False
    _, rgb = read_point_cloud(product.path)
    assert rgb is None or int(rgb.sum()) == 0


def test_rebuild_is_reproducible(tiny_scene):
    """V6: the same ROI and config must yield an identical cloud."""
    config, roi, dem_path, ortho_path, _ = tiny_scene
    first = build_point_cloud(config, roi, dem_path=dem_path, imagery_path=ortho_path)
    xyz_a, rgb_a = read_point_cloud(first.path)
    second = build_point_cloud(config, roi, dem_path=dem_path, imagery_path=ortho_path,
                               overwrite=True)
    xyz_b, rgb_b = read_point_cloud(second.path)
    np.testing.assert_array_equal(xyz_a, xyz_b)
    np.testing.assert_array_equal(rgb_a, rgb_b)


def test_point_budget_is_enforced(tiny_scene):
    config, roi, dem_path, ortho_path, _ = tiny_scene
    config.pointcloud.resolution_m = 0.5
    config.pointcloud.max_points = 100
    with pytest.raises(UnsupportedError, match="max_points"):
        build_point_cloud(config, roi, dem_path=dem_path, imagery_path=ortho_path)


def test_choose_surface_prefers_dsm_then_dem(tiny_scene, tmp_path):
    config, _, dem_path, _, _ = tiny_scene
    path, kind = choose_surface(config, dem_path, None)
    assert kind == "dem" and path == dem_path

    dsm_path = tmp_path / "dsm.tif"
    dsm_path.write_bytes(b"placeholder")
    assert choose_surface(config, dem_path, dsm_path)[1] == "dsm"

    config.pointcloud.surface = "dem"
    assert choose_surface(config, dem_path, dsm_path)[1] == "dem"

    config.pointcloud.surface = "dsm"
    with pytest.raises(UnsupportedError, match="no DSM exists"):
        choose_surface(config, dem_path, None)


def test_choose_surface_without_any_elevation(config):
    with pytest.raises(UnsupportedError):
        choose_surface(config, None, None)
