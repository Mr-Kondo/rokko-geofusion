"""Display code: smoke tests that every renderer produces something valid.

Rendering is checked, not looked at: the point is that the notebook cells
cannot break silently after a refactor.
"""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from rokko_geofusion.crs import CrsManager, GridSpec, RoiGeometry  # noqa: E402
from rokko_geofusion.io.raster import write_grid_raster  # noqa: E402
from rokko_geofusion.lidar.pointcloud import build_point_cloud  # noqa: E402
from rokko_geofusion.visualization.maps import roi_map  # noqa: E402
from rokko_geofusion.visualization.plots import (  # noqa: E402
    CLASS_COLOURS,
    grid_figure,
    histogram,
    save_figure,
    show_classes,
    show_raster,
    show_rgb,
)
from rokko_geofusion.visualization.pointcloud3d import plot_point_cloud  # noqa: E402


@pytest.fixture()
def scene(config):
    config.roi.name = "viz"
    config.roi.radius_m = 40.0
    config.paths.ensure()
    roi = RoiGeometry.from_config(config.roi, CrsManager(config.crs),
                                  grid_snap_m=config.crs.grid_snap_m)
    grid = roi.grid(config.lidar.resolution_m)
    rows, cols = np.mgrid[0:grid.height, 0:grid.width]
    dem = (50.0 + rows * 0.8).astype(np.float32)
    dem_path = write_grid_raster(config.paths.interim / "dem.tif", dem, grid,
                                 nodata=float("nan"))
    rgb_grid = roi.grid(config.imagery.resolution_m)
    rgb = np.zeros((3, rgb_grid.height, rgb_grid.width), np.uint8)
    rgb[1] = 180
    rgb_path = write_grid_raster(config.paths.interim / "orthophoto.tif", rgb, rgb_grid,
                                 nodata=None)
    classes = (rows % 3).astype(np.uint8)
    class_path = write_grid_raster(config.paths.interim / "classes.tif", classes, grid,
                                   nodata=None)
    return config, roi, dem_path, rgb_path, class_path


def test_show_raster_returns_an_axis(scene):
    _, _, dem_path, _, _ = scene
    ax = show_raster(dem_path, title="DEM", colorbar_label="m")
    assert ax.get_title() == "DEM"
    assert ax.images


def test_show_raster_rejects_an_empty_raster(config, tmp_path):
    grid = GridSpec((0.0, 0.0, 10.0, 10.0), 1.0, config.crs.projected)
    path = write_grid_raster(tmp_path / "empty.tif", np.full((10, 10), np.nan, np.float32),
                             grid, nodata=float("nan"))
    with pytest.raises(ValueError):
        show_raster(path)


def test_show_rgb_needs_three_bands(scene):
    _, _, dem_path, rgb_path, _ = scene
    assert show_rgb(rgb_path).images
    with pytest.raises(ValueError, match="expected 3"):
        show_rgb(dem_path)


def test_show_classes_draws_a_legend(scene):
    _, _, _, _, class_path = scene
    ax = show_classes(class_path, ["other", "building", "road"])
    assert ax.get_legend() is not None


def test_every_project_class_has_a_colour():
    from rokko_geofusion.config import SegmentationConfig

    for name in SegmentationConfig().classes:
        assert name in CLASS_COLOURS


def test_grid_figure_handles_a_missing_panel(scene, tmp_path):
    _, _, dem_path, rgb_path, _ = scene
    figure = grid_figure(
        [
            {"path": rgb_path, "kind": "rgb", "title": "ortho"},
            {"path": dem_path, "title": "dem"},
            {"path": tmp_path / "does_not_exist.tif", "title": "missing"},
        ],
        ncols=3,
    )
    assert len(figure.axes) >= 3
    out = save_figure(figure, tmp_path / "panel.png", dpi=60)
    assert out.is_file() and out.stat().st_size > 0


def test_histogram_ignores_nan():
    ax = histogram(np.array([1.0, 2.0, np.nan, 3.0]), bins=4, title="t")
    assert ax.get_title() == "t"


def test_roi_map_contains_the_layers(scene):
    import geopandas as gpd
    from shapely.geometry import Polygon

    config, roi, *_ = scene
    min_x, min_y, max_x, max_y = roi.bounds_projected
    frame = gpd.GeoDataFrame(
        {"osm_id": ["way/1"], "name": ["test"],
         "geometry": [Polygon([(min_x + 5, min_y + 5), (min_x + 20, min_y + 5),
                               (min_x + 20, min_y + 20)])]},
        crs=config.crs.projected,
    )
    # `_repr_html_` wraps the map in an escaped iframe, so render the root.
    # Folium serialises with ensure_ascii=True, hence the JSON-escaped compare.
    import json

    html = roi_map(config, roi, layers={"building": frame}).get_root().render()
    assert "building (1)" in html
    assert json.dumps(config.visualization.basemap_attribution)[1:-1] in html
    assert config.visualization.basemap in html


def test_point_cloud_display_is_capped_and_deterministic(scene):
    config, roi, dem_path, rgb_path, _ = scene
    product = build_point_cloud(config, roi, dem_path=dem_path, imagery_path=rgb_path)
    assert product.n_points > 50

    figure = plot_point_cloud(config, product.path, max_points=50)
    trace = figure.data[0]
    assert len(trace.x) == 50
    again = plot_point_cloud(config, product.path, max_points=50)
    np.testing.assert_array_equal(trace.x, again.data[0].x)


def test_point_cloud_display_colour_modes(scene):
    config, roi, dem_path, rgb_path, _ = scene
    product = build_point_cloud(config, roi, dem_path=dem_path, imagery_path=rgb_path)
    rgb_figure = plot_point_cloud(config, product.path, color_by="rgb", max_points=20)
    assert isinstance(rgb_figure.data[0].marker.color, (list, tuple))
    z_figure = plot_point_cloud(config, product.path, color_by="z", max_points=20)
    assert len(z_figure.data[0].marker.color) == 20
