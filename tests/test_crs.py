"""CRS handling and ROI geometry -- the foundation every other stage rests on."""

from __future__ import annotations

import math

import numpy as np
import pytest

from rokko_geofusion.config import CrsConfig, RoiConfig
from rokko_geofusion.crs import (
    CrsManager,
    GridSpec,
    RoiGeometry,
    densify_bounds,
    roi_from_config,
    snap_bounds,
)
from rokko_geofusion.exceptions import CrsError


@pytest.fixture()
def manager(config) -> CrsManager:
    return CrsManager(config.crs)


@pytest.fixture()
def roi(config) -> RoiGeometry:
    return roi_from_config(config)


# --- transformations --------------------------------------------------------
def test_geographic_to_projected_roundtrip_is_sub_millimetre(manager):
    lon, lat = 135.2348, 34.7284
    x, y = manager.geographic_to_projected(lon, lat)
    lon2, lat2 = manager.projected_to_geographic(x, y)
    # 1e-9 degrees is well under 1 mm at this latitude.
    assert lon2 == pytest.approx(lon, abs=1e-9)
    assert lat2 == pytest.approx(lat, abs=1e-9)


def test_roundtrip_over_the_whole_roi_outline(manager, roi):
    points = np.array(list(densify_bounds(roi.bounds_projected, steps=25)))
    back = manager.transform_points(
        manager.transform_points(points, manager.projected, manager.geographic),
        manager.geographic,
        manager.projected,
    )
    residual = np.abs(back - points).max()
    assert residual < 1e-3, f"round-trip error {residual} m exceeds 1 mm"


def test_projected_coordinates_are_metric_and_plausible(manager):
    """EPSG:6673 origin is 36N / 134-20E; Kobe must land SE of it."""
    x, y = manager.geographic_to_projected(135.2348, 34.7284)
    assert 70_000 < x < 95_000, x      # ~0.9 deg east of the central meridian
    assert -150_000 < y < -130_000, y  # ~1.27 deg south of 36N
    # 1 degree of latitude is ~111 km: check the scale is really metres.
    _, y2 = manager.geographic_to_projected(135.2348, 35.7284)
    assert abs(y2 - y) == pytest.approx(110_800, rel=0.02)


def test_distance_in_projected_crs_matches_geodesic(manager):
    from pyproj import Geod

    lon0, lat0 = 135.2348, 34.7284
    lon1, lat1 = 135.2448, 34.7384
    x0, y0 = manager.geographic_to_projected(lon0, lat0)
    x1, y1 = manager.geographic_to_projected(lon1, lat1)
    planar = math.hypot(x1 - x0, y1 - y0)
    geodesic = Geod(ellps="GRS80").inv(lon0, lat0, lon1, lat1)[2]
    assert planar == pytest.approx(geodesic, rel=2e-4)


def test_transform_points_preserves_z(manager):
    points = np.array([[135.2348, 34.7284, 123.5], [135.2350, 34.7286, 200.0]])
    out = manager.transform_points(points, manager.geographic, manager.projected)
    assert out.shape == (2, 3)
    np.testing.assert_allclose(out[:, 2], points[:, 2])


def test_transform_points_rejects_bad_shape(manager):
    with pytest.raises(CrsError):
        manager.transform_points(np.zeros((4, 5)), manager.geographic, manager.projected)


def test_same_crs_transform_is_identity(manager):
    points = np.array([[1.0, 2.0], [3.0, 4.0]])
    np.testing.assert_array_equal(
        manager.transform_points(points, manager.projected, manager.projected), points
    )


def test_geographic_bbox_encloses_the_projected_square(manager, roi):
    """The geographic bbox is the *envelope* of the rotated projected square.

    Its own corners therefore fall slightly outside the projected square --
    that is correct, and the containment we must guarantee is this direction:
    no part of the analysis square may be missing from the download bbox.
    """
    min_x, min_y, max_x, max_y = roi.bounds_projected
    corners = [(min_x, min_y), (max_x, max_y), (min_x, max_y), (max_x, min_y)]
    for x, y in corners:
        lon, lat = manager.projected_to_geographic(x, y)
        assert roi.bounds_geographic[0] <= lon <= roi.bounds_geographic[2]
        assert roi.bounds_geographic[1] <= lat <= roi.bounds_geographic[3]


def test_geographic_bbox_overshoot_is_bounded(manager, roi):
    """The envelope should not be wildly larger than the square it wraps."""
    back = manager.transform_bounds(
        roi.bounds_geographic, manager.geographic, manager.projected
    )
    overshoot = max(
        roi.bounds_projected[0] - back[0],
        roi.bounds_projected[1] - back[1],
        back[2] - roi.bounds_projected[2],
        back[3] - roi.bounds_projected[3],
    )
    assert 0.0 <= overshoot < 0.05 * roi.width_m


def test_misconfigured_crs_roles_are_rejected():
    with pytest.raises(CrsError):
        CrsManager(CrsConfig(geographic="EPSG:4326", projected="EPSG:4326"))
    with pytest.raises(CrsError):
        CrsManager(CrsConfig(geographic="EPSG:6673", projected="EPSG:6673"))
    with pytest.raises(CrsError):
        CrsManager(CrsConfig(geographic="EPSG:4326", projected="NOT-A-CRS"))


# --- ROI geometry -----------------------------------------------------------
def test_roi_square_matches_requested_radius(roi, config):
    assert roi.width_m == pytest.approx(2 * config.roi.radius_m)
    assert roi.height_m == pytest.approx(2 * config.roi.radius_m)
    assert roi.area_m2 == pytest.approx((2 * config.roi.radius_m) ** 2)


def test_roi_center_is_the_configured_point(roi, config, manager):
    lon, lat = manager.projected_to_geographic(*roi.center_projected)
    assert lon == pytest.approx(config.roi.center.lon, abs=1e-9)
    assert lat == pytest.approx(config.roi.center.lat, abs=1e-9)


def test_roi_from_bbox_agrees_with_center_form(manager):
    roi_bbox = RoiGeometry.from_config(
        RoiConfig(name="bb", bbox=[135.22, 34.72, 135.25, 34.74]), manager
    )
    assert roi_bbox.bounds_geographic == (135.22, 34.72, 135.25, 34.74)
    assert roi_bbox.width_m > 2000  # ~0.03 deg lon at 34.7N is ~2.7 km
    assert roi_bbox.center_geographic[0] == pytest.approx(135.235)


def test_buffered_roi_grows_symmetrically(roi):
    grown = roi.buffered(100.0)
    assert grown.width_m == pytest.approx(roi.width_m + 200.0)
    assert grown.bounds_projected[0] == pytest.approx(roi.bounds_projected[0] - 100.0)
    assert grown.center_projected == roi.center_projected


def test_contains_projected(roi):
    cx, cy = roi.center_projected
    inside = np.array([cx, cx + 100.0])
    outside_y = np.array([cy, cy + 10_000.0])
    mask = roi.contains_projected(inside, outside_y)
    assert mask.tolist() == [True, False]


def test_roi_to_dict_is_serialisable(roi):
    import json

    payload = json.loads(json.dumps(roi.to_dict()))
    assert payload["crs"]["projected"] == roi.crs.projected
    assert len(payload["bounds_projected"]) == 4


# --- grids ------------------------------------------------------------------
@pytest.mark.parametrize("resolution", [0.5, 1.0, 5.0, 10.0])
def test_snap_bounds_lands_on_multiples_and_never_shrinks(resolution):
    bounds = (81556.3, -141701.7, 83559.1, -139694.2)
    snapped = snap_bounds(bounds, resolution)
    for value in snapped:
        assert value % resolution == pytest.approx(0.0, abs=1e-9)
    assert snapped[0] <= bounds[0] and snapped[1] <= bounds[1]
    assert snapped[2] >= bounds[2] and snapped[3] >= bounds[3]


def test_grid_shape_and_transform_are_consistent(roi):
    grid = roi.grid(5.0)
    assert grid.shape == (grid.height, grid.width)
    transform = grid.transform
    assert transform.a == pytest.approx(5.0)
    assert transform.e == pytest.approx(-5.0)
    assert transform.c == pytest.approx(grid.bounds[0])
    assert transform.f == pytest.approx(grid.bounds[3])
    # Bottom-right pixel corner must land exactly on the snapped bounds.
    x, y = transform * (grid.width, grid.height)
    assert x == pytest.approx(grid.bounds[2])
    assert y == pytest.approx(grid.bounds[1])


def test_cell_centers_are_inside_the_grid(roi):
    grid = roi.grid(5.0)
    xs, ys = grid.cell_centers()
    assert xs.size == grid.width and ys.size == grid.height
    assert grid.bounds[0] < xs[0] < xs[-1] < grid.bounds[2]
    assert grid.bounds[1] < ys[-1] < ys[0] < grid.bounds[3]
    assert xs[1] - xs[0] == pytest.approx(5.0)
    assert ys[0] - ys[1] == pytest.approx(5.0)


def test_grids_of_different_rois_share_one_lattice(manager):
    """V2 by construction: DSM and DEM tiles always align."""
    a = RoiGeometry.from_config(
        RoiConfig(name="a", center={"lat": 34.7284, "lon": 135.2348}, radius_m=1000), manager
    ).grid(5.0)
    b = RoiGeometry.from_config(
        RoiConfig(name="b", center={"lat": 34.7301, "lon": 135.2372}, radius_m=700), manager
    ).grid(5.0)
    assert (a.bounds[0] - b.bounds[0]) % 5.0 == pytest.approx(0.0, abs=1e-9)
    assert (a.bounds[3] - b.bounds[3]) % 5.0 == pytest.approx(0.0, abs=1e-9)


def test_grid_matches_is_strict():
    base = GridSpec((0.0, 0.0, 100.0, 100.0), 1.0, "EPSG:6673")
    assert base.matches(GridSpec((0.0, 0.0, 100.0, 100.0), 1.0, "EPSG:6673"))
    assert not base.matches(GridSpec((0.0, 0.0, 100.0, 100.0), 2.0, "EPSG:6673"))
    assert not base.matches(GridSpec((1.0, 0.0, 101.0, 100.0), 1.0, "EPSG:6673"))
    assert not base.matches(GridSpec((0.0, 0.0, 100.0, 100.0), 1.0, "EPSG:3857"))


def test_grid_is_deterministic_across_runs(config):
    """V6: the same config must produce byte-identical grid definitions."""
    first = roi_from_config(config).grid(config.lidar.resolution_m).to_dict()
    second = roi_from_config(config).grid(config.lidar.resolution_m).to_dict()
    assert first == second
