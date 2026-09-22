"""Raster sampling at point coordinates (the image -> point-cloud colour path)."""

from __future__ import annotations

import numpy as np
import pytest

from rokko_geofusion.crs import GridSpec
from rokko_geofusion.exceptions import CrsError
from rokko_geofusion.imagery.sampler import sample_raster, sample_rgb
from rokko_geofusion.io.raster import write_grid_raster

CRS = "EPSG:6673"


@pytest.fixture()
def ramp_raster(tmp_path):
    """10x10 cells of 1 m, value == column index, origin at (1000, 2000)."""
    grid = GridSpec((1000.0, 2000.0, 1010.0, 2010.0), 1.0, CRS)
    array = np.tile(np.arange(10, dtype=np.float32), (10, 1))
    path = write_grid_raster(tmp_path / "ramp.tif", array, grid, nodata=float("nan"))
    return path, grid


@pytest.fixture()
def rgb_raster(tmp_path):
    grid = GridSpec((0.0, 0.0, 4.0, 4.0), 1.0, CRS)
    red = np.zeros((4, 4), np.uint8)
    red[0, 0] = 255       # north-west cell
    green = np.zeros((4, 4), np.uint8)
    green[3, 3] = 200     # south-east cell
    blue = np.full((4, 4), 7, np.uint8)
    array = np.stack([red, green, blue])
    return write_grid_raster(tmp_path / "rgb.tif", array, grid, nodata=None), grid


def test_nearest_sampling_hits_the_expected_cell(ramp_raster):
    path, _ = ramp_raster
    # Cell centres are at 1000.5, 1001.5, ...
    x = np.array([1000.5, 1003.5, 1009.5])
    y = np.array([2009.5, 2005.5, 2000.5])
    values = sample_raster(path, x, y, expected_crs=CRS)
    np.testing.assert_allclose(values[0], [0.0, 3.0, 9.0])


def test_sampling_anywhere_inside_a_cell_returns_that_cell(ramp_raster):
    path, _ = ramp_raster
    for offset in (0.01, 0.5, 0.99):
        value = sample_raster(path, np.array([1003.0 + offset]), np.array([2005.0 + offset]),
                              expected_crs=CRS)
        assert value[0, 0] == pytest.approx(3.0)


def test_bilinear_interpolates_between_cell_centres(ramp_raster):
    path, _ = ramp_raster
    # Exactly between the centres of columns 3 and 4.
    value = sample_raster(path, np.array([1004.0]), np.array([2005.5]),
                          expected_crs=CRS, method="bilinear")
    assert value[0, 0] == pytest.approx(3.5)


def test_bilinear_at_a_cell_centre_equals_nearest(ramp_raster):
    path, _ = ramp_raster
    x, y = np.array([1006.5]), np.array([2003.5])
    assert sample_raster(path, x, y, expected_crs=CRS, method="bilinear")[0, 0] == pytest.approx(
        sample_raster(path, x, y, expected_crs=CRS)[0, 0]
    )


def test_points_outside_the_raster_get_the_fill_value(ramp_raster):
    path, _ = ramp_raster
    values = sample_raster(path, np.array([0.0, 1005.5, 99999.0]),
                           np.array([0.0, 2005.5, 99999.0]), expected_crs=CRS)
    assert np.isnan(values[0, 0])
    assert values[0, 1] == pytest.approx(5.0)
    assert np.isnan(values[0, 2])


def test_all_points_outside_returns_all_fill(ramp_raster):
    path, _ = ramp_raster
    values = sample_raster(path, np.array([0.0, 1.0]), np.array([0.0, 1.0]), expected_crs=CRS)
    assert np.isnan(values).all()


def test_nan_nodata_is_propagated(tmp_path):
    grid = GridSpec((0.0, 0.0, 3.0, 3.0), 1.0, CRS)
    array = np.full((3, 3), 5.0, np.float32)
    array[1, 1] = np.nan
    path = write_grid_raster(tmp_path / "holes.tif", array, grid, nodata=float("nan"))
    values = sample_raster(path, np.array([1.5, 0.5]), np.array([1.5, 2.5]), expected_crs=CRS)
    assert np.isnan(values[0, 0])
    assert values[0, 1] == pytest.approx(5.0)


def test_sentinel_nodata_becomes_nan(tmp_path):
    grid = GridSpec((0.0, 0.0, 2.0, 2.0), 1.0, CRS)
    array = np.array([[1.0, -9999.0], [2.0, 3.0]], dtype=np.float32)
    path = write_grid_raster(tmp_path / "sentinel.tif", array, grid, nodata=-9999.0)
    values = sample_raster(path, np.array([1.5]), np.array([1.5]), expected_crs=CRS)
    assert np.isnan(values[0, 0])


def test_crs_mismatch_is_refused(ramp_raster):
    path, _ = ramp_raster
    with pytest.raises(CrsError, match="CRS mismatch"):
        sample_raster(path, np.array([1000.5]), np.array([2000.5]), expected_crs="EPSG:4326")


def test_sample_rgb_returns_uint8_triples(rgb_raster):
    path, _ = rgb_raster
    rgb = sample_rgb(path, np.array([0.5, 3.5]), np.array([3.5, 0.5]), expected_crs=CRS)
    assert rgb.shape == (2, 3)
    assert rgb.dtype == np.uint8
    assert tuple(rgb[0]) == (255, 0, 7)   # NW cell
    assert tuple(rgb[1]) == (0, 200, 7)   # SE cell


def test_sample_rgb_outside_is_black(rgb_raster):
    path, _ = rgb_raster
    rgb = sample_rgb(path, np.array([-100.0]), np.array([-100.0]), expected_crs=CRS)
    assert tuple(rgb[0]) == (0, 0, 0)


def test_mismatched_x_y_lengths_are_rejected(ramp_raster):
    path, _ = ramp_raster
    with pytest.raises(ValueError):
        sample_raster(path, np.array([1.0, 2.0]), np.array([1.0]), expected_crs=CRS)


def test_windowed_read_does_not_change_results(ramp_raster):
    """A single point must sample the same value as the whole-array read."""
    path, _ = ramp_raster
    single = sample_raster(path, np.array([1007.5]), np.array([2002.5]), expected_crs=CRS)
    many = sample_raster(
        path,
        np.array([1000.5, 1007.5, 1009.5]),
        np.array([2009.5, 2002.5, 2000.5]),
        expected_crs=CRS,
    )
    assert single[0, 0] == pytest.approx(many[0, 1])
