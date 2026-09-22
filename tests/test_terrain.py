"""Terrain derivatives, verified against analytically known surfaces."""

from __future__ import annotations

import numpy as np
import pytest

from rokko_geofusion.crs import GridSpec
from rokko_geofusion.exceptions import AlignmentError
from rokko_geofusion.terrain.analysis import (
    ASPECT_SECTORS,
    aspect,
    aspect_distribution,
    hillshade,
    normalised_dsm,
    point_density,
    relief,
    slope,
    terrain_statistics,
)

CELL = 5.0
CRS = "EPSG:6673"


def _plane(gradient_east: float, gradient_north: float, size: int = 21) -> np.ndarray:
    """z = ge*x + gn*y on a north-up grid (row 0 = north)."""
    rows, cols = np.mgrid[0:size, 0:size].astype(float)
    x = cols * CELL
    y = (size - 1 - rows) * CELL  # northing grows upwards, row index downwards
    return (gradient_east * x + gradient_north * y).astype(np.float32)


def _interior(array: np.ndarray) -> np.ndarray:
    """Drop the replicated edge ring, where Horn's operator is approximate."""
    return array[1:-1, 1:-1]


# --- slope ------------------------------------------------------------------
@pytest.mark.parametrize("gradient", [0.0, 0.1, 0.5, 1.0, 2.0])
def test_slope_of_a_plane_matches_the_analytic_value(gradient):
    z = _plane(gradient, 0.0)
    expected = np.degrees(np.arctan(gradient))
    np.testing.assert_allclose(_interior(slope(z, CELL)), expected, atol=1e-4)


def test_slope_is_direction_independent():
    east = slope(_plane(0.3, 0.0), CELL)
    north = slope(_plane(0.0, 0.3), CELL)
    diagonal = slope(_plane(0.3 / np.sqrt(2), 0.3 / np.sqrt(2)), CELL)
    assert _interior(east).mean() == pytest.approx(_interior(north).mean(), abs=1e-4)
    assert _interior(diagonal).mean() == pytest.approx(_interior(east).mean(), abs=1e-4)


def test_slope_in_percent():
    z = _plane(0.25, 0.0)
    np.testing.assert_allclose(_interior(slope(z, CELL, units="percent")), 25.0, atol=1e-4)


def test_slope_rejects_unknown_units():
    with pytest.raises(ValueError):
        slope(_plane(0.1, 0.0), CELL, units="radians")


def test_slope_depends_on_cell_size():
    z = _plane(0.2, 0.0)
    coarse = _interior(slope(z, CELL)).mean()
    fine = _interior(slope(z, CELL / 2)).mean()
    assert fine > coarse  # same height change over half the distance


def test_slope_propagates_nan():
    z = _plane(0.2, 0.0)
    z[5, 5] = np.nan
    assert np.isnan(slope(z, CELL)[5, 5])


# --- aspect -----------------------------------------------------------------
@pytest.mark.parametrize(
    ("gradient_east", "gradient_north", "expected_deg"),
    [
        (0.3, 0.0, 270.0),    # rises to the east  -> faces west
        (-0.3, 0.0, 90.0),    # rises to the west  -> faces east
        (0.0, 0.3, 180.0),    # rises to the north -> faces south
        (0.0, -0.3, 0.0),     # rises to the south -> faces north
        (0.3, 0.3, 225.0),    # rises to the NE    -> faces SW
    ],
)
def test_aspect_points_downhill(gradient_east, gradient_north, expected_deg):
    values = _interior(aspect(_plane(gradient_east, gradient_north), CELL))
    difference = np.abs((values - expected_deg + 180.0) % 360.0 - 180.0)
    assert difference.max() < 1e-3


def test_flat_ground_has_no_aspect():
    flat = np.full((11, 11), 42.0, np.float32)
    assert np.isnan(aspect(flat, CELL)).all()
    np.testing.assert_allclose(_interior(slope(flat, CELL)), 0.0, atol=1e-9)


def test_aspect_stays_within_the_compass_range():
    rng = np.random.default_rng(3)
    noisy = rng.normal(100.0, 5.0, size=(30, 30)).astype(np.float32)
    values = aspect(noisy, CELL)
    finite = values[np.isfinite(values)]
    assert finite.min() >= 0.0 and finite.max() < 360.0


def test_aspect_distribution_sums_to_one():
    rng = np.random.default_rng(1)
    values = rng.uniform(0, 360, size=5000)
    distribution = aspect_distribution(values)
    assert set(distribution) == set(ASPECT_SECTORS)
    assert sum(distribution.values()) == pytest.approx(1.0)
    # Uniform input -> roughly equal sectors.
    assert max(distribution.values()) < 0.2


def test_aspect_distribution_sectors_are_centred():
    assert aspect_distribution(np.array([0.0, 359.0, 10.0]))["N"] == pytest.approx(1.0)
    assert aspect_distribution(np.array([90.0]))["E"] == pytest.approx(1.0)
    assert aspect_distribution(np.array([225.0]))["SW"] == pytest.approx(1.0)


# --- relief / hillshade -----------------------------------------------------
def test_relief_on_a_ramp_equals_the_window_drop():
    z = _plane(0.2, 0.0, size=31)
    values = relief(z, 5)
    # Over a 5-cell window the ramp rises 4 * gradient * cell.
    assert values[10, 10] == pytest.approx(4 * 0.2 * CELL, abs=1e-3)


def test_relief_is_zero_on_flat_ground():
    values = relief(np.full((15, 15), 7.0, np.float32), 3)
    np.testing.assert_allclose(values, 0.0, atol=1e-6)


def test_relief_rejects_even_windows():
    with pytest.raises(ValueError):
        relief(np.zeros((9, 9), np.float32), 4)


def test_hillshade_is_a_display_range():
    values = hillshade(_plane(0.3, 0.1), CELL)
    finite = values[np.isfinite(values)]
    assert finite.min() >= 0.0 and finite.max() <= 255.0
    # Default light comes from the north-west (azimuth 315 deg). A plane that
    # rises to the east and falls to the north faces NW (aspect 315) and must
    # be lit; the opposite plane faces SE and must be in shadow.
    faces_north_west = _plane(0.4, -0.4)
    faces_south_east = _plane(-0.4, 0.4)
    assert _interior(aspect(faces_north_west, CELL)).mean() == pytest.approx(315.0, abs=1e-3)
    assert _interior(aspect(faces_south_east, CELL)).mean() == pytest.approx(135.0, abs=1e-3)
    lit = _interior(hillshade(faces_north_west, CELL)).mean()
    shadowed = _interior(hillshade(faces_south_east, CELL)).mean()
    assert lit > 200.0 > shadowed


# --- nDSM -------------------------------------------------------------------
def test_ndsm_is_the_difference():
    dem = np.full((5, 5), 100.0, np.float32)
    dsm = dem + 7.5
    np.testing.assert_allclose(normalised_dsm(dsm, dem), 7.5)


def test_ndsm_clips_negative_and_extreme_values():
    dem = np.full((3, 3), 100.0, np.float32)
    dsm = np.array([[95.0, 100.0, 103.0],
                    [100.0, 400.0, 100.0],
                    [100.0, 100.0, 100.0]], np.float32)
    values = normalised_dsm(dsm, dem, clip_min=0.0, clip_max=150.0)
    assert values[0, 0] == pytest.approx(0.0)    # DSM below DEM -> clipped
    assert values[0, 2] == pytest.approx(3.0)
    assert values[1, 1] == pytest.approx(150.0)  # absurd height -> clipped


def test_ndsm_propagates_nan():
    dem = np.full((3, 3), 100.0, np.float32)
    dsm = dem.copy()
    dsm[1, 1] = np.nan
    assert np.isnan(normalised_dsm(dsm, dem)[1, 1])


def test_ndsm_requires_matching_shapes():
    with pytest.raises(AlignmentError):
        normalised_dsm(np.zeros((4, 4), np.float32), np.zeros((5, 5), np.float32))


# --- density ----------------------------------------------------------------
def test_point_density_counts_points_per_cell():
    grid = GridSpec((0.0, 0.0, 30.0, 30.0), 10.0, CRS)
    x = np.array([5.0, 6.0, 15.0, 25.0])
    y = np.array([25.0, 25.0, 15.0, 5.0])   # north-west, north-west, centre, south-east
    counts = point_density(x, y, grid)
    assert counts.shape == (3, 3)
    assert counts[0, 0] == 2.0
    assert counts[1, 1] == 1.0
    assert counts[2, 2] == 1.0
    assert counts.sum() == 4.0


def test_point_density_ignores_points_outside_the_grid():
    grid = GridSpec((0.0, 0.0, 10.0, 10.0), 10.0, CRS)
    counts = point_density(np.array([5.0, 500.0]), np.array([5.0, 500.0]), grid)
    assert counts.sum() == 1.0


# --- statistics -------------------------------------------------------------
@pytest.fixture()
def stats_inputs():
    grid = GridSpec((0.0, 0.0, 100.0, 100.0), 10.0, CRS)
    elevation = _plane(0.1, 0.0, size=10)
    return grid, elevation


def test_statistics_report_area_and_ranges(stats_inputs):
    grid, elevation = stats_inputs
    statistics = terrain_statistics(
        grid=grid,
        elevation=elevation,
        slope_deg=slope(elevation, grid.resolution_m),
        aspect_deg=aspect(elevation, grid.resolution_m),
        relief_m=relief(elevation, 3),
    )
    assert statistics["cells"] == 100
    assert statistics["area_m2"] == pytest.approx(100 * 100.0)
    assert statistics["elevation"]["min"] == pytest.approx(float(elevation.min()))
    assert statistics["elevation"]["max"] == pytest.approx(float(elevation.max()))
    assert statistics["slope"]["mean"] > 0
    assert statistics["object_height"] is None
    assert statistics["coverage"] == pytest.approx(1.0)


def test_statistics_respect_a_mask(stats_inputs):
    grid, elevation = stats_inputs
    mask = np.zeros(elevation.shape, bool)
    mask[:5, :5] = True
    statistics = terrain_statistics(
        grid=grid,
        elevation=elevation,
        slope_deg=slope(elevation, grid.resolution_m),
        aspect_deg=aspect(elevation, grid.resolution_m),
        mask=mask,
    )
    assert statistics["cells"] == 25
    assert statistics["area_m2"] == pytest.approx(25 * 100.0)
    assert statistics["elevation"]["max"] <= float(elevation[mask].max())


def test_statistics_record_unavailable_products(stats_inputs):
    grid, elevation = stats_inputs
    statistics = terrain_statistics(
        grid=grid,
        elevation=elevation,
        slope_deg=slope(elevation, grid.resolution_m),
        aspect_deg=aspect(elevation, grid.resolution_m),
        unavailable={"ndsm": "no DSM configured"},
    )
    assert statistics["object_height"] is None
    assert "no DSM configured" in statistics["unavailable"]["ndsm"]


def test_statistics_with_object_height(stats_inputs):
    grid, elevation = stats_inputs
    heights = np.full(elevation.shape, 3.0, np.float32)
    heights[0, 0] = 25.0
    statistics = terrain_statistics(
        grid=grid,
        elevation=elevation,
        slope_deg=slope(elevation, grid.resolution_m),
        aspect_deg=aspect(elevation, grid.resolution_m),
        object_height=heights,
    )
    assert statistics["object_height"]["p50"] == pytest.approx(3.0)
    assert statistics["object_height"]["max"] == pytest.approx(25.0)


def test_statistics_on_an_all_nan_region(stats_inputs):
    grid, elevation = stats_inputs
    empty = np.full(elevation.shape, np.nan, np.float32)
    statistics = terrain_statistics(
        grid=grid, elevation=empty, slope_deg=empty, aspect_deg=empty
    )
    assert statistics["elevation"] is None
    assert statistics["coverage"] == 0.0
