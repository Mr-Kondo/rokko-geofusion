"""Spatial validation checks (V1-V6 of the project brief).

"It ran" is not the acceptance criterion; "it is in the right place" is. Each
check returns a :class:`CheckResult` with a machine-readable measurement, so
the numbers quoted in the README and in configuration comments can always be
reproduced with ``python scripts/validate.py``.

A check that cannot run (for example anything needing a DSM, when no DSM is
configured) reports ``status="unavailable"`` with the reason. It is never
silently skipped and never reported as a pass.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from rokko_geofusion.config import Config
from rokko_geofusion.crs import RoiGeometry
from rokko_geofusion.io.raster import grid_from_raster, read_raster

logger = logging.getLogger(__name__)

Status = str  # "pass" | "fail" | "warn" | "unavailable"


@dataclass
class CheckResult:
    id: str
    title: str
    status: Status
    detail: str
    measurements: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def __str__(self) -> str:
        badge = {"pass": "PASS", "fail": "FAIL", "warn": "WARN",
                 "unavailable": "N/A "}.get(self.status, "????")
        return f"[{badge}] {self.id}: {self.title}\n        {self.detail}"


def _unavailable(check_id: str, title: str, reason: str) -> CheckResult:
    return CheckResult(check_id, title, "unavailable", reason)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def rasterize_layer(config: Config, layer: str, grid, *, buffer_m: float = 0.0) -> np.ndarray:
    """Burn a GIS layer onto ``grid`` as a boolean mask."""
    from rasterio.features import rasterize

    from rokko_geofusion.gis.osm import load_layer

    frame = load_layer(config, layer)
    geometries = [
        (geometry.buffer(buffer_m) if buffer_m else geometry, 1)
        for geometry in frame.geometry
        if geometry is not None and not geometry.is_empty
    ]
    if not geometries:
        return np.zeros(grid.shape, dtype=bool)
    return rasterize(
        geometries, out_shape=grid.shape, transform=grid.transform, fill=0, dtype="uint8"
    ).astype(bool)


def mask_agreement(predicted: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    """IoU / recall / precision between two boolean masks."""
    intersection = float(np.logical_and(predicted, reference).sum())
    union = float(np.logical_or(predicted, reference).sum())
    return {
        "iou": intersection / union if union else 0.0,
        "recall": intersection / float(reference.sum()) if reference.any() else 0.0,
        "precision": intersection / float(predicted.sum()) if predicted.any() else 0.0,
        "predicted_fraction": float(predicted.mean()),
        "reference_fraction": float(reference.mean()),
    }


def best_shift(
    predicted: np.ndarray, reference: np.ndarray, max_shift_cells: int
) -> tuple[int, int, float]:
    """Row/column shift of ``predicted`` that maximises overlap with ``reference``.

    A co-registered pair peaks at ``(0, 0)``; a systematic offset shows up as a
    non-zero peak, which is exactly what a CRS or grid mistake looks like.
    """
    best = (0, 0, -1.0)
    for d_row in range(-max_shift_cells, max_shift_cells + 1):
        rolled_rows = np.roll(predicted, d_row, axis=0)
        for d_col in range(-max_shift_cells, max_shift_cells + 1):
            rolled = np.roll(rolled_rows, d_col, axis=1)
            score = float(np.logical_and(rolled, reference).sum())
            if score > best[2]:
                best = (d_row, d_col, score)
    return best


def _downsample_bool(mask: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return mask
    height = (mask.shape[0] // factor) * factor
    width = (mask.shape[1] // factor) * factor
    block = mask[:height, :width].reshape(height // factor, factor, width // factor, factor)
    return block.mean(axis=(1, 3)) > 0.5


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------
def check_v1_imagery_gis_alignment(config: Config, roi: RoiGeometry) -> CheckResult:
    """V1: the orthophoto and the GIS vectors describe the same ground."""
    title = "orthophoto and GIS vectors are co-registered"
    class_path = config.paths.raster / "segmentation_class.tif"
    if not class_path.is_file():
        return _unavailable(
            "V1", title,
            "needs segmentation_class.tif (run scripts/segment_imagery.py); the "
            "check measures the offset between predicted and mapped buildings",
        )
    try:
        reference_all = rasterize_layer(config, "building", grid_from_raster(class_path))
    except FileNotFoundError as exc:
        return _unavailable("V1", title, str(exc))

    grid = grid_from_raster(class_path)
    class_index, _ = read_raster(class_path, band=1)
    predicted_all = class_index == config.segmentation.classes.index("building")
    if not reference_all.any() or not predicted_all.any():
        return _unavailable("V1", title, "no buildings in either the map or the prediction")

    # Search the offset on a 4 m grid: fine enough to spot a CRS mistake,
    # cheap enough to brute-force.
    factor = max(1, int(round(4.0 / grid.resolution_m)))
    predicted = _downsample_bool(predicted_all, factor)
    reference = _downsample_bool(reference_all, factor)
    d_row, d_col, _ = best_shift(predicted, reference, max_shift_cells=5)
    offset_m = float(np.hypot(d_row, d_col) * grid.resolution_m * factor)
    agreement = mask_agreement(predicted_all, reference_all)

    tolerance_m = 3 * grid.resolution_m * factor
    status = "pass" if offset_m <= tolerance_m else "fail"
    return CheckResult(
        "V1", title, status,
        f"best-overlap offset between predicted and OSM buildings is {offset_m:.1f} m "
        f"(row {d_row}, col {d_col} at {grid.resolution_m * factor:.1f} m); "
        f"IoU {agreement['iou']:.3f}, recall {agreement['recall']:.3f}, "
        f"precision {agreement['precision']:.3f}. "
        f"{'No systematic displacement.' if status == 'pass' else 'SYSTEMATIC DISPLACEMENT.'}",
        {"offset_m": offset_m, "offset_cells": [d_row, d_col],
         "tolerance_m": tolerance_m, **agreement},
    )


def check_v2_grid_alignment(config: Config, roi: RoiGeometry) -> CheckResult:
    """V2: every raster product shares one lattice."""
    title = "all raster products share a common grid lattice"
    candidates = [
        config.paths.interim / "dem.tif",
        config.paths.interim / "dsm.tif",
        config.paths.interim / "orthophoto.tif",
        config.paths.raster / "slope.tif",
        config.paths.raster / "segmentation_class.tif",
    ]
    present = [path for path in candidates if path.is_file()]
    if len(present) < 2:
        return _unavailable("V2", title, "fewer than two rasters exist yet")

    grids = {path.name: grid_from_raster(path) for path in present}
    reference = next(iter(grids.values()))
    problems = []
    for name, grid in grids.items():
        if grid.crs != reference.crs:
            problems.append(f"{name}: CRS {grid.crs} != {reference.crs}")
            continue
        for axis, value, other in (("min_x", grid.bounds[0], reference.bounds[0]),
                                   ("max_y", grid.bounds[3], reference.bounds[3])):
            offset = abs(value - other)
            # Products at different resolutions must still share the origin.
            if offset % min(grid.resolution_m, reference.resolution_m) > 1e-6:
                problems.append(f"{name}: {axis} offset {offset} m is not a whole cell")
    status = "pass" if not problems else "fail"
    return CheckResult(
        "V2", title, status,
        (f"all {len(grids)} rasters snap to the same origin ({reference.crs}, "
         f"snap {config.crs.grid_snap_m:.1f} m)") if status == "pass"
        else "; ".join(problems),
        {"grids": {name: grid.to_dict() for name, grid in grids.items()}},
    )


def check_v3_dsm_above_dem(config: Config, roi: RoiGeometry) -> CheckResult:
    """V3: the surface model is normally at or above the ground model."""
    title = "DSM - DEM is non-negative for almost all cells"
    dsm_path = config.paths.interim / "dsm.tif"
    dem_path = config.paths.interim / "dem.tif"
    if not dsm_path.is_file():
        return _unavailable(
            "V3", title,
            "no DSM is configured (lidar.dsm.provider='none'); see README "
            "'Data sources' for how to supply one",
        )
    if not dem_path.is_file():
        return _unavailable("V3", title, "no DEM available")

    dsm, _ = read_raster(dsm_path, band=1)
    dem, _ = read_raster(dem_path, band=1)
    difference = dsm - dem
    finite = np.isfinite(difference)
    negative = float((difference[finite] < -0.5).mean()) if finite.any() else 1.0
    status = "pass" if negative < 0.02 else ("warn" if negative < 0.05 else "fail")
    return CheckResult(
        "V3", title, status,
        f"{negative:.2%} of cells have DSM more than 0.5 m below DEM "
        f"(pass < 2%, warn < 5%)",
        {"negative_fraction": negative,
         "median_difference_m": float(np.nanmedian(difference[finite])) if finite.any() else None},
    )


def check_v4_buildings_are_tall(config: Config, roi: RoiGeometry) -> CheckResult:
    """V4: mapped buildings coincide with tall objects (needs an nDSM)."""
    title = "mapped buildings coincide with high nDSM"
    ndsm_path = config.paths.raster / "ndsm.tif"
    if not ndsm_path.is_file():
        return _unavailable(
            "V4", title,
            "needs an nDSM, which needs a DSM (lidar.dsm.provider='none'). "
            "V1 cross-checks building positions against the imagery instead.",
        )
    grid = grid_from_raster(ndsm_path)
    heights, _ = read_raster(ndsm_path, band=1)
    buildings = rasterize_layer(config, "building", grid)
    if not buildings.any():
        return _unavailable("V4", title, "no mapped buildings in this ROI")

    inside = heights[buildings & np.isfinite(heights)]
    outside = heights[~buildings & np.isfinite(heights)]
    if inside.size == 0 or outside.size == 0:
        return _unavailable("V4", title, "not enough finite nDSM cells")
    median_inside = float(np.median(inside))
    median_outside = float(np.median(outside))
    threshold = config.fusion.thresholds.building_min_height_m
    tall_fraction = float((inside > threshold).mean())
    status = "pass" if (median_inside > median_outside + 1.0 and tall_fraction > 0.5) else "fail"
    return CheckResult(
        "V4", title, status,
        f"median height inside building footprints {median_inside:.2f} m vs "
        f"{median_outside:.2f} m outside; {tall_fraction:.1%} of footprint cells "
        f"exceed {threshold} m",
        {"median_inside_m": median_inside, "median_outside_m": median_outside,
         "tall_fraction": tall_fraction, "threshold_m": threshold},
    )


def check_v5_cloud_matches_imagery(config: Config, roi: RoiGeometry) -> CheckResult:
    """V5: the RGB point cloud seen from above reproduces the orthophoto."""
    title = "RGB point cloud viewed from above matches the orthophoto"
    from rokko_geofusion.lidar.pointcloud import read_point_cloud

    cloud_path = config.paths.pointcloud / f"cloud.{config.output.pointcloud_format}"
    imagery_path = config.paths.interim / "orthophoto.tif"
    if not cloud_path.is_file():
        return _unavailable("V5", title, "no point cloud yet (scripts/colorize_pointcloud.py)")
    if not imagery_path.is_file():
        return _unavailable("V5", title, "no orthophoto yet")

    xyz, rgb = read_point_cloud(cloud_path)
    if rgb is None:
        return _unavailable("V5", title, "the point cloud has no colour")
    unsampled = int((rgb.sum(axis=1) == 0).sum())

    from rokko_geofusion.imagery.sampler import sample_rgb

    reference = sample_rgb(
        imagery_path, xyz[:, 0], xyz[:, 1],
        expected_crs=grid_from_raster(imagery_path).crs, method="nearest",
    )
    difference = np.abs(rgb.astype(float) - reference.astype(float))
    correlation = float(np.corrcoef(rgb[:, 0].astype(float),
                                    reference[:, 0].astype(float))[0, 1])
    status = "pass" if (unsampled == 0 and correlation > 0.95) else (
        "warn" if correlation > 0.8 else "fail")
    return CheckResult(
        "V5", title, status,
        f"{len(xyz):,} points, {unsampled} without colour; re-sampling the "
        f"orthophoto at the point coordinates reproduces the stored colour with "
        f"R={correlation:.4f}, mean |difference| {difference.mean():.2f}/255",
        {"points": int(len(xyz)), "unsampled_points": unsampled,
         "red_correlation": correlation, "mean_abs_difference": float(difference.mean())},
    )


def check_v6_reproducibility(config: Config, roi: RoiGeometry) -> CheckResult:
    """V6: re-running the same ROI produces identical results."""
    title = "recomputation is deterministic"
    dem_path = config.paths.interim / "dem.tif"
    if not dem_path.is_file():
        return _unavailable("V6", title, "no DEM available")

    from rokko_geofusion.terrain.analysis import aspect, slope

    grid = grid_from_raster(dem_path)
    dem, _ = read_raster(dem_path, band=1)

    def digest() -> str:
        payload = [
            slope(dem, grid.resolution_m, units=config.terrain.slope_units).tobytes(),
            aspect(dem, grid.resolution_m).tobytes(),
            str(roi.grid(config.lidar.resolution_m).to_dict()).encode(),
            config.fingerprint().encode(),
        ]
        hasher = hashlib.sha256()
        for item in payload:
            hasher.update(item)
        return hasher.hexdigest()

    first, second = digest(), digest()
    status = "pass" if first == second else "fail"
    return CheckResult(
        "V6", title, status,
        f"terrain derivatives + grid + config digest is stable: {first[:16]}",
        {"digest": first, "stable": first == second,
         "config_fingerprint": config.fingerprint()},
    )


def check_segmentation_labels(config: Config, roi: RoiGeometry) -> CheckResult:
    """Score the segmentation classes against independent OSM geometry.

    This is the measurement that justifies (or refutes) any entry in
    ``segmentation.label_overrides``.
    """
    title = "segmentation classes agree with independent OSM geometry"
    class_path = config.paths.raster / "segmentation_class.tif"
    if not class_path.is_file():
        return _unavailable("SEG", title, "no segmentation raster yet")

    grid = grid_from_raster(class_path)
    class_index, _ = read_raster(class_path, band=1)
    names = config.segmentation.classes
    measurements: dict[str, Any] = {}
    for project_class, layer, buffer_m in (("building", "building", 0.0),
                                           ("road", "road", 4.0),
                                           ("water", "water", 3.0)):
        if project_class not in names:
            continue
        try:
            reference = rasterize_layer(config, layer, grid, buffer_m=buffer_m)
        except FileNotFoundError:
            continue
        predicted = class_index == names.index(project_class)
        measurements[project_class] = mask_agreement(predicted, reference)

    building = measurements.get("building", {})
    recall = building.get("recall", 0.0)
    status = "pass" if recall > 0.6 else ("warn" if recall > 0.3 else "fail")
    detail = "; ".join(
        f"{name}: IoU {values['iou']:.3f} recall {values['recall']:.3f} "
        f"precision {values['precision']:.3f}"
        for name, values in measurements.items()
    ) or "no comparable layers"
    if config.segmentation.label_overrides:
        detail += " (with segmentation.label_overrides applied)"
    return CheckResult("SEG", title, status, detail, measurements)


CHECKS = (
    check_v1_imagery_gis_alignment,
    check_v2_grid_alignment,
    check_v3_dsm_above_dem,
    check_v4_buildings_are_tall,
    check_v5_cloud_matches_imagery,
    check_v6_reproducibility,
    check_segmentation_labels,
)


def run_all(config: Config, roi: RoiGeometry, *, only: str | None = None) -> list[CheckResult]:
    results: list[CheckResult] = []
    for check in CHECKS:
        result = check(config, roi)
        if only and only.lower() not in (result.id.lower(), check.__name__):
            continue
        results.append(result)
        logger.info("%s", result)
    return results


def summarise(results: list[CheckResult]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1
    return {
        "checks": [result.to_dict() for result in results],
        "counts": counts,
        "failed": [result.id for result in results if result.status == "fail"],
    }


def to_markdown(results: list[CheckResult]) -> str:
    lines = ["| check | status | detail |", "|---|---|---|"]
    for result in results:
        detail = result.detail.replace("\n", " ").replace("|", "/")
        lines.append(f"| **{result.id}** {result.title} | `{result.status}` | {detail} |")
    return "\n".join(lines)
