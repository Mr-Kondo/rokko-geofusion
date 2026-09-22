"""Assemble the structured analysis payload.

This is the single place where the numbers that describe an ROI are collected:
terrain statistics, land-cover fractions, point cloud size, fusion results,
point cloud ML metrics and the validation outcome. The LLM stage consumes this
payload verbatim; it never recomputes anything, and anything missing is
reported as ``null`` with a reason rather than filled in.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from rokko_geofusion import __version__
from rokko_geofusion.config import Config
from rokko_geofusion.crs import RoiGeometry
from rokko_geofusion.utils.metadata import read_json

logger = logging.getLogger(__name__)


def _load(path: Path) -> Any | None:
    if not path.is_file():
        logger.debug("no metrics file at %s", path)
        return None
    try:
        return read_json(path)
    except (ValueError, OSError) as exc:
        logger.warning("could not read %s: %s", path, exc)
        return None


def _round(value: Any, digits: int = 2) -> Any:
    if isinstance(value, float):
        return round(value, digits)
    if isinstance(value, dict):
        return {key: _round(item, digits) for key, item in value.items()}
    if isinstance(value, list):
        return [_round(item, digits) for item in value]
    return value


def build_payload(config: Config, roi: RoiGeometry) -> dict[str, Any]:
    """Collect every computed statistic for this ROI into one document."""
    metrics = config.paths.metrics
    terrain = _load(metrics / "terrain.json")
    segmentation = _load(metrics / "segmentation.json")
    fusion = _load(metrics / "fusion.json")
    pointcloud = _load(metrics / "pointcloud.json")
    pointcloud_ml = _load(metrics / "pointcloud_ml.json")
    validation = _load(metrics / "validation.json")
    lidar = _load(metrics / "download_lidar.json")

    missing: dict[str, str] = {}
    payload: dict[str, Any] = {
        "schema_version": 1,
        "producer": f"rokko-geofusion/{__version__}",
        "config_fingerprint": config.fingerprint(),
        "roi": roi.to_dict(),
        "provenance": {
            "dem": (lidar or {}).get("products", {}).get("dem"),
            "dsm": (lidar or {}).get("products", {}).get("dsm"),
            "imagery": {
                "provider": config.imagery.provider,
                "dataset": config.imagery.dataset,
                "zoom": config.imagery.zoom,
                "resolution_m": config.imagery.resolution_m,
                "attribution": config.imagery.attribution,
            },
            "gis": {
                "providers": list(config.gis.providers),
                "layers": list(config.gis.osm.layers),
                "attribution": config.gis.osm.attribution,
            },
            "crs": {
                "geographic": config.crs.geographic,
                "projected": config.crs.projected,
                "grid_snap_m": config.crs.grid_snap_m,
            },
        },
    }

    if terrain:
        payload["terrain"] = {
            "area_m2": _round(terrain.get("area_m2")),
            "resolution_m": terrain.get("resolution_m"),
            "elevation_m": _round(terrain.get("elevation")),
            "slope_deg": _round(terrain.get("slope")),
            "relief_m": _round(terrain.get("relief")),
            "aspect": _round(terrain.get("aspect"), 3),
            "object_height_m": _round(terrain.get("object_height")),
            "unavailable": terrain.get("unavailable", {}),
        }
    else:
        missing["terrain"] = "scripts/build_terrain.py has not been run for this ROI"

    if segmentation:
        payload["image_semantics"] = {
            "model": (segmentation.get("model") or {}).get("model_id"),
            "resolution_m": (segmentation.get("grid") or {}).get("resolution_m"),
            "class_fractions": _round(segmentation.get("class_fractions"), 4),
            "mean_confidence": _round(segmentation.get("mean_confidence"), 3),
            "caveat": (segmentation.get("model") or {}).get("domain_note"),
        }
    else:
        missing["image_semantics"] = "scripts/segment_imagery.py has not been run"

    if fusion:
        payload["fusion"] = {
            "cells": fusion.get("cells"),
            "cell_size_m": fusion.get("cell_size_m"),
            "class_fractions": _round(fusion.get("class_fractions"), 4),
            "rule_counts": fusion.get("rule_counts"),
            "skipped_conditions": fusion.get("skipped_conditions", []),
            "modalities": fusion.get("modalities", {}),
            "unavailable": fusion.get("unavailable", {}),
        }
    else:
        missing["fusion"] = "scripts/fuse_modalities.py has not been run"

    if pointcloud:
        payload["pointcloud"] = {
            "points": pointcloud.get("n_points"),
            "surface": pointcloud.get("surface"),
            "has_rgb": pointcloud.get("has_rgb"),
            "is_true_lidar": pointcloud.get("is_true_lidar"),
            "z_range_m": _round(pointcloud.get("z_range_m")),
        }
    else:
        missing["pointcloud"] = "scripts/colorize_pointcloud.py has not been run"

    if pointcloud_ml:
        payload["pointcloud_ml"] = {
            "dataset": pointcloud_ml.get("dataset"),
            "evaluation_note": pointcloud_ml.get("evaluation_note"),
            "results": [
                {
                    "feature_set": item.get("feature_set"),
                    "in_channels": item.get("in_channels"),
                    "final_loss": _round(item.get("final_loss"), 4),
                    "silhouette": _round(item.get("silhouette"), 3),
                    "adjusted_mutual_information": _round(
                        item.get("adjusted_mutual_information"), 3
                    ),
                    "caveat": item.get("caveat"),
                }
                for item in pointcloud_ml.get("results", [])
            ],
        }
    else:
        missing["pointcloud_ml"] = "scripts/run_pointcloud_ml.py has not been run"

    if validation:
        payload["validation"] = {
            "counts": validation.get("counts"),
            "checks": [
                {"id": check["id"], "status": check["status"], "detail": check["detail"]}
                for check in validation.get("checks", [])
            ],
        }
    else:
        missing["validation"] = "scripts/validate.py has not been run"

    payload["missing"] = missing
    if missing:
        logger.warning("analysis payload is missing %d section(s): %s",
                       len(missing), ", ".join(sorted(missing)))
    return payload


def write_payload(config: Config, roi: RoiGeometry) -> tuple[Path, dict[str, Any]]:
    from rokko_geofusion.utils.metadata import write_json

    payload = build_payload(config, roi)
    path = write_json(config.paths.reports / "analysis_payload.json", payload)
    return path, payload
