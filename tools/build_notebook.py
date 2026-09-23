#!/usr/bin/env python
"""Generate notebooks/geofusion_demo.ipynb.

The notebook is a build artefact: it contains no processing logic, only calls
into `scripts/` and `rokko_geofusion`. Generating it from here keeps that rule
enforceable and keeps the file's JSON diff-friendly.

    python tools/build_notebook.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = "https://github.com/Mr-Kondo/rokko-geofusion"
NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "geofusion_demo.ipynb"


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.strip().splitlines(True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.strip().splitlines(True),
    }


CELLS = [
    md(f"""
# rokko-geofusion — LiDAR × imagery × GIS × GeoAI

Analysis of one geographic area by combining an elevation model, an
orthophoto, GIS vectors, image segmentation, point-cloud learning and a
VLM/LLM pair — all on one explicit coordinate reference system.

**This notebook contains no processing logic.** Every cell either calls a
script in `scripts/` or imports from `rokko_geofusion`. Change
`configs/rokko.yaml` to analyse a different place.

Default area: Kobe University Rokkodai campus (34.7284 N, 135.2348 E, ±1 km).

Repository: {REPO}
"""),

    md("## 01 · Install"),
    code("""
# Colab: clone and install. Local: this is a no-op if you are already in the repo.
import os, sys, subprocess
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules
REPO_URL = "https://github.com/Mr-Kondo/rokko-geofusion.git"   # <- your fork, if any
REPO_DIR = Path("/content/rokko-geofusion") if IN_COLAB else Path.cwd()

if IN_COLAB and not REPO_DIR.exists():
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(REPO_DIR)], check=True)
if IN_COLAB:
    os.chdir(REPO_DIR)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", ".[all]"], check=True)

print("working directory:", Path.cwd())
"""),

    md("## 02 · Runtime information\n\nDetects CPU, RAM, GPU, VRAM, CUDA, GDAL and the "
       "installed packages, and derives the tile/batch/point budgets from them. "
       "Nothing below assumes a particular GPU."),
    code("""
!python scripts/check_environment.py --config configs/rokko.yaml
"""),

    md("## 03 · Load the configuration\n\nEvery URL, EPSG code, threshold and model id "
       "lives in the YAML file — none of them are hard-coded in the package."),
    code("""
from rokko_geofusion.config import load_config
from rokko_geofusion.crs import roi_from_config
from rokko_geofusion.environment import make_resource_profile

CONFIG_PATH = "configs/rokko.yaml"
config = load_config(CONFIG_PATH)
roi = roi_from_config(config)
profile = make_resource_profile(config)

print(roi)
print("CRS:", config.crs.geographic, "->", config.crs.projected, "(tiles:", config.crs.tile + ")")
print("outputs:", config.paths.out_root)
print("config fingerprint:", config.fingerprint())
print("resource profile:", profile.to_dict())
"""),

    md("## 04 · Download the data\n\nDEM (GSI elevation tiles), orthophoto (GSI seamless "
       "photo) and GIS vectors (OpenStreetMap via Overpass). Responses are cached on "
       "disk, so re-running is cheap.\n\n"
       "**DSM:** no verifiable public DSM URL exists for the default area, so the DSM "
       "provider is `none` and every height-above-ground product reports itself as "
       "*unavailable* rather than being faked. See the README for how to supply one."),
    code("""
!python scripts/run_pipeline.py --config configs/rokko.yaml --stage lidar imagery gis
"""),

    md("## 05 · Preprocess\n\nBuilds the RGB point cloud (elevation + colour sampled "
       "from the orthophoto) and the terrain derivatives."),
    code("""
!python scripts/run_pipeline.py --config configs/rokko.yaml --stage pointcloud terrain
"""),

    md("## 06 · 2D — GIS layers on a map"),
    code("""
from rokko_geofusion.gis.osm import load_layer
from rokko_geofusion.visualization.maps import roi_map

layers = {name: load_layer(config, name) for name in config.gis.osm.layers}
for name, frame in layers.items():
    print(f"{name:<10} {len(frame):5d} features")

roi_map(config, roi, layers=layers)
"""),

    md("## 07 · 3D — the RGB point cloud\n\nThe analysis cloud stays on disk; only a "
       "downsampled copy is sent to the browser (`visualization.max_display_points`)."),
    code("""
from rokko_geofusion.visualization.pointcloud3d import plot_point_cloud

cloud_path = config.paths.pointcloud / f"cloud.{config.output.pointcloud_format}"
plot_point_cloud(config, cloud_path, color_by="rgb")
"""),

    md("## 08 · Terrain analysis\n\nSlope and aspect use Horn's 3×3 operator and are "
       "verified against analytic planes in `tests/test_terrain.py`."),
    code("""
from rokko_geofusion.utils.metadata import read_json
from rokko_geofusion.visualization.plots import grid_figure

raster = config.paths.raster
figure = grid_figure([
    {"path": config.paths.interim / "orthophoto.tif", "kind": "rgb", "title": "orthophoto"},
    {"path": raster / "elevation.tif", "title": "elevation (m)", "cmap": "terrain",
     "colorbar_label": "m"},
    {"path": raster / "hillshade.tif", "title": "hillshade", "cmap": "gray"},
    {"path": raster / "slope.tif", "title": "slope (deg)", "cmap": "magma",
     "colorbar_label": "deg"},
    {"path": raster / "aspect.tif", "title": "aspect (deg, 0=N)", "cmap": "hsv",
     "percentile_clip": (0, 100)},
    {"path": raster / "relief.tif", "title": "local relief (m)", "cmap": "cividis",
     "colorbar_label": "m"},
], ncols=3, suptitle=f"Terrain — {roi.key}")

terrain = read_json(config.paths.metrics / "terrain.json")
print("area              %.3f km2" % (terrain["area_m2"] / 1e6))
print("elevation  min/mean/max  %.1f / %.1f / %.1f m" % (
    terrain["elevation"]["min"], terrain["elevation"]["mean"], terrain["elevation"]["max"]))
print("slope      mean/max      %.1f / %.1f deg" % (
    terrain["slope"]["mean"], terrain["slope"]["max"]))
print("aspect sectors:", {k: round(v, 3) for k, v in terrain["aspect"]["sectors"].items()})
for name, reason in terrain.get("unavailable", {}).items():
    print(f"UNAVAILABLE {name}: {reason}")
figure
"""),

    md("## 09 · Image semantic segmentation\n\nA pretrained checkpoint mapped onto the "
       "project classes **by label name**. No accuracy is claimed for this area — "
       "cell 15 scores it against independent OSM geometry."),
    code("""
!python scripts/run_pipeline.py --config configs/rokko.yaml --stage segmentation
"""),
    code("""
from IPython.display import Image, display

segmentation = read_json(config.paths.metrics / "segmentation.json")
print("model:", segmentation["model"]["model_id"])
for name, fraction in segmentation["class_fractions"].items():
    print(f"  {name:<12} {fraction:6.1%}")
display(Image(filename=str(config.paths.figures / "segmentation.png")))
"""),

    md("## 10 · Multimodal fusion\n\nOne row per cell carrying geometry, colour, "
       "terrain, image semantics and GIS attributes, plus a rule-based fused class "
       "that records *which rule* assigned it."),
    code("""
!python scripts/run_pipeline.py --config configs/rokko.yaml --stage fusion
"""),
    code("""
import pandas as pd

fusion = read_json(config.paths.metrics / "fusion.json")
print(f"{fusion['cells']:,} cells at {fusion['cell_size_m']} m")
for name, fraction in fusion["class_fractions"].items():
    if fraction:
        print(f"  {name:<18} {fraction:6.1%}")
for condition in fusion.get("skipped_conditions", []):
    print("RULE NOT APPLIED:", condition)

table = pd.read_parquet(config.paths.vector / "fusion_cells.parquet")
display(table.head())
display(Image(filename=str(config.paths.figures / "fusion.png")))
"""),

    md("## 11 · Point cloud machine learning\n\nROI → tile → voxelise → sample → batch → "
       "PointNet, trained self-supervised. The four feature sets are compared; no "
       "supervised accuracy is reported because no ground truth exists here."),
    code("""
!python scripts/run_pipeline.py --config configs/rokko.yaml --stage pointcloud_ml
"""),
    code("""
ml = read_json(config.paths.metrics / "pointcloud_ml.json")
print(ml["evaluation_note"], "\\n")
display(pd.DataFrame([{
    "feature set": r["feature_set"],
    "channels": r["in_channels"],
    "final loss": round(r["final_loss"], 4),
    "silhouette": r["silhouette"],
    "AMI vs fused class": r["adjusted_mutual_information"],
    "seconds": round(r["train_seconds"], 1),
} for r in ml["results"]]))
for r in ml["results"]:
    if r.get("caveat"):
        print(f"CAVEAT ({r['feature_set']}): {r['caveat']}")
display(Image(filename=str(config.paths.figures / "pointcloud_ml.png")))
"""),

    md("## 12 · VLM — visual interpretation\n\nThe vision model sees rendered views "
       "only, and is forbidden from producing numbers. Set an API key first; without "
       "one the stage records `status: unavailable` and the notebook continues."),
    code("""
import os
# os.environ["ANTHROPIC_API_KEY"] = "sk-ant-..."   # or use Colab secrets
print("API key set:", bool(os.environ.get(config.vlm.api_key_env)))

!python scripts/run_pipeline.py --config configs/rokko.yaml --stage vlm
"""),
    code("""
vlm = read_json(config.paths.reports / "vlm_analysis.json")
if vlm["status"] == "ok":
    print("terrain:   ", vlm["terrain_description"])
    print("land cover:", vlm["land_cover_description"])
    print("built:     ", vlm["built_environment_description"])
    for item in vlm["notable_patterns"]:
        print(" pattern:", item)
    for item in vlm["uncertainties"]:
        print(" uncertainty:", item)
else:
    print("VLM unavailable:", vlm["reason"])

for path in sorted((config.paths.figures / "vlm_inputs").glob("*.png")):
    display(Image(filename=str(path), width=430))
"""),

    md("## 13 · LLM — integrated analysis\n\nMeasurements come from Python; the "
       "language model integrates and explains them, keeping Measured / Observed / "
       "Inferred / Uncertain apart."),
    code("""
!python scripts/run_pipeline.py --config configs/rokko.yaml --stage llm
"""),
    code("""
from IPython.display import Markdown

report_path = config.paths.reports / "geoai_report.md"
report = read_json(config.paths.reports / "geoai_report.json")
if report.get("status") == "ok":
    display(Markdown(report_path.read_text(encoding="utf-8")))
else:
    print("LLM unavailable:", report.get("reason"))
    print("The measurements are still available:",
          config.paths.reports / "analysis_payload.json")
"""),

    md("## 14 · Interactive ROI analysis\n\nDraw a rectangle on the map, then read its "
       "statistics straight from the processed rasters (no downloads). To analyse a "
       "*new* area end to end, call `analyze_roi(bounds=[...])`."),
    code("""
from rokko_geofusion.interactive import analyze_roi, roi_selector, summarize_subregion

leaflet_map, get_bounds = roi_selector(config)
leaflet_map
"""),
    code("""
# Statistics for the drawn box (or the fallback box) - instant, no downloads.
bounds = get_bounds() or [135.2320, 34.7265, 135.2380, 34.7305]
print("bounds:", [round(v, 5) for v in bounds])

statistics = summarize_subregion(config, bounds)
print("area           %.0f m2 (%d cells)" % (statistics["area_m2"], statistics["cells"]))
print("elevation  min/mean/max  %.1f / %.1f / %.1f m" % (
    statistics["elevation"]["min"], statistics["elevation"]["mean"],
    statistics["elevation"]["max"]))
print("slope      mean/p95      %.1f / %.1f deg" % (
    statistics["slope"]["mean"], statistics["slope"]["p95"]))
for name, fraction in statistics.get("land_cover", {}).items():
    if fraction > 0.01:
        print(f"  {name:<18} {fraction:6.1%}")
for name, reason in statistics.get("unavailable", {}).items():
    print(f"UNAVAILABLE {name}: {reason}")
"""),
    code("""
# Analyse a DIFFERENT area end to end (downloads + full pipeline).
# analysis = analyze_roi(center=(34.7100, 135.2100), radius_m=500, name="kobe_port")
# print(analysis.summary())
"""),

    md("## 15 · Validation and export\n\nSpatial checks V1–V6: co-registration, grid "
       "alignment, DSM ≥ DEM, buildings vs height, cloud vs imagery, determinism. "
       "Checks that cannot run report `unavailable`, never `pass`."),
    code("""
!python scripts/run_pipeline.py --config configs/rokko.yaml --stage validate report
"""),
    code("""
validation = read_json(config.paths.metrics / "validation.json")
for check in validation["checks"]:
    print(f"[{check['status'].upper():<11}] {check['id']}: {check['detail']}")
print("\\ncounts:", validation["counts"])
display(Markdown((config.paths.metrics / "validation.md").read_text(encoding="utf-8")))
"""),
    code("""
# Bundle the small artefacts (figures, metrics, reports) for download.
import shutil

bundle = shutil.make_archive(
    str(config.paths.out_root / f"{roi.key}_results"), "zip",
    root_dir=config.paths.out_root,
)
print("bundle:", bundle, f"({os.path.getsize(bundle) / 1e6:.1f} MB)")
print("point cloud:", config.paths.pointcloud)
print("rasters:    ", config.paths.raster)
print("fusion table:", config.paths.vector / "fusion_cells.parquet")

if IN_COLAB:
    from google.colab import files
    # files.download(bundle)
"""),
]


def build() -> dict:
    return {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
            "colab": {"provenance": [], "toc_visible": True},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> int:
    NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK.write_text(json.dumps(build(), indent=1, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    n_code = sum(1 for cell in CELLS if cell["cell_type"] == "code")
    print(f"wrote {NOTEBOOK} ({len(CELLS)} cells, {n_code} code)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
