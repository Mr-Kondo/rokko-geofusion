# rokko-geofusion

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mr-Kondo/rokko-geofusion/blob/main/notebooks/geofusion_demo.ipynb)

LiDAR-derived terrain × aerial imagery × GIS × GeoAI for one geographic area,
on one explicit coordinate reference system.

The default area is the **Kobe University Rokkodai campus** and its
surroundings (34.7284 N, 135.2348 E, ±1 km). Analysing somewhere else is a
change to `configs/rokko.yaml` and nothing else.

```
              Google Colab  (run · compare · visualise)
                            │
                  Geographic ROI selection
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
     DEM / DSM          Orthophoto          GIS vectors
   (GSI elevation      (GSI seamless         (OSM via
      tiles)              photo)             Overpass)
        │                   │                   │
        ▼                   ▼                   ▼
   point cloud        semantic seg.       spatial masks
   terrain derivs     (SegFormer)         + attributes
        │                   │                   │
        └───────────────────┼───────────────────┘
                            ▼
                  multimodal fusion  (one row per cell)
                            │
                ┌───────────┼───────────┐
                ▼           ▼           ▼
          point cloud ML   VLM         LLM
          (PointNet,    (visual      (integrated
           self-sup.)   reading)      reasoning)
                            ▼
                      GeoAI output
```

## What it actually does

| Stage | Output | Status on the default ROI |
|---|---|---|
| DEM acquisition | 5 m elevation raster, EPSG:6673 | 100 % coverage, 26.7–512.0 m |
| Orthophoto | 0.5 m RGB raster | 100 % coverage, 4020 × 4020 px |
| GIS | building / road / water / landuse | 1538 / 970 / 50 / 98 features |
| Point cloud | RGB cloud, LAZ | 161,604 points |
| Terrain | slope, aspect, relief, hillshade, nDSM | slope mean 14.2°, max 65.7° |
| Segmentation | class + confidence raster | LoveDA SegFormer, 6 classes |
| Fusion | per-cell feature table + fused class | 4,040,100 cells, 19 columns |
| Point cloud ML | tile embeddings, clusters, metrics | 1680 tiles, 4 feature sets |
| VLM / LLM | structured visual reading + report | local Qwen3-VL on the GPU, no API key |
| Validation | V1–V6 spatial checks | 5 pass, 2 unavailable (need a DSM) |

## Install

```bash
git clone https://github.com/Mr-Kondo/rokko-geofusion.git && cd rokko-geofusion
python -m venv .venv && source .venv/bin/activate     # Python 3.10+
pip install -e ".[all]"
python scripts/check_environment.py --config configs/rokko.yaml
```

Extras: `pointcloud` (LAZ I/O), `ml` (torch + transformers + scikit-learn),
`ai` (Anthropic SDK), `viz` (matplotlib/plotly/folium/ipyleaflet), `dev`
(pytest + ruff), `all` (everything).

Python 3.12 is the reference version (it matches Colab). Very new interpreters
may not have wheels for rasterio/geopandas yet.

## Use it

### Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mr-Kondo/rokko-geofusion/blob/main/notebooks/geofusion_demo.ipynb)

Open [`notebooks/geofusion_demo.ipynb` in Colab](https://colab.research.google.com/github/Mr-Kondo/rokko-geofusion/blob/main/notebooks/geofusion_demo.ipynb) and run it top to bottom (cell 01 clones
this repository and installs the package; change `REPO_URL` there if you work
from a fork). The notebook installs the package, runs the pipeline stage
by stage and displays the results. It contains **no processing logic** — that
rule is enforced by `tests/test_pipeline.py`.

### Command line

```bash
python scripts/run_pipeline.py --config configs/rokko.yaml --stage all
python scripts/run_pipeline.py --config configs/rokko.yaml --stage fusion
python scripts/run_pipeline.py --list-stages
```

Every stage also runs standalone, which is the fast path when debugging one
step:

```bash
python scripts/download_lidar.py     --config configs/rokko.yaml
python scripts/download_imagery.py   --config configs/rokko.yaml
python scripts/download_gis.py       --config configs/rokko.yaml
python scripts/colorize_pointcloud.py --config configs/rokko.yaml
python scripts/build_terrain.py      --config configs/rokko.yaml
python scripts/segment_imagery.py    --config configs/rokko.yaml
python scripts/fuse_modalities.py    --config configs/rokko.yaml
python scripts/run_pointcloud_ml.py  --config configs/rokko.yaml
python scripts/run_vlm.py            --config configs/rokko.yaml
python scripts/run_llm.py            --config configs/rokko.yaml
python scripts/validate.py           --config configs/rokko.yaml
```

Any configuration value can be overridden without editing the file:

```bash
python scripts/run_pipeline.py --config configs/rokko.yaml --stage all \
  --set roi.radius_m=500 --set imagery.zoom=17 --set segmentation.enabled=false
```

### Python API

```python
from rokko_geofusion.interactive import analyze_roi, summarize_subregion

# A new area, end to end (downloads + full pipeline).
analysis = analyze_roi(center=(34.7100, 135.2100), radius_m=500, name="kobe_port")
print(analysis.summary())

# A sub-box of the area already processed — instant, no downloads.
stats = summarize_subregion(analysis.config, [135.2320, 34.7265, 135.2380, 34.7305])
```

## Data sources

All endpoints below were verified against the live services during
development; `tests/test_network.py` re-checks them so an upstream change
fails loudly instead of looking like "no data for this ROI".

| Source | Endpoint | Notes |
|---|---|---|
| **DEM** | `cyberjapandata.gsi.go.jp/xyz/dem5a/{z}/{x}/{y}.txt` | GSI 5 m mesh, zoom 15, 256×256 CSV of orthometric heights |
| **DEM fallback** | `.../xyz/dem/{z}/{x}/{y}.txt` | DEM10B, 10 m mesh, zoom 14, fills gaps in the 5 m product |
| **Orthophoto** | `.../xyz/seamlessphoto/{z}/{x}/{y}.jpg` | nationwide seamless photo mosaic; `ort` is the alternative |
| **GIS** | `overpass-api.de/api/interpreter` | needs a descriptive `User-Agent` (406 otherwise); mirrors configured |
| **Catalogue** | `geospatial.jp/ckan/api/3/action/...` | used to *discover* datasets, never to guess a URL |

Attribution: elevation and imagery © 国土地理院 (GSI); vectors ©
OpenStreetMap contributors, ODbL.

### About the DSM (important)

The brief names the Hyogo prefecture high-accuracy 3D dataset as the preferred
LiDAR source. **No stable public download URL for it could be verified** — a
CKAN search of `www.geospatial.jp` returns no matching dataset. Rather than
invent a plausible URL, `lidar.dsm.provider` ships as `none`, and everything
that depends on a DSM reports itself as *unavailable* with the reason:

* `ndsm.tif` and `object_height` statistics are not produced;
* the height-dependent fusion rules record themselves as **skipped**;
* validation checks V3 and V4 report `unavailable`, never `pass`.

To enable them, pick one:

```yaml
lidar:
  dsm:
    provider: local_raster           # a GeoTIFF you downloaded yourself
    local: {path: data/raw/dsm.tif}
```
```yaml
lidar:
  dsm:
    provider: local_xyz              # a regular-grid XYZ text file
    local: {path: data/raw/dsm.xyz, crs: EPSG:6673}
```
```yaml
lidar:
  dsm:
    provider: ckan                   # a catalogue resource you verified
    ckan: {resource_url: "https://.../dsm.tif"}
```

`python scripts/download_lidar.py --discover "<query>"` searches the catalogue
and prints the candidate resources with their URLs.

Because the shipped elevation source is a regular gridded product resampled
from web tiles, **it is not raw LiDAR**. Every artefact records
`is_true_lidar: false` and says so in its metadata sidecar.

## Coordinate reference systems

Three roles are kept explicit at all times and all three live in the config:

| Role | Default | Used for |
|---|---|---|
| `geographic` | EPSG:4326 | how you specify the ROI, and web maps |
| `tile` | EPSG:3857 | the XYZ pyramids we download |
| `projected` | EPSG:6673 | **all** analysis (JGD2011 / Japan Plane Rectangular CS V) |

Two rules make the products interoperable:

1. every raster is reprojected **once**, explicitly, onto the projected CRS;
2. every grid snaps to a shared lattice (`crs.grid_snap_m`, default 10 m), so
   a 0.5 m orthophoto and a 5 m DEM have identical outer bounds and nest
   exactly.

Rule 2 exists because of a real bug: with per-resolution snapping the coarse
DEM stuck out past the finer orthophoto and 801 edge points sampled no colour
at all. `tests/test_crs.py` guards against its return.

No EPSG code appears anywhere in `src/` except `config.py`; a test enforces it.

## Layout

```
configs/rokko.yaml        every URL, EPSG code, threshold and model id
scripts/*.py              one thin CLI per stage + run_pipeline.py
src/rokko_geofusion/      all processing logic
  config.py  crs.py  environment.py  paths.py  pipeline.py
  report.py  interactive.py  validation.py
  io/  lidar/  imagery/  gis/  terrain/  segmentation/
  fusion/  pointcloud_ml/  vlm/  llm/  visualization/  utils/
notebooks/                the Colab demo (generated by tools/build_notebook.py)
data/raw|interim|processed    inputs and intermediates (gitignored)
outputs/<roi>/            figures, metrics, reports, rasters, cloud (gitignored)
tests/                    unit + spatial-consistency tests
```

Outputs are ROI-scoped: `outputs/rokkodai_34.72840_135.23480_r1000/…`, so two
areas never overwrite each other and re-running one reproduces its paths.

Every artefact has a `<name>.meta.json` sidecar recording CRS, source,
resolution, acquisition details, processing parameters and the configuration
fingerprint.

## Milestones

| # | Milestone | Delivered |
|---|---|---|
| 0 | Repository / runtime | typed config, resource detection, logging, metadata, CLI |
| 1 | Data acquisition | GSI elevation + imagery, OSM, CKAN discovery, HTTP cache |
| 2 | CRS / preprocessing | CrsManager, RoiGeometry, shared grid lattice |
| 3 | 2D + 3D MVP | RGB point cloud, folium map, plotly cloud |
| 4 | Terrain analysis | slope, aspect, relief, hillshade, nDSM, statistics |
| 5 | Image segmentation | tiled SegFormer with a blended overlap and an OOM ladder |
| 6 | Multimodal fusion | 19-column per-cell table, auditable rule engine |
| 7 | Point cloud ML | PointNet, self-supervised, four feature sets compared |
| 8 | VLM | rendered views, numbers forbidden, structured output |
| 9 | LLM | Measured / Observed / Inferred / Uncertain report |
| 10 | Interactive GeoAI | `analyze_roi`, `summarize_subregion`, map selection |
| 11 | End-to-end validation | V1–V6 + segmentation cross-check |

## Validation

`python scripts/validate.py --config configs/rokko.yaml` writes
`outputs/<roi>/metrics/validation.{json,md}`.

| Check | Question | Result on the default ROI |
|---|---|---|
| **V1** | do the imagery and the GIS vectors describe the same ground? | **pass** — best-overlap offset between predicted and mapped buildings is **0.0 m** |
| **V2** | are all rasters on one lattice? | **pass** — 4 rasters, one origin, EPSG:6673 |
| **V3** | is DSM ≥ DEM? | *unavailable* — no DSM configured |
| **V4** | do mapped buildings sit on high nDSM? | *unavailable* — needs a DSM |
| **V5** | does the RGB cloud from above match the orthophoto? | **pass** — 161,604 points, **0 uncoloured**, R = 1.0000 |
| **V6** | does re-running reproduce the result? | **pass** — stable digest |
| **SEG** | do the predicted classes agree with independent OSM geometry? | **pass** — building recall 0.78, road precision 0.78 |

## Reproducibility

`V6` re-derives the terrain products inside one run and compares digests. Two
stages were additionally measured across *separate* runs on the same machine:

| stage | repeat-run result |
|---|---|
| segmentation (SegFormer on MPS) | **0 of 16,160,400 cells differ** |
| fusion (4,040,100 cells) | rule counts and class fractions identical |

GeoTIFF *file* hashes do change between runs: the metadata sidecar is also
written into the TIFF tags and carries `created_utc`. Compare pixel content or
the metrics JSON, not file bytes.

One caveat worth knowing: fixing the segmentation blend-window floor (tile
corners were weighted 0.0025 instead of 0.05) shifted roughly 0.005 % of fused
cells at class boundaries. Small changes to blending weights move a few
argmax decisions; that is expected, and it is why the rule counts are recorded
in `metrics/fusion.json` rather than being left implicit.

## Segmentation and its limits

The default checkpoint is `IgorNer/segformer-b5-loveda`, fine-tuned on LoveDA
(0.3 m aerial imagery), which is in-domain for an orthophoto. Labels are mapped
onto the project classes **by name**, so any HuggingFace semantic-segmentation
checkpoint can be substituted — including a general-purpose ADE20K one, which
is much faster and markedly worse from nadir.

**A measured correction is configured.** That checkpoint publishes an
`id2label` that does not match the class indices its head learned. Scored
against independent OSM geometry over this ROI:

| interpretation | building IoU | building recall | road precision |
|---|---|---|---|
| as published | 0.038 | 0.18 | 0.12 |
| corrected (`label_overrides`) | **0.259** | **0.78** | **0.78** |

and the published class 0 is never predicted at all, which is what an ignore
class does. The correction lives in `segmentation.label_overrides` with the
evidence in the config comment; reproduce it with
`python scripts/validate.py --check SEG`.

No accuracy figure is claimed for this area. Residual known issues: the model
over-calls impervious surfaces, its water class has low precision (the fusion
prefers mapped water for that reason), and LoveDA is Chinese imagery, so
Japanese urban texture is still a domain shift.

## Fusion

One row per 1 m cell: `x, y, z, r, g, b, elevation, object_height, slope,
aspect, relief, image_class, image_confidence, in_building, on_road, in_water,
building_height_osm, fused_class, rule`.

`rule` records **which rule** assigned the class, so any cell can be traced to
its evidence. On the default ROI the GIS layers contribute 484,236 road cells
and 87,797 building cells that the image model alone missed — that is the
fusion earning its place.

Height-dependent rules (`building height ≥ 2.5 m`, `tall vegetation ≥ 3 m`,
`road ≤ 1 m`) cannot be evaluated without a DSM. They are recorded as
**skipped** in `metrics/fusion.json` and logged as `RULE NOT APPLIED`, never
treated as satisfied.

## Point cloud ML

`ROI → spatial tile → voxelisation → sampling → mini batch → PointNet`, as
specified. The fusion table is the source, so the four feature sets are the
same points with more channels:

| set | channels |
|---|---|
| A `xyz` | geometry only |
| B `xyz_rgb` | + colour |
| C `xyz_rgb_terrain` | + slope, aspect (sin/cos), relief, height (+ a *known* flag) |
| D `xyz_rgb_terrain_semantic` | + image class (confidence-weighted) and GIS flags |

There is **no labelled ground truth for this area**, so no supervised accuracy
is reported. The encoder is trained self-supervised (NT-Xent over augmented
views) and evaluated by silhouette score and by adjusted mutual information
against the independently derived fused classes.

Result on the default ROI (1680 tiles of 50 m, 4096 points each, 20 epochs):

| feature set | channels | final loss | silhouette | AMI vs fused class |
|---|---|---|---|---|
| A `xyz` | 3 | 0.116 | **0.415** | 0.184 |
| B `xyz_rgb` | 6 | 0.156 | 0.354 | 0.194 |
| C `xyz_rgb_terrain` | 12 | 0.135 | 0.222 | 0.196 |
| D `xyz_rgb_terrain_semantic` | 20 | 0.127 | 0.215 | **0.229** |

Agreement with the fused classes rises monotonically as modalities are added,
while silhouette falls: richer inputs buy semantic alignment at the cost of
globular embedding structure.

**A confound had to be fixed to get that result.** Channels such as slope or a
class one-hot are invariant to rotation and jitter, so with only geometric
augmentation the two views of a tile stayed trivially identifiable: the
contrastive loss collapsed to 0.003 and the richer feature sets scored *worse*
on every metric — an artefact of the objective, not a property of the features.
`pointcloud_ml.augment_feature_dropout` blanks whole non-geometry channels per
view, which keeps the task non-trivial (loss 0.12–0.16 across all four sets).

Set D contains the image semantics the fused classes were partly derived from,
so its AMI is inflated by construction. That caveat is attached to its result
row and printed by the CLI; it is not evidence that the model learned more.

## VLM and LLM

By default both stages run an **open-weights model on this machine's GPU**
(`provider: local`): Qwen3-VL, Apache-2.0. No API key is needed and no data
leaves the runtime. One checkpoint serves both stages: the VLM gives it the
rendered views, the LLM gives it text only.

`model: auto` picks the size from the accelerator memory actually detected,
using `local_models.auto_tiers` in the config:

| detected memory | e.g. | model | weights |
|---|---|---|---|
| >= 30 GB | A100 40/80 GB | `Qwen/Qwen3-VL-8B-Instruct` | 17.5 GB |
| >= 12 GB | T4 16 GB, L4 24 GB | `Qwen/Qwen3-VL-4B-Instruct` | 8.9 GB |
| smaller, or CPU | | `Qwen/Qwen3-VL-2B-Instruct` | 4.3 GB |

Memory means GPU VRAM on CUDA and half the unified memory on Apple silicon.
The model runs in bf16 where the GPU supports it and in fp16 on a T4. Any
Hugging Face id can replace `auto`; `llm.model` may also be a text-only model.

* Decoding is **greedy** (temperature 0) rather than the sampling the
  checkpoints ship with, so re-running an ROI reproduces the same text on the
  same device and dtype: two bf16 runs on Apple silicon were byte-identical.
  A different dtype changes the text (fp16 and bf16 runs differed), and some
  CUDA kernels are non-deterministic, which was not measured here.
* The VLM receives **rendered views only** (orthophoto, RGB cloud, elevation,
  slope, nDSM, segmentation) with captions carrying units and CRS. Its system
  prompt forbids numbers outright and its response schema has no numeric
  fields.
* The LLM receives the Python-computed payload plus the VLM's qualitative
  reading, and must keep **Measured / Observed / Inferred / Uncertain** apart.
* Every result records which model ran, on which device and dtype, and why that
  model was chosen (`runtime` in `reports/*.json`).
* A local model that fails (for example out of memory) writes
  `status: "failed"` with advice, overwriting any earlier result, so a stale
  analysis is never shown as this run's. Hosted APIs remain available:
  `provider: anthropic` (or `openai`), the model name, and the API key.

```bash
python scripts/run_vlm.py --config configs/rokko.yaml
python scripts/run_llm.py --config configs/rokko.yaml
# Inspect exactly what would be sent, without running a model:
python scripts/run_vlm.py --config configs/rokko.yaml --render-only
python scripts/run_llm.py --config configs/rokko.yaml --prompt-only
# Check that a model fits and answers on this machine (loads it for real):
RGF_TEST_LOCAL_MODEL=Qwen/Qwen3-VL-8B-Instruct python -m pytest -q -m local_model
```

If a model download stalls at 0 B/s, set `HF_HUB_DISABLE_XET=1` and run again:
it switches Hugging Face from the Xet transfer to plain HTTP, and the download
resumes from the partial files.

## Resources

Nothing assumes a particular GPU. `environment.make_resource_profile()`
detects CPU, RAM, GPU and VRAM and derives tile size, batch size and point
budgets; on out-of-memory the code walks a documented ladder — **batch size →
tile size → point count → CPU**.

| Configuration | Behaviour on the default 2 × 2 km ROI |
|---|---|
| CPU only | works; segmentation is the slow step (tiles 384 px, batch 1) |
| Colab T4 / L4 (14–24 GB) | full pipeline comfortably |
| Apple M-series (MPS) | used for this development run; segmentation ≈ 13 s |
| RAM | ~4 GB peak; the fusion table is streamed tile by tile |
| Disk | ~215 MB per ROI (112 MB of it the fusion Parquet) |
| Local VLM/LLM | first run downloads the model: 17.5 GB (8B, A100) or 8.9 GB (4B, T4) |

Downloads for the default ROI: 13 elevation tiles, 306 imagery tiles, 4
Overpass queries — all cached under `data/raw/_cache`.

## Limitations and known issues

1. **No DSM by default** → no nDSM, no measured object heights, V3/V4
   unavailable, three fusion rules skipped. This is the single biggest gap and
   it is a data-availability problem, not a code one.
2. **The elevation source is a regular grid, not raw LiDAR returns.** It cannot
   show vegetation structure, wires or building façades.
3. **Segmentation is out-of-region.** LoveDA transfers well but is not Japanese
   imagery; no accuracy is claimed.
4. **OSM completeness varies.** Building footprints are the roof outline and a
   minority carry a height tag; `height_source` records where each came from.
5. **Relation geometry is best-effort**: outer members are polygonised, inner
   rings are not subtracted, so courtyard buildings are slightly over-sized.
6. **Point cloud ML has no ground truth**, so its metrics are structural, not
   accuracy.
7. **Overpass is rate-limited** and its primary endpoint refused connections
   during development; mirrors are configured and responses are cached.
8. **The local VLM/LLM was run end to end on Apple silicon only** (M5, MPS,
   Qwen3-VL-4B in bf16): the VLM stage took 90 s for six views, the LLM stage
   5.5 min, both answers parsed, the VLM output contained no numbers, and every
   number in the report exists in the Python payload. The CUDA paths (8B in
   bf16 on an A100, 4B in fp16 on a T4) use the same code but were not run here.
   fp16 numerics were checked by converting the 4B model to fp16 on the device:
   no NaN or garbled output, and the answer parsed. (Loading *directly* as fp16
   on Apple silicon segfaults inside transformers' threaded weight conversion;
   the pipeline loads bf16 there, so it is not affected.)
   The hosted-API adapters are tested against stubbed clients only.
9. **A small local model can mislabel a correct number.** In testing, the 4B
   model quoted the local-relief maximum (50 m) as the area's elevation
   difference (485 m). The payload now states the elevation range outright and
   defines local relief; still read the Measured section against
   `reports/analysis_payload.json`.

## Tests

```bash
python -m pytest -q                    # everything
python -m pytest -q -m "not network"   # offline only
python -m pytest -q -m network         # upstream canaries
ruff check src scripts tests
```

The suite covers CRS round-trips (sub-millimetre), grid nesting, tile
arithmetic against verified indices, HTTP error classification, slope/aspect
against analytic planes, sampler correctness, fusion rule priority, the OOM
ladder, prompt contracts, and the rule that notebooks contain no logic.

## Licensing

The code is released under the [MIT License](LICENSE). The **data is not**: GSI elevation and imagery are subject to
the GSI terms of use, and OpenStreetMap data is ODbL, which has share-alike
obligations for derived databases — `fusion_cells.parquet` is one. Model
weights carry their own licences. Check all of these before redistributing
anything from `outputs/`.

## Conventions for contributors

See `CLAUDE.md`. The short version: processing logic never goes in a notebook;
CRS is never implicit; large clouds are never loaded whole; raw data is never
overwritten; external specifications are never guessed; and each milestone ends
with the tests green.
