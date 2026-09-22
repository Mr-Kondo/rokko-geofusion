# CLAUDE.md — working rules for this repository

Read this before changing anything. These rules exist because the project's
value is *spatial correctness and reproducibility*, not "the cell ran".

## Hard rules

1. **Do not put processing logic in notebooks.**
   `notebooks/` may only: install, load config, call `scripts/` or import from
   `rokko_geofusion`, read results, visualise, compare. No downloads, no CRS
   maths, no model code, no class/function definitions of substance.
2. **`scripts/` and `src/` come first.** New behaviour goes into
   `src/rokko_geofusion/<subpackage>/`, is exposed by a thin `scripts/*.py`
   CLI, and only then gets called from a notebook.
3. **Never treat a CRS implicitly.** Every dataset records its source CRS,
   the processing CRS and the output CRS. Reproject with `pyproj`/`rasterio`
   explicitly. EPSG codes live in `configs/*.yaml`, never in `src/` (enforced
   by `tests/test_config.py::test_no_epsg_codes_are_hardcoded_in_processing_modules`).
4. **Never load a large point cloud in one piece.** Use spatial tiling,
   chunked reads, voxel downsampling and `float32`. Analysis data and display
   data are separate; display is always downsampled.
5. **Never overwrite raw data.** `data/raw/` is append-only. Derived products
   go to `data/interim/`, `data/processed/` and `outputs/`.
6. **Never commit outputs or data.** `data/*` and `outputs/*` are gitignored
   except for `.gitkeep`.
7. **Never guess an external specification.** If a URL, API, file name or
   model id cannot be verified, raise `ConfigurationRequiredError` /
   `UnsupportedError` and say so in the README. A plausible-looking fake URL
   or a stub that returns synthetic data is worse than an explicit failure.
8. **Every milestone ends with passing tests.** `pytest` must be green before
   moving on; add tests with the feature, not afterwards.
9. **Do not break working behaviour to reach the next phase.** Re-run the
   previous milestone's script and tests after refactoring.

## Conventions

- **Logging, not printing.** Library code uses `logging.getLogger(__name__)`.
  Failures use `utils.logging.log_failure_context(...)` so that the log names
  *what* failed, the *target*, the *ROI*, the *CRS* and the *likely cause*.
  An upstream service change must never be logged as "file not found".
- **Configuration is typed.** `config.py` uses pydantic with `extra="forbid"`;
  a typo in YAML is an error, not a silently ignored key.
- **Every artefact gets a `.meta.json` sidecar** via `utils.metadata`
  (CRS, source, resolution, acquisition, processing parameters, config
  fingerprint). Regular gridded products are never labelled as raw LiDAR.
- **Resources are detected, never assumed.** Use
  `environment.make_resource_profile(config)` for tile/batch/point budgets and
  `ResourceProfile.downscale()` for the OOM ladder
  (batch → tile → point count → CPU). No code may assume a specific GPU.
- **Determinism.** `utils.seed.set_global_seed` is called by every stage;
  re-running one ROI must reproduce byte-comparable statistics (V6).
- **Numeric work belongs in Python.** VLM/LLM adapters interpret and explain;
  they never compute areas, slopes, ratios or elevations.

## Layout

| Path | Responsibility |
|---|---|
| `src/rokko_geofusion/` | all processing logic, importable package |
| `scripts/` | one thin CLI per stage + `run_pipeline.py` |
| `configs/` | experiment configuration (the only place with URLs/EPSG/thresholds) |
| `notebooks/` | Colab UI: call, display, compare |
| `data/raw|interim|processed` | inputs and intermediates (gitignored) |
| `outputs/` | figures, metrics, reports, clouds (gitignored) |
| `tests/` | unit + spatial-consistency tests |

## Commands

```bash
python scripts/check_environment.py --config configs/rokko.yaml
python scripts/run_pipeline.py --config configs/rokko.yaml --stage <stage>
python -m pytest -q
python -m pytest -q -m "not network"      # offline subset
```
