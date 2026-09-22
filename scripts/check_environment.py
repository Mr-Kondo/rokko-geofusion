#!/usr/bin/env python
"""Report the runtime environment and verify that the project can run here.

    python scripts/check_environment.py
    python scripts/check_environment.py --config configs/rokko.yaml --json

Exit code is non-zero when a *required* dependency is missing, so the script
doubles as a smoke test in CI and as the first cell of the Colab notebook.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow `python scripts/check_environment.py` without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rokko_geofusion import __version__  # noqa: E402
from rokko_geofusion.config import load_config  # noqa: E402
from rokko_geofusion.environment import (  # noqa: E402
    detect_environment,
    make_resource_profile,
    select_device,
)
from rokko_geofusion.exceptions import ConfigError  # noqa: E402
from rokko_geofusion.utils.logging import setup_logging  # noqa: E402

REQUIRED_PACKAGES = (
    "numpy",
    "scipy",
    "pandas",
    "pydantic",
    "pyproj",
    "shapely",
    "rasterio",
    "geopandas",
    "pyarrow",
    "requests",
    "PIL",
)
OPTIONAL_PACKAGES = {
    "laspy": "LAZ/LAS point cloud IO  -> pip install -e '.[pointcloud]'",
    "torch": "segmentation & point-cloud ML  -> pip install -e '.[ml]'",
    "transformers": "segmentation checkpoints  -> pip install -e '.[ml]'",
    "sklearn": "clustering / metrics  -> pip install -e '.[ml]'",
    "matplotlib": "figures  -> pip install -e '.[viz]'",
    "plotly": "3D point cloud display  -> pip install -e '.[viz]'",
    "folium": "2D map display  -> pip install -e '.[viz]'",
}

_OK = "OK  "
_MISS = "MISS"
_WARN = "WARN"


def _row(label: str, value: object) -> str:
    return f"  {label:<22} {value}"


def _section(title: str) -> str:
    return f"\n{title}\n" + "-" * len(title)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=None, help="also validate this config file")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument("--quiet", action="store_true", help="only report problems")
    args = parser.parse_args(argv)

    setup_logging("WARNING" if args.quiet else "INFO")
    env = detect_environment()
    lines: list[str] = []

    lines.append(_section(f"rokko-geofusion {__version__} -- environment"))
    lines.append(_row("Python", f"{env.python_version}  ({env.python_executable})"))
    lines.append(_row("Platform", env.platform))
    lines.append(_row("OS / machine", f"{env.os_name} / {env.machine}"))
    lines.append(_row("CPU", f"{env.cpu_model}  ({env.cpu_count} logical cores)"))
    lines.append(_row("RAM total", f"{env.ram_total_gb} GB" if env.ram_total_gb else "unknown"))
    lines.append(
        _row("RAM available", f"{env.ram_available_gb} GB" if env.ram_available_gb
             else "unknown (install psutil)")
    )
    lines.append(_row("Google Colab", "yes" if env.in_colab else "no"))
    lines.append(_row("Notebook kernel", "yes" if env.in_notebook else "no"))

    lines.append(_section("Accelerator"))
    lines.append(_row("PyTorch", env.torch_version or "not installed"))
    lines.append(_row("CUDA available", env.cuda_available))
    lines.append(_row("CUDA version", env.cuda_version or "-"))
    lines.append(_row("cuDNN", env.cudnn_version or "-"))
    lines.append(_row("Apple MPS", env.mps_available))
    if env.gpus:
        for index, gpu in enumerate(env.gpus):
            lines.append(
                _row(f"GPU[{index}]", f"{gpu.name}  {gpu.total_vram_gb} GB VRAM  "
                                     f"(sm_{(gpu.capability or '').replace('.', '')})")
            )
    else:
        lines.append(_row("GPU", "none detected -> CPU / MPS execution"))
    lines.append(_row("Selected device", select_device("auto", env)))

    lines.append(_section("Geospatial stack"))
    lines.append(_row("GDAL (rasterio)", env.gdal_version or "not available"))
    lines.append(_row("PDAL", env.pdal_version or "not installed (optional; laspy is used)"))

    lines.append(_section("Packages"))
    missing_required: list[str] = []
    missing_optional: list[str] = []
    for name in REQUIRED_PACKAGES:
        version = env.packages.get(name)
        status = _OK if version else _MISS
        if not version:
            missing_required.append(name)
        lines.append(_row(f"[{status}] {name}", version or "REQUIRED - not importable"))
    for name, why in OPTIONAL_PACKAGES.items():
        version = env.packages.get(name)
        if not version:
            missing_optional.append(name)
        lines.append(_row(f"[{_OK if version else _WARN}] {name}", version or f"optional: {why}"))

    config_report: dict[str, object] | None = None
    if args.config:
        lines.append(_section("Configuration"))
        try:
            config = load_config(args.config)
        except ConfigError as exc:
            lines.append(_row("[FAIL] config", str(exc).splitlines()[0]))
            missing_required.append("config")
        else:
            paths = config.paths
            profile = make_resource_profile(config, env)
            lines.append(_row("file", config.source_path))
            lines.append(_row("fingerprint", config.fingerprint()))
            lines.append(_row("ROI", config.roi.key))
            lines.append(
                _row("CRS", f"geographic={config.crs.geographic} "
                            f"projected={config.crs.projected} tile={config.crs.tile}")
            )
            lines.append(_row("DEM provider", config.lidar.dem.provider))
            lines.append(_row("DSM provider", config.lidar.dsm.provider))
            lines.append(_row("imagery", f"{config.imagery.provider}:"
                                         f"{config.imagery.dataset}@z{config.imagery.zoom}"))
            lines.append(_row("output root", paths.out_root))
            lines.append(
                _row("resource profile", f"tier={profile.tier} device={profile.device} "
                                         f"seg_tile={profile.seg_tile_px}px "
                                         f"seg_batch={profile.seg_batch_size} "
                                         f"pc_points={profile.pc_num_points}")
            )
            config_report = {
                "fingerprint": config.fingerprint(),
                "roi_key": config.roi.key,
                "profile": profile.to_dict(),
            }

    if args.json:
        payload = {
            "version": __version__,
            "environment": env.to_dict(),
            "missing_required": missing_required,
            "missing_optional": missing_optional,
            "config": config_report,
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    elif not args.quiet:
        print("\n".join(lines))

    if missing_required:
        print(
            f"\nFAIL: {len(missing_required)} required component(s) missing: "
            f"{', '.join(missing_required)}\n"
            "      Install them with:  pip install -e '.[all]'",
            file=sys.stderr,
        )
        return 1
    if not args.quiet:
        note = ""
        if missing_optional:
            note = f"  ({len(missing_optional)} optional package(s) missing: " \
                   f"{', '.join(missing_optional)})"
        print(f"\nPASS: all required components are available.{note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
