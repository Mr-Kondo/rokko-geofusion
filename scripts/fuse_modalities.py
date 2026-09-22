#!/usr/bin/env python
"""Fuse LiDAR-derived terrain, imagery semantics and GIS into one feature table.

    python scripts/fuse_modalities.py --config configs/rokko.yaml

Writes:
  outputs/<roi>/vector/fusion_cells.parquet   one row per cell, all modalities
  outputs/<roi>/raster/fused_class.tif        the rule-based fused class
  outputs/<roi>/pointcloud/cloud_fused.laz    RGB cloud, class in LAS classification
  outputs/<roi>/metrics/fusion.json           class fractions, rules fired, gaps
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rokko_geofusion.crs import roi_from_config  # noqa: E402
from rokko_geofusion.fusion.build import build_fusion  # noqa: E402
from rokko_geofusion.utils.cli import build_parser, init_stage  # noqa: E402
from rokko_geofusion.utils.metadata import write_json  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = build_parser(__doc__.splitlines()[0])
    parser.add_argument("--preview", action="store_true", default=True,
                        help="also write a comparison figure")
    args = parser.parse_args(argv)
    ctx = init_stage(args, stage_name="fuse_modalities")
    config, logger = ctx.config, ctx.logger

    roi = roi_from_config(config)
    product = build_fusion(config, roi, overwrite=ctx.overwrite)

    payload = {
        "table": str(product.table_path),
        "raster": str(product.raster_path),
        "pointcloud": str(product.pointcloud_path) if product.pointcloud_path else None,
        "grid": product.grid.to_dict(),
        "class_names": product.class_names,
        "unavailable": product.unavailable,
        **product.statistics,
    }
    write_json(config.paths.metrics / "fusion.json", payload)

    if args.preview:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from rokko_geofusion.visualization.plots import (
            save_figure,
            show_classes,
            show_rgb,
        )

        figure, axes = plt.subplots(1, 3, figsize=(16.5, 5.8))
        show_rgb(config.paths.interim / "orthophoto.tif", ax=axes[0], title="orthophoto")
        show_classes(config.paths.raster / "segmentation_class.tif",
                     config.segmentation.classes, ax=axes[1], title="image semantics")
        show_classes(product.raster_path, product.class_names, ax=axes[2],
                     title="fused class (image + terrain + GIS)")
        figure.suptitle(f"Multimodal fusion - {roi.key}", fontsize=11)
        save_figure(figure, config.paths.figures / "fusion.png",
                    dpi=config.visualization.figure_dpi)

    for name, reason in product.unavailable.items():
        logger.warning("%-14s UNAVAILABLE: %s", name, reason)
    logger.info("fusion table -> %s (%s rows)", product.table_path, f"{product.n_cells:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
