#!/usr/bin/env python
"""Run semantic segmentation on the orthophoto.

    python scripts/segment_imagery.py --config configs/rokko.yaml

Writes segmentation_class.tif (uint8 class index) and
segmentation_confidence.tif to outputs/<roi>/raster/, plus a coloured preview
and class statistics.

NOTE: the classes come from a pretrained checkpoint and no accuracy figure is
claimed for this ROI. Cross-check them with `python scripts/validate.py
--check SEG`, which scores them against independent OSM geometry.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rokko_geofusion.crs import roi_from_config  # noqa: E402
from rokko_geofusion.segmentation.runner import segment_imagery  # noqa: E402
from rokko_geofusion.utils.cli import build_parser, init_stage  # noqa: E402
from rokko_geofusion.utils.metadata import write_json  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = build_parser(__doc__.splitlines()[0])
    parser.add_argument("--preview", action="store_true", default=True,
                        help="also write a coloured preview PNG")
    args = parser.parse_args(argv)
    ctx = init_stage(args, stage_name="segment_imagery", need_profile=True)
    config = ctx.config

    roi = roi_from_config(config)
    imagery_path = config.paths.interim / "orthophoto.tif"

    product = segment_imagery(
        config, roi, imagery_path, profile=ctx.profile, overwrite=ctx.overwrite
    )

    if args.preview:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from rokko_geofusion.visualization.plots import (
            save_figure,
            show_classes,
            show_raster,
            show_rgb,
        )

        figure, axes = plt.subplots(1, 3, figsize=(16.5, 5.8))
        show_rgb(imagery_path, ax=axes[0], title="orthophoto")
        show_classes(product.class_path, product.class_names, ax=axes[1],
                     title="semantic classes")
        show_raster(product.confidence_path, ax=axes[2], title="confidence",
                    cmap="inferno", percentile_clip=(1, 99), colorbar_label="max p")
        model_info = product.metadata["model"]
        subtitle = "labels corrected by segmentation.label_overrides" \
            if model_info.get("label_overrides") else "checkpoint labels used as published"
        figure.suptitle(
            f"Segmentation - {model_info['model_id']} ({subtitle})", fontsize=11
        )
        save_figure(figure, config.paths.figures / "segmentation.png",
                    dpi=config.visualization.figure_dpi)

    write_json(
        config.paths.metrics / "segmentation.json",
        {
            "class_path": str(product.class_path),
            "confidence_path": str(product.confidence_path),
            "grid": product.grid.to_dict(),
            "class_names": product.class_names,
            **product.statistics,
            "model": product.metadata["model"],
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
