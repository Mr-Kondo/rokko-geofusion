#!/usr/bin/env python
"""Train the point cloud encoder and compare the four feature sets.

    python scripts/run_pointcloud_ml.py --config configs/rokko.yaml
    python scripts/run_pointcloud_ml.py --config configs/rokko.yaml \
        --feature-sets xyz xyz_rgb --set pointcloud_ml.epochs=5

No labelled ground truth exists for this ROI, so no supervised accuracy is
reported. Each feature set is trained self-supervised (NT-Xent over augmented
views) and evaluated by clustering quality and by agreement with the
independently derived fused classes.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from rokko_geofusion.pointcloud_ml.train import run_comparison  # noqa: E402
from rokko_geofusion.utils.cli import build_parser, init_stage  # noqa: E402
from rokko_geofusion.utils.metadata import write_json  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = build_parser(__doc__.splitlines()[0])
    parser.add_argument("--feature-sets", nargs="+", default=None,
                        help="override pointcloud_ml.feature_sets")
    parser.add_argument("--no-figure", action="store_true", help="skip the cluster map")
    args = parser.parse_args(argv)
    ctx = init_stage(args, stage_name="run_pointcloud_ml", need_profile=True)
    config, logger = ctx.config, ctx.logger

    outcome = run_comparison(config, profile=ctx.profile, feature_sets=args.feature_sets)

    embeddings_dir = config.paths.metrics
    npz_path = embeddings_dir / "pointcloud_ml_embeddings.npz"
    np.savez_compressed(
        npz_path,
        tile_origins=np.asarray(outcome["tile_origins"], dtype=np.float64),
        reference_classes=outcome["reference_classes"],
        **{f"embedding__{name}": values for name, values in outcome["embeddings"].items()},
        **{f"clusters__{name}": values for name, values in outcome["cluster_labels"].items()},
    )
    logger.info("wrote %s", npz_path)

    summary = {
        "dataset": outcome["dataset"],
        "results": outcome["results"],
        "evaluation_note": outcome["evaluation_note"],
        "embeddings_file": str(npz_path),
        "reference_class_names": outcome["reference_class_names"],
    }
    write_json(config.paths.metrics / "pointcloud_ml.json", summary)

    header = f"{'feature set':<28}{'ch':>4}{'loss':>9}{'silhouette':>13}{'AMI':>9}{'secs':>8}"
    logger.info("%s", header)
    logger.info("%s", "-" * len(header))
    for result in outcome["results"]:
        logger.info(
            "%-28s%4d%9.4f%13s%9s%8.1f",
            result["feature_set"], result["in_channels"], result["final_loss"],
            f"{result['silhouette']:.3f}" if result["silhouette"] is not None else "n/a",
            f"{result['adjusted_mutual_information']:.3f}"
            if result["adjusted_mutual_information"] is not None else "n/a",
            result["train_seconds"],
        )
        if result.get("caveat"):
            logger.warning("  %s: %s", result["feature_set"], result["caveat"])

    if not args.no_figure:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from rokko_geofusion.visualization.plots import save_figure

        origins = np.asarray(outcome["tile_origins"])
        names = list(outcome["cluster_labels"])
        figure, axes = plt.subplots(1, len(names) + 1, figsize=(4.6 * (len(names) + 1), 4.8),
                                    squeeze=False)
        size = config.pointcloud_ml.tile_size_m
        reference = outcome["reference_classes"]
        axes[0][0].scatter(origins[:, 0], origins[:, 1], c=reference, cmap="tab10",
                           s=max(4.0, (size / 8) ** 2), marker="s")
        axes[0][0].set_title("dominant fused class per tile", fontsize=9)
        for index, name in enumerate(names, start=1):
            axes[0][index].scatter(origins[:, 0], origins[:, 1],
                                   c=outcome["cluster_labels"][name], cmap="tab10",
                                   s=max(4.0, (size / 8) ** 2), marker="s")
            axes[0][index].set_title(f"clusters: {name}", fontsize=9)
        for ax in axes[0]:
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
        figure.suptitle(
            "Point cloud ML - self-supervised tile embeddings, k-means clusters "
            "(no ground truth labels)", fontsize=11,
        )
        save_figure(figure, config.paths.figures / "pointcloud_ml.png",
                    dpi=config.visualization.figure_dpi)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
