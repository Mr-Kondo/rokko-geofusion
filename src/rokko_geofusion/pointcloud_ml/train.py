"""Train, embed, cluster and compare feature sets.

There is no labelled ground truth for this ROI, so no supervised accuracy is
reported. Instead each feature set gets:

* a self-supervised contrastive encoder (no labels used in training);
* a k-means clustering of the resulting tile embeddings;
* silhouette score (structure of the embedding space) and adjusted mutual
  information against the *independently derived* fused classes.

The AMI for ``xyz_rgb_terrain_semantic`` is reported with an explicit caveat:
that feature set contains the image semantics from which the fused classes were
partly derived, so its agreement is inflated by construction. It is not
evidence that the model learned more.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from rokko_geofusion.config import Config
from rokko_geofusion.environment import ResourceProfile
from rokko_geofusion.exceptions import ResourceError, UnsupportedError
from rokko_geofusion.pointcloud_ml.dataset import (
    FusionTileDataset,
    feature_dimension,
)
from rokko_geofusion.pointcloud_ml.pointnet import (
    augment,
    build_pointnet,
    describe_model,
    nt_xent_loss,
)

logger = logging.getLogger(__name__)


@dataclass
class FeatureSetResult:
    feature_set: str
    in_channels: int
    epochs: int
    final_loss: float
    loss_curve: list[float]
    silhouette: float | None
    adjusted_mutual_information: float | None
    cluster_sizes: dict[str, int]
    train_seconds: float
    device: str
    caveat: str | None = None
    model: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _is_out_of_memory(error: BaseException) -> bool:
    text = str(error).lower()
    return "out of memory" in text or "mps backend out of memory" in text


def train_feature_set(
    config: Config,
    dataset: FusionTileDataset,
    feature_set: str,
    *,
    profile: ResourceProfile,
    seed: int = 0,
) -> tuple[FeatureSetResult, np.ndarray]:
    """Train one encoder and return ``(result, tile_embeddings)``."""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise UnsupportedError(
            "point cloud ML needs torch: pip install -e '.[ml]'"
        ) from exc

    settings = config.pointcloud_ml
    device = profile.device
    batch_size = max(2, min(profile.pc_batch_size or settings.batch_size, len(dataset)))
    in_channels = feature_dimension(feature_set, len(config.segmentation.classes))

    feature_dropout = float(settings.augment_feature_dropout)
    torch.manual_seed(seed)
    model = build_pointnet(in_channels, embedding_dim=settings.embedding_dim).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=settings.learning_rate)
    rng = np.random.default_rng(seed)

    n_tiles = len(dataset)
    loss_curve: list[float] = []
    started = time.perf_counter()

    model.train()
    for epoch in range(settings.epochs):
        order = rng.permutation(n_tiles)
        epoch_losses: list[float] = []
        for start in range(0, n_tiles - batch_size + 1, batch_size):
            indices = order[start:start + batch_size]
            raw = dataset.batch(indices, feature_set, rng=rng)
            view_a = np.stack([augment(tile, rng, feature_dropout=feature_dropout)
                               for tile in raw])
            view_b = np.stack([augment(tile, rng, feature_dropout=feature_dropout)
                               for tile in raw])
            tensor_a = torch.from_numpy(view_a).to(device)
            tensor_b = torch.from_numpy(view_b).to(device)

            try:
                _, projection_a = model(tensor_a)
                _, projection_b = model(tensor_b)
                loss = nt_xent_loss(projection_a, projection_b)
                optimiser.zero_grad(set_to_none=True)
                loss.backward()
                optimiser.step()
            except RuntimeError as exc:
                if _is_out_of_memory(exc):
                    raise ResourceError(f"out of accelerator memory: {exc}") from exc
                raise
            epoch_losses.append(float(loss.detach().cpu()))

        mean_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        loss_curve.append(mean_loss)
        if epoch == 0 or (epoch + 1) % 5 == 0 or epoch == settings.epochs - 1:
            logger.info("%-26s epoch %2d/%d  loss %.4f",
                        feature_set, epoch + 1, settings.epochs, mean_loss)

    train_seconds = time.perf_counter() - started

    # Embeddings for every tile, with augmentation switched off.
    model.eval()
    embeddings: list[np.ndarray] = []
    eval_rng = np.random.default_rng(seed)
    with torch.no_grad():
        for start in range(0, n_tiles, batch_size):
            indices = list(range(start, min(start + batch_size, n_tiles)))
            raw = dataset.batch(indices, feature_set, rng=eval_rng)
            tensor = torch.from_numpy(raw).to(device)
            if tensor.shape[0] == 1:
                # BatchNorm needs more than one sample in training mode only;
                # eval mode is fine, but keep the shape explicit.
                embedding = model(tensor, project=False)
            else:
                embedding = model(tensor, project=False)
            embeddings.append(embedding.cpu().numpy())
    tile_embeddings = np.concatenate(embeddings, axis=0)

    result = FeatureSetResult(
        feature_set=feature_set,
        in_channels=in_channels,
        epochs=settings.epochs,
        final_loss=loss_curve[-1] if loss_curve else float("nan"),
        loss_curve=loss_curve,
        silhouette=None,
        adjusted_mutual_information=None,
        cluster_sizes={},
        train_seconds=train_seconds,
        device=device,
        model={**describe_model(model), "augment_feature_dropout": feature_dropout},
    )
    return result, tile_embeddings


def evaluate_embeddings(
    embeddings: np.ndarray,
    reference_classes: np.ndarray,
    *,
    n_clusters: int,
    seed: int = 0,
) -> tuple[np.ndarray, float | None, float | None, dict[str, int]]:
    """k-means the embeddings and score them without using any labels to fit."""
    try:
        from sklearn.cluster import KMeans
        from sklearn.metrics import adjusted_mutual_info_score, silhouette_score
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise UnsupportedError(
            "clustering needs scikit-learn: pip install -e '.[ml]'"
        ) from exc

    n_clusters = int(min(n_clusters, max(2, len(embeddings) // 2)))
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed)
    labels = kmeans.fit_predict(embeddings)

    silhouette = (float(silhouette_score(embeddings, labels))
                  if len(set(labels.tolist())) > 1 else None)
    ami = (float(adjusted_mutual_info_score(reference_classes, labels))
           if reference_classes is not None and len(set(reference_classes.tolist())) > 1
           else None)
    sizes = {str(int(label)): int(count)
             for label, count in zip(*np.unique(labels, return_counts=True), strict=True)}
    return labels, silhouette, ami, sizes


def run_comparison(
    config: Config,
    *,
    profile: ResourceProfile,
    table_path: Path | None = None,
    feature_sets: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Train every configured feature set and compare them."""
    settings = config.pointcloud_ml
    if not settings.enabled:
        raise UnsupportedError("pointcloud_ml.enabled is false")

    table_path = table_path or (config.paths.vector / "fusion_cells.parquet")
    dataset = FusionTileDataset(
        config, table_path, feature_sets=feature_sets, seed=config.project.seed
    )
    reference = dataset.dominant_classes()

    results: list[FeatureSetResult] = []
    embeddings_by_set: dict[str, np.ndarray] = {}
    cluster_labels: dict[str, np.ndarray] = {}
    current = profile

    for feature_set in dataset.feature_sets:
        attempts = 0
        while True:
            try:
                result, embeddings = train_feature_set(
                    config, dataset, feature_set, profile=current, seed=config.project.seed
                )
                break
            except ResourceError as exc:
                attempts += 1
                previous = (current.device, current.pc_batch_size, current.pc_num_points)
                current = current.downscale()
                if attempts > 4 or (current.device, current.pc_batch_size,
                                    current.pc_num_points) == previous:
                    raise
                logger.warning("%s -- retrying with device=%s batch=%d points=%d",
                               exc, current.device, current.pc_batch_size,
                               current.pc_num_points)

        labels, silhouette, ami, sizes = evaluate_embeddings(
            embeddings, reference, n_clusters=settings.n_clusters, seed=config.project.seed
        )
        result.silhouette = silhouette
        result.adjusted_mutual_information = ami
        result.cluster_sizes = sizes
        if feature_set == "xyz_rgb_terrain_semantic":
            result.caveat = (
                "This feature set contains the image semantics that the fused "
                "classes were partly derived from, so its agreement with them is "
                "inflated by construction and is not comparable with the others."
            )
        results.append(result)
        embeddings_by_set[feature_set] = embeddings
        cluster_labels[feature_set] = labels
        logger.info(
            "%-26s silhouette %s  AMI %s  (%.1f s on %s)",
            feature_set,
            f"{silhouette:.3f}" if silhouette is not None else "n/a",
            f"{ami:.3f}" if ami is not None else "n/a",
            result.train_seconds, result.device,
        )

    return {
        "dataset": dataset.describe(),
        "results": [result.to_dict() for result in results],
        "embeddings": embeddings_by_set,
        "cluster_labels": cluster_labels,
        "tile_origins": dataset.tile_origins,
        "reference_classes": reference,
        "reference_class_names": list(config.fusion.classes),
        "evaluation_note": (
            "No labelled ground truth exists for this ROI. No supervised accuracy "
            "is reported. Silhouette measures embedding structure; AMI measures "
            "agreement with the rule-based fused classes, which are themselves "
            "derived, not surveyed."
        ),
    }
