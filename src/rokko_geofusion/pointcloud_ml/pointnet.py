"""A small PointNet encoder.

Deliberately minimal: shared per-point MLPs followed by a symmetric max-pool,
which is the part of PointNet that matters here (a permutation-invariant tile
embedding). The input transform network is omitted -- it triples the parameter
count and mainly helps with arbitrary object poses, while our tiles are already
in a consistent, north-up, metric frame.

No labelled ground truth exists for this ROI, so the encoder is trained with a
self-supervised contrastive objective and evaluated by clustering quality and
by agreement with the independently derived fused classes.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def build_pointnet(in_channels: int, embedding_dim: int = 256,
                   projection_dim: int = 128):  # -> torch.nn.Module
    """Construct the encoder; imported lazily so torch stays optional."""
    import torch
    from torch import nn

    class SharedMLP(nn.Module):
        def __init__(self, sizes: list[int]) -> None:
            super().__init__()
            layers: list[nn.Module] = []
            for previous, current in zip(sizes[:-1], sizes[1:], strict=True):
                layers += [nn.Conv1d(previous, current, 1), nn.BatchNorm1d(current),
                           nn.ReLU(inplace=True)]
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)

    class PointNetEncoder(nn.Module):
        """``(B, N, C)`` -> ``(B, embedding_dim)`` and a projection for the loss."""

        def __init__(self) -> None:
            super().__init__()
            self.point_mlp = SharedMLP([in_channels, 64, 128, 256])
            self.head = nn.Sequential(
                nn.Linear(256, embedding_dim), nn.BatchNorm1d(embedding_dim),
                nn.ReLU(inplace=True),
            )
            self.projection = nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim), nn.ReLU(inplace=True),
                nn.Linear(embedding_dim, projection_dim),
            )
            self.in_channels = in_channels
            self.embedding_dim = embedding_dim

        def forward(self, points, *, project: bool = True):
            # (B, N, C) -> (B, C, N) for Conv1d
            features = self.point_mlp(points.transpose(1, 2))
            pooled = torch.max(features, dim=2).values
            embedding = self.head(pooled)
            if not project:
                return embedding
            return embedding, self.projection(embedding)

    model = PointNetEncoder()
    n_parameters = sum(p.numel() for p in model.parameters())
    logger.info("PointNet encoder: %d input channels, %d-d embedding, %s parameters",
                in_channels, embedding_dim, f"{n_parameters:,}")
    return model


def nt_xent_loss(projection_a, projection_b, temperature: float = 0.1):
    """NT-Xent (SimCLR) loss over two augmented views of the same tiles."""
    import torch
    import torch.nn.functional as functional

    batch = projection_a.shape[0]
    features = functional.normalize(torch.cat([projection_a, projection_b], dim=0), dim=1)
    similarity = features @ features.t() / temperature
    similarity.fill_diagonal_(float("-inf"))

    targets = torch.arange(batch, device=features.device)
    targets = torch.cat([targets + batch, targets], dim=0)
    return functional.cross_entropy(similarity, targets)


def augment(points, rng, *, jitter: float = 0.01, dropout: float = 0.1,
            colour_jitter: float = 0.05, rotate: bool = True,
            feature_dropout: float = 0.25):
    """Random view of a tile: Z-rotation, jitter, point dropout, colour shift.

    Only the first three channels are geometry; channels 3-5, when present, are
    colour. Rotating a one-hot class or a slope value would be meaningless, so
    the remaining channels are instead perturbed by ``feature_dropout``: each
    non-geometry channel is blanked in a view with that probability.

    That step is not cosmetic. Channels such as slope or a class one-hot are
    invariant to rotation and jitter, so without it the two views of a tile
    stay trivially identifiable, the contrastive loss collapses towards zero,
    and feature sets with more channels score *worse* on every downstream
    metric purely as an artefact of the objective.
    """
    import numpy as np

    out = points.copy()
    n, channels = out.shape

    if rotate:
        angle = rng.uniform(0.0, 2.0 * np.pi)
        cos, sin = np.cos(angle), np.sin(angle)
        x, y = out[:, 0].copy(), out[:, 1].copy()
        out[:, 0] = cos * x - sin * y
        out[:, 1] = sin * x + cos * y

    out[:, :3] += rng.normal(0.0, jitter, size=(n, 3)).astype(out.dtype)

    if channels >= 6 and colour_jitter > 0:
        shift = rng.normal(0.0, colour_jitter, size=(1, 3)).astype(out.dtype)
        out[:, 3:6] = np.clip(out[:, 3:6] + shift, 0.0, 1.0)

    if feature_dropout > 0 and channels > 3:
        blanked = rng.random(channels - 3) < feature_dropout
        if blanked.any():
            out[:, 3:][:, blanked] = 0.0

    if dropout > 0:
        keep = rng.random(n) >= dropout
        if keep.sum() >= 8:
            # Keep the point count fixed by resampling the survivors.
            survivors = np.flatnonzero(keep)
            out = out[rng.choice(survivors, size=n, replace=True)]
    return out


def describe_model(model: Any) -> dict[str, Any]:
    return {
        "architecture": "pointnet-encoder (shared MLP 64-128-256 + max pool)",
        "input_transform": False,
        "in_channels": int(getattr(model, "in_channels", -1)),
        "embedding_dim": int(getattr(model, "embedding_dim", -1)),
        "parameters": int(sum(p.numel() for p in model.parameters())),
        "objective": "self-supervised NT-Xent over two augmented views",
    }
