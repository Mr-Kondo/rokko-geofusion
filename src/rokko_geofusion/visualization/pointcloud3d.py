"""3D point cloud display.

Analysis data and display data are separate: this module always thins the
cloud to ``visualization.max_display_points`` before handing anything to a
browser widget, because a few million points will freeze a Colab output cell.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from rokko_geofusion.config import Config
from rokko_geofusion.lidar.pointcloud import read_point_cloud, subsample_for_display

logger = logging.getLogger(__name__)


def _hex_colours(rgb: np.ndarray) -> list[str]:
    return [f"#{r:02x}{g:02x}{b:02x}" for r, g, b in rgb]


def plot_point_cloud(
    config: Config,
    path: Path | str,
    *,
    color_by: str = "rgb",
    max_points: int | None = None,
    point_size: float = 1.6,
    z_exaggeration: float = 1.0,
    title: str | None = None,
):
    """Render a cloud with plotly. ``color_by`` is ``rgb``, ``z`` or a constant."""
    import plotly.graph_objects as go

    budget = max_points or config.visualization.max_display_points
    xyz, rgb = read_point_cloud(path)
    total = len(xyz)
    xyz, rgb = subsample_for_display(xyz, rgb, budget, seed=config.project.seed)
    if total > len(xyz):
        logger.info("displaying %s of %s points (display budget)",
                    f"{len(xyz):,}", f"{total:,}")

    # Plot in ROI-local metres: absolute plane-rectangular coordinates are
    # large negative numbers and make the plotly axes unreadable.
    origin = xyz.min(axis=0)
    local = xyz - origin

    if color_by == "rgb" and rgb is not None:
        marker = {"size": point_size, "color": _hex_colours(rgb)}
    elif color_by == "z":
        marker = {
            "size": point_size,
            "color": xyz[:, 2],
            "colorscale": "Viridis",
            "colorbar": {"title": "elevation (m)"},
        }
    else:
        marker = {"size": point_size, "color": color_by}

    figure = go.Figure(
        data=[
            go.Scatter3d(
                x=local[:, 0],
                y=local[:, 1],
                z=local[:, 2] * z_exaggeration,
                mode="markers",
                marker=marker,
                hovertemplate="x=%{x:.1f} m<br>y=%{y:.1f} m<br>z=%{z:.1f} m<extra></extra>",
            )
        ]
    )
    # np.ptp(...) rather than ndarray.ptp(): the method was removed in NumPy 2.
    span_x = float(np.ptp(local[:, 0]))
    span_y = float(np.ptp(local[:, 1]))
    span_z = float(np.ptp(local[:, 2])) * z_exaggeration
    largest = max(span_x, span_y, 1.0)
    figure.update_layout(
        title=title or f"{Path(path).name}  ({len(xyz):,} of {total:,} points shown)",
        scene={
            "xaxis_title": "east (m)",
            "yaxis_title": "north (m)",
            "zaxis_title": f"elevation (m){'' if z_exaggeration == 1 else f' x{z_exaggeration}'}",
            "aspectmode": "manual",
            "aspectratio": {
                "x": span_x / largest,
                "y": span_y / largest,
                "z": max(span_z / largest, 0.12),
            },
        },
        margin={"l": 0, "r": 0, "t": 40, "b": 0},
        height=650,
    )
    return figure
