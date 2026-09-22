"""Render the images a vision model is allowed to look at.

A VLM never receives the raw point cloud or the raw rasters. It receives the
same rendered views a human analyst would look at, so that what the model saw
is exactly what the notebook shows and what the report cites.

Every image carries a caption stating what it is, its units and its CRS. The
model is asked to *interpret* these views; all numbers come from Python.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rokko_geofusion.config import Config
from rokko_geofusion.crs import RoiGeometry

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VlmImage:
    """One rendered view, with the caption that accompanies it."""

    name: str
    path: Path
    caption: str

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "path": str(self.path), "caption": self.caption}


def _save(figure, path: Path, dpi: int) -> Path:
    from rokko_geofusion.visualization.plots import save_figure

    return save_figure(figure, path, dpi=dpi)


def render_vlm_inputs(
    config: Config,
    roi: RoiGeometry,
    *,
    max_images: int | None = None,
    overwrite: bool = True,
) -> list[VlmImage]:
    """Render up to ``max_images`` views of the ROI for a vision model."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from rokko_geofusion.visualization.plots import show_classes, show_raster, show_rgb

    limit = max_images or config.vlm.max_images
    out_dir = config.paths.figures / "vlm_inputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    dpi = config.visualization.figure_dpi
    interim, raster = config.paths.interim, config.paths.raster
    images: list[VlmImage] = []

    def add(name: str, caption: str, render) -> None:
        if len(images) >= limit:
            return
        path = out_dir / f"{name}.png"
        if path.is_file() and not overwrite:
            images.append(VlmImage(name, path, caption))
            return
        figure, ax = plt.subplots(figsize=(7, 7))
        try:
            render(ax)
        except (ValueError, OSError) as exc:
            logger.warning("skipping VLM input %s: %s", name, exc)
            plt.close(figure)
            return
        figure.tight_layout()
        _save(figure, path, dpi)
        plt.close(figure)
        images.append(VlmImage(name, path, caption))

    extent_note = (
        f"{roi.width_m:.0f} x {roi.height_m:.0f} m, north up, {config.crs.projected}"
    )

    if (interim / "orthophoto.tif").is_file():
        add(
            "orthophoto",
            f"Aerial orthophoto of the area ({config.imagery.attribution}, "
            f"{config.imagery.resolution_m} m per pixel). {extent_note}.",
            lambda ax: show_rgb(interim / "orthophoto.tif", ax=ax, title="orthophoto"),
        )

    cloud_path = config.paths.pointcloud / f"cloud.{config.output.pointcloud_format}"
    if cloud_path.is_file():
        add(
            "pointcloud_oblique",
            "Oblique rendering of the RGB point cloud (colour sampled from the "
            "orthophoto, height from the elevation model). Vertical exaggeration "
            "1x; viewpoint from the south-east.",
            lambda ax: _render_cloud_oblique(config, cloud_path, ax),
        )

    if (raster / "elevation.tif").is_file():
        add(
            "elevation",
            f"Ground elevation (DEM) in metres above sea level, colour-mapped. "
            f"{extent_note}.",
            lambda ax: show_raster(raster / "elevation.tif", ax=ax, title="elevation (m)",
                                   cmap="terrain", colorbar_label="m"),
        )

    if (raster / "slope.tif").is_file():
        add(
            "slope",
            f"Terrain slope in degrees ({config.terrain.slope_units}); brighter is "
            f"steeper. {extent_note}.",
            lambda ax: show_raster(raster / "slope.tif", ax=ax, title="slope (deg)",
                                   cmap="magma", colorbar_label="deg"),
        )

    if (raster / "ndsm.tif").is_file():
        add(
            "ndsm",
            f"Height above ground (nDSM) in metres: DSM minus DEM. {extent_note}.",
            lambda ax: show_raster(raster / "ndsm.tif", ax=ax, title="height above ground (m)",
                                   cmap="viridis", colorbar_label="m"),
        )

    if (raster / "segmentation_class.tif").is_file():
        add(
            "segmentation",
            "Land-cover classes predicted from the orthophoto by a pretrained "
            f"segmentation model ({', '.join(config.segmentation.classes)}). "
            f"Indicative only, not surveyed. {extent_note}.",
            lambda ax: show_classes(raster / "segmentation_class.tif",
                                    config.segmentation.classes, ax=ax,
                                    title="image semantics"),
        )

    if (raster / "fused_class.tif").is_file():
        add(
            "fused",
            "Fused land-cover classes combining image semantics, terrain and "
            f"mapped geometry ({', '.join(config.fusion.classes)}). {extent_note}.",
            lambda ax: show_classes(raster / "fused_class.tif", config.fusion.classes,
                                    ax=ax, title="fused classes"),
        )

    # Extras, only if the budget allows.
    if (raster / "hillshade.tif").is_file():
        add(
            "hillshade",
            "Shaded relief of the ground surface, illuminated from the north-west "
            f"at 45 degrees. Brightness encodes slope orientation only. {extent_note}.",
            lambda ax: show_raster(raster / "hillshade.tif", ax=ax, title="hillshade",
                                   cmap="gray"),
        )

    logger.info("rendered %d VLM input image(s) in %s", len(images), out_dir)
    return images


def _render_cloud_oblique(config: Config, cloud_path: Path, ax) -> None:
    """Draw the cloud as an oblique scatter on an ordinary 2-D axis.

    A true 3-D axis is avoided on purpose: the projection below is a simple,
    reproducible isometric one, which keeps the rendering identical between
    runs (V6) and keeps the figure readable at report size.
    """
    from rokko_geofusion.lidar.pointcloud import read_point_cloud

    xyz, rgb = read_point_cloud(cloud_path, max_points=250_000, seed=config.project.seed)
    if rgb is None:
        rgb = np.full((len(xyz), 3), 160, np.uint8)

    local = xyz - xyz.min(axis=0)
    azimuth, tilt = np.radians(45.0), np.radians(35.0)
    screen_x = local[:, 0] * np.cos(azimuth) - local[:, 1] * np.sin(azimuth)
    depth = local[:, 0] * np.sin(azimuth) + local[:, 1] * np.cos(azimuth)
    screen_y = depth * np.sin(tilt) + local[:, 2] * np.cos(tilt)

    order = np.argsort(-depth)  # painter's algorithm: far points first
    # A dark canvas and overlapping markers: with a white background the gaps
    # between points wash the colours out and the surface reads as blank paper.
    ax.set_facecolor("#101010")
    # Marker area in points^2, sized so neighbouring points just touch: a
    # roughly square cloud has sqrt(n) points across a ~7 inch figure.
    points_across = max(np.sqrt(len(xyz)), 1.0)
    marker_size = max(1.0, (700.0 / points_across) ** 2)
    ax.scatter(screen_x[order], screen_y[order], c=rgb[order] / 255.0,
               s=marker_size, marker="s", linewidths=0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"RGB point cloud, oblique view ({len(xyz):,} points shown)", fontsize=10)
