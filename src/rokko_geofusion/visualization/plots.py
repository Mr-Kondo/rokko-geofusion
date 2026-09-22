"""Static figures (matplotlib) shared by the notebook, reports and the VLM.

The same renderers feed the notebook and the VLM input images, so what the
model sees is exactly what the user sees.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

#: Colours for the fused / segmentation class rasters.
CLASS_COLOURS: dict[str, str] = {
    "other": "#9e9e9e",
    "building": "#e15759",
    "road": "#f2c744",
    "vegetation": "#59a14f",
    "tall_vegetation": "#2d6a4f",
    "low_vegetation": "#a7c957",
    "bare_soil": "#b08968",
    "water": "#4e79a7",
    "unknown": "#dddddd",
}


def _read_for_display(path: Path | str, max_side_px: int = 1600):
    """Read a raster downsampled to at most ``max_side_px`` on its long side."""
    import rasterio
    from rasterio.enums import Resampling

    with rasterio.open(path) as dataset:
        scale = max(1, max(dataset.height, dataset.width) // max_side_px)
        out_shape = (dataset.count, dataset.height // scale, dataset.width // scale)
        array = dataset.read(out_shape=out_shape, resampling=Resampling.average)
        nodata = dataset.nodata
        extent = (dataset.bounds.left, dataset.bounds.right,
                  dataset.bounds.bottom, dataset.bounds.top)
    if nodata is not None and np.issubdtype(array.dtype, np.floating) \
            and not np.isnan(nodata):
        array = np.where(array == nodata, np.nan, array)
    return array, extent


def show_raster(
    path: Path | str,
    *,
    ax=None,
    title: str | None = None,
    cmap: str = "viridis",
    percentile_clip: tuple[float, float] = (2.0, 98.0),
    colorbar_label: str | None = None,
    max_side_px: int = 1600,
):
    """Display a single-band raster with a robust stretch."""
    import matplotlib.pyplot as plt

    array, extent = _read_for_display(path, max_side_px)
    band = array[0].astype(float)
    finite = np.isfinite(band)
    if not finite.any():
        raise ValueError(f"{path} has no finite values to display")
    vmin, vmax = np.nanpercentile(band[finite], percentile_clip)

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))
    image = ax.imshow(band, extent=extent, cmap=cmap, vmin=vmin, vmax=vmax,
                      interpolation="nearest")
    ax.set_title(title or Path(path).stem, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    if colorbar_label is not None:
        ax.figure.colorbar(image, ax=ax, shrink=0.78, label=colorbar_label)
    return ax


def show_rgb(path: Path | str, *, ax=None, title: str | None = None,
             max_side_px: int = 1600):
    """Display a 3-band RGB raster."""
    import matplotlib.pyplot as plt

    array, extent = _read_for_display(path, max_side_px)
    if array.shape[0] < 3:
        raise ValueError(f"{path} has {array.shape[0]} band(s); expected 3 (RGB)")
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(np.transpose(array[:3], (1, 2, 0)).astype(np.uint8), extent=extent)
    ax.set_title(title or Path(path).stem, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    return ax


def show_classes(
    path: Path | str,
    class_names: Sequence[str],
    *,
    ax=None,
    title: str | None = None,
    max_side_px: int = 1600,
    legend: bool = True,
):
    """Display an integer class raster with the project colour table."""
    import matplotlib.pyplot as plt
    import rasterio
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch
    from rasterio.enums import Resampling

    with rasterio.open(path) as dataset:
        scale = max(1, max(dataset.height, dataset.width) // max_side_px)
        array = dataset.read(
            1,
            out_shape=(dataset.height // scale, dataset.width // scale),
            resampling=Resampling.nearest,
        )
        extent = (dataset.bounds.left, dataset.bounds.right,
                  dataset.bounds.bottom, dataset.bounds.top)

    colours = [CLASS_COLOURS.get(name, "#cccccc") for name in class_names]
    cmap = ListedColormap(colours)
    norm = BoundaryNorm(np.arange(-0.5, len(class_names) + 0.5), cmap.N)

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(array, extent=extent, cmap=cmap, norm=norm, interpolation="nearest")
    ax.set_title(title or Path(path).stem, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    if legend:
        present = sorted(set(np.unique(array).tolist()))
        handles = [
            Patch(facecolor=colours[value], label=class_names[value])
            for value in present
            if 0 <= value < len(class_names)
        ]
        ax.legend(handles=handles, loc="lower right", fontsize=7, framealpha=0.85)
    return ax


def grid_figure(
    panels: Sequence[dict[str, Any]],
    *,
    ncols: int = 3,
    figsize_per_panel: tuple[float, float] = (5.4, 5.4),
    suptitle: str | None = None,
):
    """Assemble several rasters into one comparison figure.

    Each panel is ``{"path": ..., "kind": "raster|rgb|classes", ...}`` with the
    remaining keys forwarded to the matching renderer.
    """
    import matplotlib.pyplot as plt

    n = len(panels)
    ncols = min(ncols, max(n, 1))
    nrows = int(np.ceil(n / ncols))
    figure, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows),
        squeeze=False,
    )
    flat = axes.ravel()

    for ax, panel in zip(flat, panels, strict=False):
        options = dict(panel)
        kind = options.pop("kind", "raster")
        path = options.pop("path")
        try:
            if kind == "rgb":
                show_rgb(path, ax=ax, **options)
            elif kind == "classes":
                show_classes(path, options.pop("class_names"), ax=ax, **options)
            else:
                show_raster(path, ax=ax, **options)
        except (ValueError, OSError) as exc:
            logger.warning("could not render %s: %s", path, exc)
            ax.text(0.5, 0.5, f"unavailable:\n{Path(str(path)).name}", ha="center",
                    va="center", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
    for ax in flat[n:]:
        ax.axis("off")
    if suptitle:
        figure.suptitle(suptitle, fontsize=12)
    figure.tight_layout()
    return figure


def save_figure(figure, path: Path | str, *, dpi: int = 150) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, dpi=dpi, bbox_inches="tight")
    logger.info("wrote figure %s", out)
    return out


def histogram(
    values: np.ndarray,
    *,
    ax=None,
    bins: int = 60,
    title: str = "",
    xlabel: str = "",
    color: str = "#4e79a7",
):
    """Histogram of a raster/point attribute, NaNs excluded."""
    import matplotlib.pyplot as plt

    finite = np.asarray(values)[np.isfinite(values)]
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3.2))
    ax.hist(finite, bins=bins, color=color, edgecolor="none")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("cells")
    ax.grid(alpha=0.25)
    return ax
