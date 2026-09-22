"""rokko-geofusion: LiDAR x aerial imagery x GIS x GeoAI fusion.

Design rules that the rest of the package follows (see CLAUDE.md):

* processing logic lives here, never in a notebook;
* every dataset carries an explicit CRS from source to output;
* large point clouds are tiled/chunked, never loaded whole;
* external URLs and model ids come from configuration, never from guesses.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
