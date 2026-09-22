"""Exception hierarchy for rokko-geofusion.

The point of having named exceptions here is traceability: when a stage fails
we want the log to say *what kind* of failure it was, so that "the upstream
service changed its URL scheme" never gets silently swallowed as "file not
found" (see CLAUDE.md, rule: do not guess external specifications).
"""

from __future__ import annotations


class GeoFusionError(Exception):
    """Base class for every error raised by this package."""


class ConfigError(GeoFusionError):
    """The configuration file is missing, malformed or internally inconsistent."""


class ConfigurationRequiredError(ConfigError):
    """A required external resource is not configured.

    Raised instead of inventing a plausible-looking URL/path. The message must
    tell the user exactly which configuration key to set.
    """

    def __init__(self, key: str, what: str, hint: str = "") -> None:
        msg = f"configuration required: `{key}` is not set ({what})."
        if hint:
            msg += f" {hint}"
        super().__init__(msg)
        self.key = key


class UnsupportedError(GeoFusionError):
    """A code path exists but is deliberately not implemented for this input."""


class DataSourceError(GeoFusionError):
    """A remote data source failed in a way that is not a simple missing file.

    Use this for HTTP errors, schema changes and unexpected payloads so that an
    upstream API change is never reported as "no data for this ROI".
    """


class DataUnavailableError(GeoFusionError):
    """The source works but genuinely has no data for the requested ROI."""


class CrsError(GeoFusionError):
    """A coordinate reference system is missing, ambiguous or inconsistent."""


class AlignmentError(GeoFusionError):
    """Two rasters / a raster and a point cloud do not share a common grid."""


class ResourceError(GeoFusionError):
    """Not enough GPU/CPU memory even after the documented fallbacks."""


class StageError(GeoFusionError):
    """A pipeline stage failed; carries the stage name for the CLI."""

    def __init__(self, stage: str, message: str) -> None:
        super().__init__(f"stage '{stage}' failed: {message}")
        self.stage = stage
