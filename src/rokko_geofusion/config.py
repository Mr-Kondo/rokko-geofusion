"""Typed configuration for the whole pipeline.

Everything that could reasonably differ between two runs -- the ROI, the CRS
pair, tile zoom levels, thresholds, model ids -- lives in a YAML file and is
validated here. No EPSG code, URL template or threshold is hard-coded in the
processing modules; they all read from :class:`Config`.

Usage::

    from rokko_geofusion.config import load_config
    cfg = load_config("configs/rokko.yaml", overrides=["roi.radius_m=500"])
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from rokko_geofusion.exceptions import ConfigError
from rokko_geofusion.paths import Paths, find_project_root, resolve_path

logger = logging.getLogger(__name__)

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


class _Base(BaseModel):
    """Forbid unknown keys so that a typo in YAML fails loudly."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


# ---------------------------------------------------------------------------
# ROI / CRS
# ---------------------------------------------------------------------------
class CenterConfig(_Base):
    lat: float = Field(..., ge=-90.0, le=90.0)
    lon: float = Field(..., ge=-180.0, le=180.0)


class RoiConfig(_Base):
    """Region of interest.

    Either a ``center`` + ``radius_m`` (a square of side ``2*radius_m``
    centred on the point, in the projected CRS) or an explicit geographic
    ``bbox`` of ``[min_lon, min_lat, max_lon, max_lat]``.
    """

    name: str = "roi"
    center: CenterConfig | None = None
    radius_m: float = Field(1000.0, gt=0.0)
    bbox: list[float] | None = None

    @field_validator("bbox")
    @classmethod
    def _check_bbox(cls, value: list[float] | None) -> list[float] | None:
        if value is None:
            return None
        if len(value) != 4:
            raise ValueError("bbox must be [min_lon, min_lat, max_lon, max_lat]")
        min_lon, min_lat, max_lon, max_lat = value
        if not (min_lon < max_lon and min_lat < max_lat):
            raise ValueError(f"degenerate bbox: {value}")
        return [float(v) for v in value]

    @model_validator(mode="after")
    def _need_one(self) -> RoiConfig:
        if self.center is None and self.bbox is None:
            raise ValueError("roi requires either `center` (+ radius_m) or `bbox`")
        return self

    @property
    def key(self) -> str:
        """Deterministic, filesystem-safe identifier used for output folders."""
        safe = re.sub(r"[^0-9A-Za-z_.-]+", "_", self.name).strip("_") or "roi"
        if self.center is not None:
            return f"{safe}_{self.center.lat:.5f}_{self.center.lon:.5f}_r{self.radius_m:g}"
        assert self.bbox is not None
        return f"{safe}_bbox_" + "_".join(f"{v:.5f}" for v in self.bbox)


class CrsConfig(_Base):
    """The three CRS roles the project keeps explicit at all times."""

    geographic: str = "EPSG:4326"
    projected: str = "EPSG:6673"
    #: CRS of the XYZ web tiles that GSI (and most tile servers) publish.
    tile: str = "EPSG:3857"
    #: CRS written to output files; defaults to ``projected`` when omitted.
    output: str | None = None
    #: Common lattice step (metres) every derived grid snaps to. Keep it a
    #: multiple of every resolution in use so that grids of different
    #: resolutions nest exactly -- otherwise a coarse grid's edge cells fall
    #: outside a finer product and sample nothing.
    grid_snap_m: float = Field(10.0, gt=0.0)

    @property
    def effective_output(self) -> str:
        return self.output or self.projected


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------
class HttpConfig(_Base):
    timeout_s: float = 60.0
    max_retries: int = 3
    backoff_s: float = 1.5
    user_agent: str = "rokko-geofusion/0.1 (research use; https://github.com/)"
    cache_enabled: bool = True
    max_workers: int = 8
    #: Politeness delay between requests to the same host, in seconds.
    min_interval_s: float = 0.0


# ---------------------------------------------------------------------------
# LiDAR / DSM / DEM
# ---------------------------------------------------------------------------
class GsiTileSourceConfig(_Base):
    """GSI elevation tiles (https://maps.gsi.go.jp/development/ichiran.html).

    Values are plain-text CSV grids of 256x256 orthometric heights per tile,
    published on the Web-Mercator XYZ grid. ``dem5a`` (5 m mesh, z15) is the
    most accurate nationwide product; ``dem`` (DEM10B, 10 m mesh, z14) is the
    fallback where 5 m coverage is missing.
    """

    base_url: str = "https://cyberjapandata.gsi.go.jp/xyz"
    dataset: Literal["dem5a", "dem5b", "dem5c", "dem"] = "dem5a"
    zoom: int | None = None  # None -> use the documented default for `dataset`
    fallback_datasets: list[Literal["dem5a", "dem5b", "dem5c", "dem"]] = Field(
        default_factory=lambda: ["dem"]
    )
    nodata_token: str = "e"
    attribution: str = "国土地理院 (Geospatial Information Authority of Japan)"


class LocalFileSourceConfig(_Base):
    """A file the user placed under ``data/raw`` themselves."""

    path: str | None = None
    #: Explicit CRS for formats that carry none (e.g. plain XYZ text).
    crs: str | None = None
    #: Grid spacing for regular-grid XYZ files, in metres.
    grid_spacing_m: float | None = None


class CkanSourceConfig(_Base):
    """Discovery against a CKAN catalogue (e.g. G-Spatial Information Center).

    We deliberately do *not* hard-code a dataset URL: the catalogue is searched
    with :mod:`rokko_geofusion.io.ckan` and the chosen resource URL is pinned
    here by the user. See README "Data sources".
    """

    catalog_url: str = "https://www.geospatial.jp/ckan"
    search_query: str | None = None
    dataset_id: str | None = None
    resource_url: str | None = None
    format: str | None = None


ElevationProvider = Literal["gsi_tile", "local_raster", "local_xyz", "ckan", "none"]


class ElevationSourceConfig(_Base):
    provider: ElevationProvider = "none"
    gsi: GsiTileSourceConfig = Field(default_factory=GsiTileSourceConfig)
    local: LocalFileSourceConfig = Field(default_factory=LocalFileSourceConfig)
    ckan: CkanSourceConfig = Field(default_factory=CkanSourceConfig)


class LidarConfig(_Base):
    """DEM/DSM acquisition and the derived point cloud.

    ``is_true_lidar`` records whether the elevation source is a real LiDAR
    point cloud or a regular gridded product. The distinction is written into
    every metadata sidecar (project requirement: never present a regular XYZ
    grid as raw LiDAR).
    """

    dem: ElevationSourceConfig = Field(
        default_factory=lambda: ElevationSourceConfig(provider="gsi_tile")
    )
    dsm: ElevationSourceConfig = Field(default_factory=ElevationSourceConfig)
    resolution_m: float = Field(5.0, gt=0.0)
    dtype: Literal["float32", "float64"] = "float32"
    #: Resampling used when regridding tiles into the projected CRS.
    resampling: Literal["nearest", "bilinear", "cubic"] = "bilinear"
    nodata: float = -9999.0
    is_true_lidar: bool = False
    #: Safety limit on the number of elevation tiles one ROI may request.
    max_tiles: int = 2048


# ---------------------------------------------------------------------------
# Imagery
# ---------------------------------------------------------------------------
class ImageryConfig(_Base):
    provider: Literal["gsi_tile", "local_raster", "none"] = "gsi_tile"
    base_url: str = "https://cyberjapandata.gsi.go.jp/xyz"
    #: ``seamlessphoto`` = seamless nationwide photo mosaic (colour, z2-z18);
    #: ``ort`` = orthorectified aerial photographs (coverage/vintage varies).
    dataset: Literal["seamlessphoto", "ort"] = "seamlessphoto"
    extension: Literal["jpg", "png"] = "jpg"
    zoom: int = Field(18, ge=1, le=21)
    #: Target grid spacing in the projected CRS. Keep it close to the native
    #: ground sample distance of `zoom` (z18 ~ 0.49 m at 35N) -- a much finer
    #: value only interpolates, it does not add detail.
    resolution_m: float = Field(0.5, gt=0.0)
    max_tiles: int = 4096
    local: LocalFileSourceConfig = Field(default_factory=LocalFileSourceConfig)
    attribution: str = "国土地理院 (Geospatial Information Authority of Japan)"


# ---------------------------------------------------------------------------
# GIS
# ---------------------------------------------------------------------------
class OsmConfig(_Base):
    endpoint: str = "https://overpass-api.de/api/interpreter"
    mirrors: list[str] = Field(
        default_factory=lambda: [
            "https://lz4.overpass-api.de/api/interpreter",
            "https://z.overpass-api.de/api/interpreter",
        ]
    )
    timeout_s: float = 180.0
    max_retries: int = 3
    #: Overpass hands out query slots on a ~60 s cycle; retrying sooner just
    #: burns the remaining slots, so this is much larger than the HTTP default.
    backoff_s: float = 30.0
    #: Overpass rejects requests without a descriptive User-Agent (HTTP 406).
    user_agent: str = "rokko-geofusion/0.1 (academic research; contact: repository issues)"
    layers: list[Literal["building", "road", "water", "landuse", "railway"]] = Field(
        default_factory=lambda: ["building", "road", "water", "landuse"]
    )
    attribution: str = "(c) OpenStreetMap contributors, ODbL"


class GisConfig(_Base):
    providers: list[Literal["osm", "local"]] = Field(default_factory=lambda: ["osm"])
    osm: OsmConfig = Field(default_factory=OsmConfig)
    local_dir: str | None = None
    #: Building height attribute candidates, in priority order.
    building_height_keys: list[str] = Field(
        default_factory=lambda: ["height", "building:height", "building:levels"]
    )
    metres_per_level: float = 3.0


# ---------------------------------------------------------------------------
# Processing / terrain / segmentation / fusion / ML / AI
# ---------------------------------------------------------------------------
class PointCloudConfig(_Base):
    """How the XYZ(+RGB) cloud is derived from the elevation rasters.

    The cloud is a *derived* product: one point per cell of the target grid,
    taking Z from the surface raster (DSM when available, otherwise DEM) and
    RGB from the orthophoto. It is therefore a gridded cloud, not raw LiDAR --
    ``is_true_lidar`` in the metadata always says which.
    """

    #: ``None`` -> use the native resolution of the elevation raster. A finer
    #: value interpolates; it does not add measured detail.
    resolution_m: float | None = None
    surface: Literal["auto", "dsm", "dem"] = "auto"
    colorize: bool = True
    #: Tile edge used when generating and colourising, in metres.
    chunk_size_m: float = Field(500.0, gt=0.0)
    rgb_sampling: Literal["nearest", "bilinear"] = "nearest"
    #: Refuse to generate more points than this (guards against a fine
    #: resolution on a large ROI).
    max_points: int = 40_000_000


class ProcessingConfig(_Base):
    voxel_size_m: float = Field(1.0, gt=0.0)
    tile_size_m: float = Field(250.0, gt=0.0)
    dtype: Literal["float32", "float64"] = "float32"
    #: Hard ceiling on points held in RAM at once; larger clouds are chunked.
    max_points_in_memory: int = 20_000_000
    http: HttpConfig = Field(default_factory=HttpConfig)


class TerrainConfig(_Base):
    slope_units: Literal["degrees", "percent"] = "degrees"
    #: Window for local relief (max-min) in cells.
    relief_window_cells: int = Field(5, ge=3)
    #: Heights below this are clamped to 0 when computing nDSM (noise floor).
    ndsm_clip_min_m: float = 0.0
    ndsm_clip_max_m: float | None = 150.0
    point_density_cell_m: float = Field(5.0, gt=0.0)


class SegmentationConfig(_Base):
    enabled: bool = True
    provider: Literal["segformer_ade", "none"] = "segformer_ade"
    #: Any HF semantic-segmentation checkpoint; ADE20K models are trained on
    #: ground-level photos, so aerial performance is a documented limitation.
    model_id: str = "nvidia/segformer-b0-finetuned-ade-512-512"
    tile_px: int = 512
    overlap_px: int = 64
    batch_size: int = 4
    #: Auto-shrink tile/batch on CUDA OOM before falling back to CPU.
    auto_downscale: bool = True
    min_tile_px: int = 256
    classes: list[str] = Field(
        default_factory=lambda: [
            "other",
            "building",
            "road",
            "vegetation",
            "bare_soil",
            "water",
        ]
    )


class FusionThresholds(_Base):
    building_min_height_m: float = 2.5
    tall_vegetation_min_height_m: float = 3.0
    road_max_height_m: float = 1.0
    #: Minimum image-class confidence for a semantic label to be trusted.
    min_image_confidence: float = 0.35


class FusionConfig(_Base):
    enabled: bool = True
    cell_size_m: float = Field(1.0, gt=0.0)
    thresholds: FusionThresholds = Field(default_factory=FusionThresholds)


class PointCloudMlConfig(_Base):
    enabled: bool = True
    model: Literal["pointnet"] = "pointnet"
    num_points: int = 4096
    tile_size_m: float = Field(50.0, gt=0.0)
    batch_size: int = 8
    epochs: int = 20
    learning_rate: float = 1e-3
    embedding_dim: int = 256
    n_clusters: int = 6
    #: Which feature sets to compare (project requirement A/B/C/D).
    feature_sets: list[Literal["xyz", "xyz_rgb", "xyz_rgb_terrain", "xyz_rgb_terrain_semantic"]] = (
        Field(default_factory=lambda: ["xyz", "xyz_rgb", "xyz_rgb_terrain",
                                       "xyz_rgb_terrain_semantic"])
    )


class VlmConfig(_Base):
    enabled: bool = True
    provider: Literal["anthropic", "openai", "hf_local", "none"] = "anthropic"
    model: str = "claude-opus-5"
    api_key_env: str = "ANTHROPIC_API_KEY"
    max_images: int = 6
    max_output_tokens: int = 2048
    temperature: float = 0.0


class LlmConfig(_Base):
    enabled: bool = True
    provider: Literal["anthropic", "openai", "hf_local", "none"] = "anthropic"
    model: str = "claude-opus-5"
    api_key_env: str = "ANTHROPIC_API_KEY"
    max_output_tokens: int = 4096
    temperature: float = 0.0
    language: Literal["ja", "en"] = "ja"


class VisualizationConfig(_Base):
    #: Points streamed to a browser widget; the analysis cloud stays on disk.
    #: A plotly HTML scatter costs ~120 bytes per point, so 100k points is
    #: already a ~12 MB output cell -- raise this only for small ROIs.
    max_display_points: int = 100_000
    figure_dpi: int = 150
    basemap: str = "https://cyberjapandata.gsi.go.jp/xyz/pale/{z}/{x}/{y}.png"
    basemap_attribution: str = "国土地理院"


class OutputConfig(_Base):
    pointcloud_format: Literal["laz", "las", "parquet", "npz"] = "laz"
    raster_dtype: Literal["float32", "float64"] = "float32"
    compress: str = "deflate"
    overwrite: bool = False
    write_metadata_sidecar: bool = True


class RuntimeConfig(_Base):
    device: Literal["auto", "cuda", "cpu", "mps"] = "auto"
    num_workers: int = 2
    allow_network: bool = True
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"


class ProjectConfig(_Base):
    name: str = "rokko_geofusion"
    seed: int = 42
    data_root: str = "data"
    output_root: str = "outputs"


# ---------------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------------
class Config(_Base):
    project: ProjectConfig = Field(default_factory=ProjectConfig)
    roi: RoiConfig
    crs: CrsConfig = Field(default_factory=CrsConfig)
    lidar: LidarConfig = Field(default_factory=LidarConfig)
    imagery: ImageryConfig = Field(default_factory=ImageryConfig)
    gis: GisConfig = Field(default_factory=GisConfig)
    pointcloud: PointCloudConfig = Field(default_factory=PointCloudConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    terrain: TerrainConfig = Field(default_factory=TerrainConfig)
    segmentation: SegmentationConfig = Field(default_factory=SegmentationConfig)
    fusion: FusionConfig = Field(default_factory=FusionConfig)
    pointcloud_ml: PointCloudMlConfig = Field(default_factory=PointCloudMlConfig)
    vlm: VlmConfig = Field(default_factory=VlmConfig)
    llm: LlmConfig = Field(default_factory=LlmConfig)
    visualization: VisualizationConfig = Field(default_factory=VisualizationConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)

    # Populated by `load_config`; excluded from serialisation.
    source_path: str | None = Field(default=None, exclude=True, repr=False)
    project_root: str | None = Field(default=None, exclude=True, repr=False)

    # -- derived -------------------------------------------------------------
    @property
    def root(self) -> Path:
        return Path(self.project_root) if self.project_root else find_project_root()

    @property
    def paths(self) -> Paths:
        root = self.root
        return Paths(
            root=root,
            data_root=resolve_path(self.project.data_root, root),
            output_root=resolve_path(self.project.output_root, root),
            roi_key=self.roi.key,
        )

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict(), allow_unicode=True, sort_keys=False)

    def save(self, path: Path | str) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(self.to_yaml(), encoding="utf-8")
        return out

    def fingerprint(self) -> str:
        """Stable hash of the effective configuration (for run provenance)."""
        import hashlib

        payload = json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _expand_env(node: Any) -> Any:
    """Expand ``${VAR}`` / ``${VAR:-default}`` inside string values."""
    if isinstance(node, str):

        def repl(match: re.Match[str]) -> str:
            name, default = match.group(1), match.group(2)
            value = os.environ.get(name)
            if value is None:
                if default is None:
                    logger.warning("environment variable %s is not set; substituting ''", name)
                    return ""
                return default
            return value

        return _ENV_PATTERN.sub(repl, node)
    if isinstance(node, dict):
        return {key: _expand_env(value) for key, value in node.items()}
    if isinstance(node, list):
        return [_expand_env(value) for value in node]
    return node


def _coerce_scalar(text: str) -> Any:
    """Parse a CLI override value using YAML scalar rules."""
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError:
        return text


def apply_overrides(data: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    """Apply ``dotted.key=value`` overrides onto a raw config dict."""
    for item in overrides or []:
        if "=" not in item:
            raise ConfigError(f"override must look like key.subkey=value, got: {item!r}")
        dotted, raw_value = item.split("=", 1)
        keys = dotted.strip().split(".")
        cursor: dict[str, Any] = data
        for key in keys[:-1]:
            nxt = cursor.get(key)
            if not isinstance(nxt, dict):
                nxt = {}
                cursor[key] = nxt
            cursor = nxt
        cursor[keys[-1]] = _coerce_scalar(raw_value.strip())
        logger.debug("config override applied: %s", item)
    return data


def load_config(
    path: Path | str,
    *,
    overrides: list[str] | None = None,
    project_root: Path | str | None = None,
) -> Config:
    """Read, override, env-expand and validate a YAML configuration file."""
    config_path = Path(path).expanduser()
    if not config_path.is_file():
        raise ConfigError(f"config file not found: {config_path}")

    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"could not parse YAML in {config_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"config root must be a mapping, got {type(raw).__name__}")

    raw = apply_overrides(raw, overrides)
    raw = _expand_env(raw)

    try:
        config = Config.model_validate(raw)
    except Exception as exc:  # pydantic ValidationError -> ConfigError
        raise ConfigError(f"invalid configuration in {config_path}:\n{exc}") from exc

    config.source_path = str(config_path.resolve())
    config.project_root = str(Path(project_root).resolve() if project_root
                              else find_project_root(config_path))
    logger.debug("loaded config %s (fingerprint %s)", config_path, config.fingerprint())
    return config
