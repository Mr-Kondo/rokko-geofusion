"""Runtime inspection and resource-aware parameter selection.

Nothing in this project may assume a particular GPU (the project brief
explicitly forbids "A100 or bust" code). Instead we detect what is available
at run time and derive a :class:`ResourceProfile` -- tile sizes, batch sizes
and point budgets -- from it, with a documented degradation path when CUDA
runs out of memory.
"""

from __future__ import annotations

import importlib
import logging
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field, replace
from typing import Any

logger = logging.getLogger(__name__)

#: Packages whose versions we report in `check_environment.py`.
TRACKED_PACKAGES = (
    "numpy",
    "scipy",
    "pandas",
    "pydantic",
    "pyproj",
    "shapely",
    "rasterio",
    "geopandas",
    "pyogrio",
    "pyarrow",
    "laspy",
    "torch",
    "transformers",
    "sklearn",
    "matplotlib",
    "plotly",
    "folium",
    "PIL",
    "requests",
)

_GB = 1024**3


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------
def _package_version(name: str) -> str | None:
    try:
        module = importlib.import_module(name)
    except Exception:  # noqa: BLE001 - a broken optional dep must not crash us
        return None
    return str(getattr(module, "__version__", "unknown"))


def _total_ram_bytes() -> int | None:
    try:
        import psutil

        return int(psutil.virtual_memory().total)
    except ImportError:
        pass
    try:  # Linux / most POSIX
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        pass
    if sys.platform == "darwin" and shutil.which("sysctl"):
        try:
            out = subprocess.run(
                ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=5
            )
            return int(out.stdout.strip())
        except (subprocess.SubprocessError, ValueError):
            return None
    return None


def _available_ram_bytes() -> int | None:
    try:
        import psutil

        return int(psutil.virtual_memory().available)
    except ImportError:
        return None


def _cpu_model() -> str:
    if sys.platform == "darwin" and shutil.which("sysctl"):
        try:
            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if out.stdout.strip():
                return out.stdout.strip()
        except subprocess.SubprocessError:
            pass
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def _gdal_version() -> str | None:
    try:
        import rasterio

        return str(rasterio.__gdal_version__)
    except Exception:  # noqa: BLE001
        pass
    if shutil.which("gdalinfo"):
        try:
            out = subprocess.run(
                ["gdalinfo", "--version"], capture_output=True, text=True, timeout=10
            )
            return out.stdout.strip() or None
        except subprocess.SubprocessError:
            return None
    return None


def _pdal_version() -> str | None:
    """PDAL is optional: LAZ IO goes through laspy/lazrs, not PDAL."""
    if shutil.which("pdal"):
        try:
            out = subprocess.run(
                ["pdal", "--version"], capture_output=True, text=True, timeout=10
            )
            for line in out.stdout.splitlines():
                if "pdal" in line.lower():
                    return line.strip()
        except subprocess.SubprocessError:
            return None
    return None


def _in_colab() -> bool:
    return "google.colab" in sys.modules or bool(os.environ.get("COLAB_RELEASE_TAG"))


def _in_notebook() -> bool:
    try:
        shell = get_ipython().__class__.__name__  # type: ignore[name-defined]  # noqa: F821
    except NameError:
        return False
    return shell in {"ZMQInteractiveShell", "Shell"}


@dataclass(frozen=True)
class GpuInfo:
    name: str
    total_vram_gb: float
    capability: str | None = None


@dataclass(frozen=True)
class EnvironmentInfo:
    """A snapshot of the machine the pipeline is running on."""

    python_version: str
    python_executable: str
    platform: str
    os_name: str
    machine: str
    cpu_model: str
    cpu_count: int
    ram_total_gb: float | None
    ram_available_gb: float | None
    in_colab: bool
    in_notebook: bool
    torch_version: str | None
    cuda_available: bool
    cuda_version: str | None
    cudnn_version: str | None
    mps_available: bool
    gpus: tuple[GpuInfo, ...]
    gdal_version: str | None
    pdal_version: str | None
    packages: dict[str, str | None] = field(default_factory=dict)

    @property
    def total_vram_gb(self) -> float:
        return sum(gpu.total_vram_gb for gpu in self.gpus)

    @property
    def primary_gpu(self) -> GpuInfo | None:
        return self.gpus[0] if self.gpus else None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["gpus"] = [asdict(gpu) for gpu in self.gpus]
        return data


def detect_environment(*, probe_packages: bool = True) -> EnvironmentInfo:
    """Collect everything ``scripts/check_environment.py`` prints."""
    torch_version: str | None = None
    cuda_available = False
    cuda_version: str | None = None
    cudnn_version: str | None = None
    mps_available = False
    gpus: list[GpuInfo] = []

    try:
        import torch
    except ImportError:
        logger.debug("torch is not installed; GPU features unavailable")
    else:
        torch_version = torch.__version__
        cuda_available = bool(torch.cuda.is_available())
        cuda_version = getattr(torch.version, "cuda", None)
        try:
            raw_cudnn = torch.backends.cudnn.version() if cuda_available else None
            cudnn_version = str(raw_cudnn) if raw_cudnn else None
        except Exception:  # noqa: BLE001
            cudnn_version = None
        mps_available = bool(
            getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        )
        if cuda_available:
            for index in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(index)
                gpus.append(
                    GpuInfo(
                        name=props.name,
                        total_vram_gb=round(props.total_memory / _GB, 2),
                        capability=f"{props.major}.{props.minor}",
                    )
                )

    ram_total = _total_ram_bytes()
    ram_available = _available_ram_bytes()

    return EnvironmentInfo(
        python_version=platform.python_version(),
        python_executable=sys.executable,
        platform=platform.platform(),
        os_name=platform.system(),
        machine=platform.machine(),
        cpu_model=_cpu_model(),
        cpu_count=os.cpu_count() or 1,
        ram_total_gb=round(ram_total / _GB, 2) if ram_total else None,
        ram_available_gb=round(ram_available / _GB, 2) if ram_available else None,
        in_colab=_in_colab(),
        in_notebook=_in_notebook(),
        torch_version=torch_version,
        cuda_available=cuda_available,
        cuda_version=cuda_version,
        cudnn_version=cudnn_version,
        mps_available=mps_available,
        gpus=tuple(gpus),
        gdal_version=_gdal_version(),
        pdal_version=_pdal_version(),
        packages=(
            {name: _package_version(name) for name in TRACKED_PACKAGES}
            if probe_packages
            else {}
        ),
    )


def select_device(preference: str = "auto", env: EnvironmentInfo | None = None) -> str:
    """Resolve ``auto|cuda|mps|cpu`` against what is actually available."""
    env = env or detect_environment(probe_packages=False)
    preference = (preference or "auto").lower()

    if preference == "cuda":
        if env.cuda_available:
            return "cuda"
        logger.warning("device 'cuda' requested but CUDA is unavailable; using CPU")
        return "cpu"
    if preference == "mps":
        if env.mps_available:
            return "mps"
        logger.warning("device 'mps' requested but MPS is unavailable; using CPU")
        return "cpu"
    if preference == "cpu":
        return "cpu"

    if env.cuda_available:
        return "cuda"
    if env.mps_available:
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# Resource-aware parameters
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ResourceProfile:
    """Run-time parameters derived from the detected hardware.

    ``downscale()`` implements the documented OOM ladder:
    batch size -> tile size -> point count -> CPU.
    """

    device: str
    tier: str
    seg_tile_px: int
    seg_batch_size: int
    pc_num_points: int
    pc_batch_size: int
    max_points_in_memory: int
    http_max_workers: int
    display_points: int
    downscale_steps: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def downscale(self, *, min_tile_px: int = 256) -> ResourceProfile:
        """Return the next, smaller profile after an out-of-memory failure."""
        if self.seg_batch_size > 1:
            return replace(
                self,
                seg_batch_size=max(1, self.seg_batch_size // 2),
                pc_batch_size=max(1, self.pc_batch_size // 2),
                downscale_steps=self.downscale_steps + 1,
            )
        if self.seg_tile_px > min_tile_px:
            return replace(
                self,
                seg_tile_px=max(min_tile_px, self.seg_tile_px // 2),
                downscale_steps=self.downscale_steps + 1,
            )
        if self.pc_num_points > 1024:
            return replace(
                self,
                pc_num_points=max(1024, self.pc_num_points // 2),
                downscale_steps=self.downscale_steps + 1,
            )
        if self.device != "cpu":
            logger.warning("exhausted GPU downscaling options; falling back to CPU")
            return replace(self, device="cpu", tier="cpu-fallback",
                           downscale_steps=self.downscale_steps + 1)
        return self


def _tier_for(env: EnvironmentInfo, device: str) -> str:
    if device == "cpu":
        return "cpu"
    if device == "mps":
        return "mps"
    vram = env.total_vram_gb
    if vram >= 30:
        return "gpu-xl"
    if vram >= 14:
        return "gpu-l"
    if vram >= 9:
        return "gpu-m"
    return "gpu-s"


_TIER_TABLE: dict[str, dict[str, int]] = {
    # tier      seg_tile  seg_batch  pc_points  pc_batch
    "gpu-xl": {"seg_tile_px": 1024, "seg_batch_size": 16, "pc_num_points": 8192, "pc_batch_size": 32},
    "gpu-l": {"seg_tile_px": 768, "seg_batch_size": 8, "pc_num_points": 4096, "pc_batch_size": 16},
    "gpu-m": {"seg_tile_px": 512, "seg_batch_size": 4, "pc_num_points": 4096, "pc_batch_size": 8},
    "gpu-s": {"seg_tile_px": 512, "seg_batch_size": 2, "pc_num_points": 2048, "pc_batch_size": 4},
    "mps": {"seg_tile_px": 512, "seg_batch_size": 2, "pc_num_points": 2048, "pc_batch_size": 4},
    "cpu": {"seg_tile_px": 384, "seg_batch_size": 1, "pc_num_points": 1024, "pc_batch_size": 2},
}


def make_resource_profile(
    config: Any | None = None,
    env: EnvironmentInfo | None = None,
) -> ResourceProfile:
    """Derive tile/batch/point budgets from hardware, capped by ``config``.

    ``config`` is a :class:`~rokko_geofusion.config.Config`; it is typed as
    ``Any`` to keep this module importable without pydantic models loaded.
    """
    env = env or detect_environment(probe_packages=False)
    preference = getattr(getattr(config, "runtime", None), "device", "auto")
    device = select_device(preference, env)
    tier = _tier_for(env, device)
    table = _TIER_TABLE[tier]

    seg_tile_px = table["seg_tile_px"]
    seg_batch_size = table["seg_batch_size"]
    pc_num_points = table["pc_num_points"]
    pc_batch_size = table["pc_batch_size"]

    # Configuration acts as an explicit ceiling, never as a floor: the user
    # may ask for less than the hardware allows, but asking for more than the
    # detected tier would re-introduce the "assume a big GPU" failure mode.
    if config is not None:
        seg = getattr(config, "segmentation", None)
        if seg is not None:
            seg_tile_px = min(seg_tile_px, int(seg.tile_px))
            seg_batch_size = min(seg_batch_size, int(seg.batch_size))
        pcml = getattr(config, "pointcloud_ml", None)
        if pcml is not None:
            pc_num_points = min(pc_num_points, int(pcml.num_points))
            pc_batch_size = min(pc_batch_size, int(pcml.batch_size))

    ram_gb = env.ram_available_gb or env.ram_total_gb or 8.0
    # ~24 bytes/point for float32 XYZ plus overhead; keep to a third of RAM.
    ram_point_budget = int((ram_gb * _GB) / 3 / 24)
    max_points = ram_point_budget
    if config is not None and getattr(config, "processing", None) is not None:
        max_points = min(max_points, int(config.processing.max_points_in_memory))

    http_workers = 8
    display_points = 300_000
    if config is not None:
        if getattr(config, "processing", None) is not None:
            http_workers = int(config.processing.http.max_workers)
        if getattr(config, "visualization", None) is not None:
            display_points = int(config.visualization.max_display_points)

    profile = ResourceProfile(
        device=device,
        tier=tier,
        seg_tile_px=seg_tile_px,
        seg_batch_size=seg_batch_size,
        pc_num_points=pc_num_points,
        pc_batch_size=pc_batch_size,
        max_points_in_memory=max(1_000_000, max_points),
        http_max_workers=max(1, http_workers),
        display_points=display_points,
    )
    logger.info(
        "resource profile: device=%s tier=%s seg_tile=%dpx seg_batch=%d "
        "pc_points=%d pc_batch=%d max_points_in_memory=%s",
        profile.device,
        profile.tier,
        profile.seg_tile_px,
        profile.seg_batch_size,
        profile.pc_num_points,
        profile.pc_batch_size,
        f"{profile.max_points_in_memory:,}",
    )
    return profile
