"""Runtime detection and the resource-degradation ladder."""

from __future__ import annotations

from dataclasses import replace

import pytest

from rokko_geofusion.environment import (
    EnvironmentInfo,
    GpuInfo,
    ResourceProfile,
    detect_environment,
    make_resource_profile,
    select_device,
)


def _fake_env(**kwargs) -> EnvironmentInfo:
    base = dict(
        python_version="3.12.0",
        python_executable="/usr/bin/python",
        platform="test",
        os_name="Linux",
        machine="x86_64",
        cpu_model="test cpu",
        cpu_count=8,
        ram_total_gb=32.0,
        ram_available_gb=24.0,
        in_colab=False,
        in_notebook=False,
        torch_version="2.3.0",
        cuda_available=False,
        cuda_version=None,
        cudnn_version=None,
        mps_available=False,
        gpus=(),
        gdal_version="3.8.0",
        pdal_version=None,
        packages={},
    )
    base.update(kwargs)
    return EnvironmentInfo(**base)


def test_detect_environment_reports_basics():
    env = detect_environment()
    assert env.python_version
    assert env.cpu_count >= 1
    assert isinstance(env.in_colab, bool)
    assert "numpy" in env.packages


def test_select_device_prefers_cuda_when_available():
    env = _fake_env(cuda_available=True, gpus=(GpuInfo("Fake GPU", 16.0, "8.6"),))
    assert select_device("auto", env) == "cuda"
    assert select_device("cpu", env) == "cpu"


def test_select_device_falls_back_when_cuda_missing():
    env = _fake_env(cuda_available=False)
    assert select_device("cuda", env) == "cpu"
    assert select_device("auto", env) == "cpu"


def test_select_device_uses_mps_when_only_mps():
    env = _fake_env(mps_available=True)
    assert select_device("auto", env) == "mps"


@pytest.mark.parametrize(
    ("vram_gb", "expected_tier"),
    [(40.0, "gpu-xl"), (16.0, "gpu-l"), (11.0, "gpu-m"), (6.0, "gpu-s")],
)
def test_profile_tiers_scale_with_vram(vram_gb, expected_tier):
    env = _fake_env(cuda_available=True, gpus=(GpuInfo("Fake", vram_gb, "8.0"),))
    profile = make_resource_profile(None, env)
    assert profile.tier == expected_tier
    assert profile.device == "cuda"


def test_profile_respects_config_ceiling(config):
    env = _fake_env(cuda_available=True, gpus=(GpuInfo("Fake", 80.0, "9.0"),))
    profile = make_resource_profile(config, env)
    # config.segmentation.batch_size is 4, the gpu-xl tier would allow 16.
    assert profile.seg_batch_size <= config.segmentation.batch_size
    assert profile.seg_tile_px <= config.segmentation.tile_px


def test_cpu_profile_is_small():
    profile = make_resource_profile(None, _fake_env())
    assert profile.device == "cpu"
    assert profile.seg_batch_size == 1
    assert profile.pc_num_points <= 1024


def test_downscale_ladder_ends_at_cpu():
    profile = ResourceProfile(
        device="cuda",
        tier="gpu-l",
        seg_tile_px=768,
        seg_batch_size=8,
        pc_num_points=4096,
        pc_batch_size=16,
        max_points_in_memory=10_000_000,
        http_max_workers=8,
        display_points=100_000,
    )
    seen = [profile]
    for _ in range(20):
        nxt = seen[-1].downscale()
        if nxt == seen[-1]:
            break
        seen.append(nxt)
    assert seen[-1].device == "cpu"
    # batch size shrinks before tile size, tile size before point count.
    assert seen[1].seg_batch_size < profile.seg_batch_size
    assert seen[1].seg_tile_px == profile.seg_tile_px
    assert any(p.seg_tile_px < profile.seg_tile_px for p in seen)
    assert seen[-1].downscale_steps > 0


def test_downscale_is_pure():
    profile = make_resource_profile(None, _fake_env())
    before = replace(profile)
    profile.downscale()
    assert profile == before
