"""Local (in-process) VLM/LLM: model choice, prompting and decoding.

Every test here runs against fakes; loading a real checkpoint is refused by the
autouse guard in conftest.py. The one real-model test is opt-in.
"""

from __future__ import annotations

import json
import os

import pytest

from rokko_geofusion.environment import EnvironmentInfo, GpuInfo
from rokko_geofusion.exceptions import ConfigurationRequiredError, UnsupportedError
from rokko_geofusion.local_model import (
    LocalGenerator,
    LocalModelChoice,
    accelerator_memory_gb,
    choose_local_model,
)
from rokko_geofusion.vlm.adapter import image_for_model, parse_json_response
from rokko_geofusion.vlm.inputs import VlmImage


def _env(*, vram_gb: float | None = None, mps: bool = False, ram_gb: float = 32.0):
    return EnvironmentInfo(
        python_version="3.12", python_executable="python", platform="test",
        os_name="Linux", machine="x86_64", cpu_model="cpu", cpu_count=8,
        ram_total_gb=ram_gb, ram_available_gb=ram_gb, in_colab=True, in_notebook=False,
        torch_version="2.x", cuda_available=vram_gb is not None, cuda_version=None,
        cudnn_version=None, mps_available=mps,
        gpus=(GpuInfo("GPU", vram_gb, "8.0"),) if vram_gb is not None else (),
        gdal_version=None, pdal_version=None, packages={},
    )


# --- model choice -------------------------------------------------------------
@pytest.mark.parametrize(
    ("vram_gb", "expected"),
    [
        (39.4, "Qwen/Qwen3-VL-8B-Instruct"),   # A100 40 GB
        (79.1, "Qwen/Qwen3-VL-8B-Instruct"),   # A100 80 GB
        (22.0, "Qwen/Qwen3-VL-4B-Instruct"),   # L4
        (14.7, "Qwen/Qwen3-VL-4B-Instruct"),   # T4
        (8.0, "Qwen/Qwen3-VL-2B-Instruct"),
    ],
)
def test_auto_picks_the_tier_the_gpu_can_hold(config, vram_gb, expected):
    choice = choose_local_model("auto", config, _env(vram_gb=vram_gb))
    assert choice.model_id == expected
    assert choice.device == "cuda"
    assert f"{vram_gb:.1f} GB" in choice.reason


def test_apple_silicon_counts_half_the_unified_memory(config):
    env = _env(mps=True, ram_gb=32.0)
    assert accelerator_memory_gb("mps", env) == pytest.approx(16.0)
    assert choose_local_model("auto", config, env).model_id == "Qwen/Qwen3-VL-4B-Instruct"


def test_cpu_falls_back_to_the_smallest_tier_and_warns(config, caplog):
    with caplog.at_level("WARNING"):
        choice = choose_local_model("auto", config, _env())
    assert choice.device == "cpu"
    assert choice.model_id == config.local_models.auto_tiers[-1].model_id
    assert "slow" in caplog.text


def test_an_explicit_model_id_is_kept(config):
    choice = choose_local_model("Qwen/Qwen3-8B", config, _env(vram_gb=80.0))
    assert choice.model_id == "Qwen/Qwen3-8B"
    assert choice.reason == "configured explicitly"


def test_tiers_come_from_configuration(config):
    config.local_models.auto_tiers = [
        {"min_memory_gb": 100.0, "model_id": "big/model"},
        {"min_memory_gb": 0.0, "model_id": "small/model"},
    ]
    assert choose_local_model("auto", config, _env(vram_gb=39.4)).model_id == "small/model"


def test_auto_is_rejected_for_api_providers(config):
    from rokko_geofusion.llm.adapter import load_llm
    from rokko_geofusion.vlm.adapter import load_vlm

    config.vlm.provider = "anthropic"
    config.llm.provider = "openai"
    with pytest.raises(ConfigurationRequiredError, match="vlm.model"):
        load_vlm(config)
    with pytest.raises(ConfigurationRequiredError, match="llm.model"):
        load_llm(config)


def test_local_is_the_default_provider(config):
    assert config.vlm.provider == config.llm.provider == "local"
    assert config.vlm.model == config.llm.model == "auto"


# --- generation -----------------------------------------------------------------
torch = pytest.importorskip("torch")


class _FakeInputs(dict):
    def to(self, device):
        return self


class _FakeProcessor:
    def __init__(self):
        self.messages = None

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        return _FakeInputs(input_ids=torch.tensor([[1, 2, 3]]))

    def batch_decode(self, tokens, skip_special_tokens=True):
        return [f"decoded {tokens.shape[1]} new tokens"]


class _FakeModel:
    device = "cpu"

    def __init__(self):
        self.options = None

    def generate(self, input_ids, **options):
        self.options = options
        return torch.tensor([[1, 2, 3, 9, 9]])


def _generator(*, multimodal: bool = True) -> LocalGenerator:
    generator = object.__new__(LocalGenerator)
    generator._torch = torch
    generator.choice = LocalModelChoice("fake/model", "cpu", "test")
    generator.dtype = "bfloat16"
    generator.multimodal = multimodal
    generator._processor = _FakeProcessor()
    generator._model = _FakeModel()
    return generator


def test_temperature_zero_is_really_greedy():
    generator = _generator()
    text = generator.generate(system="s", content=[{"type": "text", "text": "hi"}],
                              max_new_tokens=64, temperature=0.0)
    options = generator._model.options
    assert options["do_sample"] is False
    # The checkpoint ships temperature 0.7 / top_p 0.8 / top_k 20: all cleared.
    assert options["temperature"] is None and options["top_p"] is None
    assert options["top_k"] is None
    assert options["max_new_tokens"] == 64
    assert text == "decoded 2 new tokens"   # only the new tokens are decoded


def test_a_positive_temperature_samples():
    generator = _generator()
    generator.generate(system="s", content=[{"type": "text", "text": "hi"}],
                       max_new_tokens=8, temperature=0.4)
    assert generator._model.options["do_sample"] is True
    assert generator._model.options["temperature"] == 0.4


def test_multimodal_messages_keep_the_content_parts_in_order():
    generator = _generator()
    content = [{"type": "text", "text": "a"}, {"type": "image", "image": "img"},
               {"type": "text", "text": "b"}]
    generator.generate(system="sys", content=content, max_new_tokens=8)
    system, user = generator._processor.messages
    assert system == {"role": "system", "content": [{"type": "text", "text": "sys"}]}
    assert user == {"role": "user", "content": content}


def test_text_only_models_get_plain_strings():
    generator = _generator(multimodal=False)
    generator.generate(system="sys", content=[{"type": "text", "text": "a"},
                                              {"type": "text", "text": "b"}],
                       max_new_tokens=8)
    system, user = generator._processor.messages
    assert system == {"role": "system", "content": "sys"}
    assert user == {"role": "user", "content": "a\n\nb"}


def test_text_only_models_refuse_images():
    generator = _generator(multimodal=False)
    with pytest.raises(UnsupportedError, match="cannot read images"):
        generator.generate(system="s", content=[{"type": "image", "image": "x"}],
                           max_new_tokens=8)


def test_out_of_memory_becomes_actionable_advice():
    from rokko_geofusion.exceptions import ResourceError

    generator = _generator()

    def explode(**kwargs):
        raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")

    generator._model.generate = explode
    with pytest.raises(ResourceError, match="Qwen3-VL-2B-Instruct"):
        generator.generate(system="s", content=[{"type": "text", "text": "x"}],
                           max_new_tokens=8)


# --- adapters -------------------------------------------------------------------
class _ScriptedGenerator:
    def __init__(self, reply: str):
        self.reply = reply
        self.calls: list[dict] = []
        self.choice = LocalModelChoice("fake/vl-model", "cuda", "auto: test tier")

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return self.reply

    def describe(self):
        return {**self.choice.to_dict(), "dtype": "bfloat16", "multimodal": True}


def _png(path, size):
    from PIL import Image

    Image.new("RGB", size, (30, 120, 60)).save(path)
    return path


def test_local_vlm_sends_each_view_with_its_caption(config, tmp_path):
    from rokko_geofusion.vlm.adapter import SYSTEM_PROMPT, LocalVlm

    reply = json.dumps({"terrain_description": "a south-facing slope",
                        "land_cover_description": "forest above housing",
                        "built_environment_description": "a dense grid",
                        "notable_patterns": ["housing stops at the slope break"],
                        "uncertainties": []})
    generator = _ScriptedGenerator(reply)
    config.local_models.image_max_side_px = 256
    vlm = LocalVlm(config, generator=generator)
    images = [VlmImage("orthophoto", _png(tmp_path / "a.png", (1050, 800)), "aerial"),
              VlmImage("slope", _png(tmp_path / "b.png", (600, 1200)), "slope map")]

    analysis = vlm.analyze(images, question="Describe it.")

    call = generator.calls[0]
    assert call["system"] == SYSTEM_PROMPT
    assert call["temperature"] == config.vlm.temperature
    assert call["max_new_tokens"] == config.vlm.max_output_tokens
    kinds = [part["type"] for part in call["content"]]
    assert kinds == ["text", "image", "text", "image", "text"]
    assert call["content"][0]["text"] == "orthophoto: aerial"
    for part in call["content"]:
        if part["type"] == "image":
            assert max(part["image"].size) <= 256
    assert analysis.parsed and analysis.terrain_description == "a south-facing slope"
    assert analysis.provider == "local" and analysis.model == "fake/vl-model"
    assert analysis.runtime["reason"] == "auto: test tier"


def test_image_resizing_keeps_the_aspect_ratio_and_never_upscales(tmp_path):
    wide = image_for_model(_png(tmp_path / "w.png", (2000, 1000)), 1024)
    assert wide.size == (1024, 512)
    small = image_for_model(_png(tmp_path / "s.png", (300, 200)), 1024)
    assert small.size == (300, 200)
    assert wide.mode == "RGB"


def test_local_llm_report_records_which_model_ran(config):
    from rokko_geofusion.llm.adapter import LocalLlm
    from rokko_geofusion.llm.report import run_llm_analysis

    reply = json.dumps({"summary": "s", "measured": ["mean elevation 154.2 m"],
                        "observed": [], "inferred": [], "uncertain": [],
                        "recommended_checks": []})
    generator = _ScriptedGenerator(reply)
    llm = LocalLlm(config, generator=generator)
    report = run_llm_analysis(config, {"terrain": {}}, None, adapter=llm)

    call = generator.calls[0]
    assert call["content"][0]["type"] == "text"
    assert "MEASUREMENTS" in call["content"][0]["text"]
    assert report.parsed and report.provider == "local"
    assert report.runtime["model_id"] == "fake/vl-model"


def test_reasoning_blocks_are_ignored_when_parsing():
    text = '<think>maybe {"not": "this"}</think>\n{"terrain_description": "hilly"}'
    payload, reason = parse_json_response(text)
    assert reason == ""
    assert payload == {"terrain_description": "hilly"}


def test_real_model_loading_is_refused_in_unit_tests(config):
    with pytest.raises(RuntimeError, match="must not load a real model"):
        LocalGenerator(LocalModelChoice("any/model", "cpu", "test"))


# --- opt-in: a real checkpoint ------------------------------------------------------
@pytest.mark.local_model
@pytest.mark.skipif(not os.environ.get("RGF_TEST_LOCAL_MODEL"),
                    reason="set RGF_TEST_LOCAL_MODEL=<hf model id> to load a real model")
def test_a_real_local_model_answers_in_the_schema(config, tmp_path):
    """Run on the target machine (e.g. Colab) to check the model fits and behaves."""
    from rokko_geofusion.vlm.adapter import LocalVlm

    config.vlm.model = os.environ["RGF_TEST_LOCAL_MODEL"]
    config.vlm.max_output_tokens = 512
    vlm = LocalVlm(config)
    image = VlmImage("orthophoto", _png(tmp_path / "green.png", (512, 512)),
                     "A uniformly green test image, 512 x 512 px.")
    analysis = vlm.analyze([image], question="Describe this image.")
    assert analysis.parsed, analysis.raw_text
    assert analysis.land_cover_description or analysis.terrain_description


def test_out_of_memory_while_moving_inputs_is_also_caught():
    """Regression (Codex review): the transfer used to run outside the OOM handler."""
    from rokko_geofusion.exceptions import ResourceError

    generator = _generator()

    class _Oversized(_FakeInputs):
        def to(self, device):
            raise torch.OutOfMemoryError("CUDA out of memory while copying pixel_values")

    generator._processor.apply_chat_template = lambda messages, **kwargs: _Oversized()
    with pytest.raises(ResourceError, match="smaller checkpoint"):
        generator.generate(system="s", content=[{"type": "text", "text": "x"}],
                           max_new_tokens=8)


# --- stage scripts never leave a stale result ---------------------------------------
def _stage(repo_root, name, monkeypatch, **replacements):
    """Load a stage script with some of its imported names replaced."""
    from rokko_geofusion.pipeline import _load_script

    module = _load_script(name, repo_root)
    for attribute, value in replacements.items():
        monkeypatch.setattr(module, attribute, value)
    return module


def _stage_args(config, tmp_path):
    return ["--config", str(config.source_path),
            "--set", f"project.data_root={tmp_path / 'data'}",
            "--set", f"project.output_root={tmp_path / 'outputs'}",
            "--set", "roi.radius_m=40", "--set", "roi.name=stale"]


def _stale_config(config, tmp_path):
    from rokko_geofusion.config import load_config

    return load_config(config.source_path, overrides=[
        f"project.data_root={tmp_path / 'data'}",
        f"project.output_root={tmp_path / 'outputs'}",
        "roi.radius_m=40", "roi.name=stale",
    ])


def _with_a_view_and_an_old_result(config, tmp_path, name):
    import numpy as np

    from rokko_geofusion.crs import roi_from_config
    from rokko_geofusion.io.raster import write_grid_raster
    from rokko_geofusion.utils.metadata import write_json

    stale = _stale_config(config, tmp_path)
    stale.paths.ensure()
    grid = roi_from_config(stale).grid(stale.imagery.resolution_m)
    write_grid_raster(stale.paths.interim / "orthophoto.tif",
                      np.full((3, grid.height, grid.width), 90, np.uint8), grid, nodata=None)
    write_json(stale.paths.reports / name, {"status": "ok", "summary": "from last week"})
    return stale


def _status(path):
    return json.loads(path.read_text(encoding="utf-8"))["status"]


def test_an_unexpected_vlm_crash_does_not_leave_the_old_ok_result(config, tmp_path,
                                                                   monkeypatch, repo_root):
    stale = _with_a_view_and_an_old_result(config, tmp_path, "vlm_analysis.json")

    class _Crashes:
        def analyze(self, images, *, question):
            raise KeyError("something nobody anticipated")

    stage = _stage(repo_root, "vlm", monkeypatch, load_vlm=lambda config: _Crashes())
    with pytest.raises(KeyError):
        stage.main(_stage_args(config, tmp_path))
    assert _status(stale.paths.reports / "vlm_analysis.json") == "incomplete"


def test_a_handled_vlm_failure_is_recorded_as_failed(config, tmp_path, monkeypatch, repo_root):
    from rokko_geofusion.exceptions import ResourceError

    stale = _with_a_view_and_an_old_result(config, tmp_path, "vlm_analysis.json")

    def out_of_memory(config):
        raise ResourceError("does not fit")

    stage = _stage(repo_root, "vlm", monkeypatch, load_vlm=out_of_memory)
    assert stage.main(_stage_args(config, tmp_path)) == 1
    result = json.loads((stale.paths.reports / "vlm_analysis.json").read_text())
    assert result == {"status": "failed", "reason": "does not fit"}


def test_render_only_leaves_the_existing_result_alone(config, tmp_path, monkeypatch, repo_root):
    stale = _with_a_view_and_an_old_result(config, tmp_path, "vlm_analysis.json")
    stage = _stage(repo_root, "vlm", monkeypatch)
    assert stage.main([*_stage_args(config, tmp_path), "--render-only"]) == 0
    assert _status(stale.paths.reports / "vlm_analysis.json") == "ok"


@pytest.mark.parametrize(
    ("failure", "expected_status", "expected_exit"),
    [("resource", "failed", 1), ("unexpected", "incomplete", None)],
)
def test_a_failed_llm_rerun_removes_the_old_report(config, tmp_path, monkeypatch, repo_root,
                                                    failure, expected_status, expected_exit):
    """Regression (Codex review): the old Markdown survived a failed rerun."""
    from rokko_geofusion.exceptions import ResourceError

    stale = _with_a_view_and_an_old_result(config, tmp_path, "geoai_report.json")
    markdown = stale.paths.reports / "geoai_report.md"
    markdown.write_text("# GeoAI analysis - from last week", encoding="utf-8")

    def fails(*args, **kwargs):
        raise ResourceError("does not fit") if failure == "resource" else ValueError("boom")

    stage = _stage(repo_root, "llm", monkeypatch, run_llm_analysis=fails)
    if expected_exit is None:
        with pytest.raises(ValueError):
            stage.main(_stage_args(config, tmp_path))
    else:
        assert stage.main(_stage_args(config, tmp_path)) == expected_exit
    assert _status(stale.paths.reports / "geoai_report.json") == expected_status
    assert not markdown.exists()


def test_an_unparsed_llm_answer_writes_no_markdown(config, tmp_path, monkeypatch, repo_root):
    from rokko_geofusion.llm.report import GeoAiReport

    stale = _with_a_view_and_an_old_result(config, tmp_path, "geoai_report.json")
    unparsed = GeoAiReport(raw_text="not json", parsed=False, provider="local", model="m")
    stage = _stage(repo_root, "llm", monkeypatch,
                   run_llm_analysis=lambda *args, **kwargs: unparsed)
    assert stage.main(_stage_args(config, tmp_path)) == 0
    assert _status(stale.paths.reports / "geoai_report.json") == "unparsed"
    assert not (stale.paths.reports / "geoai_report.md").exists()


@pytest.mark.parametrize(("status", "exposed"), [("ok", True), ("failed", False),
                                                  ("incomplete", False)])
def test_analyze_roi_exposes_the_report_only_after_a_successful_run(config, tmp_path,
                                                                    status, exposed):
    """Regression (Codex review): any leftover Markdown was exposed as the report."""
    from rokko_geofusion.interactive import analyze_roi
    from rokko_geofusion.utils.metadata import write_json

    stale = _stale_config(config, tmp_path)
    stale.paths.ensure()
    (stale.paths.reports / "geoai_report.md").write_text("# report", encoding="utf-8")
    write_json(stale.paths.reports / "geoai_report.json", {"status": status})

    analysis = analyze_roi(config_path=config.source_path, run=False, overrides=[
        f"project.data_root={tmp_path / 'data'}",
        f"project.output_root={tmp_path / 'outputs'}",
        "roi.radius_m=40", "roi.name=stale",
    ])
    assert ("report" in analysis.products) is exposed
