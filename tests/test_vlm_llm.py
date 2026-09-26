"""VLM and LLM adapters: prompts, parsing, and the no-fabrication contract."""

from __future__ import annotations

import json

import pytest

from rokko_geofusion.crs import CrsManager, RoiGeometry
from rokko_geofusion.exceptions import ConfigurationRequiredError, DataSourceError
from rokko_geofusion.io.raster import write_grid_raster
from rokko_geofusion.llm.adapter import load_llm
from rokko_geofusion.llm.report import SYSTEM_PROMPT as LLM_SYSTEM_PROMPT
from rokko_geofusion.llm.report import (
    GeoAiReport,
    build_prompt,
    run_llm_analysis,
)
from rokko_geofusion.report import build_payload
from rokko_geofusion.vlm.adapter import (
    RESPONSE_SCHEMA,
    SYSTEM_PROMPT,
    _to_analysis,
    build_user_prompt,
    load_vlm,
    parse_json_response,
)
from rokko_geofusion.vlm.inputs import VlmImage, render_vlm_inputs


# --- response parsing -------------------------------------------------------
def test_parse_plain_json():
    payload, reason = parse_json_response('{"a": 1}')
    assert payload == {"a": 1}
    assert reason == ""


def test_parse_fenced_json():
    payload, _ = parse_json_response("```json\n{\"a\": [1, 2]}\n```")
    assert payload == {"a": [1, 2]}


def test_parse_json_surrounded_by_prose():
    payload, _ = parse_json_response('Sure! {"terrain_description": "hilly"} Hope that helps.')
    assert payload["terrain_description"] == "hilly"


def test_parse_failure_is_reported_not_papered_over():
    payload, reason = parse_json_response("I cannot answer that.")
    assert payload is None
    assert "not valid JSON" in reason


def test_parse_rejects_a_json_array():
    payload, reason = parse_json_response("[1, 2, 3]")
    assert payload is None
    assert "expected an object" in reason or "not valid JSON" in reason


# --- VLM prompt contract ----------------------------------------------------
def test_system_prompt_forbids_numbers():
    assert "VISUAL INTERPRETATION ONLY" in SYSTEM_PROMPT
    assert "Do NOT state elevations" in SYSTEM_PROMPT


def test_user_prompt_lists_the_views_in_order(tmp_path):
    images = [
        VlmImage("orthophoto", tmp_path / "a.png", "aerial image"),
        VlmImage("elevation", tmp_path / "b.png", "DEM"),
    ]
    prompt = build_user_prompt(images, "Describe this area.")
    assert "1. orthophoto: aerial image" in prompt
    assert "2. elevation: DEM" in prompt
    for key in RESPONSE_SCHEMA:
        assert key in prompt


def test_analysis_from_a_well_formed_response(tmp_path):
    images = [VlmImage("orthophoto", tmp_path / "a.png", "aerial image")]
    text = json.dumps({
        "terrain_description": "a steep slope falling to the south",
        "land_cover_description": "forest above, dense housing below",
        "built_environment_description": "a grid of small buildings",
        "notable_patterns": ["settlement stops at the break of slope"],
        "uncertainties": ["shadowed areas are hard to read"],
    })
    analysis = _to_analysis(text, provider="test", model="m", images=images)
    assert analysis.parsed is True
    assert analysis.terrain_description.startswith("a steep slope")
    assert analysis.notable_patterns == ["settlement stops at the break of slope"]
    assert analysis.images[0]["name"] == "orthophoto"


def test_unparseable_response_keeps_the_raw_text(tmp_path):
    images = [VlmImage("orthophoto", tmp_path / "a.png", "aerial image")]
    analysis = _to_analysis("sorry, no JSON here", provider="test", model="m", images=images)
    assert analysis.parsed is False
    assert analysis.raw_text == "sorry, no JSON here"
    assert analysis.terrain_description == ""      # never invented
    assert any("parsing failed" in item for item in analysis.uncertainties)


# --- provider selection -----------------------------------------------------
def test_missing_api_key_is_a_configuration_error(config, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    for section in (config.vlm, config.llm):
        section.provider = "anthropic"
        section.model = "claude-opus-5"
    with pytest.raises(ConfigurationRequiredError, match="ANTHROPIC_API_KEY"):
        load_vlm(config)
    with pytest.raises(ConfigurationRequiredError, match="ANTHROPIC_API_KEY"):
        load_llm(config)


def test_provider_none_is_a_configuration_error(config):
    config.vlm.provider = "none"
    config.llm.provider = "none"
    with pytest.raises(ConfigurationRequiredError, match="vlm.provider"):
        load_vlm(config)
    with pytest.raises(ConfigurationRequiredError, match="llm.provider"):
        load_llm(config)


def test_disabled_stage_is_a_configuration_error(config):
    config.vlm.enabled = False
    with pytest.raises(ConfigurationRequiredError):
        load_vlm(config)


def test_anthropic_adapter_builds_image_blocks(config, monkeypatch, tmp_path):
    """The request must carry one text caption and one image per view."""
    from PIL import Image

    from rokko_geofusion.vlm.adapter import AnthropicVlm

    path = tmp_path / "view.png"
    Image.new("RGB", (8, 8), (1, 2, 3)).save(path)
    images = [VlmImage("orthophoto", path, "aerial image")]

    captured: dict[str, object] = {}

    class _Messages:
        def create(self, **kwargs):
            captured.update(kwargs)

            class _Block:
                type = "text"
                text = json.dumps({"terrain_description": "flat"})

            class _Response:
                content = [_Block()]

            return _Response()

    adapter = object.__new__(AnthropicVlm)
    adapter.model = "test-model"
    adapter.max_output_tokens = 128
    adapter.temperature = 0.0
    adapter._client = type("C", (), {"messages": _Messages()})()

    analysis = adapter.analyze(images, question="Describe this area.")
    assert analysis.parsed and analysis.terrain_description == "flat"
    content = captured["messages"][0]["content"]
    assert content[0]["type"] == "text" and "orthophoto" in content[0]["text"]
    assert content[1]["type"] == "image"
    assert content[1]["source"]["media_type"] == "image/png"
    assert captured["system"] == SYSTEM_PROMPT
    assert captured["temperature"] == 0.0


def test_provider_errors_are_surfaced(config, tmp_path):
    from PIL import Image

    from rokko_geofusion.vlm.adapter import AnthropicVlm

    path = tmp_path / "view.png"
    Image.new("RGB", (8, 8)).save(path)

    class _Messages:
        def create(self, **kwargs):
            raise RuntimeError("upstream exploded")

    adapter = object.__new__(AnthropicVlm)
    adapter.model = "m"
    adapter.max_output_tokens = 16
    adapter.temperature = 0.0
    adapter._client = type("C", (), {"messages": _Messages()})()

    with pytest.raises(DataSourceError, match="upstream exploded"):
        adapter.analyze([VlmImage("v", path, "c")], question="?")


# --- rendered inputs --------------------------------------------------------
@pytest.fixture()
def rendered_scene(config):
    import numpy as np

    config.roi.name = "vlm"
    config.roi.radius_m = 40.0
    config.paths.ensure()
    roi = RoiGeometry.from_config(config.roi, CrsManager(config.crs),
                                  grid_snap_m=config.crs.grid_snap_m)
    grid = roi.grid(config.lidar.resolution_m)
    rows, _ = np.mgrid[0:grid.height, 0:grid.width]
    write_grid_raster(config.paths.raster / "elevation.tif",
                      (100.0 + rows).astype(np.float32), grid, nodata=float("nan"))
    write_grid_raster(config.paths.raster / "slope.tif",
                      np.full(grid.shape, 12.0, np.float32), grid, nodata=float("nan"))
    rgb_grid = roi.grid(config.imagery.resolution_m)
    write_grid_raster(config.paths.interim / "orthophoto.tif",
                      np.full((3, rgb_grid.height, rgb_grid.width), 90, np.uint8),
                      rgb_grid, nodata=None)
    return config, roi


def test_rendered_views_have_captions_with_units_and_crs(rendered_scene):
    config, roi = rendered_scene
    images = render_vlm_inputs(config, roi)
    assert images
    names = [image.name for image in images]
    assert names[0] == "orthophoto"
    for image in images:
        assert image.path.is_file() and image.path.stat().st_size > 0
        assert image.caption
    assert any(config.crs.projected in image.caption for image in images)
    assert any("metres" in image.caption for image in images)


def test_rendered_views_respect_the_image_budget(rendered_scene):
    config, roi = rendered_scene
    assert len(render_vlm_inputs(config, roi, max_images=2)) == 2


# --- LLM report -------------------------------------------------------------
def test_llm_system_prompt_separates_the_four_kinds():
    for key in ("measured", "observed", "inferred", "uncertain"):
        assert f'"{key}"' in LLM_SYSTEM_PROMPT
    assert "never recompute" in LLM_SYSTEM_PROMPT


def test_llm_prompt_carries_the_measurements_and_the_visual_analysis():
    payload = {"terrain": {"elevation_m": {"mean": 154.2}}}
    visual = {"terrain_description": "a steep slope"}
    prompt = build_prompt(payload, visual, language="ja")
    assert "154.2" in prompt
    assert "a steep slope" in prompt
    assert "Japanese" in prompt


def test_llm_prompt_states_when_no_visual_analysis_exists():
    prompt = build_prompt({"terrain": {}}, None)
    assert "No visual analysis is available" in prompt


class _FakeLlm:
    provider = "fake"
    model = "fake-1"

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        return self.reply


def test_report_parses_the_four_sections(config):
    reply = json.dumps({
        "summary": "A steep, partly forested slope with dense housing below.",
        "measured": ["mean elevation 154.2 m"],
        "observed": ["forest occupies the upper slopes"],
        "inferred": ["settlement is limited by the terrain"],
        "uncertain": ["no DSM, so building heights are unknown"],
        "recommended_checks": ["obtain a DSM"],
    })
    adapter = _FakeLlm(reply)
    report = run_llm_analysis(config, {"terrain": {}}, None, adapter=adapter)
    assert report.parsed is True
    assert report.measured == ["mean elevation 154.2 m"]
    assert report.uncertain == ["no DSM, so building heights are unknown"]
    assert report.provider == "fake"
    assert adapter.calls[0][0] == LLM_SYSTEM_PROMPT


def test_report_keeps_raw_text_when_unparseable(config):
    report = run_llm_analysis(config, {}, None, adapter=_FakeLlm("no json here"))
    assert report.parsed is False
    assert report.raw_text == "no json here"
    assert report.summary == ""
    assert report.measured == []


def test_report_markdown_labels_every_section():
    report = GeoAiReport(summary="s", measured=["m"], observed=["o"], inferred=["i"],
                         uncertain=["u"], recommended_checks=["c"],
                         provider="p", model="m1")
    markdown = report.to_markdown("roi_key")
    for heading in ("Measured", "Observed", "Inferred", "Uncertain", "Recommended"):
        assert heading in markdown
    assert "Numbers come from the pipeline" in markdown


# --- payload ----------------------------------------------------------------
def test_payload_reports_missing_sections_rather_than_omitting_them(config):
    config.roi.name = "empty"
    config.paths.ensure()
    roi = RoiGeometry.from_config(config.roi, CrsManager(config.crs),
                                  grid_snap_m=config.crs.grid_snap_m)
    payload = build_payload(config, roi)
    assert set(payload["missing"]) >= {"terrain", "fusion", "validation"}
    assert payload["roi"]["key"] == roi.key
    assert payload["provenance"]["crs"]["projected"] == config.crs.projected


def test_payload_includes_terrain_when_present(config):
    from rokko_geofusion.utils.metadata import write_json

    config.roi.name = "withstats"
    config.paths.ensure()
    roi = RoiGeometry.from_config(config.roi, CrsManager(config.crs),
                                  grid_snap_m=config.crs.grid_snap_m)
    write_json(config.paths.metrics / "terrain.json", {
        "area_m2": 4040100.0,
        "resolution_m": 5.0,
        "elevation": {"min": 26.709, "mean": 154.237, "max": 511.956},
        "slope": {"mean": 14.215},
        "object_height": None,
        "unavailable": {"ndsm": "no DSM configured"},
    })
    payload = build_payload(config, roi)
    assert payload["terrain"]["elevation_m"]["mean"] == 154.24
    # The range across the area is stated explicitly, and local relief is
    # defined so it cannot be mistaken for it.
    assert payload["terrain"]["elevation_range_m"] == pytest.approx(485.25, abs=0.01)
    assert "NOT the elevation range" in payload["terrain"]["local_relief_definition"]
    assert "25 m across" in payload["terrain"]["local_relief_definition"]
    assert payload["terrain"]["object_height_m"] is None
    assert "ndsm" in payload["terrain"]["unavailable"]
    assert "terrain" not in payload["missing"]
