"""Multimodal fusion: the rule engine and the assembled feature table."""

from __future__ import annotations

import numpy as np
import pytest

from rokko_geofusion.config import FusionConfig
from rokko_geofusion.crs import CrsManager, RoiGeometry
from rokko_geofusion.exceptions import UnsupportedError
from rokko_geofusion.fusion.build import FEATURE_COLUMNS, build_fusion
from rokko_geofusion.fusion.rules import (
    RULE_NAMES,
    FusionInputs,
    apply_rules,
    class_fractions,
)
from rokko_geofusion.io.raster import grid_from_raster, write_grid_raster

IMAGE_CLASSES = ["other", "building", "road", "vegetation", "bare_soil", "water"]


def _inputs(image_class, confidence=0.9, height=None, **masks) -> FusionInputs:
    image_class = np.asarray(image_class)
    n = image_class.size
    return FusionInputs(
        image_class=image_class,
        image_confidence=np.full(n, confidence, np.float32)
        if np.isscalar(confidence) else np.asarray(confidence),
        object_height=None if height is None else np.asarray(height, np.float32),
        in_building=masks.get("in_building"),
        on_road=masks.get("on_road"),
        in_water=masks.get("in_water"),
    )


def _fused_name(config: FusionConfig, index: int) -> str:
    return config.classes[index]


# --- rules ------------------------------------------------------------------
def test_image_evidence_alone_assigns_classes():
    config = FusionConfig()
    codes = [IMAGE_CLASSES.index(n) for n in ("building", "road", "vegetation",
                                              "bare_soil", "water")]
    fused, rule, provenance = apply_rules(_inputs(codes), config=config,
                                          image_classes=IMAGE_CLASSES)
    names = [_fused_name(config, value) for value in fused]
    assert names == ["building", "road", "low_vegetation", "bare_soil", "water"]
    assert provenance["height_available"] is False
    assert provenance["unassigned"] == 0
    assert RULE_NAMES[rule[0]] == "image_building_no_height"


def test_height_separates_tall_from_low_vegetation():
    config = FusionConfig()
    vegetation = IMAGE_CLASSES.index("vegetation")
    fused, rule, provenance = apply_rules(
        _inputs([vegetation, vegetation], height=[12.0, 0.4]),
        config=config, image_classes=IMAGE_CLASSES,
    )
    assert [_fused_name(config, v) for v in fused] == ["tall_vegetation", "low_vegetation"]
    assert RULE_NAMES[rule[0]] == "image_vegetation_tall"
    assert provenance["height_available"] is True
    assert provenance["skipped_conditions"] == []


def test_height_gates_the_building_rule():
    config = FusionConfig()
    building = IMAGE_CLASSES.index("building")
    fused, _, _ = apply_rules(
        _inputs([building, building], height=[8.0, 0.2]),
        config=config, image_classes=IMAGE_CLASSES,
    )
    # A "building" pixel with no elevation above ground is not a building.
    assert _fused_name(config, fused[0]) == "building"
    assert _fused_name(config, fused[1]) != "building"


def test_missing_height_is_recorded_as_skipped_not_assumed():
    config = FusionConfig()
    building = IMAGE_CLASSES.index("building")
    fused, rule, provenance = apply_rules(_inputs([building]), config=config,
                                          image_classes=IMAGE_CLASSES)
    assert _fused_name(config, fused[0]) == "building"
    assert RULE_NAMES[rule[0]] == "image_building_no_height"
    assert len(provenance["skipped_conditions"]) == 3
    assert all("no DSM" in text for text in provenance["skipped_conditions"])


def test_nan_height_does_not_silently_pass_the_threshold():
    config = FusionConfig()
    building = IMAGE_CLASSES.index("building")
    fused, rule, _ = apply_rules(
        _inputs([building], height=[np.nan]), config=config, image_classes=IMAGE_CLASSES
    )
    assert _fused_name(config, fused[0]) == "building"
    assert RULE_NAMES[rule[0]] == "image_building_no_height"


def test_low_confidence_image_evidence_is_not_trusted():
    config = FusionConfig()
    building = IMAGE_CLASSES.index("building")
    fused, rule, _ = apply_rules(
        _inputs([building], confidence=0.1), config=config, image_classes=IMAGE_CLASSES
    )
    assert _fused_name(config, fused[0]) == "other"
    assert RULE_NAMES[rule[0]] == "image_other_low_confidence"


def test_gis_evidence_supplies_a_class_the_image_missed():
    config = FusionConfig()
    other = IMAGE_CLASSES.index("other")
    fused, rule, _ = apply_rules(
        _inputs([other, other],
                in_building=np.array([True, False]),
                on_road=np.array([False, True])),
        config=config, image_classes=IMAGE_CLASSES,
    )
    assert [_fused_name(config, v) for v in fused] == ["building", "road"]
    assert [RULE_NAMES[v] for v in rule] == ["gis_building", "gis_road"]


def test_gis_evidence_can_be_switched_off():
    config = FusionConfig(use_gis_evidence=False)
    other = IMAGE_CLASSES.index("other")
    fused, _, provenance = apply_rules(
        _inputs([other], in_building=np.array([True])),
        config=config, image_classes=IMAGE_CLASSES,
    )
    assert _fused_name(config, fused[0]) != "building"
    assert provenance["used_gis_evidence"] is False


def test_image_evidence_wins_over_gis_when_both_apply():
    config = FusionConfig()
    water = IMAGE_CLASSES.index("water")
    fused, rule, _ = apply_rules(
        _inputs([water], in_building=np.array([True])),
        config=config, image_classes=IMAGE_CLASSES,
    )
    assert _fused_name(config, fused[0]) == "water"
    assert RULE_NAMES[rule[0]] == "image_water"


def test_thresholds_come_from_configuration():
    strict = FusionConfig()
    strict.thresholds.tall_vegetation_min_height_m = 20.0
    vegetation = IMAGE_CLASSES.index("vegetation")
    fused, _, _ = apply_rules(_inputs([vegetation], height=[12.0]), config=strict,
                              image_classes=IMAGE_CLASSES)
    assert _fused_name(strict, fused[0]) == "low_vegetation"


def test_unknown_image_class_index_is_handled():
    config = FusionConfig()
    fused, rule, _ = apply_rules(_inputs([-1], confidence=0.0), config=config,
                                 image_classes=IMAGE_CLASSES)
    assert _fused_name(config, fused[0]) == "unknown"
    assert RULE_NAMES[rule[0]] == "none"


def test_class_fractions_sum_to_one():
    config = FusionConfig()
    fused = np.array([1, 1, 2, 6], np.uint8)
    fractions = class_fractions(fused, config.classes)
    assert sum(fractions.values()) == pytest.approx(1.0)
    assert fractions["building"] == pytest.approx(0.5)


# --- end to end -------------------------------------------------------------
@pytest.fixture()
def fusion_scene(config):
    """A 60 m ROI with DEM, orthophoto, slope, segmentation and GIS layers."""
    import geopandas as gpd
    from shapely.geometry import LineString, Polygon

    config.roi.name = "fuse"
    config.roi.radius_m = 30.0
    config.fusion.cell_size_m = 1.0
    config.fusion.chunk_size_m = 25.0
    config.segmentation.output_resolution_m = 1.0
    config.paths.ensure()
    roi = RoiGeometry.from_config(config.roi, CrsManager(config.crs),
                                  grid_snap_m=config.crs.grid_snap_m)

    dem_grid = roi.grid(config.lidar.resolution_m)
    write_grid_raster(config.paths.interim / "dem.tif",
                      np.full(dem_grid.shape, 100.0, np.float32), dem_grid,
                      nodata=float("nan"))
    rgb_grid = roi.grid(config.imagery.resolution_m)
    write_grid_raster(config.paths.interim / "orthophoto.tif",
                      np.full((3, rgb_grid.height, rgb_grid.width), 120, np.uint8),
                      rgb_grid, nodata=None)
    terrain_grid = dem_grid
    for name in ("slope", "aspect", "relief"):
        write_grid_raster(config.paths.raster / f"{name}.tif",
                          np.full(terrain_grid.shape, 5.0, np.float32), terrain_grid,
                          nodata=float("nan"))

    seg_grid = roi.grid(config.segmentation.output_resolution_m)
    classes = np.full(seg_grid.shape, IMAGE_CLASSES.index("vegetation"), np.uint8)
    classes[:, : seg_grid.width // 2] = IMAGE_CLASSES.index("building")
    write_grid_raster(config.paths.raster / "segmentation_class.tif", classes, seg_grid,
                      nodata=None)
    write_grid_raster(config.paths.raster / "segmentation_confidence.tif",
                      np.full(seg_grid.shape, 0.9, np.float32), seg_grid,
                      nodata=float("nan"))

    min_x, min_y, max_x, max_y = roi.bounds_projected
    gpd.GeoDataFrame(
        {"osm_id": ["way/1"], "height_m": [12.0],
         "geometry": [Polygon([(min_x + 2, min_y + 2), (min_x + 12, min_y + 2),
                               (min_x + 12, min_y + 12), (min_x + 2, min_y + 12)])]},
        crs=config.crs.projected,
    ).to_parquet(config.paths.interim / "gis_building.parquet")
    gpd.GeoDataFrame(
        {"osm_id": ["way/2"],
         "geometry": [LineString([(min_x, max_y - 5), (max_x, max_y - 5)])]},
        crs=config.crs.projected,
    ).to_parquet(config.paths.interim / "gis_road.parquet")
    return config, roi


def test_fusion_table_has_every_feature_column(fusion_scene):
    import pandas as pd

    config, roi = fusion_scene
    product = build_fusion(config, roi)
    frame = pd.read_parquet(product.table_path)
    assert list(frame.columns) == list(FEATURE_COLUMNS)
    assert len(frame) == product.n_cells == product.grid.width * product.grid.height


def test_fusion_raster_shares_the_roi_grid(fusion_scene):
    config, roi = fusion_scene
    product = build_fusion(config, roi)
    assert grid_from_raster(product.raster_path).matches(roi.grid(config.fusion.cell_size_m))


def test_fusion_records_the_missing_dsm(fusion_scene):
    config, roi = fusion_scene
    product = build_fusion(config, roi)
    assert "object_height" in product.unavailable
    assert product.statistics["skipped_conditions"]
    assert product.statistics["modalities"]["ndsm"] == "unavailable"
    assert product.statistics["modalities"]["dem"] == "available"


def test_every_cell_carries_all_modalities(fusion_scene):
    import pandas as pd

    config, roi = fusion_scene
    product = build_fusion(config, roi)
    frame = pd.read_parquet(product.table_path)
    assert frame["elevation"].notna().all()
    assert frame["slope"].notna().all()
    assert frame["image_class"].ge(0).all()
    assert frame["image_confidence"].gt(0).all()
    # object_height is genuinely unknown, and says so rather than being 0.
    assert frame["object_height"].isna().all()


def test_gis_attributes_reach_the_table(fusion_scene):
    import pandas as pd

    config, roi = fusion_scene
    product = build_fusion(config, roi)
    frame = pd.read_parquet(product.table_path)
    assert frame["in_building"].any()
    assert frame["on_road"].any()
    inside = frame[frame["in_building"]]
    np.testing.assert_allclose(inside["building_height_osm"].to_numpy(), 12.0)


def test_fused_pointcloud_classification_matches_the_table(fusion_scene):
    import pandas as pd

    from rokko_geofusion.lidar.pointcloud import (
        read_point_cloud,
        read_point_cloud_classification,
    )

    config, roi = fusion_scene
    product = build_fusion(config, roi)
    assert product.pointcloud_path is not None and product.pointcloud_path.is_file()
    frame = pd.read_parquet(product.table_path)
    xyz, rgb = read_point_cloud(product.pointcloud_path)
    classification = read_point_cloud_classification(product.pointcloud_path)
    assert len(xyz) == len(frame)
    np.testing.assert_allclose(xyz[:, 0], frame["x"].to_numpy(), atol=1e-3)
    np.testing.assert_array_equal(classification, frame["fused_class"].to_numpy())


def test_fusion_can_skip_the_pointcloud(fusion_scene):
    config, roi = fusion_scene
    config.fusion.write_pointcloud = False
    product = build_fusion(config, roi)
    assert product.pointcloud_path is None


def test_fusion_without_a_dem_is_refused(config):
    config.roi.name = "empty"
    config.roi.radius_m = 20.0
    config.paths.ensure()
    roi = RoiGeometry.from_config(config.roi, CrsManager(config.crs),
                                  grid_snap_m=config.crs.grid_snap_m)
    with pytest.raises(UnsupportedError, match="needs a DEM"):
        build_fusion(config, roi)


def test_disabled_fusion_is_refused(fusion_scene):
    config, roi = fusion_scene
    config.fusion.enabled = False
    with pytest.raises(UnsupportedError):
        build_fusion(config, roi)


def test_fusion_is_reproducible(fusion_scene):
    import pandas as pd

    config, roi = fusion_scene
    first = pd.read_parquet(build_fusion(config, roi).table_path)
    second = pd.read_parquet(build_fusion(config, roi, overwrite=True).table_path)
    pd.testing.assert_frame_equal(first, second)
