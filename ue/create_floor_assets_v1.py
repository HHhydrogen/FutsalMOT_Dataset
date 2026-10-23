"""可分发地板资产入口：独立创建或只读验收，不更换地图绑定。"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from grf_ue_bridge.floor_diversity import CATALOG

DESTINATION = "/Game/FutsalMOT/Environment/Floors/V1"
TEXTURE_PATH = DESTINATION + "/T_FloorNeutral_Albedo_v1"
MATERIAL_PATH = DESTINATION + "/M_FloorTint_v1"
SOURCE_SHA = "81bfaa6a9e0e826446649d251d638c8a6c11319c135173f7768b3a009f5166eb"
NEUTRAL_LINEAR = 0.21404114048223255
TILE_SIZE_CM = 200.0


def target_paths():
    """所有自有资产的显式范围，不扫描或改写其他目录。"""
    return [TEXTURE_PATH, MATERIAL_PATH] + [DESTINATION + "/MI_" + v["id"] for v in CATALOG]


def source_paths(project):
    """纹理与来源跟随外层项目分发，不依赖机器历史路径。"""
    folder = Path(project) / "SourceAssets/Floors/V1"
    return folder / "T_FloorNeutral_Albedo_v1.png", folder / "source_receipt.json"


def validate_existing(unreal):
    """只读核验已有资产及世界坐标采样链；不把材质球图当场地验收。"""
    library = unreal.MaterialEditingLibrary
    texture = unreal.load_asset(TEXTURE_PATH)
    material = unreal.load_asset(MATERIAL_PATH)
    assert texture and texture.get_class().get_name() == "Texture2D"
    assert material and material.get_class().get_name() == "Material"
    assert texture.get_editor_property("srgb") is True
    for axis in ("address_x", "address_y"):
        assert texture.get_editor_property(axis) == unreal.TextureAddress.TA_WRAP
    expressions = list(library.get_material_expressions(material))
    samples = [n for n in expressions if n.get_class().get_name() == "MaterialExpressionTextureSampleParameter2D"
               and str(n.get_editor_property("parameter_name")) == "FloorNeutralTexture"]
    assert len(samples) == 1
    sample = samples[0]
    assert sample.get_editor_property("texture").get_path_name() == texture.get_path_name()

    def inputs(node):
        return [n for n in library.get_inputs_for_material_expression(material, node) if n is not None]

    sample_inputs = inputs(sample)
    assert len(sample_inputs) == 1 and sample_inputs[0].get_class().get_name() == "MaterialExpressionDivide"
    divide = sample_inputs[0]
    upstream = inputs(divide)
    mask = [n for n in upstream if n.get_class().get_name() == "MaterialExpressionComponentMask"]
    tile = [n for n in upstream if n.get_class().get_name() == "MaterialExpressionScalarParameter"
            and str(n.get_editor_property("parameter_name")) == "FloorTileSizeCM"]
    assert len(upstream) == 2 and len(mask) == len(tile) == 1
    assert upstream[0] == mask[0] and upstream[1] == tile[0], "Divide A/B 必须依次为世界XY、厘米周期"
    assert abs(float(tile[0].get_editor_property("default_value")) - TILE_SIZE_CM) < 1e-6
    assert all(bool(mask[0].get_editor_property(p)) == (p in ("r", "g")) for p in ("r", "g", "b", "a"))
    world = inputs(mask[0])
    assert len(world) == 1 and world[0].get_class().get_name() == "MaterialExpressionWorldPosition"
    assert world[0].get_editor_property("world_position_shader_offset") == unreal.WorldPositionIncludedOffsets.WPT_EXCLUDE_ALL_SHADER_OFFSETS
    records = {}
    expected_scalars = {"FloorTileSizeCM": 200.0, "FloorTextureAmplitude": 0.04,
                        "FloorRoughness": 0.72, "FloorMetallic": 0.0}
    for variant in CATALOG:
        path = DESTINATION + "/MI_" + variant["id"]
        instance = unreal.load_asset(path)
        assert instance and instance.get_class().get_name() == "MaterialInstanceConstant"
        assert instance.get_editor_property("parent").get_path_name() == material.get_path_name()
        actual = library.get_material_instance_vector_parameter_value(instance, "FloorTint")
        tint = [float(actual.r), float(actual.g), float(actual.b), float(actual.a)]
        assert all(abs(a - b) < 1e-6 for a, b in zip(tint, variant["base_color_linear"] + [1.0]))
        scalar_values = {}
        for name, value in expected_scalars.items():
            read_back = float(library.get_material_instance_scalar_parameter_value(instance, name))
            assert abs(read_back - value) < 1e-6, (path, name, read_back)
            scalar_values[name] = read_back
        records[path] = {"tint": tint, "scalars": scalar_values}
    return {"world_xy_graph_verified": True, "instances": records}


def main():
    import unreal
    project = Path(unreal.Paths.project_dir()).resolve()
    source, receipt = source_paths(project)
    assert source.is_file() and receipt.is_file(), "缺少可分发纹理源或来源记录"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == SOURCE_SHA
    provenance = json.loads(receipt.read_text(encoding="utf-8-sig"))
    assert provenance.get("sha256") == SOURCE_SHA
    library = unreal.MaterialEditingLibrary
    assetlib = unreal.EditorAssetLibrary
    tools = unreal.AssetToolsHelpers.get_asset_tools()
    required_classes = ("AssetImportTask", "MaterialFactoryNew", "MaterialInstanceConstantFactoryNew",
        "MaterialExpressionWorldPosition", "MaterialExpressionComponentMask", "MaterialExpressionDivide",
        "MaterialExpressionTextureSampleParameter2D", "MaterialExpressionScalarParameter",
        "MaterialExpressionVectorParameter", "MaterialExpressionClamp", "MaterialExpressionMultiply", "MaterialExpressionAdd")
    assert all(hasattr(unreal, name) for name in required_classes)
    for name in ("create_material_expression", "connect_material_expressions", "connect_material_property",
                 "recompile_material", "set_material_instance_parent", "set_material_instance_scalar_parameter_value",
                 "set_material_instance_vector_parameter_value", "get_material_instance_scalar_parameter_value",
                 "get_material_instance_vector_parameter_value", "get_material_expressions", "get_inputs_for_material_expression"):
        assert callable(getattr(library, name, None)), name
    enum = getattr(unreal, "WorldPositionIncludedOffsets", None)
    assert enum and hasattr(enum, "WPT_EXCLUDE_ALL_SHADER_OFFSETS")
    assert unreal.EditorLevelLibrary.get_editor_world().get_path_name().split(".", 1)[0] == "/Game/FutsalMOT/Maps/L_FutsalCourt"
    dirty = [p.get_path_name() for p in list(unreal.EditorLoadingAndSavingUtils.get_dirty_content_packages())
             + list(unreal.EditorLoadingAndSavingUtils.get_dirty_map_packages())]
    assert not dirty, "存在未保存包，请先核对来源: " + repr(dirty)
    targets = target_paths()
    exists = [assetlib.does_asset_exist(p) for p in targets]
    report = {"schema": "futsalmot_floor_asset_creation", "version": 1,
              "targets": targets, "source_sha256": SOURCE_SHA,
              "apply": os.environ.get("FUTSALMOT_FLOOR_CREATE_APPLY") == "1",
              "formula": "UV=AbsoluteWorldPosition_NoShaderOffsets.xy/FloorTileSizeCM",
              "created_assets": [], "scene_binding_changed": False,
              "visual_acceptance": False, "seamless_verified": False,
              "physical_period_render_verified": False}
    if all(exists):
        report["verification"] = validate_existing(unreal)
        report["status"] = "existing_assets_verified_readonly"
        print("FLOOR_CREATE_EXISTING " + json.dumps(report))
        return report
    assert not any(exists), "部分目标资产已存在；拒绝覆盖或隐式补齐，请单独诊断并选择明确的新attempt"
    assert not assetlib.does_directory_exist(DESTINATION), "目标目录已存在，拒绝混入未知内容"
    if not report["apply"]:
        report["status"] = "preflight_passed_not_applied"
        print("FLOOR_CREATE_PREFLIGHT " + json.dumps(report))
        return report
    evidence = project / "Saved/FloorDiversity/creation"
    evidence.mkdir(parents=True, exist_ok=True)
    report_path = evidence / ("create_" + uuid.uuid4().hex + ".json")

    def flush():
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    try:
        task = unreal.AssetImportTask()
        for key, value in {"filename": str(source), "destination_path": DESTINATION,
            "destination_name": TEXTURE_PATH.rsplit("/", 1)[-1], "automated": True,
            "replace_existing": False, "save": False}.items():
            task.set_editor_property(key, value)
        tools.import_asset_tasks([task])
        texture = unreal.load_asset(TEXTURE_PATH)
        assert texture and texture.get_class().get_name() == "Texture2D"
        report["created_assets"].append(TEXTURE_PATH)
        texture.set_editor_property("srgb", True)
        for axis in ("address_x", "address_y"):
            texture.set_editor_property(axis, unreal.TextureAddress.TA_WRAP)
        assert assetlib.save_loaded_asset(texture)

        def create(path, factory):
            result = tools.create_asset(path.rsplit("/", 1)[-1], DESTINATION, None, factory)
            assert result is not None
            report["created_assets"].append(path)
            return result

        material = create(MATERIAL_PATH, unreal.MaterialFactoryNew())

        def node(cls, **properties):
            result = library.create_material_expression(material, getattr(unreal, cls))
            assert result is not None
            for key, value in properties.items():
                result.set_editor_property(key, value)
            return result

        def link(a, out, b, pin):
            assert library.connect_material_expressions(a, out, b, pin), (out, pin)

        def scalar(name, value):
            return node("MaterialExpressionScalarParameter", parameter_name=name, default_value=float(value))

        world = node("MaterialExpressionWorldPosition", world_position_shader_offset=enum.WPT_EXCLUDE_ALL_SHADER_OFFSETS)
        xy = node("MaterialExpressionComponentMask", r=True, g=True, b=False, a=False)
        link(world, "", xy, "")
        tile = scalar("FloorTileSizeCM", TILE_SIZE_CM)
        divide = node("MaterialExpressionDivide")
        link(xy, "", divide, "A")
        link(tile, "", divide, "B")
        sample = node("MaterialExpressionTextureSampleParameter2D", parameter_name="FloorNeutralTexture",
                      texture=texture, sampler_type=unreal.MaterialSamplerType.SAMPLERTYPE_COLOR)
        link(divide, "", sample, "UVs")
        clamp = node("MaterialExpressionClamp", min_default=0.0, max_default=2 * NEUTRAL_LINEAR)
        link(sample, "R", clamp, "")
        normalize = node("MaterialExpressionMultiply", const_b=1.0 / NEUTRAL_LINEAR)
        link(clamp, "", normalize, "A")
        center = node("MaterialExpressionAdd", const_b=-1.0)
        link(normalize, "", center, "A")
        amplitude = scalar("FloorTextureAmplitude", 0.04)
        scaled = node("MaterialExpressionMultiply")
        link(center, "", scaled, "A")
        link(amplitude, "", scaled, "B")
        unity = node("MaterialExpressionAdd", const_b=1.0)
        link(scaled, "", unity, "A")
        tint = node("MaterialExpressionVectorParameter", parameter_name="FloorTint",
                    default_value=unreal.LinearColor(*CATALOG[0]["base_color_linear"], 1.0))
        color = node("MaterialExpressionMultiply")
        link(unity, "", color, "A")
        link(tint, "RGB", color, "B")
        assert library.connect_material_property(color, "", unreal.MaterialProperty.MP_BASE_COLOR)
        for name, value, prop in (("FloorRoughness", .72, unreal.MaterialProperty.MP_ROUGHNESS),
                                  ("FloorMetallic", 0., unreal.MaterialProperty.MP_METALLIC)):
            assert library.connect_material_property(scalar(name, value), "", prop)
        library.recompile_material(material)
        assert assetlib.save_loaded_asset(material)
        for variant in CATALOG:
            instance = create(DESTINATION + "/MI_" + variant["id"], unreal.MaterialInstanceConstantFactoryNew())
            library.set_material_instance_parent(instance, material)
            # UE5.8 setter会写入却恒返回false，以实际读回验收。
            library.set_material_instance_vector_parameter_value(instance, "FloorTint",
                unreal.LinearColor(*variant["base_color_linear"], 1.0))
            for name, value in (("FloorTileSizeCM", 200.), ("FloorTextureAmplitude", .04),
                                ("FloorRoughness", .72), ("FloorMetallic", 0.)):
                library.set_material_instance_scalar_parameter_value(instance, name, value)
            assert assetlib.save_loaded_asset(instance)
        report["verification"] = validate_existing(unreal)
        assert hashlib.sha256(source.read_bytes()).hexdigest() == SOURCE_SHA
        report["status"] = "created_verified_not_applied"
        flush()
    except Exception as exc:
        report["status"] = "failed_partial_assets_preserved"
        report["error"] = repr(exc)
        flush()
        raise
    print("FLOOR_CREATE_COMPLETE " + json.dumps(report))
    return report


if __name__ == "__main__":
    main()
