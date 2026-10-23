"""独立 Pose17 小样配对诊断；只写新分析目录，不修改媒体或提升可见性等级。"""

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path

from grf_ue_bridge.pose17_export import COCO17_NAMES, SKELETON, project_landmarks

GPU_LBS_METHOD = "gpu_render_component_pose_cpu_lbs_no_morph_cloth_deformer_wpo_diagnostic"
GPU_LBS_FENCE = "gpu_dynamic_data_endframe_tasks_finished_render_commands_flush"
GPU_LBS_SCOPE = "pose17_gpu_render_cpu_lbs"
GPU_LBS_CHARACTER_SCOPE = "pose17_gpu_render_cpu_lbs_character_calibration"
GPU_LBS_JOINT_SCOPE = "pose17_gpu_render_cpu_lbs_joint_diagnostic"
PLAYER003_ASSET = "/Game/FutsalMOT/Characters/MPFB/Players/Player_003/MPFB_Player_003.MPFB_Player_003"
PLAYER004_ASSET = "/Game/FutsalMOT/Characters/MPFB/Players/Player_004/Mesh/MPFB_Player_004.MPFB_Player_004"
# 从实际生产mesh registry核对的literal白名单，不用GetFName或路径prefix代替精确身份。
CALIBRATION_ASSETS = {
    4: PLAYER004_ASSET,
    5: "/Game/FutsalMOT/Characters/MPFB/Players/Player_005/Mesh/MPFB_Player_005.MPFB_Player_005",
    6: "/Game/FutsalMOT/Characters/MPFB/Players/Player_006/Mesh/MPFB_Player_006.MPFB_Player_006",
    7: "/Game/FutsalMOT/Characters/MPFB/Players/Player_007/Mesh/MPFB_Player_007.MPFB_Player_007",
    8: "/Game/FutsalMOT/Characters/MPFB/Players/Player_008/Mesh/MPFB_Player_008.MPFB_Player_008",
    9: "/Game/FutsalMOT/Characters/MPFB/Players/Player_009/Mesh/MPFB_Player_009.MPFB_Player_009",
}
JOINT_ROLE_IDS = {"L0":3,"L1":4,"L2":5,"L3":6,"L4":7,"R0":8,"R1":9,"R2":3,"R3":4,"R4":5}
TRANSFORM_ENCODING = "ue_transform_translation_cm_quat_xyzw_scale_xyz_v1"
LIMB_BONES = ("upperarm_l", "upperarm_r", "lowerarm_l", "lowerarm_r", "hand_l", "hand_r",
              "thigh_l", "thigh_r", "calf_l", "calf_r", "foot_l", "foot_r")


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _sha256(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def sampling_context(plan, receipt):
    """新GPU/LBS路径须双份opt-in与真实native收据；旧CPU路径不扩大门禁。"""
    path = plan.get("pose_sampling_path", "cpu_skin_cache")
    _require(path in ("cpu_skin_cache", "gpu_render_cpu_lbs"), "未知Pose采样路径")
    if path == "cpu_skin_cache":
        _require(plan.get("diagnostic_scope") not in (GPU_LBS_SCOPE, GPU_LBS_CHARACTER_SCOPE, GPU_LBS_JOINT_SCOPE),
                 "GPU/LBS scope不能伪装为CPU采样路径")
        _require(receipt.get("pose_sampling_path", "cpu_skin_cache") == path
                 and receipt.get("gpu_lbs_diagnostic_requested", False) is False,
                 "CPU计划不能消费GPU/LBS native收据")
        return {"path": path}
    _require(plan.get("allow_gpu_render_cpu_lbs_diagnostic") is True
             and plan.get("allow_pie_cpu_preparation") is False,
             "GPU/LBS必须显式opt-in且禁用CPU渲染准备")
    scope = plan.get("diagnostic_scope")
    _require(scope == receipt.get("diagnostic_scope") and scope in (GPU_LBS_SCOPE, GPU_LBS_CHARACTER_SCOPE, GPU_LBS_JOINT_SCOPE),
             "GPU/LBS诊断scope不同")
    _require(receipt.get("pose_sampling_path") == path and receipt.get("gpu_lbs_diagnostic_requested") is True,
             "缺少原生GPU/LBS路径证据")
    if scope == GPU_LBS_SCOPE:
        _require(plan.get("allow_pie_visual_mesh_override", False) is False
                 and receipt.get("visual_mesh_override_requested") is False
                 and receipt.get("visual_mesh_overrides") == []
                 and all(not t.get("diagnostic_visual_mesh_override") for t in plan["targets"]),
                 "原GPU/LBS scope仍拒绝角色override")
    _require(receipt.get("camera_state_restored") is True and receipt.get("restore_errors") == []
             and receipt.get("streaming_state_restart_required") is False
             and receipt.get("pipeline_setup_error") == "",
             "GPU/LBS缺少干净native teardown证据")
    warmup = plan.get("gpu_lbs_warmup_frames")
    prepared = receipt.get("gpu_lbs_prepared_engine_frame")
    _require(type(warmup) is int and 8 <= warmup <= 120
             and type(prepared) is int and prepared >= 0, "GPU/LBS实际warmup身份缺失")
    bindings = {}
    manifest_cache = {}
    for target in plan["targets"]:
        entity = target["entity_id"]
        _require(entity not in bindings and target.get("component_object_name") == "MPFBVisual",
                 "GPU/LBS实体重复或不是实际MPFBVisual")
        manifest_path = Path(target["landmark_manifest"])
        if manifest_path not in manifest_cache:
            raw = manifest_path.read_bytes()
            manifest_cache[manifest_path] = (hashlib.sha256(raw).hexdigest(), json.loads(raw)["ue_identity"])
        digest, identity = manifest_cache[manifest_path]
        bindings[entity] = {**target, "manifest_sha256": digest, "ue_identity": identity}
    _require(bool(bindings), "GPU/LBS不能使用空目标")
    result = {"path": path, "scope": scope, "warmup_frames": warmup, "prepared_engine_frame": prepared, "bindings": bindings}
    if scope == GPU_LBS_CHARACTER_SCOPE:
        result["override_evidence"] = check_character_override(plan, receipt, result)
        result["calibration_player_id"] = plan["calibration_player_id"]
    elif scope == GPU_LBS_JOINT_SCOPE:
        result["overrides_by_entity"] = check_joint_context(plan, receipt, result)
        result["calibration_by_entity"] = dict(JOINT_ROLE_IDS)
        result["review_frame_indices"] = list(plan["review_frame_indices"])
        result["start_frame"] = plan["start_frame"]
    return result


def check_gpu_eligibility(eligibility, vertex_count, label="GPU/LBS"):
    """同一拒绝规则同时核验swap后和采样时；不把eligible布尔当完整证据。"""
    _require(isinstance(eligibility, dict), label + "缺资格原始字段")
    for name in ("checks_executed", "eligible", "pie_world", "weights_readable"):
        _require(eligibility.get(name) is True, label + "拒绝证据未执行/未通过: " + name)
    for name in ("render_cpu_enabled", "render_static", "has_nanite_data", "parallel_evaluation_running",
                 "leader_parallel_evaluation_running", "non_finite_morph_weights", "lod0_has_cloth_data",
                 "variable_influence_weights"):
        _require(eligibility.get(name) is False, label + "不支持的实际状态: " + name)
    for name in ("active_morph_weight_count", "manual_morph_curve_count", "external_morph_set_count",
                 "cloth_simulation_count", "lod0_cloth_section_count", "mesh_deformer_asset_count",
                 "mesh_deformer_instance_count", "material_wpo_slot_count", "null_material_slot_count"):
        _require(type(eligibility.get(name)) is int and eligibility[name] == 0,
                 label + "额外形变或材质缺失: " + name)
    for name in ("weight_vertex_count", "reference_vertex_count"):
        _require(type(eligibility.get(name)) is int and eligibility[name] == vertex_count,
                 label + "实际权重/参考顶点数不同: " + name)
    _require(type(eligibility.get("max_bone_influences")) is int and eligibility["max_bone_influences"] > 0,
             label + "缺实际固定影响数")


def check_character_override(plan, receipt, context):
    """只接收004–009白名单的显式单角色组合；原003父mesh不变。"""
    _require(plan.get("allow_pie_visual_mesh_override") is True
             and plan.get("allow_pie_driver_hide") is True
             and receipt.get("visual_mesh_override_requested") is True,
             "组合scope必须明确PIE override/隐藏driver且native确有请求")
    player_id = plan.get("calibration_player_id")
    _require(type(player_id) is int and player_id in CALIBRATION_ASSETS,
             "组合scope仅允许显式整数Player004–009校准")
    expected_asset = CALIBRATION_ASSETS[player_id]
    _require(len(plan["targets"]) == len(context["bindings"]) == 1, "组合scope只允许单角色")
    target = plan["targets"][0]
    _require(target.get("entity_id") == "L0" and target.get("sequence_binding_name") == "Player_L0"
             and target.get("diagnostic_visual_mesh_override") == expected_asset,
             "组合scope必须绑定L0/Player_L0的精确白名单角色override")
    binding = context["bindings"]["L0"]
    _require(binding["ue_identity"].get("asset_path") == expected_asset,
             "角色override必须使用该角色manifest身份，不能跨角色复用")
    override_warmup = plan.get("override_warmup_frames")
    _require(type(override_warmup) is int and override_warmup == context["warmup_frames"],
             "组合scope两套计划warmup必须相等且为8..120整数")
    overrides = receipt.get("visual_mesh_overrides")
    _require(isinstance(overrides, list) and len(overrides) == 1 and isinstance(overrides[0], dict),
             "组合scope缺唯一native override收据")
    evidence = overrides[0]
    return check_override_transaction(evidence, target, binding, context, expected_asset, (PLAYER003_ASSET,))


def check_override_transaction(evidence, target, binding, context, expected_asset, allowed_original_assets):
    """复用实际per-component事务；联合scope仅扩大精确原资产白名单。"""
    for name in ("identity_verified", "apply_succeeded", "animation_binding_preserved", "new_asset_materials_verified",
                 "restore_attempted", "restore_succeeded", "streaming_state_matched",
                 "gpu_lbs_validation_attempted", "gpu_lbs_validation_succeeded"):
        _require(evidence.get(name) is True, "组合scope实际apply/恢复/材质/GPU检查未通过: " + name)
    for name in ("apply_error", "restore_error", "gpu_lbs_validation_error"):
        _require(evidence.get(name) == "", "组合scope实际override错误: " + name)
    component = evidence.get("component_path", "")
    suffix = "." + target["actor_object_name"] + ".MPFBVisual"
    _require(isinstance(component, str) and re.search(r"/UEDPIE_\d+_", component) is not None
             and component.endswith(suffix), "组合scope override不是计划内实际PIE组件")
    _require(evidence.get("original_asset_path") in allowed_original_assets
             and evidence.get("applied_asset_path") == expected_asset,
             "组合scope原/新资产不在精确白名单，不能仅切换标签")
    skeleton = evidence.get("shared_skeleton_path", "")
    animation = evidence.get("animation_class_path", "")
    _require(isinstance(skeleton, str) and skeleton.startswith("/Game/")
             and skeleton.endswith("/SKEL_MPFB_Base.SKEL_MPFB_Base"), "组合scope不是共享MPFB skeleton")
    _require(isinstance(animation, str) and animation.startswith("/Game/")
             and animation.endswith("/ABP_MPFB_RuntimeRetarget.ABP_MPFB_RuntimeRetarget_C"),
             "组合scope不是共享runtime retarget动画class")
    instance = evidence.get("original_animation_instance_path", "")
    _require(isinstance(instance, str) and instance.startswith(component + ".") and len(instance) > len(component) + 1,
             "组合scope缺原PIE动画实例身份")
    applied = evidence.get("applied_asset_material_paths")
    original = evidence.get("original_override_material_paths")
    material_path = lambda p: isinstance(p, str) and p.startswith(("/Game/", "/Engine/")) and "." in p
    _require(isinstance(applied, list) and applied and all(material_path(p) for p in applied),
             "组合scope缺新资产实际非空材质路径")
    # 原OverrideMaterials快照中空指针由C++明确写成空串，不伪造材质hash或非空slot。
    _require(isinstance(original, list) and all(p == "" or material_path(p) for p in original),
             "组合scope缺原OverrideMaterials快照")
    applied_frame = evidence.get("applied_engine_frame")
    _require(type(applied_frame) is int and 0 <= applied_frame <= context["prepared_engine_frame"],
             "组合scope GPU准备帧不得早于真实mesh apply")
    check_gpu_eligibility(evidence.get("gpu_lbs_eligibility_after_override"), int(binding["ue_identity"]["vertex_count"]),
                          "组合scope swap后GPU/LBS")
    return evidence


def check_joint_context(plan, receipt, context):
    """一次十实体联合诊断；没有新增native角色ID或逐角色像素门槛。"""
    location_pilot = False
    if "location_pilot_contract" in plan or "location_pilot" in plan:
        from pose17_location_pilot_contract import validate_location_pilot_contract
        location_pilot = validate_location_pilot_contract(plan)
    _require(plan.get("allow_joint_pose17_diagnostic") is True
             and plan.get("allow_pie_visual_mesh_override") is True
             and plan.get("allow_pie_driver_hide") is True
             and receipt.get("visual_mesh_override_requested") is True,
             "联合scope须显式joint/GPU/PIE opt-in并隐藏driver")
    _require(type(plan.get("frame_count")) is int and plan["frame_count"] in ((120,) if location_pilot else (3, 10))
             and type(plan.get("start_frame")) is int and plan["start_frame"] >= 0,
             "联合scope仅允许显式3或10个连续输出帧")
    _require(plan.get("review_frame_indices") == [plan["frame_count"] // 2]
             and type(plan["review_frame_indices"][0]) is int,
             "联合scope仅冻结中间输出帧作为总览：3帧取1，10帧取5")
    cameras = plan.get("cameras", [])
    _require(len(cameras) == 4 and {c.get("camera_id") for c in cameras} == {"CineCam_01","CineCam_02","CineCam_03","CineCam_04"}
             and all(isinstance(c.get("source_sequence"), str) and c["source_sequence"].startswith("/Game/") for c in cameras),
             "联合scope需要四个显式源Sequence/原相机")
    _require(type(plan.get("override_warmup_frames")) is int
             and plan["override_warmup_frames"] == context["warmup_frames"], "联合scope两套warmup必须相等")
    _require(len(plan["targets"]) == len(context["bindings"]) == 10
             and set(context["bindings"]) == set(JOINT_ROLE_IDS), "联合scope实体集合不是固定十人")
    _require(len({t["actor_object_name"] for t in plan["targets"]}) == 10, "联合scope实际Actor重复")
    overrides = receipt.get("visual_mesh_overrides")
    _require(isinstance(overrides, list) and len(overrides) == 8
             and all(isinstance(e, dict) for e in overrides), "联合scope必须有八个唯一override事务")
    paths = [e.get("component_path") for e in overrides]
    _require(all(isinstance(p, str) for p in paths) and len(set(paths)) == 8, "联合scope重复override组件")
    allowed_original = tuple([PLAYER003_ASSET] + list(CALIBRATION_ASSETS.values()))
    matched = {}
    for entity, player_id in JOINT_ROLE_IDS.items():
        target = context["bindings"][entity]
        _require(type(target.get("calibration_player_id")) is int and target["calibration_player_id"] == player_id
                 and target.get("sequence_binding_name") == "Player_" + entity,
                 "联合scope entity/计划角色ID/Sequence绑定错配: " + entity)
        expected_asset = PLAYER003_ASSET if player_id == 3 else CALIBRATION_ASSETS[player_id]
        _require(target["ue_identity"].get("asset_path") == expected_asset,
                 "联合scope角色manifest身份不同: " + entity)
        suffix = "." + target["actor_object_name"] + ".MPFBVisual"
        entries = [e for e in overrides if e["component_path"].endswith(suffix)]
        if player_id == 3:
            _require(target.get("diagnostic_visual_mesh_override") is None and not entries,
                     "联合scope已为003的实体不得override: " + entity)
        else:
            _require(target.get("diagnostic_visual_mesh_override") == expected_asset and len(entries) == 1,
                     "联合scope角色/实际组件override不唯一: " + entity)
            matched[entity] = check_override_transaction(entries[0], target, target, context, expected_asset, allowed_original)
    _require(len(matched) == 8, "联合scope遗漏或多余override事务")
    return matched


def _check_transform(value, label):
    _require(isinstance(value, dict), label + "缺少显式transform")
    for field, size in (("translation_cm", 3), ("rotation_xyzw", 4), ("scale_xyz", 3)):
        vector = value.get(field)
        _require(isinstance(vector, list) and len(vector) == size
                 and all(type(v) in (int, float) and math.isfinite(v) for v in vector),
                 label + "的transform非有限或编码不符")
    _require(abs(sum(v*v for v in value["rotation_xyzw"]) - 1) <= 1e-3,
             label + "的Quaternion不是单位旋转")
    _require(all(abs(v) > 1e-8 for v in value["scale_xyz"]), label + "的scale退化")


def check_gpu_lbs_sample(row, context):
    """逐值验证原生GPU渲染/CPU-LBS；不能从方法名推定无额外形变。"""
    _require(row.get("sample_succeeded") is True and row.get("cache_fence_completed_this_engine_frame") is True
             and row.get("local_read_stable") is True and row.get("mrq_synchronization_verified") is False,
             "GPU/LBS实际采样/fence/stable布尔证据缺失或自称MRQ已验")
    for name in ("engine_frame_before", "engine_frame_after", "hook_engine_frame",
                 "pose_revision_before", "pose_revision_after", "output_frame", "sequence_source_frame"):
        _require(type(row.get(name)) is int and row[name] >= 0, "GPU/LBS当前帧/revision身份非法: " + name)
    for name in ("temporal_sample_index", "spatial_sample_index"):
        _require(type(row.get(name)) is int and row[name] == 0, "GPU/LBS单sample索引必须为0: " + name)
    _require(row.get("sampling_method") == GPU_LBS_METHOD, "GPU/LBS实际采样方法不同")
    _require(row.get("render_cpu_enabled_at_read") is False
             and row.get("render_static_at_read") is False
             and type(row.get("predicted_lod_at_read")) is int and row["predicted_lod_at_read"] == 0,
             "GPU/LBS实际渲染必须CPU=false、非static、LOD0")
    _require(row.get("pose_fence_method") == GPU_LBS_FENCE
             and row.get("end_frame_update_tasks_finished") is True,
             "GPU/LBS缺少实际end-frame任务/fence证据")
    observed = row.get("gpu_lbs_warmup_engine_frames_observed")
    _require(type(observed) is int and observed >= context["warmup_frames"]
             and observed == row["hook_engine_frame"] - context["prepared_engine_frame"],
             "GPU/LBS实际warmup与同帧准备身份不同")
    _require(row.get("ref_to_local_signature_scheme") == "ue_ref_to_local_rowmajor_f32_le_v1"
             and _sha256(row.get("ref_to_local_matrices_sha256")), "缺少当前RefToLocal签名")
    _require(row.get("skin_weight_signature_scheme") == "ue_component_lod0_fixed_skinweight_raw_v1"
             and _sha256(row.get("skin_weight_data_sha256")), "缺少实际组件LOD0权重签名")
    target = context["bindings"].get(row.get("entity_id"))
    _require(target is not None, "GPU/LBS实体不在显式计划")
    _require(row.get("manifest_sha256") == target["manifest_sha256"], "GPU/LBS当前manifest SHA不匹配")
    identity = row.get("mesh_identity", {})
    expected = target["ue_identity"]
    for key in ("asset_path", "signature_scheme", "topology_sha256", "geometry_sha256"):
        _require(identity.get(key) == expected.get(key), "GPU/LBS实际mesh身份不同: " + key)
    # FFutsalPose17MeshIdentity.LOD真实UStruct JSON是lOD，经现有normalizer得到l_od。
    _require(identity.get("signature_scheme") == "ue_lod_le_u32_f32_v1"
             and type(identity.get("l_od")) is int and identity["l_od"] == int(expected["lod"]) == 0
             and type(identity.get("vertex_count")) is int
             and identity["vertex_count"] == int(expected["vertex_count"]) > 0
             and _sha256(identity.get("topology_sha256")) and _sha256(identity.get("geometry_sha256")),
             "GPU/LBS真实LOD0拓扑/几何签名不符")
    component = row.get("component_path", "")
    suffix = "." + target["actor_object_name"] + "." + target["component_object_name"]
    _require(isinstance(component, str) and re.search(r"/UEDPIE_\d+_", component) is not None
             and component.endswith(suffix), "GPU/LBS不是计划内实际PIE组件")
    base = row.get("bone_space_base_component_path", "")
    owner = component.rsplit(".", 1)[0] + "."
    _require(isinstance(base, str) and base.startswith(owner) and base[len(owner):]
             and "." not in base[len(owner):], "GPU/LBS骨pose来源不是同一PIE Actor组件")
    names = row.get("bone_space_base_names")
    transforms = row.get("bone_space_base_transforms_ue_cm")
    _require(row.get("bone_pose_transform_encoding") == TRANSFORM_ENCODING
             and isinstance(names, list) and names and all(isinstance(n, str) and n for n in names)
             and len(set(names)) == len(names) and set(LIMB_BONES).issubset(names)
             and isinstance(transforms, list) and len(transforms) == len(names),
             "GPU/LBS当前骨pose名字/transform身份不完整")
    for transform in transforms:
        _check_transform(transform, "当前骨pose")
    _require(row.get("world_transform_encoding") == TRANSFORM_ENCODING, "GPU/LBS世界transform编码缺失")
    for name in ("actor_world_transform_ue_cm", "visual_world_transform_ue_cm"):
        _check_transform(row.get(name), name)
    eligibility = row.get("gpu_lbs_eligibility", {})
    check_gpu_eligibility(eligibility, identity["vertex_count"])
    joint = context.get("scope") == GPU_LBS_JOINT_SCOPE
    override_evidence = context.get("overrides_by_entity", {}).get(row.get("entity_id")) if joint else context.get("override_evidence")
    if context.get("scope") == GPU_LBS_CHARACTER_SCOPE or override_evidence is not None:
        _require(isinstance(override_evidence, dict) and override_evidence["component_path"] == component
                 and override_evidence["applied_asset_path"] == identity["asset_path"],
                 "组合scope每帧采样与override组件/新mesh身份不同")
        elapsed = row.get("override_warmup_engine_frames_observed")
        _require(type(elapsed) is int and elapsed == row["hook_engine_frame"] - override_evidence["applied_engine_frame"]
                 and elapsed >= context["warmup_frames"] and elapsed >= observed,
                 "组合scope真实override warmup与apply/GPU准备身份不同")
    elif joint:
        _require(context["calibration_by_entity"][row["entity_id"]] == 3,
                 "联合scope非003实体缺override事务")
    _require(isinstance(row.get("world_m"), list) and len(row["world_m"]) == 17
             and all(isinstance(p, list) and len(p) == 3
                     and all(type(v) in (int, float) and math.isfinite(v) for v in p) for p in row["world_m"]),
             "GPU/LBS世界点不完整或非有限")
    return {"path": context["path"], "sampling_method": row["sampling_method"],
            "diagnostic_scope": context.get("scope", GPU_LBS_SCOPE),
            "calibration_player_id": context.get("calibration_by_entity", {}).get(row.get("entity_id"), context.get("calibration_player_id")),
            "actual_gpu_render_cpu_lbs_checks_passed": True, "manifest_sha256": row["manifest_sha256"],
            "mesh_identity": identity, "component_path": component, "bone_space_base_component_path": base,
            "bone_pose_count": len(names), "bone_pose_transform_encoding": row["bone_pose_transform_encoding"],
            "actual_render_cpu_enabled": False, "actual_render_static": False, "actual_predicted_lod": 0,
            "ref_to_local_signature_scheme": row["ref_to_local_signature_scheme"],
            "ref_to_local_matrices_sha256": row["ref_to_local_matrices_sha256"],
            "skin_weight_signature_scheme": row["skin_weight_signature_scheme"],
            "skin_weight_data_sha256": row["skin_weight_data_sha256"], "pose_fence_method": row["pose_fence_method"],
            "gpu_lbs_eligibility": eligibility,
            "gpu_lbs_warmup_engine_frames_observed": observed,
            "character_override_evidence": override_evidence,
            "override_warmup_engine_frames_observed": row.get("override_warmup_engine_frames_observed") if override_evidence else None,
            "final_rendered_surface_equivalence_verified": False}


def media_index(directory, kind, camera_id=None):
    """按显式帧号配对；不把 ObjectID 调试 PNG 当作 RGB。"""
    found = {}
    for path in sorted(Path(directory).iterdir()):
        if not path.is_file() or path.suffix.lower() != (".png" if kind == "rgb" else ".exr"):
            continue
        if kind == "rgb":
            # 原生多pass名称含camera；FinalImage专用输出保留纯pass名称。
            # 两种都精确锚定，不能把WorldDepth/ObjectID前缀当成RGB。
            camera = re.escape(camera_id or Path(directory).name)
            match = re.fullmatch(r"FinalImage(?:" + camera + r"\.)?(\d+)\.png", path.name, re.I)
            if match is None:
                continue
        else:
            match = re.search(r"(?:^|\.)(\d+)\.exr$", path.name, re.I)
        _require(match is not None, "无法读取媒体帧号: " + path.name)
        frame = int(match.group(1))
        _require(frame not in found, "同帧媒体重复: " + str(frame))
        found[frame] = path
    return found


def ue_rotation_basis(pitch, yaw, roll):
    """UE FRotationMatrix 三个单位轴；角度单位为度。"""
    p, y, r = [math.radians(float(v)) for v in (pitch, yaw, roll)]
    cp, sp, cy, sy, cr, sr = math.cos(p), math.sin(p), math.cos(y), math.sin(y), math.cos(r), math.sin(r)
    return ([cp * cy, cp * sy, sp],
            [sr * sp * cy - cr * sy, sr * sp * sy + cr * cy, -sr * cp],
            [-(cr * sp * cy + sr * sy), cy * sr - cr * sp * sy, cr * cp])


def camera_from_exr_header(header):
    """只支持已检查的小样对称透视；保留半像素约定待视觉核验。"""
    def value(key):
        _require(key in header, "EXR 缺少渲染元数据: " + key)
        item = header[key]
        return item.decode() if isinstance(item, bytes) else item
    box = value("dataWindow")
    width, height = [int(v) for v in (box[1] - box[0] + 1)]
    _require(list(box[0]) == [0, 0] and width > 0 and height > 0, "不支持偏移 dataWindow")
    fov = float(value("unreal/camera/FinalImage/fov"))
    _require(0 < fov < 180, "FOV 无效")
    aspect = float(value("unreal/camera/FinalImage/sensorAspectRatio"))
    _require(abs(aspect - width / height) < 2e-5, "Filmback/output 宽高比不同，需要最终投影矩阵")
    _require(abs(float(value("unreal/camera/FinalImage/overscanPercent"))) < 1e-8, "不支持 overscan")
    _require(abs(float(value("unreal/camera/FinalImage/squeezeFactor")) - 1) < 1e-8, "不支持 anamorphic squeeze")
    location = [float(value("unreal/camera/curPos/" + k)) * .01 for k in "xyz"]
    axes = ue_rotation_basis(*[float(value("unreal/camera/curRot/" + k)) for k in ("pitch", "yaw", "roll")])
    focal = width / (2 * math.tan(math.radians(fov) / 2))
    return {"intrinsics": {"width": width, "height": height, "fx": focal, "fy": focal,
                            "cx": (width - 1) / 2, "cy": (height - 1) / 2},
            "extrinsics": dict(zip(("world_location_m", "forward", "right", "up"), [location] + list(axes))),
            "horizontal_fov_degrees": fov,
            "projection_method": "EXR_renderer_camera_symmetric_pinhole_pixel_center_minus_half",
            "final_projection_matrix_verified": False}


def check_pair(row, header, camera_id, frame, sampling=None):
    """检查独立帧身份和缓存证据；通过仍不等于像素级同步已验收。"""
    gpu_evidence = None
    requested_gpu = sampling is not None and sampling.get("path") == "gpu_render_cpu_lbs"
    if requested_gpu:
        _require(row.get("camera_name") == camera_id, "GPU/LBS native相机名与显式计划不同")
        gpu_evidence = check_gpu_lbs_sample(row, sampling)
    else:
        _require(row.get("sampling_method") != GPU_LBS_METHOD,
                 "GPU/LBS原生行不能绕过显式计划/收据门禁进入CPU分支")
    _require(row["sample_succeeded"], "native 采样失败: " + row.get("error", ""))
    _require(row["keypoint_names"] == list(COCO17_NAMES), "不是固定 COCO17 顺序")
    _require(row["output_frame"] == frame, "native 输出帧与文件帧不一致")
    _require(str(header["unreal/jobName"]) == camera_id, "EXR job 身份与显式相机计划不同")
    _require(int(header["unreal/sequenceFrameNumber"]) == row["sequence_source_frame"], "EXR/native 源帧不一致")
    if requested_gpu and sampling.get("scope") == GPU_LBS_JOINT_SCOPE:
        _require(int(header["unreal/sequenceFrameNumberRelative"]) == frame,
                 "联合scope EXR相对输出帧不一致")
        _require(row["sequence_source_frame"] == sampling["start_frame"] + frame,
                 "联合scope源帧未遵循冻结连续窗")
    _require(row["temporal_sample_count"] == row["spatial_sample_count"] == 1, "必须单 temporal/spatial sample")
    for name in ("spatialSampleCount", "temporalSampleCount"):
        _require(int(header["unreal/aa/" + name]) == 1, "EXR AA sample 数不为1")
    _require(abs(float(header["unreal/camera/shutterAngle"])) < 1e-6 and abs(row["motion_blur_fraction"]) < 1e-8,
             "运动模糊未关闭")
    _require(row["cache_fence_completed_this_engine_frame"] and row["local_read_stable"], "CPU 缓存/读取不稳定")
    _require(row["engine_frame_before"] == row["engine_frame_after"] == row["hook_engine_frame"], "采样引擎帧不一致")
    _require(row["pose_revision_before"] == row["pose_revision_after"], "读取期间骨骼 revision 改变")
    for name in ("projection_jitter", "off_center_projection_offset"):
        _require(all(abs(v) < 1e-8 for v in row[name]), "不支持非零 " + name)
    camera = camera_from_exr_header(header)
    _require(list(row["image_size"]) == [camera["intrinsics"][k] for k in ("width", "height")], "native/EXR尺寸不同")
    for field in ("world_location_m", "forward", "right", "up"):
        native = row["camera_" + field]
        _require(max(abs(a-b) for a,b in zip(native, camera["extrinsics"][field])) < 2e-5,
                 "native/EXR相机不同: " + field)
    _require(abs(row["camera_horizontal_fov_degrees"] - camera["horizontal_fov_degrees"]) < 2e-5, "native/EXR FOV不同")
    _require(abs(float(header["unreal/camera/ActorHitProxyMask/fov"]) - camera["horizontal_fov_degrees"]) < 2e-5,
             "RGB/ObjectID FOV不同")
    if gpu_evidence is not None:
        camera["native_sampling_evidence"] = gpu_evidence
    return camera


def check_expected_camera_plan(camera, camera_plan):
    """核对计划与实际EXR view；不把copy轨道读回冒充最终相机执行。"""
    if "closeup_location_cm" not in camera_plan and "closeup_look_at_cm" not in camera_plan:
        return {"applicable":False}
    location = [v*.01 for v in camera_plan["closeup_location_cm"]]
    target = [v*.01 for v in camera_plan["closeup_look_at_cm"]]
    direction = [b-a for a,b in zip(location,target)]
    length = math.sqrt(sum(v*v for v in direction))
    _require(length>0,"计划相机位置与look-at目标重合")
    forward = [v/length for v in direction]
    extr = camera["extrinsics"]
    location_error = max(abs(a-b) for a,b in zip(location,extr["world_location_m"]))
    direction_error = max(abs(a-b) for a,b in zip(forward,extr["forward"]))
    return {"applicable":True,"plan_camera_match_passed":location_error<=2e-5 and direction_error<=2e-5,
        "expected_location_m":location,"actual_EXR_location_m":extr["world_location_m"],
        "expected_look_at_forward":forward,"actual_EXR_forward":extr["forward"],
        "position_max_error_m":location_error,"forward_max_component_error":direction_error,
        "forward_cos_similarity":sum(a*b for a,b in zip(forward,extr["forward"])),
        "location_tolerance_m":2e-5,"forward_component_tolerance":2e-5,
        "runtime_look_at_bool_directly_observed":False}


def pose_motion(rows):
    """区分世界移动和去除参考点平移后的变化，不把平移误报为身体动态。"""
    ordered = sorted(rows, key=lambda r: r["sequence_source_frame"])
    result = []
    for a, b in zip(ordered, ordered[1:]):
        deltas = [[y-x for x,y in zip(p,q)] for p,q in zip(a["world_m"], b["world_m"])]
        # 左右髋的中点只用作去平移的数值参考，不宣称骨盆标注。
        root = [(deltas[11][i] + deltas[12][i]) / 2 for i in range(3)]
        norm = lambda v: math.sqrt(sum(x*x for x in v))
        result.append({"from_source_frame": a["sequence_source_frame"], "to_source_frame": b["sequence_source_frame"],
                       "max_world_delta_m": max(map(norm, deltas)), "hip_midpoint_translation_m": norm(root),
                       "max_translation_removed_delta_m": max(norm([v-r for v,r in zip(d,root)]) for d in deltas)})
    return result


def _draw_overlay(rgb_path, points, bbox, output):
    from PIL import Image, ImageDraw
    image = Image.open(rgb_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    for left, right in SKELETON:
        p, q = points[left-1].get("uv_px"), points[right-1].get("uv_px")
        if p and q:
            draw.line([tuple(p), tuple(q)], fill=(255, 170, 0), width=3)
    for index, point in enumerate(points):
        uv = point.get("uv_px")
        if uv and point["visibility_state"] == "unknown":
            x,y = uv
            draw.ellipse((x-5,y-5,x+5,y+5), outline=(255,0,255), width=2)
            draw.text((x+6,y-7), str(index), fill=(255,255,255))
    if bbox:
        x,y,w,h = bbox
        draw.rectangle((x,y,x+w-1,y+h-1), outline=(0,255,255), width=2)
        image.crop((max(0,x-50),max(0,y-50),min(image.width,x+w+50),min(image.height,y+h+50))).save(output.with_name(output.stem + "_crop.png"))
    image.save(output)


def cryptomatte_manifest_from_header(header):
    """联合快速配对只读公开EXR header，不访问channels或解码整帧像素。"""
    for key, value in header.items():
        if key.startswith("cryptomatte/") and key.endswith("/manifest"):
            manifest = json.loads(value) if isinstance(value, (str, bytes)) else value
            _require(isinstance(manifest, dict) and manifest
                     and all(isinstance(k, str) and isinstance(v, str) and re.fullmatch(r"[0-9a-fA-F]{8}", v)
                             for k,v in manifest.items()), "EXR header Cryptomatte manifest无效")
            return manifest
    raise ValueError("EXR header缺Cryptomatte manifest")


def _draw_joint_overview(rgb_path, rows, output):
    """一帧全员合成一张总览，不生成十张单人4K图或裁图。"""
    from PIL import Image, ImageDraw
    with Image.open(rgb_path) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    for index, row in enumerate(rows):
        color = ((67*index+70)%230+25, (113*index+50)%230+25, (157*index+30)%230+25)
        points = row["keypoints"]
        for left,right in SKELETON:
            p,q = points[left-1].get("uv_px"),points[right-1].get("uv_px")
            if p is not None and q is not None:draw.line([tuple(p),tuple(q)],fill=color,width=2)
        for point in points:
            uv = point.get("uv_px")
            if uv is not None:
                x,y = uv;draw.ellipse((x-3,y-3,x+3,y+3),outline=color,width=2)
        bbox = row["visible_instance_bbox_xywh"]
        if bbox:
            x,y,w,h=bbox;draw.rectangle((x,y,x+w-1,y+h-1),outline=color,width=2)
            draw.text((x,y),row["entity_id"],fill=color)
    image.save(output)


def inspect(attempt, output, pixel_review=True):
    """仅消费完整落盘的诊断小样；缺少收据时拒绝生成伪Pose。"""
    import numpy as np
    import OpenEXR
    from PIL import Image
    from grf_ue_bridge.cryptomatte import load_cryptomatte, build_mask
    ue_dir = Path(__file__).resolve().parents[3] / "ue"
    if str(ue_dir) not in sys.path:
        sys.path.insert(0, str(ue_dir))
    from pose17_native_receipt import normalize_native_receipt
    attempt, output = Path(attempt).resolve(), Path(output).resolve()
    _require(not output.exists(), "分析目录已存在，拒绝覆盖")
    plan = json.loads((attempt / "plan.json").read_text(encoding="utf-8"))
    receipts = {}
    sampling_contexts = {}
    for camera in plan["cameras"]:
        name = camera["camera_id"]
        path = attempt / (name + "_native_pose_restore_receipt.json")
        _require(path.is_file(), "缺少 native 收据；不能事后重建Pose: " + str(path))
        receipts[name] = normalize_native_receipt(json.loads(path.read_text(encoding="utf-8")))
        sampling_contexts[name] = sampling_context(plan, receipts[name])
    output.mkdir(parents=True)
    report = {"schema": "futsalmot_pose17_smoke_alignment_diagnostic_v1", "attempt": str(attempt),
              "MRQ_sync_verified": False, "visibility_policy": "unknown保持未知；mask命中仅是邻域诊断，不证明自遮挡",
              "source_scope": plan["source_scope"], "pose_sampling_path": plan.get("pose_sampling_path", "cpu_skin_cache"),
              "cameras": {}, "errors": []}
    joint = plan.get("diagnostic_scope") == GPU_LBS_JOINT_SCOPE
    _require(pixel_review or joint, "跳过像素抽样仅适用于已复用联合GPU通路")
    if joint:
        report["pixel_evaluation_scope"] = ("仅review_frame_indices=" + str(plan["review_frame_indices"]) + "逐camera一张全员总览；其余header/metadata配对，未做全像素审计") if pixel_review else "本次仅header/native配对，不生成重复预览；mask/bbox在数据生成阶段解码"
        report["review_frame_indices"] = list(plan["review_frame_indices"])
        report["pixel_review_performed"] = pixel_review
    mapping = {s+str(i): "Player_"+s+str(i) for s in "LR" for i in range(5)}
    for name, receipt in receipts.items():
        camera_report = {"frames": [], "motion": {}, "errors": [], "expected_camera_plan_checks": []}
        report["cameras"][name] = camera_report
        try:
            _require(not receipt["setup_error"] and not receipt["record_limit_reached"], "native setup错误或记录截断")
            rgb, exr = media_index(attempt/name, "rgb"), media_index(attempt/name, "exr")
            expected = set(range(plan["frame_count"]))
            _require(set(rgb) == set(exr) == expected, "RGB/EXR帧集合不符合显式计划")
            rows = receipt["records"]
            entities = {target["entity_id"] for target in plan["targets"]}
            _require(len(rows) == len(expected)*len(entities) and {(r["output_frame"],r["entity_id"]) for r in rows} == {(f,e) for f in expected for e in entities}, "native帧/实体集合不同或重复")
            camera_output = output/name
            camera_output.mkdir()
            for frame in sorted(expected):
                header = OpenEXR.File(str(exr[frame]), header_only=True).header()
                pixels_evaluated = not joint or (pixel_review and frame in plan["review_frame_indices"])
                mask = counts = None
                if pixels_evaluated:
                    manifest, ids = load_cryptomatte(exr[frame])
                    mask, counts = build_mask(manifest, ids, mapping)
                    Image.fromarray(mask).save(camera_output / ("%06d_mask.png" % frame))
                else:
                    manifest = cryptomatte_manifest_from_header(header)
                with Image.open(rgb[frame]) as rgb_header:
                    rgb_size = rgb_header.size
                overview_rows = []
                for row in [r for r in rows if r["output_frame"] == frame]:
                    camera = check_pair(row, header, name, frame, sampling=sampling_contexts[name])
                    camera_plan = next(c for c in plan["cameras"] if c["camera_id"]==name)
                    expected_view = check_expected_camera_plan(camera,camera_plan)
                    camera_report["expected_camera_plan_checks"].append(dict(output_frame=frame,**expected_view))
                    _require(not expected_view["applicable"] or expected_view["plan_camera_match_passed"],
                             "EXR实际相机位置/forward未执行近景计划")
                    _require(rgb_size == (camera["intrinsics"]["width"],camera["intrinsics"]["height"]), "RGB尺寸不同")
                    entity = row["entity_id"]
                    _require(mapping[entity] in manifest, "Cryptomatte缺少目标Actor")
                    source_method = row["sampling_method"] if sampling_contexts[name]["path"] == "gpu_render_cpu_lbs" else "native_actual_visual_mesh_diagnostic"
                    points = project_landmarks([{"world_m": p, "source_method": source_method} for p in row["world_m"]], camera)
                    mask_id = int(entity[1])+ (1 if entity[0]=="L" else 6)
                    bbox = None
                    if pixels_evaluated:
                        ys,xs = np.where(mask == mask_id)
                        bbox = [int(xs.min()),int(ys.min()),int(xs.max()-xs.min()+1),int(ys.max()-ys.min()+1)] if len(xs) else None
                    for point in points:
                        uv = point.get("uv_px")
                        point["instance_mask_hit_at_nearest_pixel"] = None
                        if pixels_evaluated and uv and point["visibility_state"] == "unknown":
                            x,y = int(round(uv[0])),int(round(uv[1]))
                            if 0 <= x < mask.shape[1] and 0 <= y < mask.shape[0]:
                                point["instance_mask_hit_at_nearest_pixel"] = bool(mask[y,x] == mask_id)
                    stem = "%06d_%s" % (frame,entity)
                    if not joint:
                        _draw_overlay(rgb[frame], points, bbox, camera_output/(stem+"_overlay.png"))
                    elif pixels_evaluated:
                        overview_rows.append({"entity_id":entity,"keypoints":points,"visible_instance_bbox_xywh":bbox})
                    camera_report["frames"].append({"output_frame": frame, "source_frame": row["sequence_source_frame"],
                        "entity_id": entity, "RGB": str(rgb[frame]), "EXR": str(exr[frame]), "camera": camera,
                        "keypoint_names": list(COCO17_NAMES), "keypoints": points,
                        "visible_instance_area_pixels": counts[entity] if pixels_evaluated else None,
                        "pixel_evaluation_status": "sampled_joint_overview" if joint and pixels_evaluated else "not_evaluated" if joint else "full_frame_diagnostic",
                        "visible_instance_bbox_xywh": bbox, "engine_frame": row["hook_engine_frame"],
                        "pose_revision": row["pose_revision_after"], "cache_and_metadata_checks_passed": True,
                        "pose_sampling_path": sampling_contexts[name]["path"],
                        "native_sampling_evidence": camera.get("native_sampling_evidence"),
                        "MRQ_cached_world_seconds": row.get("m_rq_world_seconds"),
                        "sample_world_seconds": row["world_time_seconds"],
                        "world_minus_MRQ_cached_seconds": (row["world_time_seconds"]-row["m_rq_world_seconds"])
                            if "m_rq_world_seconds" in row else None})
                if joint and pixels_evaluated:
                    _draw_joint_overview(rgb[frame], overview_rows, camera_output/("%06d_all_players_overlay.png" % frame))
            camera_report["motion"] = {e: pose_motion([r for r in rows if r["entity_id"] == e]) for e in sorted(entities)}
        except Exception as error:
            camera_report["errors"].append(repr(error))
            report["errors"].append(name + ": " + repr(error))
    report["structural_pairing_passed"] = not report["errors"]
    report["clock_interpretation"] = "MRQ TimeData.WorldSeconds为初始化时从UWorld取得、随后按FrameDeltaTime递增的独立缓存；采样WorldTimeSeconds为当前UWorld。两者绝对偏移单列，不能单独用作Sequence帧或RGB同步证明。实际缓存同步依据同引擎帧fence/revision及EXR源帧；尚未进行全表面像素级时序量化。"
    if report["pose_sampling_path"] == "gpu_render_cpu_lbs":
        report["gpu_lbs_scope"] = "实际GPU渲染保持不变；坐标由本帧骨pose/RefToLocal/实际权重CPU-LBS计算。fence是end-frame动态数据提交，不是CPU skin cache；共同帧/相机/资格门禁通过仍不证明最终GPU表面等价，5mm头深度及unknown可见性门禁独立保留。"
    # 只报告跨机位数值差异，不能用独立Sequence相同帧号假定同一动态状态。
    cross = {}
    for receipt in receipts.values():
        for row in receipt["records"]:
            cross.setdefault((row["sequence_source_frame"],row["entity_id"]), []).append(row)
    report["cross_camera_world_max_delta_m"] = [{"source_frame": f,"entity_id": e,
        "max_delta_m": max(math.dist(a,b) for r in rows[1:] for a,b in zip(rows[0]["world_m"],r["world_m"]))}
        for (f,e),rows in sorted(cross.items()) if len(rows)>1]
    (output/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("attempt", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--skip-review-pixels", action="store_true", help="复用已验联合通路，跳过重复抽样图；仍核对全部帧/身份/元数据")
    args = parser.parse_args()
    report = inspect(args.attempt,args.output,pixel_review=not args.skip_review_pixels)
    print(json.dumps({"structural_pairing_passed": report["structural_pairing_passed"], "MRQ_sync_verified": False,
                      "errors": report["errors"], "report": str(args.output/"report.json")},ensure_ascii=False))
    return 0 if report["structural_pairing_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
