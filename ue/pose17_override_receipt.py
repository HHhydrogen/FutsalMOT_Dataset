"""只读Player003→004诊断事务验收；不导入unreal，不把原生声明升级为图像/GT质量。"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import re

from pose17_native_receipt import normalize_native_receipt
from pose17_smoke_runner import _assert_gpu_lbs_receipt, _assert_override_receipt, validate_plan
from pose17_calibration_assets import calibration_visual_mesh, production_visual_mesh


OLD_MESH = "/Game/FutsalMOT/Characters/MPFB/Players/Player_003/MPFB_Player_003.MPFB_Player_003"
NEW_MESH = "/Game/FutsalMOT/Characters/MPFB/Players/Player_004/Mesh/MPFB_Player_004.MPFB_Player_004"
SKELETON = "/Game/FutsalMOT/Characters/MPFB/Core/Skeleton/SKEL_MPFB_Base.SKEL_MPFB_Base"
ANIM_CLASS = "/Game/FutsalMOT/Characters/MPFB/Core/Retarget/ABP_MPFB_RuntimeRetarget.ABP_MPFB_RuntimeRetarget_C"
NAMES = ["nose","left_eye","right_eye","left_ear","right_ear","left_shoulder","right_shoulder",
         "left_elbow","right_elbow","left_wrist","right_wrist","left_hip","right_hip","left_knee","right_knee","left_ankle","right_ankle"]


def _require(ok,message):
    if not ok: raise ValueError(message)


def _identity(value):
    """仅统一已知LOD缩写及JSON整数编码，不丢失路径或任何几何签名。"""
    output = dict(value)
    if "l_od" in output: output["lod"] = output.pop("l_od")
    for key in ("lod","vertex_count"):
        number = output.get(key)
        _require(type(number) is int or isinstance(number,str) and number.isdecimal(), "身份整数字段缺失/非法: "+key)
        output[key] = int(number)
    output = {k:output.get(k) for k in ("asset_path","signature_scheme","lod","vertex_count","topology_sha256","geometry_sha256")}
    _require(output["signature_scheme"]=="ue_lod_le_u32_f32_v1" and output["lod"]==0 and output["vertex_count"]>0,"身份签名方案/LOD/顶点数非法")
    for key in ("topology_sha256","geometry_sha256"):
        _require(isinstance(output[key],str) and re.fullmatch("[0-9a-f]{64}",output[key]) is not None,"缺少真实完整小写SHA256: "+key)
    return output


def _editor_mesh(report,target):
    _require(report["level"].split(".")[0]=="/Game/FutsalMOT/Maps/L_FutsalCourt"
        and not report["dirty_packages"] and not report["pie_worlds"] and report["mrq_running"] is False,"Editor Level/dirty/PIE/MRQ证据不合格")
    actors = [a for a in report["players"] if a["actor_name"]==target["actor_object_name"]]
    _require(len(actors)==1,"Editor没有唯一显式Actor")
    meshes = [m for m in actors[0]["meshes"] if m["component_name"]=="MPFBVisual"]
    _require(len(meshes)==1 and meshes[0]["asset"]==OLD_MESH,"Editor必须保留精确原Player003 MPFBVisual")
    return meshes[0]


def validate_bundle(plan, state, before, after, frozen_manifest, manifest_sha256, receipts):
    """receipts须已由native normalizer处理；逐相机失败独立留证，不自动修复/重试。"""
    report = {"schema":"futsalmot_player004_override_acceptance_v1","expected_old_mesh":OLD_MESH,
        "expected_new_mesh":NEW_MESH,"component_transaction_passed":False,"editor_postflight_passed":False,
        "streaming_restart_required":False,"safe_to_continue_without_restart":False,"acceptance_passed":False,
        "MRQ_sync_verified":False,"visibility_state":"unknown","cameras":{},"errors":[],"status":"failed_requires_diagnosis",
        "evidence_gaps":["材质/动画/恢复逐项读回由已执行原生代码的布尔和路径收据证明；未序列化全部恢复后PIE字段值，CPU不能独立重建UE资产状态",
            "sample未显式序列化CPU skin实际mode；成功路径原生门禁要求CPU缓存，不能冒充独立GPU/CPU A/B证据",
            "SkeletalMesh UObject class由原生强类型参数/runner检查约束，收据没有独立class字段",
            "这里不读取RGB/EXR，不证明实际最终camera projection、mask像素、头点语义、衣裤质量或精确COCO可见性"]}
    try:
        combined = plan.get("diagnostic_scope")=="pose17_gpu_render_cpu_lbs_character_calibration"
        new_mesh = calibration_visual_mesh(plan.get("calibration_player_id")) if combined else NEW_MESH
        report["expected_new_mesh"] = new_mesh
        report["calibration_player_id"] = plan.get("calibration_player_id") if combined else 4
        _require(plan.get("allow_pie_visual_mesh_override") is True and plan.get("diagnostic_scope") in
            ("pose17_character_calibration","pose17_gpu_render_cpu_lbs_character_calibration"),
            "计划没有明确PIE角色校准授权scope")
        _require(plan.get("allow_pie_cpu_preparation") is (False if combined else True) and plan.get("allow_pie_driver_hide") is True,"CPU/driver诊断授权缺失")
        warmup = plan.get("override_warmup_frames")
        _require(type(warmup) is int and 8<=warmup<=120,"warmup必须明确>=8")
        if combined:
            _require(plan.get("pose_sampling_path")=="gpu_render_cpu_lbs" and plan.get("allow_gpu_render_cpu_lbs_diagnostic") is True
                and type(plan.get("gpu_lbs_warmup_frames")) is int and plan["gpu_lbs_warmup_frames"]==warmup,"组合GPU/LBS授权或warmup不符")
            report["evidence_gaps"] = [gap for gap in report["evidence_gaps"] if "sample未显式序列化CPU" not in gap]
        _require(len(plan["targets"])==1,"当前仅验收一个角色override")
        target = plan["targets"][0]
        _require(target["entity_id"]=="L0" and target["component_object_name"]=="MPFBVisual"
            and target.get("diagnostic_visual_mesh_override")==new_mesh,"不是明确L0 MPFBVisual的精确白名单角色override")
        expected_identity = _identity(frozen_manifest.get("ue_identity",frozen_manifest.get("mesh_identity",{})))
        _require(expected_identity["asset_path"]==new_mesh,"冻结manifest与角色ID精确资产不同")
        _require(isinstance(manifest_sha256,str) and re.fullmatch("[0-9a-f]{64}",manifest_sha256),"冻结manifest SHA非法")
        cameras = [c["camera_id"] for c in plan["cameras"]]
        _require(len(cameras)>=2 and len(set(cameras))==len(cameras) and set(receipts)==set(cameras),"全部显式相机收据未齐或重复")
        expected_frames = set(range(plan["frame_count"]))
        start = plan["start_frame"]
        _require(state.get("executor_success") is True and not state.get("errors") and state.get("ready_to_render") is False,
            "executor未成功收尾或存在错误")
        _require(type(state.get("streaming_restart_required")) is bool,"state缺少明确streaming_restart_required")
        jobs = state.get("completed_jobs",[])
        _require(len(jobs)==len(cameras) and {j["job"] for j in jobs}==set(cameras)
            and all(j.get("sampling_passed") is True and j.get("restore_passed") is True for j in jobs),"异步相机采样/恢复结果不完整")
        old_mesh = _editor_mesh(before,target)
        restored_mesh = _editor_mesh(after,target)
        _require(old_mesh==restored_mesh and before["players"]==after["players"],"Editor mesh/动画/材质证据与原状态不同")
        _require(state.get("editor_scene_state_matched") is True and state.get("editor_mesh_state_matched") is True
            and state.get("editor_restore_probe_pending") is False,"独立postflight未成功")
        report["editor_postflight_passed"] = True
    except (ValueError,KeyError,TypeError) as error:
        report["errors"].append("bundle: "+repr(error))
        return report
    shared_new_materials = None
    for camera_id in cameras:
        result = {"passed":False,"errors":[]}
        report["cameras"][camera_id] = result
        try:
            receipt = receipts[camera_id]
            _require(receipt["schema"]=="futsalmot_pose17_native_receipt_v1"
                and receipt["lifecycle_stage"]=="pipeline_teardown_before_PIE_cleanup","native收据schema/生命周期不符")
            _require(receipt.get("visual_mesh_override_requested") is True and receipt.get("diagnostic_scope")==plan["diagnostic_scope"],
                "native没有执行明确override scope")
            if combined:
                _assert_gpu_lbs_receipt(plan,receipt)
                _assert_override_receipt(plan,receipt)
            _require(not receipt.get("setup_error") and not receipt.get("pipeline_setup_error") and not receipt.get("restore_errors")
                and receipt.get("record_limit_reached") is False and receipt.get("camera_state_restored") is True,"native setup/restore不合格")
            evidence = receipt.get("visual_mesh_overrides",[])
            _require(len(evidence)==1,"native override事务数不符")
            evidence = evidence[0]
            _require(evidence["original_asset_path"]==OLD_MESH and evidence["applied_asset_path"]==new_mesh,
                "原/新mesh并非精确003→指定白名单角色")
            _require(evidence["shared_skeleton_path"]==SKELETON and evidence["animation_class_path"]==ANIM_CLASS,
                "不是精确共享骨架及AnimBP")
            component = evidence["component_path"]
            _require("/UEDPIE_" in component and component.endswith("."+target["actor_object_name"]+".MPFBVisual"),
                "native目标不是精确PIE Actor/MPFBVisual")
            _require(evidence["original_animation_instance_path"].startswith(component+"."),"原动画实例路径不属于该组件")
            for key in ("identity_verified","apply_succeeded","animation_binding_preserved","new_asset_materials_verified","restore_attempted","restore_succeeded"):
                _require(evidence.get(key) is True,"原生事务证据缺失/失败: "+key)
            _require(not evidence.get("apply_error") and not evidence.get("restore_error"),"原生override有partial错误")
            original_materials = [m or "" for m in old_mesh["override_material_paths"]]
            # 只有PIE namespace可机械规范化，其他未知差异不能假定是同一材质。
            canonical = lambda path:re.sub(r"UEDPIE_\d+_","",path)
            _require([canonical(p) for p in evidence["original_override_material_paths"]]==[canonical(p) for p in original_materials],
                "PIE原override材质路径与Editor快照不能一致对应")
            new_materials = evidence["applied_asset_material_paths"]
            _require(isinstance(new_materials,list) and new_materials and all(isinstance(p,str) and p.startswith("/") for p in new_materials),
                "新mesh自身材质slots未完整记录")
            if shared_new_materials is None: shared_new_materials = new_materials
            _require(new_materials==shared_new_materials,"相机之间新mesh材质slots不同")
            streaming = evidence.get("streaming_state_matched")
            restart = receipt.get("streaming_state_restart_required")
            _require(type(streaming) is bool and type(restart) is bool,"streaming结果缺少明确布尔")
            _require(streaming or restart,"streaming未恢复却未声明restart_required")
            job = next(j for j in jobs if j["job"]==camera_id)
            _require(job.get("streaming_state_restart_required") is restart,"state与native streaming重启需求不同")
            result["streaming_status"] = "matched" if streaming and not restart else "restart_required"
            if restart or not streaming: report["streaming_restart_required"] = True
            records = receipt["records"]
            _require(len(records)==len(expected_frames) and {r["output_frame"] for r in records}==expected_frames,"native记录不完整/重复帧")
            applied_frame = evidence["applied_engine_frame"]
            _require(type(applied_frame) is int and applied_frame>=0,"应用engine frame缺失")
            for record in records:
                _require(record["entity_id"]==target["entity_id"] and record["component_path"]==component
                    and record["camera_name"]==camera_id,"record实际角色/组件/相机身份不同")
                _require(record["sample_succeeded"] is True and not record.get("error") and record["keypoint_names"]==NAMES,
                    "实际record未成功采样COCO17")
                _require(_identity(record["mesh_identity"])==expected_identity and record["manifest_sha256"]==manifest_sha256,
                    "record实际新mesh完整identity/manifest SHA不同")
                _require(record["sequence_source_frame"]==start+record["output_frame"],"源帧与显式小样范围不同")
                _require(record.get("cine_camera_look_at_state_observed") is True and record.get("cine_camera_look_at_enabled") is False,
                    "没有实际观测camera lookat=false，不能拿计划值冒充运行时")
                elapsed = record.get("override_warmup_engine_frames_observed")
                _require(type(elapsed) is int and elapsed>=warmup and elapsed==record["hook_engine_frame"]-applied_frame,
                    "实际warmup不足或与应用/采样engine frame不一致")
                _require(record["cache_fence_completed_this_engine_frame"] is True and record["local_read_stable"] is True
                    and record["engine_frame_before"]==record["engine_frame_after"]==record["hook_engine_frame"]
                    and record["pose_revision_before"]==record["pose_revision_after"],"当前帧CPU缓存/读取证据不一致")
                _require(record["temporal_sample_count"]==record["spatial_sample_count"]==1 and record["motion_blur_fraction"]==0,
                    "采样参数不同")
                _require(record.get("mrq_synchronization_verified") is False,"不得升级正式MRQ同步")
                _require(len(record["world_m"])==17 and all(len(p)==3 and all(math.isfinite(v) for v in p) for p in record["world_m"]),
                    "世界17点无效")
                _require(record.get("actor_world_transform_ue_cm") and record.get("visual_world_transform_ue_cm"),"缺少Actor/Visual实际变换证据")
            result.update(passed=True,records=len(records),new_asset_material_paths=new_materials,
                native_animation_material_restore_assertions_passed=True,actual_camera_lookat_false_all_records=True)
        except (ValueError,KeyError,TypeError,RuntimeError) as error:
            result["errors"].append(repr(error))
            report["errors"].append(camera_id+": "+repr(error))
    if not report["errors"]:
        maximum = 0.0
        for frame in sorted(expected_frames):
            views = [next(r for r in receipts[c]["records"] if r["output_frame"]==frame)["world_m"] for c in cameras]
            maximum = max(maximum,max(math.dist(a,b) for view in views[1:] for a,b in zip(views[0],view)))
        report["cross_view_world_max_delta_m"] = maximum
        if maximum>1e-5: report["errors"].append("跨机位同源帧世界Pose差超过预注册1e-5m，需独立诊断动态状态")
    report["component_transaction_passed"] = not report["errors"] and report["editor_postflight_passed"]
    if state["streaming_restart_required"] != report["streaming_restart_required"]:
        report["errors"].append("state与native聚合streaming重启需求不同")
        report["component_transaction_passed"] = False
    report["safe_to_continue_without_restart"] = report["component_transaction_passed"] and not report["streaming_restart_required"]
    report["acceptance_passed"] = report["safe_to_continue_without_restart"]
    report["status"] = ("native_component_switch_accepted_requires_visual_pairing" if report["acceptance_passed"] else
        "native_component_switch_passed_streaming_restart_required" if report["component_transaction_passed"] else "failed_requires_diagnosis")
    return report


def validate_joint_bundle(plan,state,before,after,manifests,manifest_hashes,receipts):
    """联合小样只做一次自动事务/身份/帧门禁；不插逐角色渲染或像素审计。"""
    report = dict(schema="futsalmot_joint_pose17_native_acceptance_v1",acceptance_passed=False,
        component_transaction_passed=False,streaming_restart_required=False,MRQ_sync_verified=False,visibility_state="unknown",errors=[])
    try:
        validate_plan(plan)
        _require(plan["diagnostic_scope"]=="pose17_gpu_render_cpu_lbs_joint_diagnostic","不是明确联合诊断scope")
        _require(state.get("executor_success") is True and not state.get("errors") and state.get("ready_to_render") is False,
            "联合executor未成功收尾")
        for editor in (before,after):
            _require(editor["level"].split(".")[0]=="/Game/FutsalMOT/Maps/L_FutsalCourt" and not editor["dirty_packages"]
                and not editor["pie_worlds"] and editor["mrq_running"] is False,"联合Editor前后门禁失败")
        _require(before["players"]==after["players"] and state.get("editor_scene_state_matched") is True
            and state.get("editor_mesh_state_matched") is True and state.get("editor_restore_probe_pending") is False,"联合Editor恢复不一致")
        cameras = {c["camera_id"] for c in plan["cameras"]}
        jobs = state["completed_jobs"]
        _require(set(receipts)==cameras and len(jobs)==4 and {j["job"] for j in jobs}==cameras
            and all(j.get("sampling_passed") is True and j.get("restore_passed") is True for j in jobs),"四相机原生收据未齐")
        targets = {t["entity_id"]:t for t in plan["targets"]}
        identities = {entity:_identity(manifests[entity].get("ue_identity",manifests[entity].get("mesh_identity",{}))) for entity in targets}
        originals = {}
        for entity,target in targets.items():
            _require(identities[entity]["asset_path"]==production_visual_mesh(target["calibration_player_id"]),"联合manifest与精确角色不同")
            actor = [a for a in before["players"] if a["actor_name"]==target["actor_object_name"]]
            _require(len(actor)==1,"联合原Actor不唯一")
            mesh = [m for m in actor[0]["meshes"] if m["component_name"]=="MPFBVisual"]
            _require(len(mesh)==1,"联合原Visual不唯一")
            originals[entity] = mesh[0]["asset"]
        expected_keys = {(frame,entity) for frame in range(plan["frame_count"]) for entity in targets}
        for camera_id,receipt in receipts.items():
            _require(receipt["lifecycle_stage"]=="pipeline_teardown_before_PIE_cleanup" and not receipt.get("setup_error")
                and not receipt.get("pipeline_setup_error") and not receipt.get("restore_errors")
                and receipt.get("record_limit_reached") is False and receipt.get("camera_state_restored") is True,"联合setup/restore失败")
            _assert_gpu_lbs_receipt(plan,receipt)
            _assert_override_receipt(plan,receipt)
            rows = receipt["records"]
            _require(len(rows)==len(expected_keys) and {(r["output_frame"],r["entity_id"]) for r in rows}==expected_keys,"联合实体/帧不完整或重复")
            for row in rows:
                entity = row["entity_id"]
                component_suffix = "."+targets[entity]["actor_object_name"]+".MPFBVisual"
                _require("/UEDPIE_" in row["component_path"] and row["component_path"].endswith(component_suffix)
                    and row["camera_name"]==camera_id and row["sequence_source_frame"]==plan["start_frame"]+row["output_frame"],"联合角色/相机/源帧不同")
                _require(_identity(row["mesh_identity"])==identities[entity] and row["manifest_sha256"]==manifest_hashes[entity],"联合实际身份/manifest SHA不同")
                _require(row["cache_fence_completed_this_engine_frame"] is True and row["local_read_stable"] is True
                    and row["engine_frame_before"]==row["engine_frame_after"]==row["hook_engine_frame"]
                    and row["pose_revision_before"]==row["pose_revision_after"] and not row.get("error"),"联合同帧fence/revision失败")
                _require(row["keypoint_names"]==NAMES and len(row["world_m"])==17 and row.get("mrq_synchronization_verified") is False
                    and all(len(p)==3 and all(math.isfinite(v) for v in p) for p in row["world_m"]),"联合17点契约失败")
            for evidence in receipt["visual_mesh_overrides"]:
                entity = next(e for e,t in targets.items() if evidence["component_path"].endswith("."+t["actor_object_name"]+".MPFBVisual"))
                _require(evidence["original_asset_path"]==originals[entity] and evidence["shared_skeleton_path"]==SKELETON
                    and evidence["animation_class_path"]==ANIM_CLASS,"联合原mesh/共享动画收据不同")
                _require(evidence["applied_asset_material_paths"] and evidence.get("streaming_state_matched") is True,
                    "联合材质/streaming恢复未通过")
            restart = receipt.get("streaming_state_restart_required")
            _require(type(restart) is bool,"联合streaming缺少明确布尔")
            report["streaming_restart_required"] |= restart
        _require(state.get("streaming_restart_required") is report["streaming_restart_required"],"联合streaming汇总不同")
        report["component_transaction_passed"] = True
        report["acceptance_passed"] = not report["streaming_restart_required"]
        report["records_per_camera"] = len(expected_keys)
    except (ValueError,KeyError,TypeError,RuntimeError,StopIteration) as error:
        report["errors"].append(repr(error))
    report["status"] = "joint_native_transaction_accepted_requires_pairing" if report["acceptance_passed"] else "joint_native_requires_diagnosis"
    return report


def inspect(attempt, output):
    attempt,output = Path(attempt).resolve(),Path(output).resolve()
    _require(not output.exists(),"验收输出已存在，不覆盖")
    read = lambda name:json.loads((attempt/name).read_text(encoding="utf-8"))
    raw_plan = (attempt/"plan.json").read_bytes()
    plan,state = json.loads(raw_plan),read("state.json")
    _require(hashlib.sha256(raw_plan).hexdigest()==state["plan_sha256"],"冻结plan SHA与state不同")
    receipts = {c["camera_id"]:normalize_native_receipt(read(c["camera_id"]+"_native_pose_restore_receipt.json")) for c in plan["cameras"]}
    if plan.get("diagnostic_scope")=="pose17_gpu_render_cpu_lbs_joint_diagnostic":
        raw_manifests = {t["entity_id"]:(attempt/(t["entity_id"]+"_landmark_snapshot.json")).read_bytes() for t in plan["targets"]}
        report = validate_joint_bundle(plan,state,read("preflight.json"),read("editor_after.json"),
            {k:json.loads(v) for k,v in raw_manifests.items()},{k:hashlib.sha256(v).hexdigest() for k,v in raw_manifests.items()},receipts)
    else:
        raw_manifest = (attempt/"L0_landmark_snapshot.json").read_bytes()
        report = validate_bundle(plan,state,read("preflight.json"),read("editor_after.json"),json.loads(raw_manifest),
            hashlib.sha256(raw_manifest).hexdigest(),receipts)
    report["attempt"] = str(attempt)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open("x",encoding="utf-8") as stream: json.dump(report,stream,ensure_ascii=False,indent=2,allow_nan=False)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("attempt",type=Path)
    parser.add_argument("output",type=Path)
    args = parser.parse_args()
    result = inspect(args.attempt,args.output)
    print(json.dumps({k:result.get(k) for k in ("status","component_transaction_passed","editor_postflight_passed",
        "streaming_restart_required","acceptance_passed","errors")},ensure_ascii=False))
    return 0 if result["acceptance_passed"] else 2 if result["component_transaction_passed"] else 1


if __name__ == "__main__": raise SystemExit(main())
