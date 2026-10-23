"""独立 Pose17 MRQ 小样准备；默认只读 preflight，绝不隐式提交渲染。

UE 内设置 FUTSAL_POSE17_SMOKE_PLAN 为计划 JSON 路径（可选），通过真实 MCP
run_python_file 执行。FUTSAL_POSE17_SMOKE_ACTION 默认 preflight；prepare 仅准备
独立Queue与新attempt；只有显式submit动作才提交。无取消/清空已有队列能力。
"""

import hashlib
import json
import os
import sys
import types
from pathlib import Path
from pose17_calibration_assets import ORIGINAL_VISUAL_MESH, JOINT_VISUAL_IDS, calibration_visual_mesh, production_visual_mesh


_PREPARED = None  # 持有 transient 队列；不会冒充已经执行的 MRQ。
_LEVEL = "/Game/FutsalMOT/Maps/L_FutsalCourt"
_SESSION = "futsalmot_pose17_smoke_runtime"


def validate_plan(plan):
    """计划只允许小样：原生4K、至少3帧/两个显式源序列，不改正式 v1。"""
    if plan.get("schema") != "futsalmot_pose17_smoke_plan_v1":
        raise ValueError("需要显式 Pose17 smoke plan schema")
    name = plan.get("attempt")
    if not isinstance(name, str) or not name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in name):
        raise ValueError("attempt 必须为安全独立目录名")
    start, count = plan.get("start_frame"), plan.get("frame_count")
    if type(start) is not int or type(count) is not int or not 3 <= count <= 30:
        raise ValueError("需要明确起始帧与3..30个连续动态诊断帧")
    cameras = plan.get("cameras", [])
    if len(cameras) < 2 or len({item["camera_id"] for item in cameras}) != len(cameras):
        raise ValueError("至少两个不同机位")
    for camera in cameras:
        camera_id = camera.get("camera_id", "")
        if not camera_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in camera_id):
            raise ValueError("camera_id 必须为安全相对目录名")
        path = camera.get("source_sequence")
        if not isinstance(path, str) or not path.startswith("/Game/") or ".." in path or "\\" in path:
            raise ValueError("每机位须显式源Sequence，不猜测 canonical 安全")
        if not isinstance(camera.get("camera_actor_object_name"), str) or not camera["camera_actor_object_name"]:
            raise ValueError("每机位须显式PIE相机Actor对象名以准备零motionblur")
    if len({item["source_sequence"] for item in cameras}) != len(cameras):
        raise ValueError("每机位单独Sequence，禁止将同源Sequence伪装两个视角")
    if not plan.get("targets"):
        raise ValueError("必须显式绑定实体/Actor/实际 VisualMesh")
    sampling_path = plan.get("pose_sampling_path","cpu_skin_cache")
    overrides = [row for row in plan["targets"] if row.get("diagnostic_visual_mesh_override") is not None]
    combined = plan.get("diagnostic_scope") == "pose17_gpu_render_cpu_lbs_character_calibration"
    joint = plan.get("diagnostic_scope") == "pose17_gpu_render_cpu_lbs_joint_diagnostic"
    if sampling_path not in ("cpu_skin_cache","gpu_render_cpu_lbs"):
        raise ValueError("未知采样路径，禁止静默回退")
    if sampling_path == "gpu_render_cpu_lbs":
        if (plan.get("allow_gpu_render_cpu_lbs_diagnostic") is not True
                or plan.get("diagnostic_scope") not in ("pose17_gpu_render_cpu_lbs","pose17_gpu_render_cpu_lbs_character_calibration","pose17_gpu_render_cpu_lbs_joint_diagnostic")
                or plan.get("allow_pie_cpu_preparation") is not False or plan.get("allow_pie_driver_hide") is not True
                or (not combined and not joint and (plan.get("allow_pie_visual_mesh_override") or overrides))):
            raise ValueError("GPU-render/LBS需要明确opt-in/scope/CPU关闭/隐藏driver及独立角色组合授权")
        warmup = plan.get("gpu_lbs_warmup_frames")
        if type(warmup) is not int or not 8 <= warmup <= 120:
            raise ValueError("新GPU-render/LBS需要8..120帧warmup，不通过去掉预热规避问题")
    elif plan.get("allow_gpu_render_cpu_lbs_diagnostic"):
        raise ValueError("GPU/LBS授权与采样路径不一致")
    if overrides or plan.get("allow_pie_visual_mesh_override"):
        if (plan.get("allow_pie_visual_mesh_override") is not True or len(overrides) != (8 if joint else 1)
                or plan.get("diagnostic_scope") not in ("pose17_character_calibration","pose17_gpu_render_cpu_lbs_character_calibration","pose17_gpu_render_cpu_lbs_joint_diagnostic")
                or plan.get("allow_pie_cpu_preparation") is not (False if combined or joint else True) or plan.get("allow_pie_driver_hide") is not True):
            raise ValueError("角色override必须显式PIE诊断scope/opt-in/CPU/隐藏driver，当前仅一次一个角色")
        warmup = plan.get("override_warmup_frames")
        if type(warmup) is not int or not 8 <= warmup <= 120:
            raise ValueError("角色override需要明确8..120帧warmup")
        path = overrides[0]["diagnostic_visual_mesh_override"]
        if (not isinstance(path,str) or not path.startswith("/Game/FutsalMOT/Characters/MPFB/Players/Player_00")
                or ".." in path or "\\" in path or path.rsplit(".",1)[-1] not in {"MPFB_Player_%03d"%i for i in range(3,10)}):
            raise ValueError("本轮override仅允许显式生产Player003至009 SkeletalMesh对象路径")
        if overrides[0].get("component_object_name") != "MPFBVisual":
            raise ValueError("override必须绑定实际MPFBVisual，禁止驱动底模")
    if combined:
        expected_mesh = calibration_visual_mesh(plan.get("calibration_player_id"))
        if (sampling_path != "gpu_render_cpu_lbs" or plan.get("allow_gpu_render_cpu_lbs_diagnostic") is not True
                or plan.get("allow_pie_visual_mesh_override") is not True or len(overrides)!=1 or len(plan["targets"])!=1
                or overrides[0]["diagnostic_visual_mesh_override"] != expected_mesh
                or overrides[0].get("entity_id") != "L0"
                or plan.get("override_warmup_frames") != plan.get("gpu_lbs_warmup_frames")):
            raise ValueError("GPU/LBS角色组合仅单L0白名单004..009双opt-in，两个warmup字段必须相等")
    if joint:
        if (plan.get("allow_joint_pose17_diagnostic") is not True or sampling_path!="gpu_render_cpu_lbs"
                or plan.get("allow_pie_visual_mesh_override") is not True or len(overrides)!=8
                or plan.get("override_warmup_frames")!=plan.get("gpu_lbs_warmup_frames")
                or len(plan["targets"])!=10 or {row.get("entity_id") for row in plan["targets"]}!=set(JOINT_VISUAL_IDS)
                or len({row.get("actor_object_name") for row in plan["targets"]})!=10
                or len(cameras)!=4):
            raise ValueError("联合GPU诊断需要显式授权、十实体/四相机/八override及相同warmup")
        for row in plan["targets"]:
            player_id = row.get("calibration_player_id")
            if (type(player_id) is not int or player_id!=JOINT_VISUAL_IDS[row["entity_id"]]
                    or row.get("component_object_name")!="MPFBVisual"
                    or row.get("sequence_binding_name")!="Player_"+row["entity_id"]
                    or row.get("diagnostic_visual_mesh_override")!=(None if player_id==3 else production_visual_mesh(player_id))):
                raise ValueError("联合实体/角色ID/精确asset/显式Sequence绑定不符合固定映射")
    elif plan.get("allow_joint_pose17_diagnostic"):
        raise ValueError("联合opt-in不能套用其他scope")


def _mesh_report(unreal, actor):
    result = []
    for mesh in actor.get_components_by_class(unreal.SkeletalMeshComponent):
        asset = mesh.get_editor_property("skeletal_mesh")
        anim_class = mesh.get_editor_property("anim_class")
        result.append({"component_name": mesh.get_name(), "path": mesh.get_path_name(),
                       "visible": bool(mesh.is_visible()), "hidden_in_game": bool(mesh.get_editor_property("hidden_in_game")),
                       "asset": asset.get_path_name() if asset else None,
                       "forced_lod_model": int(mesh.get_editor_property("forced_lod_model")),
                       "animation_class_path": anim_class.get_path_name() if anim_class else None,
                       "override_material_paths": [material.get_path_name() if material else None
                           for material in mesh.get_editor_property("override_materials")]})
        reader = getattr(getattr(unreal, "FutsalPose17Library", None), "get_visual_sampling_state", None)
        if mesh.get_name() == "MPFBVisual" and reader is not None:
            state_result = reader(mesh)
            if state_result is not None:
                state, error = state_result
                result[-1]["cpu_state"] = {name: getattr(state, name) for name in
                    ("cpu_enabled", "force_rendered_lod", "force_streamed_lod", "asset_was_streamable")} if not error else {"error": str(error)}
    return result


def _sequence_report(unreal, path):
    if not unreal.EditorAssetLibrary.does_asset_exist(path):
        return {"path": path, "error": "源资产不存在"}
    sequence = unreal.load_asset(path)
    if not isinstance(sequence, unreal.LevelSequence):
        return {"path": path, "error": "源资产不是 LevelSequence"}
    rate = sequence.get_display_rate()
    bindings = []
    for binding in sequence.get_bindings():
        tracks = []
        for track in binding.get_tracks():
            track_class = track.get_class().get_name()
            numeric_channels = []
            if track_class in ("MovieScene3DTransformTrack", "MovieSceneDoubleTrack", "MovieSceneFloatTrack"):
                for section in track.get_sections():
                    for channel in section.get_all_channels():
                        keys = channel.get_keys()
                        if keys and all(callable(getattr(key, "get_value", None)) for key in keys):
                            values = []
                            for key in keys:
                                value = key.get_value()
                                if type(value) not in (int, float):
                                    continue
                                time = key.get_time()
                                values.append({"frame": float(time.frame_number.value) + float(time.sub_frame), "value": float(value)})
                            numeric_channels.append({"name": str(channel.channel_name), "keys": values})
            tracks.append({"class": track_class, "sections": len(track.get_sections()), "numeric_channels": numeric_channels})
        bindings.append({"name": str(binding.get_display_name()), "tracks": tracks})
    # UE5.8 SequencerScripting 已将 master tracks 改名为 tracks；只用实际存在的入口。
    top_tracks = sequence.get_tracks() if callable(getattr(sequence, "get_tracks", None)) else None
    return {"path": path, "playback_start": int(sequence.get_playback_start()),
            "playback_end": int(sequence.get_playback_end()), "display_rate": [int(rate.numerator), int(rate.denominator)],
            "bindings": bindings, "top_tracks": [track.get_class().get_name() for track in top_tracks] if top_tracks is not None else None,
            "source_dynamic_pose_verified": False}


def preflight(plan=None):
    """只读当前Editor、反射API、实际网格与显式源Sequence。"""
    import unreal
    world = unreal.EditorLevelLibrary.get_editor_world()
    subsystem = unreal.get_editor_subsystem(unreal.MoviePipelineQueueSubsystem)
    actors = unreal.EditorLevelLibrary.get_all_level_actors()
    report = {
        "schema": "futsalmot_pose17_smoke_preflight_v1", "action": "preflight_read_only",
        "level": world.get_path_name(),
        "dirty_packages": [package.get_path_name() for package in list(unreal.EditorLoadingAndSavingUtils.get_dirty_content_packages())
                           + list(unreal.EditorLoadingAndSavingUtils.get_dirty_map_packages())],
        "pie_worlds": [item.get_path_name() for item in unreal.EditorLevelLibrary.get_pie_worlds(False)],
        "mrq_running": bool(subsystem.is_rendering()),
        "queue_job_count": len(subsystem.get_queue().get_jobs()),
        "pose_library": hasattr(unreal, "FutsalPose17Library"), "pose_pass": hasattr(unreal, "FutsalPose17MRQPass"),
        "filtered_png_output": hasattr(unreal, "FutsalMoviePipelinePNGOutput"),
        "players": [{"actor_name": actor.get_name(), "label": actor.get_actor_label(), "meshes": _mesh_report(unreal, actor)}
                    for actor in actors if actor.get_actor_label().startswith("Player_")],
        "source_sequences": [], "api": {}, "ready_to_render": False,
    }
    console_reader = getattr(unreal.SystemLibrary, "get_console_variable_int_value", None)
    report["skeletal_force_lod_cvar"] = int(console_reader("r.SkeletalMeshForceLOD")) if console_reader else None
    for name in ("MoviePipelineQueue", "MoviePipelinePIEExecutor", "MoviePipelineOutputSetting",
                 "MoviePipelineImageSequenceOutput_EXR", "FutsalMoviePipelinePNGOutput", "MoviePipelineObjectIdRenderPass", "FutsalPose17Library"):
        cls = getattr(unreal, name, None)
        report["api"][name] = {"present": cls is not None}
        if cls:
            for method in ("execute", "read_asset_reference_geometry", "get_visual_sampling_state", "set_visual_cpu_sampling"):
                value = getattr(cls, method, None)
                if value is not None:
                    report["api"][name][method] = str(getattr(value, "__doc__", ""))
    if plan is not None:
        validate_plan(plan)
        for item in plan["cameras"]:
            report["source_sequences"].append(_sequence_report(unreal, item["source_sequence"]))
        try:
            _assert_prepare_gate(plan, report)
            report["plan_gate_passed"] = True
        except (ValueError, RuntimeError) as error:
            report["plan_gate_passed"] = False
            report["plan_gate_error"] = str(error)
    return report


def _assert_prepare_gate(plan, report):
    if report["level"].split(".")[0] != _LEVEL or report["dirty_packages"] or report["pie_worlds"] or report["mrq_running"]:
        raise RuntimeError("地图/dirty/PIE/MRQ门禁未通过，不准备新队列")
    if not report["pose_library"] or not report["pose_pass"]:
        raise RuntimeError("新模块反射未加载")
    if report.get("filtered_png_output") is not True:
        raise RuntimeError("已验证的FinalImage专用PNG输出类未加载，不静默回退")
    players = {player["actor_name"]: player for player in report["players"]}
    for target in plan["targets"]:
        actor = players.get(target["actor_object_name"])
        meshes = [mesh for mesh in actor["meshes"] if mesh["component_name"] == target["component_object_name"]] if actor else []
        if len(meshes) != 1 or not meshes[0]["visible"] or meshes[0]["hidden_in_game"] or target["component_object_name"] == "CharacterMesh0":
            raise RuntimeError("目标不是唯一实际可见 VisualMesh")
    for sequence in report["source_sequences"]:
        if sequence.get("error") or sequence.get("top_tracks") is None or "MovieSceneCameraCutTrack" not in sequence["top_tracks"]:
            raise RuntimeError("源序列未通过 class/CameraCut 只读门禁")
        if not sequence["playback_start"] <= plan["start_frame"] < plan["start_frame"] + plan["frame_count"] <= sequence["playback_end"]:
            raise RuntimeError("小样帧范围超出显式源序列")
        if not any(any(track["class"] in ("MovieSceneSkeletalAnimationTrack", "MovieScene3DTransformTrack")
                       for track in binding["tracks"]) for binding in sequence["bindings"]):
            raise RuntimeError("源序列无动态轨道证据；轨道存在仍不证明实际三个动作帧")
        for target in plan["targets"]:
            name = target.get("sequence_binding_name", "Player_" + target["entity_id"])
            bindings = [binding for binding in sequence["bindings"] if binding["name"] == name]
            if len(bindings) != 1:
                raise RuntimeError("源Sequence没有唯一目标绑定: " + name)
            changing = []
            for track in bindings[0]["tracks"]:
                for channel in track.get("numeric_channels", []):
                    keys = [key for key in channel["keys"] if plan["start_frame"] <= key["frame"] < plan["start_frame"] + plan["frame_count"]]
                    if len(keys) >= 2 and max(key["value"] for key in keys) - min(key["value"] for key in keys) > 1e-6:
                        changing.append(channel["name"])
            if not changing:
                raise RuntimeError("目标绑定在选定范围未发现变化数值关键帧，仅轨道名不足: " + name)


def _native_targets(unreal, plan):
    """将已绑定实际UE拓扑的manifest转反射输入，未验收候选必须显式诊断授权。"""
    actors = unreal.EditorLevelLibrary.get_all_level_actors()
    targets, snapshots = [], {}
    for row in plan["targets"]:
        source = Path(row["landmark_manifest"])
        raw = source.read_bytes()
        manifest = json.loads(raw)
        accepted = manifest.get("calibration", {}).get("status") == "accepted"
        if not accepted and plan.get("allow_unaccepted_landmarks_for_diagnostic") is not True:
            raise RuntimeError("头部候选未验收，仅显式诊断计划允许渲染，不允许称正式真值")
        actor_matches = [actor for actor in actors if actor.get_name() == row["actor_object_name"]]
        if len(actor_matches) != 1:
            raise RuntimeError("无法匹配manifest目标的精确Actor")
        matches = [mesh for mesh in actor_matches[0].get_components_by_class(unreal.SkeletalMeshComponent)
                   if mesh.get_name() == row["component_object_name"]]
        if len(matches) != 1:
            raise RuntimeError("无法匹配manifest目标的实际VisualMesh")
        if plan.get("diagnostic_scope")=="pose17_gpu_render_cpu_lbs_character_calibration":
            current_asset = matches[0].get_editor_property("skeletal_mesh")
            if current_asset is None or current_asset.get_path_name()!=ORIGINAL_VISUAL_MESH:
                raise RuntimeError("组合角色校准的Editor原mesh必须为精确Player003路径")
        override_path = row.get("diagnostic_visual_mesh_override")
        override_asset = None
        if override_path:
            override_asset = unreal.load_asset(override_path)
            if not isinstance(override_asset,unreal.SkeletalMesh):
                raise RuntimeError("override不是实际SkeletalMesh资产")
            result = unreal.FutsalPose17Library.read_asset_reference_geometry(override_asset)
            if result is None:
                raise RuntimeError("override资产真实LOD0身份读取失败")
            _positions, _triangles, identity, error = result
        else:
            result = unreal.FutsalPose17Library.describe_visual_mesh(matches[0])
            if result is None:
                raise RuntimeError("实际VisualMesh身份读取失败")
            identity, error = result
        if error:
            raise RuntimeError(error)
        expected = manifest.get("ue_identity", manifest.get("mesh_identity", {}))
        if plan.get("diagnostic_scope")=="pose17_gpu_render_cpu_lbs_character_calibration":
            if expected.get("asset_path")!=calibration_visual_mesh(plan["calibration_player_id"]):
                raise RuntimeError("manifest身份与显式校准角色ID的精确资产不同")
        if plan.get("diagnostic_scope")=="pose17_gpu_render_cpu_lbs_joint_diagnostic":
            if expected.get("asset_path")!=production_visual_mesh(row["calibration_player_id"]):
                raise RuntimeError("联合manifest身份与实体角色精确资产不同")
        for key in ("asset_path", "signature_scheme", "lod", "vertex_count", "geometry_sha256", "topology_sha256"):
            if str(getattr(identity, key)) != str(expected.get(key)):
                raise RuntimeError("manifest不是当前UE导入拓扑: " + key)
        anchors = []
        for name in ("nose", "left_eye", "right_eye", "left_ear", "right_ear"):
            point = manifest["anchors"][name]
            anchor = unreal.FutsalPose17SurfaceAnchor()
            anchor.set_editor_property("keypoint", name)
            anchor.set_editor_property("triangle_vertices", unreal.IntVector(*point["triangle_vertices"]))
            anchor.set_editor_property("barycentric", unreal.Vector(*point["barycentric"]))
            anchor.set_editor_property("semantic_evidence_ref", str(source))
            anchors.append(anchor)
        target = unreal.FutsalPose17MRQTarget()
        for name in ("entity_id", "actor_object_name", "component_object_name"):
            target.set_editor_property(name, row[name])
        target.set_editor_property("expected_identity", identity)
        target.set_editor_property("head_anchors", anchors)
        target.set_editor_property("manifest_sha256", hashlib.sha256(raw).hexdigest())
        if override_asset is not None:
            target.set_editor_property("diagnostic_visual_mesh_override", override_asset)
        targets.append(target)
        snapshots[row["entity_id"]] = raw
    return targets, snapshots


def prepare(plan):
    """仅构造独立内存队列与无覆盖attempt，不执行MRQ或改现有queue。"""
    import unreal
    global _PREPARED
    if _PREPARED is not None or getattr(sys.modules.get(_SESSION), "prepared", None) is not None:
        raise RuntimeError("该模块已持有准备队列，不重复构造")
    report = preflight(plan)
    _assert_prepare_gate(plan, report)
    targets, snapshots = _native_targets(unreal, plan)
    # 当前阶段只冻结计划与门禁，后续 CPU 状态准备/恢复和校准需额外实际验收。
    root = Path(unreal.Paths.project_dir()).resolve() / "Saved/Pose17Development/smoke"
    attempt = root / plan["attempt"]
    if attempt.exists():
        raise RuntimeError("attempt 已存在，拒绝覆盖")
    attempt.mkdir(parents=True, exist_ok=False)
    encoded = json.dumps(plan, ensure_ascii=False, sort_keys=True, allow_nan=False).encode("utf-8")
    (attempt / "plan.json").write_bytes(encoded)
    (attempt / "preflight.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for entity, raw in snapshots.items():
        if entity not in {side + str(index) for side in "LR" for index in range(5)}:
            raise RuntimeError("需要固定球员实体ID L0..R4")
        (attempt / (entity + "_landmark_snapshot.json")).write_bytes(raw)
    queue = unreal.MoviePipelineQueue()
    settings = []
    for camera in plan["cameras"]:
        destination = "/Game/FutsalMOT/Development/Pose17/" + plan["attempt"] + "/" + camera["camera_id"]
        if unreal.EditorAssetLibrary.does_asset_exist(destination):
            raise RuntimeError("独立Sequence资产已存在，拒绝覆盖")
        copied = unreal.EditorAssetLibrary.duplicate_asset(camera["source_sequence"], destination)
        if copied is None or not unreal.EditorAssetLibrary.save_loaded_asset(copied):
            raise RuntimeError("复制/保存独立Sequence失败，不改原序列")
        copy_report = _sequence_report(unreal, destination)
        source_report = next(row for row in report["source_sequences"] if row["path"] == camera["source_sequence"])
        if copy_report["bindings"] != source_report["bindings"] or copy_report["top_tracks"] != source_report["top_tracks"]:
            raise RuntimeError("复制Sequence绑定/轨道不一致，禁止渲染")
        job = queue.allocate_new_job(unreal.MoviePipelineExecutorJob)
        job.job_name = camera["camera_id"]
        job.sequence = unreal.SoftObjectPath(destination)
        job.map = unreal.SoftObjectPath(_LEVEL)
        config = job.get_configuration()
        output = config.find_or_add_setting_by_class(unreal.MoviePipelineOutputSetting)
        output.output_directory = unreal.DirectoryPath(str(attempt / camera["camera_id"]))
        output.output_resolution = unreal.IntPoint(3840, 2160)
        # 文件帧为attempt内相对编号；EXR/native仍保留真实Sequence源帧。
        output.file_name_format = "{render_pass}{frame_number_rel}"
        if output.file_name_format != "{render_pass}{frame_number_rel}":
            raise RuntimeError("相对输出帧文件名配置读回失败")
        output.zero_pad_frame_numbers = 6
        output.use_custom_playback_range = True
        output.custom_start_frame = plan["start_frame"]
        output.custom_end_frame = plan["start_frame"] + plan["frame_count"]
        config.find_or_add_setting_by_class(unreal.FutsalMoviePipelinePNGOutput)
        exr = config.find_or_add_setting_by_class(unreal.MoviePipelineImageSequenceOutput_EXR)
        exr.multilayer = True
        config.find_or_add_setting_by_class(unreal.MoviePipelineDeferredPassBase)
        object_id = config.find_or_add_setting_by_class(unreal.MoviePipelineObjectIdRenderPass)
        # 与正式已验证路径一致，按Actor而不是Material/组件碎片分实例。
        object_id.set_editor_property("id_type", unreal.MoviePipelineObjectIdPassIdType.ACTOR)
        if object_id.get_editor_property("id_type") != unreal.MoviePipelineObjectIdPassIdType.ACTOR:
            raise RuntimeError("ObjectID未保持逐Actor分组，禁止提交")
        aa = config.find_or_add_setting_by_class(unreal.MoviePipelineAntiAliasingSetting)
        aa.spatial_sample_count = 1
        aa.temporal_sample_count = 1
        if plan.get("pose_sampling_path") == "gpu_render_cpu_lbs":
            aa.set_editor_property("use_camera_cut_for_warm_up",False)
            aa.set_editor_property("engine_warm_up_count",plan["gpu_lbs_warmup_frames"])
            aa.set_editor_property("render_warm_up_count",0)
            if aa.get_editor_property("engine_warm_up_count") != plan["gpu_lbs_warmup_frames"]:
                raise RuntimeError("GPU/LBS预注册warmup配置读回失败")
        if plan.get("allow_pie_visual_mesh_override") is True:
            aa.set_editor_property("use_camera_cut_for_warm_up", False)
            aa.set_editor_property("engine_warm_up_count", plan["override_warmup_frames"])
            aa.set_editor_property("render_warm_up_count", 0)
            if aa.get_editor_property("engine_warm_up_count") != plan["override_warmup_frames"]:
                raise RuntimeError("角色override warmup配置读回失败")
        setting = config.find_or_add_setting_by_class(unreal.FutsalPose17MRQPass)
        setting.max_records = plan["frame_count"] * len(plan["targets"])
        setting.targets = targets
        setting.set_editor_property("prepare_cpu_for_diagnostic", plan.get("allow_pie_cpu_preparation") is True)
        setting.set_editor_property("hide_driver_for_diagnostic", plan.get("allow_pie_driver_hide") is True)
        setting.set_editor_property("diagnostic_camera_object_names", [camera["camera_actor_object_name"]])
        setting.set_editor_property("diagnostic_receipt_file", str(attempt / (camera["camera_id"] + "_native_pose_restore_receipt.json")))
        if plan.get("allow_pie_visual_mesh_override") is True:
            setting.set_editor_property("allow_pie_visual_mesh_override", True)
            setting.set_editor_property("diagnostic_scope", plan["diagnostic_scope"])
            setting.set_editor_property("minimum_override_warmup_frames", plan["override_warmup_frames"])
        if plan.get("pose_sampling_path") == "gpu_render_cpu_lbs":
            setting.set_editor_property("use_gpu_render_cpu_lbs_for_diagnostic",True)
            setting.set_editor_property("diagnostic_scope",plan["diagnostic_scope"])
            setting.set_editor_property("minimum_gpu_lbs_warmup_frames",plan["gpu_lbs_warmup_frames"])
        if plan.get("diagnostic_scope")=="pose17_gpu_render_cpu_lbs_joint_diagnostic":
            setting.set_editor_property("allow_joint_pose17_diagnostic",True)
        settings.append(setting)
    _PREPARED = {"queue": queue, "settings": settings, "attempt": attempt, "plan": plan}
    # MCP run_python_file 可能使用新的执行globals；通过专用模块持有 transient UE 对象。
    session = types.ModuleType(_SESSION)
    session.prepared = _PREPARED
    sys.modules[_SESSION] = session
    state = {"schema": "futsalmot_pose17_smoke_state_v1", "status": "prepared_not_submitted",
             "plan_sha256": hashlib.sha256(encoded).hexdigest(), "jobs": len(plan["cameras"]),
             "ready_to_render": (plan.get("allow_pie_cpu_preparation") is True or plan.get("pose_sampling_path")=="gpu_render_cpu_lbs")
                 and plan.get("allow_pie_driver_hide") is True,
             "pose_sampling_path":plan.get("pose_sampling_path","cpu_skin_cache"),
             "production_accepted": False, "completed_jobs": [], "errors": [],
             "remaining_gates": ["runtime_CPU_LOD_prepare_restore_validation",
                                 "camera_motionblur_zero", "source_sequence_runtime_pose_dynamic", "MRQ_RGB_mask_pose_pair_validation"]}
    (attempt / "state.json").write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    _PREPARED["state"] = state
    return state


def _write_state(session):
    filename = session["attempt"] / "state.json"
    temporary = filename.with_suffix(".tmp")
    temporary.write_text(json.dumps(session["state"], ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(str(temporary), str(filename))


def _assert_override_receipt(plan, receipt):
    """纯CPU契约：新角色必须提供真实身份、材质/动画保持及恢复收据。"""
    targets = [row for row in plan["targets"] if row.get("diagnostic_visual_mesh_override")]
    if not targets:
        if receipt.get("visual_mesh_override_requested") or receipt.get("visual_mesh_overrides"):
            raise RuntimeError("未授权计划出现角色override")
        return
    if (receipt.get("visual_mesh_override_requested") is not True
            or receipt.get("diagnostic_scope") != plan.get("diagnostic_scope") or receipt.get("pipeline_setup_error")):
        raise RuntimeError("角色override native授权scope/setup门禁未通过")
    if plan.get("diagnostic_scope")=="pose17_gpu_render_cpu_lbs_character_calibration":
        player_id = plan.get("calibration_player_id")
        if targets[0]["diagnostic_visual_mesh_override"]!=calibration_visual_mesh(player_id):
            raise RuntimeError("角色override计划ID与精确资产白名单不一致")
    evidence = receipt.get("visual_mesh_overrides",[])
    if len(evidence) != len(targets):
        raise RuntimeError("角色override native事务数量不符")
    for target in targets:
        path = target["diagnostic_visual_mesh_override"]
        matching = [row for row in evidence if row.get("applied_asset_path")==path and
            (plan.get("diagnostic_scope")!="pose17_gpu_render_cpu_lbs_joint_diagnostic" or
             row.get("component_path","").endswith("."+target["actor_object_name"]+"."+target["component_object_name"]))]
        if len(matching) != 1:
            raise RuntimeError("角色override未提供对应新mesh事务")
        row = matching[0]
        for field in ("identity_verified","apply_succeeded","animation_binding_preserved","new_asset_materials_verified",
                      "restore_attempted","restore_succeeded"):
            if row.get(field) is not True:
                raise RuntimeError("角色override native证据失败: "+field)
        if row.get("apply_error") or row.get("restore_error"):
            raise RuntimeError("角色override native事务存在错误")
        samples = [r for r in receipt["records"] if r["entity_id"]==target["entity_id"]]
        if plan.get("pose_sampling_path") == "gpu_render_cpu_lbs":
            allowed_originals = {production_visual_mesh(i) for i in range(3,10)} if plan.get("diagnostic_scope")=="pose17_gpu_render_cpu_lbs_joint_diagnostic" else {ORIGINAL_VISUAL_MESH}
            if row.get("original_asset_path") not in allowed_originals:
                raise RuntimeError("角色override原mesh不是精确Player003")
            if (row.get("gpu_lbs_validation_attempted") is not True or row.get("gpu_lbs_validation_succeeded") is not True
                    or row.get("gpu_lbs_validation_error")):
                raise RuntimeError("角色override后没有实际GPU/LBS门禁成功证据")
            expected_vertices = samples[0].get("mesh_identity",{}).get("vertex_count") if samples else -1
            if type(expected_vertices) is not int: raise RuntimeError("角色override缺少新mesh实际顶点数")
            _assert_gpu_lbs_eligibility(row.get("gpu_lbs_eligibility_after_override",{}),expected_vertices)
            applied,prepared = row.get("applied_engine_frame"),receipt.get("gpu_lbs_prepared_engine_frame")
            if type(applied) is not int or type(prepared) is not int or applied<0 or prepared<applied:
                raise RuntimeError("角色apply/GPU prepare实际engine frame顺序不成立")
        if not samples or any(r["mesh_identity"]["asset_path"] != path
                or r.get("override_warmup_engine_frames_observed",-1)<plan["override_warmup_frames"] for r in samples):
            raise RuntimeError("角色override采样不是新mesh身份或实际warmup不足")
        if plan.get("pose_sampling_path") == "gpu_render_cpu_lbs":
            for sample in samples:
                elapsed = sample.get("override_warmup_engine_frames_observed")
                if (type(elapsed) is not int or elapsed!=sample["hook_engine_frame"]-applied
                        or elapsed<sample.get("gpu_lbs_warmup_engine_frames_observed",-1)):
                    raise RuntimeError("角色override与GPU prepare实际warmup关系不一致")


def _assert_gpu_lbs_eligibility(eligibility,vertex_count):
    """实际完整门禁快照；post-override可尚未绑定row，但不能缺任何检查。"""
    for field in ("checks_executed","eligible","pie_world","weights_readable"):
        if eligibility.get(field) is not True: raise RuntimeError("GPU/LBS实际门禁证据缺失: "+field)
    for field in ("render_cpu_enabled","render_static","has_nanite_data","parallel_evaluation_running",
                  "leader_parallel_evaluation_running","non_finite_morph_weights","lod0_has_cloth_data","variable_influence_weights"):
        if eligibility.get(field) is not False: raise RuntimeError("GPU/LBS不支持状态: "+field)
    for field in ("active_morph_weight_count","manual_morph_curve_count","external_morph_set_count","cloth_simulation_count",
                  "lod0_cloth_section_count","mesh_deformer_asset_count","mesh_deformer_instance_count","material_wpo_slot_count","null_material_slot_count"):
        if type(eligibility.get(field)) is not int or eligibility[field]!=0:
            raise RuntimeError("GPU/LBS不支持变形/材质计数: "+field)
    actual_count = eligibility.get("reference_vertex_count")
    if (type(actual_count) is not int or actual_count<=0
            or type(eligibility.get("weight_vertex_count")) is not int or eligibility["weight_vertex_count"]!=actual_count
            or (vertex_count is not None and (type(vertex_count) is not int or vertex_count!=actual_count))
            or type(eligibility.get("max_bone_influences")) is not int or eligibility["max_bone_influences"]<=0):
        raise RuntimeError("GPU/LBS实际weights元数据不一致")


def _assert_gpu_lbs_receipt(plan,receipt):
    """新路径独立成功契约；不能把CPU render/cached结果冒充GPU渲染的LBS诊断。"""
    if plan.get("pose_sampling_path","cpu_skin_cache") != "gpu_render_cpu_lbs":
        if receipt.get("gpu_lbs_diagnostic_requested"):
            raise RuntimeError("计划未授权GPU/LBS但native已启用")
        return
    if (receipt.get("pose_sampling_path") != "gpu_render_cpu_lbs" or receipt.get("gpu_lbs_diagnostic_requested") is not True
            or receipt.get("diagnostic_scope") != plan.get("diagnostic_scope") or receipt.get("pipeline_setup_error")):
        raise RuntimeError("GPU/LBS原生scope/路径/setup未通过")
    prepared = receipt.get("gpu_lbs_prepared_engine_frame")
    if type(prepared) is not int or prepared<0:
        raise RuntimeError("GPU/LBS缺少实际预备engine frame")
    if not receipt.get("records"):
        raise RuntimeError("GPU/LBS没有实际采样记录")
    for row in receipt["records"]:
        if (row.get("sample_succeeded") is not True or row.get("render_cpu_enabled_at_read") is not False
                or row.get("render_static_at_read") is not False or type(row.get("predicted_lod_at_read")) is not int or row["predicted_lod_at_read"] != 0
                or row.get("sampling_method") != "gpu_render_component_pose_cpu_lbs_no_morph_cloth_deformer_wpo_diagnostic"
                or row.get("pose_fence_method") != "gpu_dynamic_data_endframe_tasks_finished_render_commands_flush"
                or row.get("end_frame_update_tasks_finished") is not True):
            raise RuntimeError("GPU/LBS实际GPU模式/方法/LOD0/endframe fence证据失败")
        warmup = row.get("gpu_lbs_warmup_engine_frames_observed")
        if (type(warmup) is not int or type(row.get("hook_engine_frame")) is not int
                or warmup<plan["gpu_lbs_warmup_frames"] or warmup!=row["hook_engine_frame"]-prepared):
            raise RuntimeError("GPU/LBS实际warmup与预备/采样engine frame不一致")
        vertex_count = row.get("mesh_identity",{}).get("vertex_count")
        if type(vertex_count) is not int: raise RuntimeError("GPU/LBS记录缺少完整mesh顶点数")
        _assert_gpu_lbs_eligibility(row.get("gpu_lbs_eligibility",{}),vertex_count)


def submit():
    """显式提交专用队列，异步返回；成功仍只是待配对验收的诊断产物。"""
    import unreal
    session = getattr(sys.modules.get(_SESSION), "prepared", None)
    if session is None or session["state"]["status"] != "prepared_not_submitted" or not session["state"]["ready_to_render"]:
        raise RuntimeError("缺少当前会话已准备且未提交的独立队列")
    gate = preflight()
    if gate["dirty_packages"] or gate["pie_worlds"] or gate["mrq_running"]:
        raise RuntimeError("submit前dirty/PIE/MRQ门禁未通过")
    executor = unreal.MoviePipelinePIEExecutor()
    session["executor"] = executor

    def on_finished(_executor, success):
        state = session["state"]
        state["executor_success"] = bool(success)
        # 不订阅带PIE Pipeline/Job的延迟delegate。此处只读C++在world清理前保存的JSON。
        from pose17_native_receipt import normalize_native_receipt
        for camera in session["plan"]["cameras"]:
            name = camera["camera_id"]
            filename = session["attempt"] / (name + "_native_pose_restore_receipt.json")
            try:
                receipt = normalize_native_receipt(json.loads(filename.read_text(encoding="utf-8")))
                _assert_override_receipt(session["plan"],receipt)
                _assert_gpu_lbs_receipt(session["plan"],receipt)
                records = receipt["records"]
                expected = session["plan"]["frame_count"] * len(session["plan"]["targets"])
                normalized_path = session["attempt"] / (name + "_pose_restore_receipt.json")
                if normalized_path.exists():
                    raise RuntimeError("标准化收据已存在，拒绝覆盖")
                normalized_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
                state["completed_jobs"].append({"job": name, "receipt": normalized_path.name,
                    "sampling_passed": len(records) == expected and not receipt["record_limit_reached"]
                        and len({(row["output_frame"], row["entity_id"]) for row in records}) == expected
                        and all(row["sample_succeeded"] for row in records) and not receipt["setup_error"],
                    "restore_passed": not receipt["restore_errors"] and receipt["camera_state_restored"],
                    "streaming_state_restart_required": receipt["streaming_state_restart_required"]})
            except Exception as error:
                state["errors"].append("native_receipt: " + name + ": " + repr(error))
        jobs = state["completed_jobs"]
        all_passed = bool(success) and not state["errors"] and len(jobs) == len(session["plan"]["cameras"]) and all(
            row["sampling_passed"] and row["restore_passed"] for row in jobs)
        state["status"] = "rendered_requires_RGB_mask_pose_alignment_acceptance" if all_passed else "failed_requires_diagnosis"
        state["ready_to_render"] = False
        state["media"] = {camera["camera_id"]: {extension: len(list((session["attempt"] / camera["camera_id"]).glob("*." + extension)))
                         for extension in ("png", "exr")} for camera in session["plan"]["cameras"]}
        if any(count < session["plan"]["frame_count"] for counts in state["media"].values() for count in counts.values()):
            state["status"] = "failed_requires_diagnosis"
            state["errors"].append("PNG或ObjectID EXR数量不足；不以executor成功代替媒体完整性")
        state["streaming_restart_required"] = any(job["streaming_state_restart_required"] for job in jobs)
        state["editor_restore_probe_pending"] = True
        _write_state(session)
        print("POSE17_SMOKE_FINISHED " + json.dumps(state, ensure_ascii=False))

    session["callbacks"] = [on_finished]
    executor.on_executor_finished_delegate.add_callable_unique(on_finished)
    session["state"]["status"] = "submitted_async"
    _write_state(session)
    try:
        executor.execute(session["queue"])
    except Exception as error:
        session["state"]["status"] = "submit_failed"
        session["state"]["errors"].append(repr(error))
        _write_state(session)
        raise
    return session["state"]


def postflight():
    """异步回调返回后由独立MCP调用读取Editor，不在PIE/GC回调内访问UObject。"""
    session = getattr(sys.modules.get(_SESSION), "prepared", None)
    if session is None or not session["state"].get("editor_restore_probe_pending"):
        raise RuntimeError("尚无完成会话需要独立postflight")
    after = preflight()
    path = session["attempt"] / "editor_after.json"
    if path.exists():
        raise RuntimeError("postflight收据已存在，不覆盖")
    path.write_text(json.dumps(after, ensure_ascii=False, indent=2), encoding="utf-8")
    before = json.loads((session["attempt"] / "preflight.json").read_text(encoding="utf-8"))
    state = session["state"]
    state["editor_scene_state_matched"] = after["level"] == before["level"] and not after["dirty_packages"] and not after["pie_worlds"]
    state["editor_mesh_state_matched"] = after["players"] == before["players"]
    state["editor_restore_probe_pending"] = False
    if not state["editor_scene_state_matched"] or not state["editor_mesh_state_matched"]:
        state["status"] = "failed_requires_diagnosis"
        state["errors"].append("Editor场景/mesh恢复不一致，检查streaming并安全重启")
    _write_state(session)
    return state


def _assert_release_gate(session, report, executor_running):
    """纯状态门禁；释放只针对已成功收尾的本会话，不接受准备中/失败任务。"""
    state = session["state"]
    if (state.get("status") != "rendered_requires_RGB_mask_pose_alignment_acceptance"
            or state.get("executor_success") is not True or state.get("errors")
            or state.get("ready_to_render") is not False
            or state.get("editor_restore_probe_pending") is not False
            or state.get("editor_scene_state_matched") is not True
            or state.get("editor_mesh_state_matched") is not True
            or state.get("streaming_restart_required") is not False):
        raise RuntimeError("会话尚未成功完成executor/native恢复/postflight，不可释放")
    jobs = state.get("completed_jobs", [])
    expected = {camera["camera_id"] for camera in session["plan"]["cameras"]}
    if (len(jobs) != len(expected) or {job["job"] for job in jobs} != expected
            or not all(job.get("sampling_passed") is True and job.get("restore_passed") is True for job in jobs)):
        raise RuntimeError("本会话完成相机/采样/恢复收据不完整")
    if (executor_running or report["mrq_running"] or report["pie_worlds"] or report["dirty_packages"]
            or report["level"].split(".")[0] != _LEVEL):
        raise RuntimeError("release前实时Level/dirty/PIE/全局或本executor MRQ门禁未通过")


def release():
    """完成后解除本会话delegate及强引用；不清空队列、不删除资产或产物。"""
    global _PREPARED
    holder = sys.modules.get(_SESSION)
    session = getattr(holder, "prepared", None)
    if session is None:
        raise RuntimeError("没有待释放的本会话")
    executor = session.get("executor")
    if executor is None:
        raise RuntimeError("本会话没有已完成executor")
    # 仅访问仍被session强持有的executor；绝不访问PIE Pipeline/Job。
    report = preflight()
    _assert_release_gate(session, report, bool(executor.is_rendering()))
    attempt = session["attempt"]
    state_on_disk = json.loads((attempt / "state.json").read_text(encoding="utf-8"))
    after = json.loads((attempt / "editor_after.json").read_text(encoding="utf-8"))
    if state_on_disk != session["state"] or after["level"] != report["level"] or after["players"] != report["players"]:
        raise RuntimeError("落盘状态或当前Editor场景/mesh与postflight不一致，不可释放")
    from pose17_native_receipt import normalize_native_receipt
    receipt_files = []
    for camera in session["plan"]["cameras"]:
        path = attempt / (camera["camera_id"] + "_native_pose_restore_receipt.json")
        receipt = normalize_native_receipt(json.loads(path.read_text(encoding="utf-8")))
        _assert_override_receipt(session["plan"],receipt)
        _assert_gpu_lbs_receipt(session["plan"],receipt)
        expected = session["plan"]["frame_count"] * len(session["plan"]["targets"])
        if (receipt["lifecycle_stage"] != "pipeline_teardown_before_PIE_cleanup" or receipt["setup_error"]
                or receipt["record_limit_reached"] or receipt["restore_errors"]
                or not receipt["camera_state_restored"] or receipt["streaming_state_restart_required"]
                or len(receipt["records"]) != expected
                or len({(row["output_frame"], row["entity_id"]) for row in receipt["records"]}) != expected
                or not all(row["sample_succeeded"] for row in receipt["records"])):
            raise RuntimeError("native持久化/恢复收据未通过release门禁: " + path.name)
        receipt_files.append(path.name)
    path = attempt / "session_release.json"
    if path.exists():
        raise RuntimeError("release收据已存在，不覆盖")
    # UE5.8 PyWrapperDelegate.cpp::RemoveCallable/ContainsCallable公开API；按自己的callable解绑。
    # 不调用clear/remove_all，不接触全局队列或其他订阅者。
    delegate = executor.on_executor_finished_delegate
    callbacks = session.get("callbacks", [])
    if not callbacks or not all(callable(callback) for callback in callbacks):
        raise RuntimeError("本会话callback身份缺失，不猜测全量解绑")
    for callback in callbacks:
        delegate.remove_callable(callback)
        if delegate.contains_callable(callback):
            raise RuntimeError("本会话delegate解绑读回失败")
    result = {"schema": "futsalmot_pose17_smoke_session_release_v1", "attempt": str(attempt),
              "native_receipts": receipt_files, "owned_callbacks_removed": len(callbacks),
              "executor_is_rendering": False, "mrq_running": False, "pie_worlds": [], "dirty_packages": [],
              "artifacts_retained": True, "other_queues_untouched": True, "production_accepted": False}
    with path.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    session["state"]["session_released"] = True
    session["state"]["session_release_receipt"] = path.name
    _write_state(session)
    # 清空旧字典可断开旧run_python_file globals/closure持有的UObject引用。
    # 历史只保留纯JSON路径；不主动GC，不破坏UE/其他任务持有的对象。
    holder.last_release = result
    session.clear()
    holder.prepared = None
    _PREPARED = None
    return result


def main():
    action = os.environ.get("FUTSAL_POSE17_SMOKE_ACTION", "preflight")
    filename = os.environ.get("FUTSAL_POSE17_SMOKE_PLAN")
    plan = json.loads(Path(filename).read_text(encoding="utf-8")) if filename else None
    if action == "preflight":
        result = preflight(plan)
    elif action == "prepare" and plan is not None:
        result = prepare(plan)
    elif action == "submit":
        result = submit()
    elif action == "postflight":
        result = postflight()
    elif action == "release":
        result = release()
    else:
        raise ValueError("只支持preflight、显式计划prepare、submit、postflight及成功会话release")
    print("POSE17_SMOKE " + json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
