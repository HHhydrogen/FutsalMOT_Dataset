"""只改已prepare新attempt的相机数值轨道，不改场景Actor或canonical序列。

preflight对UE资产只读，但会把自有调度state.ready_to_render封锁为false；
只有三个复制序列均保存/读回通过的显式apply才解锁。
"""

import hashlib
import json
import math
import os
import re
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pose17_smoke_runner as runner

FIRST_PARTIAL_SHA = "06ed86909c5852a0ec3e70c07f6b1e1485f33d2adca5ff3e724d4c8e143ca09d"
TRACKING_NAME = "bEnableLookAtTracking"
TRACKING_PATH = "LookatTrackingSettings.bEnableLookAtTracking"
L0_ACTOR = "BP_FutsalPlayerBase_C_UAID_44A3BB9598B7230503_1114910274"
PRODUCTION_CAMERA_SCOPE = {
    "CineCam_01": ("/Game/FutsalMOT/Sequences/LS_Cam_01", "CineCameraActor_UAID_44A3BB9598B7AFEC02_1646179601", [-1740., 6., 100.]),
    "CineCam_02": ("/Game/FutsalMOT/Sequences/LS_Cam_02", "CineCameraActor_UAID_44A3BB9598B7B0EC02_1882079778", [-1930., -220., 100.]),
    "CineCam_03": ("/Game/FutsalMOT/Sequences/LS_Cam_03", "CineCameraActor_UAID_44A3BB9598B7B0EC02_1882548779", [-1930., 232., 100.]),
}


def production_player_asset(player):
    """生产003目录较早建立；004–009统一在Mesh子目录，绝不宽松前缀匹配。"""
    if type(player) is not int or not 3 <= player <= 9:
        raise ValueError("只允许本轮生产角色003–009")
    directory = f"/Game/FutsalMOT/Characters/MPFB/Players/Player_{player:03d}/"
    return directory + ("" if player == 3 else "Mesh/") + f"MPFB_Player_{player:03d}.MPFB_Player_{player:03d}"


def validate_calibration_scope(plan, manifest):
    """跨角色近景仅改变L0 PIE VisualMesh，角色/路径/候选身份必须精确一致。"""
    player = plan.get("calibration_player_id", 3)
    expected_asset = production_player_asset(player)
    if len(plan.get("targets", [])) != 1:
        raise ValueError("近景校准仅允许一个L0角色，禁止批量override")
    target = plan["targets"][0]
    if (target.get("entity_id") != "L0" or target.get("actor_object_name") != L0_ACTOR
            or target.get("component_object_name") != "MPFBVisual" or target.get("sequence_binding_name") != "Player_L0"):
        raise ValueError("角色校准限定已有L0/MPFBVisual，不允许其他Actor或驱动底模")
    if manifest.get("ue_identity", {}).get("asset_path") != expected_asset:
        raise ValueError("角色ID与manifest实际UE资产身份不一致")
    if Path(target.get("landmark_manifest", "")).name != f"mpfb_player{player:03d}_ue_surface_head_candidates.json":
        raise ValueError("角色ID与候选文件不一致")
    override = target.get("diagnostic_visual_mesh_override")
    sampling_path = plan.get("pose_sampling_path", "cpu_skin_cache")
    if player != 3 or override is not None or plan.get("allow_pie_visual_mesh_override"):
        if override != expected_asset:
            raise ValueError("override必须是该角色的精确生产SkeletalMesh对象路径")
        warmup = plan.get("override_warmup_frames")
        if type(warmup) is not int or not 8 <= warmup <= 120:
            raise ValueError("override必须至少8帧warmup且不超过120帧")
        if sampling_path == "gpu_render_cpu_lbs":
            if (not 4 <= player <= 9 or plan.get("diagnostic_scope") != "pose17_gpu_render_cpu_lbs_character_calibration"
                    or plan.get("allow_pie_visual_mesh_override") is not True
                    or plan.get("allow_gpu_render_cpu_lbs_diagnostic") is not True
                    or plan.get("allow_pie_cpu_preparation") is not False
                    or plan.get("allow_pie_driver_hide") is not True):
                raise ValueError("GPU-LBS角色组合仅允许004–009单角色、独立scope、双opt-in、CPU关闭与隐藏driver")
            gpu_warmup = plan.get("gpu_lbs_warmup_frames")
            if type(gpu_warmup) is not int or not 8 <= gpu_warmup <= 120 or gpu_warmup != warmup:
                raise ValueError("GPU-LBS/override两个warmup必须为相同8..120整数")
            if plan.get("start_frame") != 0 or plan.get("frame_count") != 3:
                raise ValueError("GPU-LBS角色组合限定0..2连续三帧")
        elif (sampling_path != "cpu_skin_cache" or plan.get("allow_pie_visual_mesh_override") is not True
                or plan.get("diagnostic_scope") != "pose17_character_calibration"
                or plan.get("allow_gpu_render_cpu_lbs_diagnostic")
                or plan.get("allow_pie_cpu_preparation") is not True or plan.get("allow_pie_driver_hide") is not True):
            raise ValueError("旧CPU角色校准保持显式PIE scope/opt-in/CPU采样/隐藏driver门禁")
    if "calibration_player_id" in plan:
        if not plan["attempt"].startswith(f"player{player:03d}_closeup_a"):
            raise ValueError("attempt角色命名不一致")
        if plan.get("disable_look_at_tracking_in_copies") is not True:
            raise ValueError("跨角色近景必须关闭复制相机tracking")
        if {c["camera_id"] for c in plan["cameras"]} != set(PRODUCTION_CAMERA_SCOPE):
            raise ValueError("只允许已冻结三机位")
        for camera in plan["cameras"]:
            source, actor, location = PRODUCTION_CAMERA_SCOPE[camera["camera_id"]]
            if (camera["source_sequence"] != source or camera["camera_actor_object_name"] != actor
                    or camera["closeup_location_cm"] != location
                    or camera["closeup_look_at_cm"] != [-1980., 6., 75.]):
                raise ValueError("本次角色校准不能改变冻结相机/近景几何")
    return {"calibration_player_id": player, "expected_visual_mesh": expected_asset,
            "override_is_pie_only": override is not None, "pose_sampling_path": sampling_path}


def exact_tracking_track(row):
    """仅此精确camera bool属性允许从非相机完整性比较中排除。"""
    return row.get("class") == "MovieSceneBoolTrack" and row.get("property_name") == TRACKING_NAME and row.get("property_path") == TRACKING_PATH


def sequence_report_with_bool_properties(unreal, path):
    """补全全部bool轨道属性身份/键值，不能用一个BoolTrack类名忽略未知属性。"""
    result = runner._sequence_report(unreal, path)
    sequence = unreal.load_asset(path)
    for binding, row in zip(sequence.get_bindings(), result["bindings"]):
        assert str(binding.get_display_name()) == row["name"]
        tracks = list(binding.get_tracks())
        assert len(tracks) == len(row["tracks"])
        for track, track_row in zip(tracks, row["tracks"]):
            if track.get_class().get_name() == "MovieSceneBoolTrack":
                assert callable(getattr(track, "get_property_name", None)) and callable(getattr(track, "get_property_path", None)), "嵌套bool属性读取API不支持；停止"
                track_row["property_name"] = str(track.get_property_name())
                track_row["property_path"] = str(track.get_property_path())
                track_row["bool_sections"] = [{"keys": [[{"frame": int(k.get_time().frame_number.value), "value": bool(k.get_value())}
                    for k in channel.get_keys()] for channel in section.get_all_channels()],
                    "completion_mode": str(section.get_completion_mode())} for section in track.get_sections()]
    return result


def read_runtime_tracking_gate(camera_actor, expected_forward, minimum_cosine=0.9999):
    """root在已解析PIE相机上只读调用；不创建播放器或在编辑器强制求值。"""
    path = camera_actor.get_path_name()
    if "UEDPIE_" not in path:
        raise ValueError("运行时门禁必须使用实际PIE Actor，不能使用编辑器对象")
    tracking = camera_actor.get_editor_property("lookat_tracking_settings")
    enabled = bool(tracking.get_editor_property("enable_look_at_tracking"))
    forward = camera_actor.get_actor_forward_vector()
    actual = [float(forward.x), float(forward.y), float(forward.z)]
    dot = sum(a * b for a, b in zip(actual, expected_forward))
    norm = math.sqrt(sum(x*x for x in actual) * sum(x*x for x in expected_forward))
    cosine = dot / norm if norm else -1.0
    if enabled or cosine < minimum_cosine:
        raise ValueError("PIE lookat仍启用或实际forward未匹配近景计划")
    return {"camera_actor": path, "lookat_enabled": enabled, "forward": actual,
            "expected_forward": expected_forward, "cosine": cosine, "runtime_gate_passed": True}


def assigned_key_value(channel_class, value):
    """诊断明确赋IEEE754 float32可精确表示值；不推断引擎隐式转换来源。"""
    if channel_class not in ("MovieSceneScriptingFloatChannel", "MovieSceneScriptingDoubleChannel"):
        raise ValueError("未验证的数值通道: " + channel_class)
    return float(struct.unpack("<f", struct.pack("<f", float(value)))[0])


def validate_failed_partial(previous, plan, plan_path):
    """恢复仅接受同attempt、同原计划位置、尚无copy保存的明确失败。"""
    if previous.get("status") != "failed_partial_copied_tracks_preserved" or previous.get("attempt") != plan["attempt"]:
        raise ValueError("不是当前attempt的明确失败收据")
    if previous.get("plan_path") != str(plan_path):
        raise ValueError("失败收据引用不同计划")
    if {r["camera_id"] for r in previous.get("cameras", [])} != {c["camera_id"] for c in plan["cameras"]}:
        raise ValueError("失败收据相机集合不一致")
    if any(r.get("modified_and_saved") for r in previous["cameras"]):
        raise ValueError("当前恢复仅支持首次未保存的部分写入")


def camera_values(location_cm, look_at_cm):
    """UE相机局部+X朝前；Transform旋转通道X/Y/Z为Roll/Pitch/Yaw。"""
    if len(location_cm) != 3 or len(look_at_cm) != 3 or not all(math.isfinite(v) for v in location_cm + look_at_cm):
        raise ValueError("相机与目标必须为有限三维厘米坐标")
    delta = [b - a for a, b in zip(location_cm, look_at_cm)]
    if math.sqrt(sum(v * v for v in delta)) < 1:
        raise ValueError("相机与目标过近")
    return {"Location.X": float(location_cm[0]), "Location.Y": float(location_cm[1]),
            "Location.Z": float(location_cm[2]), "Rotation.X": 0.,
            "Rotation.Y": math.degrees(math.atan2(delta[2], math.hypot(delta[0], delta[1]))),
            "Rotation.Z": math.degrees(math.atan2(delta[1], delta[0])),
            "Scale.X": 1., "Scale.Y": 1., "Scale.Z": 1.}


def validate_closeup_plan(plan):
    """固定本次L0三机位/三连续帧边界，不静默扩大诊断范围。"""
    runner.validate_plan(plan)
    if len(plan["cameras"]) != 3 or plan["start_frame"] != 0 or plan["frame_count"] != 3:
        raise ValueError("本次近景计划限定三机位和0..2连续帧")
    if plan["attempt"] == "player003_mrq_a02" or "closeup" not in plan["attempt"]:
        raise ValueError("必须显式新近景attempt")
    if len(plan["targets"]) != 1 or plan["targets"][0].get("entity_id") != "L0":
        raise ValueError("本次近景仅验证L0，其他角色须新计划")
    for camera in plan["cameras"]:
        camera_values(camera["closeup_location_cm"], camera["closeup_look_at_cm"])
    if len({tuple(c["closeup_location_cm"]) for c in plan["cameras"]}) != 3:
        raise ValueError("近景三机位位置必须不同")


def normalized_channels(section):
    channels = section.get_all_channels()
    result = {}
    for channel in channels:
        name = re.sub(r"_\d+$", "", str(channel.channel_name))
        if name in result:
            raise ValueError("重复规范通道名: " + name)
        result[name] = channel
    expected = {kind + "." + axis for kind in ("Location", "Rotation", "Scale") for axis in "XYZ"}
    if not expected.issubset(result):
        raise ValueError("相机Transform通道名未匹配；禁止按索引猜测: " + repr(list(result)))
    return {key: result[key] for key in expected}


def non_camera_signature(sequence_report, camera_name, allow_exact_tracking=False):
    """仅排除目标camera的Transform，其他绑定及CameraCut均参与比较。"""
    result = json.loads(json.dumps(sequence_report))
    for binding in result["bindings"]:
        if binding["name"] == camera_name:
            binding["tracks"] = [t for t in binding["tracks"] if t["class"] != "MovieScene3DTransformTrack"
                                 and not (allow_exact_tracking and exact_tracking_track(t))]
    return hashlib.sha256(json.dumps(result, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def assign_owned_job_sequence(job, copy_path, expected_copy_path, soft_path_factory):
    """仅声明自有transient job的明确copy目标，不依赖SoftObjectPath转换或比较。"""
    prefix = "/Game/FutsalMOT/Development/Pose17/"
    if copy_path != expected_copy_path or not copy_path.startswith(prefix) or ".." in copy_path or "\\" in copy_path:
        raise ValueError("拒绝将job指向canonical或其他attempt序列")
    if str(job.job_name) != copy_path.rsplit("/", 1)[-1]:
        raise ValueError("job_name与明确copy相机不一致")
    job.sequence = soft_path_factory(copy_path)


def main():
    import unreal
    project = Path(unreal.Paths.project_dir()).resolve()
    plan_path = Path(os.environ.get("FUTSAL_POSE17_CLOSEUP_PLAN", str(project / "Saved/Pose17Development/player003_closeup_a03.json")))
    plan = json.loads(plan_path.read_text(encoding="utf-8-sig"))
    validate_closeup_plan(plan)
    manifest_path = Path(plan["targets"][0]["landmark_manifest"])
    manifest_bytes = manifest_path.read_bytes()
    calibration_scope = validate_calibration_scope(plan, json.loads(manifest_bytes))
    disable_tracking = plan.get("disable_look_at_tracking_in_copies") is True
    action = os.environ.get("FUTSAL_POSE17_CLOSEUP_ACTION", "preflight")
    assert action in ("preflight", "apply", "preflight_resume_failed_partial", "resume_failed_partial")
    resuming = action in ("preflight_resume_failed_partial", "resume_failed_partial")
    destination_root = "/Game/FutsalMOT/Development/Pose17/" + plan["attempt"] + "/"
    subsystem = unreal.get_editor_subsystem(unreal.MoviePipelineQueueSubsystem)
    assert not subsystem.is_rendering() and not unreal.EditorLevelLibrary.get_pie_worlds(False)
    assert unreal.EditorLevelLibrary.get_editor_world().get_path_name().split(".", 1)[0] == runner._LEVEL
    dirty = [p.get_path_name() for p in list(unreal.EditorLoadingAndSavingUtils.get_dirty_content_packages())
             + list(unreal.EditorLoadingAndSavingUtils.get_dirty_map_packages())]
    allowed_dirty = {destination_root + c["camera_id"] for c in plan["cameras"]} if resuming else set()
    assert set(dirty).issubset(allowed_dirty), "范围外dirty: " + repr(dirty)
    session = getattr(sys.modules.get(runner._SESSION), "prepared", None)
    assert session and session["state"]["status"] == "prepared_not_submitted", "先通过独立runner prepare新队列"
    assert session["plan"] == plan and session["plan"]["attempt"] == plan["attempt"]
    saved_plan = (session["attempt"] / "plan.json").read_bytes()
    assert json.loads(saved_plan) == plan
    assert hashlib.sha256(saved_plan).hexdigest() == session["state"]["plan_sha256"]
    prepared_landmarks = (session["attempt"] / "L0_landmark_snapshot.json").read_bytes()
    assert hashlib.sha256(prepared_landmarks).hexdigest() == hashlib.sha256(manifest_bytes).hexdigest(), "候选manifest在prepare后发生变化"
    # 通用prepare尚未配置近景时不可submit。此处仅更新调度门禁，不写UE资产。
    session["state"]["ready_to_render"] = False
    session["state"]["closeup_configuration_status"] = "blocked_pending_copied_camera_tracks"
    runner._write_state(session)
    jobs = list(session["queue"].get_jobs())
    assert len(jobs) == 3 and {str(job.job_name) for job in jobs} == {c["camera_id"] for c in plan["cameras"]}
    failed_report_path = session["attempt"] / "closeup_camera_tracks.json"
    previous = None
    if resuming:
        original_bytes = failed_report_path.read_bytes()
        assert hashlib.sha256(original_bytes).hexdigest() == FIRST_PARTIAL_SHA, "首次失败收据变化"
        previous = json.loads(original_bytes)
        validate_failed_partial(previous, plan, plan_path)
        report_path = session["attempt"] / "closeup_camera_tracks_resume01.json"
    else:
        report_path = failed_report_path
    assert not report_path.exists(), "近景轨道收据已存在，拒绝隐式重跑"
    report = {"status": "preflight_passed", "action": action, "attempt": plan["attempt"],
              "plan_path": str(plan_path), "cameras": [], "canonical_changed": False,
              "scene_actor_transform_changed": False, "ready_to_render": False,
              "visual_acceptance": False, "source_scope": plan["source_scope"],
              "plan_sha256": session["state"]["plan_sha256"], "owned_dirty_before": dirty,
              "parent_failure_sha256": FIRST_PARTIAL_SHA if resuming else None,
              "disable_tracking_in_copies": disable_tracking,
              "runtime_lookat_false_verified": False, "calibration_scope": calibration_scope,
              "landmark_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest()}
    if disable_tracking:
        assert hasattr(unreal, "MovieSceneBoolTrack")
        assert hasattr(unreal.MovieSceneCompletionMode, "RESTORE_STATE")
    prepared = []
    editor_camera_flags = []
    if disable_tracking:
        editor_actors = list(unreal.EditorLevelLibrary.get_all_level_actors())
        for camera in plan["cameras"]:
            matches = [a for a in editor_actors if a.get_name() == camera["camera_actor_object_name"]]
            assert len(matches) == 1
            settings = matches[0].get_editor_property("lookat_tracking_settings")
            initial = bool(settings.get_editor_property("enable_look_at_tracking"))
            editor_camera_flags.append((matches[0], initial))
        report["editor_camera_tracking_before"] = {a.get_name(): value for a, value in editor_camera_flags}
    for camera in plan["cameras"]:
        copied_path = destination_root + camera["camera_id"]
        job = next(j for j in jobs if str(j.job_name) == camera["camera_id"])
        copied = unreal.load_asset(copied_path)
        assert copied and isinstance(copied, unreal.LevelSequence)
        bindings = [b for b in copied.get_bindings() if str(b.get_display_name()) == camera["camera_id"]]
        assert len(bindings) == 1, "相机binding非唯一"
        binding = bindings[0]
        assert binding.get_possessed_object_class().get_name() == "CineCameraActor"
        tracks = [t for t in binding.get_tracks() if t.get_class().get_name() == "MovieScene3DTransformTrack"]
        if disable_tracking:
            assert callable(getattr(unreal.MovieSceneBoolTrack, "set_property_name_and_path", None)), "BoolTrack属性路径API缺失；停止"
            assert callable(getattr(unreal.MovieSceneBoolTrack, "get_property_name", None))
            assert callable(getattr(unreal.MovieSceneBoolTrack, "get_property_path", None))
            existing_tracking = [t for t in binding.get_tracks() if t.get_class().get_name() == "MovieSceneBoolTrack"
                and str(t.get_property_path()) == TRACKING_PATH]
            assert not existing_tracking, "新attempt的目标tracking轨道已存在，拒绝隐式重建"
        assert len(tracks) <= 1, "多个相机Transform轨道存在歧义"
        if tracks:
            assert len(tracks[0].get_sections()) == 1, "相机Transform需唯一section"
            normalized_channels(tracks[0].get_sections()[0])
        values = camera_values(camera["closeup_location_cm"], camera["closeup_look_at_cm"])
        before = sequence_report_with_bool_properties(unreal, copied_path) if disable_tracking else runner._sequence_report(unreal, copied_path)
        source_file = project / "Content" / (camera["source_sequence"].removeprefix("/Game/") + ".uasset")
        assert source_file.is_file()
        if resuming:
            previous_rows = [r for r in previous["cameras"] if r["camera_id"] == camera["camera_id"]]
            assert len(previous_rows) == 1
            prior_row = previous_rows[0]
            assert prior_row["copy_sequence"] == copied_path and prior_row["target_values"] == values
            assert non_camera_signature(before, camera["camera_id"]) == prior_row["non_camera_sha256_before"], "失败后非相机轨道有未知变化"
            assert hashlib.sha256(source_file.read_bytes()).hexdigest() == prior_row["source_uasset_sha256_before"], "失败后canonical源变化"
        row = {"camera_id": camera["camera_id"], "view_role": camera["closeup_view_role"],
               "source_sequence": camera["source_sequence"], "copy_sequence": copied_path,
               "source_uasset_sha256_before": hashlib.sha256(source_file.read_bytes()).hexdigest(),
               "target_values": values, "non_camera_sha256_before": non_camera_signature(before, camera["camera_id"], disable_tracking),
               "before": before, "modified_and_saved": False,
               "job_sequence_assignment_enforced": False,
               "actual_EXR_camera_execution_verified": False}
        report["cameras"].append(row)
        prepared.append((camera, copied, binding, tracks, row, source_file, job))
    if action in ("preflight", "preflight_resume_failed_partial"):
        print("POSE17_CLOSEUP_PREFLIGHT " + json.dumps({k: v for k, v in report.items() if k != "cameras"}))
        print("POSE17_CLOSEUP_CAMERAS " + json.dumps([{k: v for k, v in row.items() if k != "before"} for row in report["cameras"]]))
        return report

    def flush():
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    report["status"] = "apply_intent_persisted"
    flush()
    try:
        for camera, copied, binding, tracks, row, source_file, job in prepared:
            track = tracks[0] if tracks else binding.add_track(unreal.MovieScene3DTransformTrack)
            section = track.get_sections()[0] if tracks else track.add_section()
            channels = normalized_channels(section)
            section.set_range(0, 3)
            row["precision_readback"] = {}
            for name, channel in channels.items():
                channel_class = channel.get_class().get_name()
                expected_value = assigned_key_value(channel_class, row["target_values"][name])
                for key in list(channel.get_keys()):
                    channel.remove_key(key)
                for frame in (0, 1, 2):
                    channel.add_key(time=unreal.FrameNumber(frame), new_value=expected_value,
                                    interpolation=unreal.MovieSceneKeyInterpolation.LINEAR)
                keys = list(channel.get_keys())
                assert len(keys) == 3 and all(float(k.get_value()) == expected_value for k in keys), (name, channel_class, expected_value, [float(k.get_value()) for k in keys])
                assert [int(k.get_time().frame_number.value) for k in keys] == [0, 1, 2]
                row["precision_readback"][name] = {"channel_class": channel_class,
                    "plan_original_double": row["target_values"][name], "assigned_float32": expected_value,
                    "quantization_error": expected_value - row["target_values"][name],
                    "actual": [float(k.get_value()) for k in keys]}
            if disable_tracking:
                tracking_track = binding.add_track(unreal.MovieSceneBoolTrack)
                tracking_track.set_property_name_and_path(TRACKING_NAME, TRACKING_PATH)
                assert str(tracking_track.get_property_name()) == TRACKING_NAME
                assert str(tracking_track.get_property_path()) == TRACKING_PATH
                tracking_section = tracking_track.add_section()
                tracking_section.set_range(0, 3)
                tracking_section.set_completion_mode(unreal.MovieSceneCompletionMode.RESTORE_STATE)
                assert tracking_section.get_completion_mode() == unreal.MovieSceneCompletionMode.RESTORE_STATE
                bool_channels = list(tracking_section.get_all_channels())
                assert len(bool_channels) == 1
                bool_channel = bool_channels[0]
                assert bool_channel.get_class().get_name() == "MovieSceneScriptingBoolChannel"
                for frame in (0, 1, 2):
                    bool_channel.add_key(time=unreal.FrameNumber(frame), new_value=False)
                bool_keys = list(bool_channel.get_keys())
                assert len(bool_keys) == 3 and all(k.get_value() is False for k in bool_keys)
                assert [int(k.get_time().frame_number.value) for k in bool_keys] == [0, 1, 2]
                row["tracking_override"] = {"property_name": TRACKING_NAME, "property_path": TRACKING_PATH,
                    "frames": [0, 1, 2], "values": [False, False, False], "completion_mode": "RESTORE_STATE",
                    "runtime_property_evaluation_verified": False}
            assert unreal.EditorAssetLibrary.save_loaded_asset(copied)
            after = sequence_report_with_bool_properties(unreal, row["copy_sequence"]) if disable_tracking else runner._sequence_report(unreal, row["copy_sequence"])
            assert non_camera_signature(after, camera["camera_id"], disable_tracking) == row["non_camera_sha256_before"], "非相机轨道变化"
            assert hashlib.sha256(source_file.read_bytes()).hexdigest() == row["source_uasset_sha256_before"], "canonical源文件变化"
            # 与prepare相同的已验证赋值入口，目标限定为当前复制资产；不猜测结构转换API。
            assign_owned_job_sequence(job, row["copy_sequence"], destination_root + camera["camera_id"], unreal.SoftObjectPath)
            row["job_sequence_assignment_enforced"] = True
            row["after"] = after
            row["modified_and_saved"] = True
            flush()
        report["status"] = "copied_camera_tracks_saved_not_rendered"
        if disable_tracking:
            report["editor_camera_tracking_after"] = {a.get_name(): bool(a.get_editor_property("lookat_tracking_settings").get_editor_property("enable_look_at_tracking"))
                                                       for a, _ in editor_camera_flags}
            assert report["editor_camera_tracking_after"] == report["editor_camera_tracking_before"], "地图相机本体tracking设置变化"
        report["ready_to_render"] = True
        flush()
        session["state"]["closeup_configuration_status"] = "three_copied_camera_tracks_saved_and_readback_verified"
        session["state"]["closeup_camera_tracks_receipt"] = str(report_path)
        session["state"]["closeup_camera_tracks_receipt_sha256"] = hashlib.sha256(report_path.read_bytes()).hexdigest()
        if disable_tracking:
            session["state"]["closeup_tracking_override_in_copied_sequences"] = TRACKING_PATH
            gate = "runtime_lookat_false_and_actual_EXR_camera_forward_match"
            if gate not in session["state"]["remaining_gates"]:
                session["state"]["remaining_gates"].append(gate)
        session["state"]["ready_to_render"] = True
        runner._write_state(session)
    except Exception as exc:
        report["status"] = "failed_partial_copied_tracks_preserved"
        report["error"] = repr(exc)
        flush()
        session["state"]["ready_to_render"] = False
        session["state"]["closeup_configuration_status"] = "failed_copied_camera_tracks_preserved"
        runner._write_state(session)
        raise
    print("POSE17_CLOSEUP_SAVED " + json.dumps({k: v for k, v in report.items() if k != "cameras"}))
    return report


if __name__ == "__main__":
    main()
