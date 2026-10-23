"""1048真实GRF前缀→四个全新源Sequence；默认preflight，不打开Sequencer或提交MRQ。"""

import copy
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pose17_smoke_runner as runner
from pose17_calibration_assets import JOINT_VISUAL_IDS, production_visual_mesh

PACKAGE = "/Game/FutsalMOT/Development/Pose17/joint_seed1048_a01"
ATTEMPT = "joint_seed1048_a01_gpu_lbs"
FRAMES_SHA = "cebe1bf84ff16e94ca18a6bca883ecfb8cc1847bca3041262b1f9d82a2e3ce8b"
META_SHA = "9ea713988046680a9fce1d0b21b6539e7e539bcc943d091289d69b8b543330a7"
CAMERAS = {
    "CineCam_01": ("CineCameraActor_UAID_44A3BB9598B7AFEC02_1646179601", [2500., 1250., 700.]),
    "CineCam_02": ("CineCameraActor_UAID_44A3BB9598B7B0EC02_1882079778", [-2500., 1250., 700.]),
    "CineCam_03": ("CineCameraActor_UAID_44A3BB9598B7B0EC02_1882548779", [-2500., -1250., 700.]),
    "CineCam_04": ("CineCameraActor_UAID_44A3BB9598B7B0EC02_1882935780", [2500., -1250., 700.]),
}
ACTOR_SUFFIX = {"L0": "1114910274", "L1": "1114900272", "L2": "1114930278", "L3": "1114905273",
                "L4": "1114941280", "R0": "1114916275", "R1": "1114925277", "R2": "1114920276",
                "R3": "1114935279", "R4": "1114889271"}


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_prefix(meta, frames):
    """不重基源帧，保留0..109供运动历史，最终只渲染100..109。"""
    _require(meta.get("schema") == "grf_ue_episode" and meta["source"].get("seed") == 1048
             and meta["source"].get("environment") == "google_research_football"
             and meta["source"].get("scenario") == "5_vs_5", "真实GRF1048身份不符")
    _require(meta.get("timing") == {"source_step_seconds": .1, "playback_fps": 10, "num_steps": 1800},
             "源时间基不符，不能混淆Sequence帧和GRF帧")
    _require(len(frames) == 1800, "源帧数量不符")
    selected = copy.deepcopy(frames[:110])
    for index, frame in enumerate(selected):
        _require(type(frame.get("step")) is int and frame["step"] == index
                 and type(frame.get("time_seconds")) in (float, int)
                 and math.isfinite(frame["time_seconds"])
                 and abs(frame["time_seconds"] - index * .1) <= 1e-12, "源前缀必须真实连续且不重基")
        _require(len(frame.get("players", [])) == 10
                 and {p["id"] for p in frame["players"]} == set(JOINT_VISUAL_IDS), "每源帧必须十实体齐全")
    prefix_meta = copy.deepcopy(meta)
    prefix_meta["timing"]["num_steps"] = 110
    return prefix_meta, selected


def make_joint_plan(project):
    """只生成明确诊断计划；角色覆盖是显式roster，不复现1048原随机外观。"""
    project = Path(project)
    targets = []
    for entity, player in JOINT_VISUAL_IDS.items():
        row = {"entity_id": entity, "calibration_player_id": player,
               "actor_object_name": "BP_FutsalPlayerBase_C_UAID_44A3BB9598B7230503_" + ACTOR_SUFFIX[entity],
               "component_object_name": "MPFBVisual", "sequence_binding_name": "Player_" + entity,
               "landmark_manifest": (project / "Saved/ResearchDiagnostics" /
                                     f"mpfb_player{player:03d}_ue_surface_head_candidates.json").as_posix()}
        if player != 3:
            row["diagnostic_visual_mesh_override"] = production_visual_mesh(player)
        targets.append(row)
    return {"schema": "futsalmot_pose17_smoke_plan_v1", "attempt": ATTEMPT,
            "start_frame": 100, "frame_count": 10, "pose_sampling_path": "gpu_render_cpu_lbs",
            "diagnostic_scope": "pose17_gpu_render_cpu_lbs_joint_diagnostic", "allow_joint_pose17_diagnostic": True,
            "allow_gpu_render_cpu_lbs_diagnostic": True, "allow_pie_visual_mesh_override": True,
            "allow_pie_cpu_preparation": False, "allow_pie_driver_hide": True,
            "gpu_lbs_warmup_frames": 8, "override_warmup_frames": 8,
            "allow_unaccepted_landmarks_for_diagnostic": True,
            "source_scope": "real_GRF_motion_seed1048_with_diagnostic_roster_override_and_current_scene_not_original_weather_or_appearance",
            "review_frame_indices": [5], "review_index_semantics": "capture_relative_index_not_GRF_source_frame",
            "source_motion": {"root_seed": 1048, "source_frames_sha256": FRAMES_SHA, "source_meta_sha256": META_SHA,
                              "context_source_frame_range": [0, 110], "selected_source_frame_range": [100, 110],
                              "sequence_frame_equals_source_step": True, "fps": 10,
                              "source_build_receipt": (project / "Saved/Pose17Development/joint_seed1048_a01_source_prepare.json").as_posix()},
            "targets": targets,
            "cameras": [{"camera_id": camera, "camera_actor_object_name": data[0],
                         "source_sequence": PACKAGE + "/LS_Cam_" + camera[-2:]} for camera, data in CAMERAS.items()]}


def _dump_new(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def _dirty(unreal):
    return sorted(p.get_path_name() for p in list(unreal.EditorLoadingAndSavingUtils.get_dirty_content_packages())
                  + list(unreal.EditorLoadingAndSavingUtils.get_dirty_map_packages()))


def _scene(unreal, plan):
    _require(unreal.EditorLevelLibrary.get_editor_world().get_path_name().split(".", 1)[0] == runner._LEVEL,
             "当前Level不符")
    _require(not unreal.EditorLevelLibrary.get_pie_worlds(False), "PIE仍活动")
    _require(not unreal.get_editor_subsystem(unreal.MoviePipelineQueueSubsystem).is_rendering(), "MRQ仍活动")
    actors = list(unreal.EditorLevelLibrary.get_all_level_actors())
    labels = {entity: "Player_" + entity for entity in JOINT_VISUAL_IDS}
    labels["BALL"] = "Ball_01"
    labels.update({camera: camera for camera in CAMERAS})
    selected, snapshot = {}, {}
    for key, label in labels.items():
        matches = [actor for actor in actors if actor.get_actor_label() == label]
        _require(len(matches) == 1, "Actor label不唯一: " + label)
        actor = matches[0]
        fuzzy = [candidate for candidate in actors
                 if label.lower() in (candidate.get_actor_label() or candidate.get_name()).lower()]
        _require(len(fuzzy) == 1 and fuzzy[0] == actor, "旧导入器的包含匹配存在额外候选: " + label)
        if key in JOINT_VISUAL_IDS:
            expected = next(t["actor_object_name"] for t in plan["targets"] if t["entity_id"] == key)
            _require(actor.get_name() == expected, "球员UObject身份变化: " + label)
        if key in CAMERAS:
            _require(actor.get_name() == CAMERAS[key][0] and isinstance(actor, unreal.CineCameraActor), "相机身份不符")
        loc, rot, scale = actor.get_actor_location(), actor.get_actor_rotation(), actor.get_actor_scale3d()
        record = {"object_name": actor.get_name(), "location_cm": [loc.x, loc.y, loc.z],
                  "rotation_deg": [rot.pitch, rot.yaw, rot.roll], "scale": [scale.x, scale.y, scale.z]}
        if key in CAMERAS:
            _require(record["location_cm"] == CAMERAS[key][1], "原机位位置已变化: " + key)
            component = actor.get_cine_camera_component()
            filmback = component.get_editor_property("filmback")
            tracking = actor.get_editor_property("lookat_tracking_settings")
            record["camera"] = {"focal_length": component.get_editor_property("current_focal_length"),
                                "filmback": [filmback.sensor_width, filmback.sensor_height],
                                "lookat_tracking_enabled": bool(tracking.get_editor_property("enable_look_at_tracking"))}
        if key in JOINT_VISUAL_IDS:
            record["meshes"] = runner._mesh_report(unreal, actor)
        selected[key] = actor
        snapshot[key] = record
    return selected, snapshot


def _sequence_summary(unreal, path, camera, builder):
    seq = unreal.load_asset(path)
    _require(isinstance(seq, unreal.LevelSequence), "新源Sequence未创建")
    rate = seq.get_display_rate()
    _require([int(rate.numerator), int(rate.denominator)] == [10, 1]
             and seq.get_playback_start() == 0 and seq.get_playback_end() == 110, "新Sequence时间基不符")
    bindings = {str(binding.get_display_name()): binding for binding in seq.get_bindings()}
    expected = {"Player_" + entity for entity in JOINT_VISUAL_IDS} | {"Ball_01", camera}
    _require(len(seq.get_bindings()) == 12 and set(bindings) == expected, "新绑定必须十球员+球+当前相机")
    counts = {}
    for label, binding in bindings.items():
        tracks = [t for t in binding.get_tracks() if t.get_class().get_name() == "MovieScene3DTransformTrack"]
        _require(len(tracks) == 1 and len(tracks[0].get_sections()) == 1, "Transform轨道不唯一")
        channels = builder._build_channel_map(tracks[0].get_sections()[0].get_all_channels())
        for name in ("Location.X", "Location.Y", "Location.Z"):
            keys = channels[name].get_keys()
            expected_frames = [0, 109] if label == camera else list(range(110))
            _require([int(key.get_time().frame_number.value) for key in keys] == expected_frames, "关键帧集合不符")
        counts[label] = {"location_keys_per_axis": len(channels["Location.X"].get_keys())}
        if label.startswith("Player_"):
            speed_tracks = [t for t in binding.get_tracks() if t.get_class().get_name() == "MovieSceneFloatTrack"]
            _require(len(speed_tracks) == 1 and len(speed_tracks[0].get_sections()) == 1, "运动速度轨道缺失")
            speed = speed_tracks[0].get_sections()[0].get_all_channels()
            _require(len(speed) == 1 and speed[0].get_num_keys() == 110, "运动速度关键帧不完整")
    cuts = [t for t in seq.get_tracks() if t.get_class().get_name() == "MovieSceneCameraCutTrack"]
    _require(len(cuts) == 1 and len(cuts[0].get_sections()) == 1, "CameraCut缺失或不唯一")
    cut = cuts[0].get_sections()[0]
    _require(callable(getattr(cut, "get_start_frame", None)) and callable(getattr(cut, "get_end_frame", None)),
             "CameraCut范围读取API缺失；停止")
    bounds = [int(cut.get_start_frame()), int(cut.get_end_frame())]
    _require(bounds == [0, 110], "CameraCut范围不符")
    return {"path": path, "display_rate": [10, 1], "playback_range": [0, 110], "bindings": counts,
            "camera_cut_range": bounds, "camera_binding_id": str(cut.get_camera_binding_id())}


def main():
    import unreal
    action = os.environ.get("FUTSAL_POSE17_JOINT_SEQUENCE_ACTION", "preflight")
    _require(action in {"preflight", "prepare"}, "只允许preflight/prepare，不支持render或隐式恢复")
    project = Path(unreal.Paths.project_dir()).resolve()
    root = project / "Saved/Pose17Development"
    _require(root.is_dir(), "已知Saved诊断目录不存在")
    report_path = root / ("joint_seed1048_a01_source_" + action + ".json")
    _require(not report_path.exists(), "本动作收据已存在；拒绝隐式重跑")
    plan_path = root / (ATTEMPT + ".json")
    report = {"status": "preflight_started", "action": action, "package": PACKAGE,
              "sequences": [], "mrq_submitted": False, "production_acceptance": False}
    try:
        _require(not plan_path.exists(), "联合计划已存在，拒绝覆盖")
        plan = make_joint_plan(project)
        runner.validate_plan(plan)
        source = project / "Saved/Production/v1_no_numbers_4k/sources/source_s00001048"
        _require(sha(source / "frames.jsonl") == FRAMES_SHA and sha(source / "meta.json") == META_SHA,
                 "真实源SHA与正式1048出处不一致")
        meta = json.loads((source / "meta.json").read_text(encoding="utf-8"))
        frames = [json.loads(line) for line in (source / "frames.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        prefix_meta, prefix_frames = source_prefix(meta, frames)
        mapping = json.loads((Path(__file__).parent / "actor_mapping.example.json").read_text(encoding="utf-8"))
        _require(mapping == {**{entity: "Player_" + entity for entity in JOINT_VISUAL_IDS}, "BALL": "Ball_01"},
                 "正式Actor mapping不符")
        manifest_sha = {}
        for target in plan["targets"]:
            manifest_path = Path(target["landmark_manifest"])
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            _require(manifest.get("ue_identity", {}).get("asset_path") == production_visual_mesh(target["calibration_player_id"]),
                     "各真实角色manifest身份不符")
            manifest_sha[target["entity_id"]] = sha(manifest_path)
        _require(not _dirty(unreal), "存在范围外或未知dirty包")
        actors, before = _scene(unreal, plan)
        _require(not unreal.EditorAssetLibrary.does_directory_exist(PACKAGE)
                 and not any(unreal.EditorAssetLibrary.does_asset_exist(c["source_sequence"]) for c in plan["cameras"]),
                 "新源包已存在；可能partial，保留且不猜测重试")
        canonical_files = [project / "Content/FutsalMOT/Sequences" / f"LS_Cam_{i:02d}.uasset" for i in range(1, 5)]
        canonical_files.append(project / "Content/FutsalMOT/Maps/L_FutsalCourt.umap")
        canonical_before = {str(path): sha(path) for path in canonical_files}
        report.update(status="preflight_passed", source_files={"frames": str(source / "frames.jsonl"), "meta": str(source / "meta.json")},
                      source_frames_sha256=FRAMES_SHA, source_meta_sha256=META_SHA, context_source_range=[0, 110],
                      selected_source_range=[100, 110], source_scope=plan["source_scope"], manifest_sha256=manifest_sha,
                      scene_before=before, canonical_sha256_before=canonical_before,
                      importer_sha256=sha(Path(__file__).parent / "import_grf_episode.py"),
                      builder_sha256=sha(__file__), plan=plan)
        if action == "prepare":
            # 当前Editor可能缓存过旧函数签名；仅在已确认无PIE/MRQ时显式重载导入器。
            if "import_grf_episode" in sys.modules:
                builder = importlib.reload(sys.modules["import_grf_episode"])
            else:
                import import_grf_episode as builder
            # 前置已经精确匹配全部label；沿用原导入算法但关闭Sequencer求值及_TEMP_SMOKE副作用。
            builder.create_sequence(prefix_meta, prefix_frames, mapping, replace_existing=False, package_path=PACKAGE,
                                    sequences_cfg=[{"name": "LS_Cam_" + c[-2:], "camera_actor": c} for c in CAMERAS],
                                    ball_rolling_cfg={"enabled": True, "radius_m": .11, "minimum_move_distance_m": .0001, "roll_sign": 1.},
                                    open_after_create=False, run_api_smoke=False)
            for camera in plan["cameras"]:
                report["sequences"].append(_sequence_summary(unreal, camera["source_sequence"], camera["camera_id"], builder))
            _, after = _scene(unreal, plan)
            report.update(scene_after=after, dirty_after=_dirty(unreal),
                          canonical_sha256_after={str(path): sha(path) for path in canonical_files})
            _require(before == after and not report["dirty_after"], "建源后Actor/Visual状态或dirty包变化；停止并保留partial")
            _require(report["canonical_sha256_after"] == canonical_before, "canonical资产变化；停止")
            _dump_new(plan_path, plan)
            report.update(status="source_sequences_saved_joint_plan_created_not_submitted", plan_path=str(plan_path),
                          plan_sha256=sha(plan_path), scene_preserved=True)
        _dump_new(report_path, report)
        print(json.dumps({"status": report["status"], "report": str(report_path), "mrq_submitted": False}, ensure_ascii=False))
    except Exception as error:
        report.update(status="failed_partial_preserved_no_automatic_retry", error=repr(error))
        try:
            report["dirty_after_failure"] = _dirty(unreal)
            report["existing_owned_sequences"] = [c["source_sequence"] for c in make_joint_plan(project)["cameras"]
                                                  if unreal.EditorAssetLibrary.does_asset_exist(c["source_sequence"])]
        except Exception as inspect_error:
            report["failure_inspection_error"] = repr(inspect_error)
        _dump_new(report_path, report)
        raise


if __name__ == "__main__":
    main()
