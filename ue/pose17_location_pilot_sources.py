"""四seed位置监督候选的源Sequence入口；只建源，不准备Queue或提交MRQ。"""

import copy
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pose17_joint_grf_sequences as joint

CANDIDATE_RELATIVE = "Saved/Pose17Development/location_pilot_4seed_candidate_a01/candidate_plan.json"
CANDIDATE_SHA256 = "21cdf06aa1de3b009b13855a6870fb16f57800102611a2ce7e52d7e5c7c9a672"
PACKAGE = "/Game/FutsalMOT/Development/Pose17/location_pilot_4seed_a01"
WINDOWS = {1042: (100, 220), 1044: (307, 427), 1048: (100, 220), 1102: (100, 220)}


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _plain_path(path):
    """拒绝已有路径中的符号链接或Windows目录联接，不跟随链接寻找输入。"""
    path = Path(path).absolute()
    for current in [path, *path.parents]:
        if current.exists():
            _require(not current.is_symlink() and not (getattr(current.stat(), "st_file_attributes", 0) & 0x400),
                     "拒绝联接或符号链接: " + str(current))
    return path


def source_prefix(meta, frames, seed):
    """保留源零帧开始的真实运动历史；绝不把120帧重基后冒充原GRF帧。"""
    _require(type(seed) is int and seed in WINDOWS, "仅允许已冻结四seed")
    end = WINDOWS[seed][1]
    _require(meta.get("schema") == "grf_ue_episode" and meta.get("source", {}).get("seed") == seed
             and meta["source"].get("environment") == "google_research_football"
             and meta["source"].get("scenario") == "5_vs_5", "源GRF身份不符")
    _require(meta.get("timing") == {"source_step_seconds": .1, "playback_fps": 10, "num_steps": 1800}
             and len(frames) == 1800, "源时间基或1800帧数量不符")
    for index, frame in enumerate(frames[:end]):
        _require(type(frame.get("step")) is int and frame["step"] == index
                 and type(frame.get("time_seconds")) in (float, int)
                 and math.isfinite(frame["time_seconds"])
                 and abs(frame["time_seconds"] - index * .1) <= 1e-12, "源前缀必须连续且不重基")
        players = frame.get("players", [])
        _require(len(players) == 10 and {p["id"] for p in players} == set(joint.JOINT_VISUAL_IDS), "每帧十实体必须唯一齐全")
        positions = [p.get("position_m", []) for p in players] + [frame.get("ball", {}).get("position_m", [])]
        _require(all(len(position) == 3 and all(type(v) in (int, float) and math.isfinite(v) for v in position)
                     for position in positions), "球和球员坐标必须有限")
    prefix_meta = copy.deepcopy(meta)
    prefix_meta["timing"]["num_steps"] = end
    return prefix_meta, copy.deepcopy(frames[:end])


def validate_candidate(candidate, project):
    """与短诊断分离的固定候选门禁；不把本函数当作渲染或正式输出准入。"""
    project = Path(project)
    _require(candidate.get("schema") == "futsalmot_pose17_location_pilot_candidate_v1"
             and candidate.get("status") == "candidate_not_admitted_not_runnable"
             and candidate.get("production_admitted") is False
             and candidate.get("training_authorized_by_this_file") is False, "必须为未准入位置监督候选")
    groups = candidate.get("groups", [])
    _require([row.get("root_seed") for row in groups] == list(WINDOWS), "只允许固定顺序的四seed候选")
    for row in groups:
        seed = row["root_seed"]
        start, end = WINDOWS[seed]
        split = "train" if seed != 1102 else "synthetic_dev"
        package = PACKAGE + f"/seed_{seed:08d}"
        plan = row["proposed_runtime_plan"]
        expected = joint.make_joint_plan(project)
        expected.update(attempt=f"location_pilot_s{seed:08d}_a01_gpu_lbs", start_frame=start, frame_count=120,
                        location_pilot_contract=True,
                        source_scope="real_GRF_motion_location_pilot_diagnostic_roster_override_current_scene_fixed_blue_location_only_not_original_weather_or_appearance",
                        review_frame_indices=[60])
        expected["location_pilot"] = {"schema": "futsalmot_pose17_location_pilot_v1", "batch_id": "location_pilot_4seed_a01",
                                      "seed": seed, "split": split, "source_frame_range": [start, end],
                                      "location_supervision": True, "visibility_supervision": False, "formal_pose_release": False}
        expected["source_motion"] = {"root_seed": seed, "source_frames_sha256": row["source_frames_sha256"],
                                     "source_meta_sha256": row["source_meta_sha256"], "context_source_frame_range": [0, end],
                                     "selected_source_frame_range": [start, end], "sequence_frame_equals_source_step": True,
                                     "fps": 10, "source_build_receipt": (project / CANDIDATE_RELATIVE).with_name("source_prepare.json").as_posix()}
        expected["cameras"] = [{"camera_id": camera, "camera_actor_object_name": data[0],
                                "source_sequence": package + "/LS_Cam_" + camera[-2:]} for camera, data in joint.CAMERAS.items()]
        _require(plan == expected, "候选运行字段变化，不允许范围外角色/相机/源/帧/监督开关")
        _require(row["sequence_package"] == package and row["source_prefix_range"] == [0, end]
                 and row["selected_source_frame_range"] == [start, end]
                 and row["split"] == ("train" if seed != 1102 else "synthetic-dev"), "候选窗口或split不符")
        active_start, active_end = row["original_frozen_active_source_range"]
        _require(active_start <= start < end <= active_end, "所选窗口不在原冻结活动窗")
        source_dir = project / f"Saved/Production/v1_no_numbers_4k/sources/source_s{seed:08d}"
        _require(Path(row["source_frames_path"]) == source_dir / "frames.jsonl"
                 and Path(row["source_meta_path"]) == source_dir / "meta.json", "只读冻结真实源，不接受任意路径")
        floor = row["floor_manifest"]
        _require(floor.get("application_status") == "planned_not_applied"
                 and floor.get("selection_mode") == "paired_override"
                 and floor.get("variant", {}).get("id") == "sport_blue_v1", "首批仅计划固定已验蓝地板，不冒称已应用")
    return candidate


def _load_inputs(project):
    path = _plain_path(project / CANDIDATE_RELATIVE)
    _require(joint.sha(path) == CANDIDATE_SHA256, "候选SHA变化，请显式新attempt而非隐式修改")
    candidate = validate_candidate(json.loads(path.read_text(encoding="utf-8")), project)
    _require(joint.sha(_plain_path(candidate["frozen_context_path"])) == candidate["frozen_context_sha256"], "冻结上下文变化")
    production = candidate["frozen_production_plan"]
    _require(joint.sha(_plain_path(production["path"])) == production["sha256"], "原冻结生产计划变化")
    _require(joint.sha(Path(__file__).parent / "import_grf_episode.py") == candidate["source_importer"]["sha256"], "已冻结导入器变化")
    inputs, manifest_sha = {}, {}
    for row in candidate["groups"]:
        frames_bytes = _plain_path(row["source_frames_path"]).read_bytes()
        meta_bytes = _plain_path(row["source_meta_path"]).read_bytes()
        _require(hashlib.sha256(frames_bytes).hexdigest() == row["source_frames_sha256"]
                 and hashlib.sha256(meta_bytes).hexdigest() == row["source_meta_sha256"], "真实源SHA变化")
        _require(joint.sha(_plain_path(row["frozen_clip_path"])) == row["frozen_clip_sha256"], "v1出处clip变化")
        frames = [json.loads(line) for line in frames_bytes.decode("utf-8").splitlines() if line.strip()]
        inputs[row["root_seed"]] = source_prefix(json.loads(meta_bytes.decode("utf-8")), frames, row["root_seed"])
        for target in row["proposed_runtime_plan"]["targets"]:
            manifest_path = _plain_path(target["landmark_manifest"])
            content = json.loads(manifest_path.read_text(encoding="utf-8"))
            _require(content.get("ue_identity", {}).get("asset_path") == joint.production_visual_mesh(target["calibration_player_id"]),
                     "真实角色manifest身份不符")
            manifest_sha[target["entity_id"]] = joint.sha(manifest_path)
    context = json.loads(Path(candidate["frozen_context_path"]).read_text(encoding="utf-8"))
    _require(all(manifest_sha[row["entity_id"]] == context["calibration_manifests"][str(row["calibration_player_id"])]["sha256"]
                 for row in candidate["groups"][0]["proposed_runtime_plan"]["targets"]), "校准manifest SHA变化")
    return candidate, inputs, manifest_sha


def _sequence_summary(unreal, path, camera, builder, end):
    """只读复核导入器输出；轨迹/动画算法仍全部来自原通用导入器。"""
    seq = unreal.load_asset(path)
    _require(isinstance(seq, unreal.LevelSequence), "新源Sequence未创建")
    rate = seq.get_display_rate()
    _require([int(rate.numerator), int(rate.denominator)] == [10, 1]
             and seq.get_playback_start() == 0 and seq.get_playback_end() == end, "新Sequence时间基不符")
    bindings = {str(binding.get_display_name()): binding for binding in seq.get_bindings()}
    _require(len(seq.get_bindings()) == 12 and set(bindings) == {"Player_" + e for e in joint.JOINT_VISUAL_IDS} | {"Ball_01", camera},
             "每源必须十球员+球+当前相机")
    counts = {}
    for label, binding in bindings.items():
        tracks = [track for track in binding.get_tracks() if track.get_class().get_name() == "MovieScene3DTransformTrack"]
        _require(len(tracks) == 1 and len(tracks[0].get_sections()) == 1, "Transform轨道不唯一")
        channels = builder._build_channel_map(tracks[0].get_sections()[0].get_all_channels())
        for name in ("Location.X", "Location.Y", "Location.Z"):
            keys = channels[name].get_keys()
            expected_frames = [0, end - 1] if label == camera else list(range(end))
            _require([int(key.get_time().frame_number.value) for key in keys] == expected_frames, "关键帧集合不符")
        counts[label] = {"location_keys_per_axis": len(channels["Location.X"].get_keys())}
        if label.startswith("Player_"):
            speeds = [track for track in binding.get_tracks() if track.get_class().get_name() == "MovieSceneFloatTrack"]
            _require(len(speeds) == 1 and len(speeds[0].get_sections()) == 1, "运动速度轨道缺失")
            speed_channels = speeds[0].get_sections()[0].get_all_channels()
            _require(len(speed_channels) == 1 and speed_channels[0].get_num_keys() == end, "运动速度关键帧不完整")
    cuts = [track for track in seq.get_tracks() if track.get_class().get_name() == "MovieSceneCameraCutTrack"]
    _require(len(cuts) == 1 and len(cuts[0].get_sections()) == 1, "CameraCut缺失或不唯一")
    cut = cuts[0].get_sections()[0]
    bounds = [int(cut.get_start_frame()), int(cut.get_end_frame())]
    _require(bounds == [0, end], "CameraCut范围不符")
    return {"path": path, "display_rate": [10, 1], "playback_range": [0, end], "bindings": counts,
            "camera_cut_range": bounds, "camera_binding_id": str(cut.get_camera_binding_id())}


def main():
    import unreal
    action = os.environ.get("FUTSAL_POSE17_LOCATION_SOURCE_ACTION", "preflight")
    _require(action in {"preflight", "prepare"}, "只允许preflight/prepare；无提交/渲染/隐式恢复")
    project = _plain_path(Path(unreal.Paths.project_dir()).resolve())
    report_path = _plain_path((project / CANDIDATE_RELATIVE).with_name("source_" + action + ".json"))
    _require(report_path.parent.is_dir() and not report_path.exists(), "目录不存在或收据已存在；拒绝隐式重跑")
    report = {"schema": "futsalmot_location_pilot_source_preparation_v1", "action": action, "status": "started",
              "package": PACKAGE, "candidate_sha256": CANDIDATE_SHA256, "mrq_submitted": False,
              "production_acceptance": False, "floor_applied": False, "sequences": []}
    candidate = None
    try:
        candidate, inputs, manifest_sha = _load_inputs(project)
        plan = candidate["groups"][0]["proposed_runtime_plan"]
        _require(not joint._dirty(unreal), "存在范围外或未知dirty包")
        _, before = joint._scene(unreal, plan)
        paths = [camera["source_sequence"] for row in candidate["groups"] for camera in row["proposed_runtime_plan"]["cameras"]]
        _require(not unreal.EditorAssetLibrary.does_directory_exist(PACKAGE)
                 and not any(unreal.EditorAssetLibrary.does_asset_exist(path) for path in paths), "源根目录或资产已存在；保留partial，不隐式覆盖")
        canonical = [project / "Content/FutsalMOT/Sequences" / f"LS_Cam_{i:02d}.uasset" for i in range(1, 5)]
        canonical.append(project / "Content/FutsalMOT/Maps/L_FutsalCourt.umap")
        hashes = {path.as_posix(): joint.sha(path) for path in canonical}
        mapping = json.loads((Path(__file__).parent / "actor_mapping.example.json").read_text(encoding="utf-8"))
        _require(mapping == {**{entity: "Player_" + entity for entity in joint.JOINT_VISUAL_IDS}, "BALL": "Ball_01"}, "Actor mapping不符")
        report.update(status="preflight_passed", scene_before=before, canonical_sha256_before=hashes,
                      manifest_sha256=manifest_sha, importer_sha256=candidate["source_importer"]["sha256"],
                      script_sha256=joint.sha(__file__), source_windows=[{"seed": seed, "prefix_range": [0, end],
                                                                       "selected_range": [start, end]} for seed, (start, end) in WINDOWS.items()])
        if action == "prepare":
            # 已通过Level/PIE/MRQ/dirty门禁；仅重载导入器，避免旧缓存缺少两个显式禁副作用参数。
            if "import_grf_episode" in sys.modules:
                builder = importlib.reload(sys.modules["import_grf_episode"])
            else:
                import import_grf_episode as builder
            for row in candidate["groups"]:
                seed = row["root_seed"]
                prefix_meta, prefix_frames = inputs[seed]
                builder.create_sequence(prefix_meta, prefix_frames, mapping, replace_existing=False,
                                        package_path=row["sequence_package"],
                                        sequences_cfg=[{"name": "LS_Cam_" + camera[-2:], "camera_actor": camera} for camera in joint.CAMERAS],
                                        ball_rolling_cfg={"enabled": True, "radius_m": .11, "minimum_move_distance_m": .0001, "roll_sign": 1.},
                                        open_after_create=False, run_api_smoke=False)
                for camera in row["proposed_runtime_plan"]["cameras"]:
                    summary = _sequence_summary(unreal, camera["source_sequence"], camera["camera_id"], builder, WINDOWS[seed][1])
                    summary.update(root_seed=seed, selected_source_range=list(WINDOWS[seed]))
                    report["sequences"].append(summary)
                _, after = joint._scene(unreal, plan)
                _require(before == after and not joint._dirty(unreal), "建源后15Actor或dirty变化；停止并保留partial")
            report.update(scene_after=after, dirty_after=joint._dirty(unreal),
                          canonical_sha256_after={path.as_posix(): joint.sha(path) for path in canonical})
            _require(report["canonical_sha256_after"] == hashes, "canonical或地图改变；保留partial")
            report.update(status="sixteen_source_sequences_saved_not_rendered", scene_preserved=True,
                          runtime_plans_created=False, floor_applied=False)
        joint._dump_new(report_path, report)
        print(json.dumps({"status": report["status"], "report": str(report_path), "mrq_submitted": False}, ensure_ascii=False))
    except Exception as error:
        report.update(status="failed_partial_preserved_no_automatic_retry", error=repr(error))
        try:
            report["dirty_after_failure"] = joint._dirty(unreal)
            report["existing_owned_sequences"] = [camera["source_sequence"] for row in (candidate or {}).get("groups", [])
                                                    for camera in row["proposed_runtime_plan"]["cameras"]
                                                    if unreal.EditorAssetLibrary.does_asset_exist(camera["source_sequence"])]
        except Exception as inspect_error:
            report["failure_inspection_error"] = repr(inspect_error)
        joint._dump_new(report_path, report)
        raise


if __name__ == "__main__":
    main()
