"""UE 统一入口：读取 resolved task，调用既有导入/渲染逻辑。

C5.3 正式数据流（full 模式）：
    create_sequence → export_annotations → Runtime Pose prep → render_episode（RGB+BurnIn）
    （渲染异步完成后，再单独运行 pose_capture_export → build_coco17 → P1 postprocess/audit）

不再在 full 模式调用旧 pose_export.py（Editor 逐帧采样），避免新旧两套 Pose 混入正式
episode（旧 pose_export.py 保留为 Legacy，可经 --mode annotations 显式调用）。

用法（Unreal Editor Python Console）：
    py "D:/path/to/code/ue/run_task.py" --resolved-task "D:/.../resolved-task.json" --mode full
    py "D:/path/to/code/ue/run_task.py" --resolved-task "D:/.../resolved-task.json" --mode pose-finalize

也支持环境变量 C5_RESOLVED_TASK（MCP 调用友好）：
    export C5_RESOLVED_TASK=D:/.../resolved-task.json
    py ".../run_task.py" --mode full

纯 UE Python + 标准库，不导入 gfootball/.venv 任何模块。
"""

import argparse
import importlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

RESOLVED_TASK_SCHEMA = "futsalmot_resolved_task"

# UE Python 会话内，已 import 过的 ue/ 模块会缓存在 sys.modules 中；多次执行本脚本
# 时先强制重载，否则会运行到磁盘上已修改但会话里仍是旧版本的代码。
_UE_MODULE_NAMES = (
    "camera_projection", "annotation_utils", "dataset_export",
    "scene_apply", "annotation_exporter", "render_preset", "render_episode",
    "pose_bones", "pose_export", "pose_render", "pose_capture_export",
    "player_motion", "camera_distribution",
)
for _name in _UE_MODULE_NAMES:
    if _name in sys.modules:
        importlib.reload(sys.modules[_name])


def _fail(msg: str, code: int = 2) -> int:
    print(f"ERROR: {msg}", file=sys.stderr)
    return code


POSE_KEYPOINTS_SCHEMA = "grf_ue_pose_keypoints"


def _apply_resolved_anchor_cameras(camera_entries, annotation_cfg):
    """应用并回读 resolved 的静态相机状态。"""
    import math
    import unreal
    from camera_projection import focal_length_to_fov_deg
    from scene_apply import find_actor

    render_cfg = annotation_cfg.get("render_rgb") or {}
    resolution = (
        int(render_cfg.get("output_resolution_x")),
        int(render_cfg.get("output_resolution_y")),
    ) if render_cfg.get("output_resolution_x") is not None and render_cfg.get("output_resolution_y") is not None else None
    for entry in camera_entries:
        camera_id = entry["camera_id"]
        profile = entry["profile"]
        actor = find_actor(entry["camera_actor"])
        if actor is None:
            raise RuntimeError(f"Canonical ID {camera_id} 的 UE Camera Actor 不存在: {entry['camera_actor']}")
        components = actor.get_components_by_class(unreal.CineCameraComponent)
        if len(components) != 1:
            raise RuntimeError(
                f"Canonical ID {camera_id} 的 Camera Component 数量必须为 1，实际为 {len(components)}"
            )
        component = components[0]
        location = profile["position_m"]
        rotation = profile["rotation_deg"]
        actor.set_actor_location_and_rotation(
            unreal.Vector(float(location[0]) * 100.0, float(location[1]) * 100.0, float(location[2]) * 100.0),
            unreal.Rotator(
                pitch=float(rotation[0]),
                yaw=float(rotation[1]),
                roll=float(rotation[2]),
            ),
            False,
            False,
        )
        lens = profile.get("lens") or {}
        if lens.get("focal_length_mm") is not None:
            component.set_editor_property("current_focal_length", float(lens["focal_length_mm"]))
        actual_location = actor.get_actor_location()
        actual_rotation = actor.get_actor_rotation()
        actual_focal = float(component.get_editor_property("current_focal_length"))
        actual_fov = float(component.get_editor_property("current_horizontal_fov"))
        filmback = component.get_editor_property("filmback")
        sensor_width = float(filmback.sensor_width)
        expected_focal = lens.get("focal_length_mm")
        expected_fov = focal_length_to_fov_deg(actual_focal, sensor_width)
        if expected_focal is not None and not math.isclose(actual_focal, float(expected_focal), rel_tol=0.0, abs_tol=1e-4):
            raise RuntimeError(f"Canonical ID {camera_id} 的 focal_length_mm 应用失败")
        if not math.isclose(actual_fov, expected_fov, rel_tol=0.0, abs_tol=1e-3):
            raise RuntimeError(
                f"Canonical ID {camera_id} 的 derived horizontal FOV 与实际 filmback/focal 不一致: "
                f"expected={expected_fov} actual={actual_fov}"
            )
        if resolution is not None:
            profile_resolution = tuple(profile["resolution"])
            if resolution != profile_resolution:
                raise RuntimeError(
                    f"Canonical ID {camera_id} 的 profile resolution={profile_resolution} "
                    f"与 render resolution={resolution} 不一致"
                )
        for actual, expected, field in (
            (actual_location.x / 100.0, location[0], "position_m.x"),
            (actual_location.y / 100.0, location[1], "position_m.y"),
            (actual_location.z / 100.0, location[2], "position_m.z"),
            (actual_rotation.pitch, rotation[0], "rotation_deg.pitch"),
            (actual_rotation.yaw, rotation[1], "rotation_deg.yaw"),
            (actual_rotation.roll, rotation[2], "rotation_deg.roll"),
        ):
            if not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=1e-3):
                raise RuntimeError(f"Canonical ID {camera_id} 的 {field} 回读不匹配")
        print(
            f"  Camera {camera_id}: {entry['camera_actor']} verified "
            f"focal={actual_focal:.6f}mm fov={actual_fov:.6f}"
        )


def _resolve_runtime_camera_selection(
    resolved_task,
    *,
    legacy_sequences,
    legacy_cameras,
    camera_ids=None,
):
    """解析 episode camera policy，并保留显式 smoke/debug 覆盖。"""
    from camera_distribution import resolve_runtime_camera_selection

    return resolve_runtime_camera_selection(
        resolved_task,
        legacy_sequences=legacy_sequences,
        legacy_cameras=legacy_cameras,
        camera_ids=camera_ids,
    )


def _runtime_pose_to_pose_keypoints(ep_dir, cameras, num_steps, ann_cfg):
    """把 Runtime Pose COCO17_3D 输出桥接为 P1 pose_annotator 契约的 pose_keypoints.jsonl。

    读 <ep>/coco17_3d.jsonl（每行 {actor_id, root, keypoints_3d_m: [[x,y,z]×17]}），
    写 <ep>/<cam>/pose_keypoints.jsonl（meta + frame 行，与 pose_export.py 格式一致，
    keypoints_world 单位为米，无 occluded——P1 用 mask 判定 visibility）。
    """
    from pose_bones import COCO_KEYPOINT_NAMES
    from annotation_utils import entity_id_to_track_id

    # 解析 image_width/height（来自 render_rgb 或 annotation_export 配置）
    render_cfg = ann_cfg.get("render_rgb") or {}
    image_width = int(render_cfg.get("output_resolution_x", 1920))
    image_height = int(render_cfg.get("output_resolution_y", 1080))

    c3d = ep_dir / "coco17_3d.jsonl"
    if not c3d.is_file():
        print(f"  WARN: 缺 {c3d}，跳过 pose_keypoints 生成")
        return
    rows = [json.loads(l) for l in c3d.read_text(encoding="utf-8").splitlines() if l.strip()]
    # 按 root（帧）分组
    by_root = {}
    for r in rows:
        by_root.setdefault(r["root"], []).append(r)

    for cam in cameras:
        cam_dir = ep_dir / cam
        cam_dir.mkdir(parents=True, exist_ok=True)
        meta_line = {
            "kind": "meta",
            "schema": POSE_KEYPOINTS_SCHEMA,
            "episode_id": ep_dir.name,
            "camera_id": cam,
            "image_width": image_width,
            "image_height": image_height,
            "keypoint_names": COCO_KEYPOINT_NAMES,
            "coordinate_convention": (
                "world keypoints in meters（Runtime Pose SaveGame cm / 100）；"
                "UE 左手系 X 前 Y 右 Z 上；按 COCO 17 点顺序"
            ),
            "occlusion_method": "none（P1 仅用 mask 判定 visibility）",
        }
        frame_lines = []
        for root in sorted(by_root.keys()):
            objects = []
            for r in by_root[root]:
                kps = r.get("keypoints_3d_m") or []
                if len(kps) != 17:
                    continue
                # None → 仍占位（pose_annotator 检查 len==17）
                kp_world = []
                for p in kps:
                    if p is None:
                        kp_world.append([None, None, None])
                    else:
                        kp_world.append([p[0], p[1], p[2]])
                eid = r["actor_id"]
                objects.append({
                    "entity_id": eid,
                    "track_id": entity_id_to_track_id(eid),
                    "keypoints_world": kp_world,
                })
            frame_lines.append({
                "kind": "frame",
                "frame_index": root + 1,  # GRF step+1 约定（img1 000001 对应 step 0）
                "source_step": root,
                "objects": objects,
            })
        out = [meta_line] + frame_lines
        pk_path = cam_dir / "pose_keypoints.jsonl"
        pk_path.write_text("\n".join(json.dumps(o) for o in out) + "\n", encoding="utf-8")
        print(f"  Wrote: {pk_path} ({len(frame_lines)} 帧，{len(rows)} actor×帧）")


def _load_resolved_task(args) -> dict:
    rt_path = args.resolved_task or os.environ.get("C5_RESOLVED_TASK")
    if not rt_path:
        raise SystemExit(_fail("缺少 --resolved-task（或环境变量 C5_RESOLVED_TASK）"))
    rt_path = Path(rt_path)
    if not rt_path.is_file():
        raise SystemExit(_fail(f"resolved task 不存在: {rt_path}"))
    with open(rt_path, encoding="utf-8") as f:
        rt = json.load(f)
    if rt.get("schema") != RESOLVED_TASK_SCHEMA:
        raise SystemExit(_fail(f"resolved task schema 非法: {rt.get('schema')!r}"))
    if rt.get("version") != 1:
        raise SystemExit(_fail(f"resolved task version 非法: {rt.get('version')!r}"))
    return rt


def main() -> int:
    ap = argparse.ArgumentParser(description="按 resolved task 运行 UE 导入/渲染")
    ap.add_argument("--resolved-task", required=False, help="resolved task JSON 路径（或环境变量 C5_RESOLVED_TASK）")
    ap.add_argument(
        "--mode", default=None,
        choices=["sequence", "annotations", "full", "render", "pose-finalize"],
        help="覆盖执行模式（缺省 full）",
    )
    args = ap.parse_args()
    # MCP run_python_file 不支持传 argv，用环境变量 C5_RUN_MODE 兜底（--mode 优先）
    mode = args.mode or os.environ.get("C5_RUN_MODE") or "full"
    rt = _load_resolved_task(args)

    import unreal  # noqa: F401  （确保 unreal 可用）

    ue_profile = rt.get("ue_profile") or {}
    ann_cfg = dict(ue_profile.get("annotation_export") or {})
    pose_cfg = (rt.get("postprocess") or {}).get("yolo_pose") or {}
    pose_enabled = bool(pose_cfg.get("enabled", False))
    episode_dir = Path(rt["trajectory_output"])
    dataset_root = Path(rt["dataset_root"])
    mapping_path = Path(rt["actor_mapping"])
    seq_list = ue_profile.get("sequences") or []
    seq_pkg = ue_profile.get("sequence_package_path") or "/Game/FutsalMOT/Sequences"
    replace_existing = bool(ue_profile.get("replace_existing", True))
    ball_rolling = ue_profile.get("ball_rolling") or None
    episode_name = rt.get("episode_name") or "episode"
    requested_camera_ids = os.environ.get("C5_CAMERA_IDS")
    camera_ids = (
        [camera_id.strip() for camera_id in requested_camera_ids.split(",") if camera_id.strip()]
        if requested_camera_ids
        else None
    )
    camera_selection = _resolve_runtime_camera_selection(
        rt,
        legacy_sequences=seq_list,
        legacy_cameras=ann_cfg.get("cameras") or ["CineCam_01"],
        camera_ids=camera_ids,
    )
    camera_entries = camera_selection["entries"]
    if camera_selection["canonical"]:
        _apply_resolved_anchor_cameras(camera_entries, ann_cfg)
    seq_list = camera_selection["sequences"]
    cameras = camera_selection["cameras"] or ["CineCam_01"]
    if camera_selection["canonical"]:
        # 旧模块仍接受 annotation_export.cameras；这里仅生成兼容适配值，
        # 选择来源始终是 resolved simulation.camera + ue.camera_mapping。
        ann_cfg["cameras"] = cameras
        resolution = camera_selection["resolution"]
        render_cfg = dict(ann_cfg.get("render_rgb") or {})
        configured = (
            render_cfg.get("output_resolution_x"),
            render_cfg.get("output_resolution_y"),
        )
        if configured != (None, None) and configured != tuple(resolution):
            raise RuntimeError(
                f"canonical camera resolution={tuple(resolution)} "
                f"与 render_rgb.output_resolution={configured} 不一致"
            )
        render_cfg["output_resolution_x"], render_cfg["output_resolution_y"] = resolution
        ann_cfg["render_rgb"] = render_cfg

    # 帧数：实际输出帧 = num_steps × factor（target_fps>10 时 Hermite 插值）。
    # 不能只取 export_profile.num_steps（否则 target_fps=30 时 playback 只到 num_steps，
    # 导致 sequence 有 num_steps×factor 帧却只渲染前 num_steps 帧 → img1 缺帧）。
    _export_ns = int((rt.get("export_profile") or {}).get("num_steps") or 3)
    _export_fps = int((rt.get("export_profile") or {}).get("target_fps") or 10)
    _factor = max(1, _export_fps // 10)
    meta_timing_num_steps = _export_ns * _factor

    if not episode_dir.is_dir():
        return _fail(f"trajectory 目录不存在（先运行 grf-ue task export）: {episode_dir}")
    if not mapping_path.is_file():
        return _fail(f"actor mapping 不存在: {mapping_path}")

    print(f"run_task: task={rt.get('task_id')} episode={episode_name} mode={mode}")
    print(f"  trajectory: {episode_dir}")
    print(f"  dataset output root: {dataset_root}  (episode_id -> {dataset_root / episode_name})")
    print(f"  sequences: {[s.get('name') for s in seq_list]}")
    print(f"  cameras: {cameras}")
    print(f"  num_steps={meta_timing_num_steps}  yolo_pose enabled: {pose_enabled}")

    # ── pose-finalize 模式：渲染完成后导出 Runtime Pose + 生成 COCO17 ──
    if mode == "pose-finalize":
        os.environ["C5_POSE_TASK"] = str(args.resolved_task or os.environ.get("C5_RESOLVED_TASK"))
        os.environ["C5_POSE_EXPECT"] = str(meta_timing_num_steps)
        # 防止旧 pose_capture.jsonl / pose_session.json 被误读：
        # 先删除旧文件，pose_capture_export 只在 capture_complete=True 时才写新文件。
        ep_out = dataset_root / episode_name
        for stale in ("pose_capture.jsonl", "pose_session.json"):
            stale_path = ep_out / stale
            if stale_path.exists():
                stale_path.unlink()
                print(f"  [防残留] 删除旧 {stale}（避免误读 incomplete 数据）")
        print("\n--- Runtime Pose 导出（pose_capture_export）---")
        import pose_capture_export
        pose_capture_export.main()
        # pose_capture_export 在 incomplete 时 fail-fast（不写文件）。
        # 必须检查新文件存在 + pose_session.capture_complete=True 才继续 COCO17。
        pc_path = ep_out / "pose_capture.jsonl"
        ps_path = ep_out / "pose_session.json"
        if not pc_path.is_file() or not ps_path.is_file():
            return _fail("pose_capture_export 未生成 pose_capture.jsonl（Runtime Pose 捕获 incomplete 或失败），禁止 COCO17")
        import json as _json
        ps = _json.loads(ps_path.read_text(encoding="utf-8"))
        if not ps.get("capture_complete"):
            return _fail(f"pose_session.capture_complete=False (captured={ps.get('captured_frame_count')}/expected={ps.get('expected_frame_count')})，禁止 COCO17")
        print("\n--- COCO17 3D/2D 生成（build_coco17）---")
        os.environ["C5_EPISODE_DIR"] = str(dataset_root / episode_name)
        os.environ["C5_COCO17_CAMERAS"] = ",".join(cameras)
        import build_coco17
        rc = build_coco17.main()
        if rc:
            return _fail("build_coco17 失败（见上）")
        # 生成 pose_keypoints.jsonl（P1 annotate-pose 契约：从 Runtime Pose COCO17_3D 桥接）
        print("\n--- 生成 pose_keypoints.jsonl（Runtime Pose → P1 pose_annotator 契约）---")
        _runtime_pose_to_pose_keypoints(dataset_root / episode_name, cameras,
                                        meta_timing_num_steps, ann_cfg)
        print("\nDone（pose-finalize）。")
        return 0

    # ── full / render / sequence / annotations 模式 ──
    from import_grf_episode import create_sequence, load_episode, load_mapping
    from annotation_exporter import export_annotations
    from render_episode import render_sequences

    if mode in ("full", "render"):
        meta, frames = load_episode(episode_dir)
        mapping = load_mapping(mapping_path)
        if mode == "full":
            print("\n--- 全流程：创建 Level Sequence ---")
            create_sequence(
                meta, frames, mapping, replace_existing, seq_pkg, seq_list, ball_rolling
            )
            export_annotations(episode_dir, mapping_path, dataset_root, ann_cfg)
        # 序列 playback 限到数据集帧数（BurnIn 捕获与 RGB 帧一致，避免多渲 1 帧）
        for s in seq_list:
            seq = unreal.load_asset(f"{seq_pkg}/{s['name']}")
            if seq is not None:
                seq.set_playback_start(0)
                seq.set_playback_end(meta_timing_num_steps)
                print(f"  序列 {s['name']} playback -> [0, {meta_timing_num_steps})")
        # Runtime Pose prep：多 camera 下，Recorder CDO 是共享的（5 个 actor），
        # 只能 prep 一个 camera 的 slot。Runtime Pose 捕获的是世界骨骼 transform
        # （camera 无关），故只需一个 camera 的完整捕获即可得到正确 pose_capture。
        # 用首个 camera 的 slot prep（pose_capture_export 也以首个 camera 为 primary）。
        if pose_enabled and mode == "full":
            print(f"\n--- Runtime Pose prep（primary={cameras[0]}）---")
            from pose_render import _prep_recorders
            _prep_recorders(episode_name, cameras[0])
            for i in range(5):
                try:
                    unreal.GameplayStatics.delete_game_in_slot(
                        f"PoseCapture_{episode_name}_{cameras[0]}_G{i}", 0)
                except Exception:
                    pass
            print(f"  {cameras[0]}: slots 已清空")
        render_sequences(
            seq_list, ann_cfg, seq_pkg, episode_dir, dataset_root, mapping_path
        )
        print("\n已提交。MRQ 渲染为异步执行（不阻塞编辑器），完成后自动复制 RGB 到 img1/"
              "（及统计 Instance-ID Mask 对齐帧）并写 render_summary.json。")
        print(f"Runtime Pose 捕获：BurnIn 写入 {cameras[0]} 的 5×SaveGame（世界 3D，camera 无关）。")
        print("渲染完成后运行：py run_task.py --mode pose-finalize（导出 pose_capture + COCO17）。")
        return 0

    if mode == "annotations":
        export_annotations(episode_dir, mapping_path, dataset_root, ann_cfg)
        if pose_enabled:
            print("\n--- 导出 Pose 关键点（Legacy pose_export，Editor 逐帧采样）---")
            from pose_export import export_pose_keypoints
            export_pose_keypoints(episode_dir, mapping_path, dataset_root, ann_cfg, pose_cfg)
        print("\nDone（annotations）。")
        return 0

    if mode == "sequence":
        meta, frames = load_episode(episode_dir)
        mapping = load_mapping(mapping_path)
        create_sequence(meta, frames, mapping, replace_existing, seq_pkg, seq_list, ball_rolling)
        print("\nDone（sequence）。")
        return 0

    return _fail(f"未知模式: {mode!r}")


if __name__ == "__main__":
    sys.exit(main())
