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
    "player_motion",
)
for _name in _UE_MODULE_NAMES:
    if _name in sys.modules:
        importlib.reload(sys.modules[_name])


def _fail(msg: str, code: int = 2) -> int:
    print(f"ERROR: {msg}", file=sys.stderr)
    return code


POSE_KEYPOINTS_SCHEMA = "grf_ue_pose_keypoints"


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
    cameras = ann_cfg.get("cameras") or ["CineCam_01"]

    # 帧数：来自 meta.timing.num_steps（resolved task export_profile.num_steps）
    meta_timing_num_steps = int((rt.get("export_profile") or {}).get("num_steps") or 3)

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
        print("\n--- Runtime Pose 导出（pose_capture_export）---")
        import pose_capture_export
        pose_capture_export.main()
        # 检查 pose_capture.jsonl 是否生成（capture_complete=True 才继续 COCO17）
        pc_path = dataset_root / episode_name / "pose_capture.jsonl"
        if not pc_path.is_file():
            return _fail("pose_capture.jsonl 未生成（Runtime Pose 捕获失败或 incomplete），跳过 COCO17")
        print("\n--- COCO17 3D/2D 生成（build_coco17）---")
        os.environ["C5_EPISODE_DIR"] = str(dataset_root / episode_name)
        os.environ["C5_COCO17_CAMERA"] = cameras[0]
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
            # C5.3：Runtime Pose prep（替代旧 pose_export Editor 采样）
            if pose_enabled:
                print("\n--- Runtime Pose prep（Recorder CDO + slot 清空 + 序列 playback）---")
                from pose_render import _prep_recorders
                for cam in cameras:
                    _prep_recorders(episode_name, cam)
                    # 清空动态 slot
                    for i in range(5):
                        try:
                            unreal.GameplayStatics.delete_game_in_slot(
                                f"PoseCapture_{episode_name}_{cam}_G{i}", 0)
                        except Exception:
                            pass
                    print(f"  {cam}: slots 已清空")
                # 序列 playback 限到数据集帧数（BurnIn 捕获与 RGB 帧一致，避免多渲 1 帧）
                for s in seq_list:
                    seq = unreal.load_asset(f"{seq_pkg}/{s['name']}")
                    if seq is not None:
                        seq.set_playback_start(0)
                        seq.set_playback_end(meta_timing_num_steps)
                        print(f"  序列 {s['name']} playback -> [0, {meta_timing_num_steps})")
        render_sequences(
            seq_list, ann_cfg, seq_pkg, episode_dir, dataset_root, mapping_path
        )
        print("\n已提交。MRQ 渲染为异步执行（不阻塞编辑器），完成后自动复制 RGB 到 img1/"
              "（及统计 Instance-ID Mask 对齐帧）并写 render_summary.json。")
        print("Runtime Pose 捕获：BurnIn OnOutputFrameStarted 每帧写入 5×SaveGame。")
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
