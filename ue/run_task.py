"""UE 统一入口：读取 resolved task，调用既有导入/渲染逻辑。

用法（Unreal Editor Python Console）：
    py "D:/path/to/code/ue/run_task.py" --resolved-task "D:/path/to/.futsalmot/runtime/<task_id>/resolved-task.json"

普通 Python CLI（grf-ue task ue-command <task>）与 UE 端读取**同一个**
resolved task，不分别实现两套路径解析；本入口不再隐式读取根目录
ue_import_config.json。

纯 UE Python + 标准库，不导入 gfootball/.venv 任何模块。
"""

import argparse
import importlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

RESOLVED_TASK_SCHEMA = "futsalmot_resolved_task"

# UE Python 会话内，已 import 过的 ue/ 模块会缓存在 sys.modules 中；多次执行本脚本
# 时先强制重载，否则会运行到磁盘上已修改但会话里仍是旧版本的代码
# （与 import_grf_episode.py 的 _UE_MODULE_NAMES 一致）。
_UE_MODULE_NAMES = (
    "camera_projection", "annotation_utils", "dataset_export",
    "scene_apply", "annotation_exporter", "render_preset", "render_episode",
    "pose_bones", "pose_export", "player_motion",
)
for _name in _UE_MODULE_NAMES:
    if _name in sys.modules:
        importlib.reload(sys.modules[_name])


def _fail(msg: str, code: int = 2) -> int:
    print(f"ERROR: {msg}", file=sys.stderr)
    return code


def main() -> int:
    ap = argparse.ArgumentParser(description="按 resolved task 运行 UE 导入/渲染")
    ap.add_argument("--resolved-task", required=True, help="resolved task JSON 路径")
    ap.add_argument(
        "--mode", default=None,
        choices=["sequence", "annotations", "full", "render"],
        help="覆盖执行模式（缺省 full）",
    )
    args = ap.parse_args()

    rt_path = Path(args.resolved_task)
    if not rt_path.is_file():
        return _fail(f"resolved task 不存在: {rt_path}")
    with open(rt_path, encoding="utf-8") as f:
        rt = json.load(f)

    if rt.get("schema") != RESOLVED_TASK_SCHEMA:
        return _fail(f"resolved task schema 非法: {rt.get('schema')!r}")
    if rt.get("version") != 1:
        return _fail(f"resolved task version 非法: {rt.get('version')!r}")

    ue_profile = rt.get("ue_profile") or {}
    ann_cfg = dict(ue_profile.get("annotation_export") or {})
    # YOLO Pose 配置（与 P1 postprocess 同一块，见 configs/README「yolo_pose」）
    pose_cfg = (rt.get("postprocess") or {}).get("yolo_pose") or {}
    pose_enabled = bool(pose_cfg.get("enabled", False))
    episode_dir = Path(rt["trajectory_output"])
    dataset_root = Path(rt["dataset_root"])
    mapping_path = Path(rt["actor_mapping"])
    seq_list = ue_profile.get("sequences") or []
    seq_pkg = ue_profile.get("sequence_package_path") or "/Game/FutsalMOT/Sequences"
    replace_existing = bool(ue_profile.get("replace_existing", True))
    ball_rolling = ue_profile.get("ball_rolling") or None
    mode = args.mode or "full"

    if not episode_dir.is_dir():
        return _fail(f"trajectory 目录不存在（先运行 grf-ue task export）: {episode_dir}")
    if not mapping_path.is_file():
        return _fail(f"actor mapping 不存在: {mapping_path}")

    print(f"run_task: task={rt.get('task_id')} episode={rt.get('episode_name')} mode={mode}")
    print(f"  trajectory: {episode_dir}")
    print(f"  dataset output root: {dataset_root}  (episode_id -> {dataset_root / rt.get('episode_name')})")
    print(f"  sequences: {[s.get('name') for s in seq_list]}")
    print(f"  cameras: {ann_cfg.get('cameras')}")
    print(f"  yolo_pose enabled: {pose_enabled}")

    # 复用 import_grf_episode / annotation_exporter / render_episode / pose_export 的既有逻辑
    from import_grf_episode import create_sequence, load_episode, load_mapping
    from annotation_exporter import export_annotations
    from render_episode import render_sequences
    from pose_export import export_pose_keypoints

    if mode in ("full", "render"):
        meta, frames = load_episode(episode_dir)
        mapping = load_mapping(mapping_path)
        if mode == "full":
            print("\n--- 全流程：创建 Level Sequence ---")
            create_sequence(
                meta, frames, mapping, replace_existing, seq_pkg, seq_list, ball_rolling
            )
            export_annotations(episode_dir, mapping_path, dataset_root, ann_cfg)
            if pose_enabled:
                print("\n--- 导出 Pose 关键点（世界 3D）---")
                export_pose_keypoints(episode_dir, mapping_path, dataset_root, ann_cfg, pose_cfg)
        render_sequences(
            seq_list, ann_cfg, seq_pkg, episode_dir, dataset_root, mapping_path
        )
        print("\n已提交。MRQ 渲染为异步执行（不阻塞编辑器），完成后自动复制 RGB 到 img1/"
              "（及统计 Instance-ID Mask 对齐帧）并写 render_summary.json。")
        return 0

    if mode == "annotations":
        export_annotations(episode_dir, mapping_path, dataset_root, ann_cfg)
        if pose_enabled:
            print("\n--- 导出 Pose 关键点（世界 3D）---")
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
