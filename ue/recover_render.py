"""从已有 MRQ render/ 目录恢复 img1/（无需重新渲染）。

场景：上一次 MRQ 渲染已把 PNG 输出到各 camera 的 render/，但完成回调
（finished delegate / watchdog）未触发，导致 img1/ 为空。本脚本从现有
render/ 帧复制对齐帧到 img1/，并写 render_summary.json。

用法（推荐，读取 resolved task）：
    uv run python ue/recover_render.py --resolved-task <resolved-task.json>

或 UE 控制台：
    py "D:/path/to/code/ue/recover_render.py" --resolved-task "<resolved-task.json>"

旧流程（根目录 ue_import_config.json）已移除。纯 Python，不依赖 unreal，
无需编辑器即可完成恢复。
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from render_episode import recover_render_to_img1  # noqa: E402

RESOLVED_TASK_SCHEMA = "futsalmot_resolved_task"


def main() -> int:
    ap = argparse.ArgumentParser(description="从 render/ 恢复 img1/")
    ap.add_argument("--resolved-task", required=True, help="resolved task JSON 路径")
    args = ap.parse_args()

    rt_path = Path(args.resolved_task)
    if not rt_path.is_file():
        print(f"ERROR: resolved task 不存在: {rt_path}")
        return 1
    with open(rt_path, encoding="utf-8") as f:
        rt = json.load(f)
    if rt.get("schema") != RESOLVED_TASK_SCHEMA:
        print(f"ERROR: resolved task schema 非法: {rt.get('schema')!r}")
        return 2

    episode = Path(rt["trajectory_output"])
    dataset_root = Path(rt["dataset_root"])
    ue_profile = rt.get("ue_profile") or {}
    ann = dict(ue_profile.get("annotation_export") or {})
    ann["output_dir"] = str(dataset_root)
    seqs = ue_profile.get("sequences") or []
    status, _per_cam = recover_render_to_img1(seqs, ann, episode, dataset_root)
    print(f"恢复状态: {status}")
    return 0 if status == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
