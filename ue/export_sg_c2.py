"""导出 SG_PoseCapture C2（5 骨骼）数据到 JSONL。

用法（Unreal Editor Python Console）：
    py "D:/projects/FustalMOT_UEDataset/Content/FutsalMOT/code/ue/export_sg_c2.py"

写：G:\\FutsalMOT_Dataset\\episode_bp_frame_sync\\pose_capture_c2.jsonl
每行：{root, shot, game_time, actor_id, bone, x, y, z}（共 450 行）
"""

import json
import sys
from pathlib import Path

LOG = []
LOG_PATH = Path(r"D:\projects\FustalMOT_UEDataset\Content\FutsalMOT\code\.futsalmot\export_sg_c2.log")
SLOT_NAME = "PoseCapture"
OUT_PATH = Path(r"G:\FutsalMOT_Dataset\episode_bp_frame_sync\pose_capture_c2.jsonl")


def _log(msg):
    print(msg)
    LOG.append(str(msg))


def _flush():
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        LOG_PATH.write_text("\n".join(LOG), encoding="utf-8")
    except Exception as e:
        print(f"  写日志失败: {e}")


def main():
    import unreal
    _log("======== 导出 SG_PoseCapture（C2：5 骨骼）========")

    sg = unreal.GameplayStatics.load_game_from_slot(SLOT_NAME, 0)
    if sg is None:
        _log("  ERROR: SaveGame 为空")
        _flush()
        return

    cap = list(sg.get_editor_property("capture_indices"))
    shot = list(sg.get_editor_property("shot_frames"))
    gt = list(sg.get_editor_property("game_times"))
    aid = list(sg.get_editor_property("actor_ids"))
    bone = list(sg.get_editor_property("bone_names"))
    loc = list(sg.get_editor_property("world_locations"))

    rows = []
    for i in range(len(cap)):
        rows.append({
            "root": cap[i],
            "shot": shot[i],
            "game_time": round(gt[i], 6),
            "actor_id": aid[i],
            "bone": bone[i],
            "x": round(loc[i].x, 3),
            "y": round(loc[i].y, 3),
            "z": round(loc[i].z, 3),
        })

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    _log(f"  导出 {len(rows)} 行 -> {OUT_PATH}")

    _flush()


if __name__ == "__main__":
    main()
    print("\n脚本已执行，结果写入 export_sg_c2.log")