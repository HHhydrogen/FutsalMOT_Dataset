"""导出 SG_PoseCapture C3（13 骨）数据到 JSONL。

用法（Unreal Editor Python Console）：
    py "D:/projects/FustalMOT_UEDataset/Content/FutsalMOT/code/ue/export_sg_c3.py"

写：G:\\FutsalMOT_Dataset\\episode_bp_frame_sync\\pose_capture_c3.jsonl
每行：{root, shot, game_time, actor_id, bone, x, y, z, qx, qy, qz, qw}（共 1170 行）
"""

import json
import sys
from pathlib import Path

LOG = []
LOG_PATH = Path(r"D:\projects\FustalMOT_UEDataset\Content\FutsalMOT\code\.futsalmot\export_sg_c3.log")
SLOT_NAME = "PoseCapture"
OUT_PATH = Path(r"G:\FutsalMOT_Dataset\episode_bp_frame_sync\pose_capture_c3.jsonl")


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
    _log("======== 导出 SG_PoseCapture（C3：13 骨）========")

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
    rot = list(sg.get_editor_property("world_rotations"))

    # Rotator → Quat：unreal.MathLibrary.conv_rotator_to_quaternion
    lib = getattr(unreal, "MathLibrary", None)
    rows = []
    for i in range(len(cap)):
        q = lib.conv_rotator_to_quaternion(rot[i])
        rows.append({
            "root": cap[i],
            "shot": shot[i],
            "game_time": round(gt[i], 6),
            "actor_id": aid[i],
            "bone": bone[i],
            "x": round(loc[i].x, 3),
            "y": round(loc[i].y, 3),
            "z": round(loc[i].z, 3),
            "qx": round(q.x, 6),
            "qy": round(q.y, 6),
            "qz": round(q.z, 6),
            "qw": round(q.w, 6),
        })

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    _log(f"  导出 {len(rows)} 行 -> {OUT_PATH}")

    _flush()


if __name__ == "__main__":
    main()
    print("\n脚本已执行，结果写入 export_sg_c3.log")