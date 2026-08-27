"""导出 SG_PoseCapture C4（10 actor）数据到 JSONL。

用法（Unreal Editor Python Console）：
    py "D:/projects/FustalMOT_UEDataset/Content/FutsalMOT/code/ue/export_sg_c4.py"

写：
  - pose_capture_c4.jsonl（11700 行：root/shot/game_time/actor_id/bone/x/y/z/qx/qy/qz/qw）
  - capture_durations_c4.jsonl（90 行：root + duration_s）
"""

import json
import sys
from pathlib import Path

LOG = []
LOG_PATH = Path(r"D:\projects\FustalMOT_UEDataset\Content\FutsalMOT\code\.futsalmot\export_sg_c4.log")
SLOTS = [f"PoseCaptureG{i}" for i in range(5)]
OUT_PATH = Path(r"G:\FutsalMOT_Dataset\episode_bp_frame_sync\pose_capture_c4.jsonl")
OUT_DURS = Path(r"G:\FutsalMOT_Dataset\episode_bp_frame_sync\capture_durations_c4.jsonl")


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
    _log("======== 导出 SG_PoseCapture（C4：5 slot 合并）========")

    # 合并 5 个 slot
    cap, shot, gt, aid, bone, loc, rot, durs = [], [], [], [], [], [], [], []
    for sl in SLOTS:
        sg = unreal.GameplayStatics.load_game_from_slot(sl, 0)
        _log(f"[1] load {sl} -> {sg}")
        if sg is None:
            _log(f"  ERROR: {sl} 空")
            continue
        cap += list(sg.get_editor_property("capture_indices"))
        shot += list(sg.get_editor_property("shot_frames"))
        gt += list(sg.get_editor_property("game_times"))
        aid += list(sg.get_editor_property("actor_ids"))
        bone += list(sg.get_editor_property("bone_names"))
        loc += list(sg.get_editor_property("world_locations"))
        rot += list(sg.get_editor_property("world_rotations"))
        durs += list(sg.get_editor_property("capture_durations"))

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

    # capture_durations（900 条：每 actor 段 1 条）
    roots_seen = []
    for v in cap:
        if v not in roots_seen:
            roots_seen.append(v)
    durs_rows = [{"root": roots_seen[(i * len(roots_seen)) // max(len(durs), 1)] if durs else i,
                  "duration_s": round(float(durs[i]), 6)} for i in range(len(durs))]
    with open(OUT_DURS, "w", encoding="utf-8") as f:
        for r in durs_rows:
            f.write(json.dumps(r) + "\n")
    _log(f"  导出 durations {len(durs_rows)} 行 -> {OUT_DURS}")

    _flush()


if __name__ == "__main__":
    main()
    print("\n脚本已执行，结果写入 export_sg_c4.log")