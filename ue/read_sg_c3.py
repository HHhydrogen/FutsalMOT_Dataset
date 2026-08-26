"""读取 SG_PoseCapture，验证 C3（13 骨 COCO17 source 集合）采样。

用法（Unreal Editor Python Console）：
    py "D:/projects/FustalMOT_UEDataset/Content/FutsalMOT/code/ue/read_sg_c3.py"

验证：
  1. total_samples == 1170（90 帧 × 13 骨骼）
  2. 每个 root_frame 0..89 恰好 13 条
  3. 无缺失、无重复、无跨帧
  4. 13 个骨骼名齐全（head + 12 肢体）
  5. world_rotations 有有效四元数
"""

import sys
from pathlib import Path

LOG = []
LOG_PATH = Path(r"D:\projects\FustalMOT_UEDataset\Content\FutsalMOT\code\.futsalmot\read_sg_c3.log")
SLOT_NAME = "PoseCapture"
EXPECT_BONES = [
    "head",
    "upperarm_l", "upperarm_r", "lowerarm_l", "lowerarm_r",
    "hand_l", "hand_r", "thigh_l", "thigh_r",
    "calf_l", "calf_r", "foot_l", "foot_r",
]


def _log(msg):
    print(msg)
    LOG.append(str(msg))


def _flush():
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        LOG_PATH.write_text("\n".join(LOG), encoding="utf-8")
    except Exception as e:
        print(f"  写日志失败: {e}")


def _v3(v):
    return (round(v.x, 3), round(v.y, 3), round(v.z, 3))


def _q4(q):
    return (round(q.yaw, 4), round(q.pitch, 4), round(q.roll, 4))


def main():
    import unreal
    _log("======== 读取 SG_PoseCapture（C3：13 骨验证）========")

    sg = unreal.GameplayStatics.load_game_from_slot(SLOT_NAME, 0)
    _log(f"[1] load -> {sg}")
    if sg is None:
        _log("  ERROR: 空")
        _flush()
        return

    cap = list(sg.get_editor_property("capture_indices"))
    shot = list(sg.get_editor_property("shot_frames"))
    gt = list(sg.get_editor_property("game_times"))
    aid = list(sg.get_editor_property("actor_ids"))
    bone = list(sg.get_editor_property("bone_names"))
    loc = list(sg.get_editor_property("world_locations"))
    rot = list(sg.get_editor_property("world_rotations"))
    total = sg.get_editor_property("total_samples")

    _log(f"[2] total_samples = {total}")
    _log(f"  len(cap)={len(cap)} len(shot)={len(shot)} len(gt)={len(gt)}")
    _log(f"  len(aid)={len(aid)} len(bone)={len(bone)} len(loc)={len(loc)} len(rot)={len(rot)}")

    ok_len = (len(cap) == len(shot) == len(gt) == len(aid) == len(bone) == len(loc) == len(rot))
    _log(f"  七数组等长: {ok_len}")
    _log(f"  len == 1170: {len(cap) == 1170}")

    from collections import Counter
    cnt = Counter(cap)
    bad_frames = [f for f in range(90) if cnt.get(f, 0) != 13]
    _log(f"\n[3] 每 root_frame 条数统计:")
    _log(f"  root_frame 覆盖 0..89: {all(f in cnt for f in range(90))}")
    _log(f"  每帧恰 13 条: {not bad_frames}")
    if bad_frames:
        _log(f"  异常帧(≠13条): {bad_frames[:20]} 共{len(bad_frames)}帧")
    _log(f"  root_frame 范围: {min(cap) if cap else '-'}..{max(cap) if cap else '-'}")

    # 同帧 13 条相邻连续
    seq_ok = True
    for f in range(90):
        idxs = [i for i, v in enumerate(cap) if v == f]
        if idxs and max(idxs) - min(idxs) != 12:
            seq_ok = False
    _log(f"  同帧 13 条相邻连续: {seq_ok}")

    # 骨骼名集合
    _log(f"\n[4] 骨骼名集合 ({len(set(bone))} 个): {sorted(set(bone))}")
    _log(f"  覆盖 13 骨骼: {set(EXPECT_BONES).issubset(set(bone))}")

    # 抽查帧骨骼齐全
    frame_bones_ok = True
    for f in [0, 44, 89]:
        bones_in_frame = [b for i, b in enumerate(bone) if cap[i] == f]
        if set(bones_in_frame) != set(EXPECT_BONES):
            frame_bones_ok = False
            _log(f"  frame {f} 骨骼数: {len(bones_in_frame)} <- 异常")
    _log(f"  抽查帧 0/44/89 各含 13 骨骼: {frame_bones_ok}")

    # rotation 有效性（Rotator 非全零）
    valid_rot = sum(1 for q in rot if not (q.yaw == 0 and q.pitch == 0 and q.roll == 0))
    _log(f"\n[5] world_rotations 有效数: {valid_rot}/{len(rot)}")

    # 抽查 frame 44
    _log(f"\n[6] 抽查帧 root=44:")
    for i, b in enumerate(bone):
        if cap[i] == 44:
            _log(f"  {b:<10} t={gt[i]:.6f} loc={_v3(loc[i])} q={_q4(rot[i])}")

    _flush()


if __name__ == "__main__":
    main()
    print("\n脚本已执行，结果写入 read_sg_c3.log")