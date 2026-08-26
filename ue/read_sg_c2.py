"""读取 SG_PoseCapture，验证 C2（5 骨骼）采样。

用法（Unreal Editor Python Console）：
    py "D:/projects/FustalMOT_UEDataset/Content/FutsalMOT/code/ue/read_sg_c2.py"

验证：
  1. total_samples == 450（90 帧 × 5 骨骼）
  2. 每个 root_frame 0..89 恰好 5 条记录
  3. 无缺失、无重复、无跨帧
  4. 5 个骨骼名齐全（hand_l/lowerarm_l/thigh_l/calf_l/foot_l）
"""

import sys
from pathlib import Path

LOG = []
LOG_PATH = Path(r"D:\projects\FustalMOT_UEDataset\Content\FutsalMOT\code\.futsalmot\read_sg_c2.log")
SLOT_NAME = "PoseCapture"
EXPECT_BONES = ["hand_l", "lowerarm_l", "thigh_l", "calf_l", "foot_l"]


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


def main():
    import unreal
    _log("======== 读取 SG_PoseCapture（C2：5 骨骼验证）========")

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
    total = sg.get_editor_property("total_samples")

    _log(f"[2] total_samples = {total}")
    _log(f"  len(cap)={len(cap)} len(shot)={len(shot)} len(gt)={len(gt)}")
    _log(f"  len(aid)={len(aid)} len(bone)={len(bone)} len(loc)={len(loc)}")

    ok_len = (len(cap) == len(shot) == len(gt) == len(aid) == len(bone) == len(loc))
    _log(f"  六数组等长: {ok_len}")
    _log(f"  len == 450: {len(cap) == 450}")

    # 每帧 5 条校验
    _log(f"\n[3] 每 root_frame 条数统计:")
    from collections import Counter
    cnt = Counter(cap)
    bad_frames = [f for f in range(90) if cnt.get(f, 0) != 5]
    _log(f"  root_frame 覆盖 0..89: {all(f in cnt for f in range(90))}")
    _log(f"  每帧恰 5 条: {not bad_frames}")
    if bad_frames:
        _log(f"  异常帧(≠5条): {bad_frames[:20]} 共{len(bad_frames)}帧")
    _log(f"  root_frame 取值范围: {min(cap) if cap else '-'}..{max(cap) if cap else '-'}")

    # 每帧 5 条顺序是否连续（同帧 5 条相邻）
    seq_ok = True
    for f in range(90):
        idxs = [i for i, v in enumerate(cap) if v == f]
        if idxs and max(idxs) - min(idxs) != 4:
            seq_ok = False
    _log(f"  同帧 5 条相邻连续: {seq_ok}")

    # 骨骼名齐全
    _log(f"\n[4] 骨骼名集合: {sorted(set(bone))}")
    _log(f"  覆盖 5 骨骼: {set(EXPECT_BONES).issubset(set(bone))}")

    # 帧内 5 骨骼顺序（同一 root_frame 内应含全部 5 个）
    frame_bones_ok = True
    for f in [0, 44, 89]:
        bones_in_frame = [b for i, b in enumerate(bone) if cap[i] == f]
        if set(bones_in_frame) != set(EXPECT_BONES):
            frame_bones_ok = False
            _log(f"  frame {f} 骨骼: {bones_in_frame} <- 异常")
    _log(f"  抽查帧 0/44/89 各含 5 骨骼: {frame_bones_ok}")

    # 抽查数据合理性
    _log(f"\n[5] 抽查帧 root=44:")
    for i, b in enumerate(bone):
        if cap[i] == 44:
            _log(f"  {b}: root={cap[i]} shot={shot[i]} t={gt[i]:.6f} {_v3(loc[i])}")
    _log(f"\n[6] 最后 10 条:")
    for i in range(len(cap) - 10, len(cap)):
        _log(f"  {bone[i]}: root={cap[i]} shot={shot[i]} {_v3(loc[i])}")

    _flush()


if __name__ == "__main__":
    main()
    print("\n脚本已执行，结果写入 read_sg_c2.log")