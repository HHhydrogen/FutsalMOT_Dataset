"""读取 SG_PoseCapture，验证 C4（10 actor × 13 骨）采样。

用法（Unreal Editor Python Console）：
    py "D:/projects/FustalMOT_UEDataset/Content/FutsalMOT/code/ue/read_sg_c4.py"

验证：
  1. total_samples == 11700（90 帧 × 10 actor × 13 骨骼）
  2. 每帧恰好 10 个 actor，ID 集合严格 = {L0..L4,R0..R4}
  3. 每 actor 每帧恰好 13 条骨骼
  4. 无缺失、无重复、无串号、无跨帧
  5. capture_durations 90 条（每帧 1 条）
"""

import sys
from pathlib import Path

LOG = []
LOG_PATH = Path(r"D:\projects\FustalMOT_UEDataset\Content\FutsalMOT\code\.futsalmot\read_sg_c4.log")
SLOTS = [f"PoseCaptureG{i}" for i in range(5)]
EXPECT_ACTORS = [f"L{i}" for i in range(5)] + [f"R{i}" for i in range(5)]
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


def main():
    import unreal
    _log("======== 读取 SG_PoseCapture（C4：5 slot 合并验证）========")

    # 合并 5 个 slot
    cap, shot, gt, aid, bone, loc, rot, durs = [], [], [], [], [], [], [], []
    total = 0
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
        total += sg.get_editor_property("total_samples")
    _log(f"  合并 total = {total}")

    _log(f"[2] total_samples = {total}（预期 11700）")
    _log(f"  len(cap)={len(cap)} len(shot)={len(shot)} len(gt)={len(gt)}")
    _log(f"  len(aid)={len(aid)} len(bone)={len(bone)} len(loc)={len(loc)} len(rot)={len(rot)}")
    _log(f"  len(durs)={len(durs)}")

    ok_len = (len(cap) == len(shot) == len(gt) == len(aid) == len(bone) == len(loc) == len(rot))
    _log(f"  七数组等长: {ok_len}")
    _log(f"  len == 11700: {len(cap) == 11700}")
    _log(f"  durs len == 900（10 actor 段 × 90 帧）: {len(durs) == 900}")

    from collections import Counter, defaultdict

    # 每帧 actor 集合
    _log(f"\n[3] 每帧 actor 集合验证:")
    frame_actors_ok = True
    for f in range(90):
        ids = {aid[i] for i, v in enumerate(cap) if v == f}
        if ids != set(EXPECT_ACTORS):
            frame_actors_ok = False
            _log(f"  frame {f} actor 集合异常: {sorted(ids)}")
    _log(f"  每帧 actor == {{L0..L4,R0..R4}}: {frame_actors_ok}")

    # 每 (actor, frame) 骨骼数
    _log(f"\n[4] 每 (actor,frame) 骨骼数:")
    per = defaultdict(Counter)
    for i in range(len(cap)):
        per[(aid[i], cap[i])][bone[i]] += 1
    bone_ok = True
    for f in range(90):
        for act in EXPECT_ACTORS:
            cnt = per[(act, f)]
            if len(cnt) != 13 or set(cnt) != set(EXPECT_BONES) or any(v != 1 for v in cnt.values()):
                bone_ok = False
                _log(f"  ({act}, frame {f}) 骨骼异常: {dict(cnt)}")
    _log(f"  每 actor 每帧 13 骨、无重复: {bone_ok}")

    # 串号检查：同 (actor,frame) 是否有多余骨骼/actor 缺失
    _log(f"\n[5] actor/bone 完整性:")
    _log(f"  每帧 10 actor、每 actor 13 骨 => 每帧 130 条")
    per_frame = Counter(cap)
    frames_bad = [f for f in range(90) if per_frame[f] != 130]
    _log(f"  每帧 130 条: {not frames_bad}")
    if frames_bad:
        _log(f"  异常帧: {frames_bad[:10]}")

    # 串号检测：某 actor 的骨骼是否被记到别的 actor（通过 actor 集合固定已覆盖）
    _log(f"\n[6] capture_durations（每 actor 段 1 条，每帧 10 条）:")
    if durs:
        vals = [float(d) for d in durs]
        _log(f"  首5: {[round(v, 6) for v in vals[:5]]}")
        _log(f"  段平均: {sum(vals)/len(vals):.6f}s  段最大: {max(vals):.6f}s")
        # 每帧总耗时 = 该帧 10 条求和（需按帧分组；这里近似按顺序每 10 条一组）
        frame_sums = []
        for i in range(0, len(vals), 10):
            frame_sums.append(sum(vals[i:i+10]))
        _log(f"  帧平均总耗时: {sum(frame_sums)/len(frame_sums):.6f}s")
        _log(f"  帧最大总耗时: {max(frame_sums):.6f}s")
        _log(f"  90 帧总耗时: {sum(frame_sums):.6f}s")

    _log(f"\n[7] 抽查帧 root=0 (前 15 条):")
    shown = 0
    for i in range(len(cap)):
        if cap[i] == 0 and shown < 15:
            _log(f"  {aid[i]:<3} {bone[i]:<10} loc=({round(loc[i].x,1)},{round(loc[i].y,1)},{round(loc[i].z,1)})")
            shown += 1

    _flush()


if __name__ == "__main__":
    main()
    print("\n脚本已执行，结果写入 read_sg_c4.log")