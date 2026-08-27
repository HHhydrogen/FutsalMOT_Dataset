"""C5.2 测试 export/validate（崩溃安全方案）：completeness 由 Python 权威计算。

- captured = 唯一 root frame 数（帧数语义）
- expected = C5_EXPECTED_FRAMES
- capture_complete = captured == expected
- 总是回写 SaveGame 完整性元数据（incomplete → capture_complete=false）
- incomplete → fail-fast（禁止导出）
- complete → 写 .futsalmot/c5_pose_capture.jsonl（供 Test B hash 比较）

env:
  C5_EXPECTED_FRAMES  预期输出帧数（默认 3）

用法（Unreal Editor Python，FutsalMOTTools.run_python_file）：
    py ".../c5_export.py"
"""

import json
import os
from pathlib import Path

LOG = []
LOG_PATH = Path(r"D:\projects\FustalMOT_UEDataset\Content\FutsalMOT\code\.futsalmot\c5_export.log")
POSE_JSONL = Path(r"D:\projects\FustalMOT_UEDataset\Content\FutsalMOT\code\.futsalmot\c5_pose_capture.jsonl")
EPISODE = "c5test"
CAMERA = "CineCam_01"
SLOTS = [f"PoseCapture_{EPISODE}_{CAMERA}_G{i}" for i in range(5)]
EXPECT_BONES = {
    "head", "upperarm_l", "upperarm_r", "lowerarm_l", "lowerarm_r",
    "hand_l", "hand_r", "thigh_l", "thigh_r", "calf_l", "calf_r",
    "foot_l", "foot_r",
}


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
    expected = int(os.environ.get("C5_EXPECTED_FRAMES", "3"))
    _log(f"======== C5.2 export/validate：expected={expected} ========")

    all_cap, all_aid, all_bone = [], [], []
    missing = []
    sgs = {}
    per_slot = {}
    for sl in SLOTS:
        sg = unreal.GameplayStatics.load_game_from_slot(sl, 0)
        if sg is None:
            missing.append(sl)
            _log(f"  [{sl}] -> None（无 SaveGame）")
            continue
        sgs[sl] = sg
        cap = list(sg.get_editor_property("capture_indices"))
        aid = list(sg.get_editor_property("actor_ids"))
        bone = list(sg.get_editor_property("bone_names"))
        total = sg.get_editor_property("total_samples")
        complete = sg.get_editor_property("capture_complete")
        session = sg.get_editor_property("session_id")
        captured = len(set(cap))
        roots = sorted(set(cap))
        per_slot[sl] = {"total": total, "captured": captured, "roots": roots,
                        "complete": complete, "session": session}
        _log(f"  [{sl}] total={total} captured={captured} roots={roots} complete={complete} session={session}")
        all_cap += cap
        all_aid += aid
        all_bone += bone

    if missing:
        _log(f"  ERROR: 缺失 slot: {missing}")
        _flush()
        return

    total_all = len(all_cap)
    captured_all = len(set(all_cap))
    first_root = min(all_cap)
    last_root = max(all_cap)
    capture_complete = (captured_all == expected)
    _log(f"  合并 total={total_all} captured={captured_all} expected={expected} first={first_root} last={last_root}")

    # 总是回写 SG 完整性元数据（Python 权威；incomplete 也标记 false）
    try:
        for sl, sg in sgs.items():
            sg.set_editor_property("capture_complete", capture_complete)
            sg.set_editor_property("captured_frame_count", captured_all)
            sg.set_editor_property("expected_frame_count", expected)
            sg.set_editor_property("first_root_frame", first_root)
            sg.set_editor_property("last_root_frame", last_root)
            unreal.GameplayStatics.save_game_to_slot(sg, sl, 0)
        _log(f"  SaveGame 元数据已回写（capture_complete={capture_complete}）")
    except Exception as e:
        _log(f"  回写 SG 元数据 ERR: {type(e).__name__} {e}")

    if not capture_complete:
        _log(f"  FAIL: Pose capture incomplete: expected={expected} captured={captured_all}")
        _log("  （禁止导出正式 Pose）")
        _flush()
        return

    # 完整性校验
    ok = True
    if len(set(all_aid)) != 10:
        _log(f"  actor 集合异常: {sorted(set(all_aid))}")
        ok = False
    if set(all_bone) != EXPECT_BONES:
        _log(f"  bone 集合异常: {sorted(set(all_bone))}")
        ok = False
    _log(f"  actor 集合={sorted(set(all_aid))}（期望 10）")
    _log(f"  bone 集合完整={set(all_bone) == EXPECT_BONES}")
    _log(f"  total={total_all}（期望 {expected * 10 * 13}）")
    if not ok:
        _log(f"  FAIL: 结构完整性异常")
        _flush()
        return

    rows = []
    for i in range(len(all_cap)):
        rows.append({"root": all_cap[i], "actor_id": all_aid[i], "bone": all_bone[i]})
    POSE_JSONL.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    _log(f"  pose_capture.jsonl 已写: {POSE_JSONL}（{len(rows)} 行）")
    _log(f"  完整性 PASS: total={total_all} captured={captured_all} complete={capture_complete}")
    _flush()


if __name__ == "__main__":
    main()