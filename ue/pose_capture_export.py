"""C5.2-B 正式 Pose 导出/校验（崩溃安全版）：completeness 全部由 Python 计算。

- captured = 唯一 root frame 数（帧数语义，非 raw sample 数）
- expected = C5_POSE_EXPECT（来自任务）
- capture_complete = captured == expected
- incomplete → fail-fast（禁止生成正式 Pose）
- 通过后回写每个 SaveGame 的完整性元数据（capture_complete/captured/expected/first/last）
- 写 pose_capture.jsonl + pose_session.json（正式持久化产物）

env:
  C5_POSE_TASK    resolved task JSON 路径（必须）
  C5_POSE_EXPECT  预期帧数（可选；缺省 3）

用法（Unreal Editor Python，FutsalMOTTools.run_python_file）：
    py ".../pose_capture_export.py"
"""

import json
import os
from pathlib import Path

LOG = []
LOG_PATH = Path(r"D:\projects\FustalMOT_UEDataset\Content\FutsalMOT\code\.futsalmot\pose_capture_export.log")
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
    rt_path = os.environ.get("C5_POSE_TASK")
    if not rt_path:
        _log("  ERROR: C5_POSE_TASK 未设置")
        _flush()
        return
    rt = json.loads(Path(rt_path).read_text(encoding="utf-8"))
    ann_cfg = rt["ue_profile"].get("annotation_export") or {}
    episode_id = rt.get("episode_name") or "episode"
    dataset_root = rt.get("dataset_root")
    camera = (ann_cfg.get("cameras") or ["CineCam_01"])[0]
    expected = int(os.environ.get("C5_POSE_EXPECT", "3"))
    _log(f"======== pose_capture_export：{episode_id} camera={camera} expected={expected} ========")

    slots = [f"PoseCapture_{episode_id}_{camera}_G{i}" for i in range(5)]
    cap, shot, gt, aid, bone, loc, rot = [], [], [], [], [], [], []
    sgs = {}
    missing = []
    for sl in slots:
        sg = unreal.GameplayStatics.load_game_from_slot(sl, 0)
        if sg is None:
            missing.append(sl)
            _log(f"  [{sl}] -> None")
            continue
        sgs[sl] = sg
        cap += list(sg.get_editor_property("capture_indices"))
        shot += list(sg.get_editor_property("shot_frames"))
        gt += list(sg.get_editor_property("game_times"))
        aid += list(sg.get_editor_property("actor_ids"))
        bone += list(sg.get_editor_property("bone_names"))
        loc += list(sg.get_editor_property("world_locations"))
        rot += list(sg.get_editor_property("world_rotations"))
    if missing:
        _log(f"  FAIL: 缺失 slot: {missing}（Pose 未捕获）")
        _flush()
        return

    # 帧数语义：唯一 root frame 数
    captured = len(set(cap))
    total = len(cap)
    first_root = min(cap)
    last_root = max(cap)
    capture_complete = (captured == expected)
    _log(f"  total={total} captured={captured} expected={expected} first={first_root} last={last_root} complete={capture_complete}")

    # 总是回写 SG 完整性元数据（SaveGame = 传输桥；完整性语义由 Python 落盘，incomplete 也标记 false）
    try:
        for sl, sg in sgs.items():
            sg.set_editor_property("capture_complete", capture_complete)
            sg.set_editor_property("captured_frame_count", captured)
            sg.set_editor_property("expected_frame_count", expected)
            sg.set_editor_property("first_root_frame", first_root)
            sg.set_editor_property("last_root_frame", last_root)
            unreal.GameplayStatics.save_game_to_slot(sg, sl, 0)
        _log(f"  SaveGame 元数据已回写（capture_complete={capture_complete} 等）")
    except Exception as e:
        _log(f"  回写 SG 元数据 ERR: {type(e).__name__} {e}")

    if not capture_complete:
        _log(f"  FAIL: Pose capture incomplete: expected={expected} captured={captured}")
        _log("  （禁止生成正式 Pose 数据）")
        _flush()
        return
    if len(set(aid)) != 10 or set(bone) != EXPECT_BONES:
        _log(f"  FAIL: 结构异常 actors={sorted(set(aid))} bones_ok={set(bone) == EXPECT_BONES}")
        _flush()
        return

    lib = getattr(unreal, "MathLibrary", None)
    rows = []
    for i in range(len(cap)):
        q = lib.conv_rotator_to_quaternion(rot[i])
        rows.append({
            "root": cap[i], "shot": shot[i], "game_time": round(gt[i], 6),
            "actor_id": aid[i], "bone": bone[i],
            "x": round(loc[i].x, 3), "y": round(loc[i].y, 3), "z": round(loc[i].z, 3),
            "qx": round(q.x, 6), "qy": round(q.y, 6), "qz": round(q.z, 6), "qw": round(q.w, 6),
        })
    ep_dir = Path(dataset_root) / episode_id
    ep_dir.mkdir(parents=True, exist_ok=True)
    (ep_dir / "pose_capture.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    meta = {
        "episode_id": episode_id, "camera_id": camera,
        "expected_frame_count": expected, "captured_frame_count": captured,
        "capture_complete": True, "total_samples": total,
        "first_root_frame": first_root, "last_root_frame": last_root,
        "session_id": f"{episode_id}_{camera}",
    }
    (ep_dir / "pose_session.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    _log(f"  pose_capture.jsonl 已写: {ep_dir / 'pose_capture.jsonl'}（{len(rows)} 行）")
    _log(f"  pose_session.json 已写: {ep_dir / 'pose_session.json'}")
    _log(f"  PASS: captured={captured} expected={expected} total={total}")
    _flush()


if __name__ == "__main__":
    main()