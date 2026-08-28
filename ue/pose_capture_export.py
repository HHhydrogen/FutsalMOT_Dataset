"""C5.2-B 正式 Pose 导出/校验（崩溃安全版）：completeness 全部由 Python 计算。

多 camera 支持（C5.3-P3）：
- Runtime Pose 捕获的是世界骨骼 transform（camera 无关）。
- Recorder CDO 是共享的（5 个 actor），只能 prep 一个 camera 的 slot；
  prep 首个 camera（run_task full 以 cameras[0] 为 primary）。
- 世界 3D 数据跨 camera 一致，故 pose_capture.jsonl 只写一份（primary camera）。
- pose_capture_export 只导出 primary camera 的 slot（其他 camera 的 slot 为空，符合预期）。

- captured = 唯一 root frame 数（帧数语义，非 raw sample 数）
- expected = C5_POSE_EXPECT（来自任务）
- capture_complete = captured == expected
- incomplete → fail-fast（禁止生成正式 Pose）
- 通过后回写每个 SaveGame 的完整性元数据
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


def _export_camera(unreal, episode_id, camera, expected):
    """导出单个 camera 的 SaveGame slot，返回 (rows, captured, total, sgs) 或 None（fail）。"""
    slots = [f"PoseCapture_{episode_id}_{camera}_G{i}" for i in range(5)]
    cap, shot, gt, aid, bone, loc, rot = [], [], [], [], [], [], []
    sgs = {}
    missing = []
    for sl in slots:
        sg = unreal.GameplayStatics.load_game_from_slot(sl, 0)
        if sg is None:
            missing.append(sl)
            _log(f"  [{camera}/{sl}] -> None")
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
        _log(f"  FAIL [{camera}]: 缺失 slot: {missing}（Pose 未捕获）")
        return None
    captured = len(set(cap))
    total = len(cap)
    first_root = min(cap)
    last_root = max(cap)
    complete = (captured == expected)
    _log(f"  [{camera}] total={total} captured={captured} expected={expected} "
         f"first={first_root} last={last_root} complete={complete}")
    # 回写 SG 完整性元数据
    try:
        for sl, sg in sgs.items():
            sg.set_editor_property("capture_complete", complete)
            sg.set_editor_property("captured_frame_count", captured)
            sg.set_editor_property("expected_frame_count", expected)
            sg.set_editor_property("first_root_frame", first_root)
            sg.set_editor_property("last_root_frame", last_root)
            unreal.GameplayStatics.save_game_to_slot(sg, sl, 0)
    except Exception as e:
        _log(f"  [{camera}] 回写 SG 元数据 ERR: {type(e).__name__} {e}")
    if not complete:
        return None  # incomplete → fail-fast
    if len(set(aid)) != 10 or set(bone) != EXPECT_BONES:
        _log(f"  FAIL [{camera}]: 结构异常 actors={sorted(set(aid))} bones_ok={set(bone) == EXPECT_BONES}")
        return None
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
    return {"rows": rows, "captured": captured, "total": total, "first": first_root,
            "last": last_root, "session_id": f"{episode_id}_{camera}"}


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
    cameras = ann_cfg.get("cameras") or ["CineCam_01"]
    # Runtime Pose 只从 primary camera 导出（世界 3D camera 无关，Recorder CDO 共享）
    primary = cameras[0]
    expected = int(os.environ.get("C5_POSE_EXPECT", "3"))
    _log(f"======== pose_capture_export：{episode_id} primary={primary} expected={expected} ========")
    r = _export_camera(unreal, episode_id, primary, expected)
    if r is None:
        _log(f"  FAIL: Pose capture incomplete ({primary})，禁止生成正式 Pose")
        _flush()
        return
    # 写 pose_capture.jsonl + pose_session.json
    ep_dir = Path(dataset_root) / episode_id
    ep_dir.mkdir(parents=True, exist_ok=True)
    (ep_dir / "pose_capture.jsonl").write_text(
        "\n".join(json.dumps(row) for row in r["rows"]) + "\n", encoding="utf-8")
    meta = {
        "episode_id": episode_id,
        "cameras": cameras,
        "primary_camera": primary,
        "expected_frame_count": expected,
        "captured_frame_count": r["captured"],
        "capture_complete": r["captured"] == expected,
        "total_samples": r["total"],
        "first_root_frame": r["first"],
        "last_root_frame": r["last"],
        "session_id": r["session_id"],
    }
    (ep_dir / "pose_session.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    _log(f"  pose_capture.jsonl 已写: {ep_dir / 'pose_capture.jsonl'}（{len(r['rows'])} 行，camera={primary}）")
    _log(f"  pose_session.json 已写: {ep_dir / 'pose_session.json'}（primary={primary}, cameras={cameras}）")
    _log(f"  PASS: captured={r['captured']} expected={expected} total={r['total']}")
    _flush()


if __name__ == "__main__":
    main()
