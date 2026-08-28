"""正式 pipeline：Runtime Pose 捕获 JSONL → COCO17（3D world + 2D projected）。

读：
  - <episode>/pose_capture.jsonl（C5.2-B 正式产物：root/shot/game_time/actor_id/bone/
    x/y/z/qx/qy/qz/qw，cm）
  - <episode>/<camera>/camera.json
写：
  - <episode>/coco17_3d.jsonl  每帧一行 {actor_id, root, keypoints_3d_m: [[x,y,z]×17]}
  - <episode>/coco17_2d.jsonl  每帧一行 {actor_id, root, keypoints_2d_px: [[x,y]×17], visible}

env:
  C5_EPISODE_DIR    episode 目录（必须；也可用 C5_POSE_TASK 从 resolved task 派生）
  C5_COCO17_CAMERA  camera 名（可选；缺省 CineCam_01）

可由 run_task --mode pose-finalize 自动调用（设置好 env 后 import main()），
也可独立运行：
    uv run python ue/build_coco17.py           （P1，env 驱动）
    py ".../ue/build_coco17.py"                （UE，env 驱动）

复用：
  - pose_bones.resolve_limb_bone_map / apply_head_offsets / normalize_world_keypoints
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pose_bones import (  # noqa: E402
    COCO_KEYPOINT_NAMES,
    resolve_limb_bone_map,
    apply_head_offsets,
    normalize_world_keypoints,
)

CM_TO_M = 0.01


def _log(msg):
    print(msg)


def load_camera(cam_dir):
    cam = json.loads((cam_dir / "camera.json").read_text(encoding="utf-8"))
    I = cam["intrinsics"]
    E = cam["extrinsics"]
    R = [E["forward"], E["right"], E["up"]]
    loc = E["world_location_m"]
    return I, R, loc


def project(world_m, I, R, loc):
    d = [world_m[0] - loc[0], world_m[1] - loc[1], world_m[2] - loc[2]]
    xc = sum(R[0][i] * d[i] for i in range(3))
    yc = sum(R[1][i] * d[i] for i in range(3))
    zc = sum(R[2][i] * d[i] for i in range(3))
    if xc <= 0:
        return None
    u = I["cx"] + I["fx"] * yc / xc
    v = I["cy"] - I["fy"] * zc / xc
    return (u, v)


def main() -> int:
    ep_str = os.environ.get("C5_EPISODE_DIR")
    if not ep_str and os.environ.get("C5_POSE_TASK"):
        rt = json.loads(Path(os.environ["C5_POSE_TASK"]).read_text(encoding="utf-8"))
        ep_str = rt["trajectory_output"]
    if not ep_str:
        _log("ERROR: 需要 C5_EPISODE_DIR（或 C5_POSE_TASK）")
        return 1
    ep = Path(ep_str)
    camera = os.environ.get("C5_COCO17_CAMERA", "CineCam_01")
    cam_dir = ep / camera
    pc = ep / "pose_capture.jsonl"
    if not pc.is_file():
        _log(f"ERROR: 缺 {pc}（先运行 pose_capture_export）")
        return 1
    if not (cam_dir / "camera.json").is_file():
        _log(f"ERROR: 缺 {cam_dir / 'camera.json'}")
        return 1

    _log(f"======== build_coco17：{ep.name} camera={camera} ========")
    rows = [json.loads(l) for l in pc.read_text(encoding="utf-8").splitlines()]
    _log(f"pose_capture 记录: {len(rows)}")

    actor_ids = sorted({r["actor_id"] for r in rows})
    by_actor_frame = {}
    for r in rows:
        by_actor_frame.setdefault((r["actor_id"], r["root"]), []).append(r)

    I, R, loc = load_camera(cam_dir)
    all_bones = sorted({r["bone"] for r in rows})
    limb_map = resolve_limb_bone_map(all_bones)
    _log(f"骨骼集合 ({len(all_bones)} 骨, 肢体映射 {len(limb_map)}/12)")

    out3d, out2d = [], []
    missing = {}
    frames_per_actor = {}
    for (aid, root), recs in sorted(by_actor_frame.items()):
        frames_per_actor.setdefault(aid, []).append(root)
        recmap = {r["bone"]: r for r in recs}
        limb = {}
        for coco, bone in limb_map.items():
            r = recmap.get(bone)
            if r is None:
                missing.setdefault((aid, coco), []).append(root)
                continue
            limb[coco] = [r["x"], r["y"], r["z"]]
        face = {}
        head = recmap.get("head")
        if head is not None:
            face = apply_head_offsets(
                [head["x"], head["y"], head["z"]],
                [head["qx"], head["qy"], head["qz"], head["qw"]],
            )
        kp_cm = normalize_world_keypoints(limb, face)
        kp_m = []
        for p in kp_cm:
            if p[0] is None:
                kp_m.append(None)
            else:
                kp_m.append([p[0] * CM_TO_M, p[1] * CM_TO_M, p[2] * CM_TO_M])
        kp_2d, vis = [], []
        for p in kp_m:
            if p is None:
                kp_2d.append(None)
                vis.append(False)
            else:
                px = project(p, I, R, loc)
                if px is None:
                    kp_2d.append(None)
                    vis.append(False)
                else:
                    kp_2d.append([round(px[0], 3), round(px[1], 3)])
                    vis.append(True)
        out3d.append({"actor_id": aid, "root": root, "keypoints_3d_m": kp_m})
        out2d.append({"actor_id": aid, "root": root, "keypoints_2d_px": kp_2d, "visible": vis})

    (ep / "coco17_3d.jsonl").write_text(
        "\n".join(json.dumps(o) for o in out3d) + "\n", encoding="utf-8")
    (ep / "coco17_2d.jsonl").write_text(
        "\n".join(json.dumps(o) for o in out2d) + "\n", encoding="utf-8")

    n_actor_frame = len(out3d)
    total_kp = n_actor_frame * 17
    roots = sorted({r["root"] for r in rows})
    _log(f"actor 集合: {actor_ids}（{len(actor_ids)}）")
    _log(f"root 范围: [{roots[0]}, {roots[-1]}]（{len(roots)} 帧）")
    _log(f"coco17_3d/2d 已写（{n_actor_frame} actor×帧，×17 = {total_kp} 关键点）")
    if missing:
        _log(f"缺失 bone 的点: {len(missing)} 处")
        return 1
    _log("所有肢体点 + head 齐全，无缺失")
    return 0


if __name__ == "__main__":
    sys.exit(main())
