"""P1：C3 JSONL → COCO17（3D world + 2D projected）。

复用 ue/pose_bones.py 的现有 COCO17 映射逻辑，不重设计。

用法：
    uv run python ue/build_coco17_c3.py

读：
  - pose_capture_c3.jsonl（1170 行：root/shot/game_time/actor_id/bone/x/y/z/qx/qy/qz/qw，cm）
  - camera.json（投影参数，世界米）
写：
  - coco17_3d.jsonl  每帧一行 {root, keypoints_3d_m: [[x,y,z]×17]}
  - coco17_2d.jsonl  每帧一行 {root, keypoints_2d_px: [[x,y]×17], visible: [bool×17]}

复用：
  - pose_bones.resolve_limb_bone_map / apply_head_offsets / normalize_world_keypoints
  - pose_bones.CM 转 M（肢体点）
"""

import json
import sys
from pathlib import Path

from pose_bones import (
    COCO_KEYPOINT_NAMES,
    resolve_limb_bone_map,
    apply_head_offsets,
    normalize_world_keypoints,
)

EP = Path(r"G:\FutsalMOT_Dataset\episode_bp_frame_sync")
CAM = EP / "CineCam_01"
PC = EP / "pose_capture_c3.jsonl"
CAMERA = CAM / "camera.json"
OUT_3D = EP / "coco17_3d.jsonl"
OUT_2D = EP / "coco17_2d.jsonl"

CM_TO_M = 0.01


def load_camera():
    cam = json.loads(CAMERA.read_text(encoding="utf-8"))
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


def main():
    rows = [json.loads(l) for l in PC.read_text(encoding="utf-8").splitlines()]
    print(f"总记录: {len(rows)}")

    # 按 (actor_id, root) 分组
    actor_ids = sorted({r["actor_id"] for r in rows})
    print(f"actor 集合: {actor_ids}")
    by_actor_frame = {}
    for r in rows:
        by_actor_frame.setdefault((r["actor_id"], r["root"]), []).append(r)

    I, R, loc = load_camera()

    # 全局骨骼名集合 → 映射（所有 actor 相同骨架）
    all_bones = sorted({r["bone"] for r in rows})
    limb_map = resolve_limb_bone_map(all_bones)
    print(f"骨骼名集合 ({len(all_bones)}): {all_bones}")
    print(f"肢体点映射 ({len(limb_map)}/12):")
    for coco in COCO_KEYPOINT_NAMES:
        if coco in limb_map:
            print(f"  {coco:<15} -> {limb_map[coco]}")

    out3d, out2d = [], []
    missing = {}
    frames_per_actor = {}
    for (aid, root), recs in sorted(by_actor_frame.items()):
        frames_per_actor.setdefault(aid, []).append(root)
        recmap = {r["bone"]: r for r in recs}

        # 12 肢体点（world cm）
        limb = {}
        for coco, bone in limb_map.items():
            r = recmap.get(bone)
            if r is None:
                missing.setdefault((aid, coco), []).append(root)
                continue
            limb[coco] = [r["x"], r["y"], r["z"]]

        # 脸部 5 点（head + 局部偏移）
        face = {}
        head = recmap.get("head")
        if head is not None:
            face = apply_head_offsets(
                [head["x"], head["y"], head["z"]],
                [head["qx"], head["qy"], head["qz"], head["qw"]],
            )

        # 合成 17×3 world cm
        kp_cm = normalize_world_keypoints(limb, face)
        # → 米
        kp_m = []
        for p in kp_cm:
            if p[0] is None:
                kp_m.append(None)
            else:
                kp_m.append([p[0] * CM_TO_M, p[1] * CM_TO_M, p[2] * CM_TO_M])

        # 2D 投影
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

    OUT_3D.write_text(
        "\n".join(json.dumps(o) for o in out3d) + "\n", encoding="utf-8")
    OUT_2D.write_text(
        "\n".join(json.dumps(o) for o in out2d) + "\n", encoding="utf-8")
    print(f"\n已写: {OUT_3D.name} ({len(out3d)} 帧), {OUT_2D.name} ({len(out2d)} 帧)")
    for aid in actor_ids:
        roots = sorted(frames_per_actor.get(aid, []))
        print(f"  actor={aid}: 帧 {roots[0] if roots else '-'}..{roots[-1] if roots else '-'} ({len(roots)} 帧)")

    if missing:
        print(f"\n缺失 bone 的点:")
        for key, roots in missing.items():
            print(f"  {key}: 缺 {len(roots)} 帧")
    else:
        print("\n所有肢体点 + head 齐全，无缺失")

    # 总关键点数 = 每 actor 每帧 17
    total_kp = len(out3d) * 17
    print(f"\n最终关键点总数 = {len(out3d)} (actor×帧) × 17 = {total_kp}")


if __name__ == "__main__":
    main()