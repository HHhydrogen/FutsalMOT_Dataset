"""P1：COCO17 2D keypoints overlay（双 actor，17 点 + COCO 骨架连线）。

用法：
    uv run python ue/overlay_coco17_c3.py

读：
  - coco17_2d.jsonl（每帧 17 点 2D 像素 + visible，含 actor_id）
  - img1/000001.png（RGB，1-based）
写：
  - overlay_coco17/000001.png
打印 0/15/30/45/60/75/89 关键帧。

颜色：L0（Player_L0）= 红系，R0（Player_R0）= 蓝系。连线同色。
"""

import json
from pathlib import Path

from PIL import Image, ImageDraw

from pose_bones import COCO_KEYPOINT_NAMES, COCO_SKELETON_EDGES

EP = Path(r"G:\FutsalMOT_Dataset\episode_bp_frame_sync")
CAM = EP / "CineCam_01"
KP2D = EP / "coco17_2d.jsonl"
IMG_DIR = CAM / "img1"
OUT_DIR = EP / "overlay_coco17"

CHECK_FRAMES = [0, 15, 30, 45, 60, 75, 89]

ACTOR_COLORS = {
    "Player_L0": (255, 0, 0),      # 红
    "Player_R0": (0, 128, 255),    # 蓝
}
FACE_COLOR = (255, 255, 0)  # 脸部黄色（与肢体区分）


def main():
    frames = [json.loads(l) for l in KP2D.read_text(encoding="utf-8").splitlines()]
    print(f"总帧: {len(frames)}")
    actors = sorted({f["actor_id"] for f in frames})
    print(f"actor 集合: {actors}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 按 (root, actor) 索引
    by_root = {}
    for fr in frames:
        by_root.setdefault(fr["root"], []).append(fr)

    for root, frs in sorted(by_root.items()):
        img_path = IMG_DIR / f"{root + 1:06d}.png"
        if not img_path.exists():
            continue
        img = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        for fr in frs:
            actor = fr["actor_id"]
            color = ACTOR_COLORS.get(actor, (255, 255, 255))
            pts = fr["keypoints_2d_px"]
            vis = fr["visible"]
            # 连线
            for a, b in COCO_SKELETON_EDGES:
                ia, ib = COCO_KEYPOINT_NAMES.index(a), COCO_KEYPOINT_NAMES.index(b)
                if vis[ia] and vis[ib] and pts[ia] and pts[ib]:
                    draw.line([pts[ia][0], pts[ia][1], pts[ib][0], pts[ib][1]],
                              fill=color, width=2)
            # 点（脸部用黄色）
            for i, name in enumerate(COCO_KEYPOINT_NAMES):
                if vis[i] and pts[i]:
                    u, v = pts[i]
                    c = FACE_COLOR if name in ("nose", "left_eye", "right_eye", "left_ear", "right_ear") else color
                    r = 1
                    draw.ellipse([u - r, v - r, u + r, v + r], fill=c)
        img.save(OUT_DIR / f"{root + 1:06d}.png")

    print("\n=== 关键帧 17 点（像素）===")
    for f in CHECK_FRAMES:
        frs = by_root.get(f, [])
        if not frs:
            print(f"root={f}: 无")
            continue
        print(f"root={f:>3}:")
        for fr in sorted(frs, key=lambda x: x["actor_id"]):
            print(f"  [{fr['actor_id']}]")
            for i, name in enumerate(COCO_KEYPOINT_NAMES):
                if fr["visible"][i] and fr["keypoints_2d_px"][i]:
                    u, v = fr["keypoints_2d_px"][i]
                    print(f"      {name:<15} ({u:7.1f},{v:7.1f})")
                else:
                    print(f"      {name:<15} 不可见")

    print(f"\noverlay 已写: {OUT_DIR}")
    print(f"颜色: {ACTOR_COLORS}")


if __name__ == "__main__":
    main()