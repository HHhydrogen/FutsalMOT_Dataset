"""P1：COCO17 2D keypoints overlay（10 actor，17 点 + COCO 骨架连线）。

用法：
    uv run python ue/overlay_coco17_c4.py

读：
  - coco17_2d_c4.jsonl（每帧 17 点 2D 像素 + visible，含 actor_id）
  - img1/000001.png（RGB，1-based）
写：
  - overlay_coco17_c4/000001.png
打印 0/15/30/45/60/75/89 关键帧。

颜色：L0-L4 红/橙/黄绿系色相渐变，R0-R4 蓝/青/紫系色相渐变；脸部点黄色。
连线同 actor 色。
"""

import json
import colorsys
from pathlib import Path

from PIL import Image, ImageDraw

from pose_bones import COCO_KEYPOINT_NAMES, COCO_SKELETON_EDGES

EP = Path(r"G:\FutsalMOT_Dataset\episode_bp_frame_sync")
CAM = EP / "CineCam_01"
KP2D = EP / "coco17_2d_c4.jsonl"
IMG_DIR = CAM / "img1"
OUT_DIR = EP / "overlay_coco17_c4"

CHECK_FRAMES = [0, 15, 30, 45, 60, 75, 89]


def _hsv(h, s=0.85, v=0.95):
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return (int(r * 255), int(g * 255), int(b * 255))


def make_colors():
    colors = {}
    # L 队：红系色相 0.0..0.12
    l_hues = [0.0, 0.03, 0.06, 0.09, 0.12]
    for i, h in enumerate(l_hues):
        colors[f"L{i}"] = _hsv(h)
    # R 队：蓝系色相 0.55..0.67
    r_hues = [0.55, 0.58, 0.61, 0.64, 0.67]
    for i, h in enumerate(r_hues):
        colors[f"R{i}"] = _hsv(h)
    return colors


FACE_COLOR = (255, 255, 0)  # 脸部黄色


def main():
    ACTOR_COLORS = make_colors()
    frames = [json.loads(l) for l in KP2D.read_text(encoding="utf-8").splitlines()]
    print(f"总帧: {len(frames)}")
    actors = sorted({f["actor_id"] for f in frames})
    print(f"actor 集合: {actors}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

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
            for a, b in COCO_SKELETON_EDGES:
                ia, ib = COCO_KEYPOINT_NAMES.index(a), COCO_KEYPOINT_NAMES.index(b)
                if vis[ia] and vis[ib] and pts[ia] and pts[ib]:
                    draw.line([pts[ia][0], pts[ia][1], pts[ib][0], pts[ib][1]],
                              fill=color, width=2)
            for i, name in enumerate(COCO_KEYPOINT_NAMES):
                if vis[i] and pts[i]:
                    u, v = pts[i]
                    c = FACE_COLOR if name in ("nose", "left_eye", "right_eye", "left_ear", "right_ear") else color
                    r = 1
                    draw.ellipse([u - r, v - r, u + r, v + r], fill=c)
        img.save(OUT_DIR / f"{root + 1:06d}.png")

    print("\n=== 关键帧 actor 覆盖 ===")
    for f in CHECK_FRAMES:
        frs = by_root.get(f, [])
        ids = sorted(fr["actor_id"] for fr in frs)
        print(f"root={f:>3}: {ids}")

    print(f"\noverlay 已写: {OUT_DIR}")
    print(f"颜色: {ACTOR_COLORS}")


if __name__ == "__main__":
    main()