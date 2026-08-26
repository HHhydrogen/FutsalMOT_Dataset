"""P1：把 hand_l 世界坐标投影到 RGB 画红点（验证 C1 同帧对齐）。

用法：
    uv run python ue/project_hand_overlay.py

读：
  - pose_capture.jsonl（hand 为 UE 厘米）
  - camera.json（投影参数，世界米）
  - img1/0000NN.png（RGB）
写：
  - overlay_hand/0000NN.png（红点叠加）
  - 打印 0/15/30/45/60/75/89 的投影像素坐标
"""

import json
import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw

EP = Path(r"G:\FutsalMOT_Dataset\episode_bp_frame_sync")
CAM = EP / "CineCam_01"
PC = EP / "pose_capture.jsonl"
CAMERA = CAM / "camera.json"
IMG_DIR = CAM / "img1"
OUT_DIR = EP / "overlay_hand"

CHECK_FRAMES = [0, 15, 30, 45, 60, 75, 89]


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
    I, R, loc = load_camera()
    rows = [json.loads(l) for l in PC.read_text(encoding="utf-8").splitlines()]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== 关键帧投影（红点像素坐标）===")
    for row in rows:
        root = row["root"]
        hand_cm = row["hand"]
        hand_m = [hand_cm[0] / 100.0, hand_cm[1] / 100.0, hand_cm[2] / 100.0]
        px = project(hand_m, I, R, loc)
        # 读取 RGB 画红点（img1 是 1-based：root=0 ↔ 000001.png）
        img_path = IMG_DIR / f"{root + 1:06d}.png"
        if img_path.exists() and px is not None:
            img = Image.open(img_path).convert("RGB")
            draw = ImageDraw.Draw(img)
            r = 4
            u, v = px
            draw.ellipse([u - r, v - r, u + r, v + r], fill=(255, 0, 0))
            img.save(OUT_DIR / f"{root:06d}.png")
            if root in CHECK_FRAMES:
                print(f"  root={root:>3} hand_m={[round(x,3) for x in hand_m]} -> px=({u:.1f}, {v:.1f})")

    print(f"\noverlay 已写: {OUT_DIR}（{len(rows)} 帧）")
    print(f"检查帧: {CHECK_FRAMES}")


if __name__ == "__main__":
    main()