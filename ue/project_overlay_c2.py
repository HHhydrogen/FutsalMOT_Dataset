"""P1：C2 5 骨骼投影 overlay（hand_l/lowerarm_l/thigh_l/calf_l/foot_l）。

用法：
    uv run python ue/project_overlay_c2.py

读：
  - pose_capture_c2.jsonl（450 行，UE 厘米）
  - camera.json（投影参数，世界米）
  - img1/000001.png（RGB，1-based）
写：
  - overlay_c2/000001.png（5 色点：hand红/lowerarm蓝/thigh绿/calf黄/foot紫）
打印 0/15/30/45/60/75/89 关键帧像素坐标。
"""

import json
from pathlib import Path

from PIL import Image, ImageDraw

EP = Path(r"G:\FutsalMOT_Dataset\episode_bp_frame_sync")
CAM = EP / "CineCam_01"
PC = EP / "pose_capture_c2.jsonl"
CAMERA = CAM / "camera.json"
IMG_DIR = CAM / "img1"
OUT_DIR = EP / "overlay_c2"

CHECK_FRAMES = [0, 15, 30, 45, 60, 75, 89]

BONE_COLORS = {
    "hand_l": (255, 0, 0),
    "lowerarm_l": (0, 0, 255),
    "thigh_l": (0, 255, 0),
    "calf_l": (255, 255, 0),
    "foot_l": (255, 0, 255),
}


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
    print(f"总记录: {len(rows)}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 按 root 分组
    by_frame = {}
    for r in rows:
        by_frame.setdefault(r["root"], []).append(r)

    for root, recs in sorted(by_frame.items()):
        img_path = IMG_DIR / f"{root + 1:06d}.png"
        if not img_path.exists():
            continue
        img = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        for rec in recs:
            bone = rec["bone"]
            hand_m = [rec["x"] / 100.0, rec["y"] / 100.0, rec["z"] / 100.0]
            px = project(hand_m, I, R, loc)
            if px is None:
                continue
            u, v = px
            color = BONE_COLORS.get(bone, (255, 255, 255))
            r = 4
            draw.ellipse([u - r, v - r, u + r, v + r], fill=color)
        img.save(OUT_DIR / f"{root + 1:06d}.png")

    print("\n=== 关键帧投影（5 色点像素坐标）===")
    for f in CHECK_FRAMES:
        recs = by_frame.get(f, [])
        print(f"root={f:>3}:")
        for rec in sorted(recs, key=lambda r: r["bone"]):
            hand_m = [rec["x"] / 100.0, rec["y"] / 100.0, rec["z"] / 100.0]
            px = project(hand_m, I, R, loc)
            if px:
                print(f"    {rec['bone']:<10} ({px[0]:6.1f},{px[1]:6.1f})  {[round(v,3) for v in hand_m]}")

    print(f"\noverlay 已写: {OUT_DIR}")
    print("颜色: hand红 lowerarm蓝 thigh绿 calf黄 foot紫")


if __name__ == "__main__":
    main()