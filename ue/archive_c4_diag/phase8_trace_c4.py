"""P1：Phase 8 — frame 30, Player_L2 单帧完整投影 trace。

world cm → camera space (m) → normalized → pixel。
对比：错误内参(1280x720) vs 正确内参(1920x1080)，并与 mask bbox 对照。
定位错误在 World→Camera 还是 Camera→Pixel。
"""

import json
import math
from pathlib import Path

EP = Path(r"G:\FutsalMOT_Dataset\episode_bp_frame_sync")
CAM = EP / "CineCam_01"
CAMERA = CAM / "camera.json"
POSE = EP / "pose_capture_c4.jsonl"
MB = CAM / "mask_bbox_c4.jsonl"

FRAME = 30
ACTOR = "L2"
BONES = ["head", "hand_l", "hand_r", "thigh_l", "calf_l", "foot_l"]
CM_TO_M = 0.01


def main():
    cam = json.loads(CAMERA.read_text(encoding="utf-8"))
    I = cam["intrinsics"]
    E = cam["extrinsics"]
    R = [E["forward"], E["right"], E["up"]]
    loc = E["world_location_m"]
    print(f"offline camera.json: fx={I['fx']:.2f} fy={I['fy']:.2f} cx={I['cx']} cy={I['cy']} "
          f"(W={I['width']} H={I['height']})")

    # 正确内参：1920x1080，焦距15mm，sensor 23.76x13.365
    W, H = 1920, 1080
    fx_c = 15.0 * W / 23.76
    fy_c = 15.0 * H / 13.365
    cx_c, cy_c = (W - 1) / 2, (H - 1) / 2
    print(f"correct 1920x1080: fx={fx_c:.2f} fy={fy_c:.2f} cx={cx_c} cy={cy_c}")

    rows = [json.loads(l) for l in POSE.read_text(encoding="utf-8").splitlines()]
    pose = {r["bone"]: r for r in rows if r["root"] == FRAME and r["actor_id"] == ACTOR}
    mb = next((json.loads(l) for l in MB.read_text(encoding="utf-8").splitlines()
               if json.loads(l)["root"] == FRAME and json.loads(l)["actor_id"] == ACTOR), None)
    print(f"\nmask bbox (frame {FRAME}, {ACTOR}): {mb}")

    def cam_space(world_m):
        dx = world_m[0] - loc[0]
        dy = world_m[1] - loc[1]
        dz = world_m[2] - loc[2]
        return (sum(R[0][i] * (world_m[i] - loc[i]) for i in range(3)),
                sum(R[1][i] * (world_m[i] - loc[i]) for i in range(3)),
                sum(R[2][i] * (world_m[i] - loc[i]) for i in range(3)))

    def project(cp, fx, fy, cx, cy):
        x, y, z = cp
        return (cx + fx * y / x, cy - fy * z / x)

    print(f"\n=== frame {FRAME}, {ACTOR} 完整 trace ===")
    print(f"{'bone':<10} {'world_m':<26} {'cam_space':<24} {'pix_wrong':<16} {'pix_correct':<16}")
    for b in BONES:
        r = pose.get(b)
        if r is None:
            continue
        w = [r["x"] * CM_TO_M, r["y"] * CM_TO_M, r["z"] * CM_TO_M]
        cp = cam_space(w)
        pw = project(cp, I["fx"], I["fy"], I["cx"], I["cy"])
        pc = project(cp, fx_c, fy_c, cx_c, cy_c)
        print(f"{b:<10} {[round(v,2) for v in w]} {[round(v,2) for v in cp]} "
              f"{[round(v,1) for v in pw]} {[round(v,1) for v in pc]}")

    # mask 中心/范围对照
    if mb:
        mcx = (mb["xmin"] + mb["xmax"]) / 2
        mcy = (mb["ymin"] + mb["ymax"]) / 2
        print(f"\nmask 中心 = ({mcx:.1f}, {mcy:.1f})，bbox = ({mb['xmin']},{mb['ymin']})-({mb['xmax']},{mb['ymax']})")
        print(f"foot_l 正确投影 vs mask 中心距离: 见上表")


if __name__ == "__main__":
    main()