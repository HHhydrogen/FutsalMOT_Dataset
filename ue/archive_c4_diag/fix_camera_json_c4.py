"""P1：修正 camera.json 内参为实际 RGB 分辨率 1920x1080。

extrinsics 保持（Phase 3 已确认与运行时 MRQ 相机一致且静态）。
fx = focal*sensor 推导 = 15*1920/23.76 = 1212.12；fy = 15*1080/13.365 = 1212.12。
"""

import json
import math
from pathlib import Path

EP = Path(r"G:\FutsalMOT_Dataset\episode_bp_frame_sync")
CAMERA = EP / "CineCam_01" / "camera.json"

W, H = 1920, 1080
FOCAL = 15.0
SW, SH = 23.76, 13.365
fx = FOCAL * W / SW
fy = FOCAL * H / SH
cx, cy = (W - 1) / 2, (H - 1) / 2
fov_h = 2 * math.degrees(math.atan((W / 2) / fx))
fov_v = 2 * math.degrees(math.atan((H / 2) / fy))


def main():
    cam = json.loads(CAMERA.read_text(encoding="utf-8"))
    old = dict(cam["intrinsics"])
    cam["image_width"] = W
    cam["image_height"] = H
    cam["intrinsics"] = {
        "width": W, "height": H,
        "fx": round(fx, 6), "fy": round(fy, 6),
        "cx": cx, "cy": cy,
    }
    cam["horizontal_fov_deg"] = round(fov_h, 6)
    cam["vertical_fov_deg"] = round(fov_v, 6)
    cam["resolution_note"] = "intrinsics calibrated for 1920x1080 (C4 RGB); camera static (Phase 3 verified)"
    CAMERA.write_text(json.dumps(cam, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"intrinsics: {old} -> {cam['intrinsics']}")
    print(f"fov_h={fov_h:.3f} fov_v={fov_v:.3f}")
    print(f"extrinsics 未变: loc={cam['extrinsics']['world_location_m']} rot={cam['extrinsics']['world_rotation_deg']}")
    print("已写 camera.json")


if __name__ == "__main__":
    main()