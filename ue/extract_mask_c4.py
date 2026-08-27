"""P1：从 C4 掩码 EXR 提取每 actor 像素 bbox + MRQ 真实相机元数据，存 JSONL。

读 mask_c4_render/*.exr（Cryptomatte Object ID）。
写：
  - mask_bbox_c4.jsonl  {root, actor_id, xmin, ymin, xmax, ymax}
  - mask_camera_c4.jsonl {root, curPos_cm, curRot, focal_length, fov, sensor, overscan_percent}
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from grf_ue_bridge.cryptomatte import load_cryptomatte, hex_id_to_float  # noqa: E402

EP = Path(r"G:\FutsalMOT_Dataset\episode_bp_frame_sync")
MASK_DIR = EP / "CineCam_01" / "mask_c4_render"
OUT_BBOX = EP / "CineCam_01" / "mask_bbox_c4.jsonl"
OUT_CAM = EP / "CineCam_01" / "mask_camera_c4.jsonl"

ACTORS = [f"L{i}" for i in range(5)] + [f"R{i}" for i in range(5)]
LABEL = {a: f"Player_{a}" for a in ACTORS}


def bbox_of_mask(id_channel, target_id):
    """id_channel (H,W) float32；返回 (xmin,ymin,xmax,ymax) 或 None。"""
    import numpy as np
    ys, xs = np.nonzero(id_channel == target_id)
    if len(xs) == 0:
        return None
    return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))


def main():
    import OpenEXR

    frames = sorted(MASK_DIR.glob("*.exr"))
    print(f"EXR 文件数: {len(frames)}")
    bbox_rows = []
    cam_rows = []
    for p in frames:
        root = int(p.stem)
        manifest, ids = load_cryptomatte(p)
        # float ID per entity
        import numpy as np
        fids = {}
        for a in ACTORS:
            lbl = LABEL[a]
            if lbl in manifest:
                fids[a] = hex_id_to_float(manifest[lbl])
        for a, fid in fids.items():
            bb = bbox_of_mask(ids, fid)
            if bb is not None:
                bbox_rows.append({"root": root, "actor_id": a,
                                  "xmin": bb[0], "ymin": bb[1],
                                  "xmax": bb[2], "ymax": bb[3]})
        # 相机元数据
        h = OpenEXR.InputFile(str(p)).header()
        def g(base, key):
            k = f"unreal/camera/{base}/{key}"
            try:
                v = h[k]
            except KeyError:
                return None
            if isinstance(v, bytes):
                s = v.decode("utf-8", "replace").strip()
                try:
                    return float(s)
                except ValueError:
                    return s
            return v
        cam_rows.append({
            "root": root,
            "curPos_cm": [g("curPos", "x"), g("curPos", "y"), g("curPos", "z")],
            "curRot_deg": [g("curRot", "pitch"), g("curRot", "yaw"), g("curRot", "roll")],
            "focal_length": g("ActorHitProxyMask", "focalLength"),
            "fov": g("ActorHitProxyMask", "fov"),
            "sensor_width": g("ActorHitProxyMask", "sensorWidth"),
            "sensor_height": g("ActorHitProxyMask", "sensorHeight"),
            "sensor_aspect": g("ActorHitProxyMask", "sensorAspectRatio"),
            "overscan_percent": g("ActorHitProxyMask", "overscanPercent"),
            "camera_name": g("", "cameraName") or g("cameraName", "") or None,
        })

    OUT_BBOX.write_text("\n".join(json.dumps(r) for r in bbox_rows) + "\n", encoding="utf-8")
    OUT_CAM.write_text("\n".join(json.dumps(r) for r in cam_rows) + "\n", encoding="utf-8")
    print(f"bbox 行数: {len(bbox_rows)} -> {OUT_BBOX}")
    print(f"camera 行数: {len(cam_rows)} -> {OUT_CAM}")
    for r in cam_rows[:3]:
        print("cam:", r)


if __name__ == "__main__":
    main()