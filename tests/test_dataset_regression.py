"""dataset_regression 端到端回归校验的单元测试（独立于 annotation_validator）。

直接用 `validate_dataset_regression` 验证跨产物一致性检查项：
帧数全链路、分辨率、mask 非全背景、MOT/YOLO 重新派生比对、instance-mask bbox==mask。
"""

import json
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from grf_ue_bridge.dataset_regression import validate_dataset_regression
from grf_ue_bridge.mask_annotator import annotate_masks_dir

W, H = 64, 64


def _write_png(path: Path, arr, rgb=False):
    if rgb:
        Image.fromarray(np.zeros((H, W, 3), dtype=np.uint8)).save(str(path))
    else:
        Image.fromarray(arr.astype(np.uint8)).save(str(path))


def _geo_obj(entity, track, cls, xyxy):
    x0, y0, x1, y1 = [float(v) for v in xyxy]
    return {
        "entity_id": entity, "track_id": track, "class": cls,
        "team": "left" if entity.startswith("L") else ("right" if entity.startswith("R") else None),
        "role": None, "is_goalkeeper": False, "world_position": [0.0, 0.0, 0.9],
        "in_frame": True, "truncated": False, "visibility": None,
        "raw_bbox_xyxy": [x0, y0, x1, y1], "raw_bbox_xywh": [x0, y0, x1 - x0, y1 - y0],
        "bbox_xyxy": [x0, y0, x1, y1], "bbox_xywh": [x0, y0, x1 - x0, y1 - y0],
    }


def _make_camera(root: Path, frames: int = 2) -> Path:
    """合成 camera：每帧 L0 可见矩形、R0 不可见（几何在画面内但 mask 无像素）。"""
    cam = root / "Cam_01"
    (cam / "gt").mkdir(parents=True)
    (cam / "camera.json").write_text(json.dumps({
        "camera_id": "Cam_01", "image_width": W, "image_height": H,
        "intrinsics": {"width": W, "height": H, "fx": 50.0, "fy": 50.0,
                       "cx": W / 2, "cy": H / 2},
        "extrinsics": {"world_location_m": [0.0, 0.0, 0.0], "forward": [1.0, 0.0, 0.0],
                       "right": [0.0, 1.0, 0.0], "up": [0.0, 0.0, 1.0]},
    }), encoding="utf-8")
    (cam / "seqinfo.ini").write_text("[Sequence]\nname=Cam_01\n", encoding="utf-8")
    l0 = _geo_obj("L0", 1, "player", [4, 4, 24, 24])
    r0 = _geo_obj("R0", 6, "player", [40, 40, 44, 44])
    lines = []
    for fi in range(1, frames + 1):
        lines.append(json.dumps({
            "episode_id": "reg", "camera_id": "Cam_01", "frame_index": fi,
            "source_step": fi - 1, "time_seconds": (fi - 1) * 0.1, "objects": [l0, r0],
        }))
    (cam / "annotations.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (cam / "img1").mkdir()
    (cam / "mask").mkdir()
    for fi in range(1, frames + 1):
        m = np.zeros((H, W), dtype=np.uint8)
        m[4:24, 4:24] = 1
        _write_png(cam / "img1" / f"{fi:06d}.png", m, rgb=True)
        _write_png(cam / "mask" / f"{fi:06d}.png", m)
    return cam


class TestDatasetRegression:
    def test_valid_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_camera(root)
            assert annotate_masks_dir(root) == 0
            assert validate_dataset_regression(root) == 0

    def test_all_background_mask_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cam = _make_camera(root)
            annotate_masks_dir(root)
            Image.fromarray(np.zeros((H, W), dtype=np.uint8)).save(str(cam / "mask" / "000001.png"))
            assert validate_dataset_regression(root) == 1

    def test_img_resolution_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cam = _make_camera(root)
            annotate_masks_dir(root)
            Image.fromarray(np.zeros((32, 64, 3), dtype=np.uint8)).save(str(cam / "img1" / "000001.png"))
            assert validate_dataset_regression(root) == 1

    def test_frame_count_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cam = _make_camera(root)
            annotate_masks_dir(root)
            (cam / "mask" / "000002.png").unlink()  # mask 缺帧 2
            assert validate_dataset_regression(root) == 1

    def test_mot_bbox_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cam = _make_camera(root)
            annotate_masks_dir(root)
            (cam / "gt" / "gt.txt").write_text(
                "1,1,5,5,20,20,1,1,1.00\n", encoding="utf-8"  # bbox 与 mask 不一致
            )
            assert validate_dataset_regression(root) == 1

    def test_yolo_det_bbox_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cam = _make_camera(root)
            annotate_masks_dir(root)
            (cam / "labels" / "det" / "000001.txt").write_text(
                "0 0.9 0.9 0.2 0.2\n", encoding="utf-8"  # 与 mask bbox 不符
            )
            assert validate_dataset_regression(root) == 1

    def test_annotations_bbox_not_equal_mask_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cam = _make_camera(root)
            annotate_masks_dir(root)
            frames = [json.loads(ln) for ln in
                      (cam / "annotations.jsonl").read_text(encoding="utf-8").splitlines() if ln.strip()]
            frames[0]["objects"][0]["bbox_xyxy"] = [4.0, 4.0, 25.0, 24.0]  # 与 mask min/max 不符
            with open(cam / "annotations.jsonl", "w", encoding="utf-8") as f:
                for fr in frames:
                    f.write(json.dumps(fr) + "\n")
            assert validate_dataset_regression(root) == 1
