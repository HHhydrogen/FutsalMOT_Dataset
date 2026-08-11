"""pose_validator 的测试：文件级/数值范围/实例级校验。"""

import json
from pathlib import Path

import numpy as np
from PIL import Image

from grf_ue_bridge.pose_annotator import annotate_pose_dir
from grf_ue_bridge.pose_validator import (
    _left_right_axis_consistency,
    validate_pose_dir,
)

W, H = 64, 64

CAMERA_JSON = {
    "camera_id": "Cam_01", "image_width": W, "image_height": H,
    "intrinsics": {"width": W, "height": H, "fx": 50.0, "fy": 50.0, "cx": W / 2, "cy": H / 2},
    "extrinsics": {"world_location_m": [0.0, 0.0, 0.0], "forward": [1.0, 0.0, 0.0],
                   "right": [0.0, 1.0, 0.0], "up": [0.0, 0.0, 1.0]},
}

# 站立姿势的简单 17 点世界坐标（米），肩/髋左右对称
STANDING = [
    [10.0, 0.00, 1.62], [10.0, -0.05, 1.66], [10.0, 0.05, 1.66],
    [10.0, -0.10, 1.60], [10.0, 0.10, 1.60],
    [10.0, -0.25, 1.50], [10.0, 0.25, 1.50],
    [10.0, -0.35, 1.35], [10.0, 0.35, 1.35],
    [10.0, -0.42, 1.20], [10.0, 0.42, 1.20],
    [10.0, -0.15, 0.90], [10.0, 0.15, 0.90],
    [10.0, -0.15, 0.50], [10.0, 0.15, 0.50],
    [10.0, -0.15, 0.15], [10.0, 0.15, 0.15],
]


def _make_camera(root: Path) -> Path:
    cam = root / "Cam_01"
    (cam / "gt").mkdir(parents=True, exist_ok=True)
    (cam / "camera.json").write_text(json.dumps(CAMERA_JSON), encoding="utf-8")
    (cam / "seqinfo.ini").write_text("[Sequence]\nname=Cam_01\n", encoding="utf-8")
    obj = {
        "entity_id": "L0", "track_id": 1, "class": "player", "team": "left",
        "role": None, "is_goalkeeper": False, "world_position": [0.0, 0.0, 0.9],
        "mask_id": 1, "bbox_source": "instance_mask",
        "in_frame": True, "truncated": False, "visibility": None,
        "visible_pixel_count": 100,
        "raw_bbox_xyxy": [28.0, 20.0, 38.0, 34.0], "raw_bbox_xywh": [28.0, 20.0, 10.0, 14.0],
        "bbox_xyxy": [28.0, 20.0, 38.0, 34.0], "bbox_xywh": [28.0, 20.0, 10.0, 14.0],
        "segmentation": None,
    }
    frames = [{"episode_id": "ep", "camera_id": "Cam_01", "frame_index": 1,
               "source_step": 0, "time_seconds": 0.0, "objects": [obj]}]
    with open(cam / "annotations.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps(frames[0]) + "\n")
    img1 = cam / "img1"
    mask = cam / "mask"
    img1.mkdir(parents=True, exist_ok=True)
    mask.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.zeros((H, W, 3), dtype=np.uint8)).save(str(img1 / "000001.png"))
    m = np.zeros((H, W), dtype=np.uint8)
    m[20:34, 28:38] = 1
    Image.fromarray(m.astype(np.uint8)).save(str(mask / "000001.png"))
    meta = {
        "kind": "meta", "schema": "futsalmot_pose_keypoints_v1", "episode_id": "ep",
        "camera_id": "Cam_01", "image_width": W, "image_height": H,
        "keypoint_names": ["nose", "left_eye", "right_eye", "left_ear", "right_ear",
                           "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                           "left_wrist", "right_wrist", "left_hip", "right_hip",
                           "left_knee", "right_knee", "left_ankle", "right_ankle"],
        "coordinate_convention": "meters", "occlusion_method": "none",
    }
    frame_line = {
        "kind": "frame", "frame_index": 1, "source_step": 0, "time_seconds": 0.0,
        "objects": [{"entity_id": "L0", "track_id": 1, "keypoints_world": STANDING,
                     "occluded": [False] * 17}],
    }
    with open(cam / "pose_keypoints.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps(meta) + "\n" + json.dumps(frame_line) + "\n")
    return cam


class TestStructuralValidation:
    def test_valid_labels_pass(self, tmp_path):
        _make_camera(tmp_path)
        assert annotate_pose_dir(tmp_path, write_yaml=False) == 0
        assert validate_pose_dir(tmp_path, validation_level="full") == 0

    def test_wrong_field_count_fails(self, tmp_path):
        cam = _make_camera(tmp_path)
        assert annotate_pose_dir(tmp_path, write_yaml=False) == 0
        # 篡改：去掉一个字段 → 55 字段
        label = cam / "labels_pose" / "000001.txt"
        parts = label.read_text(encoding="utf-8").split()
        parts = parts[:-1]
        label.write_text(" ".join(parts) + "\n", encoding="utf-8")
        assert validate_pose_dir(tmp_path, validation_level="quick") == 1

    def test_out_of_range_fails(self, tmp_path):
        cam = _make_camera(tmp_path)
        assert annotate_pose_dir(tmp_path, write_yaml=False) == 0
        label = cam / "labels_pose" / "000001.txt"
        parts = label.read_text(encoding="utf-8").split()
        parts[5] = "2.5"  # nose x 越界
        label.write_text(" ".join(parts) + "\n", encoding="utf-8")
        assert validate_pose_dir(tmp_path, validation_level="quick") == 1

    def test_invalid_v_fails(self, tmp_path):
        cam = _make_camera(tmp_path)
        assert annotate_pose_dir(tmp_path, write_yaml=False) == 0
        label = cam / "labels_pose" / "000001.txt"
        parts = label.read_text(encoding="utf-8").split()
        parts[7] = "5"  # nose v 非法
        label.write_text(" ".join(parts) + "\n", encoding="utf-8")
        assert validate_pose_dir(tmp_path, validation_level="quick") == 1

    def test_missing_label_file_fails(self, tmp_path):
        cam = _make_camera(tmp_path)
        assert annotate_pose_dir(tmp_path, write_yaml=False) == 0
        (cam / "labels_pose" / "000001.txt").unlink()
        assert validate_pose_dir(tmp_path, validation_level="quick") == 1

    def test_missing_pose_keypoints_fails(self, tmp_path):
        cam = _make_camera(tmp_path)
        assert annotate_pose_dir(tmp_path, write_yaml=False) == 0
        (cam / "pose_keypoints.jsonl").unlink()
        assert validate_pose_dir(tmp_path, validation_level="quick") == 1

    def test_missing_labels_pose_dir_fails(self, tmp_path):
        _make_camera(tmp_path)
        assert validate_pose_dir(tmp_path, validation_level="quick") == 1


class TestLeftRightAxis:
    def _frame_with_shoulders_hips(self, shoulder_sign=1.0, hip_sign=1.0):
        kps = [list(p) for p in STANDING]
        # 调整肩/髋左右分离方向：正 = left 在 -Y、right 在 +Y（与 STANDING 一致）
        ls, rs = 5, 6
        lh, rh = 11, 12
        kps[ls] = [10.0, -0.25 * shoulder_sign, 1.50]
        kps[rs] = [10.0, 0.25 * shoulder_sign, 1.50]
        kps[lh] = [10.0, -0.15 * hip_sign, 0.90]
        kps[rh] = [10.0, 0.15 * hip_sign, 0.90]
        return {"kind": "frame", "frame_index": 1, "source_step": 0, "time_seconds": 0.0,
                "objects": [{"entity_id": "L0", "track_id": 1, "keypoints_world": kps}]}

    def test_consistent_axes_no_error(self):
        assert _left_right_axis_consistency(self._frame_with_shoulders_hips()) == []

    def test_reversed_shoulder_axis_detected(self):
        # 肩轴反转（左/右肩交换）→ 双轴反向 → 报错
        errors = _left_right_axis_consistency(self._frame_with_shoulders_hips(shoulder_sign=-1.0))
        assert len(errors) == 1
        assert "左右" in errors[0]

    def test_reversed_hip_axis_detected(self):
        errors = _left_right_axis_consistency(self._frame_with_shoulders_hips(hip_sign=-1.0))
        assert len(errors) == 1
