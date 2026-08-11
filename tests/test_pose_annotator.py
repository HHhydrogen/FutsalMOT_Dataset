"""pose_annotator 的端到端与单元测试（合成 camera 目录，纯 numpy/PIL）。

覆盖：world→camera→image 投影、normalization、visibility 判定（mask 邻域 + occluded）、
YOLO Pose 序列化（56 字段）、labels_pose 输出、dataset YAML、与 det bbox 对齐、
以及既有 Detection/Segmentation/MOT pipeline 的回归。
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from grf_ue_bridge.mask_annotator import annotate_masks_dir
from grf_ue_bridge.pose_annotator import (
    _keypoint_visibility,
    _project_keypoints,
    annotate_pose_dir,
    compute_pose_instances,
    pose_overlay_dir,
    read_pose_keypoints,
    serialize_pose_line,
    write_pose_dataset,
)
from grf_ue_bridge.pose_validator import validate_pose_dir
from camera_projection import (
    CameraExtrinsics,
    CameraIntrinsics,
    compute_intrinsics_from_focal_length,
)

W, H = 64, 64

# 相机：位于原点朝 +X 看，fx=fy=50，cx=cy=32 → u=32+50*y/x，v=32-50*z/x
CAMERA_JSON = {
    "camera_id": "Cam_01",
    "image_width": W,
    "image_height": H,
    "intrinsics": {"width": W, "height": H, "fx": 50.0, "fy": 50.0, "cx": W / 2, "cy": H / 2},
    "extrinsics": {"world_location_m": [0.0, 0.0, 0.0], "forward": [1.0, 0.0, 0.0],
                   "right": [0.0, 1.0, 0.0], "up": [0.0, 0.0, 1.0]},
}

# L0 的 17 个世界关键点（米），球员在 x=4m（图像中高约 19px，比 x=10 时大 2.5 倍）。
# left_ankle 放在相机后方（x<0）→ 应判 v=0。
L0_WORLD = [
    [4.0, 0.00, 1.62], [4.0, -0.05, 1.66], [4.0, 0.05, 1.66],
    [4.0, -0.10, 1.60], [4.0, 0.10, 1.60],
    [4.0, -0.25, 1.50], [4.0, 0.25, 1.50],
    [4.0, -0.35, 1.35], [4.0, 0.35, 1.35],
    [4.0, -0.42, 1.20], [4.0, 0.42, 1.20],
    [4.0, -0.15, 0.90], [4.0, 0.15, 0.90],
    [4.0, -0.15, 0.50], [4.0, 0.15, 0.50],
    [-1.0, -0.15, 0.15], [4.0, 0.15, 0.15],
]

# 期望投影（像素）：u=32+50*y/4=32+12.5y，v=32-50*z/4=32-12.5z
EXPECTED_PROJ = [
    (32.0, 11.75), (31.375, 11.25), (32.625, 11.25), (30.75, 12.0), (33.25, 12.0),
    (28.875, 13.25), (35.125, 13.25), (27.625, 15.125), (36.375, 15.125),
    (26.75, 17.0), (37.25, 17.0), (30.125, 20.75), (33.875, 20.75),
    (30.125, 25.75), (33.875, 25.75), None, (33.875, 30.125),
]

# 期望可见性：right_wrist(#10) 被 L1 mask 覆盖 → 1；left_ankle(#15) 在相机后方 → 0
EXPECTED_VIS = [2] * 17
EXPECTED_VIS[10] = 1
EXPECTED_VIS[15] = 0


def _intrinsics():
    return CameraIntrinsics(width=W, height=H, fx=50.0, fy=50.0, cx=W / 2, cy=H / 2)


def _extrinsics():
    return CameraExtrinsics(location=(0.0, 0.0, 0.0), forward=(1.0, 0.0, 0.0),
                            right=(0.0, 1.0, 0.0), up=(0.0, 0.0, 1.0))


def _write_mask_png(path: Path, arr: np.ndarray) -> None:
    Image.fromarray(arr.astype(np.uint8)).save(str(path))


def _mask_frames() -> dict:
    """帧 1 的 mask：L0=[26,38)×[10,31)，L1=[36,38)×[16,19)（L1 覆盖 L0 重叠区）。

    L1 只覆盖右腕(right_wrist, 37,17)区域，避开右肩(35,13)/右肘(36,15)/右髋(34,21)
    的投影像素——保证遮挡判定无歧义（右腕被遮挡、其余可见）。
    """
    m = np.zeros((H, W), dtype=np.uint8)
    m[10:31, 26:38] = 1  # L0
    m[16:19, 36:38] = 2  # L1（后写，覆盖 → 像素值 2）
    return {"1": m}


def _pose_objects(occluded_all=False):
    occ = [occluded_all] * 17
    return [
        {"entity_id": "L0", "track_id": 1, "keypoints_world": L0_WORLD, "occluded": occ},
        # L1 所有关键点无效 → 应被跳过（不产生无意义标签行）
        {"entity_id": "L1", "track_id": 2,
         "keypoints_world": [[None, None, None]] * 17, "occluded": [False] * 17},
    ]


def _pose_keypoints_jsonl(cam_dir: Path, frames: dict = None):
    frames = frames or {"1": _pose_objects()}
    lines = [{
        "kind": "meta",
        "schema": "futsalmot_pose_keypoints_v1",
        "episode_id": "ep",
        "camera_id": "Cam_01",
        "image_width": W, "image_height": H,
        "keypoint_names": ["nose", "left_eye", "right_eye", "left_ear", "right_ear",
                           "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                           "left_wrist", "right_wrist", "left_hip", "right_hip",
                           "left_knee", "right_knee", "left_ankle", "right_ankle"],
        "coordinate_convention": "meters",
        "occlusion_method": "none",
    }]
    for frame_index, objects in sorted(frames.items(), key=lambda kv: int(kv[0])):
        lines.append({
            "kind": "frame",
            "frame_index": int(frame_index),
            "source_step": int(frame_index) - 1,
            "time_seconds": (int(frame_index) - 1) * 0.1,
            "objects": objects,
        })
    (cam_dir / "pose_keypoints.jsonl").write_text(
        "\n".join(json.dumps(l) for l in lines) + "\n", encoding="utf-8")


def _mask_primary_obj(entity, track, cls, mask_id, xyxy):
    """annotations.jsonl 中 mask-primary（annotate-masks 之后）形态的 object。"""
    xmin, ymin, xmax, ymax = [float(v) for v in xyxy]
    return {
        "entity_id": entity, "track_id": track, "class": cls,
        "team": "left" if entity.startswith("L") else "right",
        "role": None, "is_goalkeeper": False,
        "world_position": [0.0, 0.0, 0.9],
        "mask_id": mask_id,
        "bbox_source": "instance_mask",
        "in_frame": True, "truncated": False, "visibility": None,
        "visible_pixel_count": 100,
        "raw_bbox_xyxy": [xmin, ymin, xmax, ymax],
        "raw_bbox_xywh": [xmin, ymin, xmax - xmin, ymax - ymin],
        "bbox_xyxy": [xmin, ymin, xmax, ymax],
        "bbox_xywh": [xmin, ymin, xmax - xmin, ymax - ymin],
        "segmentation": None,
    }


def _make_camera_dir(root: Path, pose_frames: dict = None, with_geometry_only=False):
    """构造合成 camera 目录（camera.json + annotations.jsonl + img1 + mask + pose_keypoints）。"""
    cam = root / "Cam_01"
    (cam / "gt").mkdir(parents=True, exist_ok=True)
    (cam / "camera.json").write_text(json.dumps(CAMERA_JSON), encoding="utf-8")
    (cam / "seqinfo.ini").write_text("[Sequence]\nname=Cam_01\n", encoding="utf-8")

    if with_geometry_only:
        l0 = {
            "entity_id": "L0", "track_id": 1, "class": "player",
            "team": "left", "role": None, "is_goalkeeper": False,
            "world_position": [0.0, 0.0, 0.9],
            "in_frame": True, "truncated": False, "visibility": None,
            "raw_bbox_xyxy": [26.0, 10.0, 38.0, 31.0],
            "raw_bbox_xywh": [26.0, 10.0, 12.0, 21.0],
            "bbox_xyxy": [26.0, 10.0, 38.0, 31.0],
            "bbox_xywh": [26.0, 10.0, 12.0, 21.0],
        }
        l1 = {
            "entity_id": "L1", "track_id": 2, "class": "player",
            "team": "right", "role": None, "is_goalkeeper": False,
            "world_position": [0.0, 0.0, 0.9],
            "in_frame": True, "truncated": False, "visibility": None,
            "raw_bbox_xyxy": [36.0, 16.0, 38.0, 19.0],
            "raw_bbox_xywh": [36.0, 16.0, 2.0, 3.0],
            "bbox_xyxy": [36.0, 16.0, 38.0, 19.0],
            "bbox_xywh": [36.0, 16.0, 2.0, 3.0],
        }
        ann = [l0, l1]
    else:
        ann = [
            _mask_primary_obj("L0", 1, "player", 1, [26, 10, 38, 31]),
            _mask_primary_obj("L1", 2, "player", 2, [36, 16, 38, 19]),
        ]

    frames = [
        {"episode_id": "ep", "camera_id": "Cam_01", "frame_index": 1, "source_step": 0,
         "time_seconds": 0.0, "objects": ann},
    ]
    with open(cam / "annotations.jsonl", "w", encoding="utf-8") as f:
        for fr in frames:
            f.write(json.dumps(fr) + "\n")

    img1 = cam / "img1"
    mask = cam / "mask"
    img1.mkdir(parents=True, exist_ok=True)
    mask.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.zeros((H, W, 3), dtype=np.uint8)).save(str(img1 / "000001.png"))
    for idx, m in _mask_frames().items():
        _write_mask_png(mask / f"{int(idx):06d}.png", m)

    _pose_keypoints_jsonl(cam, pose_frames)
    return cam


class TestProjection:
    def test_known_world_point_projects_correctly(self):
        uv = _project_keypoints(L0_WORLD, _intrinsics(), _extrinsics())
        for got, want in zip(uv, EXPECTED_PROJ):
            if want is None:
                assert got is None
            else:
                assert got is not None
                assert math.isclose(got[0], want[0], abs_tol=1e-6)
                assert math.isclose(got[1], want[1], abs_tol=1e-6)

    def test_behind_camera_is_none(self):
        assert _project_keypoints([[-1.0, 0.0, 1.0], *([[10.0, 0.0, 1.0]] * 16)],
                                  _intrinsics(), _extrinsics())[0] is None


class TestSerialization:
    def test_line_has_56_fields(self):
        kps = [(10.0, 20.0, 2)] * 17
        line = serialize_pose_line(0, [0.0, 0.0, 100.0, 200.0], 1920, 1080, kps)
        assert len(line.split()) == 56

    def test_normalization_ranges(self):
        # bbox [28,20,38,34] → cx=33/64, cy=27/64, w=10/64, h=14/64
        line = serialize_pose_line(0, [28.0, 20.0, 38.0, 34.0], W, H,
                                   [(32.0, 23.9, 2)] * 17)
        parts = [float(v) for v in line.split()]
        assert parts[0] == 0.0
        cx, cy, w, h = parts[1:5]
        assert math.isclose(cx, 33 / 64, abs_tol=1e-6)
        assert math.isclose(cy, 27 / 64, abs_tol=1e-6)
        assert math.isclose(w, 10 / 64, abs_tol=1e-6)
        assert math.isclose(h, 14 / 64, abs_tol=1e-6)
        # 关键点归一化：u=32 → 32/64；v=23.9 → 23.9/64
        assert math.isclose(parts[5], 32 / 64, abs_tol=1e-6)
        assert math.isclose(parts[6], 23.9 / 64, abs_tol=1e-6)

    def test_wrong_keypoint_count_rejected(self):
        with pytest.raises(ValueError):
            serialize_pose_line(0, [0, 0, 1, 1], W, H, [(0.0, 0.0, 2)] * 16)


class TestVisibility:
    def _mask_arr(self):
        return _mask_frames()["1"]

    def test_own_mask_visible(self):
        vis = _keypoint_visibility((32.0, 23.9), W, H, self._mask_arr(), 1, False, 2)
        assert vis == 2

    def test_other_instance_occluded(self):
        # right_wrist (37.25, 17.0) 在 L1 覆盖区 [36,38)×[16,19)
        vis = _keypoint_visibility((37.25, 17.0), W, H, self._mask_arr(), 1, False, 2)
        assert vis == 1

    def test_outside_image_invalid(self):
        assert _keypoint_visibility((200.0, 200.0), W, H, self._mask_arr(), 1, False, 2) == 0
        assert _keypoint_visibility((-5.0, 3.0), W, H, self._mask_arr(), 1, False, 2) == 0

    def test_free_space_visible(self):
        # 背景像素（0）→ 非遮挡 → 2
        assert _keypoint_visibility((2.0, 2.0), W, H, self._mask_arr(), 1, False, 2) == 2

    def test_occluded_flag_forces_v1(self):
        # 在自身 mask 内但 UE trace 标记被遮挡 → 1（自遮挡 / 非 mask 几何）
        assert _keypoint_visibility((32.0, 23.9), W, H, self._mask_arr(), 1, True, 2) == 1

    def test_edge_tolerance_not_overly_occluded(self):
        # 中心像素是自身、仅 1px 其他实例贴边 → 不判遮挡（避免轮廓边缘大量错误 v=1）
        m = np.zeros((H, W), dtype=np.uint8)
        m[10:30, 10:30] = 1
        m[15, 16] = 2  # 图像坐标 (x=16, y=15) 处放 1px 其他实例，紧邻 (x=15, y=15)
        assert _keypoint_visibility((15.0, 15.0), W, H, m, 1, False, 2) == 2
        # 而关键点中心像素本身就是其他实例 → 明确遮挡
        assert _keypoint_visibility((16.0, 15.0), W, H, m, 1, False, 2) == 1

    def test_mask_none_skips_mask_check(self):
        assert _keypoint_visibility((32.0, 23.9), W, H, None, 1, False, 2) == 2
        assert _keypoint_visibility((32.0, 23.9), W, H, None, 1, True, 2) == 1


class TestEndToEnd:
    def test_annotate_pose_full_flow(self, tmp_path):
        cam = _make_camera_dir(tmp_path)
        rc = annotate_pose_dir(tmp_path, pose_cfg={"visibility_neighborhood_radius": 2}, write_yaml=True)
        assert rc == 0

        # labels_pose 帧与 pose 帧一一对应
        label = cam / "labels_pose" / "000001.txt"
        assert label.exists()
        lines = [ln for ln in label.read_text(encoding="utf-8").splitlines() if ln.strip()]
        assert len(lines) == 1  # 只有 L0（L1 全无效关键点被跳过）
        parts = [float(v) for v in lines[0].split()]
        assert len(parts) == 56
        # bbox 与 mask-primary 一致：cx=(26+38)/2/64=0.5, cy=(10+31)/2/64=20.5/64
        assert math.isclose(parts[1], 0.5, abs_tol=1e-6)
        assert math.isclose(parts[2], 20.5 / 64, abs_tol=1e-6)
        # 关键点可见性：right_wrist=1、left_ankle=0、其余 2
        vis = [int(parts[7 + 3 * i]) for i in range(17)]
        assert vis == EXPECTED_VIS

    def test_annotate_pose_is_idempotent(self, tmp_path):
        """重跑 annotate-pose（含 yolo_pose/ 暂存）不应崩溃（hardlink 幂等）。"""
        cam = _make_camera_dir(tmp_path)
        assert annotate_pose_dir(tmp_path, write_yaml=True) == 0
        assert annotate_pose_dir(tmp_path, write_yaml=True) == 0  # 第二次（staging 已存在）
        label = (cam / "labels_pose" / "000001.txt").read_text(encoding="utf-8")
        assert len(label.split()) == 56

    def test_dataset_yaml(self, tmp_path):
        cam = _make_camera_dir(tmp_path)
        assert annotate_pose_dir(tmp_path, write_yaml=True) == 0
        yaml_path = tmp_path / "futsal_pose.yaml"
        text = yaml_path.read_text(encoding="utf-8")
        assert "kpt_shape: [17, 3]" in text
        assert "flip_idx: [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]" in text
        assert "names:" in text and "0: player" in text
        assert "train: images" in text and "val: images" in text
        assert f"path: {(tmp_path / 'yolo_pose').resolve().as_posix()}" in text
        # yolo_pose 暂存目录：images 硬链接 + labels 副本
        assert (tmp_path / "yolo_pose" / "images" / "Cam_01" / "000001.png").exists()
        assert (tmp_path / "yolo_pose" / "labels" / "Cam_01" / "000001.txt").exists()

    def test_pose_validator_passes(self, tmp_path):
        cam = _make_camera_dir(tmp_path)
        assert annotate_pose_dir(tmp_path, write_yaml=False) == 0
        assert validate_pose_dir(tmp_path, validation_level="full") == 0
        assert validate_pose_dir(tmp_path, validation_level="quick") == 0

    def test_pose_does_not_touch_detection_labels(self, tmp_path):
        """回归：annotate-pose 不覆盖/不破坏 det/seg/MOT 产物。"""
        cam = _make_camera_dir(tmp_path, with_geometry_only=True)
        assert annotate_masks_dir(tmp_path, formats="json,mot,yolo-det,yolo-seg",
                                  include_ball=True) == 0
        det_before = (cam / "labels" / "det" / "000001.txt").read_text(encoding="utf-8")
        mot_before = (cam / "gt" / "gt.txt").read_text(encoding="utf-8")
        assert annotate_pose_dir(tmp_path, write_yaml=False) == 0
        assert (cam / "labels" / "det" / "000001.txt").read_text(encoding="utf-8") == det_before
        assert (cam / "gt" / "gt.txt").read_text(encoding="utf-8") == mot_before
        # 既有 pipeline 校验仍通过
        from grf_ue_bridge.annotation_validator import validate_annotation_dir
        assert validate_annotation_dir(tmp_path, workers=0, validation_level="full") == 0

    def test_pose_bbox_matches_det_bbox(self, tmp_path):
        """Pose 行 bbox 必须与同帧同球员的 YOLO det bbox 完全一致（复用 mask-primary bbox）。"""
        cam = _make_camera_dir(tmp_path, with_geometry_only=True)
        assert annotate_masks_dir(tmp_path, formats="json,mot,yolo-det", include_ball=True) == 0
        det_line = (cam / "labels" / "det" / "000001.txt").read_text(encoding="utf-8").splitlines()[0]
        assert annotate_pose_dir(tmp_path, write_yaml=False) == 0
        pose_line = (cam / "labels_pose" / "000001.txt").read_text(encoding="utf-8").splitlines()[0]
        det = det_line.split()
        pose = pose_line.split()
        # class + bbox 前 5 字段一致
        assert pose[:5] == det[:5]

    def test_empty_frame_allows_empty_txt(self, tmp_path):
        """空场景（无可见球员）允许空 txt。"""
        # 只有 BALL 且所有关键点无效 → 无 pose 行
        cam = _make_camera_dir(tmp_path)
        # 覆盖：L0/L1 全部关键点无效
        _pose_keypoints_jsonl(cam, {"1": [
            {"entity_id": "L0", "track_id": 1,
             "keypoints_world": [[None, None, None]] * 17, "occluded": [False] * 17},
        ]})
        assert annotate_pose_dir(tmp_path, write_yaml=False) == 0
        assert (cam / "labels_pose" / "000001.txt").exists()
        assert (cam / "labels_pose" / "000001.txt").read_text(encoding="utf-8").strip() == ""
        assert validate_pose_dir(tmp_path, validation_level="full") == 0


class TestOverlay:
    def test_pose_overlay_writes_images(self, tmp_path):
        cam = _make_camera_dir(tmp_path)
        assert annotate_pose_dir(tmp_path, write_yaml=False) == 0
        drawn = pose_overlay_dir(cam, frames=[1])
        assert drawn == 1
        assert (cam / "debug" / "pose" / "000001.png").exists()

    def test_pose_overlay_empty_when_no_pose(self, tmp_path):
        cam = _make_camera_dir(tmp_path)
        # 不写 pose_keypoints.jsonl → overlay 应报错返回 0
        (cam / "pose_keypoints.jsonl").unlink()
        assert pose_overlay_dir(cam, frames=[1]) == 0
