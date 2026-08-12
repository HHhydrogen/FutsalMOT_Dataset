"""debug 可视化模块的测试：全量三套图集（bbox/彩色 mask/pose）+ 三个视频 + 配置。

合成 camera 目录：img1 + mask + annotations.jsonl（mask-primary）+ pose_keypoints.jsonl，
跑 debug_annotations_dir（含视频），断言 debug/ 图集与 video_*.mp4 生成。
"""

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from grf_ue_bridge.debug import (
    _debug_image_sets,
    _frame_number,
    debug_annotations_dir,
    make_video,
    render_camera_debug,
)
from grf_ue_bridge.config.models import PostprocessTaskConfig

W, H = 96, 96

CAMERA_JSON = {
    "camera_id": "Cam_01", "image_width": W, "image_height": H,
    "intrinsics": {"width": W, "height": H, "fx": 50.0, "fy": 50.0, "cx": W / 2, "cy": H / 2},
    "extrinsics": {"world_location_m": [0.0, 0.0, 0.0], "forward": [1.0, 0.0, 0.0],
                   "right": [0.0, 1.0, 0.0], "up": [0.0, 0.0, 1.0]},
}

# 简单 17 点骨架（世界米，x=5，y 取 ±、z 从高到低）
STANDING = [
    [5.0, 0.00, 1.62], [5.0, -0.05, 1.66], [5.0, 0.05, 1.66],
    [5.0, -0.10, 1.60], [5.0, 0.10, 1.60],
    [5.0, -0.25, 1.50], [5.0, 0.25, 1.50],
    [5.0, -0.35, 1.35], [5.0, 0.35, 1.35],
    [5.0, -0.42, 1.20], [5.0, 0.42, 1.20],
    [5.0, -0.15, 0.90], [5.0, 0.15, 0.90],
    [5.0, -0.15, 0.50], [5.0, 0.15, 0.50],
    [5.0, -0.15, 0.15], [5.0, 0.15, 0.15],
]


def _make_camera(root: Path, n_frames: int = 2) -> Path:
    cam = root / "Cam_01"
    (cam / "gt").mkdir(parents=True, exist_ok=True)
    (cam / "camera.json").write_text(json.dumps(CAMERA_JSON), encoding="utf-8")
    (cam / "seqinfo.ini").write_text("[Sequence]\nname=Cam_01\nframeRate=30\n", encoding="utf-8")

    img1 = cam / "img1"
    mask_dir = cam / "mask"
    img1.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    pose_objs_all = []
    for fi in range(1, n_frames + 1):
        x = 2.0 + 0.3 * (fi - 1)  # 球员较近（骨架在图像中 ~40px，连线可见）
        kps = [[x, k[1], k[2]] for k in STANDING]
        # 剪影：关键点 bbox 画椭圆（mask 覆盖所有关键点）
        uvs = []
        for kp in kps:
            u = W / 2 + 50 * kp[1] / kp[0]
            v = H / 2 - 50 * kp[2] / kp[0]
            uvs.append((u, v))
        xs = [u for u, _ in uvs]
        ys = [v for _, v in uvs]
        cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
        rx, ry = (max(xs) - min(xs)) / 2 + 6, (max(ys) - min(ys)) / 2 + 6

        m = np.zeros((H, W), dtype=np.uint8)
        yy, xx = np.ogrid[:H, :W]
        inside = ((xx - cx) / max(1, rx)) ** 2 + ((yy - cy) / max(1, ry)) ** 2 <= 1.0
        m[inside] = 1

        # RGB：绿色场 + 红色球员剪影
        rgb = Image.new("RGB", (W, H), (36, 92, 42))
        from PIL import ImageDraw
        ImageDraw.Draw(rgb).ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=(225, 90, 90))
        rgb.save(str(img1 / f"{fi:06d}.png"))
        Image.fromarray(m.astype(np.uint8)).save(str(mask_dir / f"{fi:06d}.png"))

        bbox = [cx - rx, cy - ry, cx + rx, cy + ry]
        obj = {
            "entity_id": "L0", "track_id": 1, "class": "player", "team": "left",
            "role": None, "is_goalkeeper": False, "world_position": [x, 0.0, 0.9],
            "mask_id": 1, "bbox_source": "instance_mask",
            "in_frame": True, "truncated": False, "visibility": None,
            "visible_pixel_count": 100,
            "raw_bbox_xyxy": [round(v, 3) for v in bbox],
            "raw_bbox_xywh": [round(v, 3) for v in (bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1])],
            "bbox_xyxy": [round(v, 3) for v in bbox],
            "bbox_xywh": [round(v, 3) for v in (bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1])],
            "segmentation": None,
        }
        frames.append({"episode_id": "ep", "camera_id": "Cam_01", "frame_index": fi,
                       "source_step": fi - 1, "time_seconds": (fi - 1) * 0.1, "objects": [obj]})
        pose_objs_all.append({"entity_id": "L0", "track_id": 1, "keypoints_world": kps,
                              "occluded": [False] * 17})
    with open(cam / "annotations.jsonl", "w", encoding="utf-8") as f:
        for fr in frames:
            f.write(json.dumps(fr) + "\n")
    meta = {"kind": "meta", "schema": "futsalmot_pose_keypoints_v1", "episode_id": "ep",
            "camera_id": "Cam_01", "image_width": W, "image_height": H,
            "keypoint_names": ["nose", "left_eye", "right_eye", "left_ear", "right_ear",
                               "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                               "left_wrist", "right_wrist", "left_hip", "right_hip",
                               "left_knee", "right_knee", "left_ankle", "right_ankle"],
            "coordinate_convention": "meters", "occlusion_method": "none"}
    with open(cam / "pose_keypoints.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps(meta) + "\n")
        for fi, objs in enumerate(pose_objs_all, start=1):
            f.write(json.dumps({"kind": "frame", "frame_index": fi, "source_step": fi - 1,
                                "time_seconds": (fi - 1) * 0.1, "objects": [objs]}) + "\n")
    return cam


class TestDebugRender:
    def test_render_camera_debug_all_sets(self, tmp_path):
        cam = _make_camera(tmp_path, n_frames=2)
        counts = render_camera_debug(cam, {"pose_dot_radius": 5, "pose_edge_width": 3})
        assert counts == {"bbox": 2, "mask": 2, "pose": 2}
        assert (cam / "debug" / "000001_bbox.png").exists()
        assert (cam / "debug" / "000002_bbox.png").exists()
        assert (cam / "debug" / "000001_mask_color.png").exists()
        assert (cam / "debug" / "pose" / "000001.png").exists()

    def test_pose_debug_no_keypoint_names(self, tmp_path):
        """pose debug 图不标注关键点名（仅点 + 骨架连线），满足无文字要求。"""
        cam = _make_camera(tmp_path, n_frames=1)
        render_camera_debug(cam, {"pose_dot_radius": 5, "pose_edge_width": 3})
        img = np.array(Image.open(cam / "debug" / "pose" / "000001.png").convert("RGB"))
        # 白色文字像素应极少/无（关键点名会引入大块白字；纯点+线几乎没有纯白像素）
        white = ((img[..., 0] > 240) & (img[..., 1] > 240) & (img[..., 2] > 240)).sum()
        assert white < 50, f"pose debug 图出现疑似文字标注（白色像素 {white} 个）"

    def test_pose_debug_has_skeleton_lines(self, tmp_path):
        """pose debug 图含骨架连线（灰色），YOLO 风格点线。

        96x96 合成图里球员较小，用小点径（dot_radius=2）避免圆点盖住连线；
        真实 640x480 下默认 dot_radius=5 连线仍可见（见 100 帧 demo 的统计）。
        """
        cam = _make_camera(tmp_path, n_frames=1)
        render_camera_debug(cam, {"pose_dot_radius": 2, "pose_edge_width": 2})
        img = np.array(Image.open(cam / "debug" / "pose" / "000001.png").convert("RGB"))
        gray = ((np.abs(img[..., 0].astype(int) - 180) < 50) &
                (np.abs(img[..., 1].astype(int) - 180) < 50) &
                (np.abs(img[..., 2].astype(int) - 185) < 50)).sum()
        assert gray > 20, "pose debug 图缺少骨架连线（灰色像素过少）"

    def test_debug_annotations_dir_with_videos(self, tmp_path):
        cam = _make_camera(tmp_path, n_frames=2)
        rc = debug_annotations_dir(tmp_path, {"make_videos": True, "pose_dot_radius": 5,
                                              "pose_edge_width": 3})
        assert rc == 0
        for name in ("video_bbox.mp4", "video_mask.mp4", "video_pose.mp4"):
            p = cam / name
            assert p.exists(), f"缺少 {name}"
            assert p.stat().st_size > 1000, f"{name} 为空"

    def test_debug_no_videos_flag(self, tmp_path):
        cam = _make_camera(tmp_path, n_frames=1)
        rc = debug_annotations_dir(tmp_path, {"make_videos": False})
        assert rc == 0
        assert not (cam / "video_bbox.mp4").exists()

    def test_skip_pose_set_when_no_pose_keypoints(self, tmp_path):
        cam = _make_camera(tmp_path, n_frames=1)
        (cam / "pose_keypoints.jsonl").unlink()
        counts = render_camera_debug(cam)
        assert counts["bbox"] == 1 and counts["mask"] == 1 and counts["pose"] == 0
        assert not (cam / "debug" / "pose").exists()


class TestFrameNumber:
    def test_frame_number_parsing(self, tmp_path):
        assert _frame_number(Path("debug/000001_bbox.png")) == 1
        assert _frame_number(Path("debug/000012_mask_color.png")) == 12
        assert _frame_number(Path("debug/pose/000003.png")) == 3


class TestDebugConfig:
    def test_config_default_off(self):
        c = PostprocessTaskConfig()
        assert c.debug.enabled is False
        assert c.debug.make_videos is True

    def test_config_fields(self):
        c = PostprocessTaskConfig(debug={"enabled": True, "include_ball": True,
                                         "pose_dot_radius": 6, "video_fps": 25})
        d = c.debug
        assert d.enabled and d.include_ball
        assert d.pose_dot_radius == 6
        assert d.video_fps == 25
