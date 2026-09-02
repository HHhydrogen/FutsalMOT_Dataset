"""标注验证器的测试。"""

import json
import tempfile
from pathlib import Path

from grf_ue_bridge.annotation_validator import validate_annotation_dir


def _player_obj(entity, track, in_frame=True, xywh=None):
    """构造一个合法的 player object（in_frame=False 时 bbox 为 None）。"""
    xyxy = None
    if xywh is not None:
        xyxy = [xywh[0], xywh[1], xywh[0] + xywh[2], xywh[1] + xywh[3]]
    return {
        "entity_id": entity,
        "track_id": track,
        "class": "player",
        "team": "left",
        "role": None,
        "is_goalkeeper": False,
        "world_position": [0.0, 0.0, 0.0],
        "in_frame": in_frame,
        "truncated": False,
        "visibility": None,
        "raw_bbox_xywh": xywh,
        "raw_bbox_xyxy": xyxy,
        "bbox_xywh": xywh,
        "bbox_xyxy": xyxy,
    }


def _write_camera(dir_path: Path, objects_by_frame, width=1920, height=1080, mot=True):
    """写一个合法的 camera 子目录。"""
    (dir_path / "gt").mkdir(parents=True, exist_ok=True)
    cam = {
        "camera_id": dir_path.name,
        "image_width": width,
        "image_height": height,
        "intrinsics": {
            "width": width, "height": height,
            "fx": 1000.0, "fy": 1000.0, "cx": width / 2, "cy": height / 2,
        },
        "extrinsics": {
            "world_location_m": [0.0, 0.0, 0.0],
            "forward": [1.0, 0.0, 0.0],
            "right": [0.0, 1.0, 0.0],
            "up": [0.0, 0.0, 1.0],
        },
    }
    (dir_path / "camera.json").write_text(json.dumps(cam), encoding="utf-8")

    with open(dir_path / "annotations.jsonl", "w", encoding="utf-8") as f:
        for fi, objs in enumerate(objects_by_frame, start=1):
            f.write(json.dumps({
                "episode_id": "episode_0001",
                "camera_id": dir_path.name,
                "frame_index": fi,
                "source_step": fi - 1,
                "time_seconds": (fi - 1) * 0.1,
                "objects": objs,
            }) + "\n")

    (dir_path / "seqinfo.ini").write_text("[Sequence]\nname=x\n", encoding="utf-8")

    if mot:
        lines = []
        for fi, objs in enumerate(objects_by_frame, start=1):
            for o in objs:
                if o.get("in_frame") and o.get("bbox_xywh") is not None:
                    x, y, w, h = o["bbox_xywh"]
                    lines.append(
                        f"{fi},{o['track_id']},{int(x)},{int(y)},{int(w)},{int(h)},1,1,1.00"
                    )
        (dir_path / "gt" / "gt.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_annotation_validator_frame_discovery_accepts_rgb_suffixes(tmp_path):
    from grf_ue_bridge.annotation_validator import _frame_numbers

    img1 = tmp_path / "img1"
    img1.mkdir()
    for suffix in ("png", "jpg", "jpeg"):
        (img1 / f"000001.{suffix}").write_bytes(b"x")
    assert _frame_numbers(img1) == {1}


class TestAnnotationValidator:
    def test_valid_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cam_dir = root / "Camera_01"
            frames = [
                [_player_obj("L0", 1, True, [10.0, 10.0, 50.0, 80.0]),
                 _player_obj("L1", 2, True, [100.0, 100.0, 40.0, 60.0])],
                [_player_obj("L0", 1, True, [11.0, 11.0, 50.0, 80.0]),
                 _player_obj("L1", 2, True, [101.0, 101.0, 40.0, 60.0])],
            ]
            _write_camera(cam_dir, frames)
            assert validate_annotation_dir(root) == 0

    def test_track_conflict(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cam_dir = root / "Camera_01"
            # 同一个 entity L0 在第 2 帧换了 track
            frames = [
                [_player_obj("L0", 1, True, [10.0, 10.0, 50.0, 80.0])],
                [_player_obj("L0", 2, True, [11.0, 11.0, 50.0, 80.0])],
            ]
            _write_camera(cam_dir, frames)
            assert validate_annotation_dir(root) == 1

    def test_out_of_bounds_bbox(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cam_dir = root / "Camera_01"
            # x=2000 超出图像宽 1920
            frames = [[_player_obj("L0", 1, True, [2000.0, 10.0, 50.0, 80.0])]]
            _write_camera(cam_dir, frames)
            assert validate_annotation_dir(root) == 1

    def test_negative_width_bbox(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cam_dir = root / "Camera_01"
            frames = [[_player_obj("L0", 1, True, [10.0, 10.0, -5.0, 80.0])]]
            _write_camera(cam_dir, frames)
            assert validate_annotation_dir(root) == 1

    def test_invalid_mot_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cam_dir = root / "Camera_01"
            frames = [[_player_obj("L0", 1, True, [10.0, 10.0, 50.0, 80.0])]]
            _write_camera(cam_dir, frames, mot=True)
            (cam_dir / "gt" / "gt.txt").write_text(
                "bad line without commas\n", encoding="utf-8"
            )
            assert validate_annotation_dir(root) == 1

    def test_missing_camera_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "Camera_01").mkdir(parents=True)
            assert validate_annotation_dir(root) == 1

    def test_empty_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            assert validate_annotation_dir(Path(tmp)) == 1
