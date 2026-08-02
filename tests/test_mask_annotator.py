"""mask_annotator + 扩展 validator 的端到端测试（合成 camera 目录，纯 numpy/PIL）。"""

import json
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from grf_ue_bridge.annotation_validator import validate_annotation_dir
from grf_ue_bridge.mask_annotator import annotate_masks_dir

W, H = 64, 64


def _geo_obj(entity, track, cls, xyxy):
    xmin, ymin, xmax, ymax = [float(v) for v in xyxy]
    return {
        "entity_id": entity,
        "track_id": track,
        "class": cls,
        "team": "left" if entity.startswith("L") else ("right" if entity.startswith("R") else None),
        "role": None,
        "is_goalkeeper": False,
        "world_position": [0.0, 0.0, 0.9],
        "in_frame": True,
        "truncated": False,
        "visibility": None,
        "raw_bbox_xyxy": [xmin, ymin, xmax, ymax],
        "raw_bbox_xywh": [xmin, ymin, xmax - xmin, ymax - ymin],
        "bbox_xyxy": [xmin, ymin, xmax, ymax],
        "bbox_xywh": [xmin, ymin, xmax - xmin, ymax - ymin],
    }


def _write_png(path: Path, arr, rgb=False):
    if rgb:
        Image.fromarray(np.zeros((H, W, 3), dtype=np.uint8)).save(str(path))
    else:
        Image.fromarray(arr.astype(np.uint8)).save(str(path))


def _make_camera(root: Path) -> Path:
    """构造合成 camera 目录（几何 annotations.jsonl + img1 + mask，尚未 annotate-masks）。

    帧 1：L0 矩形 x[10,30) y[10,30)、L1 x[20,40) y[20,40)（L1 遮挡 L0 重叠区）、
          R0 无 mask（完全不可见）。
    帧 2：L0 移到 x[15,35) y[15,35)、L1 不变。
    """
    cam_dir = root / "Cam_01"
    (cam_dir / "gt").mkdir(parents=True, exist_ok=True)
    cam = {
        "camera_id": "Cam_01",
        "image_width": W,
        "image_height": H,
        "intrinsics": {
            "width": W, "height": H,
            "fx": 50.0, "fy": 50.0, "cx": W / 2, "cy": H / 2,
        },
        "extrinsics": {"world_location_m": [0.0, 0.0, 0.0], "forward": [1.0, 0.0, 0.0],
                       "right": [0.0, 1.0, 0.0], "up": [0.0, 0.0, 1.0]},
    }
    (cam_dir / "camera.json").write_text(json.dumps(cam), encoding="utf-8")
    (cam_dir / "seqinfo.ini").write_text("[Sequence]\nname=Cam_01\n", encoding="utf-8")

    l0 = _geo_obj("L0", 1, "player", [10, 10, 30, 30])
    l1 = _geo_obj("L1", 2, "player", [20, 20, 40, 40])
    r0 = _geo_obj("R0", 6, "player", [50, 50, 55, 55])
    ball = _geo_obj("BALL", 100, "ball", [0, 0, 3, 3])
    frames = [
        {"episode_id": "ep", "camera_id": "Cam_01", "frame_index": 1, "source_step": 0,
         "time_seconds": 0.0, "objects": [l0, l1, r0, ball]},
        {"episode_id": "ep", "camera_id": "Cam_01", "frame_index": 2, "source_step": 1,
         "time_seconds": 0.1, "objects": [
            dict(l0, bbox_xyxy=[15.0, 15.0, 35.0, 35.0], bbox_xywh=[15.0, 15.0, 20.0, 20.0],
                 raw_bbox_xyxy=[15.0, 15.0, 35.0, 35.0], raw_bbox_xywh=[15.0, 15.0, 20.0, 20.0]),
            l1, r0, ball]},
    ]
    with open(cam_dir / "annotations.jsonl", "w", encoding="utf-8") as f:
        for fr in frames:
            f.write(json.dumps(fr) + "\n")

    img1 = cam_dir / "img1"
    mask = cam_dir / "mask"
    img1.mkdir(parents=True, exist_ok=True)
    mask.mkdir(parents=True, exist_ok=True)

    m1 = np.zeros((H, W), dtype=np.uint8)
    m1[10:30, 10:30] = 1  # L0
    m1[20:40, 20:40] = 2  # L1（覆盖 L0 重叠区 → L0 可见 300px）
    m2 = np.zeros((H, W), dtype=np.uint8)
    m2[15:35, 15:35] = 1  # L0 移动，与 L1 重叠 15×15=225 → L0 可见 175px
    m2[20:40, 20:40] = 2
    for idx, m in ((1, m1), (2, m2)):
        _write_png(img1 / f"{idx:06d}.png", m, rgb=True)
        _write_png(mask / f"{idx:06d}.png", m)
    return cam_dir


def _load_frames(cam_dir):
    return [json.loads(line) for line in (cam_dir / "annotations.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]


class TestAnnotateMasks:
    def test_bbox_from_mask_and_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            cam = _make_camera(Path(tmp))
            assert annotate_masks_dir(Path(tmp)) == 0
            f1, f2 = _load_frames(cam)
            o = {obj["entity_id"]: obj for obj in f1["objects"]}
            # L0：被 L1 遮挡重叠区 → 可见 300px，bbox 即 mask min/max
            assert o["L0"]["bbox_source"] == "instance_mask"
            assert o["L0"]["bbox_xyxy"] == [10.0, 10.0, 30.0, 30.0]
            assert o["L0"]["bbox_xywh"] == [10.0, 10.0, 20.0, 20.0]
            assert o["L0"]["visible_pixel_count"] == 300
            assert o["L0"]["mask_id"] == 1
            assert o["L0"]["in_frame"] is True
            # L1 全可见
            assert o["L1"]["visible_pixel_count"] == 400
            assert o["L1"]["bbox_xyxy"] == [20.0, 20.0, 40.0, 40.0]
            # R0 完全不可见 → 模态 GT 无 bbox，回退几何、in_frame=False
            assert o["R0"]["bbox_source"] == "geometry"
            assert o["R0"]["visible_pixel_count"] == 0
            assert o["R0"]["in_frame"] is False

            # 帧 2 L0 遮挡后
            o2 = {obj["entity_id"]: obj for obj in f2["objects"]}
            assert o2["L0"]["bbox_xyxy"] == [15.0, 15.0, 35.0, 35.0]
            assert o2["L0"]["visible_pixel_count"] == 175

    def test_track_and_mask_id_stable_across_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            cam = _make_camera(Path(tmp))
            annotate_masks_dir(Path(tmp))
            f1, f2 = _load_frames(cam)
            for f in (f1, f2):
                for obj in f["objects"]:
                    if obj["entity_id"] == "L0":
                        assert obj["track_id"] == 1 and obj["mask_id"] == 1
                    if obj["entity_id"] == "R0":
                        assert obj["track_id"] == 6 and obj["mask_id"] == 6

    def test_mot_gt(self):
        with tempfile.TemporaryDirectory() as tmp:
            cam = _make_camera(Path(tmp))
            annotate_masks_dir(Path(tmp))
            gt = (cam / "gt" / "gt.txt").read_text(encoding="utf-8").strip().splitlines()
            # 帧1：L0(10,10,20,20)、L1(20,20,20,20)；帧2：L0(15,15,20,20)、L1(20,20,20,20)
            assert gt == [
                "1,1,10,10,20,20,1,1,1.00",
                "1,2,20,20,20,20,1,1,1.00",
                "2,1,15,15,20,20,1,1,1.00",
                "2,2,20,20,20,20,1,1,1.00",
            ]
            # R0 不可见未写入；默认不含球

    def test_yolo_det_and_seg(self):
        with tempfile.TemporaryDirectory() as tmp:
            cam = _make_camera(Path(tmp))
            annotate_masks_dir(Path(tmp))
            det1 = (cam / "labels" / "det" / "000001.txt").read_text(encoding="utf-8").strip().splitlines()
            assert det1[0].startswith("0 ")  # player 类别 0
            cx, cy, w, h = (float(v) for v in det1[0].split()[1:])
            assert (cx, cy, w, h) == (20 / W, 20 / W, 20 / W, 20 / W)
            seg1 = (cam / "labels" / "seg" / "000001.txt").read_text(encoding="utf-8").strip().splitlines()
            assert len(seg1) == 2  # L0 + L1
            for line in seg1:
                vals = [float(v) for v in line.split()[1:]]
                assert len(vals) % 2 == 0
                assert all(0.0 <= v <= 1.0 for v in vals)
            # R0（不可见）与空标签：det/seg 各 2 行
            det2 = (cam / "labels" / "det" / "000002.txt").read_text(encoding="utf-8").strip().splitlines()
            assert len(det2) == 2

    def test_include_ball(self):
        with tempfile.TemporaryDirectory() as tmp:
            cam = _make_camera(Path(tmp))
            # 给帧加一个球的 mask（mask_id=11）
            from PIL import Image as PImage
            mask = np.array(PImage.open(cam / "mask" / "000001.png"))
            mask[50:55, 50:55] = 11
            PImage.fromarray(mask.astype(np.uint8)).save(str(cam / "mask" / "000001.png"))
            annotate_masks_dir(Path(tmp), include_ball=True)
            f1 = _load_frames(cam)[0]
            ball = next(o for o in f1["objects"] if o["entity_id"] == "BALL")
            assert ball["bbox_source"] == "instance_mask"
            assert ball["bbox_xyxy"] == [50.0, 50.0, 55.0, 55.0]
            gt = (cam / "gt" / "gt.txt").read_text(encoding="utf-8")
            assert ",100," in gt  # 球 track_id=100 写入 MOT
            det1 = (cam / "labels" / "det" / "000001.txt").read_text(encoding="utf-8")
            assert "\n1 " in det1 or det1.startswith("1 ")  # 球类别 1

    def test_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            cam = _make_camera(Path(tmp))
            annotate_masks_dir(Path(tmp))
            first = _load_frames(cam)
            annotate_masks_dir(Path(tmp))
            second = _load_frames(cam)
            assert first == second


class TestValidatorMask:
    def test_valid_dir_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_camera(root)
            annotate_masks_dir(root)
            assert validate_annotation_dir(root) == 0

    def test_rgb_mask_frame_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cam = _make_camera(root)
            annotate_masks_dir(root)
            (cam / "img1" / "000003.png").write_bytes(b"x")  # 多出一张 RGB
            assert validate_annotation_dir(root) == 1

    def test_illegal_mask_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cam = _make_camera(root)
            annotate_masks_dir(root)
            from PIL import Image as PImage
            mask = np.array(PImage.open(cam / "mask" / "000001.png"))
            mask[3, 3] = 50  # 非法实例 ID
            PImage.fromarray(mask.astype(np.uint8)).save(str(cam / "mask" / "000001.png"))
            assert validate_annotation_dir(root) == 1

    def test_bbox_not_equal_mask(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cam = _make_camera(root)
            annotate_masks_dir(root)
            # 篡改 L0 的 bbox_xyxy 使其与 mask 不一致
            frames = _load_frames(cam)
            frames[0]["objects"][0]["bbox_xyxy"] = [10.0, 10.0, 31.0, 30.0]
            with open(cam / "annotations.jsonl", "w", encoding="utf-8") as f:
                for fr in frames:
                    f.write(json.dumps(fr) + "\n")
            assert validate_annotation_dir(root) == 1

    def test_yolo_out_of_range(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cam = _make_camera(root)
            annotate_masks_dir(root)
            seg = cam / "labels" / "seg" / "000001.txt"
            seg.write_text("0 0.5 0.5 1.5 0.5 1.5 1.5 0.5 1.5\n", encoding="utf-8")
            assert validate_annotation_dir(root) == 1

    def test_bad_yolo_seg_odd_points(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cam = _make_camera(root)
            annotate_masks_dir(root)
            det = cam / "labels" / "det" / "000001.txt"
            det.write_text("0 0.1 0.1 0.2\n", encoding="utf-8")  # 缺 w/h
            assert validate_annotation_dir(root) == 1

    def test_missing_mask_dir_not_error(self):
        # 没有 mask/ 的旧格式目录：validator 不报 mask 相关错误
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cam_dir = _make_camera(root)
            annotate_masks_dir(root)  # 先生成 gt.txt / labels / mask_config
            import shutil
            shutil.rmtree(cam_dir / "mask")
            assert validate_annotation_dir(root) == 0
