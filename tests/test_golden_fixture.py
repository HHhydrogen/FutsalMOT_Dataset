"""最小 golden fixture：端到端锁定 mask → bbox → MOT → YOLO Detect → YOLO Seg。

全部为确定性合成小图（64×64、2 帧），不提交任何二进制资产。
expected 数值为手工核算 + 首次运行锁定，作为整条 Instance-ID 标注链路的回归锚点：
任何一环（mask 解码 / bbox / 多边形 / MOT / YOLO 序列化）改动都会使 golden 断言失败。
"""

import json
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from grf_ue_bridge.annotation_validator import validate_annotation_dir
from grf_ue_bridge.mask_annotator import annotate_masks_dir

W, H = 64, 64


def _write_png(path: Path, arr, rgb=False):
    if rgb:
        Image.fromarray(np.zeros((H, W, 3), dtype=np.uint8)).save(str(path))
    else:
        Image.fromarray(arr.astype(np.uint8)).save(str(path))


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


def _make_golden_camera(root: Path) -> Path:
    """构造 golden camera：2 帧，无重叠矩形，mask 全部手工可核算。

    帧 1：L0 [4,24)x[4,24)、L1 [30,50)x[30,50)、BALL [55,60)x[55,60)。
    帧 2：L0 [8,28)x[8,28)，L1/BALL 不变。
    R0 几何在画面内 [40,40,44,44] 但 mask 无像素 → not_visible（不可见）。
    """
    cam_dir = root / "Cam_01"
    (cam_dir / "gt").mkdir(parents=True, exist_ok=True)
    cam = {
        "camera_id": "Cam_01",
        "image_width": W,
        "image_height": H,
        "intrinsics": {"width": W, "height": H, "fx": 50.0, "fy": 50.0,
                       "cx": W / 2, "cy": H / 2},
        "extrinsics": {"world_location_m": [0.0, 0.0, 0.0], "forward": [1.0, 0.0, 0.0],
                       "right": [0.0, 1.0, 0.0], "up": [0.0, 0.0, 1.0]},
    }
    (cam_dir / "camera.json").write_text(json.dumps(cam), encoding="utf-8")
    (cam_dir / "seqinfo.ini").write_text("[Sequence]\nname=Cam_01\n", encoding="utf-8")

    l0_1 = _geo_obj("L0", 1, "player", [4, 4, 24, 24])
    l1 = _geo_obj("L1", 2, "player", [30, 30, 50, 50])
    r0 = _geo_obj("R0", 6, "player", [40, 40, 44, 44])
    ball = _geo_obj("BALL", 100, "ball", [55, 55, 60, 60])
    l0_2 = _geo_obj("L0", 1, "player", [8, 8, 28, 28])
    frames = [
        {"episode_id": "golden", "camera_id": "Cam_01", "frame_index": 1,
         "source_step": 0, "time_seconds": 0.0, "objects": [l0_1, l1, r0, ball]},
        {"episode_id": "golden", "camera_id": "Cam_01", "frame_index": 2,
         "source_step": 1, "time_seconds": 0.1, "objects": [l0_2, l1, r0, ball]},
    ]
    with open(cam_dir / "annotations.jsonl", "w", encoding="utf-8") as f:
        for fr in frames:
            f.write(json.dumps(fr) + "\n")

    img1 = cam_dir / "img1"
    mask = cam_dir / "mask"
    img1.mkdir(parents=True, exist_ok=True)
    mask.mkdir(parents=True, exist_ok=True)

    m1 = np.zeros((H, W), dtype=np.uint8)
    m1[4:24, 4:24] = 1    # L0 帧1
    m1[30:50, 30:50] = 2  # L1
    m1[55:60, 55:60] = 11 # BALL
    m2 = np.zeros((H, W), dtype=np.uint8)
    m2[8:28, 8:28] = 1    # L0 帧2
    m2[30:50, 30:50] = 2  # L1
    m2[55:60, 55:60] = 11 # BALL
    for idx, m in ((1, m1), (2, m2)):
        _write_png(img1 / f"{idx:06d}.png", m, rgb=True)
        _write_png(mask / f"{idx:06d}.png", m)
    return cam_dir


def _load_frames(cam_dir):
    return [json.loads(line) for line in
            (cam_dir / "annotations.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]


def _read_lines(path: Path):
    if not path.exists():
        return []
    return [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


class TestGoldenFixture:
    def test_mask_to_bbox_golden(self):
        """mask 可见像素 → bbox / visible_pixel_count 精确锁定。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_golden_camera(root)
            assert annotate_masks_dir(root, include_ball=True) == 0
            f1, f2 = _load_frames(root / "Cam_01")
            o1 = {o["entity_id"]: o for o in f1["objects"]}
            o2 = {o["entity_id"]: o for o in f2["objects"]}
            # 帧 1：L0 [4,4,24,24] 400px；L1 [30,30,50,50] 400px；BALL [55,55,60,60] 25px
            assert o1["L0"]["bbox_xyxy"] == [4.0, 4.0, 24.0, 24.0]
            assert o1["L0"]["bbox_xywh"] == [4.0, 4.0, 20.0, 20.0]
            assert o1["L0"]["visible_pixel_count"] == 400
            assert o1["L1"]["bbox_xyxy"] == [30.0, 30.0, 50.0, 50.0]
            assert o1["L1"]["visible_pixel_count"] == 400
            assert o1["BALL"]["bbox_xyxy"] == [55.0, 55.0, 60.0, 60.0]
            assert o1["BALL"]["visible_pixel_count"] == 25
            # R0 不可见：bbox null、几何保留在 geometry_bbox_*（不回填）
            assert o1["R0"]["bbox_source"] == "not_visible"
            assert o1["R0"]["in_frame"] is False
            assert o1["R0"]["bbox_xyxy"] is None
            assert o1["R0"]["geometry_bbox_xyxy"] == [40.0, 40.0, 44.0, 44.0]
            # 帧 2：L0 移动
            assert o2["L0"]["bbox_xyxy"] == [8.0, 8.0, 28.0, 28.0]
            assert o2["L0"]["visible_pixel_count"] == 400
            # 几何投影保留在 geometry_bbox_*（独立字段，不等于 mask bbox）
            assert o2["L0"]["geometry_bbox_xyxy"] == [8.0, 8.0, 28.0, 28.0]

    def test_mot_golden(self):
        """MOT gt.txt：mask bbox → 整数 MOT 行精确锁定；不可见对象不出现。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_golden_camera(root)
            assert annotate_masks_dir(root, include_ball=True) == 0
            gt = _read_lines(root / "Cam_01" / "gt" / "gt.txt")
            assert gt == [
                "1,1,4,4,20,20,1,1,1.00",
                "1,2,30,30,20,20,1,1,1.00",
                "1,100,55,55,5,5,1,100,1.00",
                "2,1,8,8,20,20,1,1,1.00",
                "2,2,30,30,20,20,1,1,1.00",
                "2,100,55,55,5,5,1,100,1.00",
            ]

    def test_yolo_det_golden(self):
        """YOLO Detect：归一化 cx cy w h 精确锁定。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_golden_camera(root)
            assert annotate_masks_dir(root, include_ball=True) == 0
            det1 = _read_lines(root / "Cam_01" / "labels" / "det" / "000001.txt")
            det2 = _read_lines(root / "Cam_01" / "labels" / "det" / "000002.txt")
            assert det1 == [
                "0 0.218750 0.218750 0.312500 0.312500",   # L0 [4,24] → cx=14/64
                "0 0.625000 0.625000 0.312500 0.312500",   # L1 [30,50] → cx=40/64
                "1 0.898438 0.898438 0.078125 0.078125",   # BALL [55,60] → cx=57.5/64
            ]
            assert det2 == [
                "0 0.281250 0.281250 0.312500 0.312500",   # L0 [8,28] → cx=18/64
                "0 0.625000 0.625000 0.312500 0.312500",
                "1 0.898438 0.898438 0.078125 0.078125",
            ]

    def test_yolo_seg_golden(self):
        """YOLO Seg：矩形 → 4 角点归一化多边形精确锁定。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_golden_camera(root)
            assert annotate_masks_dir(root, include_ball=True) == 0
            seg1 = _read_lines(root / "Cam_01" / "labels" / "seg" / "000001.txt")
            assert len(seg1) == 3
            # 每行 class + 成对归一化点，且坐标 ∈ [0,1]
            for line in seg1:
                parts = line.split()
                vals = [float(v) for v in parts[1:]]
                assert len(vals) % 2 == 0
                assert all(0.0 <= v <= 1.0 for v in vals)
            # L0 帧1 矩形 [4,24]×[4,24] 的归一化角点（顺序由外轮廓跟踪决定，锁定具体值）
            l0_ring = _denorm_ring(seg1[0].split()[1:], W, H)
            # 多边形为像素坐标闭环（末像素 x=23，mask bbox 连续区间 xmax=24）
            assert _ring_bbox(l0_ring) == (4.0, 4.0, 23.0, 23.0)
            ball_ring = _denorm_ring(seg1[2].split()[1:], W, H)
            assert _ring_bbox(ball_ring) == (55.0, 55.0, 59.0, 59.0)
            # 锁定每行精确 flat 值（golden 锚点）
            assert seg1 == GOLDEN_SEG_FRAME1
            seg2 = _read_lines(root / "Cam_01" / "labels" / "seg" / "000002.txt")
            assert seg2 == GOLDEN_SEG_FRAME2

    def test_regression_validator_passes(self):
        """golden fixture 通过完整验收（annotation + dataset regression）。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_golden_camera(root)
            assert annotate_masks_dir(root, include_ball=True) == 0
            assert validate_annotation_dir(root) == 0

    def test_idempotent(self):
        """annotate-masks 重复运行结果一致。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_golden_camera(root)
            annotate_masks_dir(root, include_ball=True)
            first = _load_frames(root / "Cam_01")
            annotate_masks_dir(root, include_ball=True)
            second = _load_frames(root / "Cam_01")
            assert first == second


# ── seg golden 锚点（由实际输出锁定，改动即回归） ──────────────────────
# 每行：class + 归一化 flat 点（5 个点 = 矩形闭环，起点在末尾重复）。
GOLDEN_SEG_FRAME1 = [
    "0 0.062500 0.062500 0.359375 0.062500 0.359375 0.359375 0.062500 0.359375 0.062500 0.078125",  # L0 [4,24]²
    "0 0.468750 0.468750 0.765625 0.468750 0.765625 0.765625 0.468750 0.765625 0.468750 0.484375",  # L1 [30,50]²
    "1 0.859375 0.859375 0.921875 0.859375 0.921875 0.921875 0.859375 0.921875 0.859375 0.875000",  # BALL [55,60]²
]
GOLDEN_SEG_FRAME2 = [
    "0 0.125000 0.125000 0.421875 0.125000 0.421875 0.421875 0.125000 0.421875 0.125000 0.140625",  # L0 [8,28]²
    "0 0.468750 0.468750 0.765625 0.468750 0.765625 0.765625 0.468750 0.765625 0.468750 0.484375",  # L1 不变
    "1 0.859375 0.859375 0.921875 0.859375 0.921875 0.921875 0.859375 0.921875 0.859375 0.875000",  # BALL 不变
]


def _denorm_ring(flat, w, h):
    vals = [float(v) for v in flat]
    return [(vals[i] * w, vals[i + 1] * h) for i in range(0, len(vals), 2)]


def _ring_bbox(ring):
    xs = [p[0] for p in ring]
    ys = [p[1] for p in ring]
    return (min(xs), min(ys), max(xs), max(ys))
