"""性能优化链路的回归测试。

覆盖任务要求的 20 项验证点：
  L 单通道快速读取 / 多通道指定通道 / 零复制量化 / 非默认 scale 量化 /
  单帧多实例 pixel_count / bbox / 完全不可见 / 边界实例 / 单连通域 / 多连通域 /
  OpenCV 与旧实现一致 / polygon 栅格化质量 / Cryptomatte uint32 位模式逐像素一致 /
  串行与多进程输出一致 / 不同 worker 数一致 / --no-segmentation / --formats /
  PNG 压缩等级解码一致 / Windows multiprocessing / 现有测试（由测试套件整体保证）。

全部为确定性合成小图，无二进制资产。
"""

import json
import math
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

# 复用 test_mask_annotator 的 fixture 构造（Cam_01，2 帧）
from test_mask_annotator import _make_camera

from instance_mask import (
    _HAS_CV2,
    _mask_to_polygons_cv2,
    compute_instance_stats,
    load_mask_array,
    mask_to_polygons_with_areas,
    polygon_to_mask,
    quantize_mask_pixels,
    raster_quality_metrics,
)
from grf_ue_bridge.mask_annotator import annotate_masks_dir
from grf_ue_bridge.cryptomatte import build_mask, _resolve_actor_plan


def _write_mask_png(path: Path, arr, compress_level=1):
    Image.fromarray(np.asarray(arr, dtype=np.uint8)).save(str(path), compress_level=compress_level)


def _write_rgba_png(path: Path, arr):
    """写一张多通道 RGBA PNG（4 通道），测试指定通道读取。"""
    h, w = arr.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :, 0] = arr   # R = mask_id
    rgba[:, :, 1] = 100
    rgba[:, :, 2] = 200
    rgba[:, :, 3] = 255
    Image.fromarray(rgba).save(str(path))


# ── 1/2. L 单通道快速读取 + 多通道指定通道读取 ──────────────────────

class TestLoadMaskArray:
    def test_l_mode_returns_2d_uint8(self, tmp_path):
        p = tmp_path / "m.png"
        arr = np.array([[0, 1, 2], [3, 0, 11]], dtype=np.uint8)
        _write_mask_png(p, arr)
        out = load_mask_array(p, "r")
        assert out.ndim == 2 and out.shape == arr.shape
        assert out.dtype == np.uint8
        assert np.array_equal(out, arr)
        # 任意 r/g/b 通道在 L 图下都等于灰度值
        for ch in ("r", "g", "b", "gray", "l"):
            assert np.array_equal(load_mask_array(p, ch), arr)

    def test_multichannel_specific_channel(self, tmp_path):
        p = tmp_path / "m_rgba.png"
        arr = np.array([[0, 1, 2], [3, 0, 11]], dtype=np.uint8)
        _write_rgba_png(p, arr)
        assert np.array_equal(load_mask_array(p, "r"), arr)
        assert np.array_equal(load_mask_array(p, "b"), np.full_like(arr, 200))

    def test_l_mode_a_channel_is_255(self, tmp_path):
        p = tmp_path / "m.png"
        _write_mask_png(p, np.array([[1, 2]], dtype=np.uint8))
        assert np.array_equal(load_mask_array(p, "a"), np.array([[255, 255]], dtype=np.uint8))


# ── 3/4. 量化：默认零复制 + 非默认 scale/offset ──────────────────────

class TestQuantize:
    def test_default_integer_passthrough_no_copy(self):
        arr = np.array([[0, 1, 2], [3, 11, 0]], dtype=np.uint8)
        out = quantize_mask_pixels(arr)
        assert out is arr  # 零复制：直接返回原数组
        assert out.dtype == np.uint8

    def test_default_float_rounds(self):
        arr = np.array([[0.4, 0.7, 1.6]], dtype=np.float32)
        out = quantize_mask_pixels(arr)
        assert out.tolist() == [[0, 1, 2]]

    def test_nondefault_scale_offset(self):
        arr = np.array([[0, 25, 50], [75, 0, 25]], dtype=np.int64)
        out = quantize_mask_pixels(arr, id_scale=25)
        assert out.tolist() == [[0, 1, 2], [3, 0, 1]]
        arr2 = np.array([[1, 26]], dtype=np.int64)
        assert quantize_mask_pixels(arr2, id_scale=25, id_offset=1).tolist() == [[0, 1]]


# ── 5/6/7/8. 单次扫描实例统计 ────────────────────────────────────────

def _mask_with_instances():
    """合成标签图：id1 矩形 [2,6)x[3,7)、id2 矩形 [10,13)x[10,14)、id11 孤立像素 (20,20)。"""
    m = np.zeros((32, 32), dtype=np.uint8)
    m[3:7, 2:6] = 1     # 4×4 = 16 px，bbox (2,3,6,7)
    m[10:14, 10:13] = 2 # 4×3 = 12 px，bbox (10,10,13,14)
    m[20, 20] = 11      # 1 px，bbox (20,20,21,21)
    return m


class TestComputeInstanceStats:
    def test_counts(self):
        stats = compute_instance_stats(_mask_with_instances())
        assert stats[1].pixel_count == 16
        assert stats[2].pixel_count == 12
        assert stats[11].pixel_count == 1

    def test_bboxes(self):
        stats = compute_instance_stats(_mask_with_instances())
        assert stats[1].bbox_xyxy == (2, 3, 6, 7)   # xmax=max_x+1、ymax=max_y+1
        assert stats[2].bbox_xyxy == (10, 10, 13, 14)
        assert stats[11].bbox_xyxy == (20, 20, 21, 21)

    def test_roi_slice(self):
        stats = compute_instance_stats(_mask_with_instances())
        m = _mask_with_instances()
        roi = m[stats[1].roi_slice]
        assert roi.shape == (4, 4)  # rows [3,7)、cols [2,6)
        assert (roi == 1).all()

    def test_invisible_absent(self):
        stats = compute_instance_stats(np.zeros((16, 16), dtype=np.uint8))
        assert stats == {}

    def test_boundary_instance(self):
        # 顶到图像左上角边界 (0,0)
        m = np.zeros((8, 8), dtype=np.uint8)
        m[0:2, 0:3] = 5
        stats = compute_instance_stats(m)
        assert stats[5].bbox_xyxy == (0, 0, 3, 2)
        # ROI 切片不得负回绕
        assert m[stats[5].roi_slice].shape == (2, 3)

    def test_consistent_with_mask_to_bbox(self):
        # 与旧 mask_to_bbox（全图 nonzero）逐值一致
        from instance_mask import mask_to_bbox
        m = _mask_with_instances()
        stats = compute_instance_stats(m)
        for mid, st in stats.items():
            bb_old = mask_to_bbox(m == mid)
            assert tuple(st.bbox_xyxy) == tuple(bb_old)


# ── 9/10. 单/多连通域多边形（cv2 路径）────────────────────────────────

class TestPolygonsViaCv2:
    def test_single_component(self):
        m = np.zeros((64, 64), dtype=np.uint8)
        m[10:30, 10:30] = 1
        polys, areas = mask_to_polygons_with_areas(m == 1)
        assert len(polys) == 1 and areas == [400]

    def test_multi_component(self):
        m = np.zeros((64, 64), dtype=np.uint8)
        m[5:12, 20:30] = 1   # 头
        m[20:35, 15:35] = 1  # 躯干
        polys, areas = mask_to_polygons_with_areas(m == 1)
        assert len(polys) == 2
        assert areas == [70, 300]  # 7×10、20×15

    def test_border_components(self):
        # 两个都贴边界的组件（左侧 + 右上角）
        m = np.zeros((40, 40), dtype=np.uint8)
        m[0:5, 0:5] = 1
        m[35:40, 35:40] = 1
        polys, areas = mask_to_polygons_with_areas(m == 1)
        assert len(polys) == 2
        assert sum(areas) == 50

    def test_polygon_rasterization_quality(self):
        # 矩形外轮廓 → even-odd 栅格化：边界环天然欠填（内 2×2）
        block = np.zeros((10, 10), dtype=bool)
        block[1:4, 1:4] = True
        from instance_mask import connected_components, trace_outer_contour
        comp = connected_components(block)[0]
        contour = trace_outer_contour(comp)
        raster = polygon_to_mask(contour, 10, 10)
        assert int(raster.sum()) == 4
        m = raster_quality_metrics(raster, block)
        assert m["iou"] > 0


# ── 11. OpenCV 快速实现与旧实现一致 ──────────────────────────────────

class TestOpenCvMatchesFallback:
    @pytest.mark.parametrize("binary", [
        (lambda m: (m.__setitem__((slice(4, 24), slice(4, 24)), True), m)[1])(np.zeros((64, 64), dtype=bool)),
        (lambda m: (m.__setitem__((slice(5, 12), slice(20, 30)), True),
                    m.__setitem__((slice(20, 35), slice(15, 35)), True), m)[-1])(np.zeros((64, 64), dtype=bool)),
        (lambda m: (m.__setitem__((slice(1, 4), slice(1, 4)), True),
                    m.__setitem__((slice(1, 4), slice(10, 13)), True), m)[-1])(np.zeros((64, 64), dtype=bool)),
    ], ids=["rect", "occluded-person", "two-blobs"])
    def test_cv2_equals_pure_python(self, monkeypatch, binary):
        # cv2 路径（当前默认）
        polys_cv2, areas_cv2 = mask_to_polygons_with_areas(binary)
        # 强制走纯 Python fallback
        monkeypatch.setattr("instance_mask._HAS_CV2", False)
        polys_py, areas_py = mask_to_polygons_with_areas(binary)
        assert polys_cv2 == polys_py  # 逐点一致
        assert areas_cv2 == areas_py

    def test_cv2_path_is_active_by_default(self):
        # 本环境 cv2 已安装 → 默认走 OpenCV 快速路径
        import instance_mask as im
        assert im._HAS_CV2 is True
        assert _mask_to_polygons_cv2(np.ones((5, 5), dtype=bool)) != ([], [])


# ── 13. Cryptomatte uint32 位模式逐像素一致 ──────────────────────────

def _float_to_hex_be(v):
    import struct
    return struct.pack(">f", v).hex()


class TestCryptomatteBitExact:
    def test_build_mask_bit_exact_matches_float(self):
        # 构造 float32 ID 通道：像素值 == 实体的 float32 Actor ID（0 = 背景）
        mapping = {"L0": "Player_L0", "R3": "Player_R3", "BALL": "Ball_01"}
        plan = _resolve_actor_plan(mapping)
        ids_float = {}
        manifest = {}
        for eid, label in mapping.items():
            v = np.float32(np.random.RandomState(abs(hash(eid)) % 1000).uniform(1e-20, 1e-10))
            ids_float[eid] = v
            manifest[label] = _float_to_hex_be(v)
        h = np.zeros((32, 32), dtype=np.float32)
        h[3:7, 3:7] = ids_float["L0"]       # 4×4 = 16
        h[10:12, 20:24] = ids_float["R3"]   # 2×4 = 8
        h[30, 30] = ids_float["BALL"]       # 1
        out_new, counts_new = build_mask(manifest, h, mapping, plan=plan)
        # 旧 float 路径
        from grf_ue_bridge.cryptomatte import hex_id_to_float
        from annotation_utils import entity_id_to_mask_id
        out_old = np.zeros_like(out_new)
        counts_old = {}
        for eid, label in mapping.items():
            m = h == np.float32(hex_id_to_float(manifest[label]))
            n = int(m.sum())
            counts_old[eid] = n
            if n > 0:
                out_old[m] = entity_id_to_mask_id(eid)
        assert counts_new == counts_old
        assert np.array_equal(out_new, out_old)
        # 像素数核对
        assert counts_new["L0"] == 16 and counts_new["R3"] == 8 and counts_new["BALL"] == 1

    def test_bitwise_no_float_confusion(self):
        # 两个相邻的 float32 位模式必须映射到各自 mask_id（绝不混淆）
        from grf_ue_bridge.cryptomatte import hex_id_to_float
        mapping = {"L0": "A", "L1": "B"}
        plan = _resolve_actor_plan(mapping)
        manifest = {"A": "1f21a2e4", "B": "1f21a2e5"}  # 相邻 float32 位模式
        ids = np.zeros((8, 8), dtype=np.float32)
        ids[0, 0] = np.float32(hex_id_to_float("1f21a2e4"))  # 精确命中 A
        out, counts = build_mask(manifest, ids, mapping, plan=plan)
        assert counts["L0"] == 1 and counts["L1"] == 0
        assert int(out[0, 0]) == 1  # mask_id 1 = L0
        ids2 = np.zeros((8, 8), dtype=np.float32)
        ids2[0, 0] = np.float32(hex_id_to_float("1f21a2e5"))  # 精确命中 B
        out2, counts2 = build_mask(manifest, ids2, mapping, plan=plan)
        assert counts2["L1"] == 1 and counts2["L0"] == 0
        assert int(out2[0, 0]) == 2  # mask_id 2 = L1


# ── 14/15/19. 串行 vs 多进程输出一致 + Windows multiprocessing ──────

def _make_two_cameras(root: Path) -> Path:
    """构造含两个相机的目录（Cam_01 + Cam_02，内容相同）。"""
    cam1 = _make_camera(root)
    shutil.copytree(cam1, root / "Cam_02")
    return root / "Cam_02"


class TestParallelDeterminism:
    def test_serial_equals_multiprocess(self, tmp_path):
        root = Path(tmp_path)
        cam1 = _make_camera(root)
        shutil.copytree(cam1, root / "Cam_02")
        root_a = Path(tmp_path) / "a"
        root_b = Path(tmp_path) / "b"
        shutil.copytree(root, root_a)
        shutil.copytree(root, root_b)
        assert annotate_masks_dir(root_a, workers=1) == 0
        assert annotate_masks_dir(root_b, workers=2) == 0
        # 逐字节比较所有产物
        for sub in ("annotations.jsonl", "mask_config.json"):
            assert (root_a / "Cam_01" / sub).read_bytes() == (root_b / "Cam_01" / sub).read_bytes()
        for cam in ("Cam_01", "Cam_02"):
            for rel in ("gt/gt.txt", "labels/det/000001.txt", "labels/seg/000001.txt",
                        "labels/det/000002.txt", "labels/seg/000002.txt"):
                assert (root_a / cam / rel).read_bytes() == (root_b / cam / rel).read_bytes(), rel

    def test_different_worker_counts_identical(self, tmp_path):
        src = Path(tmp_path) / "src"
        _make_camera(src)
        shutil.copytree(src / "Cam_01", src / "Cam_02")
        outputs = {}
        for w in (1, 2, 3):
            d = Path(tmp_path) / f"w{w}"
            shutil.copytree(src, d)  # 只复制干净源，避免把上一轮输出嵌套进去
            assert annotate_masks_dir(d, workers=w) == 0
            outputs[w] = sorted(p.read_bytes() for p in d.rglob("*") if p.is_file())
        assert outputs[1] == outputs[2] == outputs[3]

    def test_windows_multiprocessing_runs(self, tmp_path):
        # 直接在 pytest（Windows spawn）里触发 ProcessPoolExecutor，验证可正常执行
        from concurrent.futures import ProcessPoolExecutor
        from grf_ue_bridge.mask_annotator import AnnotationConfig, AnnotationTask, _annotate_slice_task
        cam_dir = _make_camera(Path(tmp_path))
        config = AnnotationConfig(
            mask_channel="r", include_ball=False, polygon_tolerance_px=1.0,
            max_polygon_points=64, id_scale=1.0, id_offset=0.0,
            formats=frozenset({"json", "mot", "yolo-det", "yolo-seg"}),
            no_segmentation=False, clean_stale=True,
        )
        task = AnnotationTask(camera_dir=cam_dir, start_index=0, end_index=2, config=config)
        with ProcessPoolExecutor(max_workers=2) as ex:
            result = list(ex.map(_annotate_slice_task, [task]))
        assert len(result[0]) == 2  # 2 帧升级成功


# ── 16/17. --no-segmentation / --formats ──────────────────────────────

class TestFastExportModes:
    def test_no_segmentation_skips_seg(self, tmp_path):
        root = Path(tmp_path)
        _make_camera(root)
        assert annotate_masks_dir(root, no_segmentation=True) == 0
        cam = root / "Cam_01"
        assert not (cam / "labels" / "seg").exists()  # 不生成 seg
        det = (cam / "labels" / "det" / "000001.txt").read_text(encoding="utf-8")
        assert det.strip()  # det 正常
        gt = (cam / "gt" / "gt.txt").read_text(encoding="utf-8")
        assert gt.strip()  # MOT 正常
        frames = [json.loads(l) for l in (cam / "annotations.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        for obj in frames[0]["objects"]:
            if obj["entity_id"] == "L0":
                assert obj["bbox_source"] == "instance_mask"
                assert obj["segmentation"] is None
                assert obj["segmentation_components"] == 0
                assert obj["visible_pixel_count"] > 0

    def test_formats_only_selected(self, tmp_path):
        root = Path(tmp_path)
        _make_camera(root)
        assert annotate_masks_dir(root, formats="json,mot") == 0
        cam = root / "Cam_01"
        assert (cam / "gt" / "gt.txt").exists()          # mot 生成
        assert not (cam / "labels" / "det").exists()     # 未选 yolo-det
        assert not (cam / "labels" / "seg").exists()     # 未选 yolo-seg
        assert (cam / "annotations.jsonl").exists()      # json 总是写

    def test_formats_mot_only(self, tmp_path):
        root = Path(tmp_path)
        _make_camera(root)
        assert annotate_masks_dir(root, formats="mot") == 0
        cam = root / "Cam_01"
        assert (cam / "gt" / "gt.txt").exists()
        assert not (cam / "labels").exists() or not (cam / "labels" / "det").exists()

    def test_invalid_format_rejected(self, tmp_path):
        root = Path(tmp_path)
        _make_camera(root)
        with pytest.raises(ValueError):
            annotate_masks_dir(root, formats="bogus")

    def test_no_seg_plus_formats_yolo_det(self, tmp_path):
        root = Path(tmp_path)
        _make_camera(root)
        assert annotate_masks_dir(root, formats="json,mot,yolo-det", no_segmentation=True) == 0
        cam = root / "Cam_01"
        assert (cam / "labels" / "det" / "000001.txt").exists()
        assert not (cam / "labels" / "seg").exists()


# ── 18. PNG 压缩等级解码一致 ─────────────────────────────────────────

class TestPngCompressLevel:
    @pytest.mark.parametrize("level", [0, 1, 6, 9])
    def test_decode_identical(self, tmp_path, level):
        rng = np.random.RandomState(42)
        arr = rng.randint(0, 12, size=(64, 64), dtype=np.uint8)
        p = tmp_path / f"m_{level}.png"
        _write_mask_png(p, arr, compress_level=level)
        out = load_mask_array(p, "r")
        assert np.array_equal(out, arr)


# ── 基准工具 smoke（可解析可运行即可，不测性能数值）────────────────────

class TestBenchmarkToolSmoke:
    def test_benchmark_script_syntactically_valid(self):
        import ast
        src = Path(__file__).resolve().parent.parent / "scripts" / "benchmark_postprocess.py"
        ast.parse(src.read_text(encoding="utf-8"))
