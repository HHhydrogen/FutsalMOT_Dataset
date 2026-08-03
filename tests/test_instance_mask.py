"""instance_mask.py 纯函数测试：mask 解码 / bbox / 连通域 / 轮廓 / 多边形。"""

import numpy as np
import pytest

from annotation_utils import entity_id_to_mask_id, mask_id_to_entity_id, valid_mask_ids
from instance_mask import (
    connected_components,
    decode_mask_pixels,
    det_xyxy_to_yolo_norm,
    mask_to_bbox,
    mask_to_polygons,
    polygon_to_yolo_flat,
    quantize_mask_pixels,
    rdp_simplify,
    trace_outer_contour,
    visible_pixel_count,
    yolo_class_id,
)
from instance_mask import polygon_to_mask
from instance_mask import dilate8, raster_quality_metrics
# import 增加
from instance_mask import (
    _ring_has_proper_crossing,
    _segments_properly_cross,
    merge_to_single_ring,
    mask_to_polygons_with_areas,
)


def _rect_mask(h=20, w=20, x0=5, y0=5, x1=15, y1=15):
    """构造矩形区域为 True 的 bool mask（x∈[x0,x1)、y∈[y0,y1)）。"""
    m = np.zeros((h, w), dtype=bool)
    m[y0:y1, x0:x1] = True
    return m


class TestMaskIdMapping:
    def test_entity_to_mask_id(self):
        assert entity_id_to_mask_id("L0") == 1
        assert entity_id_to_mask_id("L4") == 5
        assert entity_id_to_mask_id("R0") == 6
        assert entity_id_to_mask_id("R4") == 10
        assert entity_id_to_mask_id("BALL") == 11

    def test_roundtrip(self):
        for eid in ["L0", "L3", "R2", "BALL"]:
            assert mask_id_to_entity_id(entity_id_to_mask_id(eid)) == eid

    def test_valid_ids(self):
        assert list(valid_mask_ids()) == list(range(1, 12))

    def test_track_ball_separate(self):
        # 球员 mask_id 与 track_id 同构，但 BALL 用 11（track 用 100）
        assert entity_id_to_mask_id("BALL") != 100


class TestDecode:
    def test_decode_exact(self):
        arr = np.array([[0, 1, 0], [1, 2, 1], [0, 1, 0]])
        m1 = decode_mask_pixels(arr, 1)
        assert m1.tolist() == [
            [False, True, False],
            [True, False, True],
            [False, True, False],
        ]

    def test_decode_with_scale_offset(self):
        # 材质把 stencil 乘 25 → 像素值 25/50/75 代表 ID 1/2/3
        arr = np.array([[0, 25, 50], [75, 0, 25]])
        assert decode_mask_pixels(arr, 1, id_scale=25).tolist() == [
            [False, True, False],
            [False, False, True],
        ]
        assert decode_mask_pixels(arr, 2, id_scale=25).any()
        # 偏移：1→0、26→1
        arr2 = np.array([[1, 26]])
        assert decode_mask_pixels(arr2, 1, id_scale=25, id_offset=1).tolist() == [[False, True]]

    def test_quantize_antialias(self):
        # 抗锯齿边缘亚像素值就近归到合法 ID
        arr = np.array([[0.4, 0.7, 1.6]])
        q = quantize_mask_pixels(arr)
        assert q.tolist() == [[0, 1, 2]]


class TestMaskToBbox:
    def test_single_rect(self):
        bb = mask_to_bbox(_rect_mask(x0=2, y0=3, x1=8, y1=7))
        assert bb == (2.0, 3.0, 8.0, 7.0)  # xmax=max_x+1、ymax=max_y+1

    def test_multi_region_tight(self):
        # 两个分离区域 → bbox 覆盖全部
        m = np.zeros((20, 20), dtype=bool)
        m[1:3, 1:4] = True
        m[10:12, 10:13] = True
        assert mask_to_bbox(m) == (1.0, 1.0, 13.0, 12.0)

    def test_empty(self):
        assert mask_to_bbox(np.zeros((10, 10), dtype=bool)) is None

    def test_width_height_is_pixel_count(self):
        bb = mask_to_bbox(_rect_mask(x0=5, y0=5, x1=10, y1=8))
        assert (bb[2] - bb[0], bb[3] - bb[1]) == (5.0, 3.0)


class TestConnectedComponents:
    def test_two_components(self):
        m = np.zeros((10, 10), dtype=bool)
        m[1:3, 1:3] = True
        m[6:8, 6:8] = True
        comps = connected_components(m)
        assert len(comps) == 2
        assert sum(int(c.sum()) for c in comps) == 8

    def test_diagonal_connected(self):
        m = np.zeros((5, 5), dtype=bool)
        m[1, 1] = m[2, 2] = True  # 对角相邻 → 8 连通算一个
        comps = connected_components(m)
        assert len(comps) == 1


class TestTraceContour:
    def test_3x3_square_perimeter(self):
        m = np.zeros((5, 5), dtype=bool)
        m[1:4, 1:4] = True
        contour = trace_outer_contour(m)
        assert len(contour) == 8  # 3×3 方块外边界像素 = 8 个
        assert set(contour) == {
            (1, 1), (2, 1), (3, 1),
            (3, 2), (3, 3),
            (2, 3), (1, 3), (1, 2),
        }

    def test_single_pixel(self):
        m = np.zeros((5, 5), dtype=bool)
        m[2, 2] = True
        contour = trace_outer_contour(m)
        assert contour == [(2, 2)]

    def test_empty(self):
        assert trace_outer_contour(np.zeros((5, 5), dtype=bool)) == []


class TestRdp:
    def test_collinear_collapses(self):
        pts = [(0, 0), (1, 0), (2, 0), (3, 0)]
        assert rdp_simplify(pts, 0.5) == [(0, 0), (3, 0)]

    def test_keeps_corner(self):
        pts = [(0, 0), (5, 0), (10, 5)]
        assert rdp_simplify(pts, 1.0) == [(0, 0), (5, 0), (10, 5)]

    def test_large_tolerance_drops_mid(self):
        pts = [(0, 0), (4, 0.1), (10, 0)]
        assert len(rdp_simplify(pts, 5.0)) == 2


class TestMaskToPolygons:
    def test_rectangle_polygon_corners(self):
        m = _rect_mask(x0=0, y0=0, x1=10, y1=10)
        polys = mask_to_polygons(m, tolerance=1.0)
        assert len(polys) == 1
        pts = set(polys[0])
        assert pts >= {(0, 0), (10, 0), (0, 10), (10, 10)} or (0.0, 0.0) in pts

    def test_two_components_two_polygons(self):
        m = np.zeros((20, 20), dtype=bool)
        m[1:3, 1:3] = True
        m[10:12, 10:12] = True
        polys = mask_to_polygons(m, tolerance=1.0)
        assert len(polys) == 2

    def test_empty(self):
        assert mask_to_polygons(np.zeros((10, 10), dtype=bool)) == []


class TestYoloSerialization:
    def test_polygon_to_yolo_flat_normalized(self):
        polys = [[(5.0, 5.0), (15.0, 5.0), (15.0, 15.0), (5.0, 15.0)]]
        flat = polygon_to_yolo_flat(polys, width=100, height=100)
        assert len(flat) == 8
        assert all(0.0 <= v <= 1.0 for v in flat)
        assert flat[0] == 0.05 and flat[1] == 0.05

    def test_multi_polygon_merged(self):
        # 两个三角形 → 桥接合并为单个 ring（8 点 16 坐标），不再简单串联
        polys = [
            [(0.0, 0.0), (10.0, 0.0), (0.0, 10.0)],
            [(50.0, 50.0), (60.0, 50.0), (50.0, 60.0)],
        ]
        flat = polygon_to_yolo_flat(polys, width=100, height=100)
        assert len(flat) == 16  # 8 个点 × 2 坐标（含桥接往返锚点）
        assert all(0.0 <= v <= 1.0 for v in flat)

    def test_det_xyxy_to_norm(self):
        cx, cy, w, h = det_xyxy_to_yolo_norm([10, 10, 30, 30], width=100, height=50)
        assert (cx, cy, w, h) == (0.2, 0.4, 0.2, 0.4)

    def test_yolo_class_id(self):
        assert yolo_class_id("L0") == 0
        assert yolo_class_id("R3") == 0
        assert yolo_class_id("BALL") == 1


class TestYoloMultiComponent:
    def test_single_component_normalized(self):
        m = np.zeros((64, 64), dtype=bool)
        m[10:40, 10:40] = True
        polys = mask_to_polygons(m)
        flat = polygon_to_yolo_flat(polys, width=64, height=64)
        assert len(flat) % 2 == 0 and len(flat) >= 6
        assert all(0.0 <= v <= 1.0 for v in flat)

    def test_two_components_no_wrong_connection(self):
        # 两碎片：桥接 ring rasterize 回 mask 覆盖两碎片、无大片背景
        m = np.zeros((64, 64), dtype=bool)
        m[5:12, 20:30] = True
        m[20:35, 15:35] = True
        polys, areas = mask_to_polygons_with_areas(m)
        ring, meta = merge_to_single_ring(polys, areas)
        flat = polygon_to_yolo_flat(polys, width=64, height=64)
        assert meta["merged"] is True
        assert all(0.0 <= v <= 1.0 for v in flat)
        # 面积膨胀：merged 不显著超过 Σ 分量（= 原始 mask 语义）
        raster_m = polygon_to_mask(ring, 64, 64)
        assert int(raster_m.sum()) <= int(m.sum()) * 1.2 + 20

    def test_occluded_person_rasterize_back(self):
        # 遮挡人体：头+躯干被遮挡带分开
        m = np.zeros((100, 100), dtype=bool)
        m[5:20, 30:55] = True   # 头
        m[35:70, 20:65] = True  # 躯干
        polys, areas = mask_to_polygons_with_areas(m)
        ring, meta = merge_to_single_ring(polys, areas)
        assert meta["merged"] is True and meta["fallback"] is None
        raster_m = polygon_to_mask(ring, 100, 100)
        # 两碎片均被覆盖
        for poly in polys:
            assert (raster_m & polygon_to_mask(poly, 100, 100)).any()
        # 无明显额外前景面积
        assert int(raster_m.sum()) <= int(m.sum()) * 1.15 + 20

    def test_small_ball_normalized(self):
        m = np.zeros((64, 64), dtype=bool)
        m[30:33, 30:33] = True  # 3×3 小球
        polys = mask_to_polygons(m)
        flat = polygon_to_yolo_flat(polys, width=64, height=64)
        assert all(0.0 <= v <= 1.0 for v in flat)
        assert len(flat) % 2 == 0


class TestPixelCount:
    def test_count(self):
        m = _rect_mask(x0=1, y0=1, x1=6, y1=6)  # 5×5 = 25
        assert visible_pixel_count(m) == 25


class TestPolygonToMask:
    def test_rectangle_full(self):
        m = polygon_to_mask([(0, 0), (4, 0), (4, 4), (0, 4)], 4, 4)
        assert int(m.sum()) == 16
        assert m.all()

    def test_rectangle_offset_fills_centers(self):
        m = polygon_to_mask([(1, 1), (3, 1), (3, 3), (1, 3)], 6, 6)
        ys, xs = np.nonzero(m)
        assert set(zip(xs.tolist(), ys.tolist())) == {(1, 1), (2, 1), (1, 2), (2, 2)}

    def test_triangle_area(self):
        m = polygon_to_mask([(0, 0), (4, 0), (0, 4)], 5, 5)
        assert int(m.sum()) == 6

    def test_contour_underfills_boundary_ring(self):
        # 3×3 方块外轮廓 → 内 2×2（边界环天然欠填，属固有特性）
        block = np.zeros((10, 10), dtype=bool)
        block[1:4, 1:4] = True
        contour = trace_outer_contour(block)
        m = polygon_to_mask(contour, 10, 10)
        assert int(m.sum()) == 4

    def test_empty(self):
        assert not polygon_to_mask([], 4, 4).any()

    def test_offscreen_polygon_no_wrap(self):
        # 完全在图像左侧的多边形：切片不得因负下标回绕而误填整行
        m = polygon_to_mask([(-3, 1), (-1, 1), (-1, 3), (-3, 3)], 5, 5)
        assert not m.any()


class TestRasterQuality:
    def test_dilate8_single_pixel(self):
        m = np.zeros((5, 5), dtype=bool)
        m[2, 2] = True
        d = dilate8(m)
        assert int(d.sum()) == 9
        assert d[1:4, 1:4].all()

    def test_dilate8_corner_clip(self):
        m = np.zeros((3, 3), dtype=bool)
        m[0, 0] = True
        d = dilate8(m)
        assert d[0:2, 0:2].all()
        assert int(d.sum()) == 4

    def test_metrics_perfect(self):
        truth = np.zeros((5, 5), dtype=bool); truth[1:4, 1:4] = True
        m = raster_quality_metrics(truth.copy(), truth)
        assert m["extra_ratio"] == 0.0
        assert m["refined_missing_ratio"] == 0.0
        assert m["iou"] == 1.0
        assert m["raw_missing_ratio"] == 0.0

    def test_metrics_extra_background(self):
        truth = np.zeros((5, 5), dtype=bool); truth[1:4, 1:4] = True
        raster = truth.copy(); raster[0, 0] = True
        m = raster_quality_metrics(raster, truth)
        assert m["extra_ratio"] == pytest.approx(1.0 / 9.0)
        assert m["iou"] < 1.0

    def test_refined_missing_ignores_boundary_ring(self):
        truth = np.zeros((6, 6), dtype=bool); truth[1:4, 1:4] = True  # 9 px
        raster = np.zeros((6, 6), dtype=bool); raster[2, 2] = True    # 中心 1 px
        m = raster_quality_metrics(raster, truth)
        # 边界环缺失像素与 raster 8 邻接 → refined_missing == 0，但 raw_missing > 0
        assert m["refined_missing_ratio"] == 0.0
        assert m["raw_missing_ratio"] > 0.0

    def test_refined_missing_detects_dropped_blob(self):
        truth = np.zeros((10, 10), dtype=bool)
        truth[1:4, 1:4] = True; truth[6:8, 6:8] = True
        raster = np.zeros((10, 10), dtype=bool); raster[1:4, 1:4] = True  # 丢右下碎片
        m = raster_quality_metrics(raster, truth)
        assert m["refined_missing_ratio"] == pytest.approx(4.0 / 13.0)

    def test_metrics_empty_truth(self):
        m = raster_quality_metrics(np.zeros((3, 3), dtype=bool),
                                   np.zeros((3, 3), dtype=bool))
        assert m["iou"] == 1.0


class TestProperCrossing:
    def test_unit_cross(self):
        assert _segments_properly_cross((0, 0), (4, 4), (0, 4), (4, 0)) is True

    def test_collinear_overlap_not_cross(self):
        assert _segments_properly_cross((0, 0), (4, 0), (1, 0), (3, 0)) is False

    def test_t_junction_not_cross(self):
        assert _segments_properly_cross((0, 0), (4, 0), (2, 0), (2, 4)) is False

    def test_shared_endpoint_not_cross(self):
        assert _segments_properly_cross((0, 0), (4, 0), (4, 0), (8, 0)) is False

    def test_naive_concat_ring_crosses(self):
        # 两个三角形 naive 串联 → 闭合边穿过第一个三角形的斜边（即旧 bug）
        tri1 = [(0.0, 0.0), (10.0, 0.0), (0.0, 10.0)]
        tri2 = [(50.0, 50.0), (60.0, 50.0), (50.0, 60.0)]
        assert _ring_has_proper_crossing(tri1 + tri2) is True

    def test_rectangle_ring_no_cross(self):
        assert _ring_has_proper_crossing([(1, 1), (3, 1), (3, 3), (1, 3)]) is False


def _two_hsep_blobs():
    A = np.zeros((20, 20), dtype=bool); A[1:4, 1:4] = True
    B = np.zeros((20, 20), dtype=bool); B[1:4, 10:13] = True
    return A, B


class TestMergeToSingleRing:
    def test_single_passthrough(self):
        poly = [(1, 1), (4, 1), (4, 4), (1, 4)]
        ring, meta = merge_to_single_ring([poly])
        assert ring == poly
        assert meta == {"n_components": 1, "merged": False, "fallback": None,
                        "crossing_detected": False, "fallback_reason": None}

    def test_two_components_bridge_single_ring(self):
        A, B = _two_hsep_blobs()
        ca, cb = trace_outer_contour(A), trace_outer_contour(B)
        ring, meta = merge_to_single_ring([ca, cb])
        assert meta["n_components"] == 2 and meta["merged"] is True
        assert meta["fallback"] is None
        assert _ring_has_proper_crossing(ring) is False
        # 两分量都被覆盖（m 的栅格化覆盖各分量栅格化：ma ⊆ m）
        m = polygon_to_mask(ring, 20, 20)
        assert (m & polygon_to_mask(ca, 20, 20) == polygon_to_mask(ca, 20, 20)).all()
        assert (m & polygon_to_mask(cb, 20, 20) == polygon_to_mask(cb, 20, 20)).all()

    def test_three_components_merged(self):
        A = np.zeros((20, 20), dtype=bool); A[1:4, 1:4] = True
        B = np.zeros((20, 20), dtype=bool); B[1:4, 8:11] = True
        C = np.zeros((20, 20), dtype=bool); C[1:4, 15:18] = True
        polys = [trace_outer_contour(x) for x in (A, B, C)]
        ring, meta = merge_to_single_ring(polys)
        assert meta["n_components"] == 3 and meta["merged"] is True
        assert _ring_has_proper_crossing(ring) is False

    def test_occluded_person_two_components(self):
        # 头 + 躯干被 7px 遮挡带分开
        mask = np.zeros((60, 60), dtype=bool)
        mask[5:12, 20:30] = True   # 头
        mask[20:35, 15:35] = True  # 躯干
        polys, areas = mask_to_polygons_with_areas(mask)
        assert len(polys) == 2
        assert areas == [70, 300]  # 头 7×10、躯干 20×15
        ring, meta = merge_to_single_ring(polys, areas)
        assert meta["merged"] is True and meta["fallback"] is None
        assert _ring_has_proper_crossing(ring) is False

    def test_far_apart_geometry_merge_ok(self):
        # 远距组件在几何层桥接合法（无 crossing，fallback 保持 None）；
        # 面积回退由 mask_annotator 的面积 gate 触发（见 Task 6）
        big = [(0, 0), (40, 0), (40, 40), (0, 40)]
        small = [(60, 60), (64, 60), (64, 64), (60, 64)]
        areas = [1600, 16]
        ring, meta = merge_to_single_ring([small, big], areas)
        assert meta["merged"] is True and meta["fallback"] is None
        assert _ring_has_proper_crossing(ring) is False

    def test_largest_component_fallback_direct(self):
        # 直接测回退分支：防御性 crossing 检查触发的行为
        from instance_mask import _largest_component_fallback
        big = [(0, 0), (40, 0), (40, 40), (0, 40)]
        small = [(60, 60), (64, 60), (64, 64), (60, 64)]
        ring, meta = _largest_component_fallback([small, big], [16, 1600],
                                                 reason="proper_crossing_detected")
        assert ring == big  # 面积最大者
        assert meta["merged"] is False
        assert meta["fallback"] == "largest_component"
        assert meta["fallback_reason"] == "proper_crossing_detected"

    def test_small_ball_single(self):
        ball = np.zeros((64, 64), dtype=bool)
        ball[30:33, 30:33] = True
        polys, areas = mask_to_polygons_with_areas(ball)
        assert len(polys) == 1 and areas == [9]
        ring, meta = merge_to_single_ring(polys, areas)
        assert meta["merged"] is False
        assert len(ring) >= 3


class TestMaskToPolygonsWithAreas:
    def test_areas_match_components(self):
        m = np.zeros((20, 20), dtype=bool)
        m[1:3, 1:3] = True      # 4 px
        m[10:12, 10:12] = True  # 4 px
        polys, areas = mask_to_polygons_with_areas(m)
        assert len(polys) == 2
        assert areas == [4, 4]

    def test_empty(self):
        polys, areas = mask_to_polygons_with_areas(np.zeros((10, 10), dtype=bool))
        assert polys == [] and areas == []
