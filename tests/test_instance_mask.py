"""instance_mask.py 纯函数测试：mask 解码 / bbox / 连通域 / 轮廓 / 多边形。"""

import numpy as np

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
        polys = [
            [(0.0, 0.0), (10.0, 0.0), (0.0, 10.0)],
            [(50.0, 50.0), (60.0, 50.0), (50.0, 60.0)],
        ]
        flat = polygon_to_yolo_flat(polys, width=100, height=100)
        assert len(flat) == 12  # 两个三角形 → 12 个归一化坐标

    def test_det_xyxy_to_norm(self):
        cx, cy, w, h = det_xyxy_to_yolo_norm([10, 10, 30, 30], width=100, height=50)
        assert (cx, cy, w, h) == (0.2, 0.4, 0.2, 0.4)

    def test_yolo_class_id(self):
        assert yolo_class_id("L0") == 0
        assert yolo_class_id("R3") == 0
        assert yolo_class_id("BALL") == 1


class TestPixelCount:
    def test_count(self):
        m = _rect_mask(x0=1, y0=1, x1=6, y1=6)  # 5×5 = 25
        assert visible_pixel_count(m) == 25
