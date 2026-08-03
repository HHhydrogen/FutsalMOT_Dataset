# 多连通域 YOLO Seg 转换实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复一个实例的 visible mask 含多个 disconnected components 时 YOLO Segment 的错误转换——用最近点桥接合并成单 ring + 面积膨胀检查，失败回退最大连通域，raw mask 与 bbox 不变。

**Architecture:** 在 `ue/instance_mask.py` 新增纯函数（栅格化器、质量指标、交叉检测、桥接合并、面积变体），在 `src/grf_ue_bridge/mask_annotator.py` 集成面积检查与记录字段。单连通域路径逐字节不变；YOLO 单多边形约束下的多连通域输出为记录过的派生近似。

**Tech Stack:** Python 3.9（.venv），numpy + PIL（不新增依赖），pytest（`uv run pytest`）。

## Global Constraints

- **提交纪律（CLAUDE.md）**：绝不自动提交 git；每个 commit 前必须向用户请求并等待明确确认；commit message 与代码注释/文档用简体中文。
- **环境隔离**：`ue/instance_mask.py` 是 numpy+PIL 纯模块，UE 侧不 import；所有新代码保持纯函数、不依赖 unreal。
- **不动的东西**：raw Instance-ID Mask PNG、`mask_to_bbox`、bbox 字段、MRQ、GRF、RGB pipeline、`annotation_validator`、CLI 参数。
- **默认值不变**：RDP tolerance=1.0px、max_points=64；轻量简化在 merge 之前逐分量做，merged ring 不再二次简化。
- **单连通域行为不变**：`polygon_to_yolo_flat` 对单连通域输出与旧实现逐字节一致。
- **面积检查阈值**（mask_annotator 模块常量，不进 CLI）：`extra_ratio ≤ 0.10`、`refined_missing_ratio ≤ 0.05`、`iou ≥ 0.75`；`raw_missing_ratio` 仅诊断不 gate。
- **测试命令**：`uv run pytest tests/test_instance_mask.py -v`、`uv run pytest tests/test_mask_annotator.py -v`。
- 参照设计文档：`docs/superpowers/specs/2026-08-03-multi-component-yolo-seg-design.md`。

---

### Task 1: even-odd 扫描线栅格化器 `polygon_to_mask`

**Files:**
- Modify: `ue/instance_mask.py`（在文件末尾「YOLO 归一化」一节前追加）
- Test: `tests/test_instance_mask.py`

**Interfaces:**
- Produces: `polygon_to_mask(points, width, height) -> np.ndarray(bool)` — 闭合 ring 栅格化；像素 `(px,py)` 中心在多边形内即填充；`points` 为空或 `<3` 返回全 False。

- [ ] **Step 1: 写失败测试**（追加到 `tests/test_instance_mask.py` 的 import 并新增类）

```python
# import 增加
from instance_mask import polygon_to_mask


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
```

- [ ] **Step 2: 运行确认失败**

Run: `uv run pytest tests/test_instance_mask.py::TestPolygonToMask -v`
Expected: `FAIL`（`ImportError: cannot import name 'polygon_to_mask'`）

- [ ] **Step 3: 实现 `polygon_to_mask`**

```python
# ── 多边形栅格化（even-odd 扫描线）─────────────────────────────────────

def polygon_to_mask(points, width, height):
    """把闭合多边形栅格化为 bool mask（even-odd 规则）。

    像素 (px, py) 的中心 (px+0.5, py+0.5) 在多边形内即填充。点序为闭合 ring
    （首尾自动相连），坐标可为 int/float。
    """
    n = len(points)
    mask = np.zeros((height, width), dtype=bool)
    if n < 3:
        return mask
    for py in range(height):
        y = py + 0.5
        xs = []
        for i in range(n):
            x1, y1 = points[i]
            x2, y2 = points[(i + 1) % n]
            if y1 == y2:
                continue  # 水平边不贡献交点
            if (y >= y1) == (y >= y2):
                continue  # 半开区间：顶点只计一次
            xs.append(x1 + (y - y1) * (x2 - x1) / (y2 - y1))
        xs.sort()
        for k in range(0, len(xs) - 1, 2):
            xl, xr = xs[k], xs[k + 1]
            c0 = int(np.ceil(xl - 0.5))
            c1 = int(np.ceil(xr - 0.5))
            if c1 > c0:
                mask[py, max(c0, 0):min(c1, width)] = True
    return mask
```

- [ ] **Step 4: 运行确认通过**

Run: `uv run pytest tests/test_instance_mask.py::TestPolygonToMask -v`
Expected: `PASS`（5 passed）

- [ ] **Step 5: 提交**（先征求用户确认）

```bash
git add ue/instance_mask.py tests/test_instance_mask.py
git commit -m "feat: 新增 even-odd 扫描线多边形栅格化器 polygon_to_mask"
```

---

### Task 2: 面积质量指标 `dilate8` + `raster_quality_metrics`

**Files:**
- Modify: `ue/instance_mask.py`
- Test: `tests/test_instance_mask.py`

**Interfaces:**
- Consumes: `polygon_to_mask`（Task 1，本任务不直接调用，供集成测试用）。
- Produces:
  - `dilate8(binary_mask: np.ndarray) -> np.ndarray(bool)` — 8 邻域膨胀（radius=1）。
  - `raster_quality_metrics(raster, truth) -> dict` — 返回 `{"extra_ratio", "refined_missing_ratio", "iou", "raw_missing_ratio"}`（全 float；`truth` 空时返回理想值）。

- [ ] **Step 1: 写失败测试**

```python
# import 增加
from instance_mask import dilate8, raster_quality_metrics


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
```

- [ ] **Step 2: 运行确认失败**

Run: `uv run pytest tests/test_instance_mask.py::TestRasterQuality -v`
Expected: `FAIL`（import 错误）

- [ ] **Step 3: 实现**

```python
# ── 面积膨胀检查辅助（rasterize 回 mask 的质量指标）────────────────────

def dilate8(binary_mask):
    """8 邻域膨胀（radius=1）。"""
    h, w = binary_mask.shape
    out = binary_mask.copy()
    if h > 1:
        out[1:, :] |= binary_mask[:-1, :]
        out[:-1, :] |= binary_mask[1:, :]
    if w > 1:
        out[:, 1:] |= binary_mask[:, :-1]
        out[:, :-1] |= binary_mask[:, 1:]
    if h > 1 and w > 1:
        out[1:, 1:] |= binary_mask[:-1, :-1]
        out[:-1, 1:] |= binary_mask[1:, :-1]
        out[1:, :-1] |= binary_mask[:-1, 1:]
        out[:-1, :-1] |= binary_mask[1:, 1:]
    return out


def raster_quality_metrics(raster, truth):
    """栅格化多边形 vs 原始 mask 的像素质量指标。

    raster/truth: 同形状 bool ndarray。
    返回 dict：
      extra_ratio          = |raster ∧ ¬truth| / |truth|（背景被填成前景）
      refined_missing_ratio = |{truth∧¬raster 且 8 邻域内无 raster}| / |truth|
                              （被完整丢弃的碎片；边界环天然欠填不计）
      iou                  = |raster ∧ truth| / |raster ∨ truth|
      raw_missing_ratio    = |truth ∧ ¬raster| / |truth|（仅诊断）
    truth 为空时返回理想值（无缺失、IoU=1）。
    """
    truth_count = int(truth.sum())
    if truth_count == 0:
        return {"extra_ratio": 0.0, "refined_missing_ratio": 0.0,
                "iou": 1.0, "raw_missing_ratio": 0.0}
    inter = raster & truth
    union = raster | truth
    extra = raster & ~truth
    raw_missing = truth & ~raster
    refined_missing = raw_missing & ~dilate8(raster)
    return {
        "extra_ratio": int(extra.sum()) / truth_count,
        "refined_missing_ratio": int(refined_missing.sum()) / truth_count,
        "iou": int(inter.sum()) / max(1, int(union.sum())),
        "raw_missing_ratio": int(raw_missing.sum()) / truth_count,
    }
```

- [ ] **Step 4: 运行确认通过**

Run: `uv run pytest tests/test_instance_mask.py::TestRasterQuality -v`
Expected: `PASS`

- [ ] **Step 5: 提交**（先征求用户确认）

```bash
git add ue/instance_mask.py tests/test_instance_mask.py
git commit -m "feat: 新增 rasterize 回 mask 的像素质量指标（dilate8 / raster_quality_metrics）"
```

---

### Task 3: 交叉检测 `_segments_properly_cross` + `_ring_has_proper_crossing`

**Files:**
- Modify: `ue/instance_mask.py`
- Test: `tests/test_instance_mask.py`

**Interfaces:**
- Produces:
  - `_segments_properly_cross(p1, p2, p3, p4) -> bool` — 两条开线段在各自内部点处严格相交；共线重叠/端点相接不算。
  - `_ring_has_proper_crossing(ring) -> bool` — 闭合 ring 是否存在非相邻边的严格交叉。

- [ ] **Step 1: 写失败测试**

```python
# import 增加
from instance_mask import _ring_has_proper_crossing, _segments_properly_cross


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
```

- [ ] **Step 2: 运行确认失败**

Run: `uv run pytest tests/test_instance_mask.py::TestProperCrossing -v`
Expected: `FAIL`（import 错误）

- [ ] **Step 3: 实现**

```python
# ── 多边形合法性检查（proper crossing）────────────────────────────────

def _segments_properly_cross(p1, p2, p3, p4):
    """两条开线段 (p1-p2, p3-p4) 是否在各自内部点处严格相交。

    排除：端点相接、共线重叠（往返零宽桥因此不被误判）。
    """
    def orient(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    d1 = orient(p3, p4, p1)
    d2 = orient(p3, p4, p2)
    d3 = orient(p1, p2, p3)
    d4 = orient(p1, p2, p4)
    return ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
           ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0))


def _ring_has_proper_crossing(ring):
    """闭合 ring 是否存在非相邻边的严格交叉。"""
    n = len(ring)
    for i in range(n):
        p1, p2 = ring[i], ring[(i + 1) % n]
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue  # 边 n-1 与边 0 相邻（闭合）
            q1, q2 = ring[j], ring[(j + 1) % n]
            if _segments_properly_cross(p1, p2, q1, q2):
                return True
    return False
```

- [ ] **Step 4: 运行确认通过**

Run: `uv run pytest tests/test_instance_mask.py::TestProperCrossing -v`
Expected: `PASS`

- [ ] **Step 5: 提交**（先征求用户确认）

```bash
git add ue/instance_mask.py tests/test_instance_mask.py
git commit -m "feat: 新增多边形严格交叉检测（_segments_properly_cross / _ring_has_proper_crossing）"
```

---

### Task 4: 桥接合并 `merge_to_single_ring` + `mask_to_polygons_with_areas`

**Files:**
- Modify: `ue/instance_mask.py`
- Test: `tests/test_instance_mask.py`

**Interfaces:**
- Consumes: `_ring_has_proper_crossing`（Task 3）、`trace_outer_contour`/`rdp_simplify`/`cap_polygon_points`/`mask_to_bbox`/`connected_components`（现有）、`polygon_to_mask`（Task 1，测试用）。
- Produces:
  - `merge_to_single_ring(polygons, areas=None) -> (ring: List[Tuple[float,float]], meta: dict)`
    - `meta = {"n_components": int, "merged": bool, "fallback": None|"largest_component", "crossing_detected": bool, "fallback_reason": None|str}`。
    - `n==1` → 原样返回（`merged=False`）；`n==0` → `([], meta)`。
    - 排序 → 逐次桥接 → 交叉检查（防御性）→ 失败回退最大连通域。
  - `mask_to_polygons_with_areas(binary_mask, tolerance=1.0, max_points=64) -> (polys, areas)`；`areas[i]` 为 `polys[i]` 对应连通域像素数，一一对应。
  - `mask_to_polygons` 保持签名（内部改为委托 `mask_to_polygons_with_areas`）。

- [ ] **Step 1: 写失败测试**

```python
# import 增加
from instance_mask import (
    mask_to_polygons_with_areas,
    merge_to_single_ring,
    _ring_has_proper_crossing,
)


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
        # 两分量都被覆盖
        m = polygon_to_mask(ring, 20, 20)
        assert (m & polygon_to_mask(ca, 20, 20)).all()
        assert (m & polygon_to_mask(cb, 20, 20)).all()

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
```

- [ ] **Step 2: 运行确认失败**

Run: `uv run pytest tests/test_instance_mask.py::TestMergeToSingleRing tests/test_instance_mask.py::TestMaskToPolygonsWithAreas -v`
Expected: `FAIL`（import 错误）

- [ ] **Step 3: 实现**

```python
# ── 多连通域 → 单 ring（YOLO 单多边形约束的派生近似）───────────────────

def mask_to_polygons_with_areas(binary_mask, tolerance=1.0, max_points=64):
    """mask_to_polygons 的变体：同时返回每个多边形的连通域像素数。

    返回 (polys, areas)，两者同序一一对应；mask_to_polygons 跳过的极小
    连通域不计入。
    """
    polys = []
    areas = []
    for comp in connected_components(binary_mask):
        contour = trace_outer_contour(comp)
        if len(contour) < 3:
            continue
        simp = rdp_simplify(contour, tolerance)
        simp = cap_polygon_points(simp, max_points)
        if len(simp) < 3:
            bb = mask_to_bbox(comp)
            if bb is not None:
                xmin, ymin, xmax, ymax = bb
                simp = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
        if len(simp) >= 3:
            polys.append(simp)
            areas.append(int(comp.sum()))
    return polys, areas


def _polygon_bbox_center(poly):
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return ((min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0)


def _nearest_point_pair(ring_a, ring_b):
    """两轮廓间最近边界点对 (ia, ib)；平局取最小索引，确定性。"""
    best = None
    for ia, pa in enumerate(ring_a):
        for ib, pb in enumerate(ring_b):
            d = (pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2
            if best is None or d < best[0]:
                best = (d, ia, ib)
    return best[1], best[2]


def _bridge_splice(ring, poly):
    """把 poly 桥接进 ring：最近边界点对处，各沿原方向走满一圈后拼接。

    结果闭合 ring：往返桥为同段反向（零宽、共线重叠）。由最近点性质可证
    桥接边不穿过任一分量内部，故不产生 strict crossing。
    """
    ia, ib = _nearest_point_pair(ring, poly)
    ring_rot = ring[ia:] + ring[:ia]   # [a, ..., a]
    poly_rot = poly[ib:] + poly[:ib]   # [b, ..., b]
    return ring_rot + poly_rot         # 闭合边 (b, a) = 返回桥


def _shoelace_area(poly):
    s = 0.0
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


def _largest_component_fallback(polygons, areas, reason):
    """回退：只保留最大连通域（areas 缺省用 shoelace 面积）。"""
    if areas is None:
        areas = [_shoelace_area(p) for p in polygons]
    best = max(range(len(polygons)), key=lambda i: areas[i])
    return list(polygons[best]), {
        "n_components": len(polygons), "merged": False,
        "fallback": "largest_component", "crossing_detected": True,
        "fallback_reason": reason,
    }


def merge_to_single_ring(polygons, areas=None):
    """多连通域多边形 → 单个 ring（YOLO 单多边形约束的派生近似）。

    策略：按 bbox 中心排序，逐次最近点桥接；桥接结果做合法性检查
    （防御性，正常输入不触发——最近点桥接边不穿过分量内部）。
    失败 → 回退只保留最大连通域。

    areas: 每分量像素数（可选，回退选最大分量用；缺省用 shoelace 面积）。

    返回 (ring, meta)。
    """
    if not polygons:
        return [], {"n_components": 0, "merged": False, "fallback": None,
                    "crossing_detected": False, "fallback_reason": None}
    if len(polygons) == 1:
        return list(polygons[0]), {"n_components": 1, "merged": False,
                                   "fallback": None, "crossing_detected": False,
                                   "fallback_reason": None}
    ordered = sorted(range(len(polygons)),
                     key=lambda i: _polygon_bbox_center(polygons[i]))
    ring = list(polygons[ordered[0]])
    for idx in ordered[1:]:
        ring = _bridge_splice(ring, polygons[idx])
    if _ring_has_proper_crossing(ring):
        return _largest_component_fallback(polygons, areas,
                                           reason="proper_crossing_detected")
    return ring, {"n_components": len(polygons), "merged": True,
                  "fallback": None, "crossing_detected": False,
                  "fallback_reason": None}
```

同时把 `mask_to_polygons` 改为委托（保持签名与行为）：

```python
def mask_to_polygons(binary_mask, tolerance=1.0, max_points=64):
    """从 binary mask 提取每个连通域的外轮廓并做轻量简化。

    返回像素坐标多边形列表 [[(x, y), ...], ...]（每个连通域一个）。
    """
    polys, _areas = mask_to_polygons_with_areas(binary_mask, tolerance, max_points)
    return polys
```

- [ ] **Step 4: 运行确认通过**

Run: `uv run pytest tests/test_instance_mask.py -v`
Expected: `PASS`（含既有测试 `TestMaskToPolygons`，行为未变）

- [ ] **Step 5: 提交**（先征求用户确认）

```bash
git add ue/instance_mask.py tests/test_instance_mask.py
git commit -m "feat: 多连通域最近点桥接合并为单 ring（merge_to_single_ring）+ 分量面积变体"
```

---

### Task 5: `ring_to_yolo_flat` + `polygon_to_yolo_flat` 接入 merge + 场景测试

**Files:**
- Modify: `ue/instance_mask.py`
- Modify: `tests/test_instance_mask.py`（更新 `TestYoloSerialization.test_multi_polygon_merged`，新增场景测试）

**Interfaces:**
- Consumes: `merge_to_single_ring`（Task 4）。
- Produces:
  - `ring_to_yolo_flat(ring, width, height, precision=6) -> List[float]` — 单 ring → 归一化 flat（坐标 ∈[0,1]）。
  - `polygon_to_yolo_flat(polygons, width, height, precision=6) -> List[float]` — 内部先 `merge_to_single_ring` 再 flatten；单连通域输出与旧实现一致。

- [ ] **Step 1: 更新 `test_multi_polygon_merged`（旧断言固化错误串联行为）**

```python
    def test_multi_polygon_merged(self):
        # 两个三角形 → 桥接合并为单个 ring（8 点 16 坐标），不再简单串联
        polys = [
            [(0.0, 0.0), (10.0, 0.0), (0.0, 10.0)],
            [(50.0, 50.0), (60.0, 50.0), (50.0, 60.0)],
        ]
        flat = polygon_to_yolo_flat(polys, width=100, height=100)
        assert len(flat) == 16  # 8 个点 × 2 坐标（含桥接往返锚点）
        assert all(0.0 <= v <= 1.0 for v in flat)
```

- [ ] **Step 2: 新增场景测试（单连通域 / 遮挡人体 / 小 BALL / rasterize 回 mask）**

```python
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
```

- [ ] **Step 3: 运行确认失败**

Run: `uv run pytest tests/test_instance_mask.py -v`
Expected: `test_multi_polygon_merged` FAIL（仍输出 12 坐标），新场景测试 FAIL（`mask_to_polygons_with_areas` 等已存在，但 `polygon_to_yolo_flat` 尚未 merge）

- [ ] **Step 4: 实现**

```python
# ── YOLO 归一化（单 ring）────────────────────────────────────────────

def ring_to_yolo_flat(ring, width, height, precision=6):
    """单 ring → YOLO seg 归一化 flat 点列表 [x1, y1, x2, y2, ...]。

    坐标 = 像素 / 尺寸，全部 ∈ [0, 1]。
    """
    pts = []
    for x, y in ring:
        pts.append(round(x / width, precision))
        pts.append(round(y / height, precision))
    return pts


def polygon_to_yolo_flat(polygons, width, height, precision=6):
    """把像素坐标多边形列表转为 YOLO seg 归一化 flat 点列表。

    多连通域先桥接合并为单 ring（YOLO 单多边形约束）；单连通域输出与旧
    实现一致。
    """
    ring, _meta = merge_to_single_ring(polygons)
    return ring_to_yolo_flat(ring, width, height, precision)
```

- [ ] **Step 5: 运行确认通过**

Run: `uv run pytest tests/test_instance_mask.py -v`
Expected: `PASS`（全部）

- [ ] **Step 6: 提交**（先征求用户确认）

```bash
git add ue/instance_mask.py tests/test_instance_mask.py
git commit -m "feat: YOLO seg 多连通域输出接入桥接合并（ring_to_yolo_flat）+ 场景测试"
```

---

### Task 6: `mask_annotator` 集成面积检查与记录字段

**Files:**
- Modify: `src/grf_ue_bridge/mask_annotator.py`
- Test: `tests/test_mask_annotator.py`

**Interfaces:**
- Consumes: `mask_to_polygons_with_areas`、`merge_to_single_ring`、`ring_to_yolo_flat`、`polygon_to_mask`、`raster_quality_metrics`（全部 instance_mask，Task 1-5）。
- Produces: `annotations.jsonl` object 新增字段 `segmentation_components` / `segmentation_merged` / `segmentation_fallback` / `segmentation_fallback_reason`。
- 常量：`_AREA_TOL_EXTRA_RATIO=0.10`、`_AREA_TOL_MISSING_RATIO=0.05`、`_AREA_TOL_IOU=0.75`。

- [ ] **Step 1: 写失败测试（端到端：两连通域 / 单连通域 / 回退）**

```python
class TestAnnotateMasksMultiComponent:
    def test_occluded_two_components_merged(self):
        # L0 被 L1 遮挡带一分为二（头 + 躯干），L1 填充中间带
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cam = _make_multi_component_camera(root)  # 见下方 helper
            annotate_masks_dir(root)
            frames = _load_frames(cam)
            o = {obj["entity_id"]: obj for obj in frames[0]["objects"]}
            assert o["L0"]["bbox_source"] == "instance_mask"
            assert o["L0"]["segmentation_components"] == 2
            assert o["L0"]["segmentation_merged"] is True
            assert o["L0"]["segmentation_fallback"] is None
            seg = o["L0"]["segmentation"]
            assert seg is not None and len(seg) % 2 == 0
            assert all(0.0 <= v <= 1.0 for v in seg)
            # bbox 仍严格等于 mask min/max
            assert o["L0"]["bbox_xyxy"] == [15.0, 5.0, 35.0, 35.0]
            # YOLO seg 单行且无跨区连接
            segtxt = (cam / "labels" / "seg" / "000001.txt").read_text(encoding="utf-8").strip().splitlines()
            assert len(segtxt) == 2  # L0 + L1 各一行

    def test_single_component_no_merge_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            cam = _make_camera(Path(tmp))  # 现有 helper，单连通域
            annotate_masks_dir(Path(tmp))
            f1 = _load_frames(cam)[0]
            o = {obj["entity_id"]: obj for obj in f1["objects"]}
            assert o["L0"]["segmentation_components"] == 1
            assert o["L0"]["segmentation_merged"] is False
            assert o["L0"]["segmentation_fallback"] is None

    def test_far_apart_fallback_largest(self):
        # 两碎片相距过远 → 面积 gate 失败 → 回退最大连通域
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cam = _make_far_apart_camera(root)
            annotate_masks_dir(root)
            frames = _load_frames(cam)
            o = {obj["entity_id"]: obj for obj in frames[0]["objects"]}
            assert o["L0"]["segmentation_components"] == 2
            assert o["L0"]["segmentation_merged"] is False
            assert o["L0"]["segmentation_fallback"] == "largest_component"
            assert o["L0"]["segmentation_fallback_reason"]
```

配套 helper（追加到测试文件顶部，`_geo_obj` 之后）：

```python
def _make_multi_component_camera(root):
    """L0 = 头(5..12,20..30) + 躯干(20..35,15..35)，L1 覆盖中间带(12..20,15..35)。"""
    cam_dir = root / "Cam_01"
    (cam_dir / "gt").mkdir(parents=True, exist_ok=True)
    W, H = 64, 64
    cam = {
        "camera_id": "Cam_01", "image_width": W, "image_height": H,
        "intrinsics": {"width": W, "height": H, "fx": 50.0, "fy": 50.0,
                       "cx": W / 2, "cy": H / 2},
        "extrinsics": {"world_location_m": [0, 0, 0], "forward": [1, 0, 0],
                       "right": [0, 1, 0], "up": [0, 0, 1]},
    }
    (cam_dir / "camera.json").write_text(json.dumps(cam), encoding="utf-8")
    (cam_dir / "seqinfo.ini").write_text("[Sequence]\nname=Cam_01\n", encoding="utf-8")
    l0 = _geo_obj("L0", 1, "player", [15, 5, 35, 35])
    l1 = _geo_obj("L1", 2, "player", [15, 12, 35, 20])
    frames = [{"episode_id": "ep", "camera_id": "Cam_01", "frame_index": 1,
               "source_step": 0, "time_seconds": 0.0, "objects": [l0, l1]}]
    with open(cam_dir / "annotations.jsonl", "w", encoding="utf-8") as f:
        for fr in frames:
            f.write(json.dumps(fr) + "\n")
    img1 = cam_dir / "img1"; mask = cam_dir / "mask"
    img1.mkdir(parents=True, exist_ok=True); mask.mkdir(parents=True, exist_ok=True)
    m = np.zeros((H, W), dtype=np.uint8)
    m[5:12, 20:30] = 1    # L0 头
    m[20:35, 15:35] = 1   # L0 躯干
    m[12:20, 15:35] = 2   # L1 遮挡带
    _write_png(img1 / "000001.png", m, rgb=True)
    _write_png(mask / "000001.png", m)
    return cam_dir


def _make_far_apart_camera(root):
    """L0 两个相距极远的碎片 → 桥接 wedge 巨大 → 面积 gate 失败回退。"""
    cam_dir = root / "Cam_01"
    (cam_dir / "gt").mkdir(parents=True, exist_ok=True)
    W, H = 64, 64
    cam = {
        "camera_id": "Cam_01", "image_width": W, "image_height": H,
        "intrinsics": {"width": W, "height": H, "fx": 50.0, "fy": 50.0,
                       "cx": W / 2, "cy": H / 2},
        "extrinsics": {"world_location_m": [0, 0, 0], "forward": [1, 0, 0],
                       "right": [0, 1, 0], "up": [0, 0, 1]},
    }
    (cam_dir / "camera.json").write_text(json.dumps(cam), encoding="utf-8")
    (cam_dir / "seqinfo.ini").write_text("[Sequence]\nname=Cam_01\n", encoding="utf-8")
    l0 = _geo_obj("L0", 1, "player", [5, 5, 55, 55])
    frames = [{"episode_id": "ep", "camera_id": "Cam_01", "frame_index": 1,
               "source_step": 0, "time_seconds": 0.0, "objects": [l0]}]
    with open(cam_dir / "annotations.jsonl", "w", encoding="utf-8") as f:
        for fr in frames:
            f.write(json.dumps(fr) + "\n")
    img1 = cam_dir / "img1"; mask = cam_dir / "mask"
    img1.mkdir(parents=True, exist_ok=True); mask.mkdir(parents=True, exist_ok=True)
    m = np.zeros((H, W), dtype=np.uint8)
    m[5:15, 5:15] = 1    # 碎片1（10×10）
    m[45:55, 45:55] = 1  # 碎片2（10×10，gap ~30px）
    _write_png(img1 / "000001.png", m, rgb=True)
    _write_png(mask / "000001.png", m)
    return cam_dir
```

- [ ] **Step 2: 运行确认失败**

Run: `uv run pytest tests/test_mask_annotator.py -v`
Expected: `FAIL`（`segmentation_components` 字段缺失 / `KeyError`）

- [ ] **Step 3: 实现 `mask_annotator.py` 改动**

顶部 import 增加 `math`，并从 `instance_mask` 增加导入：

```python
import math  # 新增（现有 import json/sys/Path/Dict/List/Optional 保留）

from instance_mask import (  # 扩展现有 import 块
    decode_mask_pixels,
    det_xyxy_to_yolo_norm,
    load_mask_array,
    mask_to_bbox,
    mask_to_polygons_with_areas,
    merge_to_single_ring,
    polygon_to_mask,
    raster_quality_metrics,
    ring_to_yolo_flat,
    visible_pixel_count,
    yolo_class_id,
)
```

常量（模块级，`_MOT_VISIBILITY_MODE` 附近）：

```python
# 多连通域合并的面积膨胀检查阈值（仅多连通域时生效；单连通域不进入）
_AREA_TOL_EXTRA_RATIO = 0.10
_AREA_TOL_MISSING_RATIO = 0.05
_AREA_TOL_IOU = 0.75
```

替换 `_upgrade_object` 的 mask 可见分支：

```python
    if binary.any():
        bbox = mask_to_bbox(binary)
        xywh = xyxy_to_xywh(bbox)
        polys, comp_areas = mask_to_polygons_with_areas(
            binary, polygon_tolerance_px, max_polygon_points
        )
        n_components = len(polys)
        ring, meta = merge_to_single_ring(polys, comp_areas) if polys else ([], None)
        seg_flat = ring_to_yolo_flat(ring, width, height) if ring else None
        fallback = meta.get("fallback") if meta else None
        fallback_reason = meta.get("fallback_reason") if meta else None
        # 面积膨胀检查（仅多连通域且桥接成功时）
        if n_components > 1 and fallback is None and seg_flat:
            reason = _check_area_quality(ring, binary, width, height)
            if reason:
                best = max(range(len(polys)), key=lambda i: comp_areas[i])
                ring = list(polys[best])
                seg_flat = ring_to_yolo_flat(ring, width, height)
                fallback = "largest_component"
                fallback_reason = reason
        obj.update({
            "mask_id": mask_id,
            "bbox_source": "instance_mask",
            "in_frame": True,
            "truncated": False,  # mask bbox 必在图像内
            "visibility": None,
            "visible_pixel_count": visible_pixel_count(binary),
            "bbox_xyxy": [round(v, 3) for v in bbox],
            "bbox_xywh": [round(v, 3) for v in xywh],
            "segmentation": seg_flat,
            "segmentation_components": n_components,
            "segmentation_merged": n_components > 1 and fallback is None,
            "segmentation_fallback": fallback,
            "segmentation_fallback_reason": fallback_reason,
            "geometry_bbox_xyxy": [round(v, 3) for v in geo_xyxy] if geo_xyxy else None,
            "geometry_bbox_xywh": [round(v, 3) for v in geo_xywh] if geo_xywh else None,
        })
```

几何 fallback 分支（`binary` 空）补全新字段：

```python
    else:
        obj.update({
            "mask_id": mask_id,
            "bbox_source": "geometry",
            "in_frame": False,
            "visible_pixel_count": 0,
            "segmentation": None,
            "segmentation_components": 0,
            "segmentation_merged": False,
            "segmentation_fallback": None,
            "segmentation_fallback_reason": None,
            "bbox_xyxy": geo_xyxy,
            "bbox_xywh": geo_xywh,
            "geometry_bbox_xyxy": geo_xyxy,
            "geometry_bbox_xywh": geo_xywh,
        })
```

新增 `_check_area_quality`（`_upgrade_object` 之前）：

```python
def _check_area_quality(ring, binary, width, height):
    """ROI 内栅格化 merged ring 与原始 mask 对比；返回失败原因字符串或 None。

    Gate：extra_ratio ≤ tol、refined_missing_ratio ≤ tol、iou ≥ tol；
    任一项失败或栅格化异常即返回原因（由调用方回退最大连通域）。
    """
    try:
        xs = [p[0] for p in ring]
        ys = [p[1] for p in ring]
        x0 = max(0, int(math.floor(min(xs))))
        y0 = max(0, int(math.floor(min(ys))))
        x1 = min(width, int(math.ceil(max(xs))) + 1)
        y1 = min(height, int(math.ceil(max(ys))) + 1)
        roi_w, roi_h = x1 - x0, y1 - y0
        if roi_w <= 0 or roi_h <= 0:
            return "empty_roi"
        shifted = [(x - x0, y - y0) for x, y in ring]
        raster = polygon_to_mask(shifted, roi_w, roi_h)
        truth = binary[y0:y1, x0:x1]
        m = raster_quality_metrics(raster, truth)
        if m["extra_ratio"] > _AREA_TOL_EXTRA_RATIO:
            return "extra_ratio=%.3f>%.2f" % (m["extra_ratio"], _AREA_TOL_EXTRA_RATIO)
        if m["refined_missing_ratio"] > _AREA_TOL_MISSING_RATIO:
            return "refined_missing_ratio=%.3f>%.2f" % (
                m["refined_missing_ratio"], _AREA_TOL_MISSING_RATIO)
        if m["iou"] < _AREA_TOL_IOU:
            return "iou=%.3f<%.2f" % (m["iou"], _AREA_TOL_IOU)
        return None
    except Exception as e:
        return "rasterization_error: %s" % e
```

- [ ] **Step 4: 运行确认通过**

Run: `uv run pytest tests/test_mask_annotator.py -v`
Expected: `PASS`（既有测试 + 新测试；`test_bbox_from_mask_and_sources` 等不受新字段影响）

- [ ] **Step 5: 提交**（先征求用户确认）

```bash
git add src/grf_ue_bridge/mask_annotator.py tests/test_mask_annotator.py
git commit -m "feat: annotate-masks 集成多连通域面积膨胀检查与 fallback 记录字段"
```

---

### Task 7: 文档同步（README + CLAUDE.md）

**Files:**
- Modify: `README.md`（「CV Dataset Annotation Export」的 `segmentation` 字段说明 + `labels/seg/` 描述）
- Modify: `CLAUDE.md`（架构段 `instance_mask.py` / `mask_annotator.py` 一句话同步）

- [ ] **Step 1: 更新 README `segmentation` 字段说明**

定位 `README.md` 第 320 行附近：

```markdown
- `segmentation`：模态分割，YOLO 归一化 flat 点列表 `[x1,y1,x2,y2,...]`。单连通域为单个多边形；**多连通域（`segmentation_components>1`）时为派生近似**——YOLO 单多边形限制下用最近点桥接合并为单个 ring（弱简单，含零宽连接），raw Instance-ID Mask 始终为 canonical GT。`segmentation_fallback` 非空表示合并失败已回退为最大连通域。
- `segmentation_components`：可见连通域碎片数（1 = 单连通域；0 = 完全不可见）。
- `segmentation_merged`：是否经桥接合并为单个 ring（多连通域且未回退时为 true）。
- `segmentation_fallback`：`null` 或 `"largest_component"`（合并的合法性/面积膨胀检查失败时回退只保留最大连通域）。
- `segmentation_fallback_reason`：回退原因（如 `extra_ratio=0.42>0.10`）；无回退时为 `null`。
```

同时更新第 440 行附近的 `labels/seg/` 描述：

```markdown
- YOLO Segment `labels/seg/`：每行 `class x1 y1 x2 y2 ...`（归一化多边形，RDP 简化后 ≤ `max_polygon_points` 点，多连通域用最近点桥接合并为单行；raw mask 为最高精度 GT）。
```

- [ ] **Step 2: 更新 CLAUDE.md 架构段**

`ue/instance_mask.py` 描述追加（第 77 行附近）：

```markdown
- `ue/instance_mask.py`（纯 numpy+PIL，pytest 可测，**UE 侧不 import**——UE Python 无 numpy）— mask 解码/量化、pixel-tight bbox、连通域、Moore 边界跟踪、RDP 多边形简化、YOLO 归一化。多连通域 → YOLO 单多边形：`merge_to_single_ring` 最近点桥接合并成单 ring + `polygon_to_mask` even-odd 栅格化做面积膨胀检查，失败回退最大连通域（详见 specs/2026-08-03-multi-component-yolo-seg-design.md）。
```

`mask_annotator.py` 描述（第 84 行附近）追加：

```markdown
- `src/grf_ue_bridge/mask_annotator.py`（P1 纯 Python，import `ue/instance_mask`）— `grf-ue annotate-masks`：读 `mask/` + 几何 `annotations.jsonl` → 覆盖写 mask-primary `annotations.jsonl`（多连通域合并面积检查，记录 `segmentation_components/merged/fallback`）/ MOT / YOLO det / YOLO seg（幂等）。
```

- [ ] **Step 3: 提交**（先征求用户确认）

```bash
git add README.md CLAUDE.md
git commit -m "docs: 同步多连通域 YOLO Seg 桥接合并与回退字段说明（README + CLAUDE.md）"
```

---

## 收尾

- [ ] 全量测试：`uv run pytest -v`，确认全部通过（现有 + 新增）。
- [ ] 运行 `git status` 确认工作区只含预期改动。
- [ ] 汇总修改文件、采用的 multi-component 策略与测试结果，向用户汇报。
- [ ] 按用户原任务要求提交一个独立 git commit（在用户确认后）。
