"""Instance-ID Mask 纯函数处理模块（numpy + PIL）。

mask 约定
---------
- mask 像素值 == 实体稳定 mask_id（L0..L4→1..5、R0..R4→6..10、BALL→11），背景 = 0。
  mask_id 映射见 annotation_utils（纯 Python，UE 侧可 import 用于打 Custom Depth Stencil）。
- 解码时对像素值做量化（round）以吸收抗锯齿边缘的亚像素混叠。
- 本模块依赖 numpy/PIL，**不依赖 unreal**，供 P1 CLI（grf-ue annotate-masks）与 pytest 使用；
  UE 侧不 import 本模块（UE Python 无 numpy）。

bbox 约定（与 geometry 投影一致的连续坐标）：
- pixel-tight bbox 返回 (xmin, ymin, xmax, ymax)，其中 xmax = max_x + 1、ymax = max_y + 1，
  即左闭右开的连续区间，恰好覆盖全部 mask 像素。宽度 = xmax - xmin = x 向像素数。
"""

import glob
import math
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

# 8 邻域方向（图像坐标：x 向右、y 向下），顺时针排列（索引增大 = 顺时针）
_DIRS: Tuple[Tuple[int, int], ...] = [
    (0, -1),   # 0 上
    (1, -1),   # 1 右上
    (1, 0),    # 2 右
    (1, 1),    # 3 右下
    (0, 1),    # 4 下
    (-1, 1),   # 5 左下
    (-1, 0),   # 6 左
    (-1, -1),  # 7 左上
]

# YOLO 类别编号：0 = player，1 = ball
YOLO_CLASS_PLAYER = 0
YOLO_CLASS_BALL = 1


# ── mask 读取与解码 ─────────────────────────────────────────────────────

def load_mask_array(path, channel: str = "r"):
    """读取 mask PNG 并返回携带实例 ID 的单通道数组 (H, W) uint8。

    channel: "r"/"g"/"b"/"a" → 取 RGBA 的对应通道；"gray"/"l" → 转灰度单通道。
    """
    from PIL import Image

    img = Image.open(str(path))
    if channel in ("r", "g", "b", "a"):
        rgba = img.convert("RGBA")
        idx = {"r": 0, "g": 1, "b": 2, "a": 3}[channel]
        arr = np.asarray(rgba, dtype=np.uint8)[:, :, idx]
    else:
        arr = np.asarray(img.convert("L"), dtype=np.uint8)
    return arr


def quantize_mask_pixels(
    mask_img: np.ndarray,
    id_scale: float = 1.0,
    id_offset: float = 0.0,
) -> np.ndarray:
    """把 mask 单通道数组量化为整数实例 ID 数组。

    量化值 = round((v - id_offset) / id_scale)，用于吸收抗锯齿边缘混叠、
    以及材质把 stencil 乘了 id_scale / 加了 id_offset 的情形。
    """
    if id_scale != 1.0 or id_offset != 0.0:
        vals = (mask_img.astype(np.float64) - id_offset) / id_scale
    else:
        vals = mask_img.astype(np.float64)
    return np.rint(vals).astype(np.int64)


def decode_mask_pixels(
    mask_img: np.ndarray,
    mask_id: int,
    id_scale: float = 1.0,
    id_offset: float = 0.0,
) -> np.ndarray:
    """返回 bool 2D mask：像素值量化后 == mask_id。

    mask_img: (H, W) 单通道数组。量化值 = round((v - id_offset) / id_scale)。
    """
    quantized = quantize_mask_pixels(mask_img, id_scale, id_offset)
    return quantized == int(mask_id)


def visible_pixel_count(binary_mask: np.ndarray) -> int:
    """可见像素数（= mask 非零像素数）。"""
    return int(np.count_nonzero(binary_mask))


def mask_to_bbox(binary_mask: np.ndarray) -> Optional[Tuple[float, float, float, float]]:
    """从 binary mask 计算 pixel-tight bbox。

    返回 (xmin, ymin, xmax, ymax)，xmax = max_x + 1、ymax = max_y + 1（连续坐标，
    恰好覆盖全部 mask 像素）。空 mask 返回 None。
    """
    ys, xs = np.nonzero(binary_mask)
    if not xs.size:
        return None
    return (
        float(xs.min()),
        float(ys.min()),
        float(xs.max() + 1),
        float(ys.max() + 1),
    )


# ── 连通域标记（8 连通，矢量泛洪）──────────────────────────────────────

def _flood_fill(binary: np.ndarray, seed: Tuple[int, int]) -> np.ndarray:
    """从 seed 像素做 8 连通泛洪，返回该连通域的 bool mask。"""
    comp = np.zeros_like(binary, dtype=bool)
    frontier = np.zeros_like(binary, dtype=bool)
    frontier[seed[0], seed[1]] = True
    while frontier.any():
        comp |= frontier
        neigh = np.zeros_like(frontier, dtype=bool)
        # 4 邻域
        neigh[1:, :] |= frontier[:-1, :]   # 下
        neigh[:-1, :] |= frontier[1:, :]   # 上
        neigh[:, 1:] |= frontier[:, :-1]   # 右
        neigh[:, :-1] |= frontier[:, 1:]   # 左
        # 4 对角
        neigh[1:, 1:] |= frontier[:-1, :-1]     # 右下
        neigh[:-1, 1:] |= frontier[1:, :-1]     # 右上
        neigh[1:, :-1] |= frontier[:-1, 1:]     # 左下
        neigh[:-1, :-1] |= frontier[1:, 1:]     # 左上
        neigh &= binary & ~comp
        frontier = neigh
    return comp


def connected_components(binary_mask: np.ndarray) -> List[np.ndarray]:
    """把 binary mask 分成 8 连通域，返回每个连通域的 bool mask 列表。"""
    components: List[np.ndarray] = []
    remaining = binary_mask.copy()
    while remaining.any():
        ys, xs = np.nonzero(remaining)
        comp = _flood_fill(remaining, (int(ys[0]), int(xs[0])))
        components.append(comp)
        remaining &= ~comp
    return components


# ── 外轮廓提取（Moore-Neighbor 边界跟踪）────────────────────────────────

def _neighbor_dir(cx: int, cy: int, target: Tuple[int, int]) -> Optional[int]:
    """target 相对 (cx, cy) 的 8 邻域方向索引；不相邻返回 None。"""
    dx, dy = target[0] - cx, target[1] - cy
    try:
        return _DIRS.index((dx, dy))
    except ValueError:
        return None


def trace_outer_contour(binary: np.ndarray) -> List[Tuple[int, int]]:
    """Moore-Neighbor 边界跟踪：返回单连通域的外边界像素点序列（图像坐标）。

    起点取最上、最左的 mask 像素（np.nonzero 行优先），沿外边界顺时针走一圈，
    回到起点前停止。孤立像素返回 [起点]。输入为单连通域 bool mask。
    """
    ys, xs = np.nonzero(binary)
    if not xs.size:
        return []
    h, w = binary.shape
    start = (int(xs[0]), int(ys[0]))
    # 起点最上最左 → 其西侧像素必为背景
    bg = (start[0] - 1, start[1])
    bg_dir = 6  # 起点背景方向（正西）
    cur = start
    contour: List[Tuple[int, int]] = []
    limit = h * w + 1
    while len(contour) < limit:
        contour.append(cur)
        cx, cy = cur
        rel = _neighbor_dir(cx, cy, bg)
        start_dir = rel if rel is not None else bg_dir
        found = None
        for k in range(1, 9):
            d = (start_dir + k) % 8
            nx, ny = cx + _DIRS[d][0], cy + _DIRS[d][1]
            if 0 <= ny < h and 0 <= nx < w and binary[ny, nx]:
                found = (nx, ny)
                new_bg_dir = (d - 1) % 8
                bg = (cx + _DIRS[new_bg_dir][0], cy + _DIRS[new_bg_dir][1])
                bg_dir = new_bg_dir
                break
        if found is None:
            return contour  # 孤立像素
        if found == start and len(contour) > 1:
            return contour
        cur = found
    return contour


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
            lo = max(c0, 0)
            hi = min(c1, width)
            if hi > lo:
                mask[py, lo:hi] = True
    return mask


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


# ── 多边形简化（Ramer-Douglas-Peucker）与归一化 ─────────────────────────

def _point_segment_distance(p, a, b) -> float:
    """点到线段（a-b）的垂直距离。"""
    x, y = p
    x1, y1 = a
    x2, y2 = b
    dx, dy = x2 - x1, y2 - y1
    if dx == 0.0 and dy == 0.0:
        return math.hypot(x - x1, y - y1)
    t = max(0.0, min(1.0, ((x - x1) * dx + (y - y1) * dy) / (dx * dx + dy * dy)))
    px, py = x1 + t * dx, y1 + t * dy
    return math.hypot(x - px, y - py)


def rdp_simplify(points: Sequence[Tuple[float, float]], tolerance: float) -> List[Tuple[float, float]]:
    """Ramer-Douglas-Peucker 折线简化。返回简化后的点列表（含首尾点）。"""
    if len(points) <= 2:
        return list(points)
    start, end = points[0], points[-1]
    dmax, index = 0.0, -1
    for i in range(1, len(points) - 1):
        d = _point_segment_distance(points[i], start, end)
        if d > dmax:
            dmax, index = d, i
    if dmax > tolerance:
        left = rdp_simplify(points[: index + 1], tolerance)
        right = rdp_simplify(points[index:], tolerance)
        return left[:-1] + right
    return [start, end]


def cap_polygon_points(
    points: Sequence[Tuple[float, float]], max_points: int
) -> List[Tuple[float, float]]:
    """多边形点过多时均匀抽样到 max_points 个点（保留首尾）。"""
    if len(points) <= max_points:
        return list(points)
    idx = np.linspace(0, len(points) - 1, max_points).round().astype(np.int64)
    idx = np.unique(idx)
    return [points[i] for i in idx]


def mask_to_polygons(
    binary_mask: np.ndarray,
    tolerance: float = 1.0,
    max_points: int = 64,
) -> List[List[Tuple[float, float]]]:
    """从 binary mask 提取每个连通域的外轮廓并做轻量简化。

    返回像素坐标多边形列表 [[(x, y), ...], ...]（每个连通域一个）。
    """
    polys, _areas = mask_to_polygons_with_areas(binary_mask, tolerance, max_points)
    return polys


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
    """把 poly 桥接进 ring：最近边界点对 (a, b) 处，各沿原方向走满一圈后拼接。

    结果闭合 ring：往返桥为 (a→b) 与 (b→a) 同段反向（零宽、共线重叠）。
    由最近点性质可证桥接边 (a, b) 不穿过任一分量内部（否则其内部点会比
    a 更接近 b，矛盾），故不产生 proper crossing。
    """
    ia, ib = _nearest_point_pair(ring, poly)
    a, b = ring[ia], poly[ib]
    ring_rot = ring[ia:] + ring[:ia] + [a]   # [a, ..., a]
    poly_rot = poly[ib:] + poly[:ib] + [b]   # [b, ..., b]
    return ring_rot + poly_rot               # 闭合边 (b, a) = 返回桥


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


# ── YOLO 归一化（单 ring）────────────────────────────────────────────

def ring_to_yolo_flat(
    ring: Sequence[Tuple[float, float]],
    width: int,
    height: int,
    precision: int = 6,
) -> List[float]:
    """单 ring → YOLO seg 归一化 flat 点列表 [x1, y1, x2, y2, ...]。

    坐标 = 像素 / 尺寸，全部 ∈ [0, 1]。
    """
    pts: List[float] = []
    for x, y in ring:
        pts.append(round(x / width, precision))
        pts.append(round(y / height, precision))
    return pts


def polygon_to_yolo_flat(
    polygons: Sequence[Sequence[Tuple[float, float]]],
    width: int,
    height: int,
    precision: int = 6,
) -> List[float]:
    """把像素坐标多边形列表转为 YOLO seg 归一化 flat 点列表。

    多连通域先桥接合并为单 ring（YOLO 单多边形约束）；单连通域输出与旧
    实现一致。
    """
    ring, _meta = merge_to_single_ring(polygons)
    return ring_to_yolo_flat(ring, width, height, precision)


def yolo_class_id(entity_id: str) -> int:
    """YOLO 类别编号：球员 0，球 1。"""
    return YOLO_CLASS_BALL if entity_id == "BALL" else YOLO_CLASS_PLAYER


def det_xyxy_to_yolo_norm(
    xyxy: Sequence[float], width: int, height: int, precision: int = 6
) -> Tuple[float, float, float, float]:
    """bbox (xmin, ymin, xmax, ymax) → YOLO detect 归一化 (cx, cy, w, h)。"""
    xmin, ymin, xmax, ymax = xyxy
    cx = (xmin + xmax) / 2.0 / width
    cy = (ymin + ymax) / 2.0 / height
    w = (xmax - xmin) / width
    h = (ymax - ymin) / height
    return (
        round(cx, precision),
        round(cy, precision),
        round(w, precision),
        round(h, precision),
    )


# ── mask 彩色可视化（仅查看，不改变数据契约）──────────────────────────────

# 背景深色 + 11 个鲜艳实例色（固定调色板，mask_id 1..11 → 对应颜色）
_MASK_BG_COLOR = (24, 24, 34)
_MASK_COLOR_PALETTE = [
    (230, 84, 84),     # 1  L0 红
    (76, 178, 240),    # 2  L1 蓝
    (76, 216, 118),    # 3  L2 绿
    (240, 202, 62),    # 4  L3 黄
    (196, 116, 240),   # 5  L4 紫
    (242, 140, 40),    # 6  R0 橙
    (56, 216, 216),    # 7  R1 青
    (242, 120, 180),   # 8  R2 粉
    (162, 216, 78),    # 9  R3 黄绿
    (138, 140, 255),   # 10 R4 浅蓝
    (255, 92, 200),    # 11 BALL 品红
]


def mask_to_color_image(mask_arr):
    """把单通道实例 ID 数组转成 (H, W, 3) uint8 彩色可视化图。

    背景(0)=深色；mask_id 1..11 → 固定鲜艳调色板（与数据值无关，仅供肉眼查看）；
    非法 ID → 亮黄（醒目提示）。**不改写 mask 数据本身**（数据契约仍是 0/1..11）。
    """
    mask = np.asarray(mask_arr)
    h, w = mask.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    out[..., 0] = _MASK_BG_COLOR[0]
    out[..., 1] = _MASK_BG_COLOR[1]
    out[..., 2] = _MASK_BG_COLOR[2]
    for mid, rgb in enumerate(_MASK_COLOR_PALETTE, start=1):
        sel = mask == mid
        if sel.any():
            out[sel] = rgb
    illegal = (mask != 0) & ((mask < 1) | (mask > len(_MASK_COLOR_PALETTE)))
    if illegal.any():
        out[illegal] = (255, 240, 0)  # 非法 ID 亮黄
    return out


# ── mask 输出校准探针（纯函数部分）─────────────────────────────────────

def analyze_mask_dir(mask_dir, sample_frames: int = 5) -> Optional[dict]:
    """读取若干 mask PNG，统计各通道取值分布，返回校准建议。

    用于确认 MRQ 实际输出编码（灰度 / R / G / B / A 哪一通道携带 stencil 值）。
    mask_dir 不存在或没有 PNG 时返回 None。
    """
    files = sorted(glob.glob(str(Path(mask_dir) / "*.png")))[:sample_frames]
    if not files:
        return None
    from PIL import Image

    channel_vals: dict = {"gray": set(), "r": set(), "g": set(), "b": set(), "a": set()}
    for f in files:
        arr = load_mask_array(f, "gray")
        channel_vals["gray"].update(np.unique(arr).tolist())
        rgba = np.asarray(Image.open(f).convert("RGBA"), dtype=np.uint8)
        for i, name in enumerate(("r", "g", "b", "a")):
            channel_vals[name].update(np.unique(rgba[:, :, i]).tolist())
    return {
        "sample_files": len(files),
        "channel_unique_values": {k: sorted(v) for k, v in channel_vals.items()},
        "note": (
            "观察哪个通道的取值集合作密集且含 1..11（或乘以 id_scale），即实例 ID 所在通道；"
            "用该值设 instance_mask.mask_channel 与 id_scale。背景通常为 0 或 255。"
        ),
    }


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
