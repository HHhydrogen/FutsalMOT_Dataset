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
    polys: List[List[Tuple[float, float]]] = []
    for comp in connected_components(binary_mask):
        contour = trace_outer_contour(comp)
        if len(contour) < 3:
            continue
        simp = rdp_simplify(contour, tolerance)
        simp = cap_polygon_points(simp, max_points)
        if len(simp) < 3:
            # 简化后不足三角形：用该连通域 bbox 四角兜底
            bb = mask_to_bbox(comp)
            if bb is not None:
                xmin, ymin, xmax, ymax = bb
                simp = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
        if len(simp) >= 3:
            polys.append(simp)
    return polys


def polygon_to_yolo_flat(
    polygons: Sequence[Sequence[Tuple[float, float]]],
    width: int,
    height: int,
    precision: int = 6,
) -> List[float]:
    """把像素坐标多边形列表转为 YOLO seg 归一化 flat 点列表 [x1, y1, x2, y2, ...]。

    多连通域合并为同一条 YOLO 行（一个实体一行，坐标全部 ∈ [0, 1]）。
    """
    pts: List[float] = []
    for poly in polygons:
        for x, y in poly:
            pts.append(round(x / width, precision))
            pts.append(round(y / height, precision))
    return pts


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
