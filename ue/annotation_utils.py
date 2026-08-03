"""标注相关的纯数据工具：bbox 转换、图像裁剪、track_id 映射。

不依赖 unreal/numpy，可在 pytest / P1 中独立测试，也可被 UE 侧脚本复用。
"""

import math
from typing import Dict, Optional, Sequence, Tuple

# 实体 ID → track_id 的确定性映射（见 CLAUDE.md / README 的文档说明）。
#   L0..L4 -> 1..5
#   R0..R4 -> 6..10
#   BALL   -> 100（独立高位 ID，与球员不冲突）
BALL_TRACK_ID = 100

# 实体 ID → Instance-ID mask_id 的确定性映射（与 track_id 同构，一个 episode 内恒定）。
#   L0..L4 -> 1..5
#   R0..R4 -> 6..10
#   BALL   -> 11
# mask 像素值 == mask_id（背景 = 0）。UE 侧用它给 actor 打 Custom Depth Stencil，
# P1 侧用它解码 Instance-ID Mask。本模块保持纯 Python（无 numpy），UE Python 可 import。
BALL_MASK_ID = 11

# 图像坐标系的边界常量（用于裁剪判断）
IMAGE_LEFT = 0.0
IMAGE_TOP = 0.0

# bbox_source 取值：区分「可见像素 GT」与「几何投影 GT」。
#   "geometry"      —— 几何投影 bbox（UE 导出初始标注；无 mask 数据时的 legacy fallback）
#   "instance_mask" —— primary GT：bbox 由 Instance-ID Mask 可见像素 min/max 计算
#   "not_visible"   —— 有有效 mask 帧但实体可见像素为 0（完全遮挡/离屏），可见 GT 为 null，
#                      几何 bbox 只保留在 geometry_bbox_*，绝不回填到 bbox_*。
BBOX_SOURCE_GEOMETRY = "geometry"
BBOX_SOURCE_INSTANCE_MASK = "instance_mask"
BBOX_SOURCE_NOT_VISIBLE = "not_visible"


def entity_id_to_track_id(entity_id: str) -> int:
    """实体 ID 到稳定 track_id 的确定性映射。

    一个 episode 内，同一个 entity_id 永远映射到同一个 track_id。
    """
    if entity_id == "BALL":
        return BALL_TRACK_ID
    prefix = entity_id[0]
    idx = int(entity_id[1:])
    if prefix == "L":
        return idx + 1
    if prefix == "R":
        return idx + 6
    raise ValueError(f"未知实体 ID: {entity_id!r}")


def entity_id_to_mask_id(entity_id: str) -> int:
    """实体 ID 到 Instance-ID mask_id 的确定性映射。

    与 track_id 同构：L0..L4→1..5、R0..R4→6..10、BALL→11。
    """
    if entity_id == "BALL":
        return BALL_MASK_ID
    prefix = entity_id[0]
    idx = int(entity_id[1:])
    if prefix == "L":
        return idx + 1
    if prefix == "R":
        return idx + 6
    raise ValueError(f"未知实体 ID: {entity_id!r}")


def mask_id_to_entity_id(mask_id: int) -> str:
    """mask_id 到实体 ID 的确定性逆映射。非法 mask_id 抛 ValueError。"""
    mask_id = int(mask_id)
    if mask_id == BALL_MASK_ID:
        return "BALL"
    if 1 <= mask_id <= 5:
        return f"L{mask_id - 1}"
    if 6 <= mask_id <= 10:
        return f"R{mask_id - 6}"
    raise ValueError(f"未知 mask_id: {mask_id!r}")


def valid_mask_ids() -> range:
    """合法的实体 mask_id 集合（1..11）。"""
    return range(1, BALL_MASK_ID + 1)


def entity_class(entity_id: str) -> str:
    """实体类别：球员为 'player'，球为 'ball'。"""
    return "ball" if entity_id == "BALL" else "player"


def entity_team(entity_id: str) -> Optional[str]:
    """实体队伍：L->left，R->right，球为 None。"""
    if entity_id == "BALL":
        return None
    return "left" if entity_id[0] == "L" else "right"


def xyxy_to_xywh(xyxy: Sequence[float]) -> Tuple[float, float, float, float]:
    """(xmin, ymin, xmax, ymax) -> (x, y, w, h)。"""
    xmin, ymin, xmax, ymax = xyxy
    return (xmin, ymin, xmax - xmin, ymax - ymin)


def xywh_to_xyxy(xywh: Sequence[float]) -> Tuple[float, float, float, float]:
    """(x, y, w, h) -> (xmin, ymin, xmax, ymax)。"""
    x, y, w, h = xywh
    return (x, y, x + w, y + h)


def clip_bbox_to_image(
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    width: int,
    height: int,
) -> Tuple[float, float, float, float]:
    """把 bbox 裁剪到图像边界，返回裁剪后的 (xmin, ymin, xmax, ymax)。"""
    cxmin = max(xmin, IMAGE_LEFT)
    cymin = max(ymin, IMAGE_TOP)
    cxmax = min(xmax, float(width))
    cymax = min(ymax, float(height))
    return (cxmin, cymin, cxmax, cymax)


def analyze_bbox(
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    width: int,
    height: int,
) -> Dict:
    """分析 bbox 与图像矩形的关系。

    返回 dict：
      in_frame    : bbox 与图像矩形是否有非零面积交集
      truncated   : in_frame 且 raw bbox 有部分超出图像边界
      clipped_xyxy: 裁剪后的合法 bbox（仅 in_frame 时有效）
      clipped_xywh: 裁剪后的合法 bbox 的 (x, y, w, h)
    """
    cxmin, cymin, cxmax, cymax = clip_bbox_to_image(xmin, ymin, xmax, ymax, width, height)
    inter_w = cxmax - cxmin
    inter_h = cymax - cymin
    in_frame = inter_w > 0 and inter_h > 0
    truncated = in_frame and (
        xmin < IMAGE_LEFT or ymin < IMAGE_TOP or xmax > width or ymax > height
    )
    return {
        "in_frame": in_frame,
        "truncated": truncated,
        "clipped_xyxy": (cxmin, cymin, cxmax, cymax),
        "clipped_xywh": (cxmin, cymin, cxmax - cxmin, cymax - cymin),
    }


def bbox_area(xyxy: Sequence[float]) -> float:
    """计算 bbox 面积（xyxy）。若宽高非正返回 0。"""
    xmin, ymin, xmax, ymax = xyxy
    w = max(0.0, xmax - xmin)
    h = max(0.0, ymax - ymin)
    return w * h
