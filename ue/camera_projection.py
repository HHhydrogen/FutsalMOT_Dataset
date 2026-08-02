"""相机投影的纯数学模块。

不依赖 unreal/numpy，可在普通 Python 环境（pytest / P1）中独立测试，
也可被 UE 侧脚本直接复用。

坐标系约定
----------
- Unreal 世界坐标：左手系，X 前、Y 右、Z 上。本模块按米处理（UE 世界默认单位
  为 cm，调用方负责把 cm 换算为米；投影对整体单位缩放不变）。
- 相机坐标系：原点在相机位置，X=相机前向（forward）、Y=相机右向（right）、
  Z=相机上向（up）。相机空间坐标：
      camera = R @ (world - location)
  其中 R 的行向量依次为 forward / right / up。
- 图像坐标系：原点在左上角，x 向右、y 向下，单位像素。
- pinhole 投影：
      u = cx + fx * (y_cam / x_cam)
      v = cy - fy * (z_cam / x_cam)
  x_cam <= 0 视为在相机后方（不可见）。
"""

from dataclasses import dataclass
import math
from typing import List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class CameraIntrinsics:
    """相机内参（像素单位）。"""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass(frozen=True)
class CameraExtrinsics:
    """相机外参：世界→相机 的旋转基向量与位置。

    camera = R @ (world - location)，其中 R 的行向量依次为
    forward（相机前向）、right（相机右向）、up（相机上向）。
    """

    location: Tuple[float, float, float]
    forward: Tuple[float, float, float]
    right: Tuple[float, float, float]
    up: Tuple[float, float, float]


def compute_intrinsics_from_fov(
    horizontal_fov_deg: float,
    width: int,
    height: int,
) -> CameraIntrinsics:
    """由水平视场角与图像尺寸计算内参。

    垂直视场角由图像宽高比推导，保证像素纵横比一致：
      fx = (width / 2) / tan(fov_h / 2)
      fy = (height / 2) / tan(fov_v / 2)
    """
    fov_h = math.radians(horizontal_fov_deg)
    aspect = width / height
    fov_v = 2 * math.atan(math.tan(fov_h / 2) / aspect)
    fx = (width / 2) / math.tan(fov_h / 2)
    fy = (height / 2) / math.tan(fov_v / 2)
    return CameraIntrinsics(width, height, fx, fy, (width - 1) / 2, (height - 1) / 2)


def compute_intrinsics_from_vertical_fov(
    vertical_fov_deg: float,
    width: int,
    height: int,
) -> CameraIntrinsics:
    """由垂直视场角与图像尺寸计算内参（CineCamera 的 current_fov 为垂直 FOV）。

      fov_h = 2 * atan(tan(fov_v / 2) * aspect)
      fy = (height / 2) / tan(fov_v / 2)
      fx = (width / 2) / tan(fov_h / 2)
    """
    fov_v = math.radians(vertical_fov_deg)
    aspect = width / height
    fov_h = 2 * math.atan(math.tan(fov_v / 2) * aspect)
    fy = (height / 2) / math.tan(fov_v / 2)
    fx = (width / 2) / math.tan(fov_h / 2)
    return CameraIntrinsics(width, height, fx, fy, (width - 1) / 2, (height - 1) / 2)


def compute_intrinsics_from_focal_length(
    focal_length_mm: float,
    sensor_width_mm: float,
    sensor_height_mm: float,
    width: int,
    height: int,
) -> CameraIntrinsics:
    """由焦距与传感器尺寸计算内参。

      fx = focal * width / sensor_width
      fy = focal * height / sensor_height
    """
    fx = focal_length_mm * width / sensor_width_mm
    fy = focal_length_mm * height / sensor_height_mm
    return CameraIntrinsics(width, height, fx, fy, (width - 1) / 2, (height - 1) / 2)


def focal_length_to_fov_deg(focal_length_mm: float, sensor_size_mm: float) -> float:
    """焦距（mm）→ 视场角（度）。sensor_size_mm 为对应轴（宽或高）的尺寸。"""
    return 2 * math.degrees(math.atan(sensor_size_mm / (2 * focal_length_mm)))


def fov_deg_to_focal_length(fov_deg: float, sensor_size_mm: float) -> float:
    """视场角（度）→ 焦距（mm）。sensor_size_mm 为对应轴的尺寸。"""
    return sensor_size_mm / (2 * math.tan(math.radians(fov_deg) / 2))


def world_to_camera(
    world_point: Sequence[float],
    extrinsics: CameraExtrinsics,
) -> Tuple[float, float, float]:
    """把世界坐标变换到相机坐标。

    返回 (x_cam, y_cam, z_cam) = (前向深度, 右向, 上向)，与相机同单位。
    """
    dx = world_point[0] - extrinsics.location[0]
    dy = world_point[1] - extrinsics.location[1]
    dz = world_point[2] - extrinsics.location[2]
    x_cam = (
        extrinsics.forward[0] * dx
        + extrinsics.forward[1] * dy
        + extrinsics.forward[2] * dz
    )
    y_cam = (
        extrinsics.right[0] * dx
        + extrinsics.right[1] * dy
        + extrinsics.right[2] * dz
    )
    z_cam = (
        extrinsics.up[0] * dx + extrinsics.up[1] * dy + extrinsics.up[2] * dz
    )
    return (x_cam, y_cam, z_cam)


def _project_camera_point(
    cam_point: Sequence[float],
    intrinsics: CameraIntrinsics,
) -> Optional[Tuple[float, float]]:
    """把相机空间点投影到图像坐标（像素，浮点）。

    返回 None 表示在相机后方（x_cam <= 0）或产生非有限值。
    """
    x_cam, y_cam, z_cam = cam_point
    if x_cam <= 0.0:
        return None
    u = intrinsics.cx + intrinsics.fx * (y_cam / x_cam)
    v = intrinsics.cy - intrinsics.fy * (z_cam / x_cam)
    if not (math.isfinite(u) and math.isfinite(v)):
        return None
    return (u, v)


def project_world_to_image(
    world_point: Sequence[float],
    intrinsics: CameraIntrinsics,
    extrinsics: CameraExtrinsics,
) -> Optional[Tuple[float, float]]:
    """把世界点投影到图像坐标（像素，浮点）。

    返回 None 表示在相机后方（x_cam <= 0）或产生非有限值。
    """
    cam = world_to_camera(world_point, extrinsics)
    return _project_camera_point(cam, intrinsics)


def world_bbox_corners(
    origin: Sequence[float],
    extent: Sequence[float],
) -> List[Tuple[float, float, float]]:
    """从世界 AABB 的中心与半边长生成 8 个角点。"""
    ox, oy, oz = origin
    ex, ey, ez = extent
    corners = []
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                corners.append((ox + sx * ex, oy + sy * ey, oz + sz * ez))
    return corners


# 立方体 8 个角点（与 world_bbox_corners 的索引一致）之间的 12 条边
_BOX_EDGES: List[Tuple[int, int]] = [
    (0, 4), (1, 5), (2, 6), (3, 7),  # 沿 X
    (0, 2), (1, 3), (4, 6), (5, 7),  # 沿 Y
    (0, 1), (2, 3), (4, 5), (6, 7),  # 沿 Z
]


def _clip_camera_segment_to_near(
    a: Sequence[float],
    b: Sequence[float],
    near: float,
) -> List[Tuple[float, float, float]]:
    """把相机空间线段按近平面（x_cam >= near）裁剪。

    返回可见端点（0、1 或 2 个点）。跨界时按比例求近平面交点。
    """
    a_in = a[0] >= near
    b_in = b[0] >= near
    if a_in and b_in:
        return [(a[0], a[1], a[2]), (b[0], b[1], b[2])]
    if not a_in and not b_in:
        return []
    # 一端在内一端在外，求线段与 x=near 平面的交点
    t = (near - a[0]) / (b[0] - a[0])
    inter = (near, a[1] + t * (b[1] - a[1]), a[2] + t * (b[2] - a[2]))
    if a_in:
        return [(a[0], a[1], a[2]), inter]
    return [inter, (b[0], b[1], b[2])]


def project_box_corners_to_image_xyxy(
    corners: Sequence[Sequence[float]],
    intrinsics: CameraIntrinsics,
    extrinsics: CameraExtrinsics,
    near: float = 0.1,
) -> Optional[Tuple[float, float, float, float]]:
    """把 8 个世界角点投影为 2D bbox。

    对 12 条边做近平面裁剪，再把可见部分投影到像素平面，取所有投影点的
    min/max 得到 (xmin, ymin, xmax, ymax)。所有角点都在近平面之后返回 None。
    """
    cam = [world_to_camera(c, extrinsics) for c in corners]
    if all(c[0] <= near for c in cam):
        return None

    projected: List[Tuple[float, float]] = []
    for i, j in _BOX_EDGES:
        seg = _clip_camera_segment_to_near(cam[i], cam[j], near)
        for pt in seg:
            uv = _project_camera_point(pt, intrinsics)
            if uv is not None:
                projected.append(uv)

    if not projected:
        return None
    us = [p[0] for p in projected]
    vs = [p[1] for p in projected]
    return (min(us), min(vs), max(us), max(vs))
