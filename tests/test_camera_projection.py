"""相机投影纯数学模块的测试。"""

import math

from camera_projection import (
    CameraExtrinsics,
    CameraIntrinsics,
    compute_intrinsics_from_focal_length,
    compute_intrinsics_from_fov,
    compute_intrinsics_from_vertical_fov,
    focal_length_to_fov_deg,
    fov_deg_to_focal_length,
    project_box_corners_to_image_xyxy,
    project_world_to_image,
    world_bbox_corners,
)


def _camera_at_origin():
    """位于原点、朝 +X 看的相机（正方形图像 1080x1080）。"""
    intr = CameraIntrinsics(
        width=1080, height=1080, fx=1000.0, fy=1000.0, cx=540.0, cy=540.0
    )
    extr = CameraExtrinsics(
        location=(0.0, 0.0, 0.0),
        forward=(1.0, 0.0, 0.0),
        right=(0.0, 1.0, 0.0),
        up=(0.0, 0.0, 1.0),
    )
    return intr, extr


class TestIntrinsics:
    def test_from_fov_90_horizontal(self):
        intr = compute_intrinsics_from_fov(90.0, 1920, 1080)
        assert intr.width == 1920
        assert intr.height == 1080
        # fx = (1920/2)/tan(45°) = 960；fy 由垂直 FOV 推导，像素应为正方形
        assert math.isclose(intr.fx, 960.0, rel_tol=1e-6)
        assert math.isclose(intr.fy, 960.0, rel_tol=1e-6)
        assert math.isclose(intr.cx, 959.5, rel_tol=1e-9)

    def test_from_focal_length(self):
        intr = compute_intrinsics_from_focal_length(50.0, 36.0, 24.0, 1920, 1080)
        assert math.isclose(intr.fx, 50.0 * 1920 / 36.0, rel_tol=1e-9)
        assert math.isclose(intr.fy, 50.0 * 1080 / 24.0, rel_tol=1e-9)
        assert math.isclose(intr.cx, 959.5, rel_tol=1e-9)

    def test_from_vertical_fov(self):
        intr = compute_intrinsics_from_vertical_fov(60.0, 1920, 1080)
        fy = (1080 / 2) / math.tan(math.radians(30.0))
        assert math.isclose(intr.fy, fy, rel_tol=1e-6)
        # 正方形图像时垂直 FOV 等于水平 FOV
        square = compute_intrinsics_from_vertical_fov(60.0, 1080, 1080)
        assert math.isclose(square.fx, square.fy, rel_tol=1e-9)

    def test_fov_focal_roundtrip(self):
        fov = focal_length_to_fov_deg(50.0, 36.0)
        focal = fov_deg_to_focal_length(fov, 36.0)
        assert math.isclose(focal, 50.0, rel_tol=1e-9)


class TestProjection:
    def test_center_point(self):
        intr, extr = _camera_at_origin()
        u, v = project_world_to_image((10.0, 0.0, 0.0), intr, extr)
        assert math.isclose(u, 540.0, rel_tol=1e-9)
        assert math.isclose(v, 540.0, rel_tol=1e-9)

    def test_right_point(self):
        intr, extr = _camera_at_origin()
        u, v = project_world_to_image((10.0, 1.0, 0.0), intr, extr)
        assert math.isclose(u, 640.0, rel_tol=1e-9)  # 540 + 1000 * (1/10)
        assert math.isclose(v, 540.0, rel_tol=1e-9)

    def test_up_point(self):
        intr, extr = _camera_at_origin()
        u, v = project_world_to_image((10.0, 0.0, 1.0), intr, extr)
        assert math.isclose(u, 540.0, rel_tol=1e-9)
        assert math.isclose(v, 440.0, rel_tol=1e-9)  # 540 - 1000 * (1/10)

    def test_down_point(self):
        intr, extr = _camera_at_origin()
        u, v = project_world_to_image((10.0, 0.0, -1.0), intr, extr)
        assert math.isclose(u, 540.0, rel_tol=1e-9)
        assert math.isclose(v, 640.0, rel_tol=1e-9)  # 540 - 1000 * (-1/10)

    def test_behind_camera_is_none(self):
        intr, extr = _camera_at_origin()
        assert project_world_to_image((-5.0, 0.0, 0.0), intr, extr) is None
        assert project_world_to_image((0.0, 0.0, 0.0), intr, extr) is None

    def test_non_finite_is_none(self):
        intr, extr = _camera_at_origin()
        assert project_world_to_image((float("nan"), 0.0, 0.0), intr, extr) is None


class TestBoxProjection:
    def test_box_in_front(self):
        intr, extr = _camera_at_origin()
        # 中心 (10,0,0)，半边长 1 → 每轴宽 2m，距相机 10m
        corners = world_bbox_corners((10.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        xyxy = project_box_corners_to_image_xyxy(corners, intr, extr)
        assert xyxy is not None
        xmin, ymin, xmax, ymax = xyxy
        # box 有深度（x 从 9 到 11），近处角点投影更大 → 边界由 x=9 的角点决定。
        # u = 540 ± 1000*(1/9) → 428.89 / 651.11（这正体现了透视下远近尺寸不同）
        assert math.isclose(xmin, 540.0 - 1000.0 / 9.0, rel_tol=1e-6)
        assert math.isclose(xmax, 540.0 + 1000.0 / 9.0, rel_tol=1e-6)
        assert math.isclose(ymin, 540.0 - 1000.0 / 9.0, rel_tol=1e-6)
        assert math.isclose(ymax, 540.0 + 1000.0 / 9.0, rel_tol=1e-6)

    def test_box_behind_camera(self):
        intr, extr = _camera_at_origin()
        corners = world_bbox_corners((-10.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        assert project_box_corners_to_image_xyxy(corners, intr, extr) is None

    def test_box_straddling_near_plane(self):
        intr, extr = _camera_at_origin()
        # 盒子中心在 x=0，跨过近平面（x 从 -1 到 1）
        corners = world_bbox_corners((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        xyxy = project_box_corners_to_image_xyxy(corners, intr, extr)
        assert xyxy is not None
        xmin, ymin, xmax, ymax = xyxy
        # 近平面附近的点投影到左侧很远
        assert xmin < 0
        assert xmax > 540.0
