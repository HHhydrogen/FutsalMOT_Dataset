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
from render_preset import (
    resolve_output_resolution,
    validate_render_vs_calibration,
)
import json

import pytest
from PIL import Image


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


class TestC5ResolutionAutomation:
    """C5.1：camera intrinsics 必须随真实输出分辨率自动生成。

    根因：calibration 曾按 1280×720 生成，而 MRQ 实际渲染 1920×1080，
    导致 Pose 整体错位。本测试验证 fx/fy/cx/cy 随分辨率线性缩放，
    且 1280×720 / 1920×1080 / 2560×1440 均正确。
    """

    FOCAL = 15.0
    SW, SH = 23.76, 13.365

    def test_intrinsics_scaling_across_resolutions(self):
        for w, h in [(1280, 720), (1920, 1080), (2560, 1440)]:
            intr = compute_intrinsics_from_focal_length(self.FOCAL, self.SW, self.SH, w, h)
            assert math.isclose(intr.fx, self.FOCAL * w / self.SW, rel_tol=1e-9)
            assert math.isclose(intr.fy, self.FOCAL * h / self.SH, rel_tol=1e-9)
            assert math.isclose(intr.cx, (w - 1) / 2, rel_tol=1e-9)
            assert math.isclose(intr.cy, (h - 1) / 2, rel_tol=1e-9)

    def test_16_9_scaling_ratio(self):
        i1280 = compute_intrinsics_from_focal_length(self.FOCAL, self.SW, self.SH, 1280, 720)
        i1920 = compute_intrinsics_from_focal_length(self.FOCAL, self.SW, self.SH, 1920, 1080)
        # 1280 → 1920 = 1.5×
        assert math.isclose(i1920.fx / i1280.fx, 1.5, rel_tol=1e-9)
        assert math.isclose(i1920.fy / i1280.fy, 1.5, rel_tol=1e-9)

    def test_c4_1920_1080_expected_values(self):
        intr = compute_intrinsics_from_focal_length(self.FOCAL, self.SW, self.SH, 1920, 1080)
        assert math.isclose(intr.fx, 1212.1212, rel_tol=1e-3)
        assert math.isclose(intr.fy, 1212.1212, rel_tol=1e-3)
        assert intr.cx == 959.5
        assert intr.cy == 539.5

    def test_resolve_output_resolution_from_render(self):
        cfg = {"render_rgb": {"output_resolution_x": 1920, "output_resolution_y": 1080}}
        assert resolve_output_resolution(cfg) == (1920, 1080)

    def test_resolve_output_resolution_legacy_fallback(self):
        cfg = {"image_width": 1280, "image_height": 720}
        assert resolve_output_resolution(cfg) == (1280, 720)

    def test_resolve_output_resolution_mismatch_rejected(self):
        # 负测试：task render = 1920×1080，但 annotation image = 1280×720 → 拒绝
        cfg = {
            "image_width": 1280,
            "image_height": 720,
            "render_rgb": {"output_resolution_x": 1920, "output_resolution_y": 1080},
        }
        with pytest.raises(ValueError):
            resolve_output_resolution(cfg)

    def test_resolve_output_resolution_missing_fails_fast(self):
        with pytest.raises(ValueError):
            resolve_output_resolution({})
        with pytest.raises(ValueError):
            resolve_output_resolution(None)

    def test_validate_render_vs_calibration_mismatch(self, tmp_path):
        render = tmp_path / "render"
        render.mkdir()
        _write_fake_png(render / "0000.png", 1920, 1080)
        cam = tmp_path / "camera.json"
        cam.write_text(json.dumps({"image_width": 1280, "image_height": 720, "intrinsics": {}}), encoding="utf-8")
        with pytest.raises(RuntimeError):
            validate_render_vs_calibration(render, cam)

    def test_validate_render_vs_calibration_ok(self, tmp_path):
        render = tmp_path / "render"
        render.mkdir()
        _write_fake_png(render / "0000.png", 1920, 1080)
        cam = tmp_path / "camera.json"
        cam.write_text(
            json.dumps({"image_width": 1920, "image_height": 1080,
                        "intrinsics": {"width": 1920, "height": 1080}}),
            encoding="utf-8",
        )
        validate_render_vs_calibration(render, cam)  # 不抛错

    def test_validate_render_vs_calibration_jpeg_only(self, tmp_path):
        render = tmp_path / "render"
        render.mkdir()
        Image.new("RGB", (1920, 1080), "black").save(render / "0000.jpg")
        cam = tmp_path / "camera.json"
        cam.write_text(json.dumps({"image_width": 1920, "image_height": 1080}), encoding="utf-8")
        validate_render_vs_calibration(render, cam)


def _write_fake_png(path, width, height):
    """写仅含 PNG 签名 + IHDR 头的伪 PNG（validate 只读前 24 字节）。"""
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write((13).to_bytes(4, "big"))
        f.write(b"IHDR")
        f.write((width).to_bytes(4, "big"))
        f.write((height).to_bytes(4, "big"))
