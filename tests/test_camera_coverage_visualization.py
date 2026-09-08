"""单相机 Camera Coverage debug 几何测试。"""

import pytest

from grf_ue_bridge.tools.camera_coverage_visualization import (
    camera_footprint_from_basis,
    load_c1_debug_camera,
    load_camera_set,
    render_c1_debug,
    render_camera_coverage,
    ray_plane_intersection,
)


def test_ray_plane_intersection_hits_ground_plane():
    assert ray_plane_intersection((1.0, 2.0, 10.0), (0.0, 0.0, -2.0)) == pytest.approx(
        (1.0, 2.0, 0.0)
    )


def test_ue_basis_footprint_has_four_ground_vertices():
    footprint = camera_footprint_from_basis(
        position_m=(0.0, 0.0, 10.0),
        forward=(0.8660254, 0.0, -0.5),
        right=(0.0, 1.0, 0.0),
        up=(0.5, 0.0, 0.8660254),
        horizontal_fov_deg=90.0,
        resolution=(1920, 1080),
    )

    assert footprint.forward == pytest.approx((0.8660254, 0.0, -0.5))
    assert len(footprint.ground_polygon) == 4
    assert all(point[2] == pytest.approx(0.0) for point in footprint.ground_polygon)


def test_c1_debug_uses_explicit_ue_basis_not_yaw_math(repo_root):
    camera = load_c1_debug_camera(repo_root / "configs" / "p2_6_anchor_only_100f.json")

    assert camera["camera_id"] == "C1"
    assert camera["position_m"] == pytest.approx((25.0, 12.5, 7.0))
    assert camera["forward"][0] < 0.0
    assert camera["forward"][1] < 0.0
    assert camera["position_m"][0] > 20.0
    assert camera["position_m"][1] > 10.0


def test_c1_footprint_expands_toward_court(repo_root):
    camera = load_c1_debug_camera(repo_root / "configs" / "p2_6_anchor_only_100f.json")
    footprint = camera["footprint"]

    assert footprint.forward[0] < 0.0
    assert footprint.forward[1] < 0.0
    assert min(point[0] for point in footprint.ground_polygon) < 20.0
    assert min(point[1] for point in footprint.ground_polygon) < 10.0


def test_render_c1_debug_writes_png_svg_and_text(tmp_path, repo_root):
    outputs = render_c1_debug(
        repo_root / "configs" / "p2_6_anchor_only_100f.json", tmp_path
    )

    assert outputs.png.name == "C1_debug_camera_coverage.png"
    assert outputs.svg.name == "C1_debug_camera_coverage.svg"
    assert outputs.debug.name == "C1_debug_camera_coverage.txt"
    assert outputs.png.stat().st_size > 1000
    debug = outputs.debug.read_text(encoding="utf-8")
    assert "Camera:\nC1" in debug
    assert "Forward (derived from profile rotation):" in debug
    assert "Ground footprint vertices:" in debug


def test_multi_camera_set_uses_profile_position_and_rotation_for_all_cameras(repo_root):
    cameras = load_camera_set(
        repo_root / "configs" / "p2_6_anchor_plus_one_partial_100f.json",
        "anchor_plus_one_partial",
    )

    assert list(cameras) == ["C1", "C2", "C3", "C4", "C5", "P01"]
    assert cameras["C1"]["position_m"] == pytest.approx((25.0, 12.5, 7.0))
    assert cameras["P01"]["position_m"] == pytest.approx((-10.0, -25.0, 16.0))
    assert cameras["P01"]["rotation_deg"] == pytest.approx(
        (-32.47119229084849, 90.0, 0.0)
    )


def test_full_field_semantics_are_defined_at_camera_set_level(repo_root):
    cameras = load_camera_set(
        repo_root / "configs" / "p2_6_anchor_only_100f.json", "anchor_only"
    )

    assert list(cameras) == ["C1", "C2", "C3", "C4", "C5"]
    assert all(len(camera["footprint"].ground_polygon) == 4 for camera in cameras.values())

    def covered_by_set(point):
        x, y = point
        for camera in cameras.values():
            polygon = camera["footprint"].ground_polygon
            inside = False
            for start, end in zip(polygon, polygon[1:] + polygon[:1]):
                if ((start[1] > y) != (end[1] > y)) and x < (
                    (end[0] - start[0]) * (y - start[1]) / (end[1] - start[1])
                    + start[0]
                ):
                    inside = not inside
            if inside:
                return True
        return False

    court_samples = [
        (x, y)
        for x in (-20.0, -15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0)
        for y in (-10.0, -5.0, 0.0, 5.0, 10.0)
    ]
    assert all(covered_by_set(sample) for sample in court_samples)


def test_p01_multi_camera_transform_comes_from_profile(repo_root):
    cameras = load_camera_set(
        repo_root / "configs" / "p2_6_anchor_plus_one_partial_100f.json",
        "anchor_plus_one_partial",
    )

    assert cameras["P01"]["position_m"] == pytest.approx((-10.0, -25.0, 16.0))
    assert cameras["P01"]["forward"][0] == pytest.approx(0.0, abs=1e-9)
    assert cameras["P01"]["forward"][1] > 0.0


def test_render_multi_camera_outputs_are_still_available(tmp_path, repo_root):
    outputs = render_camera_coverage(
        repo_root / "configs/p2_6_anchor_only_100f.json",
        "anchor_only",
        tmp_path,
    )

    assert outputs.png.exists()
    assert outputs.svg.exists()
    assert "C1" in outputs.svg.read_text(encoding="utf-8")
