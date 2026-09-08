"""基于 Camera Profile 的 C1/多相机地面覆盖可视化。

本模块只用于分析和展示，不参与 task、UE runtime 或数据生成流程。
默认从 profile 的 UE rotation 生成 basis，不依赖 Unreal。
`full_field` 的语义属于选定 camera set 的联合覆盖目标，不表示单个
camera footprint 必须独立覆盖整个球场。
"""

from dataclasses import dataclass
import argparse
import json
import math
from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont

import sys

_REPO_ROOT = Path(__file__).resolve().parents[3]
_UE_DIR = _REPO_ROOT / "ue"
if str(_UE_DIR) not in sys.path:
    sys.path.insert(0, str(_UE_DIR))

from camera_projection import focal_length_to_fov_deg  # noqa: E402


Point3 = Tuple[float, float, float]
Point2 = Tuple[float, float]


CAMERA_ID = "C1"
FIELD_LENGTH_M = 40.0
FIELD_WIDTH_M = 20.0
DISPLAY_BOUNDS_M = (-45.0, 45.0, -35.0, 35.0)
SENSOR_WIDTH_MM = 23.76


@dataclass(frozen=True)
class CameraFootprint:
    origin: Point3
    ground_polygon: Tuple[Point3, ...]
    forward: Point3


@dataclass(frozen=True)
class RenderOutputs:
    png: Path
    svg: Path
    debug: Optional[Path] = None


def ray_plane_intersection(
    origin: Sequence[float], direction: Sequence[float], plane_z: float = 0.0
) -> Point3:
    """计算射线与水平地面的交点。"""
    dz = float(direction[2])
    if abs(dz) < 1e-12:
        raise ValueError("ray is parallel to the ground plane")
    scale = (float(plane_z) - float(origin[2])) / dz
    if scale <= 0.0:
        raise ValueError("ray does not intersect the ground plane in front of camera")
    return tuple(float(origin[i]) + scale * float(direction[i]) for i in range(3))  # type: ignore[return-value]


def camera_footprint_from_basis(
    position_m: Sequence[float],
    forward: Sequence[float],
    right: Sequence[float],
    up: Sequence[float],
    horizontal_fov_deg: float,
    resolution: Sequence[int],
    ground_height: float = 0.0,
) -> CameraFootprint:
    """使用真实 UE basis 计算四角视锥与地面的交集 polygon。"""
    width, height = (float(value) for value in resolution)
    if width <= 0.0 or height <= 0.0:
        raise ValueError("camera resolution must be positive")
    half_horizontal = math.radians(float(horizontal_fov_deg)) / 2.0
    half_vertical = math.atan(math.tan(half_horizontal) * height / width)
    horizontal_scale = math.tan(half_horizontal)
    vertical_scale = math.tan(half_vertical)
    corner_signs = ((1.0, -1.0), (1.0, 1.0), (-1.0, 1.0), (-1.0, -1.0))
    ground_polygon = []
    for vertical_sign, horizontal_sign in corner_signs:
        ray = tuple(
            float(forward[i])
            + horizontal_sign * horizontal_scale * float(right[i])
            + vertical_sign * vertical_scale * float(up[i])
            for i in range(3)
        )
        ground_polygon.append(ray_plane_intersection(position_m, ray, ground_height))
    return CameraFootprint(
        origin=tuple(float(value) for value in position_m),  # type: ignore[arg-type]
        ground_polygon=tuple(ground_polygon),
        forward=tuple(float(value) for value in forward),  # type: ignore[arg-type]
    )


def _basis_from_rotation(rotation_deg: Sequence[float]) -> Tuple[Point3, Point3, Point3]:
    """按 profile 的 UE Rotator 顺序生成可复现的 transform basis。"""
    pitch, yaw, roll = (math.radians(float(value)) for value in rotation_deg)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    cr, sr = math.cos(roll), math.sin(roll)
    forward = (cp * cy, cp * sy, sp)
    right = (sr * sp * cy - cr * sy, sr * sp * sy + cr * cy, -sr * cp)
    up = (cr * sp * cy + sr * sy, cr * sp * sy - sr * cy, cr * cp)
    return forward, right, up


def load_c1_debug_camera(config_path: Path):
    """读取 C1 profile，作为默认 visualization 数据源。"""
    config = json.loads(config_path.read_text(encoding="utf-8"))
    profile = config["simulation"]["camera"]["profiles"][CAMERA_ID]
    focal = float(profile["lens"]["focal_length_mm"])
    resolution = tuple(profile.get("resolution", (1920, 1080)))
    sensor_width = float(profile.get("sensor_width_mm", SENSOR_WIDTH_MM))
    fov = focal_length_to_fov_deg(focal, sensor_width)
    basis = _basis_from_rotation(profile["rotation_deg"])
    footprint = camera_footprint_from_basis(
        profile["position_m"],
        basis[0],
        basis[1],
        basis[2],
        fov,
        resolution,
    )
    return {
        "camera_id": CAMERA_ID,
        "position_m": tuple(float(value) for value in profile["position_m"]),
        "rotation_deg": tuple(float(value) for value in profile["rotation_deg"]),
        "forward": basis[0],
        "right": basis[1],
        "up": basis[2],
        "focal_length_mm": focal,
        "sensor_width_mm": sensor_width,
        "resolution": resolution,
        "horizontal_fov_deg": fov,
        "footprint": footprint,
    }


def load_camera_set(config_path: Path, camera_set: str):
    """读取 profile 参数，并使用 profile transform 计算各相机。"""
    config = json.loads(config_path.read_text(encoding="utf-8"))
    profiles = config["simulation"]["camera"]["profiles"]
    camera_ids = ("C1", "C2", "C3", "C4", "C5")
    if camera_set == "anchor_plus_one_partial":
        camera_ids += ("P01",)
    if camera_set != "anchor_only" and camera_set != "anchor_plus_one_partial":
        raise ValueError("unknown camera set: %s" % camera_set)
    cameras = {}
    for camera_id in camera_ids:
        profile = profiles[camera_id]
        position = tuple(float(value) for value in profile["position_m"])
        basis = _basis_from_rotation(profile["rotation_deg"])
        focal = float(profile["lens"]["focal_length_mm"])
        resolution = tuple(profile.get("resolution", (1920, 1080)))
        fov = focal_length_to_fov_deg(focal, SENSOR_WIDTH_MM)
        cameras[camera_id] = {
            "camera_id": camera_id,
            "position_m": position,
            "rotation_deg": tuple(profile["rotation_deg"]),
            "forward": basis[0],
            "right": basis[1],
            "up": basis[2],
            "focal_length_mm": focal,
            "resolution": resolution,
            "horizontal_fov_deg": fov,
            "footprint": camera_footprint_from_basis(
                position, basis[0], basis[1], basis[2], fov, resolution
            ),
        }
    return cameras


def render_c1_debug(config_path: Path, output_dir: Path) -> RenderOutputs:
    """生成 C1 单相机 PNG、SVG 和文本 debug 输出。"""
    camera = load_c1_debug_camera(config_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    png = output_dir / "C1_debug_camera_coverage.png"
    svg = output_dir / "C1_debug_camera_coverage.svg"
    debug = output_dir / "C1_debug_camera_coverage.txt"
    _render_png(png, camera)
    _render_svg(svg, camera)
    _write_debug(debug, camera)
    return RenderOutputs(png=png, svg=svg, debug=debug)


def render_camera_coverage(config_path: Path, camera_set: str, output_dir: Path) -> RenderOutputs:
    """生成多相机展示，数据源固定为 camera profile。"""
    cameras = load_camera_set(config_path, camera_set)
    output_dir.mkdir(parents=True, exist_ok=True)
    png = output_dir / (camera_set + "_camera_coverage.png")
    svg = output_dir / (camera_set + "_camera_coverage.svg")
    _render_multi_png(png, cameras)
    _render_multi_svg(svg, cameras, camera_set)
    return RenderOutputs(png=png, svg=svg)


def _projector(width: int, height: int):
    min_x, max_x, min_y, max_y = DISPLAY_BOUNDS_M
    margin = 120.0
    scale = min((width - 2 * margin) / (max_x - min_x), (height - 2 * margin) / (max_y - min_y))

    def project(point: Sequence[float]) -> Point2:
        return (
            margin + (float(point[0]) - min_x) * scale,
            height - margin - (float(point[1]) - min_y) * scale,
        )

    return project


def _font(size: int):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _render_png(path: Path, camera: Mapping) -> None:
    width, height = 1800, 1200
    image = Image.new("RGB", (width, height), "#f5f7fa")
    draw = ImageDraw.Draw(image, "RGBA")
    project = _projector(width, height)
    _draw_court_background(draw, project)
    polygon = [project(point) for point in camera["footprint"].ground_polygon]
    draw.polygon(polygon, fill=(40, 108, 180, 48), outline="#286cae")
    _draw_court_lines(draw, project)
    ox, oy = project(camera["position_m"])
    target = project((camera["position_m"][0] + camera["forward"][0] * 5.0, camera["position_m"][1] + camera["forward"][1] * 5.0))
    draw.line((ox, oy, target[0], target[1]), fill="#286cae", width=5)
    draw.ellipse((ox - 10, oy - 10, ox + 10, oy + 10), fill="#286cae")
    draw.text((ox + 16, oy - 22), "C1", fill="#17212b", font=_font(28))
    draw.text((110, 42), "C1 Camera Transform to Ground Coverage", fill="#17212b", font=_font(36))
    draw.text((110, 86), "Camera profile transform | z = 0 ground plane | units: m", fill="#627786", font=_font(21))
    _draw_legend(draw, width, height)
    image.save(path, "PNG", optimize=True)


def _draw_court_background(draw, project) -> None:
    draw.rectangle((project((-20.0, 10.0)), project((20.0, -10.0))), fill="#e8f1e8")


def _draw_court_lines(draw, project) -> None:
    p = project
    draw.rectangle((p((-20.0, 10.0)), p((20.0, -10.0))), outline="#304657", width=5)
    draw.line((p((0.0, 10.0)), p((0.0, -10.0))), fill="#627786", width=3)
    center = p((0.0, 0.0))
    radius = abs(p((3.0, 0.0))[0] - center[0])
    draw.ellipse((center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius), outline="#627786", width=3)
    for side in (-1, 1):
        goal_x = side * 20.0
        penalty_x = goal_x - side * 6.0
        for y_sign in (-1.0, 1.0):
            points = []
            center_y = y_sign * 1.5
            start = math.pi if side < 0 else 0.0
            end = start + (-math.pi / 2.0 if y_sign < 0 else math.pi / 2.0)
            for step in range(17):
                angle = start + (end - start) * step / 16.0
                points.append(p((goal_x - side * 6.0 * math.cos(angle), center_y + 6.0 * math.sin(angle))))
            draw.line(points, fill="#627786", width=3, joint="curve")
        draw.line((p((penalty_x, -7.5)), p((penalty_x, 7.5))), fill="#627786", width=3)
        draw.ellipse((p((goal_x - side * 6.0, 0.0))[0] - 4, p((goal_x - side * 6.0, 0.0))[1] - 4, p((goal_x - side * 6.0, 0.0))[0] + 4, p((goal_x - side * 6.0, 0.0))[1] + 4), fill="#627786")
        draw.ellipse((p((goal_x - side * 10.0, 0.0))[0] - 4, p((goal_x - side * 10.0, 0.0))[1] - 4, p((goal_x - side * 10.0, 0.0))[0] + 4, p((goal_x - side * 10.0, 0.0))[1] + 4), outline="#627786", width=2)
        draw.line((p((goal_x, -1.5)), p((goal_x, 1.5))), fill="#627786", width=3)
        draw.line((p((goal_x, -1.5)), p((goal_x + side, -1.5))), fill="#8a9aa4", width=2)
        draw.line((p((goal_x, 1.5)), p((goal_x + side, 1.5))), fill="#8a9aa4", width=2)
        draw.line((p((goal_x + side, -1.5)), p((goal_x + side, 1.5))), fill="#8a9aa4", width=2)


def _draw_legend(draw, width: int, height: int) -> None:
    x, y = width - 390, height - 125
    draw.rectangle((x - 20, y - 22, width - 28, height - 32), fill="#ffffff", outline="#b8c3cc", width=2)
    draw.ellipse((x, y - 3, x + 18, y + 15), fill="#286cae")
    draw.text((x + 30, y - 8), "C1 ground footprint", fill="#17212b", font=_font(21))


def _render_multi_png(path: Path, cameras: Mapping[str, Mapping]) -> None:
    width, height = 1800, 1200
    image = Image.new("RGB", (width, height), "#f5f7fa")
    draw = ImageDraw.Draw(image, "RGBA")
    project = _projector(width, height)
    _draw_court_background(draw, project)
    for camera_id, camera in cameras.items():
        color = (234, 129, 57, 58) if camera_id == "P01" else (40, 108, 180, 48)
        line = "#d9782e" if camera_id == "P01" else "#286cae"
        polygon = [project(point) for point in camera["footprint"].ground_polygon]
        draw.polygon(polygon, fill=color, outline=line)
    _draw_court_lines(draw, project)
    for camera_id, camera in cameras.items():
        line = "#d9782e" if camera_id == "P01" else "#286cae"
        ox, oy = project(camera["position_m"])
        target = project((camera["position_m"][0] + camera["forward"][0] * 4.0, camera["position_m"][1] + camera["forward"][1] * 4.0))
        draw.line((ox, oy, target[0], target[1]), fill=line, width=4)
        draw.ellipse((ox - 9, oy - 9, ox + 9, oy + 9), fill=line)
        draw.text((ox + 14, oy - 18), camera_id, fill="#17212b", font=_font(25))
    draw.text((110, 42), "FutsalMOT Camera Coverage | %s" % " + ".join(cameras), fill="#17212b", font=_font(36))
    draw.text((110, 86), "UE basis vectors | ground plane z = 0 | units: m", fill="#627786", font=_font(21))
    _draw_legend_multi(draw, width, height)
    image.save(path, "PNG", optimize=True)


def _draw_legend_multi(draw, width: int, height: int) -> None:
    x, y = width - 390, height - 125
    draw.rectangle((x - 20, y - 22, width - 28, height - 32), fill="#ffffff", outline="#b8c3cc", width=2)
    draw.ellipse((x, y - 3, x + 18, y + 15), fill="#286cae")
    draw.text((x + 30, y - 8), "Anchor footprint", fill="#17212b", font=_font(21))
    draw.ellipse((x, y + 32, x + 18, y + 50), fill="#d9782e")
    draw.text((x + 30, y + 27), "Partial footprint", fill="#17212b", font=_font(21))


def _render_svg(path: Path, camera: Mapping) -> None:
    width, height = 1800, 1200
    project = _projector(width, height)

    def point(value):
        x, y = project(value)
        return "%.2f,%.2f" % (x, y)

    court_tl = project((-20.0, 10.0))
    court_br = project((20.0, -10.0))
    center_top = project((0.0, 10.0))
    center_bottom = project((0.0, -10.0))
    center = project((0.0, 0.0))
    radius = abs(project((3.0, 0.0))[0] - center[0])
    footprint = camera["footprint"].ground_polygon
    origin = project(camera["position_m"])
    target = project((camera["position_m"][0] + camera["forward"][0] * 5.0, camera["position_m"][1] + camera["forward"][1] * 5.0))
    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1800" height="1200" viewBox="0 0 1800 1200">',
        '<rect width="100%" height="100%" fill="#f5f7fa"/>',
        '<text x="110" y="62" font-family="Arial,sans-serif" font-size="36" font-weight="700" fill="#17212b">C1 Camera Transform to Ground Coverage</text>',
        '<text x="110" y="98" font-family="Arial,sans-serif" font-size="21" fill="#627786">UE basis vectors | z = 0 ground plane | units: m</text>',
        '<rect x="%.2f" y="%.2f" width="%.2f" height="%.2f" fill="#e8f1e8"/>' % (court_tl[0], court_tl[1], court_br[0] - court_tl[0], court_br[1] - court_tl[1]),
        '<polygon points="%s" fill="#286cae" fill-opacity="0.20" stroke="#286cae" stroke-width="3"/>' % " ".join(point(vertex) for vertex in footprint),
        '<line x1="%.2f" y1="%.2f" x2="%.2f" y2="%.2f" stroke="#627786" stroke-width="3"/>' % (center_top[0], center_top[1], center_bottom[0], center_bottom[1]),
        '<circle cx="%.2f" cy="%.2f" r="%.2f" fill="none" stroke="#627786" stroke-width="3"/>' % (center[0], center[1], radius),
        '<rect x="%.2f" y="%.2f" width="%.2f" height="%.2f" fill="none" stroke="#304657" stroke-width="5"/>' % (court_tl[0], court_tl[1], court_br[0] - court_tl[0], court_br[1] - court_tl[1]),
        '<line x1="%.2f" y1="%.2f" x2="%.2f" y2="%.2f" stroke="#286cae" stroke-width="5"/>' % (origin[0], origin[1], target[0], target[1]),
        '<circle cx="%.2f" cy="%.2f" r="10" fill="#286cae"/><text x="%.2f" y="%.2f" font-family="Arial,sans-serif" font-size="28" fill="#17212b">C1</text>' % (origin[0], origin[1], origin[0] + 16, origin[1] - 12),
        '<rect x="1450" y="1030" width="320" height="90" fill="#fff" stroke="#b8c3cc" stroke-width="2"/><circle cx="1475" cy="1065" r="9" fill="#286cae"/><text x="1500" y="1072" font-family="Arial,sans-serif" font-size="21" fill="#17212b">C1 ground footprint</text>',
        '<text x="1670" y="170" font-family="Arial,sans-serif" font-size="22" fill="#17212b">+X</text><text x="1680" y="200" font-family="Arial,sans-serif" font-size="22" fill="#17212b">↑</text>',
        '</svg>',
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _render_multi_svg(path: Path, cameras: Mapping[str, Mapping], camera_set: str) -> None:
    width, height = 1800, 1200
    project = _projector(width, height)

    def point(value):
        x, y = project(value)
        return "%.2f,%.2f" % (x, y)

    court_tl = project((-20.0, 10.0))
    court_br = project((20.0, -10.0))
    center_top = project((0.0, 10.0))
    center_bottom = project((0.0, -10.0))
    center = project((0.0, 0.0))
    radius = abs(project((3.0, 0.0))[0] - center[0])
    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1800" height="1200" viewBox="0 0 1800 1200">',
        '<rect width="100%" height="100%" fill="#f5f7fa"/>',
        '<text x="110" y="62" font-family="Arial,sans-serif" font-size="36" font-weight="700" fill="#17212b">FutsalMOT Camera Coverage | %s</text>' % camera_set,
        '<text x="110" y="98" font-family="Arial,sans-serif" font-size="21" fill="#627786">Camera profile transform | ground plane z = 0 | units: m</text>',
        '<rect x="%.2f" y="%.2f" width="%.2f" height="%.2f" fill="#e8f1e8"/>' % (court_tl[0], court_tl[1], court_br[0] - court_tl[0], court_br[1] - court_tl[1]),
    ]
    for camera_id, camera in cameras.items():
        fill = "#d9782e" if camera_id == "P01" else "#286cae"
        lines.append('<polygon points="%s" fill="%s" fill-opacity="%s" stroke="%s" stroke-width="3"/>' % (" ".join(point(vertex) for vertex in camera["footprint"].ground_polygon), fill, "0.30" if camera_id == "P01" else "0.20", fill))
    lines.extend([
        '<line x1="%.2f" y1="%.2f" x2="%.2f" y2="%.2f" stroke="#627786" stroke-width="3"/>' % (center_top[0], center_top[1], center_bottom[0], center_bottom[1]),
        '<circle cx="%.2f" cy="%.2f" r="%.2f" fill="none" stroke="#627786" stroke-width="3"/>' % (center[0], center[1], radius),
        '<rect x="%.2f" y="%.2f" width="%.2f" height="%.2f" fill="none" stroke="#304657" stroke-width="5"/>' % (court_tl[0], court_tl[1], court_br[0] - court_tl[0], court_br[1] - court_tl[1]),
    ])
    for camera_id, camera in cameras.items():
        fill = "#d9782e" if camera_id == "P01" else "#286cae"
        origin = project(camera["position_m"])
        target = project((camera["position_m"][0] + camera["forward"][0] * 4.0, camera["position_m"][1] + camera["forward"][1] * 4.0))
        lines.append('<line x1="%.2f" y1="%.2f" x2="%.2f" y2="%.2f" stroke="%s" stroke-width="4"/>' % (origin[0], origin[1], target[0], target[1], fill))
        lines.append('<circle cx="%.2f" cy="%.2f" r="9" fill="%s"/><text x="%.2f" y="%.2f" font-family="Arial,sans-serif" font-size="25" fill="#17212b">%s</text>' % (origin[0], origin[1], fill, origin[0] + 14, origin[1] - 10, camera_id))
    lines.extend([
        '<rect x="1420" y="1030" width="350" height="90" fill="#fff" stroke="#b8c3cc" stroke-width="2"/><circle cx="1445" cy="1065" r="9" fill="#286cae"/><text x="1470" y="1072" font-family="Arial,sans-serif" font-size="21" fill="#17212b">Anchor footprint</text><circle cx="1445" cy="1100" r="9" fill="#d9782e"/><text x="1470" y="1107" font-family="Arial,sans-serif" font-size="21" fill="#17212b">Partial footprint</text>',
        '</svg>',
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_debug(path: Path, camera: Mapping) -> None:
    footprint = camera["footprint"].ground_polygon
    lines = [
        "Camera:",
        CAMERA_ID,
        "",
        "Position:",
        "(%0.9f, %0.9f, %0.9f)" % tuple(camera["position_m"]),
        "",
        "Rotation (profile, informational only):",
        "(%0.9f, %0.9f, %0.9f)" % tuple(camera["rotation_deg"]),
        "",
        "Forward (derived from profile rotation):",
        "(%0.12f, %0.12f, %0.12f)" % tuple(camera["forward"]),
        "",
        "Right (derived from profile rotation):",
        "(%0.12f, %0.12f, %0.12f)" % tuple(camera["right"]),
        "",
        "Up (derived from profile rotation):",
        "(%0.12f, %0.12f, %0.12f)" % tuple(camera["up"]),
        "",
        "Horizontal FOV (degrees):",
        "%0.9f" % camera["horizontal_fov_deg"],
        "",
        "Ground footprint vertices:",
    ]
    lines.extend("(%0.9f, %0.9f, %0.9f)" % tuple(point) for point in footprint)
    lines.extend(["", "Data source:", "C1 simulation.camera.profiles (default source: profile)", "Ground plane: z=0"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--camera-set", choices=("c1_debug", "anchor_only", "anchor_plus_one_partial"), default="c1_debug")
    args = parser.parse_args()
    if args.camera_set == "c1_debug":
        outputs = render_c1_debug(args.config, args.output_dir)
    else:
        outputs = render_camera_coverage(args.config, args.camera_set, args.output_dir)
    result = {"png": str(outputs.png), "svg": str(outputs.svg)}
    if outputs.debug is not None:
        result["debug"] = str(outputs.debug)
    print(json.dumps(result, ensure_ascii=True))


if __name__ == "__main__":
    main()
