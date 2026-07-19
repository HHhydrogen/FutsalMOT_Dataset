#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FutsalMOT 布局检查 + 后处理.

执行流程：
- 读取 objects_bbox_2d_clean_<seq_id>.json
- 按 --step 间隔绘制布局检查图（bbox + 球员关键点 + 场地关键点 + 边界线）
- 写入 YOLO / MOT 标签

布局检查图输出到：
    Saved/FutsalMOT/layout_check/<seq_id>/
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError

CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from futsalmot.core.io import read_json, write_json_atomic, write_text_atomic
from futsalmot.core.paths import CODE_DIR, CURRENT_RUN_POINTER, PROJECT_ROOT

SCRIPT_VERSION = "A1_5_POSTPROCESS_V2"
SCRIPT_DIR = CODE_DIR
DEFAULT_PROJECT_ROOT = PROJECT_ROOT


class PostprocessError(RuntimeError):
    """Fatal input or integrity error."""


def normalize_path_text(path: Path) -> str:
    return path.resolve().as_posix()


def default_annotation_path(project_root: Path) -> Path:
    if CURRENT_RUN_POINTER.is_file():
        try:
            pointer = read_json(CURRENT_RUN_POINTER)
            seq_id = str(pointer.get("seq_id", pointer.get("episode_id", ""))).strip()
            if seq_id:
                return (
                    project_root
                    / "Saved"
                    / "FutsalMOT"
                    / "annotations"
                    / "objects_bbox_2d_clean_{}.json".format(seq_id)
                )
        except Exception:
            pass
    return (
        project_root
        / "Saved"
        / "FutsalMOT"
        / "annotations"
        / "objects_bbox_2d_clean_episode_random_0001_t1.json"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FutsalMOT overlay + YOLO/MOT conversion + integrity check."
    )
    parser.add_argument(
        "--annotation",
        type=Path,
        default=None,
        help=(
            "objects_bbox_2d_clean_<seq_id>.json. If omitted, the current "
            "pipeline pointer is used."
        ),
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Override project root. Otherwise inferred from annotation path.",
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Do not remove this sequence's prior overlay/YOLO/MOT outputs.",
    )
    parser.add_argument(
        "--skip-overlay",
        action="store_true",
        help="Validate and convert labels without writing overlay PNGs.",
    )
    parser.add_argument(
        "--draw-keypoints",
        action="store_true",
        help="Draw player keypoints_2d on overlay PNGs when present.",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=1,
        help="每隔 N 帧绘制一张布局检查图（默认 1，建议 5）",
    )
    parser.add_argument(
        "--field-keypoints",
        type=Path,
        default=None,
        help="场地关键点 JSON 文件路径；默认根据 annotation 自动推导",
    )
    parser.add_argument(
        "--layout-dir",
        type=Path,
        default=None,
        help="布局检查图输出目录；默认自动推导到 Saved/FutsalMOT/layout_check/<seq_id>",
    )
    return parser.parse_args()


def infer_project_root(annotation_path: Path, override: Optional[Path]) -> Path:
    if override is not None:
        return override.expanduser().resolve()
    resolved = annotation_path.expanduser().resolve()
    # Expected: <project>/Saved/FutsalMOT/annotations/file.json
    parents = resolved.parents
    if len(parents) >= 4 and parents[0].name.lower() == "annotations":
        if parents[1].name.lower() == "futsalmot" and parents[2].name.lower() == "saved":
            return parents[3]
    return DEFAULT_PROJECT_ROOT.resolve()


def load_annotation_file(path: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    if not path.is_file():
        raise PostprocessError("找不到标注文件：{}".format(path))
    try:
        data = read_json(path)
    except json.JSONDecodeError as exc:
        raise PostprocessError(
            "标注 JSON 解析失败：{} line={} column={}".format(
                exc.msg, exc.lineno, exc.colno
            )
        ) from exc
    except OSError as exc:
        raise PostprocessError("无法读取标注文件：{}".format(exc)) from exc

    if isinstance(data, dict):
        records = data.get("records")
        metadata = data
    elif isinstance(data, list):
        records = data
        metadata = {}
    else:
        raise PostprocessError("标注 JSON 顶层必须是 object 或 records 数组")
    if not isinstance(records, list) or not records:
        raise PostprocessError("标注 JSON 缺少非空 records 数组")
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise PostprocessError("records[{}] 必须是 object".format(index))
    return metadata, records


def require_int(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise PostprocessError("{} 必须是整数".format(field))
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise PostprocessError("{} 必须是整数，当前={!r}".format(field, value)) from exc
    if isinstance(value, float) and not value.is_integer():
        raise PostprocessError("{} 必须是整数，当前={!r}".format(field, value))
    return result


def sorted_unique(values: Iterable[str]) -> List[str]:
    return sorted(set(values))


def infer_object_ids(records: Sequence[Dict[str, Any]]) -> List[str]:
    result: List[str] = []
    seen = set()
    for record in records:
        for obj in record.get("objects", []):
            if not isinstance(obj, dict):
                continue
            object_id = str(obj.get("object_id", "")).strip()
            if object_id and object_id not in seen:
                seen.add(object_id)
                result.append(object_id)
    return result


def infer_class_id_map(records: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    result: Dict[str, int] = {}
    for record in records:
        for obj in record.get("objects", []):
            if not isinstance(obj, dict):
                continue
            category = str(obj.get("category", "")).strip()
            value = obj.get("class_id")
            if not category or value is None:
                continue
            class_id = require_int(value, "class_id")
            if category in result and result[category] != class_id:
                raise PostprocessError(
                    "category={} 出现不同 class_id：{} / {}".format(
                        category, result[category], class_id
                    )
                )
            result[category] = class_id
    return result


def infer_track_id_map(records: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    result: Dict[str, int] = {}
    used: Dict[int, str] = {}
    for record in records:
        for obj in record.get("objects", []):
            if not isinstance(obj, dict):
                continue
            object_id = str(obj.get("object_id", "")).strip()
            value = obj.get("track_id")
            if not object_id or value is None:
                continue
            track_id = require_int(value, "track_id")
            if object_id in result and result[object_id] != track_id:
                raise PostprocessError(
                    "object_id={} 出现不同 track_id：{} / {}".format(
                        object_id, result[object_id], track_id
                    )
                )
            if track_id in used and used[track_id] != object_id:
                raise PostprocessError(
                    "track_id={} 被 {} 和 {} 重复使用".format(
                        track_id, used[track_id], object_id
                    )
                )
            result[object_id] = track_id
            used[track_id] = object_id
    return result


def resolve_rgb_path(
    record: Dict[str, Any],
    project_root: Path,
    seq_id: str,
    camera_id: str,
    frame_id: int,
) -> Tuple[Path, str]:
    raw = record.get("rgb_path")
    if isinstance(raw, str) and raw.strip():
        candidate = Path(raw.strip())
        if not candidate.is_absolute():
            # UE builder writes paths relative to Saved/FutsalMOT.
            candidate = project_root / "Saved" / "FutsalMOT" / candidate
        candidate = candidate.resolve()
        if candidate.is_file():
            return candidate, "record.rgb_path"

    image_root = project_root / "Saved" / "FutsalMOT" / "images_clean" / seq_id / camera_id
    candidates = [
        image_root / "{:06d}.png".format(frame_id),
        image_root / "{:04d}.png".format(frame_id),
        image_root / "{}.png".format(frame_id),
    ]
    existing = [path.resolve() for path in candidates if path.is_file()]
    if existing:
        return existing[0], "legacy_fallback"
    return candidates[0].resolve(), "missing"


def parse_bbox(value: Any, field: str) -> Tuple[float, float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise PostprocessError("{} 必须是 [x,y,w,h]".format(field))
    converted: List[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise PostprocessError("{} 包含非数值".format(field))
        number = float(item)
        if not math.isfinite(number):
            raise PostprocessError("{} 包含非有限数值".format(field))
        converted.append(number)
    return converted[0], converted[1], converted[2], converted[3]


def load_font(size: int = 18) -> ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/msyh.ttc"),
    ]
    for path in candidates:
        if path.is_file():
            try:
                return ImageFont.truetype(str(path), size)
            except Exception:
                pass
    return ImageFont.load_default()


def object_color(obj: Dict[str, Any]) -> Tuple[int, int, int]:
    category = str(obj.get("category", ""))
    if category == "ball":
        return 40, 120, 255
    if category == "player":
        return 20, 220, 80
    return 255, 255, 0


def draw_object_keypoints(draw: ImageDraw.ImageDraw, obj: Dict[str, Any]) -> None:
    keypoints = obj.get("keypoints_2d")
    if not isinstance(keypoints, list):
        return
    for kp in keypoints:
        if not isinstance(kp, dict):
            continue
        if require_int(kp.get("visibility", 0), "keypoint.visibility") != 2:
            continue
        x = kp.get("x")
        y = kp.get("y")
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            continue
        radius = 3.0
        draw.ellipse(
            [float(x) - radius, float(y) - radius, float(x) + radius, float(y) + radius],
            fill=(255, 230, 40),
            outline=(0, 0, 0),
        )


def resolve_field_keypoints_path(annotation_path: Path) -> Path:
    """从 bbox 标注路径推导同序列的场地关键点文件路径。"""
    stem = annotation_path.stem
    project_root = annotation_path.parents[3]
    seq_id = stem
    prefix = "objects_bbox_2d_clean_"
    if seq_id.startswith(prefix):
        seq_id = seq_id[len(prefix):]
    candidates = [
        project_root / "Saved" / "FutsalMOT" / "annotations" / "field_keypoints_2d_clean_frame000000.json",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def load_field_keypoints(path: Path) -> Dict[str, Dict[int, List[Dict[str, Any]]]]:
    """加载场地关键点，返回 kp_by_camera_frame[camera_id][frame_id] = [kps]。"""
    result: Dict[str, Dict[int, List[Dict[str, Any]]]] = {}
    if not path.is_file():
        return result
    data = read_json(path)
    records = data.get("records") if isinstance(data, dict) else data
    if not isinstance(records, list):
        return result
    for rec in records:
        cam = str(rec.get("camera_id", ""))
        fid = int(rec.get("frame_id", 0))
        kps = rec.get("field_keypoints", [])
        if cam not in result:
            result[cam] = {}
        result[cam][fid] = kps
    return result


def draw_field_keypoints(draw: ImageDraw.ImageDraw, kps: List[Dict[str, Any]]) -> None:
    """在 draw 上绘制场地关键点。"""
    for kp in kps:
        if not kp.get("in_image", False):
            continue
        uv = kp.get("clean_uv")
        if not isinstance(uv, (list, tuple)) or len(uv) < 2:
            continue
        x, y = float(uv[0]), float(uv[1])
        radius = 4.0
        draw.ellipse(
            [x - radius, y - radius, x + radius, y + radius],
            fill=(255, 100, 100),
            outline=(200, 50, 50),
        )
        name = str(kp.get("name", ""))
        if name:
            draw.text((x + 5, y - 8), name, fill=(255, 200, 200), font=load_font(10))


COURT_BOUNDARY_SEGMENTS = [
    ("KP01", "KP02"), ("KP02", "KP03"), ("KP03", "KP04"), ("KP04", "KP01"),
    ("KP05", "KP06"), ("KP06", "KP07"), ("KP07", "KP08"), ("KP08", "KP05"),
    ("KP09", "KP10"), ("KP10", "KP11"), ("KP11", "KP12"), ("KP12", "KP09"),
    ("KP13", "KP17"), ("KP17", "KP14"), ("KP14", "KP18"), ("KP18", "KP13"),
    ("KP15", "KP16"),
]


def build_kp_name_map(kps: List[Dict[str, Any]]) -> Dict[str, List[float]]:
    m: Dict[str, List[float]] = {}
    for kp in kps:
        name = str(kp.get("name", ""))
        uv = kp.get("clean_uv")
        if name and isinstance(uv, (list, tuple)) and len(uv) >= 2 and kp.get("in_image", False):
            m[name] = [float(uv[0]), float(uv[1])]
    return m


def draw_court_boundaries(draw: ImageDraw.ImageDraw, kps: List[Dict[str, Any]]) -> None:
    name_map = build_kp_name_map(kps)
    for a, b in COURT_BOUNDARY_SEGMENTS:
        if a in name_map and b in name_map:
            draw.line(
                [name_map[a][0], name_map[a][1], name_map[b][0], name_map[b][1]],
                fill=(150, 200, 255),
                width=2,
            )


def yolo_line(obj: Dict[str, Any], width: int, height: int) -> Optional[str]:
    bbox = obj.get("bbox_2d_clean")
    if bbox is None:
        return None
    x, y, w, h = parse_bbox(bbox, "bbox_2d_clean")
    if w <= 1.0 or h <= 1.0:
        return None
    class_id = require_int(obj.get("class_id"), "class_id")
    xc = (x + w / 2.0) / float(width)
    yc = (y + h / 2.0) / float(height)
    wn = w / float(width)
    hn = h / float(height)
    return "{} {:.6f} {:.6f} {:.6f} {:.6f}".format(class_id, xc, yc, wn, hn)


def mot_line(frame_id: int, frame_start: int, obj: Dict[str, Any]) -> Optional[str]:
    bbox = obj.get("bbox_2d_clean")
    if bbox is None:
        return None
    x, y, w, h = parse_bbox(bbox, "bbox_2d_clean")
    if w <= 1.0 or h <= 1.0:
        return None
    track_id = require_int(obj.get("track_id"), "track_id")
    frame_mot = frame_id - frame_start + 1
    return (
        "{},{},{:.2f},{:.2f},{:.2f},{:.2f},1,-1,-1,-1".format(
            frame_mot, track_id, x, y, w, h
        )
    )


def main() -> int:
    args = parse_args()
    tentative_root = (
        args.project_root.expanduser().resolve()
        if args.project_root is not None
        else DEFAULT_PROJECT_ROOT.resolve()
    )
    annotation_path = (
        args.annotation.expanduser().resolve()
        if args.annotation is not None
        else default_annotation_path(tentative_root).resolve()
    )
    project_root = infer_project_root(annotation_path, args.project_root)

    try:
        metadata, records = load_annotation_file(annotation_path)
        first = records[0]
        frame_values = [require_int(record.get("frame_id"), "record.frame_id") for record in records]
        camera_values = [str(record.get("camera_id", "")).strip() for record in records]
        if any(not value for value in camera_values):
            raise PostprocessError("record.camera_id 不能为空")

        seq_id = str(metadata.get("seq_id", first.get("seq_id", ""))).strip()
        if not seq_id:
            prefix = "objects_bbox_2d_clean_"
            seq_id = annotation_path.stem
            if seq_id.startswith(prefix):
                seq_id = seq_id[len(prefix):]
        if not seq_id:
            raise PostprocessError("无法确定 seq_id")

        frame_start = require_int(metadata.get("frame_start", min(frame_values)), "frame_start")
        frame_end = require_int(metadata.get("frame_end", max(frame_values)), "frame_end")
        if frame_end < frame_start:
            raise PostprocessError("frame_end 小于 frame_start")
        image_width = require_int(
            metadata.get("image_width", first.get("image_width", 1920)), "image_width"
        )
        image_height = require_int(
            metadata.get("image_height", first.get("image_height", 1080)), "image_height"
        )
        if image_width <= 0 or image_height <= 0:
            raise PostprocessError("图像尺寸必须为正数")

        raw_cameras = metadata.get("camera_ids")
        camera_ids = (
            [str(value) for value in raw_cameras]
            if isinstance(raw_cameras, list) and raw_cameras
            else sorted_unique(camera_values)
        )
        raw_objects = metadata.get("object_ids")
        object_ids = (
            [str(value) for value in raw_objects]
            if isinstance(raw_objects, list) and raw_objects
            else infer_object_ids(records)
        )
        if not camera_ids or not object_ids:
            raise PostprocessError("camera_ids/object_ids 不能为空")

        raw_class_map = metadata.get("class_id_map")
        class_id_map = (
            {str(key): require_int(value, "class_id_map") for key, value in raw_class_map.items()}
            if isinstance(raw_class_map, dict) and raw_class_map
            else infer_class_id_map(records)
        )
        raw_track_map = metadata.get("track_id_map")
        track_id_map = (
            {str(key): require_int(value, "track_id_map") for key, value in raw_track_map.items()}
            if isinstance(raw_track_map, dict) and raw_track_map
            else infer_track_id_map(records)
        )

        grouped: Dict[Tuple[str, int], Dict[str, Any]] = {}
        resolved_images: Dict[Tuple[str, int], Path] = {}
        resolution_sources: Dict[str, int] = {}
        errors: List[str] = []
        expected_object_set = set(object_ids)

        for index, record in enumerate(records):
            camera_id = str(record.get("camera_id", "")).strip()
            frame_id = require_int(record.get("frame_id"), "records[{}].frame_id".format(index))
            key = camera_id, frame_id
            if key in grouped:
                errors.append("重复 record：camera={} frame={}".format(camera_id, frame_id))
                continue
            grouped[key] = record
            image_path, source = resolve_rgb_path(record, project_root, seq_id, camera_id, frame_id)
            resolved_images[key] = image_path
            resolution_sources[source] = resolution_sources.get(source, 0) + 1

            objects = record.get("objects")
            if not isinstance(objects, list):
                errors.append("{} frame={} objects 不是数组".format(camera_id, frame_id))
                continue
            actual_ids = [str(obj.get("object_id", "")) for obj in objects if isinstance(obj, dict)]
            if len(objects) != len(object_ids):
                errors.append(
                    "{} frame={} objects={} expected={}".format(
                        camera_id, frame_id, len(objects), len(object_ids)
                    )
                )
            if len(set(actual_ids)) != len(actual_ids):
                errors.append("{} frame={} object_id 重复".format(camera_id, frame_id))
            if set(actual_ids) != expected_object_set:
                errors.append(
                    "{} frame={} 对象集合不匹配 missing={} extra={}".format(
                        camera_id,
                        frame_id,
                        sorted(expected_object_set - set(actual_ids)),
                        sorted(set(actual_ids) - expected_object_set),
                    )
                )

            record_w = require_int(record.get("image_width", image_width), "record.image_width")
            record_h = require_int(record.get("image_height", image_height), "record.image_height")
            if record_w != image_width or record_h != image_height:
                errors.append(
                    "{} frame={} record 尺寸 {}x{} 与 metadata {}x{} 不一致".format(
                        camera_id, frame_id, record_w, record_h, image_width, image_height
                    )
                )

            for obj_index, obj in enumerate(objects):
                if not isinstance(obj, dict):
                    errors.append("{} frame={} objects[{}] 非 object".format(camera_id, frame_id, obj_index))
                    continue
                object_id = str(obj.get("object_id", ""))
                expected_track = track_id_map.get(object_id)
                if expected_track is not None:
                    try:
                        actual_track = require_int(obj.get("track_id"), "track_id")
                    except PostprocessError as exc:
                        errors.append(str(exc))
                    else:
                        if actual_track != expected_track:
                            errors.append(
                                "{} frame={} {} track_id={} expected={}".format(
                                    camera_id, frame_id, object_id, actual_track, expected_track
                                )
                            )
                if obj.get("visible", False):
                    try:
                        x, y, w, h = parse_bbox(
                            obj.get("bbox_2d_clean"),
                            "{} frame={} {} bbox".format(camera_id, frame_id, object_id),
                        )
                    except PostprocessError as exc:
                        errors.append(str(exc))
                        continue
                    if w <= 1.0 or h <= 1.0:
                        errors.append(
                            "{} frame={} {} bbox 非正尺寸 {}".format(
                                camera_id, frame_id, object_id, [x, y, w, h]
                            )
                        )
                    tolerance = 1.0
                    if (
                        x < -tolerance
                        or y < -tolerance
                        or x + w > image_width + tolerance
                        or y + h > image_height + tolerance
                    ):
                        errors.append(
                            "{} frame={} {} bbox 越界 {} image={}x{}".format(
                                camera_id, frame_id, object_id, [x, y, w, h], image_width, image_height
                            )
                        )
                elif obj.get("bbox_2d_clean") is not None:
                    errors.append(
                        "{} frame={} {} visible=false 但 bbox 非 null".format(
                            camera_id, frame_id, object_id
                        )
                    )

        expected_frames = frame_end - frame_start + 1
        expected_records = len(camera_ids) * expected_frames
        if len(records) != expected_records:
            errors.append("records={} expected={}".format(len(records), expected_records))

        # Validate exact camera/frame grid and RGB files before deleting prior outputs.
        for camera_id in camera_ids:
            for frame_id in range(frame_start, frame_end + 1):
                key = camera_id, frame_id
                if key not in grouped:
                    errors.append("缺少 record：{} frame={}".format(camera_id, frame_id))
                    continue
                path = resolved_images[key]
                if not path.is_file():
                    errors.append("缺少图片：{}".format(path))
                    continue
                try:
                    with Image.open(path) as image:
                        image.verify()
                    with Image.open(path) as image:
                        actual_size = image.size
                except (OSError, UnidentifiedImageError) as exc:
                    errors.append("图片无法读取：{} ({})".format(path, exc))
                    continue
                if actual_size != (image_width, image_height):
                    errors.append(
                        "图片尺寸不一致：{} got={} expected={}".format(
                            path, actual_size, (image_width, image_height)
                        )
                    )

        # Flag stale/extra numeric PNGs because they can hide frame-range mistakes.
        expected_paths = {path.resolve() for path in resolved_images.values() if path.is_file()}
        for camera_id in camera_ids:
            camera_dir = project_root / "Saved" / "FutsalMOT" / "images_clean" / seq_id / camera_id
            if not camera_dir.is_dir():
                continue
            for path in camera_dir.glob("*.png"):
                if path.resolve() not in expected_paths and path.stem.isdigit():
                    errors.append("发现额外/陈旧图片：{}".format(path))

        print("=" * 72)
        print("FutsalMOT postprocess {}".format(SCRIPT_VERSION))
        print("annotation={}".format(annotation_path))
        print("project_root={}".format(project_root))
        print("seq_id={}".format(seq_id))
        print("frames={}..{} cameras={} records={}".format(
            frame_start, frame_end, camera_ids, len(records)
        ))
        print("objects={} image={}x{}".format(object_ids, image_width, image_height))
        print("rgb_resolution_sources={}".format(resolution_sources))
        print("=" * 72)

        if errors:
            print("PRECHECK FAILED: {} errors".format(len(errors)), file=sys.stderr)
            for message in errors[:200]:
                print("[ERROR] {}".format(message), file=sys.stderr)
            if len(errors) > 200:
                print("[ERROR] ... {} more".format(len(errors) - 200), file=sys.stderr)
            return 1

        saved_root = project_root / "Saved" / "FutsalMOT"
        step = max(1, int(args.step))
        overlay_root = saved_root / "overlay_objects_bbox_{}".format(seq_id)
        layout_root = saved_root / "layout_check" / seq_id
        yolo_root = saved_root / "labels_yolo_clean" / seq_id
        mot_root = saved_root / "labels_mot_clean" / seq_id
        manifest_path = saved_root / "annotations" / "manifest_{}.json".format(seq_id)

        field_kp_path = (
            args.field_keypoints.expanduser().resolve()
            if args.field_keypoints is not None
            else resolve_field_keypoints_path(annotation_path)
        )
        field_kp_data = load_field_keypoints(field_kp_path)
        has_field_kp = bool(field_kp_data)

        if not args.no_clean:
            for path in (overlay_root, yolo_root, mot_root):
                if path.exists():
                    shutil.rmtree(path)
        if not args.skip_overlay:
            overlay_root.mkdir(parents=True, exist_ok=True)
        layout_root.mkdir(parents=True, exist_ok=True)
        yolo_root.mkdir(parents=True, exist_ok=True)
        mot_root.mkdir(parents=True, exist_ok=True)

        font = load_font(18)
        yolo_file_count = 0
        yolo_line_count = 0
        mot_stats: Dict[str, int] = {}
        manifest_records: List[Dict[str, Any]] = []
        layout_count = 0

        for camera_id in camera_ids:
            if not args.skip_overlay:
                (overlay_root / camera_id).mkdir(parents=True, exist_ok=True)
            (layout_root / camera_id).mkdir(parents=True, exist_ok=True)
            (yolo_root / camera_id).mkdir(parents=True, exist_ok=True)
            mot_lines: List[str] = []

            for frame_id in range(frame_start, frame_end + 1):
                key = camera_id, frame_id
                record = grouped[key]
                image_path = resolved_images[key]
                objects = sorted(
                    record.get("objects", []),
                    key=lambda obj: require_int(obj.get("track_id"), "track_id"),
                )

                yolo_lines: List[str] = []
                for obj in objects:
                    if not obj.get("visible", False):
                        continue
                    line = yolo_line(obj, image_width, image_height)
                    if line is not None:
                        yolo_lines.append(line)
                    mot = mot_line(frame_id, frame_start, obj)
                    if mot is not None:
                        mot_lines.append(mot)

                yolo_path = yolo_root / camera_id / "{:06d}.txt".format(frame_id)
                write_text_atomic(
                    yolo_path,
                    ("\n".join(yolo_lines) + "\n") if yolo_lines else "",
                )
                yolo_file_count += 1
                yolo_line_count += len(yolo_lines)

                do_layout = (frame_id - frame_start) % step == 0 or frame_id == frame_end

                overlay_path: Optional[Path] = None
                if not args.skip_overlay and do_layout:
                    with Image.open(image_path) as source:
                        image = source.convert("RGB")
                    draw = ImageDraw.Draw(image)

                    # Draw field keypoints + court boundaries
                    cam_kp = field_kp_data.get(camera_id, {})
                    frame_kps = cam_kp.get(frame_id, cam_kp.get(0, []))
                    if frame_kps:
                        draw_court_boundaries(draw, frame_kps)
                        draw_field_keypoints(draw, frame_kps)

                    # Draw objects
                    for obj in objects:
                        if not obj.get("visible", False):
                            continue
                        x, y, w, h = parse_bbox(obj["bbox_2d_clean"], "bbox")
                        color = object_color(obj)
                        draw.rectangle([x, y, x + w, y + h], outline=color, width=3)
                        label = "{} id={}".format(
                            obj.get("category", "obj"), obj.get("track_id", "")
                        )
                        tx = max(0.0, min(float(image_width - 165), x))
                        ty = max(0.0, y - 22.0)
                        draw.rectangle([tx, ty, tx + 165, ty + 22], fill=(0, 0, 0))
                        draw.text((tx + 4, ty + 2), label, fill=color, font=font)
                        if args.draw_keypoints:
                            draw_object_keypoints(draw, obj)

                    overlay_path = layout_root / camera_id / "{:06d}.png".format(frame_id)
                    image.save(overlay_path)
                    layout_count += 1

                manifest_records.append(
                    {
                        "camera_id": camera_id,
                        "frame_id": frame_id,
                        "image_path": normalize_path_text(image_path),
                        "overlay_path": normalize_path_text(overlay_path) if overlay_path else None,
                        "yolo_label_path": normalize_path_text(yolo_path),
                    }
                )

            gt_path = mot_root / camera_id / "gt" / "gt.txt"
            write_text_atomic(gt_path, ("\n".join(mot_lines) + "\n") if mot_lines else "")
            mot_stats[camera_id] = len(mot_lines)

        manifest = {
            "schema_version": "1.0",
            "generator_version": SCRIPT_VERSION,
            "seq_id": seq_id,
            "frame_start": frame_start,
            "frame_end": frame_end,
            "frame_end_exclusive": frame_end + 1,
            "image_width": image_width,
            "image_height": image_height,
            "annotation_json": normalize_path_text(annotation_path),
            "project_root": normalize_path_text(project_root),
            "camera_ids": camera_ids,
            "object_ids": object_ids,
            "class_id_map": class_id_map,
            "track_id_map": track_id_map,
            "expected_records": expected_records,
            "records": manifest_records,
        }
        write_json_atomic(manifest_path, manifest)

        # Final output counts.
        final_errors: List[str] = []
        expected_yolo = expected_records
        actual_yolo = sum(1 for _ in yolo_root.rglob("*.txt"))
        if actual_yolo != expected_yolo:
            final_errors.append("YOLO files={} expected={}".format(actual_yolo, expected_yolo))
        for camera_id in camera_ids:
            gt_path = mot_root / camera_id / "gt" / "gt.txt"
            if not gt_path.is_file():
                final_errors.append("缺少 MOT gt：{}".format(gt_path))

        print("[CHECK] expected_objects_per_record={}".format(len(object_ids)))
        print("[CHECK] records={} expected={}".format(len(records), expected_records))
        print("[CHECK] yolo_files={} expected={}".format(actual_yolo, expected_yolo))
        print("[CHECK] yolo_total_lines={}".format(yolo_line_count))
        print("[CHECK] layout_check_images={} (间隔 N={})".format(layout_count, step))
        for camera_id in camera_ids:
            print("[CHECK] {} MOT lines={}".format(camera_id, mot_stats[camera_id]))
        print("")
        print("=" * 72)
        print("布局检查图输出目录:")
        print("  {}".format(layout_root.resolve().as_posix()))
        print("")
        print("包含内容：")
        print("  - 所有目标的 bbox")
        print("  - 球员骨骼关键点" + (" (已启用)" if args.draw_keypoints else ""))
        print("  - 场地关键点" + (" (已加载)" if has_field_kp else " (未找到场地关键点文件)"))
        print("  - 场地边界线" + (" (已绘制)" if has_field_kp else ""))
        print("=" * 72)

        if final_errors:
            print("CHECK FAILED", file=sys.stderr)
            for message in final_errors:
                print("[ERROR] {}".format(message), file=sys.stderr)
            return 1

        print("CHECK PASSED")
        print("ALL DONE")
        return 0

    except PostprocessError as exc:
        print("[ERROR] {}".format(exc), file=sys.stderr)
        return 2
    except Exception as exc:
        print(
            "[UNEXPECTED ERROR] {}: {}".format(type(exc).__name__, exc),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
