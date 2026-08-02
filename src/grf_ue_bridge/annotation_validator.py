"""验证导出的 CV 标注目录。纯 Python，不依赖 unreal。

内部 annotation 约定（见 ue/annotation_exporter.py 与 README）：
  - 每个 camera 子目录含 camera.json、annotations.jsonl、gt/gt.txt、seqinfo.ini。
  - annotations.jsonl 每行：episode_id / camera_id / frame_index（1 基）/
    source_step / time_seconds / objects[]。
  - object 字段：entity_id / track_id / class / team / role / is_goalkeeper /
    world_position / raw_bbox_xywh / raw_bbox_xyxy / bbox_xywh / bbox_xyxy /
    in_frame / truncated / visibility。in_frame=true 时 bbox_xywh 必须是合法 bbox。
"""

import json
import math
import sys
from pathlib import Path
from typing import Dict, List

# 把仓库的 ue/ 目录加入 sys.path（与 tests/conftest.py 一致），以便 import 纯模块
_UE_DIR = Path(__file__).resolve().parent.parent.parent / "ue"
if str(_UE_DIR) not in sys.path:
    sys.path.insert(0, str(_UE_DIR))

from annotation_utils import entity_id_to_mask_id  # noqa: E402


def _validate_camera(cam_dir: Path) -> List[str]:
    """验证单个 camera 子目录。返回错误列表（空表示通过）。"""
    errors: List[str] = []
    cam_label = cam_dir.name

    # ── camera.json ──────────────────────────────────────────────
    cam_json_path = cam_dir / "camera.json"
    if not cam_json_path.exists():
        errors.append(f"[{cam_label}] 缺少 camera.json")
        return errors
    try:
        cam = json.loads(cam_json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        errors.append(f"[{cam_label}] camera.json 不是合法 JSON: {e}")
        return errors

    intr = cam.get("intrinsics")
    if not isinstance(intr, dict):
        errors.append(f"[{cam_label}] camera.json 缺少 intrinsics")
        return errors
    for key in ("width", "height", "fx", "fy", "cx", "cy"):
        if not isinstance(intr.get(key), (int, float)):
            errors.append(f"[{cam_label}] intrinsics.{key} 缺失或非数字")
    extr = cam.get("extrinsics")
    if not isinstance(extr, dict):
        errors.append(f"[{cam_label}] camera.json 缺少 extrinsics")

    width = int(intr.get("width", 0))
    height = int(intr.get("height", 0))
    if width <= 0 or height <= 0:
        errors.append(f"[{cam_label}] 图像尺寸非法: {width}x{height}")

    # ── annotations.jsonl ────────────────────────────────────────
    ann_path = cam_dir / "annotations.jsonl"
    if not ann_path.exists():
        errors.append(f"[{cam_label}] 缺少 annotations.jsonl")
        return errors

    entity_to_track: Dict[str, int] = {}
    track_to_entity: Dict[int, str] = {}
    prev_frame = 0

    with open(ann_path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                frame = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"[{cam_label}] annotations.jsonl 第 {line_no} 行不是合法 JSON: {e}")
                continue
            if not isinstance(frame, dict):
                errors.append(f"[{cam_label}] 第 {line_no} 行不是 JSON 对象")
                continue

            frame_index = frame.get("frame_index")
            if not isinstance(frame_index, int) or frame_index < 1:
                errors.append(f"[{cam_label}] 第 {line_no} 行 frame_index 非法: {frame_index!r}")
            if line_no > 1 and isinstance(frame_index, int) and frame_index != prev_frame + 1:
                errors.append(
                    f"[{cam_label}] 第 {line_no} 行 frame_index={frame_index}，"
                    f"期望 {prev_frame + 1}（连续递增）"
                )
            if isinstance(frame_index, int):
                prev_frame = frame_index

            objects = frame.get("objects")
            if not isinstance(objects, list):
                errors.append(f"[{cam_label}] 第 {line_no} 行缺少 objects 列表")
                continue

            for oi, obj in enumerate(objects):
                label = f"[{cam_label}] 第 {line_no} 行 objects[{oi}]"
                if not isinstance(obj, dict):
                    errors.append(f"{label} 不是对象")
                    continue
                entity_id = obj.get("entity_id")
                track_id = obj.get("track_id")
                if not isinstance(entity_id, str) or not isinstance(track_id, int):
                    errors.append(f"{label} entity_id/track_id 缺失或类型非法")
                    continue
                # entity ↔ track 双向一致
                if entity_id in entity_to_track and entity_to_track[entity_id] != track_id:
                    errors.append(
                        f"{label} entity_id={entity_id} 之前映射到 track {entity_to_track[entity_id]}，"
                        f"现在却为 {track_id}"
                    )
                if track_id in track_to_entity and track_to_entity[track_id] != entity_id:
                    errors.append(
                        f"{label} track_id={track_id} 之前对应 {track_to_entity[track_id]}，"
                        f"现在却为 {entity_id}"
                    )
                entity_to_track.setdefault(entity_id, track_id)
                track_to_entity.setdefault(track_id, entity_id)

                # entity_id ↔ mask_id 确定性映射稳定（mask-primary 标注必须一致）
                mask_id = obj.get("mask_id")
                if isinstance(mask_id, int):
                    try:
                        if mask_id != entity_id_to_mask_id(entity_id):
                            errors.append(
                                f"{label} mask_id={mask_id} 与 entity_id={entity_id} 的"
                                f"确定性映射不符（期望 {entity_id_to_mask_id(entity_id)}）"
                            )
                    except (ValueError, TypeError):
                        pass

                in_frame = obj.get("in_frame")
                if not isinstance(in_frame, bool):
                    errors.append(f"{label} in_frame 缺失或非布尔")
                    continue
                if in_frame:
                    _check_bbox(errors, obj, label, width, height)

    # ── Instance-ID Mask 校验（存在 mask/ 时生效）────────────────
    if (cam_dir / "mask").exists():
        _validate_mask_dir(errors, cam_label, cam_dir, width, height)

    # ── MOT gt.txt / seqinfo.ini ─────────────────────────────────
    gt_path = cam_dir / "gt" / "gt.txt"
    if gt_path.exists():
        with open(gt_path, encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                parts = line.strip().split(",")
                if len(parts) != 9:
                    errors.append(f"[{cam_label}] gt.txt 第 {line_no} 行字段数 != 9: {line.strip()!r}")
                    continue
                frame_f, tid, x, y, w, h, conf, cls, vis = parts
                for idx, val in enumerate((frame_f, tid, x, y, w, h, conf, cls, vis)):
                    try:
                        float(val)
                    except ValueError:
                        errors.append(f"[{cam_label}] gt.txt 第 {line_no} 行第 {idx + 1} 列不是数字: {val!r}")
                try:
                    if int(frame_f) < 1:
                        errors.append(f"[{cam_label}] gt.txt 第 {line_no} 行 frame < 1")
                    if int(tid) < 1:
                        errors.append(f"[{cam_label}] gt.txt 第 {line_no} 行 track_id < 1")
                    if int(w) <= 0 or int(h) <= 0:
                        errors.append(f"[{cam_label}] gt.txt 第 {line_no} 行 bbox 宽/高非正")
                    if int(x) < 0 or int(y) < 0:
                        errors.append(f"[{cam_label}] gt.txt 第 {line_no} 行 x/y 为负")
                    if width > 0 and int(x) + int(w) > width:
                        errors.append(f"[{cam_label}] gt.txt 第 {line_no} 行 x+w 超出图像宽")
                    if height > 0 and int(y) + int(h) > height:
                        errors.append(f"[{cam_label}] gt.txt 第 {line_no} 行 y+h 超出图像高")
                except ValueError:
                    pass
    else:
        errors.append(f"[{cam_label}] 缺少 gt/gt.txt（如需 MOT 导出）")

    if not (cam_dir / "seqinfo.ini").exists():
        errors.append(f"[{cam_label}] 缺少 seqinfo.ini")

    return errors


def _check_bbox(errors: List[str], obj: dict, label: str, width: int, height: int) -> None:
    """校验 in_frame 对象的 bbox 合法。"""
    for key in ("bbox_xywh", "bbox_xyxy"):
        box = obj.get(key)
        if not isinstance(box, list) or len(box) != 4:
            errors.append(f"{label} {key} 缺失或不是 4 元素列表")
            continue
        for i, v in enumerate(box):
            if not isinstance(v, (int, float)) or not math.isfinite(v):
                errors.append(f"{label} {key}[{i}] 不是有限数字: {v!r}")
    box = obj.get("bbox_xywh")
    if isinstance(box, list) and len(box) == 4:
        try:
            x, y, w, h = (float(v) for v in box)
            if w <= 0 or h <= 0:
                errors.append(f"{label} bbox_xywh 宽/高非正: {box}")
            if x < 0 or y < 0:
                errors.append(f"{label} bbox_xywh x/y 为负: {box}")
            if width > 0 and x + w > width + 1e-6:
                errors.append(f"{label} bbox_xywh x+w 超出图像宽: {box}")
            if height > 0 and y + h > height + 1e-6:
                errors.append(f"{label} bbox_xywh y+h 超出图像高: {box}")
        except (TypeError, ValueError):
            errors.append(f"{label} bbox_xywh 含非法值: {box!r}")


def _frame_numbers(dir_path: Path) -> set:
    """从目录里的 PNG 文件名解析帧号集合。"""
    if not dir_path.exists():
        return set()
    nums = set()
    for p in dir_path.glob("*.png"):
        digits = "".join(ch for ch in p.stem if ch.isdigit())
        if digits:
            nums.add(int(digits))
    return nums


def _read_mask_config(cam_dir: Path) -> dict:
    """读取 annotate-masks 写入的 mask_config.json（解码参数）。"""
    cfg_path = cam_dir / "mask_config.json"
    if cfg_path.exists():
        try:
            return json.loads(cfg_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _validate_mask_dir(
    errors: List[str], cam_label: str, cam_dir: Path, width: int, height: int
) -> None:
    """校验 Instance-ID Mask：RGB/mask 帧一一对应、分辨率、合法 ID、bbox==mask、YOLO。"""
    from instance_mask import (
        load_mask_array,
        mask_to_bbox,
        quantize_mask_pixels,
    )

    mask_dir = cam_dir / "mask"
    decode = _read_mask_config(cam_dir)
    channel = decode.get("mask_channel", "r")
    id_scale = float(decode.get("id_scale", 1.0))
    id_offset = float(decode.get("id_offset", 0.0))

    mask_frames = _frame_numbers(mask_dir)
    if not mask_frames:
        errors.append(f"[{cam_label}] mask/ 目录为空")
        return

    # RGB 与 mask 帧一一对应
    img_frames = _frame_numbers(cam_dir / "img1")
    if img_frames and img_frames != mask_frames:
        only_img = sorted(img_frames - mask_frames)[:5]
        only_mask = sorted(mask_frames - img_frames)[:5]
        errors.append(
            f"[{cam_label}] img1/ 与 mask/ 帧号集合不一致（仅 img1: {only_img}，仅 mask: {only_mask}）"
        )

    ann_path = cam_dir / "annotations.jsonl"
    if not ann_path.exists():
        return
    with open(ann_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            frame = json.loads(line)
            fi = frame.get("frame_index")
            if not isinstance(fi, int):
                continue
            mask_path = mask_dir / f"{fi:06d}.png"
            if not mask_path.exists():
                continue
            mask_img = load_mask_array(mask_path, channel)
            mh, mw = mask_img.shape
            if (mw, mh) != (width, height):
                errors.append(
                    f"[{cam_label}] mask {fi} 分辨率 {mw}x{mh} != camera {width}x{height}"
                )
            quantized = quantize_mask_pixels(mask_img, id_scale, id_offset)
            vals = set(quantized.reshape(-1).tolist())
            legal = {0} | set(range(1, 12))  # 背景 0 + 合法实体 mask_id 1..11
            illegal = sorted(v for v in vals if v not in legal)
            if illegal:
                errors.append(
                    f"[{cam_label}] mask {fi} 含非法实例 ID 值: {illegal[:10]}"
                )
            for obj in frame.get("objects", []):
                entity_id = obj.get("entity_id")
                mask_id = obj.get("mask_id")
                try:
                    expected = entity_id_to_mask_id(entity_id)
                except (ValueError, TypeError):
                    continue
                if isinstance(mask_id, int) and mask_id != expected:
                    errors.append(
                        f"[{cam_label}] frame {fi} {entity_id} mask_id={mask_id} != 期望 {expected}"
                    )
                if obj.get("bbox_source") == "instance_mask" and isinstance(mask_id, int):
                    bbox = mask_to_bbox(quantized == mask_id)
                    xyxy = obj.get("bbox_xyxy")
                    if bbox is None or not isinstance(xyxy, list) or len(xyxy) != 4:
                        errors.append(
                            f"[{cam_label}] frame {fi} {entity_id} bbox_source=instance_mask"
                            f" 但 mask 空或 bbox 缺失"
                        )
                        continue
                    for a, b in zip(bbox, xyxy):
                        if abs(float(a) - float(b)) > 0.5:
                            errors.append(
                                f"[{cam_label}] frame {fi} {entity_id} bbox_xyxy 与 mask"
                                f" min/max 不一致: mask={[round(v,2) for v in bbox]}, ann={xyxy}"
                            )
                            break

    _validate_yolo(errors, cam_label, cam_dir, "det")
    _validate_yolo(errors, cam_label, cam_dir, "seg")


def _validate_yolo(errors: List[str], cam_label: str, cam_dir: Path, sub: str) -> None:
    """校验 YOLO 标签：坐标 ∈ [0,1]；det 行 5 值且 w/h>0；seg 行偶数点。"""
    label_dir = cam_dir / "labels" / sub
    if not label_dir.exists():
        return  # 未导出 YOLO 属可选，不算错误
    for p in sorted(label_dir.glob("*.txt")):
        for line_no, line in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
            parts = line.strip().split()
            if not parts:
                continue
            try:
                floats = [float(v) for v in parts[1:]]
            except ValueError:
                errors.append(f"[{cam_label}] {p.name} 第 {line_no} 行含非数字: {line.strip()!r}")
                continue
            if sub == "det":
                if len(parts) != 5:
                    errors.append(
                        f"[{cam_label}] {p.name} 第 {line_no} 行 det 应 5 值"
                        f"（class cx cy w h），实际 {len(parts)} 值"
                    )
                    continue
                cx, cy, w, h = floats
                if w <= 0 or h <= 0:
                    errors.append(f"[{cam_label}] {p.name} 第 {line_no} 行 w/h 非正")
            else:
                if len(floats) % 2 != 0:
                    errors.append(
                        f"[{cam_label}] {p.name} 第 {line_no} 行 seg 点数为奇数（应成对 x/y）"
                    )
            for v in floats:
                if not (0.0 <= v <= 1.0):
                    errors.append(
                        f"[{cam_label}] {p.name} 第 {line_no} 行坐标越界 [0,1]: {v}"
                    )
                    break


def _report(errors: List[str]) -> None:
    """打印错误列表。"""
    if not errors:
        return
    print(f"ANNOTATION VALIDATOR: Found {len(errors)} error(s)")
    for err in errors[:50]:
        print(f"  ERROR: {err}")
    if len(errors) > 50:
        print(f"  ... and {len(errors) - 50} more errors")


def validate_annotation_dir(annotation_dir: Path) -> int:
    """验证一个标注输出目录。

    支持两种层级：<root>/<camera>/ 或 <root>/<episode_id>/<camera>/。
    通过递归查找包含 camera.json 的目录来定位 camera 子目录。

    Returns:
        0 表示通过，1 表示失败。
    """
    errors: List[str] = []
    camera_dirs = sorted(d.parent for d in annotation_dir.rglob("camera.json"))
    if not camera_dirs:
        errors.append(f"目录 {annotation_dir} 下没有 camera 子目录（缺少 camera.json）")
    for cam_dir in camera_dirs:
        errors += _validate_camera(cam_dir)
    _report(errors)
    if not errors:
        print(f"ANNOTATION VALIDATOR: {annotation_dir} PASSED all checks")
        return 0
    return 1
