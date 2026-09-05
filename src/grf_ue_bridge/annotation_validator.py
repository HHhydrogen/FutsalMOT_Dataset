"""验证导出的 CV 标注目录。纯 Python，不依赖 unreal。

内部 annotation 约定（见 ue/annotation_exporter.py 与 README）：
  - 每个 camera 子目录含 camera.json、annotations.jsonl、gt/gt.txt、seqinfo.ini。
  - annotations.jsonl 每行：episode_id / camera_id / frame_index（1 基）/
    source_step / time_seconds / objects[]。
  - object 字段：entity_id / track_id / class / team / role / is_goalkeeper /
    world_position / raw_bbox_xywh / raw_bbox_xyxy / bbox_xywh / bbox_xyxy /
    in_frame / truncated / visibility。in_frame=true 时 bbox_xywh 必须是合法 bbox。

可见像素 GT 与几何投影 GT 语义分离（本校验器强制执行）：
  - bbox_source="instance_mask"：bbox 由 mask 可见像素派生 → in_frame=true、
    visible_pixel_count>0。
  - bbox_source="not_visible"：mask 存在但实体像素为 0 → in_frame=false、
    visible_pixel_count=0、bbox_xyxy/xywh=null，几何只保留在 geometry_bbox_*。
  - 无 bbox_source / bbox_source="geometry"：legacy 几何标注（无 mask 数据时保留）。
  - 不可见对象不得进入 MOT gt.txt 与 YOLO 标签（交叉校验）。
"""

import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# 把仓库的 ue/ 目录加入 sys.path（与 tests/conftest.py 一致），以便 import 纯模块
_UE_DIR = Path(__file__).resolve().parent.parent.parent / "ue"
if str(_UE_DIR) not in sys.path:
    sys.path.insert(0, str(_UE_DIR))

from annotation_utils import (  # noqa: E402
    BBOX_SOURCE_GEOMETRY,
    BBOX_SOURCE_INSTANCE_MASK,
    BBOX_SOURCE_NOT_VISIBLE,
    entity_id_to_mask_id,
)
from .validation_result import CheckStatus, ValidationResult

# bbox_source 合法取值（None = legacy 几何，无 mask 数据时保留）
_BBOX_SOURCES = (
    BBOX_SOURCE_GEOMETRY,
    BBOX_SOURCE_INSTANCE_MASK,
    BBOX_SOURCE_NOT_VISIBLE,
)

# quick 验证级别下每相机最多重算的 mask 帧数（均匀抽样，保证确定性）
_QUICK_MASK_SAMPLE = 10


def _validate_camera(
    cam_dir: Path,
    validation_level: str = "full",
    require_mot: bool = True,
    require_mask: bool = True,
    require_yolo_det: bool = True,
    require_yolo_seg: bool = True,
) -> List[str]:
    """验证单个 camera 子目录。返回错误列表（空表示通过）。

    validation_level：full=完整逐帧重算；quick=结构检查 + 抽样重算 mask 帧。
    """
    errors: List[str] = []
    cam_label = cam_dir.name

    # ── camera.json ──────────────────────────────────────────────
    cam_json_path = cam_dir / "camera.json"
    if not cam_json_path.exists():
        errors.append(f"[{cam_label}] 缺少 camera.json")
        return errors
    try:
        cam = json.loads(cam_json_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as e:
        errors.append(f"[{cam_label}] camera.json 不是合法 JSON: {e}")
        return errors
    if not isinstance(cam, dict):
        errors.append(f"[{cam_label}] camera.json 顶层必须是 JSON 对象")
        return errors

    intr = cam.get("intrinsics")
    if not isinstance(intr, dict):
        errors.append(f"[{cam_label}] camera.json 缺少 intrinsics")
        intr = {}
    for key in ("width", "height", "fx", "fy", "cx", "cy"):
        if not isinstance(intr.get(key), (int, float)):
            errors.append(f"[{cam_label}] intrinsics.{key} 缺失或非数字")
    extr = cam.get("extrinsics")
    if not isinstance(extr, dict):
        errors.append(f"[{cam_label}] camera.json 缺少 extrinsics")
        extr = {}

    try:
        width = int(intr.get("width", 0))
        height = int(intr.get("height", 0))
    except (TypeError, ValueError, OverflowError):
        width = height = 0
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
    # frame_index → {track_id: obj}（供 MOT 交叉校验）；frame_index → (n_players, n_balls)
    frame_track_objects: Dict[int, Dict[int, dict]] = {}
    yolo_counts_by_frame: Dict[int, Tuple[int, int]] = {}

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

            if isinstance(frame_index, int):
                yolo_counts_by_frame[frame_index] = _frame_instance_counts(objects)

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
                if isinstance(frame_index, int):
                    frame_track_objects.setdefault(frame_index, {})[track_id] = obj
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
                _check_semantics(errors, obj, label, width, height)
                if in_frame:
                    _check_bbox(errors, obj, label, width, height)

    # ── Instance-ID Mask 校验（存在 mask/ 时生效）────────────────
    if require_mask and (cam_dir / "mask").exists():
        sample = None if validation_level != "quick" else _QUICK_MASK_SAMPLE
        _validate_mask_dir(errors, cam_label, cam_dir, width, height, sample_frames=sample)

    # ── YOLO 标签校验（行数须与 instance_mask 可见对象一致，不可见对象不得进入）──
    if require_yolo_det:
        _validate_yolo(errors, cam_label, cam_dir, "det", yolo_counts_by_frame)
    if require_yolo_seg:
        _validate_yolo(errors, cam_label, cam_dir, "seg", yolo_counts_by_frame)

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
                values = (frame_f, tid, x, y, w, h, conf, cls, vis)
                valid_numeric = True
                for idx, val in enumerate(values):
                    try:
                        parsed = float(val)
                    except ValueError:
                        errors.append(f"[{cam_label}] gt.txt 第 {line_no} 行第 {idx + 1} 列不是数字: {val!r}")
                        valid_numeric = False
                        continue
                    if not math.isfinite(parsed):
                        errors.append(
                            f"[{cam_label}] gt.txt 第 {line_no} 行第 {idx + 1} 列不是有限数字: {val!r}"
                        )
                        valid_numeric = False
                if not valid_numeric:
                    continue

                integer_values = {}
                valid_integers = True
                for idx in (0, 1, 2, 3, 4, 5, 7):
                    try:
                        integer_values[idx] = int(values[idx])
                    except (ValueError, OverflowError):
                        errors.append(
                            f"[{cam_label}] gt.txt 第 {line_no} 行第 {idx + 1} 列不是整数: {values[idx]!r}"
                        )
                        valid_integers = False
                if not valid_integers:
                    continue

                frame_i = integer_values[0]
                track_i = integer_values[1]
                x_i = integer_values[2]
                y_i = integer_values[3]
                w_i = integer_values[4]
                h_i = integer_values[5]
                if frame_i < 1:
                    errors.append(f"[{cam_label}] gt.txt 第 {line_no} 行 frame < 1")
                if track_i < 1:
                    errors.append(f"[{cam_label}] gt.txt 第 {line_no} 行 track_id < 1")
                if w_i <= 0 or h_i <= 0:
                    errors.append(f"[{cam_label}] gt.txt 第 {line_no} 行 bbox 宽/高非正")
                if x_i < 0 or y_i < 0:
                    errors.append(f"[{cam_label}] gt.txt 第 {line_no} 行 x/y 为负")
                if width > 0 and x_i + w_i > width:
                    errors.append(f"[{cam_label}] gt.txt 第 {line_no} 行 x+w 超出图像宽")
                if height > 0 and y_i + h_i > height:
                    errors.append(f"[{cam_label}] gt.txt 第 {line_no} 行 y+h 超出图像高")
                # 交叉校验：不可见对象（not_visible / visible_pixel_count==0）不得进入 MOT
                gt_obj = frame_track_objects.get(frame_i, {}).get(track_i)
                if gt_obj is not None and (
                    gt_obj.get("in_frame") is False
                    or gt_obj.get("bbox_source") == BBOX_SOURCE_NOT_VISIBLE
                    or gt_obj.get("visible_pixel_count") == 0
                ):
                    errors.append(
                        f"[{cam_label}] gt.txt 第 {line_no} 行不可见对象进入 MOT: "
                        f"frame={frame_f} track={tid}（bbox_source={gt_obj.get('bbox_source')!r}, "
                        f"visible_pixel_count={gt_obj.get('visible_pixel_count')!r}）"
                    )
    elif require_mot:
        errors.append(f"[{cam_label}] 缺少 gt/gt.txt（如需 MOT 导出）")

    if require_mot and not (cam_dir / "seqinfo.ini").exists():
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
            value = json.loads(cfg_path.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise ValueError("顶层必须是 JSON 对象")
            return value
        except (json.JSONDecodeError, OSError, UnicodeError):
            pass
    return {}


def _validate_mask_dir(
    errors: List[str],
    cam_label: str,
    cam_dir: Path,
    width: int,
    height: int,
    sample_frames: Optional[int] = None,
) -> None:
    """校验 Instance-ID Mask：RGB/mask 帧一一对应、分辨率、合法 ID、bbox==mask、YOLO。

    sample_frames 非 None 时只对该数量的帧做昂贵的逐帧 mask 重算（均匀抽样，
    确定性），但帧集合对应关系（glob 文件名）仍全量检查——即 quick 验证模式。
    """
    from instance_mask import (
        load_mask_array,
        mask_to_bbox,
        quantize_mask_pixels,
    )

    mask_dir = cam_dir / "mask"
    try:
        decode = _read_mask_config(cam_dir)
    except ValueError as exc:
        errors.append(f"[{cam_label}] mask_config.json 结构非法: {exc}")
        return
    channel = decode.get("mask_channel", "r")
    try:
        id_scale = float(decode.get("id_scale", 1.0))
        id_offset = float(decode.get("id_offset", 0.0))
    except (TypeError, ValueError, OverflowError) as exc:
        errors.append(f"[{cam_label}] mask_config.json 解码参数非法: {exc}")
        return

    mask_frames = _frame_numbers(mask_dir)
    if not mask_frames:
        errors.append(f"[{cam_label}] mask/ 目录为空")
        return

    # RGB 与 mask 帧一一对应（结构检查，全量）
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
        lines = [ln.strip() for ln in f if ln.strip()]
    n = len(lines)
    if sample_frames is not None and n > sample_frames:
        step = max(1, math.ceil(n / sample_frames))
        sampled_idx = list(range(0, n, step))
    else:
        sampled_idx = list(range(n))
    for li in sampled_idx:
        try:
            frame = json.loads(lines[li])
        except (UnicodeError, json.JSONDecodeError):
            continue  # 行级 JSON 错误由主循环报告
        if not isinstance(frame, dict):
            errors.append(f"[{cam_label}] annotations.jsonl 第 {li + 1} 行不是 JSON 对象")
            continue
        fi = frame.get("frame_index")
        if not isinstance(fi, int):
            continue
        mask_path = mask_dir / f"{fi:06d}.png"
        if not mask_path.exists():
            continue
        try:
            mask_img = load_mask_array(mask_path, channel)
        except Exception as exc:
            errors.append(f"[{cam_label}] mask {fi:06d}.png 读取/解码失败: {exc}")
            continue
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
        objects = frame.get("objects", [])
        if not isinstance(objects, list):
            errors.append(
                f"[{cam_label}] annotations.jsonl 第 {li + 1} 行 objects 必须是列表"
            )
            continue
        for oi, obj in enumerate(objects):
            if not isinstance(obj, dict):
                errors.append(
                    f"[{cam_label}] annotations.jsonl 第 {li + 1} 行 objects[{oi}] 不是对象"
                )
                continue
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
            if obj.get("bbox_source") == BBOX_SOURCE_INSTANCE_MASK and isinstance(mask_id, int):
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
            elif obj.get("bbox_source") == BBOX_SOURCE_NOT_VISIBLE and isinstance(mask_id, int):
                # 反向一致性：标记为不可见的实体，mask 里必须真的没有可见像素
                if (quantized == mask_id).any():
                    errors.append(
                        f"[{cam_label}] frame {fi} {entity_id} bbox_source=not_visible"
                        f" 但 mask 含可见像素（{int((quantized == mask_id).sum())} px）"
                    )


def _validate_yolo(
    errors: List[str],
    cam_label: str,
    cam_dir: Path,
    sub: str,
    counts_by_frame: Dict[int, Tuple[int, int]],
) -> None:
    """校验 YOLO 标签：坐标 ∈ [0,1]；det 行 5 值且 w/h>0；seg 行偶数点。

    额外做「不可见对象不得进入 YOLO」的一致性检查：每帧行数必须落在
    [instance_mask 球员数, instance_mask 球员数 + instance_mask 球数] 区间
    （球是否导出取决于 include_ball，故只做区间而非精确匹配）。
    """
    label_dir = cam_dir / "labels" / sub
    if not label_dir.exists():
        return  # 未导出 YOLO 属可选，不算错误
    for p in sorted(label_dir.glob("*.txt")):
        digits = "".join(ch for ch in p.stem if ch.isdigit())
        fi = int(digits) if digits else None
        n_players, n_balls = counts_by_frame.get(fi, (0, 0))
        n_lines = 0
        for line_no, line in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
            parts = line.strip().split()
            if not parts:
                continue
            n_lines += 1
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

        # 一致性：行数必须在 [instance_mask 球员数, 球员数+球数] 内（球视 include_ball
        # 而定）。超出上限 = 不可见对象（not_visible / legacy 几何）泄漏进 YOLO。
        if n_lines < n_players or n_lines > n_players + n_balls:
            errors.append(
                f"[{cam_label}] {p.name} 行数 {n_lines} 与 instance_mask 可见对象不符"
                f"（players={n_players}, balls={n_balls}）——不可见对象不得进入 YOLO"
            )


def _frame_instance_counts(objects: List[dict]) -> Tuple[int, int]:
    """统计该帧 instance_mask 可见对象数，返回 (球员数, 球数)。"""
    n_players = 0
    n_balls = 0
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        if obj.get("bbox_source") != BBOX_SOURCE_INSTANCE_MASK:
            continue
        if obj.get("class") == "ball":
            n_balls += 1
        else:
            n_players += 1
    return n_players, n_balls


def _check_semantics(errors: List[str], obj: dict, label: str, width: int, height: int) -> None:
    """校验「可见像素 GT」与「几何投影 GT」的语义分离（规则见模块 docstring）。

    - bbox_source 取值必须合法（None / geometry / instance_mask / not_visible）。
    - instance_mask → in_frame=true 且 visible_pixel_count>0。
    - visible_pixel_count==0 → 可见 bbox/raw bbox/segmentation 必须为 null，
      且 bbox_source 应为 not_visible（不可回填成可见的几何 bbox）。
    - not_visible → in_frame=false、visible_pixel_count==0。
    - in_frame=false → bbox_xyxy/xywh 必须为 null。
    - geometry_bbox_* 为独立字段：存在时须是合法 4 元素 bbox（数值上不受可见 bbox 约束）。
    """
    src = obj.get("bbox_source")
    if src is not None and src not in _BBOX_SOURCES:
        errors.append(f"{label} bbox_source 取值非法: {src!r}")

    vpc = obj.get("visible_pixel_count")
    if vpc is not None and not isinstance(vpc, int):
        errors.append(f"{label} visible_pixel_count 非整数: {vpc!r}")

    if src == BBOX_SOURCE_INSTANCE_MASK:
        if obj.get("in_frame") is not True:
            errors.append(f"{label} bbox_source=instance_mask 但 in_frame 非 true")
        if not (isinstance(vpc, int) and vpc > 0):
            errors.append(f"{label} bbox_source=instance_mask 但 visible_pixel_count 非正数")

    if isinstance(vpc, int) and vpc == 0:
        if src != BBOX_SOURCE_NOT_VISIBLE:
            errors.append(
                f"{label} visible_pixel_count=0 但 bbox_source={src!r}"
                f"（应为 {BBOX_SOURCE_NOT_VISIBLE!r}）"
            )
        for key in ("bbox_xyxy", "bbox_xywh", "raw_bbox_xyxy", "raw_bbox_xywh"):
            if obj.get(key) is not None:
                errors.append(
                    f"{label} visible_pixel_count=0 但 {key} 非 null"
                    f"（几何 bbox 只应保存在 geometry_bbox_*）"
                )
        if obj.get("segmentation") is not None:
            errors.append(f"{label} visible_pixel_count=0 但 segmentation 非 null")

    if src == BBOX_SOURCE_NOT_VISIBLE:
        if obj.get("in_frame") is not False:
            errors.append(f"{label} bbox_source=not_visible 但 in_frame 非 false")
        if not (isinstance(vpc, int) and vpc == 0):
            errors.append(f"{label} bbox_source=not_visible 但 visible_pixel_count 非 0")

    # in_frame=false → 可见 bbox 必须为 null（覆盖几何离屏与 not_visible 两种情形）
    if obj.get("in_frame") is False:
        for key in ("bbox_xyxy", "bbox_xywh"):
            if obj.get(key) is not None:
                errors.append(f"{label} in_frame=false 但 {key} 非 null")

    # geometry_bbox_* 独立字段：存在时须为合法 4 元素 bbox
    for key in ("geometry_bbox_xyxy", "geometry_bbox_xywh"):
        gb = obj.get(key)
        if gb is None:
            continue
        if not isinstance(gb, list) or len(gb) != 4:
            errors.append(f"{label} {key} 非 4 元素列表: {gb!r}")
            continue
        for i, v in enumerate(gb):
            if not isinstance(v, (int, float)) or not math.isfinite(v):
                errors.append(f"{label} {key}[{i}] 不是有限数字: {v!r}")


def _report(errors: List[str]) -> None:
    """打印错误列表。"""
    if not errors:
        return
    print(f"ANNOTATION VALIDATOR: Found {len(errors)} error(s)")
    for err in errors[:50]:
        print(f"  ERROR: {err}")
    if len(errors) > 50:
        print(f"  ... and {len(errors) - 50} more errors")


def _validate_camera_task(task: tuple) -> List[str]:
    """进程池 worker：验证单个 camera（结构校验 + 可选 dataset regression）。

    模块级函数（可 pickle，Windows spawn 安全）。返回错误列表。
    """
    (
        cam_str,
        validation_level,
        require_mot,
        require_mask,
        require_yolo_det,
        require_yolo_seg,
    ) = task
    cam_dir = Path(cam_str)
    errors = _validate_camera(
        cam_dir,
        validation_level=validation_level,
        require_mot=require_mot,
        require_mask=require_mask,
        require_yolo_det=require_yolo_det,
        require_yolo_seg=require_yolo_seg,
    )
    if validation_level != "quick":
        try:
            from .dataset_regression import _validate_camera as _reg_camera
            errors += _reg_camera(
                cam_dir,
                require_mot=require_mot,
                require_mask=require_mask,
                require_yolo_det=require_yolo_det,
                require_yolo_seg=require_yolo_seg,
            )
        except Exception as e:
            errors.append(f"[{cam_dir.name}] DATASET REGRESSION: 执行异常: {e}")
    return errors


def _resolve_workers(workers: int, n_tasks: int) -> int:
    """解析 --workers：0=自动（min(任务数, max(1, cpu_count//2))），1=串行，>1=指定。"""
    if workers == 0:
        return min(max(1, n_tasks), max(1, (os.cpu_count() or 1) // 2))
    return max(1, workers)


def validate_annotation_dir(
    annotation_dir: Path,
    workers: int = 0,
    validation_level: str = "full",
    require_mot: bool = True,
    require_mask: bool = True,
    require_yolo_det: bool = True,
    require_yolo_seg: bool = True,
) -> int:
    """验证一个标注输出目录。

    支持两种层级：<root>/<camera>/ 或 <root>/<episode_id>/<camera>/。
    通过递归查找包含 camera.json 的目录来定位 camera 子目录。

    除逐字段语义/掩码校验外，还会运行端到端 dataset regression（RGB/mask/annotation
    帧数、分辨率、MOT/YOLO 重新派生比对、多连通域 quality gate 复验），作为最终验收。

    workers：0=自动，1=串行，>1=相机级多进程并行（结果与串行一致）。
    require_mot：是否要求每个 camera 都存在有效 gt/gt.txt 和 seqinfo.ini；关闭时只
                 跳过缺失文件，已存在 gt/gt.txt 仍会验证。
    validation_level：
      full  —— 完整语义：逐帧 mask 重算、重新派生并比较 MOT/YOLO、完整检查全部帧
                （默认，保持现有行为）。
      quick —— 结构检查（文件存在/帧数/文件名对应/mask ID 合法/bbox 范围/track
                映射/MOT/YOLO 语法）+ 每相机抽样有限帧的 mask 重算；跳过昂贵的
                全量重派生。

    Returns:
        0 表示通过，1 表示失败。
    """
    return validate_annotation_result(
        annotation_dir,
        workers=workers,
        validation_level=validation_level,
        require_mot=require_mot,
        require_mask=require_mask,
        require_yolo_det=require_yolo_det,
        require_yolo_seg=require_yolo_seg,
    ).exit_code


def validate_annotation_result(
    annotation_dir: Path,
    workers: int = 0,
    validation_level: str = "full",
    require_mot: bool = True,
    require_mask: bool = True,
    require_yolo_det: bool = True,
    require_yolo_seg: bool = True,
) -> ValidationResult:
    """验证标注目录并返回结构化结果，同时保留旧 CLI 的打印行为。"""
    if validation_level not in ("full", "quick"):
        raise ValueError(f"未知 validation_level: {validation_level!r}（可选 full/quick）")

    errors: List[str] = []
    camera_dirs = sorted(d.parent for d in annotation_dir.rglob("camera.json"))
    if not camera_dirs:
        errors.append(f"目录 {annotation_dir} 下没有 camera 子目录（缺少 camera.json）")

    nworkers = _resolve_workers(workers, len(camera_dirs))
    if nworkers <= 1 or len(camera_dirs) <= 1:
        camera_errors = [
            _validate_camera_task(
                (
                    str(cam_dir),
                    validation_level,
                    require_mot,
                    require_mask,
                    require_yolo_det,
                    require_yolo_seg,
                )
            )
            for cam_dir in camera_dirs
        ]
    else:
        tasks = [
            (
                str(cam_dir),
                validation_level,
                require_mot,
                require_mask,
                require_yolo_det,
                require_yolo_seg,
            )
            for cam_dir in camera_dirs
        ]
        with ProcessPoolExecutor(max_workers=nworkers) as ex:
            camera_errors = list(ex.map(_validate_camera_task, tasks))
    for cam_errors in camera_errors:
        errors += cam_errors

    result = ValidationResult(errors=list(errors))
    mot_missing = any(
        not (cam_dir / "gt" / "gt.txt").exists() for cam_dir in camera_dirs
    )
    mot_sidecar_missing = any(
        not (cam_dir / "seqinfo.ini").exists() for cam_dir in camera_dirs
    )
    mot_errors = [
        error
        for error in errors
        if "gt.txt" in error or "seqinfo.ini" in error
    ]
    if mot_errors:
        result.add_check(
            "mot_export",
            status=CheckStatus.FAILED,
            required=require_mot,
            message=mot_errors[0],
        )
    elif (mot_missing or mot_sidecar_missing) and not require_mot:
        result.add_check(
            "mot_export",
            status=CheckStatus.SKIPPED,
            required=False,
            message="未配置 MOT 导出，且 gt/gt.txt 或 seqinfo.ini 不完整",
        )
    else:
        result.add_check(
            "mot_export",
            status=CheckStatus.PASSED,
            required=require_mot,
        )

    result.add_check(
        "annotation",
        status=CheckStatus.FAILED if result.errors else CheckStatus.PASSED,
        required=True,
        message=result.errors[0] if result.errors else None,
    )
    result.finalize()
    _report(result.errors)
    if not result.errors:
        print(f"ANNOTATION VALIDATOR: {annotation_dir} PASSED all checks")
    return result
