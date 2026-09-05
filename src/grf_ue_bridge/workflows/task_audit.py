#!/usr/bin/env python
"""soak episode 完整性审计。

对一次完整渲染+后处理的相机数据集目录做只读审计，输出 JSON + Markdown 双报告，
并用退出码标识是否通过。检查维度：

  - 相机目录数量 / 期望帧数
  - render/（RGB 原始）、render_mask/（Object ID EXR 原始）数量与目标帧覆盖
  - img1/、mask/、annotations.jsonl、labels/det/、labels/seg/、gt/gt.txt 数量
  - 缺帧、重复帧、零字节文件
  - 跨相机时间同步（time_seconds / source_step / episode_id / track_id / mask_id）
  - camera.json 标定合法性（分辨率一致、内参有限且为正、外参有限、相机不重复）
  - render_summary.json 状态与每相机 ok
  - 可选：进程内运行 validate-annotations（quick/full）

CLI 入口：`grf-ue task audit <task>`（推荐）；或 `python -m grf_ue_bridge.workflows.task_audit --input ...`。

退出码：0 = 全部通过；1 = 存在任何失败项（缺帧/重复/零字节/同步/映射/标定/数量不符）。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from grf_ue_bridge.validation_result import CheckStatus, ValidationResult

# 实体 → track_id / mask_id 的稳定映射（与 annotation_utils 一致，本地复述避免依赖）
TRACK_MAP = {f"L{i}": i + 1 for i in range(5)}
TRACK_MAP.update({f"R{i}": i + 6 for i in range(5)})
TRACK_MAP["BALL"] = 100
MASK_MAP = {f"L{i}": i + 1 for i in range(5)}
MASK_MAP.update({f"R{i}": i + 6 for i in range(5)})
MASK_MAP["BALL"] = 11


def _is_finite(v) -> bool:
    return isinstance(v, (int, float)) and math.isfinite(v)


def _as_bool(value) -> bool:
    return str(value).lower() in ("true", "1", "yes", "on")


def _extend_unique(target: List[str], values: Sequence[str]) -> None:
    for value in values:
        if value not in target:
            target.append(value)


def _frame_numbers(files: Sequence[Path]) -> List[int]:
    """从文件名单 stem 数字解析帧号（升序，含重复）。"""
    out = []
    for p in files:
        digits = "".join(ch for ch in p.stem if ch.isdigit())
        if digits:
            out.append(int(digits))
    return sorted(out)


def _counts_and_missing(
    files: Sequence[Path], expected: int
) -> Tuple[int, List[int], List[int]]:
    """返回 (文件数, 缺失编号, 重复编号)。文件名为 {frame_index:06d} 约定。"""
    nums = _frame_numbers(files)
    uniq = sorted(set(nums))
    missing = [i for i in range(1, expected + 1) if i not in uniq]
    dups = sorted({n for n in nums if nums.count(n) > 1})
    return len(nums), missing, dups


def _zero_byte(patterns: Sequence[Path]) -> List[Path]:
    return [p for p in patterns if p.exists() and p.stat().st_size == 0]


def _validate_required_mot(
    path: Path,
    cam_id: str,
    image_width: Optional[int] = None,
    image_height: Optional[int] = None,
) -> List[str]:
    """检查已有 MOT 的基本结构，兼容 audit none 的轻量路径。"""
    errors: List[str] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        return [f"{cam_id}: gt/gt.txt 读取失败: {exc}"]
    for line_no, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        if len(line.strip().split(",")) != 9:
            errors.append(
                f"{cam_id}: gt/gt.txt 第 {line_no} 行字段数 != 9: {line.strip()!r}"
            )
            continue
        fields = line.strip().split(",")
        valid_numeric = True
        for field_no, value in enumerate(fields, start=1):
            try:
                parsed = float(value)
            except ValueError:
                errors.append(
                    f"{cam_id}: gt/gt.txt 第 {line_no} 行第 {field_no} 列不是数字: {value!r}"
                )
                valid_numeric = False
                continue
            if not math.isfinite(parsed):
                errors.append(
                    f"{cam_id}: gt/gt.txt 第 {line_no} 行第 {field_no} 列不是有限数字: {value!r}"
                )
                valid_numeric = False
        if not valid_numeric:
            continue
        for field_no in (1, 2, 3, 4, 5, 6, 8):
            try:
                int(fields[field_no - 1])
            except (ValueError, OverflowError):
                errors.append(
                    f"{cam_id}: gt/gt.txt 第 {line_no} 行第 {field_no} 列不是整数: "
                    f"{fields[field_no - 1]!r}"
                )
        try:
            frame_id = int(fields[0])
            track_id = int(fields[1])
            x = int(fields[2])
            y = int(fields[3])
            width = int(fields[4])
            height = int(fields[5])
        except (ValueError, OverflowError):
            continue
        if frame_id < 1:
            errors.append(f"{cam_id}: gt/gt.txt 第 {line_no} 行 frame_id < 1")
        if track_id < 1:
            errors.append(f"{cam_id}: gt/gt.txt 第 {line_no} 行 track_id < 1")
        if width <= 0 or height <= 0:
            errors.append(
                f"{cam_id}: gt/gt.txt 第 {line_no} 行 bbox 宽/高非正: {width}x{height}"
            )
        if x < 0 or y < 0:
            errors.append(f"{cam_id}: gt/gt.txt 第 {line_no} 行 x/y 为负: ({x},{y})")
        if image_width is not None and x + width > image_width:
            errors.append(
                f"{cam_id}: gt/gt.txt 第 {line_no} 行 x+w 超出图像宽 "
                f"{image_width}: x={x}, w={width}"
            )
        if image_height is not None and y + height > image_height:
            errors.append(
                f"{cam_id}: gt/gt.txt 第 {line_no} 行 y+h 超出图像高 "
                f"{image_height}: y={y}, h={height}"
            )
    return errors


def _read_frame_meta(cam_dir: Path) -> Tuple[Optional[List[dict]], List[str]]:
    """读取 annotations.jsonl 的帧级元数据（不含 objects），失败返回错误列表。"""
    ann = cam_dir / "annotations.jsonl"
    if not ann.exists():
        return None, [f"缺少 annotations.jsonl: {ann}"]
    frames: List[dict] = []
    errors: List[str] = []
    try:
        with open(ann, encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    errors.append(
                        f"annotations.jsonl 第 {line_no} 行 JSON 解析失败: {exc}"
                    )
                    continue
                if not isinstance(obj, dict):
                    errors.append(
                        f"annotations.jsonl 第 {line_no} 行顶层必须是 JSON 对象"
                    )
                    continue
                objects = obj.get("objects", [])
                if not isinstance(objects, list):
                    errors.append(
                        f"annotations.jsonl 第 {line_no} 行 objects 必须是列表，"
                        f"实际为 {type(objects).__name__}"
                    )
                    objects = []
                for object_index, object_value in enumerate(objects):
                    if not isinstance(object_value, dict):
                        errors.append(
                            f"annotations.jsonl 第 {line_no} 行 objects[{object_index}] "
                            "必须是 JSON 对象"
                        )
                frame_index = obj.get("frame_index")
                if (
                    not isinstance(frame_index, int)
                    or isinstance(frame_index, bool)
                    or frame_index < 1
                ):
                    errors.append(
                        f"annotations.jsonl 第 {line_no} 行 frame_index 非法: {frame_index!r}"
                    )
                frames.append({
                    "frame_index": frame_index,
                    "source_step": obj.get("source_step"),
                    "time_seconds": obj.get("time_seconds"),
                    "episode_id": obj.get("episode_id"),
                    "objects": objects,
                })
    except (OSError, UnicodeError) as exc:
        return None, [f"annotations.jsonl 读取失败: {exc}"]
    return frames, errors


def _read_jsonl_rows(path: Path, label: str, errors: List[str]) -> Tuple[int, bool]:
    """读取 JSONL 行数，并返回是否每个非空行都是 JSON 对象。"""
    valid = True
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        errors.append(f"pose: {label} 读取失败: {exc}")
        return 0, False
    count = 0
    for line_no, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        count += 1
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"pose: {label} 第 {line_no} 行 JSON 解析失败: {exc}")
            valid = False
            continue
        if not isinstance(value, dict):
            errors.append(
                f"pose: {label} 第 {line_no} 行顶层必须是 JSON 对象，"
                f"实际为 {type(value).__name__}"
            )
            valid = False
    return count, valid


def _derive_keep_indices(
    frames: List[dict],
    render_fps: int,
    source_step_seconds: Optional[float],
) -> List[int]:
    """根据标注帧的 source_step / time_seconds 推导 MRQ 应保留的渲染帧号。

    source_step_seconds 缺失时从 time_seconds / source_step 反推（取第一个 step>0 的帧）。
    """
    step_sec = source_step_seconds
    if not isinstance(step_sec, (int, float)) or not math.isfinite(step_sec) or step_sec <= 0:
        step_sec = None
    if step_sec is None:
        for fr in frames:
            s = fr.get("source_step")
            t = fr.get("time_seconds")
            if (
                isinstance(s, (int, float))
                and not isinstance(s, bool)
                and s > 0
                and math.isfinite(s)
                and isinstance(t, (int, float))
                and not isinstance(t, bool)
                and math.isfinite(t)
            ):
                step_sec = t / s
                break
    if step_sec is None or not math.isfinite(step_sec) or step_sec <= 0:
        step_sec = 0.1
    keep = []
    for position, fr in enumerate(frames):
        s = fr.get("source_step")
        if (
            not isinstance(s, (int, float))
            or isinstance(s, bool)
            or not math.isfinite(s)
        ):
            frame_index = fr.get("frame_index")
            s = frame_index - 1 if isinstance(frame_index, int) and frame_index >= 1 else position
        keep.append(int(round(s * step_sec * render_fps)))
    return keep


def _load_episode_meta(episode_dir: Path) -> Optional[dict]:
    meta_path = episode_dir / "meta.json"
    if not meta_path.exists():
        return None
    try:
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        return meta if isinstance(meta, dict) else None
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None


def check_camera(
    cam_dir: Path,
    expected_frames: int,
    keep_indices: List[int],
    errors: List[str],
    warns: List[str],
    mask_enabled: bool = True,
    mot_required: bool = False,
    yolo_det_required: bool = False,
    yolo_seg_required: bool = False,
    enforce_mot_files: bool = True,
    render_required: bool = True,
) -> dict:
    """审计单个相机目录，返回统计 dict，并把失败项写入 errors。

    mask_enabled=False（任务未启用 instance_mask）时跳过 mask/render_mask 相关检查。
    mot_required、yolo_*_required 控制对应派生产物是否必须存在。
    enforce_mot_files=False 时由 task-aware annotation validation 负责 MOT 文件门禁，
    避免同一缺失项被审计层重复报告。
    """
    cam_id = cam_dir.name
    img1_dir, mask_dir = cam_dir / "img1", cam_dir / "mask"
    render_dir, rmask_dir = cam_dir / "render", cam_dir / "render_mask"
    det_dir, seg_dir = cam_dir / "labels" / "det", cam_dir / "labels" / "seg"
    gt_txt = cam_dir / "gt" / "gt.txt"
    st: dict = {
        "camera_id": cam_id,
        "ok": True,
        "core_ok": True,
        "mask_required": bool(mask_enabled),
        "mask_present": mask_dir.exists() or rmask_dir.exists(),
        "mask_complete": False,
        "mask_ok": not bool(mask_enabled),
        "mot_required": bool(mot_required),
        "mot_present": gt_txt.exists() or (cam_dir / "seqinfo.ini").exists(),
        "mot_complete": gt_txt.exists() and (cam_dir / "seqinfo.ini").exists(),
        "mot_ok": not bool(mot_required),
        "det_required": bool(yolo_det_required),
        "det_present": det_dir.exists(),
        "det_complete": False,
        "det_ok": not bool(yolo_det_required),
        "seg_required": bool(yolo_seg_required),
        "seg_present": seg_dir.exists(),
        "seg_complete": False,
        "seg_ok": not bool(yolo_seg_required),
    }

    # 标注帧
    frames, errs = _read_frame_meta(cam_dir)
    if errs:
        errors.extend(errs)
        st["annotations_frames"] = 0
        st["core_ok"] = False
        st["ok"] = False
        return st
    st["mask_ok"] = True
    st["mot_ok"] = True
    st["det_ok"] = True
    st["seg_ok"] = True
    ann_idx = [
        fr["frame_index"]
        for fr in frames
        if isinstance(fr.get("frame_index"), int)
        and not isinstance(fr.get("frame_index"), bool)
        and fr.get("frame_index") >= 1
    ]
    st["annotations_frames"] = len(ann_idx)
    if len(ann_idx) != expected_frames:
        errors.append(f"{cam_id}: annotations 帧数 {len(ann_idx)} != 预期 {expected_frames}")
        st["core_ok"] = False
        st["ok"] = False
    missing_ann = [i for i in range(1, expected_frames + 1) if i not in set(ann_idx)]
    dup_ann = sorted({n for n in ann_idx if ann_idx.count(n) > 1})
    if missing_ann:
        errors.append(f"{cam_id}: annotations 缺帧 {missing_ann[:10]}...")
        st["core_ok"] = False
        st["ok"] = False
    if dup_ann:
        errors.append(f"{cam_id}: annotations 重复帧 {dup_ann[:10]}")
        st["core_ok"] = False
        st["ok"] = False

    # 目录产物统计
    img1_files = sorted(img1_dir.glob("*.png")) if img1_dir.exists() else []
    mask_files = sorted(mask_dir.glob("*.png")) if mask_dir.exists() else []
    render_files = sorted(render_dir.rglob("*.png")) if render_dir.exists() else []
    exr_files = sorted(rmask_dir.rglob("*.exr")) if rmask_dir.exists() else []
    det_files = sorted(det_dir.glob("*.txt")) if det_dir.exists() else []
    seg_files = sorted(seg_dir.glob("*.txt")) if seg_dir.exists() else []

    st["render_rgb_png"] = len(render_files)
    st["render_mask_exr"] = len(exr_files)
    st["img1_png"] = len(img1_files)
    st["mask_png"] = len(mask_files)
    st["det_txt"] = len(det_files)
    st["seg_txt"] = len(seg_files)
    st["gt_txt_exists"] = gt_txt.exists()
    if gt_txt.exists():
        try:
            with open(gt_txt, encoding="utf-8") as f:
                st["gt_txt_lines"] = sum(1 for _ in f)
        except OSError as exc:
            errors.append(f"{cam_id}: gt/gt.txt 读取失败: {exc}")
            st["mot_ok"] = False
            st["ok"] = False
    else:
        st["gt_txt_lines"] = 0
    # img1 / mask 缺帧与重复
    _n1, miss_img1, dup_img1 = _counts_and_missing(img1_files, expected_frames)
    st["img1_missing"] = miss_img1
    st["img1_dup"] = dup_img1
    if miss_img1:
        errors.append(f"{cam_id}: img1 缺帧 {miss_img1[:10]}...")
        st["core_ok"] = False
        st["ok"] = False
    if dup_img1:
        errors.append(f"{cam_id}: img1 重复帧 {dup_img1[:10]}")
        st["core_ok"] = False
        st["ok"] = False
    _n2, miss_mask, dup_mask = _counts_and_missing(mask_files, expected_frames)
    st["mask_missing"] = miss_mask
    st["mask_dup"] = dup_mask
    if mask_enabled and not mask_dir.exists():
        errors.append(f"{cam_id}: 缺少 mask/（已要求 instance mask）")
        st["mask_ok"] = False
        st["ok"] = False
    elif mask_enabled and miss_mask:
        errors.append(f"{cam_id}: mask 缺帧 {miss_mask[:10]}...")
        st["mask_ok"] = False
        st["ok"] = False
    if mask_enabled and dup_mask:
        errors.append(f"{cam_id}: mask 重复帧 {dup_mask[:10]}")
        st["mask_ok"] = False
        st["ok"] = False

    # render / render_mask 是否覆盖全部 keep_indices（均为 transient 渲染产物：
    # img1 是 canonical RGB；zero-waste 已删除 render/，render_mask/ 由 cleanup 删除后均不要求存在）
    render_nums = set(_frame_numbers(render_files))
    exr_nums = set(_frame_numbers(exr_files))
    miss_render = sorted(set(keep_indices) - render_nums)
    miss_exr = sorted(set(keep_indices) - exr_nums)
    # 仅当 render/ 仍有 PNG 时才校验其完整性（img1 已保证 RGB 完整性；
    # zero-waste 删除 render/ PNG 后 render_files 为空，跳过）
    if render_required and render_files and miss_render:
        errors.append(f"{cam_id}: render/ 缺少目标帧 {miss_render[:10]}...")
        st["core_ok"] = False
        st["ok"] = False
    if mask_enabled and miss_exr:
        errors.append(f"{cam_id}: render_mask/ 缺少目标 EXR 帧 {miss_exr[:10]}...")
        st["mask_ok"] = False
        st["ok"] = False

    st["mask_complete"] = bool(mask_files) and bool(exr_files) and not miss_mask and not miss_exr

    # 零字节文件
    z = _zero_byte(img1_files)
    if render_required:
        z += _zero_byte(render_files)
    if mask_enabled:
        z += _zero_byte(exr_files) + _zero_byte(mask_files)
    if yolo_det_required:
        z += _zero_byte(det_files)
    if yolo_seg_required:
        z += _zero_byte(seg_files)
    st["zero_byte"] = len(z)
    if z:
        errors.append(f"{cam_id}: 零字节文件 {[str(p.name) for p in z][:10]}")
        st["ok"] = False
        for path in z:
            if mask_dir in path.parents or rmask_dir in path.parents:
                st["mask_ok"] = False
            if det_dir in path.parents:
                st["det_ok"] = False
            if seg_dir in path.parents:
                st["seg_ok"] = False
            if img1_dir in path.parents:
                st["core_ok"] = False

    # labels/det 与 labels/seg 帧号覆盖
    def _detect_nums(files):
        out = set()
        for p in files:
            d = "".join(c for c in p.stem if c.isdigit())
            if d:
                out.add(d)
        return out
    det_nums = _detect_nums(det_files)
    seg_nums = _detect_nums(seg_files)
    st["det_frames"] = len(det_nums)
    st["seg_frames"] = len(seg_nums)
    miss_det = [i for i in range(1, expected_frames + 1) if f"{i:06d}" not in det_nums]
    miss_seg = [i for i in range(1, expected_frames + 1) if f"{i:06d}" not in seg_nums]
    st["det_complete"] = bool(det_files) and not miss_det
    st["seg_complete"] = bool(seg_files) and not miss_seg
    if yolo_det_required and not det_files:
        errors.append(f"{cam_id}: 缺少 labels/det/*.txt（已要求 YOLO detect）")
        st["det_ok"] = False
        st["ok"] = False
    elif yolo_det_required and miss_det:
        errors.append(f"{cam_id}: labels/det 缺帧 {miss_det[:10]}...")
        st["det_ok"] = False
        st["ok"] = False
    if yolo_seg_required and not seg_files:
        errors.append(f"{cam_id}: 缺少 labels/seg/*.txt（已要求 YOLO segment）")
        st["seg_ok"] = False
        st["ok"] = False
    elif yolo_seg_required and miss_seg:
        errors.append(f"{cam_id}: labels/seg 缺帧 {miss_seg[:10]}...")
        st["seg_ok"] = False
        st["ok"] = False

    # 零字节 gt.txt
    if gt_txt.exists() and gt_txt.stat().st_size == 0:
        warns.append(f"{cam_id}: gt/gt.txt 为空（可能所有对象不可见或未导出 MOT）")

    # 数据契约文件存在性；MOT sidecar 只在要求 MOT 时才是门禁。
    for req in ("camera.json", "annotations.jsonl"):
        if not (cam_dir / req).exists():
            errors.append(f"{cam_id}: 缺少 {req}")
            st["core_ok"] = False
            st["ok"] = False
    if mot_required and not gt_txt.exists():
        st["mot_ok"] = False
        st["ok"] = False
        if enforce_mot_files:
            errors.append(f"{cam_id}: 缺少 gt/gt.txt（已要求 MOT 导出）")
    if mot_required and not (cam_dir / "seqinfo.ini").exists():
        st["mot_ok"] = False
        st["ok"] = False
        if enforce_mot_files:
            errors.append(f"{cam_id}: 缺少 seqinfo.ini（已要求 MOT 导出）")
    if gt_txt.exists() and (enforce_mot_files or not mot_required):
        image_width = None
        image_height = None
        camera_json = cam_dir / "camera.json"
        try:
            camera_config = json.loads(camera_json.read_text(encoding="utf-8"))
            if isinstance(camera_config, dict):
                image_width = camera_config.get("image_width")
                image_height = camera_config.get("image_height")
                if not isinstance(image_width, int) or isinstance(image_width, bool):
                    intrinsics = camera_config.get("intrinsics")
                    image_width = (
                        intrinsics.get("width")
                        if isinstance(intrinsics, dict)
                        else None
                    )
                if not isinstance(image_height, int) or isinstance(image_height, bool):
                    intrinsics = camera_config.get("intrinsics")
                    image_height = (
                        intrinsics.get("height")
                        if isinstance(intrinsics, dict)
                        else None
                    )
                if not isinstance(image_width, int) or isinstance(image_width, bool):
                    image_width = None
                if not isinstance(image_height, int) or isinstance(image_height, bool):
                    image_height = None
        except (OSError, UnicodeError, json.JSONDecodeError):
            pass
        mot_errors = _validate_required_mot(
            gt_txt,
            cam_id,
            image_width=image_width,
            image_height=image_height,
        )
        if mot_errors:
            errors.extend(mot_errors)
            st["mot_ok"] = False
            st["ok"] = False
    return st


def check_sync(
    cameras_meta: Dict[str, List[dict]],
    errors: List[str],
) -> dict:
    """跨相机时间/track/mask 同步检查。cameras_meta: cam_id -> frame meta 列表。"""
    st = {"ok": True, "checked_frames": 0}
    ids = list(cameras_meta.keys())
    if not ids:
        return st
    lengths = {camera_id: len(cameras_meta[camera_id]) for camera_id in ids}
    n = min(lengths.values())
    if len(set(lengths.values())) > 1:
        errors.append(f"同步: 相机 annotations 帧数不一致 {lengths}")
        st["ok"] = False
    for i in range(n):
        ref = cameras_meta[ids[0]][i]
        for cam in ids[1:]:
            cur = cameras_meta[cam][i]
            if cur.get("frame_index") != ref.get("frame_index"):
                errors.append(f"同步: 相机 {cam} 帧 {i} frame_index 不一致")
                st["ok"] = False
                continue
            for key in ("time_seconds", "source_step", "episode_id"):
                if cur.get(key) != ref.get(key):
                    errors.append(
                        f"同步: 相机 {cam} 帧 {i} 的 {key} "
                        f"({cur.get(key)!r}) != {ids[0]} ({ref.get(key)!r})"
                    )
                    st["ok"] = False
            # track_id / mask_id 跨相机一致
            ref_objs = {
                o.get("entity_id"): o
                for o in ref.get("objects", [])
                if isinstance(o, dict)
            }
            for o in cur.get("objects", []):
                if not isinstance(o, dict):
                    continue
                eid = o.get("entity_id")
                ref_o = ref_objs.get(eid)
                if ref_o is None:
                    continue
                for key in ("track_id", "mask_id"):
                    if o.get(key) != ref_o.get(key):
                        errors.append(
                            f"同步: {eid} 在相机 {cam} 帧 {i} 的 {key} "
                            f"({o.get(key)!r}) != {ids[0]} ({ref_o.get(key)!r})"
                        )
                        st["ok"] = False
    st["checked_frames"] = n
    return st


def check_track_mask_mapping(cameras_meta: Dict[str, List[dict]], errors: List[str],
                            mask_enabled: bool = True) -> dict:
    """校验每个实体的 track_id / mask_id 与稳定映射一致，且同帧不冲突。

    mask_enabled=False 时仅校验 track_id（mask_id 可为 None）。
    """
    st = {"ok": True, "checked_objects": 0}
    seen: Dict[str, Tuple[int, int]] = {}
    for cam, frames in cameras_meta.items():
        for fr in frames:
            for o in fr.get("objects", []):
                if not isinstance(o, dict):
                    continue
                eid = o.get("entity_id")
                if not eid:
                    continue
                st["checked_objects"] += 1
                expect_track = TRACK_MAP.get(eid)
                expect_mask = MASK_MAP.get(eid)
                if expect_track is None:
                    errors.append(f"映射: {cam} 帧 {fr.get('frame_index')} 未知 entity_id {eid!r}")
                    st["ok"] = False
                    continue
                got = (o.get("track_id"), o.get("mask_id"))
                # mask 关闭时 mask_id 可为 None，仅校验 track_id
                if mask_enabled:
                    if got != (expect_track, expect_mask):
                        errors.append(
                            f"映射: {eid} 在 {cam} 帧 {fr.get('frame_index')} "
                            f"track/mask={got}，期望 ({expect_track},{expect_mask})"
                        )
                        st["ok"] = False
                else:
                    if o.get("track_id") != expect_track:
                        errors.append(
                            f"映射: {eid} 在 {cam} 帧 {fr.get('frame_index')} "
                            f"track={o.get('track_id')}，期望 {expect_track}"
                        )
                        st["ok"] = False
                key = got if mask_enabled else (o.get("track_id"),)
                if eid in seen and seen[eid] != key:
                    errors.append(f"映射: {eid} 在不同位置出现不一致 {got} vs {seen[eid]}")
                    st["ok"] = False
                else:
                    seen[eid] = key
    return st


def check_camera_json(cam_dirs: List[Path], expected_frames: int, errors: List[str]) -> dict:
    """检查每个 camera.json 标定合法、分辨率一致、四相机外参不重复、无 NaN。"""
    st = {"ok": True, "cameras": {}}
    resolutions = set()
    locations: List[list] = []
    for cam in cam_dirs:
        path = cam / "camera.json"
        info = {"camera_id": cam.name, "ok": True}
        local_errors: List[str] = []
        try:
            with open(path, encoding="utf-8") as f:
                cfg = json.load(f)
        except (OSError, UnicodeError, json.JSONDecodeError) as e:
            local_errors.append(f"标定: {cam.name} camera.json 读取失败: {e}")
        else:
            if not isinstance(cfg, dict):
                local_errors.append("标定: camera.json 顶层必须是 JSON 对象")
                cfg = None
            if cfg is None:
                st["cameras"][cam.name] = info
                info["ok"] = False
                st["ok"] = False
                errors.extend(local_errors)
                continue
            w = cfg.get("image_width")
            h = cfg.get("image_height")
            if not isinstance(w, int) or not isinstance(h, int) or w <= 0 or h <= 0:
                local_errors.append(f"标定: {cam.name} 分辨率非法 {w}x{h}")
            else:
                resolutions.add((w, h))
            intr = cfg.get("intrinsics")
            if not isinstance(intr, dict):
                local_errors.append(
                    f"标定: {cam.name} intrinsics 必须是 JSON 对象，实际为 "
                    f"{type(intr).__name__}"
                )
                intr = {}
            for k in ("fx", "fy", "cx", "cy"):
                v = intr.get(k)
                if not _is_finite(v) or v <= 0:
                    local_errors.append(f"标定: {cam.name} intrinsics.{k} 非法 {v!r}")
            extr = cfg.get("extrinsics")
            if not isinstance(extr, dict):
                local_errors.append(
                    f"标定: {cam.name} extrinsics 必须是 JSON 对象，实际为 "
                    f"{type(extr).__name__}"
                )
                extr = {}
            loc = extr.get("world_location_m")
            if not isinstance(loc, list) or len(loc) != 3 or not all(_is_finite(v) for v in loc):
                local_errors.append(f"标定: {cam.name} extrinsics.world_location_m 非法 {loc!r}")
            else:
                locations.append(loc)
            for axis in ("forward", "right", "up"):
                vec = extr.get(axis)
                if not isinstance(vec, list) or len(vec) != 3 or not all(_is_finite(v) for v in vec):
                    local_errors.append(f"标定: {cam.name} extrinsics.{axis} 非法 {vec!r}")
        if local_errors:
            errors.extend(local_errors)
            info["ok"] = False
            st["ok"] = False
        st["cameras"][cam.name] = info
    if len(resolutions) > 1:
        errors.append(f"标定: 相机分辨率不一致 {resolutions}")
        st["ok"] = False
    # 仅当存在多个相机时检查外参是否重复（单相机无意义）
    if len(cam_dirs) >= 2 and len(locations) == len(cam_dirs) \
            and len({tuple(v) for v in locations}) == 1:
        errors.append("标定: 相机 world_location 完全相同（外参疑似重复/未标定）")
        st["ok"] = False
    st["resolutions"] = sorted(resolutions)
    return st


def check_render_summary(
    dataset_dir: Path,
    errors: List[str],
    warns: List[str],
    required: bool = True,
    expected_camera_ids: Optional[Sequence[str]] = None,
) -> dict:
    summary_path = dataset_dir / "render_summary.json"
    if not summary_path.exists():
        message = "缺少 render_summary.json（请先完成 UE RGB/MRQ 渲染）"
        if required:
            errors.append(message)
            return {"ok": False, "status": "missing", "required": True, "message": message}
        warns.append("缺少 render_summary.json（未要求 RGB 渲染，跳过）")
        return {"ok": True, "status": "missing", "required": False, "skipped": True,
                "warnings": warns[-1:]}
    try:
        with open(summary_path, encoding="utf-8") as f:
            summary = json.load(f)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        message = f"render_summary: 无法读取 render_summary.json: {exc}"
        if required:
            errors.append(message)
        else:
            warns.append(message)
        return {"ok": False, "status": "invalid", "required": bool(required),
                "skipped": not required, "error": str(exc), "message": message}
    if not isinstance(summary, dict):
        message = "render_summary: 顶层必须是 JSON 对象"
        if required:
            errors.append(message)
        else:
            warns.append(message)
        return {"ok": False, "status": "invalid", "required": bool(required),
                "skipped": not required, "error": message, "message": message}
    status = summary.get("status")
    per = summary.get("cameras")
    structure_errors: List[str] = []
    if not isinstance(per, dict):
        structure_errors.append("缺少 cameras 对象")
        per = {}
    elif not per:
        structure_errors.append("cameras 对象不能为空")
    else:
        bad = {
            c: e
            for c, e in per.items()
            if not isinstance(e, dict) or e.get("ok") is not True
        }
        if bad:
            structure_errors.append(f"未通过相机 {sorted(bad)}")
    if expected_camera_ids:
        missing = sorted(set(expected_camera_ids) - set(per))
        if missing:
            structure_errors.append(f"cameras 缺少预期相机 {missing}")
    if status != "success" or structure_errors:
        details = ", ".join(structure_errors)
        message = f"render_summary: status={status!r}，{details or '未通过'}"
        if required:
            errors.append(message)
        else:
            warns.append(message)
        return {"ok": False, "status": status, "required": bool(required),
                "skipped": not required, "cameras": per, "message": message}
    return {"ok": True, "status": status, "required": bool(required), "cameras": per}


def check_pose_coco17(
    dataset_dir: Path,
    expected_frames: int,
    errors: List[str],
    skip: bool = False,
    required: bool = False,
    expected_camera_ids: Optional[Sequence[str]] = None,
) -> dict:
    """检查 Runtime Pose + COCO17 完整性（C5.3 正式 Pose 来源）。

    仅当 pose_session.json 存在时校验（= Runtime Pose 已运行）：
    - capture_complete=True
    - pose_capture.jsonl 行数 == 10 actor × expected_frames × 13 bone
    - coco17_3d.jsonl 行数 == 10 actor × expected_frames
    pose_session.json 不存在 → 未启用 yolo_pose 时跳过；要求 Pose 时失败。
    skip=True（research_minimal 已 cleanup，raw pose 属有意删除的 transient）→ 跳过。
    """
    st = {"ok": True, "pose_session": False, "capture_complete": False,
          "pose_capture_rows": 0, "coco17_3d_rows": 0, "coco17_total_kp": 0, "skipped": False}
    if skip or not required:
        st["skipped"] = True
        st["required"] = False
        return st
    ps_path = dataset_dir / "pose_session.json"
    if not ps_path.exists():
        st["required"] = bool(required)
        if required:
            errors.append(
                f"pose: 缺少 pose_session.json（已要求 Runtime Pose；请先运行 UE pose-finalize）"
            )
            st["ok"] = False
        else:
            st["skipped"] = True
        return st
    pc_path = dataset_dir / "pose_capture.jsonl"
    c3d_path = dataset_dir / "coco17_3d.jsonl"
    # pose_session 存在 → 校验 capture_complete
    try:
        ps = json.loads(ps_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        errors.append(f"pose: pose_session.json 解析失败: {exc}")
        st["ok"] = False
        st["required"] = bool(required)
        return st
    if not isinstance(ps, dict):
        errors.append("pose: pose_session.json 顶层必须是 JSON 对象")
        st["ok"] = False
        st["required"] = bool(required)
        return st
    st["pose_session"] = True
    st["required"] = bool(required)
    st["capture_complete"] = ps.get("capture_complete") is True
    if not st["capture_complete"]:
        errors.append(
            f"pose: pose_session.capture_complete=False (captured={ps.get('captured_frame_count')}"
            f"/expected={ps.get('expected_frame_count')})，incomplete，禁止标注成功"
        )
        st["ok"] = False
    # pose_capture.jsonl 行数
    if pc_path.exists():
        st["pose_capture_rows"], capture_valid = _read_jsonl_rows(
            pc_path, "pose_capture.jsonl", errors
        )
        if not capture_valid:
            st["ok"] = False
        expected_capture = 10 * expected_frames * 13
        if st["pose_capture_rows"] != expected_capture:
            errors.append(
                f"pose_capture: {st['pose_capture_rows']} 行 != 预期 {expected_capture}"
                f"（10 actor × {expected_frames} 帧 × 13 bone）"
            )
            st["ok"] = False
    else:
        errors.append("pose: 缺少 pose_capture.jsonl")
        st["ok"] = False
    # coco17_3d 行数 == 10 × expected_frames（10 actor × N 帧）
    expected_coco = 10 * expected_frames
    if c3d_path.exists():
        st["coco17_3d_rows"], coco3d_valid = _read_jsonl_rows(
            c3d_path, "coco17_3d.jsonl", errors
        )
        if not coco3d_valid:
            st["ok"] = False
        st["coco17_total_kp"] = st["coco17_3d_rows"] * 17
        if st["coco17_3d_rows"] != expected_coco:
            errors.append(f"coco17_3d: {st['coco17_3d_rows']} 行 != 预期 {expected_coco}（10 actor × {expected_frames} 帧）")
            st["ok"] = False
    else:
        errors.append("coco17: 缺少 coco17_3d.jsonl")
        st["ok"] = False
    # coco17_2d 按 camera 分文件：检查每个 camera 子目录
    camera_ids = list(expected_camera_ids or [])
    if not camera_ids:
        camera_ids = sorted(
            d.name for d in dataset_dir.iterdir()
            if d.is_dir() and (d / "coco17_2d.jsonl").exists()
        )
    if not camera_ids:
        errors.append("coco17: 缺少任何 <camera>/coco17_2d.jsonl")
        st["ok"] = False
    else:
        for camera_id in camera_ids:
            c2d_path = dataset_dir / camera_id / "coco17_2d.jsonl"
            if not c2d_path.exists():
                errors.append(f"coco17: 缺少 {camera_id}/coco17_2d.jsonl")
                st["ok"] = False
                continue
            row_count, c2d_valid = _read_jsonl_rows(
                c2d_path,
                f"{camera_id}/coco17_2d.jsonl",
                errors,
            )
            if not c2d_valid:
                st["ok"] = False
            if row_count != expected_coco:
                errors.append(
                    f"coco17: {camera_id}/coco17_2d.jsonl 行数 {row_count} "
                    f"!= 预期 {expected_coco}（10 actor × {expected_frames} 帧）"
                )
                st["ok"] = False
    st["coco17_2d_cameras"] = camera_ids
    return st


def check_cross_camera_identity(cam_dirs: List[Path], errors: List[str]) -> dict:
    """跨 camera canonical identity audit：所有 camera 的所有 visible object 的
    track_id / mask_id 必须与 entity_id 的 canonical mapping 一致。

    允许某 camera 某 frame 不出现某 entity（遮挡/out-of-FOV），
    但一旦出现，track_id/mask_id 必须匹配 canonical mapping。
    """
    st = {"ok": True, "checked_objects": 0, "identity_errors": 0}
    for cam in cam_dirs:
        ann_path = cam / "annotations.jsonl"
        if not ann_path.exists():
            continue
        try:
            f = open(ann_path, encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            errors.append(f"identity: {cam.name} annotations.jsonl 读取失败: {exc}")
            st["ok"] = False
            continue
        with f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    fr = json.loads(line)
                except (UnicodeError, json.JSONDecodeError):
                    continue
                if not isinstance(fr, dict):
                    continue
                fi = fr.get("frame_index")
                objects = fr.get("objects", [])
                if not isinstance(objects, list):
                    continue
                for obj in objects:
                    if not isinstance(obj, dict):
                        continue
                    eid = obj.get("entity_id", "")
                    tid = obj.get("track_id")
                    mid = obj.get("mask_id")
                    st["checked_objects"] += 1
                    expected_tid = TRACK_MAP.get(eid)
                    expected_mid = MASK_MAP.get(eid)
                    if expected_tid is not None and tid != expected_tid:
                        errors.append(
                            f"identity: {cam.name} f{fi} {eid} track_id={tid} expected={expected_tid}"
                        )
                        st["identity_errors"] += 1
                        st["ok"] = False
                    if expected_mid is not None and mid is not None and mid != expected_mid:
                        errors.append(
                            f"identity: {cam.name} f{fi} {eid} mask_id={mid} expected={expected_mid}"
                        )
                        st["identity_errors"] += 1
                        st["ok"] = False
    return st


def write_reports(
    dataset_dir: Path,
    report: dict,
    out_dir: Path,
    args,
) -> Tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    jpath = out_dir / "soak_audit_report.json"
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    lines = [
        "# Soak 审计报告",
        "",
        f"- 输入: `{dataset_dir}`",
        f"- 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 期望: {args.expected_cameras} 相机 × {args.expected_frames_per_camera} 帧",
        f"- 退出码: {report['exit_code']}",
        f"- 结论: **{'PASS' if report['passed'] else 'FAIL'}**",
        "",
        "## 汇总",
        "",
        f"- 相机: {len(report['cameras'])} / {args.expected_cameras}",
        f"- 检查失败项: {len(report['errors'])}",
        f"- 警告: {len(report['warnings'])}",
        "",
        "## 每相机统计",
        "",
        "| 相机 | render PNG | render_mask EXR | img1 | mask | annotations | det | seg | gt.txt | 缺帧 | 重复 | 零字节 |",
        "|------|-----------:|----------------:|-----:|-----:|------------:|----:|----:|-------:|:----:|:----:|:------:|",
    ]
    for cid in sorted(report["cameras"]):
        c = report["cameras"][cid]
        lines.append(
            f"| {cid} | {c.get('render_rgb_png', '-')} | {c.get('render_mask_exr', '-')} "
            f"| {c.get('img1_png', '-')} | {c.get('mask_png', '-')} | {c.get('annotations_frames', '-')} "
            f"| {c.get('det_txt', '-')} | {c.get('seg_txt', '-')} | {c.get('gt_txt_lines', '-')} "
            f"| {len(c.get('img1_missing', []))} | {len(c.get('img1_dup', []))} | {c.get('zero_byte', '-')} |"
        )
    if report["sync"]:
        lines.append("")
        lines.append(f"## 跨相机同步\n\n- 检查帧数: {report['sync'].get('checked_frames', 0)}，OK: {report['sync'].get('ok')}")
    if report["mapping"]:
        lines.append("")
        lines.append(f"## track/mask 映射\n\n- 检查对象数: {report['mapping'].get('checked_objects', 0)}，OK: {report['mapping'].get('ok')}")
    if report["calibration"]:
        lines.append("")
        lines.append(f"## 相机标定\n\n- 分辨率: {report['calibration'].get('resolutions')}，OK: {report['calibration'].get('ok')}")
    if report["render_summary"]:
        lines.append("")
        lines.append(f"## render_summary\n\n- status: {report['render_summary'].get('status')}，OK: {report['render_summary'].get('ok')}")
    if report.get("pose_coco17"):
        pc = report["pose_coco17"]
        lines.append("")
        lines.append(
            f"## Runtime Pose + COCO17\n\n- pose_session: {pc.get('pose_session')}，"
            f"capture_complete: {pc.get('capture_complete')}\n"
            f"- pose_capture rows: {pc.get('pose_capture_rows')}\n"
            f"- coco17_3d rows: {pc.get('coco17_3d_rows')}，total_kp: {pc.get('coco17_total_kp')}，OK: {pc.get('ok')}"
        )
    if report.get("cross_camera_identity"):
        ci = report["cross_camera_identity"]
        lines.append("")
        lines.append(
            f"## Cross-Camera Identity Audit\n\n- checked_objects: {ci.get('checked_objects')}，"
            f"identity_errors: {ci.get('identity_errors')}，OK: {ci.get('ok')}"
        )
    if report["validation"]:
        lines.append("")
        lines.append(f"## validate-annotations（{report['validation'].get('level')}）\n\n- 退出码: {report['validation'].get('exit_code')}")
    if report["errors"]:
        lines.append("")
        lines.append("## 失败项\n")
        for e in report["errors"]:
            lines.append(f"- {e}")
    if report["warnings"]:
        lines.append("")
        lines.append("## 警告\n")
        for w in report["warnings"]:
            lines.append(f"- {w}")
    mpath = out_dir / "soak_audit_report.md"
    with open(mpath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return jpath, mpath


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="soak episode 完整性审计")
    ap.add_argument("--input", required=True, help="相机数据集根目录（含多个 camera 子目录）")
    ap.add_argument("--expected-cameras", type=int, default=4)
    ap.add_argument("--expected-frames-per-camera", type=int, default=300)
    ap.add_argument("--episode", default=None, help="episode 目录（读 meta 时序，精确校验渲染帧覆盖）")
    ap.add_argument("--render-fps", type=int, default=30, help="MRQ 渲染帧率（缺省 30）")
    ap.add_argument("--validation-level", default="quick", choices=["quick", "full", "none"])
    ap.add_argument("--mask-enabled", default="true", help="是否启用 instance_mask 校验（true/false，缺省 true）")
    ap.add_argument("--pose-skip", default="false", help="是否跳过 pose 完整性校验（true/false，缺省 false）")
    ap.add_argument(
        "--render-required", default="true",
        help="是否要求 render_summary.json 成功（true/false，缺省 true）",
    )
    ap.add_argument(
        "--mot-required", default="false",
        help="是否要求 MOT 的 gt/gt.txt 与 seqinfo.ini（true/false，缺省 false）",
    )
    ap.add_argument(
        "--yolo-det-required", default="false",
        help="是否要求 labels/det/*.txt（true/false，缺省 false）",
    )
    ap.add_argument(
        "--yolo-seg-required", default="false",
        help="是否要求 labels/seg/*.txt（true/false，缺省 false）",
    )
    ap.add_argument(
        "--pose-required", default="false",
        help="是否要求 Runtime Pose/COCO17（true/false，缺省 false）",
    )
    ap.add_argument("--output", default=None, help="报告输出目录（默认 <input>/audit）")
    args = ap.parse_args(argv)

    mask_enabled = _as_bool(args.mask_enabled)
    pose_skip = _as_bool(args.pose_skip)
    render_required = _as_bool(args.render_required)
    mot_required = _as_bool(args.mot_required)
    yolo_det_required = _as_bool(args.yolo_det_required)
    yolo_seg_required = _as_bool(args.yolo_seg_required)
    pose_required = _as_bool(args.pose_required)

    dataset_dir = Path(args.input)
    if not dataset_dir.is_dir():
        print(f"ERROR: 输入目录不存在: {dataset_dir}", file=sys.stderr)
        return 1

    errors: List[str] = []
    warns: List[str] = []
    report: dict = {"input": str(dataset_dir), "expected": {
        "cameras": args.expected_cameras,
        "frames_per_camera": args.expected_frames_per_camera,
    }, "requirements": {
        "render": render_required,
        "instance_mask": mask_enabled,
        "mot": mot_required,
        "yolo_det": yolo_det_required,
        "yolo_seg": yolo_seg_required,
        "pose": pose_required,
    }}

    # 相机发现
    cam_dirs = sorted(p.parent for p in dataset_dir.rglob("camera.json"))
    if len(cam_dirs) != args.expected_cameras:
        errors.append(f"相机数量 {len(cam_dirs)} != 预期 {args.expected_cameras}")

    # episode 时序 → keep_indices
    meta = _load_episode_meta(Path(args.episode)) if args.episode else None
    source_step_seconds = None
    keep_indices = list(range(args.expected_frames_per_camera))
    if meta:
        timing = meta.get("timing") or {}
        if isinstance(timing, dict):
            try:
                num_steps = int(timing.get("num_steps", args.expected_frames_per_camera))
                source_step_seconds = float(timing.get("source_step_seconds", 0.1))
                fps = int(timing.get("playback_fps", args.render_fps))
                if num_steps >= 0 and source_step_seconds > 0 and fps > 0:
                    keep_indices = [int(round(i * source_step_seconds * fps)) for i in range(num_steps)]
                    report["timing"] = {
                        "source_step_seconds": source_step_seconds,
                        "playback_fps": fps,
                        "num_steps": num_steps,
                        "max_keep_index": max(keep_indices) if keep_indices else 0,
                    }
            except (TypeError, ValueError, OverflowError):
                errors.append("meta.json timing 字段非法，无法推导渲染帧覆盖")
    else:
        # 从第一个相机标注反推
        if cam_dirs:
            frames, _ = _read_frame_meta(cam_dirs[0])
            if frames:
                keep_indices = _derive_keep_indices(frames, args.render_fps, None)
                if source_step_seconds is None and frames:
                    for fr in frames:
                        s = fr.get("source_step")
                        t = fr.get("time_seconds")
                        if (
                            isinstance(s, (int, float))
                            and not isinstance(s, bool)
                            and s > 0
                            and math.isfinite(s)
                            and isinstance(t, (int, float))
                            and not isinstance(t, bool)
                            and math.isfinite(t)
                        ):
                            source_step_seconds = t / s
                            break

    # 每相机
    cameras: Dict[str, dict] = {}
    cameras_meta: Dict[str, List[dict]] = {}
    for cam in cam_dirs:
        st = check_camera(
            cam,
            args.expected_frames_per_camera,
            keep_indices,
            errors,
            warns,
            mask_enabled=mask_enabled,
            mot_required=mot_required,
            yolo_det_required=yolo_det_required,
            yolo_seg_required=yolo_seg_required,
            enforce_mot_files=args.validation_level == "none",
            render_required=render_required,
        )
        frames, _ = _read_frame_meta(cam)
        cameras_meta[cam.name] = frames or []
        cameras[cam.name] = st
    report["cameras"] = cameras

    # 跨相机同步 + 映射
    sync = check_sync(cameras_meta, errors)
    mapping = check_track_mask_mapping(cameras_meta, errors, mask_enabled)
    calib = check_camera_json(cam_dirs, args.expected_frames_per_camera, errors)
    rsummary = check_render_summary(
        dataset_dir,
        errors,
        warns,
        required=render_required,
        expected_camera_ids=[cam.name for cam in cam_dirs],
    )
    pose_coco = check_pose_coco17(
        dataset_dir,
        args.expected_frames_per_camera,
        errors,
        skip=pose_skip,
        required=pose_required,
        expected_camera_ids=[cam.name for cam in cam_dirs],
    )
    cross_identity = check_cross_camera_identity(cam_dirs, errors)
    report["sync"] = sync
    report["mapping"] = mapping
    report["calibration"] = calib
    report["render_summary"] = rsummary
    report["pose_coco17"] = pose_coco
    report["cross_camera_identity"] = cross_identity

    # 可选进程内验证
    validation = None
    annotation_result = None
    if args.validation_level != "none":
        from grf_ue_bridge.annotation_validator import validate_annotation_result

        try:
            annotation_result = validate_annotation_result(
                dataset_dir,
                workers=0,
                validation_level=args.validation_level,
                require_mot=mot_required,
                require_mask=mask_enabled,
                require_yolo_det=yolo_det_required,
                require_yolo_seg=yolo_seg_required,
            )
        except Exception as exc:
            message = (
                f"validate-annotations({args.validation_level}) 执行异常: "
                f"{type(exc).__name__}: {exc}；请检查 annotations.jsonl、mask/ 和标签文件"
            )
            errors.append(message)
            annotation_result = ValidationResult(errors=[message])
            annotation_result.add_check(
                "annotation",
                CheckStatus.FAILED,
                required=True,
                message=message,
            )
            annotation_result.finalize()
        validation = {
            "level": args.validation_level,
            "exit_code": annotation_result.exit_code,
            "passed": annotation_result.passed,
            "errors": list(annotation_result.errors),
            "warnings": list(annotation_result.warnings),
            "checks": annotation_result.checks,
        }
        _extend_unique(errors, annotation_result.errors)
        _extend_unique(warns, annotation_result.warnings)
    report["validation"] = validation

    result = ValidationResult(errors=list(errors), warnings=list(warns))
    result.add_check(
        "camera_count",
        CheckStatus.PASSED if len(cam_dirs) == args.expected_cameras else CheckStatus.FAILED,
        required=True,
        message=errors[0] if len(cam_dirs) != args.expected_cameras else None,
        actual=len(cam_dirs),
        expected=args.expected_cameras,
    )
    result.add_check(
        "camera_core",
        CheckStatus.PASSED
        if cameras and all(c.get("core_ok", False) for c in cameras.values())
        else CheckStatus.FAILED,
        required=True,
        cameras={cid: c.get("core_ok", False) for cid, c in cameras.items()},
    )
    result.add_check(
        "render",
        CheckStatus.SKIPPED if not render_required else (
            CheckStatus.PASSED if rsummary.get("ok") else CheckStatus.FAILED
        ),
        required=render_required,
        message=rsummary.get("message") if render_required and not rsummary.get("ok") else None,
        detail=rsummary,
    )
    result.add_check(
        "instance_mask",
        CheckStatus.SKIPPED if not mask_enabled else (
            CheckStatus.PASSED
            if cameras and all(c.get("mask_ok", False) for c in cameras.values())
            else CheckStatus.FAILED
        ),
        required=mask_enabled,
        cameras={cid: c.get("mask_ok", False) for cid, c in cameras.items()},
    )
    annotation_mot_status = None
    if annotation_result is not None:
        annotation_mot_status = annotation_result.checks.get("mot_export", {}).get("status")
    if annotation_mot_status == CheckStatus.FAILED.value:
        mot_status = CheckStatus.FAILED
    elif annotation_mot_status == CheckStatus.SKIPPED.value:
        mot_status = CheckStatus.SKIPPED
    elif any(not c.get("mot_ok", True) for c in cameras.values()):
        mot_status = CheckStatus.FAILED
    elif mot_required:
        mot_status = (
            CheckStatus.PASSED
            if cameras and all(c.get("mot_complete", False) and c.get("mot_ok", False) for c in cameras.values())
            else CheckStatus.FAILED
        )
    elif not cameras or not all(c.get("mot_complete", False) for c in cameras.values()):
        mot_status = CheckStatus.SKIPPED
    else:
        mot_status = CheckStatus.PASSED
    result.add_check(
        "mot_export",
        mot_status,
        required=mot_required,
        cameras={cid: {"present": c.get("mot_present"), "ok": c.get("mot_ok")} for cid, c in cameras.items()},
    )
    for name, required, detail_key in (
        ("yolo_det", yolo_det_required, "det_ok"),
        ("yolo_seg", yolo_seg_required, "seg_ok"),
    ):
        present = any(c.get("det_present" if name == "yolo_det" else "seg_present") for c in cameras.values())
        valid = bool(cameras) and all(c.get(detail_key, False) for c in cameras.values())
        status = CheckStatus.SKIPPED if not required else (
            CheckStatus.PASSED if valid and present else CheckStatus.FAILED
        )
        result.add_check(
            name,
            status,
            required=required,
            cameras={cid: {"present": c.get("det_present" if name == "yolo_det" else "seg_present"), "ok": c.get(detail_key)} for cid, c in cameras.items()},
        )
    result.add_check(
        "sync",
        CheckStatus.PASSED if sync.get("ok") else CheckStatus.FAILED,
        required=True,
        detail=sync,
    )
    result.add_check(
        "mapping",
        CheckStatus.PASSED if mapping.get("ok") else CheckStatus.FAILED,
        required=True,
        detail=mapping,
    )
    result.add_check(
        "calibration",
        CheckStatus.PASSED if calib.get("ok") else CheckStatus.FAILED,
        required=True,
        detail=calib,
    )
    result.add_check(
        "runtime_pose",
        CheckStatus.SKIPPED if pose_coco.get("skipped") else (
            CheckStatus.PASSED if pose_coco.get("ok") else CheckStatus.FAILED
        ),
        required=pose_required and not pose_coco.get("skipped"),
        detail=pose_coco,
    )
    result.add_check(
        "cross_camera_identity",
        CheckStatus.PASSED if cross_identity.get("ok") else CheckStatus.FAILED,
        required=True,
        detail=cross_identity,
    )
    if annotation_result is None:
        result.add_check(
            "annotation_validation",
            CheckStatus.SKIPPED,
            required=False,
            level="none",
        )
    else:
        annotation_check = annotation_result.checks.get("annotation", {})
        result.add_check(
            "annotation_validation",
            annotation_check.get(
                "status",
                CheckStatus.PASSED if annotation_result.passed else CheckStatus.FAILED,
            ),
            required=True,
            detail=validation,
        )

    result.finalize()
    report.update(result.to_dict())

    out_dir = Path(args.output) if args.output else dataset_dir / "audit"
    jpath, mpath = write_reports(dataset_dir, report, out_dir, args)

    # 人类可读摘要
    print(f"soak 审计完成: {'PASS' if report['passed'] else 'FAIL'}")
    print(f"  相机: {len(cam_dirs)}/{args.expected_cameras}")
    print(f"  失败项: {len(errors)}")
    for e in errors[:20]:
        print(f"    FAIL  {e}")
    for w in warns[:10]:
        print(f"    WARN  {w}")
    print(f"  报告: {jpath}")
    print(f"        {mpath}")
    return report["exit_code"]


if __name__ == "__main__":
    sys.exit(main())
