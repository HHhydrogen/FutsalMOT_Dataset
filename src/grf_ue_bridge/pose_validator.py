"""验证 YOLO Pose 标注目录（labels_pose/）。纯 Python + numpy，不依赖 unreal。

校验项（见 docs/design/2026-08-11-yolo-pose-export.md）：
  文件级：
    - labels_pose/ 与 pose_keypoints.jsonl 帧一一对应；空场景允许空 txt。
    - 每行必须恰好 56 个字段（5 + 17×3）。
  数值范围：
    - bbox：0 <= xc/yc/w/h <= 1，且 w/h > 0。
    - 关键点：0 <= x <= 1、0 <= y <= 1、v ∈ {0, 1, 2}。
  实例级（full 级别重新派生并逐行比对；quick 级别结构 + 行数比对）：
    - 每实例恰好 17 个关键点（由 56 字段保证）。
    - track/player 不重复、bbox 与 instance mask 一致（行数与 bbox 来自同一
      annotations.jsonl mask-primary 数据）。
    - 左右关节点映射未反转：用世界坐标验证双肩轴与双髋轴方向一致
      （shoulder_vec · hip_vec > 0，任一轴反转则点积为负）。

调用入口：grf-ue validate-pose <annotation_dir>（见 cli.py）；task postprocess 集成。
"""

import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# 把仓库的 ue/ 目录加入 sys.path（与 tests/conftest.py 一致）
_UE_DIR = Path(__file__).resolve().parent.parent.parent / "ue"
if str(_UE_DIR) not in sys.path:
    sys.path.insert(0, str(_UE_DIR))

from grf_ue_bridge.pose_annotator import (  # noqa: E402
    LABELS_POSE_DIRNAME,
    compute_pose_instances,
    instance_to_line,
    read_pose_keypoints,
    _load_camera,
)
from pose_bones import (  # noqa: E402
    COCO_KEYPOINT_NAMES,
    NUM_COCO_KEYPOINTS,
    YOLO_POSE_FIELDS,
    bone_index_of,
)
from instance_mask import load_mask_array, quantize_mask_pixels  # noqa: E402


def _camera_dirs(annotation_dir: Path) -> List[Path]:
    return sorted(d.parent for d in annotation_dir.rglob("camera.json"))


def _read_mask_config(cam_dir: Path) -> dict:
    cfg_path = cam_dir / "mask_config.json"
    if cfg_path.exists():
        try:
            return json.loads(cfg_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _load_ann_by_frame(cam_dir: Path) -> Dict[int, dict]:
    ann_by_frame: Dict[int, dict] = {}
    p = cam_dir / "annotations.jsonl"
    if not p.exists():
        return ann_by_frame
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                fr = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(fr.get("frame_index"), int):
                ann_by_frame[fr["frame_index"]] = fr
    return ann_by_frame


def _left_right_axis_consistency(pose_frame: dict) -> List[str]:
    """世界坐标下检查左右未反转：双肩轴与双髋轴应同向（点积 > 0）。

    对每个实例，用 left/right shoulder 与 left/right hip 的连线方向做点积。
    若映射把左右之一调换（如 clavicle_l 误配到 right_shoulder），两轴方向相反。
    """
    errors: List[str] = []
    i_ls = bone_index_of("left_shoulder")
    i_rs = bone_index_of("right_shoulder")
    i_lh = bone_index_of("left_hip")
    i_rh = bone_index_of("right_hip")
    for obj in pose_frame.get("objects", []):
        kps = obj.get("keypoints_world")
        if not kps or len(kps) != NUM_COCO_KEYPOINTS:
            continue

        def _vec(a: int, b: int) -> Optional[Tuple[float, float, float]]:
            pa, pb = kps[a], kps[b]
            if pa is None or pb is None or pa[0] is None or pb[0] is None:
                return None
            return (pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2])

        shoulder = _vec(i_ls, i_rs)  # left -> right
        hip = _vec(i_lh, i_rh)
        if shoulder is None or hip is None:
            continue
        s = math.sqrt(sum(c * c for c in shoulder))
        h = math.sqrt(sum(c * c for c in hip))
        if s == 0 or h == 0:
            continue
        dot = sum(a * b for a, b in zip(shoulder, hip)) / (s * h)
        if dot < 0.0:
            errors.append(
                f"[左右轴] {obj.get('entity_id')} 双肩轴与双髋轴反向"
                f"（dot={dot:.2f}）——左右关节点映射可能反转"
            )
    return errors


def _validate_label_file(
    errors: List[str],
    cam_label: str,
    p: Path,
) -> Dict[int, List[str]]:
    """校验单个 labels_pose/*.txt：56 字段 + 数值范围。返回 {frame: lines}。"""
    lines_by_frame: Dict[int, List[str]] = {}
    digits = "".join(ch for ch in p.stem if ch.isdigit())
    fi = int(digits) if digits else None
    with open(p, encoding="utf-8") as f:
        raw_lines = [ln.strip() for ln in f if ln.strip()]
    lines_by_frame[fi] = raw_lines
    for line_no, line in enumerate(raw_lines, start=1):
        parts = line.split()
        if len(parts) != YOLO_POSE_FIELDS:
            errors.append(
                f"[{cam_label}] {p.name} 第 {line_no} 行字段数 {len(parts)} "
                f"!= {YOLO_POSE_FIELDS}（5 + 17×3）：{line[:80]!r}"
            )
            continue
        try:
            floats = [float(v) for v in parts]
        except ValueError:
            errors.append(f"[{cam_label}] {p.name} 第 {line_no} 行含非数字")
            continue
        # class 必须是 player（0）
        if int(floats[0]) != 0:
            errors.append(f"[{cam_label}] {p.name} 第 {line_no} 行 class 应为 0（player）：{parts[0]!r}")
        # bbox：xc yc w h ∈ [0,1]，w/h > 0
        cx, cy, w, h = floats[1:5]
        for name, val in (("xc", cx), ("yc", cy), ("w", w), ("h", h)):
            if not (0.0 <= val <= 1.0):
                errors.append(f"[{cam_label}] {p.name} 第 {line_no} 行 bbox {name} 越界 [0,1]: {val}")
        if w <= 0 or h <= 0:
            errors.append(f"[{cam_label}] {p.name} 第 {line_no} 行 bbox w/h 非正")
        # 17 关键点：x,y ∈ [0,1]，v ∈ {0,1,2}
        for i in range(NUM_COCO_KEYPOINTS):
            x, y, v = floats[5 + 3 * i], floats[6 + 3 * i], floats[7 + 3 * i]
            name = COCO_KEYPOINT_NAMES[i]
            if not (0.0 <= x <= 1.0) or not (0.0 <= y <= 1.0):
                errors.append(
                    f"[{cam_label}] {p.name} 第 {line_no} 行 {name}(#{i}) 坐标越界 [0,1]: "
                    f"({x}, {y})"
                )
            if v not in (0, 1, 2):
                errors.append(
                    f"[{cam_label}] {p.name} 第 {line_no} 行 {name}(#{i}) v={v} 非法（应为 0/1/2）"
                )
    return lines_by_frame


def _recompute_frame_lines(
    cam_dir: Path,
    cam_label: str,
    pose_frame: dict,
    ann_frame: dict,
    radius: int,
    use_mask: bool,
    errors: List[str],
) -> Optional[List[str]]:
    """重新派生一帧的 YOLO Pose 行（与 annotate-pose 同一代码路径），用于比对。"""
    intrinsics, extrinsics, size = _load_camera(cam_dir)
    if intrinsics is None or size is None:
        return None
    width, height = size
    mask_ids = None
    if use_mask:
        decode = _read_mask_config(cam_dir)
        mask_dir = cam_dir / "mask"
        fi = pose_frame.get("frame_index")
        if mask_dir.is_dir() and isinstance(fi, int):
            mask_path = mask_dir / f"{fi:06d}.png"
            if mask_path.exists():
                try:
                    mask_img = load_mask_array(mask_path, decode.get("mask_channel", "r"))
                    mask_ids = quantize_mask_pixels(
                        mask_img,
                        float(decode.get("id_scale", 1.0)),
                        float(decode.get("id_offset", 0.0)),
                    )
                except Exception as e:  # noqa: BLE001
                    errors.append(f"[{cam_label}] mask {fi} 重算失败: {e}")
    instances = compute_pose_instances(
        pose_frame, ann_frame, mask_ids, intrinsics, extrinsics,
        width, height, radius,
    )
    return [instance_to_line(inst, width, height) for inst in instances]


def _validate_pose_camera(cam_dir: Path, radius: int, validation_level: str) -> List[str]:
    """验证单个 camera 的 labels_pose/。返回错误列表（空 = 通过）。"""
    errors: List[str] = []
    cam_label = cam_dir.name
    pose_dir = cam_dir / LABELS_POSE_DIRNAME
    if not pose_dir.is_dir():
        errors.append(f"[{cam_label}] 缺少 {LABELS_POSE_DIRNAME}/（先运行 annotate-pose）")
        return errors

    meta, pose_frames = read_pose_keypoints(cam_dir)
    if meta is None:
        errors.append(f"[{cam_label}] 缺少 pose_keypoints.jsonl")
        return errors
    ann_by_frame = _load_ann_by_frame(cam_dir)

    # 帧集合对应：labels_pose 帧号 == pose 帧号（空 txt 也允许）
    label_frames = set()
    for p in pose_dir.glob("*.txt"):
        digits = "".join(ch for ch in p.stem if ch.isdigit())
        if digits:
            label_frames.add(int(digits))
        _validate_label_file(errors, cam_label, p)
    pose_frames_set = {f.get("frame_index") for f in pose_frames if isinstance(f.get("frame_index"), int)}
    if label_frames != pose_frames_set:
        only_label = sorted(label_frames - pose_frames_set)[:5]
        only_pose = sorted(pose_frames_set - label_frames)[:5]
        errors.append(
            f"[{cam_label}] labels_pose/ 帧号与 pose_keypoints 不一致"
            f"（仅 label: {only_label}，仅 pose: {only_pose}）"
        )

    # 图片 ↔ 标签对应（img1 存在时）
    img1 = cam_dir / "img1"
    if img1.is_dir():
        img_frames = set()
        for p in img1.glob("*.png"):
            digits = "".join(ch for ch in p.stem if ch.isdigit())
            if digits:
                img_frames.add(int(digits))
        missing = sorted(pose_frames_set - img_frames)[:5]
        if missing:
            errors.append(f"[{cam_label}] pose 帧缺少对应 img1 图片: {missing}")

    # 实例级：左右轴一致性（世界坐标，无需 mask）
    for pose_frame in pose_frames:
        errors += _left_right_axis_consistency(pose_frame)

    # 行数/内容比对（重新派生）
    use_mask = validation_level != "quick"
    label_lines: Dict[int, List[str]] = {}
    for p in sorted(pose_dir.glob("*.txt")):
        digits = "".join(ch for ch in p.stem if ch.isdigit())
        fi = int(digits) if digits else None
        with open(p, encoding="utf-8") as f:
            label_lines[fi] = [ln.strip() for ln in f if ln.strip()]
    for pose_frame in pose_frames:
        fi = pose_frame.get("frame_index")
        if not isinstance(fi, int):
            continue
        ann_frame = ann_by_frame.get(fi)
        if ann_frame is None:
            errors.append(f"[{cam_label}] pose 帧 {fi} 缺对应 annotations.jsonl 行")
            continue
        expected = _recompute_frame_lines(
            cam_dir, cam_label, pose_frame, ann_frame, radius, use_mask, errors
        )
        if expected is None:
            continue
        actual = label_lines.get(fi, [])
        if validation_level == "quick":
            if len(actual) != len(expected):
                errors.append(
                    f"[{cam_label}] 帧 {fi} 标签行数 {len(actual)} != 期望 {len(expected)}"
                    f"（track 重复 / 漏导出 / 多余行）"
                )
        else:
            if actual != expected:
                errors.append(
                    f"[{cam_label}] 帧 {fi} 标签与重新派生产物不一致"
                    f"（{len(actual)} vs {len(expected)} 行；首处差异见下）"
                )
                for a, b in zip(actual, expected):
                    if a != b:
                        errors.append(f"[{cam_label}] 帧 {fi} 差异:\n  label: {a}\n  expect: {b}")
                        break
    return errors


def _validate_pose_task(task: tuple) -> Tuple[str, List[str]]:
    cam_str, radius, validation_level = task
    return cam_str, _validate_pose_camera(Path(cam_str), radius, validation_level)


def _resolve_workers(workers: int, n_tasks: int) -> int:
    if workers == 0:
        return min(max(1, n_tasks), max(1, (os.cpu_count() or 1) // 2))
    return max(1, workers)


def _report(errors: List[str]) -> None:
    if not errors:
        return
    print(f"POSE VALIDATOR: Found {len(errors)} error(s)")
    for err in errors[:50]:
        print(f"  ERROR: {err}")
    if len(errors) > 50:
        print(f"  ... and {len(errors) - 50} more errors")


def validate_pose_dir(
    annotation_dir: Path,
    workers: int = 0,
    validation_level: str = "full",
    visibility_neighborhood_radius: int = 2,
) -> int:
    """验证一个标注输出目录的 YOLO Pose 标签。返回 0=通过 / 1=失败。

    validation_level：full=结构 + 逐帧 mask 重算比对；quick=结构 + 行数比对。
    """
    if validation_level not in ("full", "quick"):
        raise ValueError(f"未知 validation_level: {validation_level!r}（可选 full/quick）")
    camera_dirs = _camera_dirs(annotation_dir)
    errors: List[str] = []
    if not camera_dirs:
        errors.append(f"目录 {annotation_dir} 下没有 camera 子目录（缺少 camera.json）")
    nworkers = _resolve_workers(workers, len(camera_dirs))
    radius = int(visibility_neighborhood_radius)
    if nworkers <= 1 or len(camera_dirs) <= 1:
        for cam_dir in camera_dirs:
            errors += _validate_pose_camera(cam_dir, radius, validation_level)
    else:
        tasks = [(str(cam_dir), radius, validation_level) for cam_dir in camera_dirs]
        with ProcessPoolExecutor(max_workers=nworkers) as ex:
            results = list(ex.map(_validate_pose_task, tasks))
        for cam_str, cam_errors in results:
            errors += cam_errors
    _report(errors)
    if not errors:
        print(f"POSE VALIDATOR: {annotation_dir} PASSED all checks")
        return 0
    return 1
