"""端到端 dataset 回归校验：从 mask + annotations 重新派生 bbox / MOT / YOLO 并与落盘产物比对。

与 annotation_validator 的分工：
  - annotation_validator 逐 camera 校验 annotations 的字段语义与 mask 合法性（单产物内）。
  - 本模块做**跨产物一致性**回归：RGB / mask / annotations 帧数严格对应、分辨率一致、
    mask 非全背景、entity→class/track/mask 规则、instance_mask bbox == mask 像素、
    MOT / YOLO Det / YOLO Seg == 从 annotations 重新派生的期望值、多连通域 merge 的
    raster quality gate 复验。

不修改任何标注生成逻辑——只读 mask/annotations/MOT/YOLO，重新跑一遍生成公式并比对。
对缺 mask 的 legacy 目录（纯几何标注）放行相应检查；对 mask-primary 目录则要求
RGB/mask/annotation 全链路一致。

入口：
  - `validate_dataset_regression(annotation_dir) -> int`（0=通过，1=失败，打印报告）
  - `collect_dataset_regression_errors(annotation_dir) -> List[str]`（供 validate_annotation_dir 合并）
  两者由 `grf-ue validate-annotations` 统一触发。
"""

import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

_UE_DIR = Path(__file__).resolve().parent.parent.parent / "ue"
if str(_UE_DIR) not in sys.path:
    sys.path.insert(0, str(_UE_DIR))

from annotation_utils import (  # noqa: E402
    BBOX_SOURCE_INSTANCE_MASK,
    BBOX_SOURCE_NOT_VISIBLE,
    entity_class,
    entity_id_to_mask_id,
    entity_id_to_track_id,
)

RGB_SUFFIXES = {".png", ".jpg", ".jpeg"}
from dataset_export import build_mot_gt  # noqa: E402
from instance_mask import (  # noqa: E402
    det_xyxy_to_yolo_norm,
    load_mask_array,
    mask_to_bbox,
    polygon_to_mask,
    quantize_mask_pixels,
    raster_quality_metrics,
    yolo_class_id,
)
from .mask_annotator import (  # noqa: E402
    _AREA_TOL_EXTRA_RATIO,
    _AREA_TOL_IOU,
    _AREA_TOL_MISSING_RATIO,
)


def _camera_dirs(annotation_dir: Path) -> List[Path]:
    """递归找出所有含 camera.json 的 camera 子目录。"""
    return sorted(d.parent for d in annotation_dir.rglob("camera.json"))


def _frame_numbers(dir_path: Path, suffixes=RGB_SUFFIXES) -> set:
    """从目录里的 RGB 文件名解析帧号集合。"""
    if not dir_path.exists():
        return set()
    nums = set()
    for p in dir_path.iterdir():
        if not p.is_file() or p.suffix.lower() not in suffixes:
            continue
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


def _png_size(path: Path) -> Optional[Tuple[int, int]]:
    """读取 PNG 尺寸 (w, h)。失败返回 None。"""
    from PIL import Image

    try:
        with Image.open(str(path)) as img:
            return img.size
    except Exception:
        return None


# ── 从 annotations 重新派生期望产物 ──────────────────────────────────────

def _expected_yolo_det_lines(objects: Sequence[dict], width: int, height: int,
                             include_ball: bool) -> List[str]:
    """与 mask_annotator._write_yolo_labels 相同的 det 行生成公式。"""
    lines: List[str] = []
    for obj in objects:
        if obj.get("bbox_source") != BBOX_SOURCE_INSTANCE_MASK:
            continue
        if not obj.get("in_frame") or not obj.get("bbox_xyxy"):
            continue
        if obj.get("class") == "ball" and not include_ball:
            continue
        cid = yolo_class_id(obj["entity_id"])
        cx, cy, w, h = det_xyxy_to_yolo_norm(obj["bbox_xyxy"], width, height)
        lines.append(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines


def _expected_yolo_seg_lines(objects: Sequence[dict], width: int, height: int,
                             include_ball: bool) -> List[str]:
    """与 mask_annotator._write_yolo_labels 相同的 seg 行生成公式。"""
    lines: List[str] = []
    for obj in objects:
        if obj.get("bbox_source") != BBOX_SOURCE_INSTANCE_MASK:
            continue
        if not obj.get("in_frame") or not obj.get("bbox_xyxy"):
            continue
        if obj.get("class") == "ball" and not include_ball:
            continue
        seg = obj.get("segmentation")
        if not seg:
            continue
        cid = yolo_class_id(obj["entity_id"])
        lines.append(f"{cid} " + " ".join(f"{v:.6f}" for v in seg))
    return lines


def _read_lines(path: Path) -> List[str]:
    """读取文本文件非空行。文件不存在返回 []。"""
    if not path.exists():
        return []
    try:
        return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    except OSError:
        return []


# ── 逐 camera 回归校验 ──────────────────────────────────────────────────

def _validate_camera(cam_dir: Path) -> List[str]:
    errors: List[str] = []
    label = f"[{cam_dir.name}]"

    cam_json_path = cam_dir / "camera.json"
    ann_path = cam_dir / "annotations.jsonl"
    if not cam_json_path.exists() or not ann_path.exists():
        # 结构缺失由 annotation_validator 报告，这里不重复
        return errors
    try:
        cam = json.loads(cam_json_path.read_text(encoding="utf-8"))
        intr = cam.get("intrinsics") or {}
        width = int(intr.get("width", 0))
        height = int(intr.get("height", 0))
    except (json.JSONDecodeError, OSError, ValueError, TypeError):
        return errors
    if width <= 0 or height <= 0:
        return errors

    frames: List[dict] = []
    try:
        with open(ann_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    frames.append(json.loads(line))
    except (json.JSONDecodeError, OSError):
        return errors
    if not frames:
        errors.append(f"{label} annotations.jsonl 为空")
        return errors

    ann_frames = {int(f.get("frame_index")) for f in frames if isinstance(f.get("frame_index"), int)}
    img1_dir, mask_dir = cam_dir / "img1", cam_dir / "mask"
    img_frames = _frame_numbers(img1_dir)
    mask_frames = _frame_numbers(mask_dir, {".png"})
    has_mask = mask_dir.exists()
    has_img = img1_dir.exists()

    # ── 1. 帧数全链路：annotation == img1 == mask ───────────────────
    if has_img and img_frames != ann_frames:
        errors.append(
            f"{label} img1/ 帧数({len(img_frames)}) != annotations 帧数({len(ann_frames)})"
            f"（缺/多: {sorted(ann_frames ^ img_frames)[:5]}）"
        )
    if has_mask and mask_frames != ann_frames:
        errors.append(
            f"{label} mask/ 帧数({len(mask_frames)}) != annotations 帧数({len(ann_frames)})"
            f"（缺/多: {sorted(ann_frames ^ mask_frames)[:5]}）"
        )
    if has_mask and not has_img:
        errors.append(f"{label} mask/ 存在但缺 img1/（RGB 未渲染，mask-primary 数据集不完整）")

    decode = _read_mask_config(cam_dir)
    channel = decode.get("mask_channel", "r")
    id_scale = float(decode.get("id_scale", 1.0))
    id_offset = float(decode.get("id_offset", 0.0))
    frames_by_index = {int(f["frame_index"]): f for f in frames if isinstance(f.get("frame_index"), int)}

    # 逐帧：分辨率 / mask 非全背景 / 逐对象 mask 派生比对
    for fi in sorted(ann_frames):
        # 2. RGB 分辨率 == mask 分辨率 == camera 分辨率
        if has_img:
            img_path = img1_dir / f"{fi:06d}.jpg"
            if not img_path.exists():
                errors.append(f"{label} img1/{fi:06d}.jpg 缺失")
            else:
                sz = _png_size(img_path)
                if sz != (width, height):
                    errors.append(f"{label} img1/{fi:06d}.jpg 分辨率 {sz} != camera {width}x{height}")
        quantized = None
        mask_path = mask_dir / f"{fi:06d}.png" if has_mask else None
        if has_mask:
            if not mask_path.exists():
                errors.append(f"{label} mask/{fi:06d}.png 缺失")
            else:
                mask_img = load_mask_array(mask_path, channel)
                mh, mw = mask_img.shape
                if (mw, mh) != (width, height):
                    errors.append(f"{label} mask/{fi:06d}.png 分辨率 {mw}x{mh} != camera {width}x{height}")
                quantized = quantize_mask_pixels(mask_img, id_scale, id_offset)
                # 3. mask 不得全背景
                if not quantized.any():
                    errors.append(f"{label} mask/{fi:06d}.png 全背景（无任何实体像素，疑似渲染失败）")
        frame = frames_by_index.get(fi)
        if frame is None:
            continue
        for oi, obj in enumerate(frame.get("objects", [])):
            olabel = f"{label} frame {fi} objects[{oi}]"
            _check_object(errors, olabel, obj, quantized, width, height, mask_dir)

    # ── 4. MOT / YOLO 重新派生比对 ─────────────────────────────────
    per_frame_objects = [f.get("objects", []) for f in frames]
    include_ball_mot = any(",100," in ln for ln in _read_lines(cam_dir / "gt" / "gt.txt"))
    expected_mot = build_mot_gt(per_frame_objects, width, height, include_ball_mot)
    actual_mot = _read_lines(cam_dir / "gt" / "gt.txt")
    if expected_mot != actual_mot:
        errors.append(f"{label} MOT gt.txt 与从 annotations 重新派生不一致（不可见对象不应出现、bbox 应一致）")

    include_ball_yolo = False
    det_dir, seg_dir = cam_dir / "labels" / "det", cam_dir / "labels" / "seg"
    if det_dir.exists():
        for p in sorted(det_dir.glob("*.txt")):
            if any(ln.startswith("1 ") for ln in _read_lines(p)):
                include_ball_yolo = True
                break
        for fi in sorted(ann_frames):
            det_path = det_dir / f"{fi:06d}.txt"
            expected = _expected_yolo_det_lines(frames_by_index[fi].get("objects", []), width, height,
                                                include_ball_yolo)
            actual = _read_lines(det_path)
            if actual != expected:
                errors.append(
                    f"{label} YOLO det/{fi:06d}.txt 与从 annotations 重新派生不一致"
                    f"（期望 {len(expected)} 行，实际 {len(actual)} 行）"
                )
    if seg_dir.exists():
        for fi in sorted(ann_frames):
            seg_path = seg_dir / f"{fi:06d}.txt"
            expected = _expected_yolo_seg_lines(frames_by_index[fi].get("objects", []), width, height,
                                                include_ball_yolo)
            actual = _read_lines(seg_path)
            if actual != expected:
                errors.append(
                    f"{label} YOLO seg/{fi:06d}.txt 与从 annotations 重新派生不一致"
                    f"（期望 {len(expected)} 行，实际 {len(actual)} 行）"
                )
    return errors


def _check_object(errors: List[str], olabel: str, obj: dict, quantized,
                  width: int, height: int, mask_dir: Path) -> None:
    """校验单个对象：entity/class/track/mask 规则 + mask 派生一致性。"""
    eid = obj.get("entity_id")
    if not isinstance(eid, str):
        return
    # BALL / 球员 的 class / track_id / mask_id 确定性规则
    exp_class = entity_class(eid)
    if obj.get("class") != exp_class:
        errors.append(f"{olabel} class={obj.get('class')!r} != 期望 {exp_class!r}")
    try:
        exp_track = entity_id_to_track_id(eid)
        exp_mask = entity_id_to_mask_id(eid)
    except (ValueError, TypeError):
        return
    if obj.get("track_id") != exp_track:
        errors.append(f"{olabel} track_id={obj.get('track_id')!r} != 期望 {exp_track!r}")
    if obj.get("mask_id") not in (None, exp_mask):
        errors.append(f"{olabel} mask_id={obj.get('mask_id')!r} != 期望 {exp_mask!r}")

    src = obj.get("bbox_source")
    mask_id = exp_mask
    if quantized is None:
        return  # 无 mask 数据：legacy/几何标注，跳过 mask 派生检查

    if src == BBOX_SOURCE_INSTANCE_MASK:
        binary = quantized == mask_id
        bb = mask_to_bbox(binary)
        ann_bb = obj.get("bbox_xyxy")
        if bb is None:
            errors.append(f"{olabel} bbox_source=instance_mask 但 mask 无像素")
            return
        if not isinstance(ann_bb, (list, tuple)) or len(ann_bb) != 4:
            errors.append(f"{olabel} bbox_source=instance_mask 但 bbox_xyxy 缺失")
        else:
            for a, b in zip(bb, ann_bb):
                if abs(float(a) - float(b)) > 0.5:
                    errors.append(
                        f"{olabel} instance-mask bbox {[round(v, 2) for v in bb]}"
                        f" != annotations bbox {ann_bb}"
                    )
                    break
        vpc = obj.get("visible_pixel_count")
        if vpc != int(binary.sum()):
            errors.append(f"{olabel} visible_pixel_count={vpc!r} != mask 像素数 {int(binary.sum())}")
        _check_seg_gate(errors, olabel, obj, binary, width, height)
    elif src == BBOX_SOURCE_NOT_VISIBLE:
        binary = quantized == mask_id
        if binary.any():
            errors.append(f"{olabel} bbox_source=not_visible 但 mask 含可见像素（{int(binary.sum())} px）")


def _check_seg_gate(errors: List[str], olabel: str, obj: dict, binary, width: int, height: int) -> None:
    """复验多连通域 merge 的 raster quality gate（放宽阈值吸收舍入误差）。

    只在「多连通域已成功合并」（segmentation_merged=True）时复验面积膨胀检查——annotate
    时已用精确 ring 认证过 gate，这里对**反归一化的存储 ring** 做 sanity 复验：
      - 6 位小数 YOLO 归一化有 <0.001px 舍入误差，可能在临界处翻转 1~2 个栅格像素
        （实测 iou 0.748 vs 0.75）；故阈值放宽 ±0.05，仍能捕获 gross 违例。
      - 极小可见对象（如 1~5px 的球，全部为孤立单像素连通域）无法提取多边形 →
        annotate 合法输出 segmentation=None，不属于错误（YOLO seg 自动跳过该行）。
    单连通域与回退对象不在此 gate 范围。
    """
    seg = obj.get("segmentation")
    if not seg:
        # 极小/全孤立单像素可见对象：无法多边形化，annotate 合法输出 None（跳过）
        return
    if len(seg) % 2 != 0:
        errors.append(f"{olabel} segmentation 点数为奇数（应成对 x/y）")
        return
    if not bool(obj.get("segmentation_merged")) or obj.get("segmentation_fallback"):
        return  # 非多连通域成功合并：不在此 gate 范围
    ring = [(float(seg[i]) * width, float(seg[i + 1]) * height)
            for i in range(0, len(seg), 2)]
    # ROI 栅格化 + 质量指标（同 mask_annotator._check_area_quality，阈值放宽 ±0.05）
    xs = [p[0] for p in ring]
    ys = [p[1] for p in ring]
    x0 = max(0, int(math.floor(min(xs))))
    y0 = max(0, int(math.floor(min(ys))))
    x1 = min(width, int(math.ceil(max(xs))) + 1)
    y1 = min(height, int(math.ceil(max(ys))) + 1)
    if x1 <= x0 or y1 <= y0:
        return
    shifted = [(x - x0, y - y0) for x, y in ring]
    raster = polygon_to_mask(shifted, x1 - x0, y1 - y0)
    truth = binary[y0:y1, x0:x1]
    m = raster_quality_metrics(raster, truth)
    if (m["extra_ratio"] > _AREA_TOL_EXTRA_RATIO + 0.05
            or m["refined_missing_ratio"] > _AREA_TOL_MISSING_RATIO + 0.05
            or m["iou"] < _AREA_TOL_IOU - 0.05):
        errors.append(
            f"{olabel} segmentation 多连通域 merge quality gate 未通过"
            f"（iou={m['iou']:.3f}, extra={m['extra_ratio']:.3f}, missing={m['refined_missing_ratio']:.3f}）"
        )


# ── 统一入口 ────────────────────────────────────────────────────────────

def collect_dataset_regression_errors(annotation_dir: Path) -> List[str]:
    """返回全部回归错误列表（供 validate_annotation_dir 合并）。"""
    errors: List[str] = []
    camera_dirs = _camera_dirs(annotation_dir)
    if not camera_dirs:
        errors.append(f"目录 {annotation_dir} 下没有 camera 子目录（缺少 camera.json）")
    for cam_dir in camera_dirs:
        errors += _validate_camera(cam_dir)
    return errors


def validate_dataset_regression(annotation_dir: Path) -> int:
    """端到端回归校验。返回 0（通过）或 1（失败）。"""
    errors = collect_dataset_regression_errors(annotation_dir)
    if not errors:
        print(f"DATASET REGRESSION: {annotation_dir} PASSED")
        return 0
    print(f"DATASET REGRESSION: Found {len(errors)} error(s)")
    for err in errors[:50]:
        print(f"  ERROR: {err}")
    if len(errors) > 50:
        print(f"  ... and {len(errors) - 50} more errors")
    return 1
