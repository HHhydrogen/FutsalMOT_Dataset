"""从渲染出的 Instance-ID Mask 生成 mask-primary 的 CV 标注（P1 侧，纯 Python）。

数据流（在 UE 渲染完成、mask/*.png 落盘后运行）：
  读 camera.json（分辨率） + UE 导出的 annotations.jsonl（实体元数据 + 几何 bbox）
  + mask/{frame_index:06d}.png（Instance-ID Mask，像素值 == mask_id）
  → 每实体由 mask 像素计算 pixel-tight bbox / visible_pixel_count / 模态分割 polygon
  → 覆盖写 annotations.jsonl（bbox_source="instance_mask"；几何 bbox 保留在 geometry_*）
  → 重写 MOT gt.txt、写 YOLO detect/segment 标签
  → 写 mask_config.json（记录解码参数，供 validator 复用）

语义约定（可见像素 GT 与几何投影 GT 严格分离）：
  - mask 中该实体有像素  → bbox_source="instance_mask"，in_frame=true，
    bbox_xyxy/xywh = mask 可见 bbox，visible_pixel_count > 0，segmentation 为模态多边形。
  - mask 中该实体像素为 0（完全遮挡/离屏）→ bbox_source="not_visible"，in_frame=false，
    bbox_xyxy/xywh = null、visible_pixel_count = 0、segmentation = null；
    几何投影只保留在 geometry_bbox_*，绝不回填到 bbox_*。
  - 无 mask 数据（无 mask/ 目录或缺某帧 mask）→ 保持 UE 几何标注原样（legacy fallback，
    无 bbox_source 字段）。MOT / YOLO 只导出 bbox_source="instance_mask" 的对象。

幂等：可重复运行。几何 fallback 每次从 geometry_bbox_*（无则首次的 bbox_*）读取，
mask 则重新解码计算，结果一致。

调用入口：grf-ue annotate-masks <output_dir>（见 cli.py）。
"""

import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# 把仓库的 ue/ 目录加入 sys.path（与 tests/conftest.py 一致），以便 import 纯模块
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_UE_DIR = _REPO_ROOT / "ue"
if str(_UE_DIR) not in sys.path:
    sys.path.insert(0, str(_UE_DIR))

from annotation_utils import (  # noqa: E402
    BBOX_SOURCE_INSTANCE_MASK,
    BBOX_SOURCE_NOT_VISIBLE,
    entity_id_to_mask_id,
    xyxy_to_xywh,
)
from dataset_export import (  # noqa: E402
    build_mot_gt,
    ensure_dir,
    write_json_atomic,
    write_jsonl_atomic,
    write_text_atomic,
)
from instance_mask import (  # noqa: E402
    compute_instance_stats,
    det_xyxy_to_yolo_norm,
    load_mask_array,
    mask_to_polygons_with_areas,
    merge_to_single_ring,
    polygon_to_mask,
    quantize_mask_pixels,
    raster_quality_metrics,
    ring_to_yolo_flat,
    yolo_class_id,
)

# MOT visibility：mask-primary 已把 bbox 裁剪到图像内，truncation 模式无意义，固定 unoccluded
_MOT_VISIBILITY_MODE = "unoccluded"

# 多连通域合并的面积膨胀检查阈值（仅多连通域时生效；单连通域不进入）
_AREA_TOL_EXTRA_RATIO = 0.10
_AREA_TOL_MISSING_RATIO = 0.05
_AREA_TOL_IOU = 0.75


def _camera_dirs(annotation_dir: Path) -> List[Path]:
    """递归找出所有含 camera.json 的 camera 子目录（与 annotation_validator 一致）。"""
    return sorted(d.parent for d in annotation_dir.rglob("camera.json"))


# ── 快速导出模式（--formats / --no-segmentation）────────────────────────

# 合法的导出格式；annotations.jsonl（json）总是写入，本集合只过滤派生产物
_FORMAT_JSON = "json"
_FORMAT_MOT = "mot"
_FORMAT_YOLO_DET = "yolo-det"
_FORMAT_YOLO_SEG = "yolo-seg"
_ALL_FORMATS = (_FORMAT_JSON, _FORMAT_MOT, _FORMAT_YOLO_DET, _FORMAT_YOLO_SEG)


def _parse_formats(formats: str) -> Set[str]:
    """解析 --formats 参数（逗号组合；'all' = 全部格式）。非法值抛 ValueError。"""
    if not formats or formats.strip().lower() == "all":
        return set(_ALL_FORMATS)
    out: Set[str] = set()
    for part in formats.split(","):
        part = part.strip().lower()
        if part not in _ALL_FORMATS:
            raise ValueError(f"未知导出格式: {part!r}（可选 all/mot/yolo-det/yolo-seg/json，逗号组合）")
        out.add(part)
    return out


def _resolve_workers(workers: int, n_tasks: int) -> int:
    """解析 --workers：0=自动（min(任务数, max(1, cpu_count//2))），1=串行，>1=指定。"""
    if workers == 0:
        return min(max(1, n_tasks), max(1, (os.cpu_count() or 1) // 2))
    return max(1, workers)


def annotate_masks_dir(
    annotation_dir: Path,
    mask_channel: str = "r",
    include_ball: bool = False,
    polygon_tolerance_px: float = 1.0,
    max_polygon_points: int = 64,
    id_scale: float = 1.0,
    id_offset: float = 0.0,
    workers: int = 0,
    chunk_size: int = 0,
    formats: str = "all",
    no_segmentation: bool = False,
) -> int:
    """对输出目录下所有 camera 子目录执行 mask → 标注 转换。返回退出码（0 成功 / 1 失败）。

    workers：0=自动，1=串行，>1=相机级/帧级多进程并行（输出逐字节确定）。
    chunk_size：>0 时按该帧数把单相机的帧切成连续 chunk（相机数少于 workers 时用）。
    formats：all/mot/yolo-det/yolo-seg/json 或逗号组合，控制写哪些派生产物
      （annotations.jsonl 总是写入）。
    no_segmentation：跳过轮廓/polygon/桥接/质量检查，segmentation=null，不生成
      labels/seg/；bbox、像素数、MOT、YOLO Det 正常生成。
    """
    cam_dirs = _camera_dirs(annotation_dir)
    if not cam_dirs:
        print(f"ERROR: {annotation_dir} 下没有 camera 子目录（缺少 camera.json）")
        return 1
    fmt = _parse_formats(formats)
    nworkers = _resolve_workers(workers, len(cam_dirs))
    if no_segmentation:
        fmt.discard(_FORMAT_YOLO_SEG)

    if nworkers <= 1 or len(cam_dirs) <= 1:
        failures = 0
        for cam_dir in cam_dirs:
            try:
                ok = _annotate_camera(
                    cam_dir,
                    mask_channel=mask_channel,
                    include_ball=include_ball,
                    polygon_tolerance_px=polygon_tolerance_px,
                    max_polygon_points=max_polygon_points,
                    id_scale=id_scale,
                    id_offset=id_offset,
                    formats=fmt,
                    no_segmentation=no_segmentation,
                )
            except Exception as e:
                print(f"  ERROR: {cam_dir.name}: {e}")
                ok = False
            if not ok:
                failures += 1
        if failures:
            print(f"annotate-masks 完成，但有 {failures} 个 camera 目录失败")
            return 1
        print("annotate-masks 完成")
        return 0

    # 并行：相机级 + 相机内帧分块
    failures = 0
    n_cameras = len(cam_dirs)
    per_cam_chunks = max(1, nworkers // n_cameras)
    opts = _SliceOpts(
        mask_channel=mask_channel,
        include_ball=include_ball,
        polygon_tolerance_px=polygon_tolerance_px,
        max_polygon_points=max_polygon_points,
        id_scale=id_scale,
        id_offset=id_offset,
        formats=frozenset(fmt),
        no_segmentation=no_segmentation,
    )
    tasks: List[tuple] = []
    for cam_dir in cam_dirs:
        nf = _annotation_line_count(cam_dir)
        if nf <= 0:
            # 空 annotations.jsonl：仍生成任务（(0,0) 切片返回空帧），保证与串行
            # 一致的「空产物」写入（annotations.jsonl / mask_config / 空 MOT / YOLO）。
            tasks.append((str(cam_dir), (0, 0), opts))
            continue
        if per_cam_chunks <= 1:
            tasks.append((str(cam_dir), (0, nf), opts))
        else:
            step = chunk_size if chunk_size > 0 else max(1, math.ceil(nf / per_cam_chunks))
            for start in range(0, nf, step):
                tasks.append((str(cam_dir), (start, min(start + step, nf)), opts))

    with ProcessPoolExecutor(max_workers=nworkers) as ex:
        results = list(ex.map(_annotate_slice_task, tasks))

    # 按 camera 合并：同一相机的各 chunk 按起点排序拼接，再写最终产物
    merged: Dict[str, List[Tuple[int, List[dict]]]] = {}
    for (cam_str, (start, _end), _o), upgraded in zip(tasks, results):
        merged.setdefault(cam_str, []).append((start, upgraded))
    for cam_str, chunks in merged.items():
        cam_dir = Path(cam_str)
        if not (cam_dir / "mask").exists():
            # 与串行 _annotate_camera 一致：无 mask/ 目录则跳过（保持几何 fallback）
            continue
        chunks.sort(key=lambda c: c[0])
        all_frames: List[dict] = []
        for _start, up in chunks:
            all_frames.extend(up)
        try:
            width, height = _camera_size(cam_dir)
            ok = _write_annotate_outputs(
                cam_dir, all_frames, width, height, opts.include_ball, opts.mask_channel,
                opts.polygon_tolerance_px, opts.max_polygon_points,
                opts.id_scale, opts.id_offset, opts.formats,
            )
        except Exception as e:
            print(f"  ERROR: {cam_dir.name}: {e}")
            ok = False
        if not ok:
            failures += 1
    if failures:
        print(f"annotate-masks 完成，但有 {failures} 个 camera 目录失败")
        return 1
    print("annotate-masks 完成")
    return 0


def _annotate_camera(cam_dir: Path, mask_channel: str, include_ball: bool,
                     polygon_tolerance_px: float, max_polygon_points: int,
                     id_scale: float, id_offset: float,
                     formats: Optional[Set[str]] = None,
                     no_segmentation: bool = False) -> bool:
    """转换单个 camera 目录（串行路径）。返回是否成功（无 mask 目录也算跳过成功）。

    formats 为 _parse_formats 解析出的集合（None = 全部格式）。输出逐字节确定，
    与并行路径一致。
    """
    if formats is None:
        formats = set(_ALL_FORMATS)
    cam_json_path = cam_dir / "camera.json"
    if not cam_json_path.exists():
        print(f"  SKIP {cam_dir.name}: 缺 camera.json")
        return False
    cam = json.loads(cam_json_path.read_text(encoding="utf-8"))
    width = int(cam.get("image_width", 0) or cam.get("intrinsics", {}).get("width", 0))
    height = int(cam.get("image_height", 0) or cam.get("intrinsics", {}).get("height", 0))
    if width <= 0 or height <= 0:
        print(f"  SKIP {cam_dir.name}: camera.json 图像尺寸非法")
        return False

    ann_path = cam_dir / "annotations.jsonl"
    if not ann_path.exists():
        print(f"  SKIP {cam_dir.name}: 缺 annotations.jsonl（先运行 UE --mode annotations）")
        return False
    frames = [json.loads(line) for line in ann_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    mask_dir = cam_dir / "mask"
    if not mask_dir.exists():
        print(f"  SKIP {cam_dir.name}: 无 mask/ 目录，保持几何 bbox（fallback）")
        return True

    upgraded = []
    mask_used = 0
    for frame in frames:
        up, used = _upgrade_one_frame(
            frame, mask_dir, width, height, mask_channel,
            polygon_tolerance_px, max_polygon_points, id_scale, id_offset,
            no_segmentation,
        )
        upgraded.append(up)
        mask_used += 1 if used else 0

    _write_annotate_outputs(
        cam_dir, upgraded, width, height, include_ball, mask_channel,
        polygon_tolerance_px, max_polygon_points, id_scale, id_offset, formats,
    )

    print(
        f"  [OK] {cam_dir.name}: {mask_used}/{len(frames)} 帧 mask 已转为标注"
        f"（可见 → bbox_source=instance_mask，不可见 → not_visible；MOT/YOLO 已更新）"
    )
    return True


def _camera_size(cam_dir: Path) -> Tuple[int, int]:
    """读取 camera.json 的图像尺寸 (width, height)。"""
    cam = json.loads((cam_dir / "camera.json").read_text(encoding="utf-8"))
    width = int(cam.get("image_width", 0) or cam.get("intrinsics", {}).get("width", 0))
    height = int(cam.get("image_height", 0) or cam.get("intrinsics", {}).get("height", 0))
    return width, height


def _annotation_line_count(cam_dir: Path) -> int:
    """annotations.jsonl 的非空行数（并行分块用）。"""
    p = cam_dir / "annotations.jsonl"
    if not p.exists():
        return 0
    n = 0
    with open(p, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def _read_frames_slice(cam_dir: Path, start: int, end: int) -> List[dict]:
    """读取 annotations.jsonl 的第 [start, end) 行（0 基，非空行计数）。"""
    out: List[dict] = []
    with open(cam_dir / "annotations.jsonl", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < start:
                continue
            if i >= end:
                break
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _upgrade_one_frame(frame: dict, mask_dir: Path, width: int, height: int,
                       mask_channel: str, polygon_tolerance_px: float,
                       max_polygon_points: int, id_scale: float, id_offset: float,
                       no_segmentation: bool) -> Tuple[dict, bool]:
    """升级单帧：有 mask 则就地升级对象并返回 (frame, True)；无 mask 返回 (frame, False)。

    每帧只读一次 mask、量化一次、单次扫描统计；所有对象复用同一份结果。
    """
    frame_index = int(frame.get("frame_index", 0))
    mask_path = mask_dir / f"{frame_index:06d}.png"
    if not mask_path.exists():
        return frame, False
    mask_img = load_mask_array(mask_path, mask_channel)
    mask_ids = quantize_mask_pixels(mask_img, id_scale, id_offset)
    stats = compute_instance_stats(mask_ids)
    for obj in frame.get("objects", []):
        _upgrade_object(
            obj, mask_ids, stats, width, height,
            polygon_tolerance_px, max_polygon_points,
            do_segmentation=not no_segmentation,
        )
    return frame, True


def _write_annotate_outputs(
    cam_dir: Path,
    frames: List[dict],
    width: int,
    height: int,
    include_ball: bool,
    mask_channel: str,
    polygon_tolerance_px: float,
    max_polygon_points: int,
    id_scale: float,
    id_offset: float,
    formats: Set[str],
) -> None:
    """写 annotations.jsonl + mask_config.json +（按 formats）MOT / YOLO det / YOLO seg。

    串行与并行路径共用，保证同一份 frames 得到逐字节一致的产物。
    """
    ann_path = cam_dir / "annotations.jsonl"
    write_jsonl_atomic(ann_path, frames)
    write_json_atomic(
        cam_dir / "mask_config.json",
        {
            "mask_channel": mask_channel,
            "id_scale": id_scale,
            "id_offset": id_offset,
            "polygon_tolerance_px": polygon_tolerance_px,
            "max_polygon_points": max_polygon_points,
            "note": "由 grf-ue annotate-masks 写入；validator 据此解码 mask 做一致性校验。",
        },
    )
    if _FORMAT_MOT in formats:
        rows = build_mot_gt(
            [f.get("objects", []) for f in frames], width, height,
            include_ball, _MOT_VISIBILITY_MODE,
        )
        write_text_atomic(cam_dir / "gt" / "gt.txt", "\n".join(rows) + ("\n" if rows else ""))
    if _FORMAT_YOLO_DET in formats or _FORMAT_YOLO_SEG in formats:
        _write_yolo_labels(
            cam_dir, frames, width, height, include_ball,
            write_det=_FORMAT_YOLO_DET in formats,
            write_seg=_FORMAT_YOLO_SEG in formats,
        )
    return True


@dataclass(frozen=True)
class _SliceOpts:
    """并行任务的不变参数（可 pickle，Windows spawn 安全）。"""

    mask_channel: str
    include_ball: bool
    polygon_tolerance_px: float
    max_polygon_points: int
    id_scale: float
    id_offset: float
    formats: frozenset
    no_segmentation: bool


def _annotate_slice_task(task: tuple) -> List[dict]:
    """进程池 worker：处理 (cam_dir, (start, end), _SliceOpts)，返回升级后的帧列表。

    模块级函数（可 pickle）；只读 mask/ 与 annotations.jsonl，不写任何文件——
    最终产物由主进程统一合并排序后原子写入（保证确定性与幂等）。
    """
    cam_str, (start, end), opts = task
    cam_dir = Path(cam_str)
    width, height = _camera_size(cam_dir)
    frames = _read_frames_slice(cam_dir, start, end)
    mask_dir = cam_dir / "mask"
    upgraded: List[dict] = []
    for frame in frames:
        up, _used = _upgrade_one_frame(
            frame, mask_dir, width, height, opts.mask_channel,
            opts.polygon_tolerance_px, opts.max_polygon_points,
            opts.id_scale, opts.id_offset, opts.no_segmentation,
        )
        upgraded.append(up)
    return upgraded


def _check_area_quality(ring, binary):
    """ROI 内栅格化 merged ring 与原始 mask 对比；返回失败原因字符串或 None。

    ring 为 ROI 坐标系（未偏移回全图）的闭合多边形，binary 为对应的 ROI 二值
    mask（mask_ids[roi_slice] == mask_id）。栅格化与质量指标与全图坐标下逐点一致
    （多边形栅格化对平移不变）。

    Gate：extra_ratio ≤ tol、refined_missing_ratio ≤ tol、iou ≥ tol；
    任一项失败或栅格化异常即返回原因（由调用方回退最大连通域）。
    """
    try:
        xs = [p[0] for p in ring]
        ys = [p[1] for p in ring]
        x0 = max(0, int(math.floor(min(xs))))
        y0 = max(0, int(math.floor(min(ys))))
        x1 = min(binary.shape[1], int(math.ceil(max(xs))) + 1)
        y1 = min(binary.shape[0], int(math.ceil(max(ys))) + 1)
        roi_w, roi_h = x1 - x0, y1 - y0
        if roi_w <= 0 or roi_h <= 0:
            return "empty_roi"
        shifted = [(x - x0, y - y0) for x, y in ring]
        raster = polygon_to_mask(shifted, roi_w, roi_h)
        truth = binary[y0:y1, x0:x1]
        m = raster_quality_metrics(raster, truth)
        if m["extra_ratio"] > _AREA_TOL_EXTRA_RATIO:
            return "extra_ratio=%.3f>%.2f" % (m["extra_ratio"], _AREA_TOL_EXTRA_RATIO)
        if m["refined_missing_ratio"] > _AREA_TOL_MISSING_RATIO:
            return "refined_missing_ratio=%.3f>%.2f" % (
                m["refined_missing_ratio"], _AREA_TOL_MISSING_RATIO)
        if m["iou"] < _AREA_TOL_IOU:
            return "iou=%.3f<%.2f" % (m["iou"], _AREA_TOL_IOU)
        return None
    except Exception as e:
        return "rasterization_error: %s" % e


def _upgrade_object(obj: dict, mask_ids, stats, width: int, height: int,
                    polygon_tolerance_px: float, max_polygon_points: int,
                    do_segmentation: bool = True) -> None:
    """就地升级单个 object：mask 可见 → bbox_source=instance_mask；mask 空 → not_visible。

    可见像素 GT 与几何投影 GT 严格分离：
      - mask 有像素：bbox_xyxy/xywh 为 mask 可见 bbox，geometry_bbox_* 保留几何投影。
      - mask 无像素：in_frame=false、bbox_xyxy/xywh/raw_bbox_* 全部为 null，
        segmentation 为 null，geometry 只保留在 geometry_bbox_*（不回填）。

    mask_ids 为已量化的整帧实例 ID 数组（uint8），stats 为该帧的单次扫描统计
    （compute_instance_stats 结果）。多边形提取只在实例的紧凑 ROI 内进行，
    完成后偏移回全图坐标——输出与整图处理逐点一致。

    do_segmentation=False（快速模式）：跳过轮廓/polygon/桥接/质量检查，
    segmentation 相关字段置空，但 bbox、像素数、MOT、YOLO Det 正常生成。
    """
    entity_id = obj.get("entity_id")
    try:
        mask_id = entity_id_to_mask_id(entity_id)
    except (ValueError, TypeError):
        return  # 未知实体，保持原样

    # 几何 fallback：优先读已存的 geometry_*，否则用当前 bbox_*（首次运行即几何值）
    geo_xyxy = obj.get("geometry_bbox_xyxy") or obj.get("bbox_xyxy")
    geo_xywh = obj.get("geometry_bbox_xywh") or obj.get("bbox_xywh")

    st = stats.get(mask_id) if stats is not None else None
    if st is None or st.pixel_count == 0:
        # 完全不可见（遮挡 / 离屏）：模态 GT 无可见 bbox。几何投影只保留在
        # geometry_bbox_*，可见 bbox 与 raw bbox 一律为 null，绝不回填（不进入 MOT/YOLO）。
        obj.update({
            "mask_id": mask_id,
            "bbox_source": BBOX_SOURCE_NOT_VISIBLE,
            "in_frame": False,
            "truncated": False,
            "visibility": None,
            "visible_pixel_count": 0,
            "segmentation": None,
            "segmentation_components": 0,
            "segmentation_merged": False,
            "segmentation_fallback": None,
            "segmentation_fallback_reason": None,
            "bbox_xyxy": None,
            "bbox_xywh": None,
            "raw_bbox_xyxy": None,
            "raw_bbox_xywh": None,
            "geometry_bbox_xyxy": [round(v, 3) for v in geo_xyxy] if geo_xyxy else None,
            "geometry_bbox_xywh": [round(v, 3) for v in geo_xywh] if geo_xywh else None,
        })
        return

    bbox = [float(v) for v in st.bbox_xyxy]
    xywh = xyxy_to_xywh(bbox)

    if not do_segmentation:
        # 快速模式：不做任何 polygon 计算
        obj.update({
            "mask_id": mask_id,
            "bbox_source": BBOX_SOURCE_INSTANCE_MASK,
            "in_frame": True,
            "truncated": False,
            "visibility": None,
            "visible_pixel_count": st.pixel_count,
            "bbox_xyxy": [round(v, 3) for v in bbox],
            "bbox_xywh": [round(v, 3) for v in xywh],
            "segmentation": None,
            "segmentation_components": 0,
            "segmentation_merged": False,
            "segmentation_fallback": None,
            "segmentation_fallback_reason": None,
            "geometry_bbox_xyxy": [round(v, 3) for v in geo_xyxy] if geo_xyxy else None,
            "geometry_bbox_xywh": [round(v, 3) for v in geo_xywh] if geo_xywh else None,
        })
        return

    # 只在 ROI 内创建二值 mask（避免对整张 1080p 图逐对象处理）
    roi_binary = mask_ids[st.roi_slice] == mask_id
    polys, comp_areas = mask_to_polygons_with_areas(
        roi_binary, polygon_tolerance_px, max_polygon_points
    )
    n_components = len(polys)
    # ring 目前是 ROI 坐标系
    ring, meta = merge_to_single_ring(polys, comp_areas) if polys else ([], None)
    fallback = meta.get("fallback") if meta else None
    fallback_reason = meta.get("fallback_reason") if meta else None
    # 面积膨胀检查（仅多连通域且桥接成功时；ring 为 ROI 坐标，与 roi_binary 同系）
    if n_components > 1 and fallback is None and ring:
        reason = _check_area_quality(ring, roi_binary)
        if reason:
            best = max(range(len(polys)), key=lambda i: comp_areas[i])
            ring = list(polys[best])
            fallback = "largest_component"
            fallback_reason = reason
    # 偏移回全图坐标后做 YOLO 归一化
    x0, y0 = st.x0, st.y0
    if ring:
        ring = [(px + x0, py + y0) for px, py in ring]
    seg_flat = ring_to_yolo_flat(ring, width, height) if ring else None

    obj.update({
        "mask_id": mask_id,
        "bbox_source": BBOX_SOURCE_INSTANCE_MASK,
        "in_frame": True,
        "truncated": False,  # mask bbox 必在图像内
        "visibility": None,
        "visible_pixel_count": st.pixel_count,
        "bbox_xyxy": [round(v, 3) for v in bbox],
        "bbox_xywh": [round(v, 3) for v in xywh],
        "segmentation": seg_flat,
        "segmentation_components": n_components,
        "segmentation_merged": n_components > 1 and fallback is None,
        "segmentation_fallback": fallback,
        "segmentation_fallback_reason": fallback_reason,
        "geometry_bbox_xyxy": [round(v, 3) for v in geo_xyxy] if geo_xyxy else None,
        "geometry_bbox_xywh": [round(v, 3) for v in geo_xywh] if geo_xywh else None,
    })


def _write_yolo_labels(cam_dir: Path, frames: List[dict], width: int, height: int,
                       include_ball: bool, write_det: bool = True,
                       write_seg: bool = True) -> None:
    """写 YOLO detect / segment 标签到 labels/det/ 与 labels/seg/（与帧一一对应）。

    write_det / write_seg 控制是否写对应目录（--formats / --no-segmentation）。
    """
    det_dir = cam_dir / "labels" / "det"
    seg_dir = cam_dir / "labels" / "seg"
    if write_det:
        ensure_dir(det_dir)
    if write_seg:
        ensure_dir(seg_dir)
    for frame in frames:
        frame_index = int(frame.get("frame_index", 0))
        det_lines: List[str] = []
        seg_lines: List[str] = []
        for obj in frame.get("objects", []):
            # 只导出具有有效 instance-mask 可见 GT 的对象；不可见（not_visible /
            # 无 mask 的 legacy 几何）不进入 YOLO 训练标签。
            if obj.get("bbox_source") != BBOX_SOURCE_INSTANCE_MASK:
                continue
            if not obj.get("in_frame") or not obj.get("bbox_xyxy"):
                continue  # 防御：schema 不一致时跳过（validator 会报）
            cls = obj.get("class")
            if cls == "ball" and not include_ball:
                continue
            if write_det:
                cid = yolo_class_id(obj["entity_id"])
                cx, cy, w, h = det_xyxy_to_yolo_norm(obj["bbox_xyxy"], width, height)
                det_lines.append(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            if write_seg:
                seg = obj.get("segmentation")
                if seg:
                    cid = yolo_class_id(obj["entity_id"])
                    seg_lines.append(f"{cid} " + " ".join(f"{v:.6f}" for v in seg))
        if write_det:
            write_text_atomic(det_dir / f"{frame_index:06d}.txt", "\n".join(det_lines) + ("\n" if det_lines else ""))
        if write_seg:
            write_text_atomic(seg_dir / f"{frame_index:06d}.txt", "\n".join(seg_lines) + ("\n" if seg_lines else ""))
