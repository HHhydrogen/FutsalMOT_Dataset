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
import sys
from pathlib import Path
from typing import Dict, List, Optional

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
    decode_mask_pixels,
    det_xyxy_to_yolo_norm,
    load_mask_array,
    mask_to_bbox,
    mask_to_polygons_with_areas,
    merge_to_single_ring,
    polygon_to_mask,
    raster_quality_metrics,
    ring_to_yolo_flat,
    visible_pixel_count,
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


def annotate_masks_dir(
    annotation_dir: Path,
    mask_channel: str = "r",
    include_ball: bool = False,
    polygon_tolerance_px: float = 1.0,
    max_polygon_points: int = 64,
    id_scale: float = 1.0,
    id_offset: float = 0.0,
) -> int:
    """对输出目录下所有 camera 子目录执行 mask → 标注 转换。返回退出码（0 成功 / 1 失败）。"""
    cam_dirs = _camera_dirs(annotation_dir)
    if not cam_dirs:
        print(f"ERROR: {annotation_dir} 下没有 camera 子目录（缺少 camera.json）")
        return 1
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
                     id_scale: float, id_offset: float) -> bool:
    """转换单个 camera 目录。返回是否成功（无 mask 目录也算跳过成功）。"""
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
        frame_index = int(frame.get("frame_index", 0))
        mask_path = mask_dir / f"{frame_index:06d}.png"
        if not mask_path.exists():
            # 该帧无 mask：保留几何标注
            upgraded.append(frame)
            continue
        mask_img = load_mask_array(mask_path, mask_channel)
        for obj in frame.get("objects", []):
            _upgrade_object(
                obj, mask_img, width, height,
                polygon_tolerance_px, max_polygon_points, id_scale, id_offset,
            )
        mask_used += 1
        upgraded.append(frame)

    write_jsonl_atomic(ann_path, upgraded)
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

    # MOT
    rows = build_mot_gt(
        [f.get("objects", []) for f in upgraded], width, height,
        include_ball, _MOT_VISIBILITY_MODE,
    )
    write_text_atomic(cam_dir / "gt" / "gt.txt", "\n".join(rows) + ("\n" if rows else ""))

    # YOLO
    _write_yolo_labels(cam_dir, upgraded, width, height, include_ball)

    print(
        f"  [OK] {cam_dir.name}: {mask_used}/{len(frames)} 帧 mask 已转为标注"
        f"（可见 → bbox_source=instance_mask，不可见 → not_visible；MOT/YOLO 已更新）"
    )
    return True


def _check_area_quality(ring, binary, width, height):
    """ROI 内栅格化 merged ring 与原始 mask 对比；返回失败原因字符串或 None。

    Gate：extra_ratio ≤ tol、refined_missing_ratio ≤ tol、iou ≥ tol；
    任一项失败或栅格化异常即返回原因（由调用方回退最大连通域）。
    """
    try:
        xs = [p[0] for p in ring]
        ys = [p[1] for p in ring]
        x0 = max(0, int(math.floor(min(xs))))
        y0 = max(0, int(math.floor(min(ys))))
        x1 = min(width, int(math.ceil(max(xs))) + 1)
        y1 = min(height, int(math.ceil(max(ys))) + 1)
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


def _upgrade_object(obj: dict, mask_img, width: int, height: int,
                    polygon_tolerance_px: float, max_polygon_points: int,
                    id_scale: float, id_offset: float) -> None:
    """就地升级单个 object：mask 可见 → bbox_source=instance_mask；mask 空 → not_visible。

    可见像素 GT 与几何投影 GT 严格分离：
      - mask 有像素：bbox_xyxy/xywh 为 mask 可见 bbox，geometry_bbox_* 保留几何投影。
      - mask 无像素：in_frame=false、bbox_xyxy/xywh/raw_bbox_* 全部为 null，
        segmentation 为 null，geometry 只保留在 geometry_bbox_*（不回填）。
    """
    entity_id = obj.get("entity_id")
    try:
        mask_id = entity_id_to_mask_id(entity_id)
    except (ValueError, TypeError):
        return  # 未知实体，保持原样
    binary = decode_mask_pixels(mask_img, mask_id, id_scale, id_offset)

    # 几何 fallback：优先读已存的 geometry_*，否则用当前 bbox_*（首次运行即几何值）
    geo_xyxy = obj.get("geometry_bbox_xyxy") or obj.get("bbox_xyxy")
    geo_xywh = obj.get("geometry_bbox_xywh") or obj.get("bbox_xywh")

    if binary.any():
        bbox = mask_to_bbox(binary)
        xywh = xyxy_to_xywh(bbox)
        polys, comp_areas = mask_to_polygons_with_areas(
            binary, polygon_tolerance_px, max_polygon_points
        )
        n_components = len(polys)
        ring, meta = merge_to_single_ring(polys, comp_areas) if polys else ([], None)
        seg_flat = ring_to_yolo_flat(ring, width, height) if ring else None
        fallback = meta.get("fallback") if meta else None
        fallback_reason = meta.get("fallback_reason") if meta else None
        # 面积膨胀检查（仅多连通域且桥接成功时）
        if n_components > 1 and fallback is None and seg_flat:
            reason = _check_area_quality(ring, binary, width, height)
            if reason:
                best = max(range(len(polys)), key=lambda i: comp_areas[i])
                ring = list(polys[best])
                seg_flat = ring_to_yolo_flat(ring, width, height)
                fallback = "largest_component"
                fallback_reason = reason
        obj.update({
            "mask_id": mask_id,
            "bbox_source": BBOX_SOURCE_INSTANCE_MASK,
            "in_frame": True,
            "truncated": False,  # mask bbox 必在图像内
            "visibility": None,
            "visible_pixel_count": visible_pixel_count(binary),
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
    else:
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


def _write_yolo_labels(cam_dir: Path, frames: List[dict], width: int, height: int,
                       include_ball: bool) -> None:
    """写 YOLO detect / segment 标签到 labels/det/ 与 labels/seg/（与帧一一对应）。"""
    det_dir = cam_dir / "labels" / "det"
    seg_dir = cam_dir / "labels" / "seg"
    ensure_dir(det_dir)
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
            cid = yolo_class_id(obj["entity_id"])
            cx, cy, w, h = det_xyxy_to_yolo_norm(obj["bbox_xyxy"], width, height)
            det_lines.append(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            seg = obj.get("segmentation")
            if seg:
                seg_lines.append(f"{cid} " + " ".join(f"{v:.6f}" for v in seg))
        write_text_atomic(det_dir / f"{frame_index:06d}.txt", "\n".join(det_lines) + ("\n" if det_lines else ""))
        write_text_atomic(seg_dir / f"{frame_index:06d}.txt", "\n".join(seg_lines) + ("\n" if seg_lines else ""))
