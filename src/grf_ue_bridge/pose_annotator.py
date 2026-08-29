"""从 UE 导出的世界关键点生成 YOLO Pose 标签（P1 侧，纯 Python + numpy + PIL）。

数据流（在 UE 完成 keypoint 导出 + annotate-masks 之后运行）：
  读 pose_keypoints.jsonl（UE 写的世界 3D 关键点 + occluded 标志）
  + camera.json（真实内参/外参）+ annotations.jsonl（mask-primary bbox/track）
  + mask/{frame:06d}.png（Instance-ID Mask，可见性判定）
  → 逐相机逐帧把 17 个世界关键点投影到图像，判定 visibility
  → 写 labels_pose/{frame:06d}.txt（YOLO Pose，每行 56 字段）
  → 可选写 yolo_pose/（images+labels 硬链接暂存 + futsal_pose.yaml，可直接训练）

可见性规则（v = 0/1/2，见 docs/design/2026-08-11-yolo-pose-export.md）：
  - v=0：关键点 3D 无效 / 在相机后方 / 投影失败 / 明显位于图像有效区域之外。
  - v=1：投影位置被**其他实例**遮挡（Instance-ID Mask 邻域判定，真实渲染结果），
        或 UE 遮挡 trace 判定被几何（自遮挡 / 球 / 围挡等非 mask 几何）遮挡。
  - v=2：其余（关键点落在自身 mask 邻域或自由空间，且无遮挡）。

bbox 复用 annotations.jsonl 的 mask-primary bbox（与 YOLO det 完全一致），
不根据关键点重新生成。track_id / mask_id / 17 keypoints 始终来自同一球员。

调用入口：grf-ue annotate-pose <output_dir>（见 cli.py）；task postprocess 自动集成。
"""

import json
import math
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# 把仓库的 ue/ 目录加入 sys.path（与 tests/conftest.py 一致），以便 import 纯模块
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_UE_DIR = _REPO_ROOT / "ue"
if str(_UE_DIR) not in sys.path:
    sys.path.insert(0, str(_UE_DIR))

from annotation_utils import BBOX_SOURCE_INSTANCE_MASK  # noqa: E402
from camera_projection import (  # noqa: E402
    CameraExtrinsics,
    CameraIntrinsics,
    project_world_to_image,
)
from instance_mask import (  # noqa: E402
    det_xyxy_to_yolo_norm,
    load_mask_array,
    quantize_mask_pixels,
    yolo_class_id,
)
from pose_bones import (  # noqa: E402
    COCO_FLIP_IDX,
    COCO_KEYPOINT_NAMES,
    COCO_SKELETON_EDGES,
    NUM_COCO_KEYPOINTS,
    NOSE_SHOULDER_MIDPOINT_EDGE,
    YOLO_POSE_FIELDS,
    bone_index_of,
)
from dataset_export import ensure_dir, write_json_atomic, write_text_atomic  # noqa: E402

# YOLO Pose 标签目录名（与 labels/det、labels/seg 平级，不覆盖现有标签）
LABELS_POSE_DIRNAME = "labels_pose"

# YOLO Pose 直接可训练的数据集目录名（episode 根下）
YOLO_POSE_STAGE_DIRNAME = "yolo_pose"
POSE_YAML_FILENAME = "futsal_pose.yaml"

# 可见性枚举（YOLO Pose 约定）
VIS_INVALID = 0   # 点不存在 / 无效 / 无法可靠生成
VIS_OCCLUDED = 1  # 点存在但被遮挡
VIS_VISIBLE = 2   # 点存在且可见


@dataclass(frozen=True)
class PoseConfig:
    """annotate-pose 的不变参数（可 pickle，Windows spawn 安全）。"""

    visibility_neighborhood_radius: int = 2   # mask 邻域判定半径（像素）
    write_yaml: bool = True                    # 是否生成 yolo_pose/ 暂存 + YAML
    mask_channel: str = "r"
    id_scale: float = 1.0
    id_offset: float = 0.0


def _resolve_pose_config(pose_cfg: Optional[dict]) -> PoseConfig:
    pose_cfg = pose_cfg or {}
    return PoseConfig(
        visibility_neighborhood_radius=int(
            pose_cfg.get("visibility_neighborhood_radius", 2)),
        write_yaml=bool(pose_cfg.get("write_dataset_yaml", True)),
        mask_channel=str((pose_cfg.get("mask_channel")) or "r"),
        id_scale=float(pose_cfg.get("id_scale", 1.0)),
        id_offset=float(pose_cfg.get("id_offset", 0.0)),
    )


# ── pose_keypoints.jsonl 读取 ─────────────────────────────────────────

def read_pose_keypoints(cam_dir: Path) -> Tuple[Optional[dict], List[dict]]:
    """读取 camera 目录下的 pose_keypoints.jsonl。

    返回 (meta, frames)。首行 kind=="meta" 为元数据，其余 kind=="frame" 为帧。
    文件不存在返回 (None, [])。
    """
    p = cam_dir / "pose_keypoints.jsonl"
    if not p.exists():
        return None, []
    meta = None
    frames: List[dict] = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("kind") == "meta":
                meta = row
            elif row.get("kind") == "frame":
                frames.append(row)
    return meta, frames


def _frame_to_pose_objects(frame: dict) -> Dict[int, dict]:
    """按 track_id 索引一帧的 pose objects。"""
    out: Dict[int, dict] = {}
    for obj in frame.get("objects", []):
        tid = obj.get("track_id")
        if isinstance(tid, int):
            out[tid] = obj
    return out


def _frame_to_ann_objects(frame: dict) -> Dict[int, dict]:
    """按 track_id 索引一帧的 annotations objects。"""
    out: Dict[int, dict] = {}
    for obj in frame.get("objects", []):
        tid = obj.get("track_id")
        if isinstance(tid, int):
            out[tid] = obj
    return out


# ── 投影 ─────────────────────────────────────────────────────────────

def _load_camera(cam_dir: Path) -> Tuple[Optional[CameraIntrinsics], Optional[CameraExtrinsics],
                                         Optional[Tuple[int, int]]]:
    """从 camera.json 构造内参/外参，返回 (intrinsics, extrinsics, (width, height))。

    解析失败返回 (None, None, None)。
    """
    p = cam_dir / "camera.json"
    if not p.exists():
        return None, None, None
    try:
        cam = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None, None, None
    intr = cam.get("intrinsics") or {}
    extr = cam.get("extrinsics") or {}
    try:
        width = int(intr["width"])
        height = int(intr["height"])
        intrinsics = CameraIntrinsics(
            width=width, height=height,
            fx=float(intr["fx"]), fy=float(intr["fy"]),
            cx=float(intr["cx"]), cy=float(intr["cy"]),
        )
    except (KeyError, TypeError, ValueError):
        return None, None, None
    try:
        loc = [float(v) for v in extr["world_location_m"]]
        fwd = [float(v) for v in extr["forward"]]
        right = [float(v) for v in extr["right"]]
        up = [float(v) for v in extr["up"]]
        extrinsics = CameraExtrinsics(
            location=(loc[0], loc[1], loc[2]),
            forward=(fwd[0], fwd[1], fwd[2]),
            right=(right[0], right[1], right[2]),
            up=(up[0], up[1], up[2]),
        )
    except (KeyError, TypeError, ValueError, IndexError):
        return None, None, None
    return intrinsics, extrinsics, (width, height)


def _project_keypoints(kps_world: List[List[Optional[float]]],
                       intrinsics: CameraIntrinsics,
                       extrinsics: CameraExtrinsics) -> List[Optional[Tuple[float, float]]]:
    """把 17 个世界关键点投影到图像坐标（像素，浮点）。

    返回长度 17 的列表；投影失败（3D 无效 / 相机后方 / 非有限值）为 None。
    """
    out: List[Optional[Tuple[float, float]]] = []
    for kp in kps_world:
        if kp is None or len(kp) != 3 or kp[0] is None:
            out.append(None)
            continue
        try:
            uv = project_world_to_image((float(kp[0]), float(kp[1]), float(kp[2])),
                                        intrinsics, extrinsics)
        except (TypeError, ValueError):
            uv = None
        out.append(uv)
    return out


# ── visibility ───────────────────────────────────────────────────────

def _keypoint_visibility(
    uv: Optional[Tuple[float, float]],
    width: int,
    height: int,
    mask_ids,
    own_mask_id: int,
    occluded: Optional[bool],
    radius: int,
) -> int:
    """单个关键点的 visibility（0/1/2），见模块 docstring 规则。

    uv: 投影像素坐标；mask_ids: 整帧量化实例 ID 数组；own_mask_id: 该球员 mask_id；
    occluded: UE 遮挡 trace 标志（None 视为 False）。
    """
    if uv is None:
        return VIS_INVALID
    u, v = uv
    if not (math.isfinite(u) and math.isfinite(v)):
        return VIS_INVALID
    x, y = int(round(u)), int(round(v))
    if x < 0 or y < 0 or x >= width or y >= height:
        # 明显位于图像有效区域之外，无法作为有效标注使用
        return VIS_INVALID

    if mask_ids is None:
        # 无 mask 数据（overlay / 快速校验）：仅依赖 UE trace
        return VIS_OCCLUDED if occluded else VIS_VISIBLE

    # 邻域采样：避免关节点恰好落在人体轮廓边缘附近就误判遮挡。
    y0, y1 = max(0, y - radius), min(height - 1, y + radius)
    x0, x1 = max(0, x - radius), min(width - 1, x + radius)
    patch = mask_ids[y0:y1 + 1, x0:x1 + 1]
    center = int(mask_ids[y, x])
    own = int((patch == own_mask_id).sum())
    other = int(((patch > 0) & (patch != own_mask_id)).sum())

    # 1) 中心像素被其他实例（球员/球）占据 → 真实渲染结果判定遮挡
    if own_mask_id is not None and own_mask_id > 0 and center != 0 and center != own_mask_id:
        return VIS_OCCLUDED
    # 2) 中心是自身像素，或完全自由空间 → 可见（除非 UE trace 标记遮挡）
    if own > 0 or other == 0:
        return VIS_OCCLUDED if occluded else VIS_VISIBLE
    # 3) 中心为背景但邻域被其他实例填满（关键点紧贴遮挡物）→ 保守判遮挡
    return VIS_OCCLUDED


# ── 实例构建（labeler 与 overlay 共用）────────────────────────────────

def compute_pose_instances(
    pose_frame: dict,
    ann_frame: dict,
    mask_ids,
    intrinsics: CameraIntrinsics,
    extrinsics: CameraExtrinsics,
    width: int,
    height: int,
    radius: int,
) -> List[dict]:
    """为一帧构建 Pose 实例列表（对齐 pose 关键点与 annotations 的 bbox/track）。

    返回每个可见球员：
      {entity_id, track_id, mask_id, bbox_xyxy, bbox_xywh, keypoints: [(u, v, vis)]}

    只输出与 annotations.jsonl 中 bbox_source=="instance_mask"（可见像素 GT）对齐的
    球员——与 YOLO det 的导出集合一致，保证 bbox 与 detection 数据完全一致。
    """
    pose_objs = _frame_to_pose_objects(pose_frame)
    ann_objs = _frame_to_ann_objects(ann_frame)
    instances: List[dict] = []
    for tid, ann in ann_objs.items():
        if ann.get("class") == "ball":
            continue
        if ann.get("bbox_source") != BBOX_SOURCE_INSTANCE_MASK:
            continue  # 不可见（not_visible / legacy 几何）不进入 Pose（与 YOLO det 一致）
        if not ann.get("in_frame") or not ann.get("bbox_xyxy"):
            continue
        pose = pose_objs.get(tid)
        if pose is None:
            continue
        kps_world = pose.get("keypoints_world")
        if not kps_world or len(kps_world) != NUM_COCO_KEYPOINTS:
            continue
        occluded = pose.get("occluded")
        mask_id = ann.get("mask_id")
        uv_list = _project_keypoints(kps_world, intrinsics, extrinsics)
        kps = []
        has_valid = False
        for i, uv in enumerate(uv_list):
            occl = None
            if isinstance(occluded, list) and i < len(occluded):
                occl = bool(occluded[i])
            vis = _keypoint_visibility(
                uv, width, height, mask_ids, int(mask_id) if mask_id is not None else -1,
                occl, radius,
            )
            kps.append((uv[0] if uv is not None else None, uv[1] if uv is not None else None, vis))
            if vis != VIS_INVALID:
                has_valid = True
        if not has_valid:
            continue  # 无任何有效关键点，不产生无意义标签行
        instances.append({
            "entity_id": ann.get("entity_id"),
            "track_id": tid,
            "mask_id": mask_id,
            "bbox_xyxy": ann.get("bbox_xyxy"),
            "bbox_xywh": ann.get("bbox_xywh"),
            "keypoints": kps,
        })
    return instances


# ── 序列化（YOLO Pose txt）───────────────────────────────────────────

def _normalize_kp(px: Optional[float], py: Optional[float], width: int, height: int,
                  precision: int = 6) -> Tuple[float, float]:
    """把像素坐标归一化到 [0,1]（越界 clamp）。无效点返回 (0.0, 0.0)。"""
    if px is None or py is None:
        return 0.0, 0.0
    x = max(0.0, min(1.0, float(px) / width))
    y = max(0.0, min(1.0, float(py) / height))
    return round(x, precision), round(y, precision)


def serialize_pose_line(
    class_id: int,
    bbox_xyxy: List[float],
    width: int,
    height: int,
    keypoints: List[Tuple[Optional[float], Optional[float], int]],
    precision: int = 6,
) -> str:
    """序列化一行 YOLO Pose：class xc yc w h x1 y1 v1 ... x17 y17 v17（56 字段）。

    bbox 由 bbox_xyxy 归一化（与 YOLO det 完全一致，复用 det_xyxy_to_yolo_norm）。
    keypoints 为 17 个 (px, py, visibility)，px/py 为像素坐标（None = 无效）。
    """
    if len(keypoints) != NUM_COCO_KEYPOINTS:
        raise ValueError(
            f"关键点数 {len(keypoints)} != {NUM_COCO_KEYPOINTS}（COCO 17 点）"
        )
    cx, cy, w, h = det_xyxy_to_yolo_norm(bbox_xyxy, width, height, precision)
    fields = [str(int(class_id)), f"{cx:.6f}", f"{cy:.6f}", f"{w:.6f}", f"{h:.6f}"]
    for px, py, vis in keypoints:
        nx, ny = _normalize_kp(px, py, width, height, precision)
        fields.append(f"{nx:.6f}")
        fields.append(f"{ny:.6f}")
        fields.append(str(int(vis)))
    return " ".join(fields)


def instance_to_line(inst: dict, width: int, height: int, precision: int = 6) -> str:
    """把 Pose 实例序列化为 YOLO Pose 行。"""
    return serialize_pose_line(
        yolo_class_id(inst["entity_id"]),
        inst["bbox_xyxy"], width, height,
        inst["keypoints"], precision,
    )


# ── 单 camera 处理 ───────────────────────────────────────────────────

def _read_mask_config(cam_dir: Path) -> dict:
    """读取 annotate-masks 写入的 mask_config.json（解码参数）。"""
    cfg_path = cam_dir / "mask_config.json"
    if cfg_path.exists():
        try:
            return json.loads(cfg_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _annotate_pose_camera(cam_dir: Path, config: PoseConfig) -> Tuple[bool, List[str]]:
    """处理单个 camera：写 labels_pose/{frame}.txt。返回 (ok, 错误列表)。"""
    errors: List[str] = []
    cam_label = cam_dir.name

    intrinsics, extrinsics, size = _load_camera(cam_dir)
    if intrinsics is None or size is None:
        return False, [f"[{cam_label}] camera.json 缺失或内参/外参非法"]

    meta, pose_frames = read_pose_keypoints(cam_dir)
    if meta is None:
        return False, [f"[{cam_label}] 缺少 pose_keypoints.jsonl（先启用 "
                       f"postprocess.yolo_pose 并在 UE 运行 --mode annotations/full）"]
    width, height = size

    ann_path = cam_dir / "annotations.jsonl"
    if not ann_path.exists():
        return False, [f"[{cam_label}] 缺少 annotations.jsonl（先运行 annotate-masks）"]
    ann_by_frame: Dict[int, dict] = {}
    with open(ann_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                fr = json.loads(line)
            except json.JSONDecodeError:
                continue
            fi = fr.get("frame_index")
            if isinstance(fi, int):
                ann_by_frame[fi] = fr

    # mask 解码参数：优先用 mask_config.json（annotate-masks 写入），否则用 config
    decode = _read_mask_config(cam_dir)
    mask_channel = decode.get("mask_channel") or config.mask_channel
    id_scale = float(decode.get("id_scale", config.id_scale))
    id_offset = float(decode.get("id_offset", config.id_offset))
    mask_dir = cam_dir / "mask"
    mask_available = mask_dir.is_dir()

    out_dir = cam_dir / LABELS_POSE_DIRNAME
    ensure_dir(out_dir)
    n_frames = 0
    n_instances = 0
    for pose_frame in pose_frames:
        fi = pose_frame.get("frame_index")
        if not isinstance(fi, int):
            errors.append(f"[{cam_label}] pose 帧缺 frame_index: {pose_frame!r}")
            continue
        ann_frame = ann_by_frame.get(fi)
        if ann_frame is None:
            # 该帧不在 annotations 中（帧范围不一致）——给出提示但不视为致命
            continue

        mask_ids = None
        if mask_available:
            mask_path = mask_dir / f"{fi:06d}.png"
            if mask_path.exists():
                try:
                    mask_img = load_mask_array(mask_path, mask_channel)
                    mask_ids = quantize_mask_pixels(mask_img, id_scale, id_offset)
                except Exception as e:  # noqa: BLE001
                    errors.append(f"[{cam_label}] mask {fi} 读取失败: {e}")
                    mask_ids = None

        instances = compute_pose_instances(
            pose_frame, ann_frame, mask_ids, intrinsics, extrinsics,
            width, height, config.visibility_neighborhood_radius,
        )
        lines = [instance_to_line(inst, width, height) for inst in instances]
        write_text_atomic(out_dir / f"{fi:06d}.txt", "\n".join(lines) + ("\n" if lines else ""))
        n_frames += 1
        n_instances += len(instances)

    print(f"  [OK] {cam_label}: {n_frames} 帧 / {n_instances} 实例 -> "
          f"{out_dir.name}/（{len(errors)} 警告）")
    return not errors, errors


def _annotate_pose_task(task: tuple) -> Tuple[str, bool, List[str]]:
    """进程池 worker：处理单个 camera。返回 (cam_str, ok, errors)。"""
    cam_str, cfg_dict = task
    config = PoseConfig(**cfg_dict)
    ok, errors = _annotate_pose_camera(Path(cam_str), config)
    return cam_str, ok, errors


def _camera_dirs(annotation_dir: Path) -> List[Path]:
    """递归找出所有含 camera.json 的 camera 子目录（与 annotation_validator 一致）。"""
    return sorted(d.parent for d in annotation_dir.rglob("camera.json"))


def annotate_pose_dir(
    annotation_dir: Path,
    pose_cfg: Optional[dict] = None,
    workers: int = 0,
    write_yaml: bool = True,
) -> int:
    """对输出目录下所有 camera 子目录生成 YOLO Pose 标签。返回退出码（0 成功 / 1 失败）。

    pose_cfg: postprocess.yolo_pose 块（enabled 外的可调参数）。
    workers: 0=自动（min(相机数, cpu//2)），1=串行，>1=相机级多进程。
    write_yaml: 是否在 episode 根生成 yolo_pose/ 暂存目录 + futsal_pose.yaml。
    """
    config = _resolve_pose_config(pose_cfg)
    cam_dirs = _camera_dirs(annotation_dir)
    if not cam_dirs:
        print(f"ERROR: {annotation_dir} 下没有 camera 子目录（缺少 camera.json）")
        return 1

    nworkers = workers
    if nworkers == 0:
        nworkers = min(max(1, len(cam_dirs)), max(1, (os.cpu_count() or 1) // 2))
    cfg_dict = {
        "visibility_neighborhood_radius": config.visibility_neighborhood_radius,
        "write_yaml": config.write_yaml,
        "mask_channel": config.mask_channel,
        "id_scale": config.id_scale,
        "id_offset": config.id_offset,
    }

    failures = 0
    if nworkers <= 1 or len(cam_dirs) <= 1:
        for cam_dir in cam_dirs:
            ok, _errs = _annotate_pose_camera(cam_dir, config)
            if not ok:
                failures += 1
    else:
        tasks = [(str(cam_dir), cfg_dict) for cam_dir in cam_dirs]
        with ProcessPoolExecutor(max_workers=nworkers) as ex:
            results = list(ex.map(_annotate_pose_task, tasks))
        for cam_str, ok, errs in results:
            if not ok:
                failures += 1
                for e in errs:
                    print(f"  ERROR: {e}")

    if failures:
        print(f"annotate-pose 完成，但有 {failures} 个 camera 目录失败")
        return 1

    # 生成数据集 YAML + 可训练暂存目录（episode 级，一次）
    if write_yaml and cam_dirs:
        episode_root = cam_dirs[0].parent
        camera_names = [c.name for c in cam_dirs]
        write_pose_dataset(episode_root, camera_names)
    print("annotate-pose 完成")
    return 0


# ── 数据集 YAML + 可训练暂存目录 ─────────────────────────────────────

def _hardlink_or_copy(src: Path, dst: Path) -> None:
    """优先硬链接（同卷，省磁盘），失败回退复制。幂等：已存在的目标先删除。

    重跑 annotate-pose 时 staging 已存在（可能为上次的硬链接/副本），先 unlink
    再重建，保证可重复执行不报 SameFileError / FileExistsError。
    """
    ensure_dir(dst.parent)
    if dst.exists():
        try:
            dst.unlink()
        except OSError:
            pass
    try:
        os.link(str(src), str(dst))
    except OSError:
        shutil.copy2(str(src), str(dst))


def _write_pose_yaml(episode_root: Path, stage_dir: Path) -> Path:
    """写 futsal_pose.yaml。

    C6-P1.7 zero-waste：不生成 yolo_pose/images 副本，yaml 的 path 指向 episode 根，
    注释说明 images 位于各 camera 的 img1。labels 位于 yolo_pose/labels/<cam>/。
    """
    data = {
        "path": episode_root.resolve().as_posix(),
        "names": {0: "player"},
        "kpt_shape": [NUM_COCO_KEYPOINTS, 3],
        "flip_idx": COCO_FLIP_IDX,
    }
    lines = [f"path: {data['path']}",
             "# zero-waste: 不复制 RGB。images 为各 camera 的 img1（<episode>/<camera>/img1）；",
             "# 训练时请把 images 指向这些 img1 目录（labels 位于 yolo_pose/labels/<camera>/）。",
             "names:", "  0: player",
             f"kpt_shape: [{NUM_COCO_KEYPOINTS}, 3]",
             "flip_idx: [" + ", ".join(str(i) for i in COCO_FLIP_IDX) + "]", ""]
    path = episode_root / POSE_YAML_FILENAME
    write_text_atomic(path, "\n".join(lines))
    stage_yaml = stage_dir / POSE_YAML_FILENAME
    write_text_atomic(stage_yaml, "\n".join(lines))
    return path


def write_pose_dataset(episode_root: Path, camera_names: List[str]) -> Path:
    """在 episode 根生成 yolo_pose/ 暂存目录（labels）与 futsal_pose.yaml。

    C6-P1.7 zero-waste：**不复制/硬链 RGB 到 yolo_pose/images**——img1 是唯一 RGB
    物理副本，YOLO 训练直接引用各 camera 的 img1（见 yaml 注释）。
    yolo_pose/ 只保留 labels + 指向 img1 的 yaml。
    """
    stage = episode_root / YOLO_POSE_STAGE_DIRNAME
    for cam in camera_names:
        src_lbl = episode_root / cam / LABELS_POSE_DIRNAME
        dst_lbl = stage / "labels" / cam
        if src_lbl.is_dir():
            ensure_dir(dst_lbl)
            for p in sorted(src_lbl.glob("*.txt")):
                shutil.copy2(str(p), str(dst_lbl / p.name))
    print(f"  YOLO Pose 可训练数据集: {stage}（labels；images 引用各 camera img1）")
    return _write_pose_yaml(episode_root, stage)


# ── 可视化 Debug（overlay）───────────────────────────────────────────

# visibility → 绘制颜色：v=2 绿、v=1 橙、v=0 红
_VIS_COLORS = {
    VIS_VISIBLE: (0, 200, 0),
    VIS_OCCLUDED: (255, 160, 0),
    VIS_INVALID: (255, 40, 40),
}


def _edge_indices(a: str, b: str) -> Tuple[int, int]:
    return bone_index_of(a), bone_index_of(b)


def draw_pose_overlay(img, instances: List[dict], skeleton_edges: Optional[List[Tuple[str, str]]] = None,
                      draw_midpoint: bool = True, dot_radius: int = 3,
                      edge_width: int = 2, keypoint_names: bool = False):
    """在 RGB 帧上绘制 17 关键点、骨架连线、bbox、track_id。返回新 PIL Image。

    关键点颜色区分 visibility（绿=可见 / 橙=遮挡 / 红=无效）；骨架连线只连接
    v>0 的点。bbox 与 track_id 沿用 annotations 的 mask-primary bbox。
    dot_radius / edge_width 可调（高清/放大预览用）；keypoint_names=True 时给每个
    关键点标注 COCO 名（调试用）。
    """
    from PIL import ImageDraw

    edges = skeleton_edges if skeleton_edges is not None else COCO_SKELETON_EDGES
    img = img.convert("RGB")
    draw = ImageDraw.Draw(img)

    def _coord(kp) -> Optional[Tuple[float, float]]:
        u, v, _vis = kp
        if u is None or v is None:
            return None
        return (float(u), float(v))

    for inst in instances:
        kps = inst["keypoints"]
        pts = [_coord(k) for k in kps]
        # 骨架连线（只连 v>0 的点）
        for a, b in edges:
            ia, ib = _edge_indices(a, b)
            pa, pb = pts[ia], pts[ib]
            if pa is None or pb is None:
                continue
            if kps[ia][2] == VIS_INVALID or kps[ib][2] == VIS_INVALID:
                continue
            draw.line([pa, pb], fill=(180, 180, 185), width=edge_width)
        # 可选：nose → 双肩中点
        if draw_midpoint and "nose" in COCO_KEYPOINT_NAMES and \
                "left_shoulder" in COCO_KEYPOINT_NAMES and "right_shoulder" in COCO_KEYPOINT_NAMES:
            inose = bone_index_of("nose")
            ils = bone_index_of("left_shoulder")
            irs = bone_index_of("right_shoulder")
            pn, pls, prs = pts[inose], pts[ils], pts[irs]
            if pn is not None and pls is not None and prs is not None and \
                    kps[inose][2] != VIS_INVALID:
                mid = ((pls[0] + prs[0]) / 2.0, (pls[1] + prs[1]) / 2.0)
                draw.line([pn, mid], fill=(180, 180, 185), width=edge_width)
        # 关键点（先画黑描边大点，再画彩色内核，保证在亮色/暗色背景上都可见）
        for i, kp in enumerate(kps):
            p = pts[i]
            if p is None:
                continue
            color = _VIS_COLORS.get(kp[2], (255, 255, 255))
            r = dot_radius if kp[2] != VIS_INVALID else max(2, dot_radius - 1)
            x, y = p
            draw.ellipse([x - r - 1, y - r - 1, x + r + 1, y + r + 1], fill=(0, 0, 0))
            draw.ellipse([x - r, y - r, x + r, y + r], fill=color)
        # 关键点名（可选，调试）
        if keypoint_names:
            for i, kp in enumerate(kps):
                p = pts[i]
                if p is None or kp[2] == VIS_INVALID:
                    continue
                x, y = p
                draw.text((x + dot_radius + 1, y - dot_radius), COCO_KEYPOINT_NAMES[i],
                          fill=(255, 255, 255), font=None)
        # bbox + track_id
        bbox = inst.get("bbox_xyxy")
        if bbox:
            xmin, ymin, xmax, ymax = (float(v) for v in bbox)
            draw.rectangle([xmin, ymin, xmax, ymax], outline=(60, 120, 255), width=2)
            label = f"{inst.get('entity_id')} #{inst.get('track_id')}"
            draw.text((xmin, max(0, ymin - 14)), label, fill=(60, 120, 255))
    return img


def pose_overlay_dir(
    cam_dir: Path,
    frames: Optional[List[int]] = None,
    out_dir: Optional[Path] = None,
    visibility_neighborhood_radius: int = 2,
    write_jpeg: bool = False,
    dot_radius: int = 3,
    edge_width: int = 2,
    keypoint_names: bool = False,
) -> int:
    """把 Pose 关键点/骨架/bbox 画到 img1/ 的 RGB 帧上，输出到 debug/pose/。

    frames: 只处理这些帧号（1 基）；None = 全部。返回绘制的帧数。
    dot_radius / edge_width / keypoint_names：透传给 draw_pose_overlay（放大预览用）。
    需要 pillow（项目已依赖）。
    """
    try:
        from PIL import Image
    except ImportError:
        print("需要 pillow：请运行 `uv sync --extra overlay` 或 `uv pip install pillow`")
        return 0

    intrinsics, extrinsics, size = _load_camera(cam_dir)
    if intrinsics is None or size is None:
        print(f"ERROR: camera.json 缺失或非法: {cam_dir}")
        return 0
    width, height = size
    meta, pose_frames = read_pose_keypoints(cam_dir)
    if meta is None:
        print(f"ERROR: 缺少 pose_keypoints.jsonl: {cam_dir}")
        return 0
    ann_by_frame: Dict[int, dict] = {}
    ann_path = cam_dir / "annotations.jsonl"
    if ann_path.exists():
        with open(ann_path, encoding="utf-8") as f:
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

    decode = _read_mask_config(cam_dir)
    mask_channel = decode.get("mask_channel", "r")
    id_scale = float(decode.get("id_scale", 1.0))
    id_offset = float(decode.get("id_offset", 0.0))
    mask_dir = cam_dir / "mask"

    img_dir = cam_dir / "img1"
    out = out_dir or (cam_dir / "debug" / "pose")
    ensure_dir(out)
    drawn = 0
    for pose_frame in pose_frames:
        fi = pose_frame.get("frame_index")
        if not isinstance(fi, int):
            continue
        if frames is not None and fi not in frames:
            continue
        img_path = img_dir / f"{fi:06d}.png"
        if not img_path.exists():
            continue
        ann_frame = ann_by_frame.get(fi)
        if ann_frame is None:
            continue
        mask_ids = None
        if mask_dir.is_dir():
            mask_path = mask_dir / f"{fi:06d}.png"
            if mask_path.exists():
                try:
                    mask_img = load_mask_array(mask_path, mask_channel)
                    mask_ids = quantize_mask_pixels(mask_img, id_scale, id_offset)
                except Exception:  # noqa: BLE001
                    mask_ids = None
        instances = compute_pose_instances(
            pose_frame, ann_frame, mask_ids, intrinsics, extrinsics,
            width, height, visibility_neighborhood_radius,
        )
        if not instances:
            continue
        with Image.open(img_path) as im:
            overlay = draw_pose_overlay(im, instances, dot_radius=dot_radius,
                                        edge_width=edge_width, keypoint_names=keypoint_names)
        suffix = ".jpg" if write_jpeg else ".png"
        overlay.save(out / f"{fi:06d}{suffix}")
        drawn += 1
    print(f"Pose overlay 完成: {drawn} 帧 -> {out}")
    return drawn
