"""UE 内运行的标注导出器：读取 CineCamera 标定与 Actor 世界 bounds，生成 CV 标注。

依赖：
  - unreal（UE 内置，一律延迟 import）
  - camera_projection / annotation_utils / dataset_export / scene_apply（同目录模块）

流程：
  1. 读取 episode（meta.json + frames.jsonl）与 actor 映射。
  2. 对每个选中的 Camera 读取真实标定（世界变换、焦距、传感器）。
  3. 逐帧应用与 Level Sequence 相同的 actor 变换（scene_apply）。
  4. 对每个 actor 读取世界空间 AABB → 8 角点 → 投影 → 裁剪 → 构造标注。
  5. 输出每个 Camera 的 camera.json / annotations.jsonl / MOT gt.txt / seqinfo.ini。
"""

import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# 保证在 UE 中运行时能 import 同目录的纯模块
sys.path.insert(0, str(Path(__file__).resolve().parent))

from annotation_utils import (
    analyze_bbox,
    entity_class,
    entity_id_to_track_id,
    entity_team,
    xyxy_to_xywh,
)
from camera_projection import (
    CameraExtrinsics,
    CameraIntrinsics,
    compute_intrinsics_from_focal_length,
    compute_intrinsics_from_vertical_fov,
    focal_length_to_fov_deg,
    project_box_corners_to_image_xyxy,
    world_bbox_corners,
)
from dataset_export import (
    build_mot_gt,
    build_seqinfo,
    ensure_dir,
    load_episode,
    load_mapping,
    write_json_atomic,
    write_jsonl_atomic,
    write_text_atomic,
)
from scene_apply import (
    apply_preview_frame,
    find_all_actors,
    find_actor,
)

CM_TO_M = 0.01  # 厘米 -> 米


def _v3(v):
    """把 unreal.Vector / 带 x/y/z 的对象转成 (float, float, float)。"""
    return (float(v.x), float(v.y), float(v.z))


def _mesh_world_bounds(actor):
    """用 actor 网格组件（Skeletal/Static Mesh）的本地 bounds 换算世界 AABB。

    部分 actor 的 get_actor_bounds 会并入超大组件/子 actor 而偏大（例如把远处
    子 actor 也算进来），改用网格组件自身的 bounds 更可靠。
    返回 (origin, extent_half) 或 None（没有网格组件 / API 不可用）。
    """
    import unreal

    comp = None
    for cls in (unreal.SkeletalMeshComponent, unreal.StaticMeshComponent):
        comps = actor.get_components_by_class(cls)
        if comps:
            comp = comps[0]
            break
    if comp is None:
        return None
    try:
        local_origin, local_extent = comp.get_local_bounds()
        world_tf = comp.get_component_to_world_transform()
        lo, le = _v3(local_origin), _v3(local_extent)
        xs, ys, zs = [], [], []
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                for sz in (-1.0, 1.0):
                    p = unreal.MathLibrary.transform_location(
                        world_tf,
                        unreal.Vector(
                            lo[0] + sx * le[0],
                            lo[1] + sy * le[1],
                            lo[2] + sz * le[2],
                        ),
                    )
                    xs.append(float(p.x))
                    ys.append(float(p.y))
                    zs.append(float(p.z))
    except Exception:
        return None
    origin = ((min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0, (min(zs) + max(zs)) / 2.0)
    extent = ((max(xs) - min(xs)) / 2.0, (max(ys) - min(ys)) / 2.0, (max(zs) - min(zs)) / 2.0)
    return origin, extent


def get_world_bounds(actor):
    """获取 actor 在世界空间的 AABB（用于 2D bbox）。

    返回 (origin(x,y,z), extent_half(x,y,z))，单位为 UE 世界单位（cm）。

    优先级：
      1. Character 的 CapsuleComponent（稳定、近似人形体积）。球员只有 yaw 旋转
         （直立），胶囊 AABB 在 X/Y/Z 上仍轴对齐，可直接用 (r, r, half_height)。
      2. Skeletal/Static Mesh 组件的本地 bounds 换算世界 AABB。
      3. get_actor_bounds(False, False) 兜底。
    """
    import unreal

    capsules = actor.get_components_by_class(unreal.CapsuleComponent)
    if capsules:
        cap = capsules[0]
        radius = float(cap.get_scaled_capsule_radius())
        half_height = float(cap.get_scaled_capsule_half_height())
        loc = cap.get_world_location()
        return _v3(loc), (radius, radius, half_height)

    mesh = _mesh_world_bounds(actor)
    if mesh is not None:
        return mesh

    # 注意：get_actor_bounds 在不同 UE 版本返回形态不同（BoxSphereBounds 结构体
    # 或 (origin, box_extent) 元组），这里做兼容处理。
    result = actor.get_actor_bounds(False, False)
    if hasattr(result, "origin"):
        origin = result.origin
        extent = result.box_extent
    else:
        origin, extent = result

    return _v3(origin), _v3(extent)


def _actor_basis(actor):
    """获取 actor 的 forward/right/up 基向量（世界坐标，元组）。

    优先用 actor 自身的基向量方法（与渲染用同一套定义）；不可用时尝试
    unreal.MathLibrary（不同 UE 版本数学库名称不同，unreal.KismetMathLibrary
    可能不存在）。
    """
    import unreal

    try:
        return (
            _v3(actor.get_actor_forward_vector()),
            _v3(actor.get_actor_right_vector()),
            _v3(actor.get_actor_up_vector()),
        )
    except Exception:
        pass
    try:
        rot = actor.get_actor_rotation()
        return (
            _v3(unreal.MathLibrary.get_forward_vector(rot)),
            _v3(unreal.MathLibrary.get_right_vector(rot)),
            _v3(unreal.MathLibrary.get_up_vector(rot)),
        )
    except Exception:
        pass
    raise RuntimeError(
        f"Actor {actor.get_actor_label() or actor.get_name()}：无法获取基向量"
        f"（get_actor_forward_vector / unreal.MathLibrary 均不可用）"
    )


def _read_property_float(component, prop_names: List[str]) -> Optional[float]:
    """按顺序尝试读取组件属性（浮点）。属性名在不同 UE 版本可能不同。"""
    for name in prop_names:
        try:
            val = component.get_editor_property(name)
            if val is not None:
                fv = float(val)
                if math.isfinite(fv) and fv > 0:
                    return fv
        except Exception:
            continue
    return None


def _read_sensor_size(component, image_width: int, image_height: int):
    """读取 CineCamera 传感器尺寸（mm）。返回 (sensor_w, sensor_h) 或 None。

    filmback 属性名与字段在不同 UE 版本可能不同，这里做多级 fallback。
    """
    filmback = None
    for name in ("filmback", "filmback_settings"):
        try:
            filmback = component.get_editor_property(name)
            if filmback is not None:
                break
        except Exception:
            continue
    if filmback is None:
        return None
    try:
        sw = float(filmback.sensor_width)
        if sw <= 0:
            return None
        sh = float(filmback.sensor_height)
        if sh <= 0:
            try:
                sh = sw / float(filmback.sensor_aspect_ratio)
            except Exception:
                sh = sw * image_height / image_width
        if sh <= 0:
            sh = sw * image_height / image_width
        return sw, sh
    except Exception:
        return None


def read_camera_calibration(camera_actor, image_width: int, image_height: int):
    """从 CineCamera actor 读取相机标定。

    读取实际世界变换（位置/旋转）、forward/right/up 基向量，以及焦距（mm）
    与传感器尺寸（mm），换算成像素内参。返回
    (intrinsics, extrinsics, cam_meta)。

    UE 世界单位为 cm；本模块统一换算为米后再投影（投影对整体单位缩放不变）。
    焦距/filmback 属性名在不同 UE 版本可能不同，做多级 fallback；
    若 filmback 缺失，回退用 current_fov（垂直 FOV）推导内参。
    """
    import unreal

    comps = camera_actor.get_components_by_class(unreal.CineCameraComponent)
    if not comps:
        raise RuntimeError(
            f"Camera actor '{camera_actor.get_actor_label() or camera_actor.get_name()}' "
            f"没有 CineCameraComponent"
        )
    comp = comps[0]

    # 相机基向量：用 actor 自身的基向量方法（与渲染同一套定义）。
    # 注意：unreal.KismetMathLibrary 在不同 UE 版本可能不可用，故不依赖它。
    # CineCamera 组件通常与 actor 对齐，相机前向 == actor 前向。
    loc = camera_actor.get_actor_location()
    rot = camera_actor.get_actor_rotation()
    fwd, right, up = _actor_basis(camera_actor)
    location_m = tuple(v * CM_TO_M for v in _v3(loc))

    focal_mm = _read_property_float(
        comp, ("current_focal_length", "focal_length")
    )
    sensor = _read_sensor_size(comp, image_width, image_height)

    if sensor is not None and focal_mm is not None:
        sensor_w, sensor_h = sensor
        intrinsics = compute_intrinsics_from_focal_length(
            focal_mm, sensor_w, sensor_h, image_width, image_height
        )
        fov_h = focal_length_to_fov_deg(focal_mm, sensor_w)
        fov_v = focal_length_to_fov_deg(focal_mm, sensor_h)
        sensor_list = [round(sensor_w, 6), round(sensor_h, 6)]
    elif sensor is None and focal_mm is not None:
        # fallback：用 CineCamera 的 current_fov（垂直 FOV）+ 图像宽高比推导
        fov_v = _read_property_float(comp, ("current_fov", "current_fov_deg", "fov"))
        if fov_v is None:
            raise RuntimeError(
                f"Camera '{camera_actor.get_actor_label() or camera_actor.get_name()}': "
                f"无法读取 filmback 传感器尺寸，也没有 current_fov 可回退"
            )
        intrinsics = compute_intrinsics_from_vertical_fov(
            fov_v, image_width, image_height
        )
        fov_h = 2 * math.degrees(
            math.atan(math.tan(math.radians(fov_v) / 2) * (image_width / image_height))
        )
        sensor_list = None
    else:
        raise RuntimeError(
            f"Camera '{camera_actor.get_actor_label() or camera_actor.get_name()}': "
            f"无法读取焦距（current_focal_length / focal_length）"
        )

    extrinsics = CameraExtrinsics(location_m, fwd, right, up)

    cam_meta = {
        "camera_id": camera_actor.get_actor_label() or camera_actor.get_name(),
        "image_width": image_width,
        "image_height": image_height,
        "intrinsics": {
            "width": image_width,
            "height": image_height,
            "fx": round(intrinsics.fx, 6),
            "fy": round(intrinsics.fy, 6),
            "cx": round(intrinsics.cx, 6),
            "cy": round(intrinsics.cy, 6),
        },
        "extrinsics": {
            "world_location_m": [round(v, 6) for v in location_m],
            "world_rotation_deg": [
                round(float(rot.pitch), 6),
                round(float(rot.yaw), 6),
                round(float(rot.roll), 6),
            ],
            "forward": [round(v, 6) for v in fwd],
            "right": [round(v, 6) for v in right],
            "up": [round(v, 6) for v in up],
            "matrix_direction": "camera = R @ (world - location)，R 行向量为 forward/right/up",
        },
        "focal_length_mm": round(focal_mm, 6) if focal_mm is not None else None,
        "sensor_size_mm": sensor_list,
        "horizontal_fov_deg": round(fov_h, 6),
        "vertical_fov_deg": round(fov_v, 6),
        "units": {
            "world": "meters（UE 厘米除以 100）",
            "image": "pixels，原点在左上角，x 向右 y 向下",
        },
        "coordinate_convention": (
            "unreal: 左手系 X 前 Y 右 Z 上；camera: X=前向 Y=右向 Z=上向；"
            "image: 原点左上，x 右 y 下；投影 u=cx+fx*(y_cam/x_cam), v=cy-fy*(z_cam/x_cam)"
        ),
    }
    return intrinsics, extrinsics, cam_meta


def build_object_annotation(
    entity_id: str,
    actor,
    entity_info: dict,
    intrinsics: CameraIntrinsics,
    extrinsics: CameraExtrinsics,
    image_width: int,
    image_height: int,
    ball_radius_m: float = None,
) -> dict:
    """为一个 actor 构造 CV 标注 dict。

    bbox 来自 actor 世界空间 AABB 的 8 个角点投影（非固定尺寸），因此远处
    球员 bbox 视觉上更小、近处更大。所有 bbox 数值保留 3 位小数。

    球：若传入 ball_radius_m，直接用该半径生成球 bbox（不依赖 mesh bounds），
    用于球 mesh 资产包围盒数据异常的情况。
    """
    if entity_id == "BALL" and ball_radius_m is not None:
        loc = actor.get_actor_location()
        r_cm = float(ball_radius_m) * 100.0
        origin_cm = (float(loc.x), float(loc.y), float(loc.z))
        extent_cm = (r_cm, r_cm, r_cm)
    else:
        origin_cm, extent_cm = get_world_bounds(actor)
    origin_m = tuple(v * CM_TO_M for v in origin_cm)
    extent_m = tuple(v * CM_TO_M for v in extent_cm)
    corners = world_bbox_corners(origin_m, extent_m)
    xyxy = project_box_corners_to_image_xyxy(corners, intrinsics, extrinsics)

    loc = actor.get_actor_location()
    world_position = [round(loc.x * CM_TO_M, 6), round(loc.y * CM_TO_M, 6), round(loc.z * CM_TO_M, 6)]

    info = entity_info.get(entity_id, {})
    base = {
        "entity_id": entity_id,
        "track_id": entity_id_to_track_id(entity_id),
        "class": entity_class(entity_id),
        "team": entity_team(entity_id),
        "role": info.get("role"),
        "is_goalkeeper": bool(info.get("is_goalkeeper", False)),
        "world_position": world_position,
    }

    def _not_in_frame():
        base.update({
            "in_frame": False,
            "truncated": False,
            "visibility": None,
            "raw_bbox_xywh": None,
            "raw_bbox_xyxy": None,
            "bbox_xywh": None,
            "bbox_xyxy": None,
        })
        return base

    if xyxy is None:
        return _not_in_frame()

    xmin, ymin, xmax, ymax = xyxy
    res = analyze_bbox(xmin, ymin, xmax, ymax, image_width, image_height)
    if not res["in_frame"]:
        return _not_in_frame()

    raw_xywh = xyxy_to_xywh((xmin, ymin, xmax, ymax))
    clipped_xyxy = res["clipped_xyxy"]
    clipped_xywh = res["clipped_xywh"]
    base.update({
        "in_frame": True,
        "truncated": res["truncated"],
        "visibility": None,
        "raw_bbox_xywh": [round(v, 3) for v in raw_xywh],
        "raw_bbox_xyxy": [round(v, 3) for v in (xmin, ymin, xmax, ymax)],
        "bbox_xywh": [round(v, 3) for v in clipped_xywh],
        "bbox_xyxy": [round(v, 3) for v in clipped_xyxy],
    })
    return base


def export_annotations(
    episode_dir: Path,
    mapping_path: Path,
    output_dir: Path,
    cfg: dict,
) -> None:
    """导出 CV 标注。

    cfg 关键字段：
      image_width / image_height : 输出图像分辨率
      cameras                    : 要导出的 Camera actor 名称列表
      frame_start / frame_end    : 0 基 GRF step 范围（frame_end 为开区间）
      export_internal_jsonl      : 是否导出内部 annotations.jsonl
      export_mot                 : 是否导出 MOTChallenge gt.txt / seqinfo.ini
      include_ball               : MOT 是否包含球
      mot_visibility_mode        : "unoccluded" 或 "truncation"
      ball_scale                 : 球 actor 缩放（须与 Level Sequence bake 一致，默认 0.5）
    """
    import unreal

    meta, frames = load_episode(episode_dir)
    mapping = load_mapping(mapping_path)
    actors = find_all_actors(mapping)
    if not actors:
        return

    image_width = int(cfg.get("image_width", 1920))
    image_height = int(cfg.get("image_height", 1080))
    camera_names = cfg.get("cameras") or []
    if not camera_names:
        print("WARNING: annotation_export.cameras 为空，跳过标注导出")
        return
    frame_start = int(cfg.get("frame_start", 0))
    frame_end = cfg.get("frame_end")
    include_ball = bool(cfg.get("include_ball", False))
    visibility_mode = cfg.get("mot_visibility_mode", "unoccluded")
    export_mot = bool(cfg.get("export_mot", True))
    export_internal_jsonl = bool(cfg.get("export_internal_jsonl", True))
    ball_scale = cfg.get("ball_scale")  # None = 不覆盖球的 scale
    ball_radius_m = cfg.get("ball_radius_m")  # None = 用 mesh bounds；否则用该半径生成球 bbox
    mot_frame_rate = int(meta["timing"].get("playback_fps", 30))
    source_step_seconds = float(meta["timing"].get("source_step_seconds", 0.1))
    episode_id = meta.get("episode_id") or episode_dir.name

    # 实体元信息（role / is_goalkeeper 来自 meta.json 的 entities）
    entity_info = {}
    for e in meta.get("entities", []):
        entity_info[e.get("id")] = e

    # 帧范围（0 基 GRF step）
    selected = [f for f in frames if f["step"] >= frame_start]
    if frame_end is not None:
        selected = [f for f in selected if f["step"] < int(frame_end)]
    if not selected:
        print("ERROR: 没有选中任何帧")
        return

    # 读取 Camera 标定
    cameras = []
    for name in camera_names:
        cam = find_actor(name)
        if not cam:
            print(f"  WARNING: Camera actor '{name}' 未找到，跳过")
            continue
        intrinsics, extrinsics, cam_meta = read_camera_calibration(cam, image_width, image_height)
        cameras.append((name, cam, intrinsics, extrinsics, cam_meta))
    if not cameras:
        print("ERROR: 没有可用的 camera actor")
        return

    # 球缩放：仅当 ball_scale 显式配置时覆盖（默认不覆盖，保持关卡里的实际 scale）
    if ball_scale is not None and "BALL" in actors:
        actors["BALL"].set_actor_scale3d(unreal.Vector(ball_scale, ball_scale, ball_scale))

    # 逐帧收集
    per_camera_lines: Dict[str, List[dict]] = {name: [] for name, *_ in cameras}
    per_camera_objects: Dict[str, List[list]] = {name: [] for name, *_ in cameras}
    prev_yaws: Dict[str, float] = {}
    prev_positions: Dict[str, object] = {}

    for frame_data in selected:
        apply_preview_frame(actors, frame_data, prev_yaws, prev_positions)
        step = frame_data["step"]
        time_seconds = frame_data.get("time_seconds", step * source_step_seconds)
        for name, cam, intrinsics, extrinsics, cam_meta in cameras:
            objects = []
            for entity_id, actor in actors.items():
                objects.append(
                    build_object_annotation(
                        entity_id, actor, entity_info, intrinsics, extrinsics,
                        image_width, image_height, ball_radius_m=ball_radius_m,
                    )
                )
            per_camera_objects[name].append(objects)
            per_camera_lines[name].append({
                "episode_id": episode_id,
                "camera_id": name,
                "frame_index": step + 1,  # 1 基，与 MOT 帧号 / 图片文件名一致
                "source_step": step,      # 0 基，frames.jsonl 行号
                "time_seconds": time_seconds,
                "objects": objects,
            })
        if step % 50 == 0:
            print(f"  Annotation: step {step}/{len(frames)}")

    # 写文件
    for name, cam, intrinsics, extrinsics, cam_meta in cameras:
        cam_out = Path(output_dir) / episode_id / name
        ensure_dir(cam_out / "img1")  # 建立 RGB 契约目录（本阶段不渲染假图）

        if export_internal_jsonl:
            write_json_atomic(cam_out / "camera.json", cam_meta)
            write_jsonl_atomic(cam_out / "annotations.jsonl", per_camera_lines[name])
            print(f"  Wrote: {cam_out / 'camera.json'}")
            print(f"  Wrote: {cam_out / 'annotations.jsonl'} ({len(per_camera_lines[name])} lines)")

        if export_mot:
            rows = build_mot_gt(
                per_camera_objects[name], image_width, image_height,
                include_ball, visibility_mode,
            )
            gt_text = "\n".join(rows) + ("\n" if rows else "")
            write_text_atomic(cam_out / "gt" / "gt.txt", gt_text)
            write_text_atomic(
                cam_out / "seqinfo.ini",
                build_seqinfo(
                    episode_id, "img1", mot_frame_rate,
                    len(selected), image_width, image_height,
                ),
            )
            print(f"  Wrote: {cam_out / 'gt' / 'gt.txt'} ({len(rows)} rows)")

    print(f"Annotation export done: {Path(output_dir) / episode_id}")
