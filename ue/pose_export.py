"""UE 内运行的关键点导出器：逐帧读取球员 SkeletalMesh 骨骼 transform → COCO 17 世界坐标。

依赖：
  - unreal（UE 内置，一律延迟 import）
  - pose_bones / scene_apply / dataset_export / annotation_utils（同目录模块，纯 Python）

流程（与 annotation_exporter 使用同一套 actor 变换规则）：
  1. 读取 episode（meta.json + frames.jsonl）与 actor 映射。
  2. 解析「UE bone → COCO keypoint」映射（pose_bones，可用 bone_overrides 覆盖）。
     解析用**逐个候选骨 probe world transform** 的方式（不依赖列出骨骼名——
     `get_ref_skeleton` 在 UE 5.8 Python 不可用），对每个 COCO 点取第一个能读到
     有效 transform 的候选骨。
  3. 逐帧应用与 Level Sequence 相同的 actor 变换（scene_apply）。
  4. 对每个球员读取 17 个关键点的世界坐标（cm → m 存储）：
       - 12 个肢体点：真实骨骼 transform（子骨骼原点 = 关节，见 pose_bones 注释）；
       - 脸部 5 点：head 骨骼局部偏移（骨骼中无眼/鼻/耳，见 HEAD_OFFSET_CM）。
  5. 对每个 camera，可选对每个关键点做遮挡 trace（射线从相机到关键点，命中距离
     明显小于关键点距离即视为被几何遮挡）→ occluded 标志。
     trace API 在任何 UE 版本缺失/变化时**绝不崩溃**：失败即跳过该关键点遮挡判定
     （返回 None），P1 侧退化为仅 Instance-ID Mask 判定。
  6. 输出每个 Camera 的 pose_keypoints.jsonl（世界 3D 关键点 + occluded）。

单位约定：UE 世界为 cm；本模块把关键点换算为**米**（与 camera.json 的
world_location_m / frames.jsonl 一致）后再落盘，P1 侧可直接投影。

注意：骨骼 transform 取自**导出时**关卡中的实际姿势（编辑器/视口状态）。若角色的
AnimBlueprint 在 MRQ PIE 渲染期间播放动画，编辑器姿势与渲染姿势可能不同，导致
关键点与渲染图不完全对齐——用 pose-overlay 验证；必要时把 mesh 动画置空（参考姿势）
或确认导出与渲染姿势一致（见 docs/design/2026-08-11-yolo-pose-export.md）。
"""

import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pose_bones import (  # noqa: E402
    COCO_KEYPOINT_NAMES,
    HEAD_BONE_NAME,
    LIMB_BONE_CANDIDATES,
    apply_head_offsets,
)
from annotation_utils import entity_id_to_track_id  # noqa: E402
from dataset_export import (  # noqa: E402
    ensure_dir,
    load_episode,
    load_mapping,
    write_jsonl_atomic,
)
from scene_apply import (  # noqa: E402
    apply_preview_frame,
    find_all_actors,
    find_actor,
)
from player_motion import gk_entity_ids_from_meta  # noqa: E402

POSE_KEYPOINTS_SCHEMA = "futsalmot_pose_keypoints_v1"
CM_TO_M = 0.01

# 遮挡 trace 的默认容差（cm）：命中距离 < 关键点距离 - 该容差 即视为被遮挡。
# 关键点位于骨骼表面内（身体半径），可见关节的表面命中距离通常 < 15cm；
# 真正被遮挡时遮挡物明显更近。容差按最大身体部位半厚度设置。
DEFAULT_TRACE_TOLERANCE_CM = 20.0


def _v3(v) -> Tuple[float, float, float]:
    """把 unreal.Vector / 带 x/y/z 的对象转成 (float, float, float)。"""
    return (float(v.x), float(v.y), float(v.z))


def _valid_world_transform(pos: Tuple[float, float, float],
                           quat_xyzw: Tuple[float, float, float, float]) -> bool:
    """校验骨骼 world transform 是否有效（有限值 + 非全零四元数）。

    缺失/未初始化的骨骼（get_bone_transform 对不存在 bone 返回零值 transform）用
    全零四元数识别，避免 probe 误判「骨骼存在」。
    """
    for v in pos:
        if not math.isfinite(v):
            return False
    for v in quat_xyzw:
        if not math.isfinite(v):
            return False
    if quat_xyzw[0] == 0.0 and quat_xyzw[1] == 0.0 and \
            quat_xyzw[2] == 0.0 and quat_xyzw[3] == 0.0:
        return False  # 全零四元数 = 未初始化/缺失
    return True


def _bone_world_transform(comp, bone_name: str) -> Optional[Tuple[Tuple[float, float, float],
                                                                  Tuple[float, float, float, float]]]:
    """读取骨骼的世界 transform，返回 (location_cm, quat_xyzw)。

    quat_xyzw 与 unreal.Quat 一致：元素序为 (x, y, z, w)。返回 None 表示 API 不可用
    或骨骼缺失（transform 无效）。
    """
    import unreal

    # 首选 get_bone_transform（返回 Transform，rotation 即 Quat，无需 Rotator 转换）
    try:
        tf = comp.get_bone_transform(bone_name, False)
        if tf is not None:
            loc = tf.location
            q = tf.rotation
            pos = (float(loc.x), float(loc.y), float(loc.z))
            quat = (float(q.x), float(q.y), float(q.z), float(q.w))
            if _valid_world_transform(pos, quat):
                return pos, quat
    except Exception:
        pass
    # 回退：socket 位置 + 旋转（旋转为 Rotator → 尽量转 Quat）。
    # 先确认 bone 存在（get_bone_index >= 0），避免缺失骨骼的 socket 回退返回
    # (0,0,0)+单位四元数被误判为有效 transform。
    try:
        idx = comp.get_bone_index(bone_name)
        if idx is not None and int(idx) < 0:
            return None
    except Exception:
        pass  # get_bone_index 不可用时不强求
    try:
        loc = comp.get_socket_location(bone_name)
        if loc is None:
            return None
        pos = (float(loc.x), float(loc.y), float(loc.z))
        q = _rotator_to_quat_xyzw(comp, bone_name)
        if _valid_world_transform(pos, q):
            return pos, q
    except Exception:
        pass
    return None


def _rotator_to_quat_xyzw(comp, bone_name: str) -> Tuple[float, float, float, float]:
    """把骨骼世界 Rotator 转成 quat_xyzw。用 unreal.MathLibrary 优先，失败回退单位四元数。"""
    import unreal

    try:
        rot = comp.get_socket_rotation(bone_name)
        if rot is None:
            return (0.0, 0.0, 0.0, 1.0)
        lib = getattr(unreal, "MathLibrary", None) or getattr(unreal, "KismetMathLibrary", None)
        if lib is not None:
            q = lib.quat_from_rotator(rot)
            return (float(q.x), float(q.y), float(q.z), float(q.w))
    except Exception:
        pass
    return (0.0, 0.0, 0.0, 1.0)


def _skeletal_mesh_component(actor):
    """取 actor 的第一个 SkeletalMeshComponent（无则返回 None）。"""
    import unreal

    comps = actor.get_components_by_class(unreal.SkeletalMeshComponent)
    return comps[0] if comps else None


def _list_actor_bones(actor) -> List[str]:
    """列出 actor 骨骼网格的全部骨骼名（**仅诊断用**，失败返回 [] 绝不抛错）。

    用 Skeleton 资产的 bone_tree 读取（UE 5.8 Python 无 mesh.get_ref_skeleton）。
    骨骼解析已改为 probe 方式（_resolve_bone_map_by_probe），不依赖本函数。
    """
    import unreal

    comp = _skeletal_mesh_component(actor)
    if comp is None:
        return []
    names: List[str] = []
    try:
        mesh = None
        getter = getattr(comp, "get_skeletal_mesh_asset", None)
        if getter is not None:
            try:
                mesh = getter()
            except Exception:
                mesh = None
        if mesh is None:
            mesh = getattr(comp, "skeletal_mesh", None)
        skel = None
        if mesh is not None:
            getter2 = getattr(mesh, "get_skeleton", None)
            if getter2 is not None:
                try:
                    skel = getter2()
                except Exception:
                    skel = None
            if skel is None:
                skel = getattr(mesh, "skeleton", None)
        if skel is not None:
            tree = None
            try:
                tree = skel.get_editor_property("bone_tree")
            except Exception:
                tree = None
            if tree:
                for bn in tree:
                    nm = getattr(bn, "name", None) or getattr(bn, "bone_name", None)
                    if nm is not None:
                        names.append(str(nm))
    except Exception:
        names = []
    return names


def dump_actor_bones(actor_name: str) -> List[str]:
    """打印某球员 actor 的全部骨骼名（UE 控制台验证映射用）。"""
    actor = find_actor(actor_name)
    if actor is None:
        print(f"ERROR: 找不到 actor: {actor_name}")
        return []
    bones = _list_actor_bones(actor)
    print(f"Actor {actor_name}: {len(bones)} 根骨骼")
    for b in bones:
        print(f"  {b}")
    return bones


def _resolve_bone_map_by_probe(actor, overrides: Dict[str, str]) -> Dict[str, str]:
    """逐个候选骨 probe world transform，为每个 COCO 肢体点选第一个有效的骨。

    不依赖列出骨骼名（get_ref_skeleton 在 UE 5.8 Python 不可用）。missing bone 的
    transform 是零值（全零四元数），被 _valid_world_transform 拒绝。
    """
    comp = _skeletal_mesh_component(actor)
    if comp is None:
        return {}
    bone_map: Dict[str, str] = {}
    for coco, cands in LIMB_BONE_CANDIDATES.items():
        cand_list = [overrides[coco]] if coco in overrides else cands
        for bone in cand_list:
            if _bone_world_transform(comp, bone) is not None:
                bone_map[coco] = bone
                break
    return bone_map


def _actor_keypoints_world_m(actor, bone_map: Dict[str, str],
                             head_offsets: Dict[str, List[float]]) -> List[List[Optional[float]]]:
    """计算单个 actor 的 17 个关键点世界坐标（米）。

    返回 [[x, y, z], ...] 长度 17；无法取得的点填充 [None, None, None]。
    """
    comp = _skeletal_mesh_component(actor)
    if comp is None:
        return [[None, None, None]] * len(COCO_KEYPOINT_NAMES)

    # 12 个肢体点：子骨骼原点 = 关节
    limb: Dict[str, List[float]] = {}
    for coco, bone in bone_map.items():
        tf = _bone_world_transform(comp, bone)
        if tf is None:
            continue
        loc_cm, _q = tf
        limb[coco] = [loc_cm[0] * CM_TO_M, loc_cm[1] * CM_TO_M, loc_cm[2] * CM_TO_M]

    # 脸部 5 点：head 骨骼局部偏移
    face: Dict[str, List[float]] = {}
    head_tf = _bone_world_transform(comp, HEAD_BONE_NAME)
    if head_tf is not None:
        loc_cm, quat_xyzw = head_tf
        face_cm = apply_head_offsets(loc_cm, quat_xyzw, head_offsets)
        for name, p in face_cm.items():
            face[name] = [p[0] * CM_TO_M, p[1] * CM_TO_M, p[2] * CM_TO_M]

    merged = dict(limb)
    merged.update(face)
    out: List[List[Optional[float]]] = []
    for name in COCO_KEYPOINT_NAMES:
        p = merged.get(name)
        if p is None:
            out.append([None, None, None])
        else:
            out.append([round(float(v), 6) for v in p])
    return out


def _camera_location_cm(camera_actor) -> Optional[Tuple[float, float, float]]:
    import unreal

    try:
        loc = camera_actor.get_actor_location()
        return (float(loc.x), float(loc.y), float(loc.z))
    except Exception:
        return None


def _get_world(actor):
    """获取 actor 所在的 world（每相机解析一次，避免对每个关键点重复查）。"""
    import unreal

    for getter in (
        lambda: actor.get_world(),
        lambda: unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_editor_world(),
    ):
        try:
            w = getter()
            if w is not None:
                return w
        except Exception:
            continue
    return None


def _trace_occluded(world, cam_loc_cm: Tuple[float, float, float],
                    keypoint_m: List[Optional[float]],
                    tolerance_cm: float) -> Optional[bool]:
    """对单个关键点做遮挡 trace。

    射线从相机位置（cm）到关键点世界位置（米 → cm）。任一 blocking 命中且
    命中距离 < 关键点距离 - tolerance 即视为被几何遮挡（含自遮挡与非 mask 几何，
    如球、围挡）。返回 None 表示 trace 不可用（调用方回退 False，仅 mask 判定）。
    """
    import unreal

    if keypoint_m[0] is None or world is None:
        return None
    try:
        kp = [float(keypoint_m[0]) * 100.0, float(keypoint_m[1]) * 100.0, float(keypoint_m[2]) * 100.0]
        start = unreal.Vector(cam_loc_cm[0], cam_loc_cm[1], cam_loc_cm[2])
        end = unreal.Vector(kp[0], kp[1], kp[2])
        if not all(math.isfinite(v) for v in kp + list(cam_loc_cm)):
            return None
        keypoint_dist = math.dist(cam_loc_cm, tuple(kp))

        result = _line_trace(world, start, end)
        if result is None:
            return None
        hit, hit_actor, hit_distance = result
        if not hit or hit_actor is None:
            return False
        if hit_distance is None:
            return False
        # 命中距离必须明显小于关键点距离才判遮挡（吸收关节在身体表面内部的深度差）
        return bool(hit_distance < keypoint_dist - tolerance_cm)
    except Exception:
        return None


def _line_trace(world, start, end):
    """执行单次射线检测，返回 (hit: bool, hit_actor, hit_distance_cm) 或 None。

    UE Python 的 trace API 在不同版本名称/枚举/签名差异很大（如 5.8 无
    unreal.ETraceTypeQuery）。本函数**全防御式**：任何一步失败都返回 None
    （调用方跳过遮挡判定），绝不向导出流程抛异常。
    """
    import unreal

    # 1) 找 trace channel 枚举（名字随版本变化；都找不到时尝试不带 channel 参数）
    trace_channel = None
    for enum_name in ("ETraceTypeQuery", "TraceTypeQuery", "ECollisionChannel"):
        enum_cls = getattr(unreal, enum_name, None)
        if enum_cls is None:
            continue
        for member in ("TraceTypeQuery1", "TraceTypeQuery_MAX", "ECC_VISIBILITY", "ECC_Visibility"):
            try:
                v = getattr(enum_cls, member, None)
            except Exception:
                v = None
            if v is not None:
                trace_channel = v
                break
        if trace_channel is not None:
            break

    # 2) draw_debug_type（可选）
    draw = None
    try:
        draw = getattr(unreal.EDrawDebugTrace, "NONE", None) if hasattr(unreal, "EDrawDebugTrace") else None
    except Exception:
        draw = None

    # 3) params 对象（TraceParams / RayTraceParams 因版本而异）
    params = None
    for cls_name in ("TraceParams", "RayTraceParams"):
        cls = getattr(unreal, cls_name, None)
        if cls is None:
            continue
        try:
            params = cls()
            break
        except Exception:
            params = None

    # 4) trace 函数本体（World 上）
    fn = getattr(world, "line_trace_single_by_channel", None)
    if fn is None:
        # 有的版本挂在 SystemLibrary / EditorLevelLibrary 上（不同签名，这里不强求）
        print("  WARNING: unreal.World.line_trace_single_by_channel 不可用（跳过遮挡 trace）")
        return None

    # 5) 多组参数尝试：优先带 channel，失败退化为不带
    attempts = []
    if trace_channel is not None:
        attempts.append({"trace_channel": trace_channel})
    attempts.append({})
    for kwargs in attempts:
        try:
            if params is not None:
                result = fn(start=start, end=end, params=params,
                            actors_to_ignore=[], draw_debug_type=draw,
                            ignore_self=True, trace_complex=False, **kwargs)
            else:
                result = fn(start=start, end=end, actors_to_ignore=[], draw_debug_type=draw,
                            ignore_self=True, trace_complex=False, **kwargs)
        except Exception:
            continue
        if result is None:
            continue
        # 兼容两种返回形态：HitResult 对象 或 (hit, hit_result) 元组
        try:
            hr = result[1] if isinstance(result, tuple) and len(result) > 1 else result
            hit = bool(hr.b_blocking_hit) if hasattr(hr, "b_blocking_hit") else True
            actor = None
            try:
                actor = hr.get_actor() or getattr(hr, "actor", None)
            except Exception:
                actor = getattr(hr, "actor", None)
            dist = None
            try:
                dist = float(hr.distance)
            except Exception:
                dist = None
            return hit, actor, dist
        except Exception:
            continue
    print("  WARNING: 遮挡 trace 调用失败（跳过 occlusion，mask 仍生效）")
    return None


def export_pose_keypoints(
    episode_dir: Path,
    mapping_path: Path,
    output_dir: Path,
    ann_cfg: dict,
    pose_cfg: dict,
) -> None:
    """写每个 camera 的 pose_keypoints.jsonl（世界 3D 关键点 + occluded）。

    pose_cfg 关键字段：
      bone_overrides   : {COCO 名: UE bone 名} 覆盖默认映射
      head_offsets_cm  : {脸部 COCO 名: [x, y, z] cm} head 局部偏移覆盖
      occlusion_trace  : bool，是否对每个关键点做遮挡 trace（默认 true）
      trace_tolerance_cm: 遮挡 trace 容差（cm，默认 20.0）
    ann_cfg 关键字段：cameras / frame_start / frame_end / image_width / image_height。
    """
    import unreal  # noqa: F401  # 确保 UE 会话存在（脚本在 UE 内运行）

    meta, frames = load_episode(episode_dir)
    gk_ids = gk_entity_ids_from_meta(meta)
    mapping = load_mapping(mapping_path)
    actors = find_all_actors(mapping)
    if not actors:
        return

    episode_id = meta.get("episode_id") or episode_dir.name
    camera_names = ann_cfg.get("cameras") or []
    frame_start = int(ann_cfg.get("frame_start", 0))
    frame_end = ann_cfg.get("frame_end")
    occlusion_trace = bool(pose_cfg.get("occlusion_trace", True))
    trace_tol = float(pose_cfg.get("trace_tolerance_cm", DEFAULT_TRACE_TOLERANCE_CM))
    overrides = pose_cfg.get("bone_overrides") or {}
    head_offsets = pose_cfg.get("head_offsets_cm") or {}
    image_width = int(ann_cfg.get("image_width", 1920))
    image_height = int(ann_cfg.get("image_height", 1080))
    source_step_seconds = float(meta["timing"].get("source_step_seconds", 0.1))

    player_actors = {eid: a for eid, a in actors.items() if eid != "BALL"}
    if not player_actors:
        print("ERROR: 没有球员 actor，无法导出关键点")
        return

    # 骨骼解析：probe 方式（不依赖列出骨骼名）。逐球员 probe 一次并缓存，
    # 兼容不同球员使用不同 mesh 的情况。
    bone_maps: Dict[str, Dict[str, str]] = {}
    for eid, actor in player_actors.items():
        bone_maps[eid] = _resolve_bone_map_by_probe(actor, overrides)
    # 诊断：用任一球员尝试列出实际骨骼名（best-effort）
    sample_actor = next(iter(player_actors.values()))
    available = _list_actor_bones(sample_actor)
    if not available:
        print("WARNING: 无法列出球员骨骼名（仅诊断；关键点由 transform probe 解析，不受影响）")
    n_total = sum(len(bm) for bm in bone_maps.values())
    if n_total == 0:
        print("WARNING: 任何球员都没有解析到肢体骨骼（probe 全失败）——"
              "请检查球员是否有 SkeletalMeshComponent；关键点将全部为无效")
    elif len(available):
        n_lack = sum(1 for bm in bone_maps.values() if len(bm) < 12)
        if n_lack:
            print(f"WARNING: {n_lack} 个球员肢体骨解析不完整（<12/12）")
    print(f"  [Pose] bone 映射（probe）: 各球员 {len(bone_maps[list(player_actors)[0]])}/12 肢体点"
          f"（骨骼列表诊断: {len(available)} 根）")

    # 帧范围（与 annotation_exporter 一致，0 基 GRF step）
    selected = [f for f in frames if f["step"] >= frame_start]
    if frame_end is not None:
        selected = [f for f in selected if f["step"] < int(frame_end)]
    if not selected:
        print("ERROR: 没有选中任何帧，跳过 pose 导出")
        return

    # 相机列表
    cameras = []
    for name in camera_names:
        cam = find_actor(name)
        if not cam:
            print(f"  WARNING: Camera actor '{name}' 未找到，跳过")
            continue
        loc = _camera_location_cm(cam)
        cameras.append((name, cam, loc))
    if not cameras:
        print("ERROR: 没有可用的 camera actor，跳过 pose 导出")
        return

    # 逐帧收集世界关键点（相机无关，只算一次）
    trackers: Dict[str, object] = {}
    per_frame_kps: Dict[int, Dict[str, List[List[Optional[float]]]]] = {}
    for frame_data in selected:
        apply_preview_frame(actors, frame_data, trackers, gk_entity_ids=gk_ids)
        step = frame_data["step"]
        kps = {}
        for entity_id, actor in player_actors.items():
            kps[entity_id] = _actor_keypoints_world_m(actor, bone_maps[entity_id], head_offsets)
        per_frame_kps[step] = kps
        if step % 50 == 0:
            print(f"  Pose: step {step}/{len(frames)}")

    # 逐相机写 pose_keypoints.jsonl（遮挡 trace 依赖相机位置，逐相机计算）
    # world 每相机解析一次（600k 次 trace 不该每次都查 world）
    world = _get_world(next(iter(player_actors.values())))
    if occlusion_trace and world is None:
        print("  WARNING: 无法获取 world，遮挡 trace 跳过（仅 mask 判定）")
    for cam_idx, (name, cam, cam_loc) in enumerate(cameras):
        cam_out = Path(output_dir) / episode_id / name
        ensure_dir(cam_out)
        print(f"  Pose trace: 相机 {cam_idx + 1}/{len(cameras)}（{name}，{len(selected)} 帧）...")
        lines = [{
            "kind": "meta",
            "schema": POSE_KEYPOINTS_SCHEMA,
            "episode_id": episode_id,
            "camera_id": name,
            "image_width": image_width,
            "image_height": image_height,
            "keypoint_names": COCO_KEYPOINT_NAMES,
            "coordinate_convention": (
                "world keypoints in meters（UE 厘米 / 100）；UE 左手系 X 前 Y 右 Z 上；"
                "按 COCO 17 点顺序"
            ),
            "occlusion_method": (
                f"UE line trace（相机→关键点，命中距离 < 关键点距离 - {trace_tol}cm 即被遮挡）"
                if occlusion_trace else "none（P1 仅用 mask 判定）"
            ),
        }]
        for fi, frame_data in enumerate(selected):
            step = frame_data["step"]
            frame_index = step + 1
            objects = []
            for entity_id, kps in per_frame_kps[step].items():
                occluded = None
                if occlusion_trace and cam_loc is not None and world is not None:
                    occluded = []
                    for kp in kps:
                        occluded.append(_trace_occluded(world, cam_loc, kp, trace_tol))
                obj = {
                    "entity_id": entity_id,
                    "track_id": entity_id_to_track_id(entity_id),
                    "keypoints_world": kps,
                }
                # 只有 trace 真正执行过（至少一个非 None）才写 occluded 字段；
                # 全部 None 表示 trace 不可用，P1 退化为仅 mask 判定
                if occluded is not None and any(o is not None for o in occluded):
                    obj["occluded"] = [bool(o) if o is not None else False for o in occluded]
                objects.append(obj)
            lines.append({
                "kind": "frame",
                "frame_index": frame_index,
                "source_step": step,
                "time_seconds": frame_data.get("time_seconds", step * source_step_seconds),
                "objects": objects,
            })
            if fi % 50 == 0:
                print(f"  Pose trace: {name} 帧 {frame_index}/{len(selected)}")
        path = cam_out / "pose_keypoints.jsonl"
        write_jsonl_atomic(path, lines)
        print(f"  Wrote: {path} ({len(lines) - 1} 帧)")

    print(f"Pose keypoint export done: {Path(output_dir) / episode_id}")
