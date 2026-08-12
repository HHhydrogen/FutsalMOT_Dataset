"""COCO 17 关键点 → UE Skeletal Mesh 骨骼的集中映射层（纯 Python，无 unreal/numpy）。

本模块是「UE 骨骼 bone → COCO keypoint」映射的**唯一权威来源**，同时供：
  - UE 侧（ue/pose_export.py）：逐帧把骨骼 transform 转成 17 个世界坐标关键点；
  - P1 侧（pose_annotator.py / 测试）：读取同一套常量与映射做投影 / 校验 / 可视化。

不依赖 unreal/numpy，UE Python 可直接 import。

骨骼名的确认依据（2026-08 实测读取 SkeletalMesh 资产 `SKM_Quinn_Simple`）：
  球员 Actor 为 BP_ThirdPersonCharacter，Mesh 组件使用 `SKM_Quinn_Simple`
  （UE5 Manny/Quinn Mannequin 骨架，`SKM_Manny_Simple` 同骨架）。资产中**实际存在**的
  骨骼：pelvis / spine_01..spine_05 / neck_01 / neck_02 / head / clavicle_l,r /
  upperarm_l,r / lowerarm_l,r / hand_l,r / thigh_l,r / calf_l,r / foot_l,r / ball_l,r。
  **不存在**眼 / 鼻 / 耳等脸部骨骼（b_eye_l 等均无）——脸部五点必须由 head 骨骼的
  局部偏移推导（见 HEAD_OFFSET_CM 与 apply_head_offsets）。

关节近似规则（关键点 = 子骨骼原点，UE 骨骼原点即父关节处）：
  肩 = upperarm_* 原点（肩关节；**不用 clavicle_* 原点**——其在胸骨处，会使双肩过近），
  肘 = lowerarm_* 原点，腕 = hand_* 原点，髋 = thigh_* 原点，膝 = calf_* 原点，踝 = foot_* 原点。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

# ── COCO 17 关键点：名称与顺序（严禁改动顺序，YOLO Pose 依赖）────────────

COCO_KEYPOINT_NAMES: List[str] = [
    "nose",                    # 0
    "left_eye",                # 1
    "right_eye",               # 2
    "left_ear",                # 3
    "right_ear",               # 4
    "left_shoulder",           # 5
    "right_shoulder",          # 6
    "left_elbow",              # 7
    "right_elbow",             # 8
    "left_wrist",              # 9
    "right_wrist",             # 10
    "left_hip",                # 11
    "right_hip",               # 12
    "left_knee",               # 13
    "right_knee",              # 14
    "left_ankle",              # 15
    "right_ankle",             # 16
]

NUM_COCO_KEYPOINTS = len(COCO_KEYPOINT_NAMES)  # 17

# YOLO Pose 每行字段数 = 5（class + bbox xywh）+ 17×3（x, y, v）
YOLO_POSE_FIELDS = 5 + NUM_COCO_KEYPOINTS * 3  # 56

# 水平翻转时关键点索引映射（左右互换，COCO 标准 flip_idx）
COCO_FLIP_IDX: List[int] = [
    0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15,
]

# 需要 head 骨骼局部偏移推导的脸部关键点（骨骼中不存在对应 bone）
FACE_KEYPOINT_NAMES: List[str] = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
]

# 由真实骨骼 transform 直接确定的关键点（每点：候选 UE bone 名，按优先级）
# 关节近似规则：关键点 = 子骨骼原点（UE 骨骼原点即父关节处）。
#   肩 = upperarm_* 原点（肩关节）。注意：**不能用 clavicle_* 原点**——锁骨起点在
#   胸骨/颈根（双肩会挤在一起），只有 upperarm 原点才是肩关节（锁骨头远端）。
LIMB_BONE_CANDIDATES: Dict[str, List[str]] = {
    "left_shoulder":  ["upperarm_l", "clavicle_l", "shoulder_l"],
    "right_shoulder": ["upperarm_r", "clavicle_r", "shoulder_r"],
    "left_elbow":     ["lowerarm_l", "elbow_l"],
    "right_elbow":    ["lowerarm_r", "elbow_r"],
    "left_wrist":     ["hand_l", "wrist_l"],
    "right_wrist":    ["hand_r", "wrist_r"],
    "left_hip":       ["thigh_l", "hip_l"],
    "right_hip":      ["thigh_r", "hip_r"],
    "left_knee":      ["calf_l", "knee_l"],
    "right_knee":     ["calf_r", "knee_r"],
    "left_ankle":     ["foot_l", "ankle_l"],
    "right_ankle":    ["foot_r", "ankle_r"],
}

HEAD_BONE_NAME = "head"

# 脸部五点相对 head 骨骼局部空间的偏移（单位：cm，局部坐标）。
# 假设 head 骨骼局部轴：+X=角色前向、+Y=角色右向、+Z=角色上向（UE 约定，左手系）。
# 数值按 Mannequin 头部尺寸（高 ~24cm、宽 ~16cm、深 ~20cm）的解剖学估算，
# **必须**用 pose-overlay 或 UE 侧 dump 验证后再微调（见 docs 设计文档）。
HEAD_OFFSET_CM: Dict[str, List[float]] = {
    "nose":      [7.5, 0.0, 9.5],   # 面部中央，head 原点前上方
    "left_eye":  [7.0, -3.2, 12.5], # 角色左侧 = 局部 -Y
    "right_eye": [7.0, 3.2, 12.5],  # 角色右侧 = 局部 +Y
    "left_ear":  [0.5, -8.0, 11.0],
    "right_ear": [0.5, 8.0, 11.0],
}

# COCO 常用骨架连线（可视化用；定义 17 点不随之改变）
COCO_SKELETON_EDGES: List[Tuple[str, str]] = [
    ("left_ankle", "left_knee"),
    ("left_knee", "left_hip"),
    ("right_ankle", "right_knee"),
    ("right_knee", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("nose", "left_eye"),
    ("nose", "right_eye"),
    ("left_eye", "left_ear"),
    ("right_eye", "right_ear"),
]
# 可选：nose → 双肩中点（仅可视化，不进入 17 点定义）
NOSE_SHOULDER_MIDPOINT_EDGE: Tuple[str, str] = ("nose", "shoulder_midpoint")


def bone_index_of(name: str) -> int:
    """COCO 关键点名 → 索引（0 基）。未知名抛 ValueError。"""
    try:
        return COCO_KEYPOINT_NAMES.index(name)
    except ValueError:
        raise ValueError(
            f"未知 COCO 关键点名: {name!r}（可选 {'/'.join(COCO_KEYPOINT_NAMES)}）"
        )


def resolve_limb_bone_map(
    available_bones: Sequence[str],
    overrides: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """从实际骨骼名解析「COCO 关键点 → UE bone」映射（12 个肢体点）。

    overrides: {coco 名: bone 名}，用户级覆盖（config postprocess.yolo_pose.bone_overrides）。
    只返回在 available_bones 中**存在**的候选 bone 对应的映射；找不到的点不包含。
    """
    avail = {str(b).lower() for b in available_bones}
    overrides = overrides or {}
    mapping: Dict[str, str] = {}
    for coco in ("left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                 "left_wrist", "right_wrist", "left_hip", "right_hip",
                 "left_knee", "right_knee", "left_ankle", "right_ankle"):
        if coco in overrides:
            bone = str(overrides[coco])
            if bone.lower() in avail:
                mapping[coco] = bone
            continue
        for cand in LIMB_BONE_CANDIDATES.get(coco, []):
            if cand.lower() in avail:
                mapping[coco] = cand
                break
    return mapping


def missing_limb_bones(
    available_bones: Sequence[str],
    overrides: Optional[Dict[str, str]] = None,
) -> List[str]:
    """列出无法从 available_bones 解析的 COCO 肢体点（供 UE 侧告警）。"""
    avail = {str(b).lower() for b in available_bones}
    missing = []
    for coco in ("left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                 "left_wrist", "right_wrist", "left_hip", "right_hip",
                 "left_knee", "right_knee", "left_ankle", "right_ankle"):
        cands = [str(overrides.get(coco))] if coco in (overrides or {}) \
            else LIMB_BONE_CANDIDATES.get(coco, [])
        if not any(c.lower() in avail for c in cands):
            missing.append(coco)
    return missing


def merged_head_offsets(
    head_offsets_cm: Optional[Dict[str, List[float]]] = None,
) -> Dict[str, List[float]]:
    """合并用户 head 偏移覆盖；只接受脸部关键点名，非法名抛 ValueError。"""
    merged = {k: list(v) for k, v in HEAD_OFFSET_CM.items()}
    for name, off in (head_offsets_cm or {}).items():
        if name not in FACE_KEYPOINT_NAMES:
            raise ValueError(
                f"head_offsets_cm 键 {name!r} 非法（只接受脸部五点："
                f"{'/'.join(FACE_KEYPOINT_NAMES)}）"
            )
        off = [float(v) for v in off]
        if len(off) != 3:
            raise ValueError(f"head_offsets_cm[{name}] 须为 3 元素 [x, y, z] cm")
        merged[name] = off
    return merged


# ── 纯数学：head 局部偏移 → 世界坐标（不依赖 unreal，可 pytest）────────────

def quat_rotate_vector(v: Sequence[float], q_xyzw: Sequence[float]) -> Tuple[float, float, float]:
    """用四元数旋转向量（标准 v' = q v q⁻¹）。

    q_xyzw 与 unreal.Quat 一致：元素序为 (x, y, z, w)。
    """
    x, y, z = float(v[0]), float(v[1]), float(v[2])
    qx, qy, qz, qw = float(q_xyzw[0]), float(q_xyzw[1]), float(q_xyzw[2]), float(q_xyzw[3])
    # t = 2 * (q.xyz × v)
    tx = 2.0 * (qy * z - qz * y)
    ty = 2.0 * (qz * x - qx * z)
    tz = 2.0 * (qx * y - qy * x)
    # v' = v + w*t + q.xyz × t
    ox = x + qw * tx + (qy * tz - qz * ty)
    oy = y + qw * ty + (qz * tx - qx * tz)
    oz = z + qw * tz + (qx * ty - qy * tx)
    return (ox, oy, oz)


def apply_head_offsets(
    head_location: Sequence[float],
    head_quat_xyzw: Sequence[float],
    head_offsets_cm: Optional[Dict[str, List[float]]] = None,
) -> Dict[str, List[float]]:
    """把脸部五点局部偏移转到世界坐标（与 head 骨骼单位一致，通常 cm）。

    每个点 = head_location + R(head) @ offset。左右点严格对应角色 anatomical 左右
    （由局部偏移 +Y/-Y 方向保证，与图像左右无关）。
    """
    offsets = merged_head_offsets(head_offsets_cm)
    out: Dict[str, List[float]] = {}
    hx, hy, hz = (float(head_location[0]), float(head_location[1]), float(head_location[2]))
    for name, off in offsets.items():
        ox, oy, oz = quat_rotate_vector(off, head_quat_xyzw)
        out[name] = [hx + ox, hy + oy, hz + oz]
    return out


def normalize_world_keypoints(
    limb_positions: Dict[str, Sequence[float]],
    face_positions: Dict[str, Sequence[float]],
) -> List[List[float]]:
    """把 12 个肢体点 + 5 个脸部点的世界坐标合成按 COCO 顺序的 17×3 列表。

    缺失的点填充 [None, None, None]（调用方据此判定 v=0）。返回长度为 17。
    """
    merged = dict(limb_positions)
    merged.update(face_positions)
    out: List[List[float]] = []
    for name in COCO_KEYPOINT_NAMES:
        p = merged.get(name)
        if p is None:
            out.append([None, None, None])
        else:
            out.append([float(p[0]), float(p[1]), float(p[2])])
    return out
