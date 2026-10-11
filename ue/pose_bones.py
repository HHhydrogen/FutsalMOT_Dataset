"""数据集 Pose 12 个肢体关键点到 UE Skeletal Mesh 骨骼的映射。

本模块是 P1/P2 共用的最终 Pose 顺序和骨骼映射来源，不依赖 unreal/numpy。
Recorder 原始采样仍额外包含 head，用于保持 13 骨 capture 完整性；head 不属于最终
训练标注，也不再推算鼻、眼、耳位置。

关节近似规则：关键点取子骨骼原点（UE 骨骼原点即父关节处）：肩取 upperarm、肘取
lowerarm、腕取 hand、髋取 thigh、膝取 calf、踝取 foot。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence


POSE_KEYPOINT_NAMES: List[str] = [
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

NUM_POSE_KEYPOINTS = len(POSE_KEYPOINT_NAMES)
POSE_FLIP_IDX: List[int] = [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10]
POSE_FIELDS = 5 + NUM_POSE_KEYPOINTS * 3
YOLO_POSE_FIELDS = POSE_FIELDS

LIMB_BONE_CANDIDATES: Dict[str, List[str]] = {
    "left_shoulder": ["upperarm_l", "clavicle_l", "shoulder_l"],
    "right_shoulder": ["upperarm_r", "clavicle_r", "shoulder_r"],
    "left_elbow": ["lowerarm_l", "elbow_l"],
    "right_elbow": ["lowerarm_r", "elbow_r"],
    "left_wrist": ["hand_l", "wrist_l"],
    "right_wrist": ["hand_r", "wrist_r"],
    "left_hip": ["thigh_l", "hip_l"],
    "right_hip": ["thigh_r", "hip_r"],
    "left_knee": ["calf_l", "knee_l"],
    "right_knee": ["calf_r", "knee_r"],
    "left_ankle": ["foot_l", "ankle_l"],
    "right_ankle": ["foot_r", "ankle_r"],
}

POSE_SKELETON_EDGES = [
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
]


def bone_index_of(name: str) -> int:
    """关键点名映射到 0 基索引；非 12 个肢体点名称抛 ValueError。"""
    try:
        return POSE_KEYPOINT_NAMES.index(name)
    except ValueError as exc:
        raise ValueError(
            f"未知 Pose 关键点名: {name!r}（可选 {'/'.join(POSE_KEYPOINT_NAMES)}）"
        ) from exc


def resolve_limb_bone_map(
    available_bones: Sequence[str],
    overrides: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """从 mesh 实际骨骼解析 12 个关键点对应的 UE bone。"""
    available = {str(bone).lower() for bone in available_bones}
    overrides = overrides or {}
    mapping: Dict[str, str] = {}
    for point in POSE_KEYPOINT_NAMES:
        if point in overrides:
            bone = str(overrides[point])
            if bone.lower() in available:
                mapping[point] = bone
            continue
        for candidate in LIMB_BONE_CANDIDATES[point]:
            if candidate.lower() in available:
                mapping[point] = candidate
                break
    return mapping


def missing_limb_bones(
    available_bones: Sequence[str],
    overrides: Optional[Dict[str, str]] = None,
) -> List[str]:
    """返回无法映射到现有骨骼的肢体关键点。"""
    available = {str(bone).lower() for bone in available_bones}
    overrides = overrides or {}
    missing = []
    for point in POSE_KEYPOINT_NAMES:
        candidates = [str(overrides[point])] if point in overrides else LIMB_BONE_CANDIDATES[point]
        if not any(candidate.lower() in available for candidate in candidates):
            missing.append(point)
    return missing


def normalize_world_keypoints(
    limb_positions: Dict[str, Sequence[float]],
) -> List[List[float]]:
    """按固定 12 点顺序合成世界坐标；缺失点占位 ``[None, None, None]``。"""
    output: List[List[float]] = []
    for point in POSE_KEYPOINT_NAMES:
        position = limb_positions.get(point)
        if position is None:
            output.append([None, None, None])
        else:
            output.append([float(position[0]), float(position[1]), float(position[2])])
    return output
