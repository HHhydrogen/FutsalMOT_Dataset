"""pose_bones 纯映射层测试：12 点顺序、flip_idx 和骨骼映射。"""

import math

import pytest

from pose_bones import (
    LIMB_BONE_CANDIDATES,
    NUM_POSE_KEYPOINTS,
    POSE_FLIP_IDX,
    POSE_KEYPOINT_NAMES,
    POSE_SKELETON_EDGES,
    POSE_FIELDS,
    YOLO_POSE_FIELDS,
    bone_index_of,
    missing_limb_bones,
    normalize_world_keypoints,
    resolve_limb_bone_map,
)

# 实测 SKM_Quinn_Simple（UE5 Manny/Quinn 骨架）的骨骼名
MANNEQUIN_BONES = [
    "root", "pelvis", "spine_01", "spine_02", "spine_03", "spine_04", "spine_05",
    "neck_01", "neck_02", "head",
    "clavicle_l", "clavicle_r", "upperarm_l", "upperarm_r",
    "lowerarm_l", "lowerarm_r", "hand_l", "hand_r",
    "thigh_l", "thigh_r", "calf_l", "calf_r", "foot_l", "foot_r",
    "ball_l", "ball_r",
]


class TestPoseDefinition:
    def test_12_limb_points_in_required_order(self):
        assert POSE_KEYPOINT_NAMES == [
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist", "left_hip", "right_hip",
            "left_knee", "right_knee", "left_ankle", "right_ankle",
        ]
        assert NUM_POSE_KEYPOINTS == 12
        assert len(set(POSE_KEYPOINT_NAMES)) == 12

    def test_flip_idx(self):
        assert POSE_FLIP_IDX == [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10]
        for i, j in enumerate(POSE_FLIP_IDX):
            assert POSE_FLIP_IDX[j] == i

    def test_yolo_pose_fields_41(self):
        assert POSE_FIELDS == 5 + NUM_POSE_KEYPOINTS * 3
        assert YOLO_POSE_FIELDS == 41

    def test_bone_index_of(self):
        assert bone_index_of("left_shoulder") == 0
        assert bone_index_of("left_ankle") == 10
        assert bone_index_of("right_ankle") == 11
        with pytest.raises(ValueError):
            bone_index_of("nose")

    def test_skeleton_edges_required_connections(self):
        required = {
            ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
            ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
            ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
            ("left_hip", "right_hip"),
            ("left_shoulder", "right_shoulder"),
            ("left_ankle", "left_knee"), ("left_knee", "left_hip"),
            ("right_ankle", "right_knee"), ("right_knee", "right_hip"),
        }
        edges = {tuple(sorted(e)) for e in POSE_SKELETON_EDGES}
        for edge in required:
            assert tuple(sorted(edge)) in edges, f"缺少骨架连线: {edge}"


class TestBoneMapping:
    def test_mannequin_bones_all_12_limb_points(self):
        m = resolve_limb_bone_map(MANNEQUIN_BONES)
        assert set(m.keys()) == {
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist", "left_hip", "right_hip",
            "left_knee", "right_knee", "left_ankle", "right_ankle",
        }
        # 左右严格对应 anatomical left/right（_l ↔ left）
        assert m["left_shoulder"].endswith("_l")
        assert m["right_shoulder"].endswith("_r")
        # 肩 = upperarm 原点（肩关节），不能用 clavicle 原点（其在胸骨，双肩过近）
        assert m["left_shoulder"] == "upperarm_l"
        assert m["right_shoulder"] == "upperarm_r"
        assert m["left_elbow"] == "lowerarm_l"
        assert m["right_elbow"] == "lowerarm_r"
        assert m["left_wrist"] == "hand_l"
        assert m["right_wrist"] == "hand_r"
        assert m["left_hip"] == "thigh_l"
        assert m["right_hip"] == "thigh_r"
        assert m["left_knee"] == "calf_l"
        assert m["right_knee"] == "calf_r"
        assert m["left_ankle"] == "foot_l"
        assert m["right_ankle"] == "foot_r"

    def test_missing_bones_reported(self):
        # 左肩有两个候选（upperarm_l 主、clavicle_l 回退），两个都去掉才真正缺失
        partial = [b for b in MANNEQUIN_BONES if b not in ("upperarm_l", "clavicle_l", "foot_r", "hand_l")]
        m = resolve_limb_bone_map(partial)
        assert "left_shoulder" not in m
        assert "right_ankle" not in m
        assert "left_wrist" not in m
        missing = missing_limb_bones(partial)
        assert "left_shoulder" in missing
        assert "right_ankle" in missing
        assert "left_wrist" in missing

    def test_overrides_take_precedence(self):
        m = resolve_limb_bone_map(MANNEQUIN_BONES, overrides={"left_elbow": "upperarm_l"})
        assert m["left_elbow"] == "upperarm_l"
        # override 指向不存在的骨骼时该点视为缺失（不静默回退到默认候选）
        m2 = resolve_limb_bone_map(MANNEQUIN_BONES, overrides={"left_elbow": "elbow_l"})
        assert "left_elbow" not in m2

    def test_all_limb_points_have_candidates(self):
        for coco in ("left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                     "left_wrist", "right_wrist", "left_hip", "right_hip",
                     "left_knee", "right_knee", "left_ankle", "right_ankle"):
            assert coco in LIMB_BONE_CANDIDATES
            assert LIMB_BONE_CANDIDATES[coco]  # 至少一个候选


class TestNormalizeWorldKeypoints:
    def test_12_order(self):
        limb = {"left_shoulder": [1, 2, 3], "right_ankle": [4, 5, 6]}
        out = normalize_world_keypoints(limb)
        assert len(out) == 12
        assert out[0] == [1, 2, 3]
        assert out[10] == [None, None, None]
        assert out[11] == [4, 5, 6]
