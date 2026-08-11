"""pose_bones 纯映射层的测试：COCO 17 点顺序/数量、flip_idx、bone 解析、head 偏移。"""

import math

import pytest

from pose_bones import (
    COCO_FLIP_IDX,
    COCO_KEYPOINT_NAMES,
    COCO_SKELETON_EDGES,
    FACE_KEYPOINT_NAMES,
    HEAD_OFFSET_CM,
    LIMB_BONE_CANDIDATES,
    NUM_COCO_KEYPOINTS,
    YOLO_POSE_FIELDS,
    apply_head_offsets,
    bone_index_of,
    merged_head_offsets,
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


class TestCoco17Definition:
    def test_17_points_in_required_order(self):
        # 顺序必须严格符合 COCO Human Pose 定义
        assert COCO_KEYPOINT_NAMES == [
            "nose", "left_eye", "right_eye", "left_ear", "right_ear",
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist", "left_hip", "right_hip",
            "left_knee", "right_knee", "left_ankle", "right_ankle",
        ]
        assert NUM_COCO_KEYPOINTS == 17
        assert len(set(COCO_KEYPOINT_NAMES)) == 17  # 无重复

    def test_flip_idx(self):
        assert COCO_FLIP_IDX == [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
        # 双射：左右互换两次回到自身
        for i, j in enumerate(COCO_FLIP_IDX):
            assert COCO_FLIP_IDX[j] == i
        # nose 与左右不对称的点保持原位
        assert COCO_FLIP_IDX[0] == 0

    def test_yolo_pose_fields_56(self):
        # 5（class + bbox xywh）+ 17×3 = 56
        assert YOLO_POSE_FIELDS == 5 + NUM_COCO_KEYPOINTS * 3
        assert YOLO_POSE_FIELDS == 56

    def test_bone_index_of(self):
        assert bone_index_of("nose") == 0
        assert bone_index_of("left_ankle") == 15
        assert bone_index_of("right_ankle") == 16
        with pytest.raises(ValueError):
            bone_index_of("football")

    def test_skeleton_edges_required_connections(self):
        required = {
            ("left_ankle", "left_knee"), ("left_knee", "left_hip"),
            ("right_ankle", "right_knee"), ("right_knee", "right_hip"),
            ("left_hip", "right_hip"),
            ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
            ("left_shoulder", "right_shoulder"),
            ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
            ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
            ("nose", "left_eye"), ("nose", "right_eye"),
            ("left_eye", "left_ear"), ("right_eye", "right_ear"),
        }
        edges = {tuple(sorted(e)) for e in COCO_SKELETON_EDGES}
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
        partial = [b for b in MANNEQUIN_BONES if b not in ("clavicle_l", "foot_r", "hand_l")]
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


class TestHeadOffsets:
    def test_face_points_distinct_not_all_same(self):
        # 关键点：5 个脸部点不得退化为同一个坐标
        q_identity = (0.0, 0.0, 0.0, 1.0)
        face = apply_head_offsets((150.0, 0.0, 150.0), q_identity)
        assert set(face.keys()) == set(FACE_KEYPOINT_NAMES)
        coords = {tuple(round(v, 3) for v in p) for p in face.values()}
        assert len(coords) == 5  # 全部不同

    def test_left_right_anatomical(self):
        # 左/右点由局部偏移方向（-Y=角色左、+Y=角色右）唯一决定，与图像左右无关。
        q_identity = (0.0, 0.0, 0.0, 1.0)
        face = apply_head_offsets((0.0, 0.0, 0.0), q_identity)
        assert face["left_eye"][1] < face["right_eye"][1]
        assert face["left_ear"][1] < face["right_ear"][1]
        # 角色绕上向转 180°：左右点世界 Y 分量符号翻转，但 anatomical 身份不变
        # （left_eye 永远是局部 -Y 的那只眼，随角色转身）。
        q_yaw180 = (0.0, 0.0, 1.0, 0.0)
        face2 = apply_head_offsets((0.0, 0.0, 0.0), q_yaw180)
        assert face2["left_eye"][1] == -face["left_eye"][1]
        assert face2["right_eye"][1] == -face["right_eye"][1]
        # 两侧始终不同（不会都退化为 head 中心）
        assert face2["left_eye"] != face2["right_eye"]

    def test_head_offset_override(self):
        offs = {"nose": [1.0, 2.0, 3.0]}
        face = apply_head_offsets((10.0, 10.0, 10.0), (0.0, 0.0, 0.0, 1.0), offs)
        assert face["nose"] == [11.0, 12.0, 13.0]

    def test_invalid_head_offset_name_rejected(self):
        with pytest.raises(ValueError):
            merged_head_offsets({"football": [1.0, 0.0, 0.0]})

    def test_quat_rotation_90deg_yaw(self):
        # 绕 +Z 转 90°：(1,0,0) -> (0,1,0)（右手定则）
        q = (0.0, 0.0, math.sin(math.radians(45)), math.cos(math.radians(45)))
        x, y, z = apply_head_offsets((0.0, 0.0, 0.0), q)["nose"]
        # 默认 nose 偏移 [7.5, 0, 9.5]，绕 Z 转 90° → (7.5,0,9.5)→(0,7.5,9.5)
        assert math.isclose(x, 0.0, abs_tol=1e-6)
        assert math.isclose(y, 7.5, abs_tol=1e-6)
        assert math.isclose(z, 9.5, abs_tol=1e-6)


class TestNormalizeWorldKeypoints:
    def test_17_order(self):
        limb = {"left_ankle": [1, 2, 3]}
        face = {"nose": [4, 5, 6]}
        out = normalize_world_keypoints(limb, face)
        assert len(out) == 17
        assert out[0] == [4, 5, 6]   # nose 在前
        assert out[15] == [1, 2, 3]  # left_ankle 第 16
        assert out[16] == [None, None, None]  # 缺 right_ankle
