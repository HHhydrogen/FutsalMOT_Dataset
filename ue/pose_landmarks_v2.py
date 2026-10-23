"""COCO17 表面锚点实验契约；不替换正式十二点导出，也不提供猜测头部偏移。

三角锚点作用于已经完成蒙皮/形变的同一拓扑顶点。Blender 顶点编号不能直接
用于 UE：导入可能拆点、合并或重排，必须另行建立并验证逐 mesh 的映射。
"""

import hashlib
import json
import math

COCO17_NAMES = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
    "right_knee", "left_ankle", "right_ankle",
)
COCO17_FLIP_IDX = (0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15)
HEAD_NAMES = COCO17_NAMES[:5]


def _vector(value, size, label):
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise ValueError(label + " 维度错误")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
           for v in value):
        raise ValueError(label + " 必须为有限数")
    return tuple(float(v) for v in value)


def mesh_fingerprints(vertices, triangles):
    """签名包括有序坐标/三角索引，必须对同坐标系、同 LOD 比较。"""
    points = [_vector(v, 3, "vertex") for v in vertices]
    faces = []
    for face in triangles:
        if len(face) != 3 or any(type(i) is not int or i < 0 or i >= len(points) for i in face):
            raise ValueError("triangle 索引非法")
        if len(set(face)) != 3:
            raise ValueError("triangle 重复顶点")
        faces.append(list(face))
    def digest(value):
        return hashlib.sha256(json.dumps(value, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return {"vertex_count": len(points), "topology_sha256": digest(faces),
            "geometry_sha256": digest(points)}


def evaluate_surface_anchor(vertices, triangle_vertices, barycentric):
    """对形变后三角形插值；坐标系/单位完全继承输入，不偷偷缩放。"""
    if len(triangle_vertices) != 3 or len(set(triangle_vertices)) != 3:
        raise ValueError("必须是三个不同的三角顶点")
    if any(type(i) is not int or i < 0 or i >= len(vertices) for i in triangle_vertices):
        raise ValueError("triangle 索引越界")
    weights = _vector(barycentric, 3, "barycentric")
    if any(w < 0 or w > 1 for w in weights) or abs(sum(weights) - 1) > 1e-8:
        raise ValueError("barycentric 必须为和为一的非负权重")
    points = [_vector(vertices[i], 3, "vertex") for i in triangle_vertices]
    ab = [points[1][j] - points[0][j] for j in range(3)]
    ac = [points[2][j] - points[0][j] for j in range(3)]
    cross = [ab[1]*ac[2]-ab[2]*ac[1], ab[2]*ac[0]-ab[0]*ac[2], ab[0]*ac[1]-ab[1]*ac[0]]
    if sum(x*x for x in cross) == 0:
        raise ValueError("退化三角形不允许作为表面锚点")
    return [sum(weights[i] * points[i][axis] for i in range(3)) for axis in range(3)]


def validate_landmark_manifest(manifest, mesh_identity, vertices, triangles):
    """缺标定、身份不匹配或无可追溯精度验收时拒绝作为生产锚点。"""
    if manifest.get("schema") != "futsalmot_head_surface_landmarks" or manifest.get("version") != 1:
        raise ValueError("不支持的 landmark manifest")
    actual = dict(mesh_identity)
    actual.update(mesh_fingerprints(vertices, triangles))
    expected = manifest.get("mesh_identity", {})
    for field in ("asset_path", "lod", "vertex_count", "geometry_sha256", "topology_sha256"):
        if field not in actual or field not in expected or actual[field] != expected[field]:
            raise ValueError("mesh 身份不匹配: " + field)
    if actual["lod"] != 0:
        raise ValueError("当前仅支持已标定 LOD0")
    if manifest.get("coordinate_frame") not in ("blender_object_m", "ue_component_cm"):
        raise ValueError("必须显式声明坐标系及单位")
    if manifest.get("source_method") != "deformed_surface_barycentric":
        raise ValueError("当前仅支持真实形变顶点的表面插值")
    calibration = manifest.get("calibration", {})
    if calibration.get("status") != "accepted" or not calibration.get("evidence_ref"):
        raise ValueError("缺少校准验收证据")
    # 误差门槛由实验预先指定；不在库中捏造通用精度阈值。
    for field in ("max_reprojection_error_px", "acceptance_threshold_px"):
        _vector([calibration.get(field)], 1, field)
        if calibration[field] < 0:
            raise ValueError("误差/阈值不可负数")
    if calibration["max_reprojection_error_px"] > calibration["acceptance_threshold_px"]:
        raise ValueError("重投影误差未通过预设阈值")
    anchors = manifest.get("anchors", {})
    if set(anchors) != set(HEAD_NAMES):
        raise ValueError("必须提供且只提供五个头部关键点")
    face_set = {frozenset(face) for face in triangles}
    for name in HEAD_NAMES:
        anchor = anchors[name]
        review = anchor.get("semantic_review", {})
        if review.get("reviewed") is not True or not review.get("method") or not review.get("evidence_ref"):
            raise ValueError(name + " 缺少语义标定")
        if frozenset(anchor.get("triangle_vertices", [])) not in face_set:
            raise ValueError(name + " 不属于已签名三角拓扑")
        evaluate_surface_anchor(vertices, anchor["triangle_vertices"], anchor.get("barycentric"))
    return actual


def mpfb_openpose_head_candidates(rig):
    """读取 MPFB 自带 OpenPose MEAN 语义；仅候选，不自动接受为 COCO 真值。

    MPFB 的 bone 名代表 tail 语义。Neck.tail 是鼻候选；眼骨的 head 同为鼻，
    耳骨的 head 是眼，因此不可按骨名直接取 head。
    """
    names = {"nose": "Neck", "left_eye": "LEye", "right_eye": "REye",
             "left_ear": "LEar", "right_ear": "REar"}
    result = {}
    for name, bone in names.items():
        tail = rig["bones"][bone]["tail"]
        indices = tail.get("vertex_indices", [])
        if tail.get("strategy") != "MEAN" or not indices or any(type(i) is not int or i < 0 for i in indices):
            raise ValueError("不支持的 MPFB 语义策略: " + name)
        result[name] = {"vertex_indices": list(indices), "source": bone + ".tail",
                        "status": "candidate_unreviewed"}
    return result
