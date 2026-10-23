"""离线UE矩阵/LBS候选：原始RefToLocal SHA完全一致才允许计算表面。

只实现当前Win64非FMA逐项float32路径，不猜隐藏骨、leader或ref-pose override；
未记录状态由最终SHA拒绝，计算表面也不自动证明GPU/RGB或可见性。
"""

import base64
import hashlib
import math
import re

import numpy as np


ARITHMETIC = "ue58_win64_transform_f64_to_f32_sse_order_non_fma_v1"
REF_SCHEME = "ue_ref_to_local_rowmajor_f32_le_v1"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def normalize_keys(value):
    """与已发布native JSON键规范相同；不转换标量类型或伪造缺字段。"""
    if isinstance(value, list):
        return [normalize_keys(item) for item in value]
    if not isinstance(value, dict):
        return value
    output = {}
    for key, item in value.items():
        if len(key) > 1 and key[0] == "b" and key[1].isupper():
            key = key[1:]
        key = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", key)
        key = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", key).lower()
        _require(key not in output, "规范化后出现重复键")
        output[key] = normalize_keys(item)
    return output


def transform_matrix_f32(transform):
    """复现FTransform<double>.ToMatrixWithScale后cast FMatrix44f的行主序。

    先按源码独立double加/乘构造，禁止先把quat降成float32或使用BLAS。
    """
    values = []
    for key, count in (("translation_cm", 3), ("rotation_xyzw", 4), ("scale_xyz", 3)):
        vector = transform.get(key)
        _require(isinstance(vector, list) and len(vector) == count
                 and all(type(v) in (int, float) and math.isfinite(v) for v in vector), "Transform显式编码缺失/非有限")
        values.append([float(v) for v in vector])
    translation, (x, y, z, w), scale = values
    _require(abs(sum(v*v for v in (x, y, z, w))-1) <= 1e-3, "非单位四元数不能猜归一化")
    _require(all(abs(v) > 1e-12 for v in scale), "原骨Transform scale退化")
    x2, y2, z2 = x+x, y+y, z+z
    xx2, yy2, zz2 = x2*x, y2*y, z2*z
    xy2, yz2, xz2 = x*y2, y*z2, x*z2
    wz2, wx2, wy2 = w*z2, w*x2, w*y2
    sx, sy, sz = scale
    matrix = [[(1-(yy2+zz2))*sx, (xy2+wz2)*sx, (xz2-wy2)*sx, 0.],
              [(xy2-wz2)*sy, (1-(xx2+zz2))*sy, (yz2+wx2)*sy, 0.],
              [(xz2+wy2)*sz, (yz2-wx2)*sz, (1-(xx2+yy2))*sz, 0.],
              [*translation, 1.]]
    result = np.asarray(matrix, dtype=np.float32)
    _require(np.all(np.isfinite(result)), "Transform转float32溢出")
    return result


def multiply_f32_sse_order(left, right):
    """VectorMatrixMultiply非FMA路径：逐列依次乘、加并每次舍入f32。"""
    left, right = np.asarray(left, dtype=np.float32), np.asarray(right, dtype=np.float32)
    _require(left.shape == right.shape == (4, 4) and np.all(np.isfinite(left)) and np.all(np.isfinite(right)), "矩阵维度/数值错误")
    result = np.empty((4, 4), dtype=np.float32)
    for row in range(4):
        temp = np.float32(left[row, 0] * right[0])
        for index in range(1, 4):
            temp = np.float32(np.float32(left[row, index] * right[index]) + temp)
        result[row] = temp
    _require(np.all(np.isfinite(result)), "矩阵乘法float32溢出")
    return result


def matrix_bytes(matrices):
    """与native相同：每骨16个row-major f32 little-endian，保留有符号零。"""
    matrices = np.asarray(matrices, dtype=np.float32)
    _require(matrices.ndim == 3 and matrices.shape[1:] == (4, 4) and len(matrices) > 0
             and np.all(np.isfinite(matrices)), "矩阵数组无效")
    return matrices.astype("<f4", copy=False).tobytes(order="C")


def canonical_identity(identity):
    """只容纳现有LOD命名差异，不能靠名称/顶点数近似匹配。"""
    identity = normalize_keys(identity)
    result = {key: identity[key] for key in ("asset_path", "signature_scheme", "vertex_count", "topology_sha256", "geometry_sha256")}
    result["lod"] = identity["lod"] if "lod" in identity else identity["l_od"]
    _require(type(result["lod"]) is int and result["lod"] == 0 and type(result["vertex_count"]) is int and result["vertex_count"] > 0
             and result["signature_scheme"] == "ue_lod_le_u32_f32_v1"
             and all(isinstance(result[key], str) and re.fullmatch(r"[0-9a-f]{64}", result[key]) for key in ("topology_sha256", "geometry_sha256")),
             "实际LOD0 identity类型/签名无效")
    return result


def decode_skin_snapshot(snapshot):
    """读取独立C++只读API实际UStruct；调用方须显式选定包装层，不猜文件。

    Base64原字节优先，JSON解码值逐槽复核。当前仅固定1/2字节索引/权重，
    资产基础权重与历史component SHA匹配仍不声称当前component有效权重。
    """
    data = normalize_keys(snapshot)
    _require(data.get("schema") == "futsalmot_pose17_asset_base_skin_data_v1", "未知静态蒙皮schema")
    for key in ("read_succeeded", "historical_native_weight_sha256_matched", "local_read_stable"):
        _require(data.get(key) is True, "静态导出未通过: " + key)
    for key in ("scene_or_asset_mutation_performed", "component_effective_weights_read", "dynamic_surface_reconstruction_verified"):
        _require(data.get(key) is False, "静态导出范围不符: " + key)
    _require(data.get("weight_source") == "asset_lod0_imported_base_skin_weight_buffer"
             and data.get("skin_weight_signature_scheme") == "ue_component_lod0_fixed_skinweight_raw_v1", "不能猜profile/effective权重来源")
    _require(type(data.get("engine_frame_before")) is int and data["engine_frame_before"] == data.get("engine_frame_after"), "静态读取帧不稳定")
    identity = canonical_identity(data["mesh_identity"])
    vertex_count = identity["vertex_count"]
    raw = base64.b64decode(data["raw_weight_data_base64"], validate=True)
    _require(_sha(raw) == data["skin_weight_data_sha256"] == data["expected_historical_skin_weight_sha256"], "基础权重原始SHA不匹配")
    _require(type(data.get("raw_weight_data_byte_count")) is int and len(raw) == data["raw_weight_data_byte_count"], "基础权重原始字节数不同")
    index_bytes, weight_bytes, influences = (data[k] for k in ("bone_index_byte_size", "bone_weight_byte_size", "max_bone_influences"))
    _require(all(type(v) is int for v in (index_bytes, weight_bytes, influences)) and index_bytes in (1, 2)
             and weight_bytes in (1, 2) and influences > 0, "固定权重布局类型错误")
    stride, offset = (index_bytes+weight_bytes)*influences, index_bytes*influences
    _require(data["constant_vertex_stride_bytes"] == stride and data["constant_weights_offset_bytes"] == offset
             and len(raw) == stride*vertex_count and data["max_raw_bone_weight"] == 65535
             and np.float32(data["raw_weight_to_float_multiplier"]) == np.float32(1/65535), "固定权重布局与字节数不同")
    raw_vertices = np.frombuffer(raw, dtype=np.uint8).reshape(vertex_count, stride)
    local = raw_vertices[:, :offset].copy().view("<u%d" % index_bytes).reshape(vertex_count, influences).astype(np.int32)
    raw_u16 = raw_vertices[:, offset:].copy().view("<u%d" % weight_bytes).reshape(vertex_count, influences).astype(np.int32)
    if weight_bytes == 1:
        raw_u16 *= 257
    normalized = np.float32(raw_u16.astype(np.float32) * np.float32(1/65535))
    bones = data["bones"]
    _require(isinstance(bones, list) and bones, "缺实际参考骨")
    names = [b["name"] for b in bones]
    _require(len(set(names)) == len(names) and all(isinstance(n, str) and n for n in names), "实际参考骨名错误")
    for index, bone in enumerate(bones):
        _require(type(bone["parent_index"]) is int and -1 <= bone["parent_index"] < index, "实际参考骨parent错误")
    inverse_raw = base64.b64decode(data["inverse_reference_matrices_base64"], validate=True)
    _require(data["inverse_reference_signature_scheme"] == "ue_inverse_ref_pose_rowmajor_f32_le_v1"
             and len(inverse_raw) == 64*len(bones) and _sha(inverse_raw) == data["inverse_reference_matrices_sha256"], "逆参考矩阵原始字节/签名错误")
    inverses = np.frombuffer(inverse_raw, dtype="<f4").reshape(len(bones), 4, 4).copy()
    _require(np.all(np.isfinite(inverses)), "逆参考矩阵非有限")
    for index, bone in enumerate(bones):
        decoded = np.asarray(bone["inverse_reference_row_major_f32"], dtype=np.float32)
        _require(decoded.shape == (16,) and decoded.astype("<f4").tobytes() == inverses[index].astype("<f4").tobytes(), "JSON逆参考矩阵与Base64不同")
    global_indices = np.empty_like(local)
    visited = np.zeros(vertex_count, dtype=bool)
    section_indices = np.full(vertex_count, -1, dtype=np.int32)
    for section_index, section in enumerate(data["sections"]):
        start, count = section["base_vertex_index"], section["vertex_count"]
        _require(type(start) is int and type(count) is int and start >= 0 and count > 0 and start+count <= vertex_count
                 and section["section_index"] == section_index and section["disabled"] is False and not visited[start:start+count].any(), "实际section范围/重叠错误")
        bone_map = section["bone_map"]
        _require(isinstance(bone_map, list) and bone_map and all(type(i) is int and 0 <= i < len(bones) for i in bone_map), "section BoneMap全局骨错误")
        indices = local[start:start+count]
        _require(np.all(indices < len(bone_map)), "section局部骨越界")
        global_indices[start:start+count] = np.asarray(bone_map, dtype=np.int32)[indices]
        visited[start:start+count] = True
        section_indices[start:start+count] = section_index
    _require(visited.all() and len(data["vertices"]) == vertex_count and np.all(raw_u16.sum(axis=1) > 0), "权重顶点覆盖错误或全零影响")
    for index, vertex in enumerate(data["vertices"]):
        _require(vertex["vertex_index"] == index and vertex["section_index"] == int(section_indices[index])
                 and vertex["section_local_bone_indices"] == local[index].tolist()
                 and vertex["global_bone_indices"] == global_indices[index].tolist()
                 and vertex["raw_weights_u16"] == raw_u16[index].tolist()
                 and np.asarray(vertex["normalized_weights"], dtype="<f4").tobytes() == normalized[index].astype("<f4").tobytes(),
                 "实际权重JSON解码值与原始Base64/section映射不同")
    return {"identity": identity, "weights_sha256": _sha(raw), "inverse_matrices_sha256": _sha(inverse_raw),
            "bone_names": names, "active_bone_indices": data["active_bone_indices"], "inverse_matrices": inverses,
            "global_bone_indices": global_indices, "normalized_weights": normalized, "raw_weights_u16": raw_u16}


def candidate_ref_to_local(row, skin):
    """仅返回固定数学候选；原SHA不一致时不返回可用矩阵。"""
    _require(row.get("ref_to_local_signature_scheme") == REF_SCHEME, "未知历史RefToLocal签名")
    expected = row.get("ref_to_local_matrices_sha256")
    _require(isinstance(expected, str) and re.fullmatch(r"[0-9a-f]{64}", expected), "历史矩阵SHA缺失")
    _require(canonical_identity(row["mesh_identity"]) == skin["identity"], "历史实际mesh与静态蒙皮身份不同")
    _require(row.get("skin_weight_signature_scheme") == "ue_component_lod0_fixed_skinweight_raw_v1"
             and row.get("skin_weight_data_sha256") == skin["weights_sha256"], "历史component有效权重与静态基础字节不同")
    _require(row.get("component_path") == row.get("bone_space_base_component_path"), "leader骨映射未落盘，拒绝猜测")
    _require(row.get("bone_pose_transform_encoding") == "ue_transform_translation_cm_quat_xyzw_scale_xyz_v1", "骨Transform编码不符")
    _require(row.get("bone_space_base_names") == skin["bone_names"], "实际骨名/顺序与资产不同")
    transforms = row.get("bone_space_base_transforms_ue_cm")
    _require(isinstance(transforms, list) and len(transforms) == len(skin["bone_names"]), "骨Transform数量不符")
    inverses = np.asarray(skin["inverse_matrices"], dtype=np.float32)
    _require(inverses.shape == (len(transforms), 4, 4), "逆参考矩阵数量不符")
    matrices = np.repeat(np.eye(4, dtype=np.float32)[None, :, :], len(transforms), axis=0)
    active = skin["active_bone_indices"]
    _require(isinstance(active, list) and len(set(active)) == len(active)
             and all(type(i) is int and 0 <= i < len(transforms) for i in active), "实际活跃骨集合无效")
    for index in active:
        matrices[index] = multiply_f32_sse_order(inverses[index], transform_matrix_f32(transforms[index]))
    actual = _sha(matrix_bytes(matrices))
    report = {"arithmetic": ARITHMETIC, "candidate_sha256": actual, "expected_native_sha256": expected,
              "ref_to_local_sha256_matched": actual == expected, "inactive_bones_are_identity": True,
              "unrecorded_visibility_and_ref_pose_override_assumed_absent_candidate_only": True,
              "render_surface_equivalence_verified": False, "visibility_state": "unknown"}
    # 拒绝把不一致的矩阵偷偷传给后续LBS；报告只保留可复核SHA。
    return report, matrices if actual == expected else None


def skin_component_positions_f32(reference_positions, global_indices, weights, matrices):
    """固定槽序逐项float32 LBS；调用者必须已通过原SHA门禁。"""
    positions = np.asarray(reference_positions, dtype=np.float32)
    indices = np.asarray(global_indices)
    weights, matrices = np.asarray(weights, dtype=np.float32), np.asarray(matrices, dtype=np.float32)
    _require(positions.ndim == 2 and positions.shape[1] == 3 and indices.ndim == 2
             and indices.shape == weights.shape and len(indices) == len(positions)
             and np.issubdtype(indices.dtype, np.integer) and np.all(indices >= 0) and np.all(indices < len(matrices))
             and np.all(np.isfinite(positions)) and np.all(np.isfinite(weights)) and np.all(weights >= 0), "LBS输入错误")
    output = np.zeros_like(positions)
    for influence in range(indices.shape[1]):
        selected = matrices[indices[:, influence]]
        transformed = np.float32(selected[:, 0, :3] * positions[:, 0, None])
        for axis in (1, 2):
            transformed = np.float32(np.float32(selected[:, axis, :3] * positions[:, axis, None]) + transformed)
        transformed = np.float32(selected[:, 3, :3] + transformed)
        output = np.float32(output + np.float32(transformed * weights[:, influence, None]))
    _require(np.all(np.isfinite(output)), "LBS输出非有限")
    return output


def reconstruct_surface(row, skin, reference_positions):
    """每次内部强制重算原SHA；任何不匹配均拒绝表面输出。"""
    report, matrices = candidate_ref_to_local(row, skin)
    _require(matrices is not None, "RefToLocal原始SHA不一致，停止动态表面重建: " + report["candidate_sha256"])
    return report, skin_component_positions_f32(reference_positions, skin["global_bone_indices"], skin["normalized_weights"], matrices)


def decode_reference_geometry(payload, skin):
    """既有参考geometry JSON逐字节核验；仅兼容旧导出的十进制整数字符串。"""
    payload = normalize_keys(payload)
    identity = dict(payload["identity"])
    for key in ("lod", "l_od", "vertex_count"):
        if key in identity and isinstance(identity[key], str):
            _require(re.fullmatch(r"0|[1-9][0-9]*", identity[key]), "参考几何整数字符串无效")
            identity[key] = int(identity[key])
    identity = canonical_identity(identity)
    _require(identity == skin["identity"], "参考几何与静态skin mesh身份不同")
    positions = np.asarray(payload["positions_cm"], dtype=np.float32)
    triangles = np.asarray(payload["triangles"])
    _require(positions.shape == (identity["vertex_count"], 3) and np.all(np.isfinite(positions)), "参考位置维度/数值错误")
    _require(triangles.ndim == 2 and triangles.shape[1] == 3 and len(triangles) > 0
             and np.issubdtype(triangles.dtype, np.integer) and np.all(triangles >= 0)
             and np.all(triangles < len(positions)), "有序参考拓扑错误")
    _require(_sha(positions.astype("<f4").tobytes()) == identity["geometry_sha256"]
             and _sha(triangles.astype("<u4").tobytes()) == identity["topology_sha256"], "参考位置/有序拓扑原始SHA不同")
    return positions, triangles.astype(np.int32)


def component_points_to_world_m(points_cm, transform):
    """UE非FMA Q.Rotate(Scale*P)+Translation的double顺序，最后cm转米。

    每机位保留原native世界Transform；不平均或按容差共享轨迹。
    """
    transform_matrix_f32(transform)  # 仅复用显式有限/单位quat/scale编码门禁。
    points = np.asarray(points_cm, dtype=np.float64)
    _require(points.ndim == 2 and points.shape[1] == 3 and np.all(np.isfinite(points)), "组件点维度/数值错误")
    q = np.asarray(transform["rotation_xyzw"], dtype=np.float64)
    scaled = points*np.asarray(transform["scale_xyz"], dtype=np.float64)
    def cross_left_q(values):
        return np.stack((q[1]*values[:,2]-q[2]*values[:,1],
                         q[2]*values[:,0]-q[0]*values[:,2],
                         q[0]*values[:,1]-q[1]*values[:,0]),axis=1)
    t = cross_left_q(scaled)
    t = t+t
    rotated = (q[3]*t+scaled)+cross_left_q(t)
    return (rotated+np.asarray(transform["translation_cm"],dtype=np.float64))*.01
