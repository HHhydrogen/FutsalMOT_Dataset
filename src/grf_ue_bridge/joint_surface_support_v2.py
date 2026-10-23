"""十二关节表面支持候选；保持骨中心坐标，永不升级visibility。"""

import hashlib
import json
import math

import numpy as np


JOINT_CHAINS = {
    "left_shoulder": ("upperarm_l", ("clavicle_l", "upperarm_l")),
    "right_shoulder": ("upperarm_r", ("clavicle_r", "upperarm_r")),
    "left_elbow": ("lowerarm_l", ("upperarm_l", "lowerarm_l")),
    "right_elbow": ("lowerarm_r", ("upperarm_r", "lowerarm_r")),
    "left_wrist": ("hand_l", ("lowerarm_l", "hand_l")),
    "right_wrist": ("hand_r", ("lowerarm_r", "hand_r")),
    "left_hip": ("thigh_l", ("pelvis", "thigh_l")),
    "right_hip": ("thigh_r", ("pelvis", "thigh_r")),
    "left_knee": ("calf_l", ("thigh_l", "calf_l")),
    "right_knee": ("calf_r", ("thigh_r", "calf_r")),
    "left_ankle": ("foot_l", ("calf_l", "foot_l")),
    "right_ankle": ("foot_r", ("calf_r", "foot_r")),
}
TOLERANCE_M = .005


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def prepare_region(positions_m, triangles, joint_weights, joint_center_m, radius_m, min_mean_weight):
    """向量化复现现有prepare_support_region：三角均权重+参考中心半径。"""
    positions = np.asarray(positions_m, dtype=np.float64)
    triangles = np.asarray(triangles)
    weights = np.asarray(joint_weights, dtype=np.float64)
    center = np.asarray(joint_center_m, dtype=np.float64)
    _require(positions.ndim == 2 and positions.shape[1] == 3 and np.all(np.isfinite(positions))
             and triangles.ndim == 2 and triangles.shape[1] == 3 and np.issubdtype(triangles.dtype, np.integer)
             and np.all(triangles >= 0) and np.all(triangles < len(positions))
             and weights.shape == (len(positions),) and np.all(np.isfinite(weights)) and np.all((weights >= 0) & (weights <= 1))
             and center.shape == (3,) and np.all(np.isfinite(center))
             and type(radius_m) in (int, float) and math.isfinite(radius_m) and radius_m > 0
             and type(min_mean_weight) in (int, float) and math.isfinite(min_mean_weight) and 0 <= min_mean_weight <= 1,
             "支持区几何、权重或候选参数无效")
    _require(np.all(triangles[:,0] != triangles[:,1]) and np.all(triangles[:,0] != triangles[:,2])
             and np.all(triangles[:,1] != triangles[:,2]), "有重复顶点的三角索引")
    selected = (weights[triangles].mean(axis=1) >= min_mean_weight) & (np.linalg.norm(positions[triangles].mean(axis=1)-center, axis=1) <= radius_m)
    return np.flatnonzero(selected).astype(np.int32)


def build_regions(skin, reference_positions_cm, triangles, radius_m=.08, min_mean_weight=.25):
    """只生成未审查支持区；阈值为预先声明候选，不能由像素结果自动拟合。"""
    names = skin["bone_names"]
    regions = {}
    for joint, (bone, chain) in JOINT_CHAINS.items():
        _require(all(name in names for name in chain), "实际骨链缺失: " + joint)
        index = names.index(bone)
        inverse = skin["inverse_matrices"][index].astype(np.float64)
        _require(np.allclose(inverse[:,3], [0,0,0,1], rtol=0, atol=0), "逆参考矩阵不是仿射编码")
        reference_center = np.linalg.inv(inverse)[3,:3]*.01
        chain_indices = [names.index(name) for name in chain]
        raw = np.where(np.isin(skin["global_bone_indices"], chain_indices), skin["raw_weights_u16"], 0).sum(axis=1, dtype=np.int64)
        chain_weights = raw.astype(np.float64)/65535.
        selected = prepare_region(reference_positions_cm*.01, triangles, chain_weights, reference_center, radius_m, min_mean_weight)
        region = {"status": "candidate_unreviewed", "joint_name": joint, "bone": bone, "bone_chain": list(chain),
                  "radius_m": radius_m, "min_mean_weight": min_mean_weight,
                  "reference_center_method": "inverse_of_actual_asset_inverse_reference_f32_matrix_candidate_only",
                  "reference_joint_center_m": reference_center.tolist(), "triangle_indices": selected.tolist(),
                  "mesh_identity": skin["identity"], "skin_weight_sha256": skin["weights_sha256"],
                  "coordinate_semantics": "bone_center_unchanged_visibility_support_only"}
        region["region_sha256"] = hashlib.sha256(json.dumps(region,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
        regions[joint] = region
    return regions


def pixel_ray(camera, x, y):
    """与已校准投影同一离散像素中心反投影，不用骨中心射线代替。"""
    intr, extr = camera["intrinsics"], camera["extrinsics"]
    direction = (np.asarray(extr["forward"],dtype=np.float64)
                 + np.asarray(extr["right"],dtype=np.float64)*((x-intr["cx"])/intr["fx"])
                 - np.asarray(extr["up"],dtype=np.float64)*((y-intr["cy"])/intr["fy"]))
    direction /= np.linalg.norm(direction)
    return np.asarray(extr["world_location_m"],dtype=np.float64), direction


def intersect_region(origin, direction, world_vertices_m, triangles, triangle_indices):
    """双面Möller–Trumbore最近真实三角命中；无closest-point补点。"""
    ids = np.asarray(triangle_indices, dtype=np.int32)
    if len(ids) == 0:
        return None
    direction = np.asarray(direction,dtype=np.float64)
    direction /= np.linalg.norm(direction)
    a,b,c = np.asarray(world_vertices_m,dtype=np.float64)[np.asarray(triangles)[ids]].transpose(1,0,2)
    edge1,edge2 = b-a,c-a
    h = np.cross(direction,edge2)
    det = np.sum(edge1*h,axis=1)
    valid = np.abs(det)>1e-12
    inv = np.zeros_like(det);inv[valid] = 1/det[valid]
    delta = np.asarray(origin,dtype=np.float64)-a
    u = np.sum(delta*h,axis=1)*inv
    q = np.cross(delta,edge1)
    v = np.sum(direction*q,axis=1)*inv
    t = np.sum(edge2*q,axis=1)*inv
    valid &= (u>=0)&(v>=0)&(u+v<=1)&(t>0)
    if not valid.any():
        return None
    local = np.flatnonzero(valid)[np.argmin(t[valid])]
    return {"triangle_index": int(ids[local]), "barycentric": [float(1-u[local]-v[local]),float(u[local]),float(v[local])],
            "world_m": (np.asarray(origin)+t[local]*direction).tolist(), "ray_distance_m": float(t[local])}


def observe_joint_support(uv_px, camera, world_vertices_m, triangles, region, depth_m, crypto_layers, target_token, actor_by_token):
    """九个实际像素中心各自求支持面，和同像素view-Z/纯coverage比较。

    返回候选状态，但visibility始终unknown；不得传字符串把候选变成正式标注。
    """
    result = {"visibility_state": "unknown", "candidate_state": "unknown", "occlusion_kind": None,
              "visibility_method": "not_measured", "region_sha256": region["region_sha256"],
              "region_status": region["status"], "support_triangle_count": len(region["triangle_indices"]),
              "reason": "not_evaluated", "pixel_samples": [], "tolerance_m": TOLERANCE_M}
    if not region["triangle_indices"]:
        result["reason"] = "empty_reference_support_region";return result
    if uv_px is None or any(not math.isfinite(float(v)) for v in uv_px):
        result["reason"] = "joint_outside_or_behind_camera";return result
    height,width = depth_m.shape
    x,y = [math.floor(float(v)+.5) for v in uv_px]
    result["joint_uv_px"] = list(uv_px);result["nearest_pixel_xy"] = [x,y]
    if x-1<0 or x+1>=width or y-1<0 or y+1>=height:
        result["reason"] = "image_boundary_or_insufficient_support";return result
    _require(len(crypto_layers)==3 and all(layer.shape == (height,width,4) for layer in crypto_layers), "需要三rank Cryptomatte和匹配深度尺寸")
    ys,xs = slice(y-1,y+2),slice(x-1,x+2)
    first = crypto_layers[0][ys,xs]
    token_grid = np.ascontiguousarray(first[:,:,0],dtype=np.float32).view(np.uint32)
    all_weights = np.stack([layer[ys,xs,k] for layer in crypto_layers for k in (1,3)])
    pure = bool(np.all(np.isfinite(all_weights)) and np.all(np.abs(first[:,:,1]-1)<1e-6)
                and np.all(np.abs(all_weights[1:])<1e-6))
    uniform = bool(np.all(token_grid==token_grid[1,1]))
    target_coverage = np.zeros((3,3),dtype=np.float64)
    for layer in crypto_layers:
        for token_channel,weight_channel in ((0,1),(2,3)):
            tokens = np.ascontiguousarray(layer[ys,xs,token_channel],dtype=np.float32).view(np.uint32)
            target_coverage += np.where(tokens==target_token,layer[ys,xs,weight_channel],0)
    result.update(pure_single_instance_coverage=pure, uniform_first_instance=uniform,
                  target_coverage_at_joint_pixel=float(target_coverage[1,1]) if math.isfinite(float(target_coverage[1,1])) else None,first_hit_token_hex="%08x"%int(token_grid[1,1]),
                  first_hit_actor=actor_by_token.get(int(token_grid[1,1])))
    origin = np.asarray(camera["extrinsics"]["world_location_m"],dtype=np.float64)
    forward = np.asarray(camera["extrinsics"]["forward"],dtype=np.float64)
    for py in range(y-1,y+2):
        for px in range(x-1,x+2):
            _,direction = pixel_ray(camera,px,py)
            hit = intersect_region(origin,direction,world_vertices_m,triangles,region["triangle_indices"])
            observed = float(depth_m[py,px])
            sample = {"pixel_xy": [px,py], "support_hit": hit, "observed_forward_depth_m": observed if math.isfinite(observed) else None}
            if hit is not None:
                z = float(np.dot(np.asarray(hit["world_m"])-origin,forward))
                sample.update(support_forward_depth_m=z,depth_delta_m=z-observed if math.isfinite(observed) else None)
            result["pixel_samples"].append(sample)
    hit_count = sum(s["support_hit"] is not None for s in result["pixel_samples"])
    result["support_ray_hit_count"] = hit_count
    if hit_count != 9:
        result["reason"] = "support_region_not_intersected_at_all_9_pixel_centres";return result
    observed = np.asarray([s["observed_forward_depth_m"] for s in result["pixel_samples"]],dtype=np.float64)
    if not np.all(np.isfinite(observed)) or np.any(observed<=0):
        result["reason"] = "invalid_or_background_depth";return result
    result["observed_depth_spread_m"] = float(observed.max()-observed.min())
    if not pure or not uniform:
        result["reason"] = "mixed_instance_or_partial_coverage";return result
    if result["observed_depth_spread_m"] > 2*TOLERANCE_M:
        result["reason"] = "depth_spread_exceeds_10mm_conservative_gate";return result
    deltas = np.asarray([s["depth_delta_m"] for s in result["pixel_samples"]])
    result["depth_delta_range_m"] = [float(deltas.min()),float(deltas.max())]
    first_token = int(token_grid[1,1])
    if first_token == target_token and np.all(np.abs(deltas)<=TOLERANCE_M):
        result.update(candidate_state="visible",reason="associated_joint_surface_first_hit_candidate")
    elif np.all(deltas>TOLERANCE_M):
        if first_token == target_token:
            result.update(candidate_state="occluded",occlusion_kind="self",reason="nearer_own_body_surface_candidate")
        elif first_token in actor_by_token:
            result.update(candidate_state="occluded",occlusion_kind="external",reason="nearer_other_actor_surface_candidate")
        else:
            result["reason"] = "nearer_unmapped_geometry_unknown"
    elif np.all(deltas < -TOLERANCE_M):
        result["reason"] = "support_in_front_of_rendered_surface_inconsistent"
    else:
        result["reason"] = "mixed_surface_depth_relation_or_other_instance_at_similar_depth"
    return result
