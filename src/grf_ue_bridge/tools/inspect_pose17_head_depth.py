"""头5个真实表面锚点的深度残差诊断；不写COCO或升级visibility/sync。"""

import argparse
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np

from grf_ue_bridge.pose17_export import COCO17_NAMES, project_landmarks
from grf_ue_bridge.tools.inspect_pose17_smoke import camera_from_exr_header, check_pair, media_index, sampling_context


TOLERANCE_M = 0.005


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def pixel_observation(uv, expected_depth_m, depth_m, crypto_layers, target_bits, radius=1):
    """最近像素及3x3原始支持；混合ID/coverage或深度边界保持unknown。"""
    result = {"uv_px": uv, "expected_forward_depth_m": expected_depth_m, "visibility_state": "unknown",
              "tolerance_m": TOLERANCE_M, "candidate": "unknown", "unknown_reasons": []}
    height,width = depth_m.shape
    if uv is None or expected_depth_m <= 0:
        result["unknown_reasons"] = ["outside_or_behind_camera"]
        return result
    x,y = [math.floor(float(v)+.5) for v in uv]
    result["nearest_pixel_xy"] = [x,y]
    result["pixel_center_offset_xy"] = [x-uv[0],y-uv[1]]
    if x-radius < 0 or x+radius >= width or y-radius < 0 or y+radius >= height:
        result["unknown_reasons"] = ["image_boundary_or_insufficient_support"]
        return result
    ys,xs = slice(y-radius,y+radius+1),slice(x-radius,x+radius+1)
    values = depth_m[ys,xs]
    _require(all(layer.shape == (height,width,4) for layer in crypto_layers), "Cryptomatte/depth尺寸或通道不同")
    first = crypto_layers[0]
    patch_ids = np.ascontiguousarray(first[ys,xs,0],dtype=np.float32).view(np.uint32)
    center_bits = int(np.asarray(first[y,x,0],dtype=np.float32).view(np.uint32))
    center_depth = float(depth_m[y,x])
    center_weights = [float(layer[y,x,k]) if math.isfinite(float(layer[y,x,k])) else None
                      for layer in crypto_layers for k in (1,3)]
    target_coverage = 0.0
    for layer in crypto_layers:
        for id_channel,weight_channel in ((0,1),(2,3)):
            token = int(np.asarray(layer[y,x,id_channel],dtype=np.float32).view(np.uint32))
            if token == target_bits:
                weight = float(layer[y,x,weight_channel])
                if math.isfinite(weight): target_coverage += weight
    center_valid = math.isfinite(center_depth) and center_depth>0
    result.update(first_hit_dominant_id_hex="%08x" % center_bits, target_instance_hit=center_bits==target_bits,
        target_coverage=target_coverage, cryptomatte_rank_coverage=center_weights,
        cryptomatte_coverage_sum=sum(center_weights) if None not in center_weights else None,
        observed_forward_depth_m=center_depth if center_valid else None,
        signed_residual_m=expected_depth_m-center_depth if center_valid else None,
        absolute_residual_m=abs(expected_depth_m-center_depth) if center_valid else None,
        neighborhood_radius_px=radius, neighborhood_first_id_hex=["%08x" % int(v) for v in patch_ids.flat])
    valid_depth = bool(np.all(np.isfinite(values)) and np.all(values>0))
    uniform_id = bool(np.all(patch_ids==center_bits))
    pure_coverage = bool(np.all(np.abs(first[ys,xs,1]-1)<1e-6)
        and all(np.all(np.abs(layer[ys,xs,k])<1e-6) for n,layer in enumerate(crypto_layers)
                for k in ((3,) if n==0 else (1,3))))
    result.update(valid_depth_support=valid_depth, instance_boundary=not uniform_id,
                  full_single_instance_coverage_support=pure_coverage)
    if valid_depth:
        result["neighborhood_depth_range_m"] = [float(values.min()),float(values.max())]
        result["neighborhood_depth_spread_m"] = float(values.max()-values.min())
        result["depth_discontinuity_conservative_flag"] = result["neighborhood_depth_spread_m"] > 2*TOLERANCE_M
    reasons = result["unknown_reasons"]
    if not valid_depth: reasons.append("invalid_depth")
    if not uniform_id: reasons.append("instance_boundary")
    if not pure_coverage: reasons.append("mixed_or_partial_coverage")
    if result.get("depth_discontinuity_conservative_flag"): reasons.append("depth_spread_exceeds_10mm_conservative_gate")
    result["target_interior_support"] = valid_depth and uniform_id and pure_coverage and center_bits==target_bits
    if not reasons:
        delta = result["signed_residual_m"]
        result["candidate"] = ("surface_agreement_candidate" if abs(delta)<=TOLERANCE_M and center_bits==target_bits
            else "nearer_first_hit_self_candidate" if delta>TOLERANCE_M and center_bits==target_bits
            else "nearer_first_hit_other_candidate" if delta>TOLERANCE_M
            else "point_in_front_of_rendered_surface_inconsistent" if delta < -TOLERANCE_M
            else "different_instance_at_similar_depth_unknown")
    return result


def _stats(values):
    return {"count":len(values), "median_abs_residual_m": statistics.median(values) if values else None,
            "max_abs_residual_m":max(values) if values else None,
            "within_5mm":sum(v<=TOLERANCE_M for v in values)}


def summarize(rows):
    """同时报告每组有效率及共同支持，避免错配大量丢点后中位数产生误导。"""
    output = {}
    for offset in (-1,0,1):
        selected = [r for r in rows if r["pose_frame_offset"]==offset]
        interior = [r for r in selected if r.get("target_interior_support")]
        conservative = [r for r in interior if not r["unknown_reasons"]]
        output[str(offset)] = {"attempted_points":len(selected),
            "target_hit_points":sum(r.get("target_instance_hit",False) for r in selected),
            "target_interior_before_depth_spread_gate":_stats([r["absolute_residual_m"] for r in interior]),
            "conservative_nonboundary_target_support":_stats([r["absolute_residual_m"] for r in conservative]),
            "candidate_counts":{c:sum(r["candidate"]==c for r in selected) for c in sorted({r["candidate"] for r in selected})}}
    groups = {}
    for row in rows:
        groups.setdefault((row["image_frame"],row["entity_id"],row["keypoint_name"]),{})[row["pose_frame_offset"]] = row
    common = [group for group in groups.values() if set(group)=={-1,0,1}
        and all(r.get("target_interior_support") for r in group.values())]
    output["common_triplet_target_interior"] = {"points":len(common), "offsets":{
        str(offset):_stats([group[offset]["absolute_residual_m"] for group in common]) for offset in (-1,0,1)},
        "note":"同一个image_frame/entity/keypoint在三个Pose帧偏移均有目标纯实例内部支持；不是自动相位判决"}
    return output


def inspect(attempt, paired_report, calibration_ref, output):
    """消费已配对小样，R线性前向cm转换为m；只生成新诊断报告。"""
    import OpenEXR
    ue_dir = Path(__file__).resolve().parents[3]/"ue"
    if str(ue_dir) not in sys.path: sys.path.insert(0,str(ue_dir))
    from pose17_native_receipt import normalize_native_receipt
    attempt,output = Path(attempt).resolve(),Path(output).resolve()
    paired_report,calibration_ref = Path(paired_report).resolve(),Path(calibration_ref).resolve()
    _require(not output.exists(),"分析目录已存在，拒绝覆盖")
    pairing = json.loads(paired_report.read_text(encoding="utf-8"))
    _require(pairing.get("structural_pairing_passed") is True and Path(pairing["attempt"]).resolve()==attempt,
        "需要相同attempt已通过的结构配对报告")
    calibration = json.loads(calibration_ref.read_text(encoding="utf-8"))
    _require(calibration.get("schema")=="futsalmot_depth_encoding_calibration_v1","需要真实平面深度编码标定")
    _require(calibration.get("rows") and all(r["summary"]["R_cm_to_view_z_error_m"]["max_abs"]<=TOLERANCE_M for r in calibration["rows"]),
        "预注册5mm容差小于实际平面标定最大误差")
    plan = json.loads((attempt/"plan.json").read_text(encoding="utf-8"))
    report = {"schema":"futsalmot_head_surface_depth_residual_diagnostic_v1","attempt":str(attempt),
        "pairing_report_ref":str(paired_report),"depth_encoding_calibration_ref":str(calibration_ref),
        "calibration_sha256":hashlib.sha256(calibration_ref.read_bytes()).hexdigest(),
        "pre_registered_tolerance_m":TOLERANCE_M,"pixel_rule":"nearest_center_floor_uv_plus_half; center_minus_half_projection",
        "depth_semantics":"FinalImageWorldDepth.R float32 linear_camera_forward_z_cm / 100",
        "coverage_semantics":"ActorHitProxyMask ranks RG=ID/coverage, BA=ID/coverage; dominant token is not guaranteed same transparent surface as depth",
        "visibility_state":"unknown","MRQ_sync_verified":False,"COCO_written":False,
        "limitations":["平面标定不自动证明头部材质/WPO/头发/眼球最终表面一致","像素footprint和头表面斜率影响残差；不靠放宽5mm阈值解决",
            "3x3实例或coverage边界保持unknown；深度spread>10mm保守拒绝候选","相邻帧对照只支持诊断；不自动选择最佳相位修改数据"],
        "cameras":{},"errors":[]}
    output.mkdir(parents=True)
    for camera_plan in plan["cameras"]:
        name = camera_plan["camera_id"]
        camera_result = {"points":[],"errors":[]}
        report["cameras"][name] = camera_result
        try:
            receipt = normalize_native_receipt(json.loads((attempt/(name+"_native_pose_restore_receipt.json")).read_text(encoding="utf-8")))
            sampling = sampling_context(plan, receipt)
            rows = {(r["output_frame"],r["entity_id"]):r for r in receipt["records"]}
            exrs = media_index(attempt/name,"exr")
            for frame,path in sorted(exrs.items()):
                exr = OpenEXR.File(str(path))
                header,channels = exr.header(),exr.channels()
                camera = camera_from_exr_header(header)
                depth_layers = [v.pixels for k,v in channels.items() if k in ("FinalImageWorldDepth","FinalImageWorldDepth.RGBA")]
                _require(len(depth_layers)==1,"无法唯一选择深度层")
                _require(depth_layers[0].dtype==np.float32,"深度层不是float32")
                depth_m = depth_layers[0][:,:,0]*.01
                crypto = [v.pixels for k,v in sorted(channels.items()) if k.startswith("ActorHitProxyMask")]
                _require(len(crypto)==3,"需要真实Cryptomatte三rank层及coverage")
                manifest_keys = [k for k in header if k.startswith("cryptomatte/") and k.endswith("/manifest")]
                _require(len(manifest_keys)==1,"Cryptomatte manifest必须唯一")
                manifest = json.loads(header[manifest_keys[0]])
                actor_by_id = {str(token).lower():label for label,token in manifest.items()}
                for target in plan["targets"]:
                    entity = target["entity_id"]
                    current = rows[(frame,entity)]
                    check_pair(current,header,name,frame,sampling)
                    token = int(manifest["Player_"+entity],16)
                    for offset in (-1,0,1):
                        pose = rows.get((frame+offset,entity))
                        if pose is None: continue
                        _require(pose["sample_succeeded"] and pose["keypoint_names"][:5]==list(COCO17_NAMES[:5]),"头5点采样/顺序错误")
                        projected = project_landmarks([{"world_m":p} for p in pose["world_m"][:5]],camera)
                        for point_index,point in enumerate(projected):
                            observed = pixel_observation(point.get("uv_px"),point.get("camera_forward_depth_m",-1),depth_m,crypto,token)
                            observed["first_hit_dominant_actor_label"] = actor_by_id.get(observed.get("first_hit_dominant_id_hex"))
                            observed.update(image_frame=frame,pose_frame=frame+offset,pose_frame_offset=offset,
                                image_sequence_source_frame=current["sequence_source_frame"],pose_sequence_source_frame=pose["sequence_source_frame"],
                                entity_id=entity,keypoint_name=COCO17_NAMES[point_index],world_m=pose["world_m"][point_index],
                                native_sampling_method=pose["sampling_method"],manifest_sha256=pose["manifest_sha256"])
                            camera_result["points"].append(observed)
                # 不把每个EXR的全分辨率数组保留到下一帧。
                del channels,exr,depth_layers,depth_m,crypto
            camera_result["summary"] = summarize(camera_result["points"])
        except Exception as error:
            camera_result["errors"].append(repr(error))
            report["errors"].append(name+": "+repr(error))
    (output/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("attempt","paired_report","calibration_ref","output"): parser.add_argument(field,type=Path)
    args = parser.parse_args()
    report = inspect(args.attempt,args.paired_report,args.calibration_ref,args.output)
    print(json.dumps({"errors":report["errors"],"MRQ_sync_verified":False,"visibility_state":"unknown",
        "summaries":{k:v.get("summary") for k,v in report["cameras"].items()},"report":str(args.output/"report.json")},ensure_ascii=False))
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
