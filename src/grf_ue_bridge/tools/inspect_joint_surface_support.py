"""复用四张既有EXR和离线frame5表面，只写十二关节候选/unknown报告。"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from grf_ue_bridge.joint_surface_support_v2 import JOINT_CHAINS, TOLERANCE_M, build_regions, observe_joint_support
from grf_ue_bridge.pose17_export import COCO17_NAMES, project_landmarks
from grf_ue_bridge.pose17_offline_lbs_v2 import canonical_identity, component_points_to_world_m, decode_reference_geometry, decode_skin_snapshot
from grf_ue_bridge.tools.inspect_pose17_smoke import camera_from_exr_header, check_pair, sampling_context


def read_json(path):
    raw = Path(path).read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def inspect(surface_report_path, paired_report_path, calibration_path, output, radius_m=.08, min_mean_weight=.25):
    """只支持明确frame5四cam×十人×十二点；每camera只打开一次实际EXR。"""
    import OpenEXR
    output = Path(output).resolve()
    _require(not output.exists(), "新诊断目录已存在，拒绝覆盖")
    surface_path = Path(surface_report_path).resolve()
    surface, surface_sha = read_json(surface_path)
    pairing, pairing_sha = read_json(paired_report_path)
    calibration, calibration_sha = read_json(calibration_path)
    _require(surface.get("all_requested_native_matrix_sha_matched") is True and surface.get("errors") == []
             and surface.get("matched_rows") == 400 and surface.get("mismatched_rows") == 0,
             "需真实四cam400条全部精确矩阵SHA通过")
    _require(pairing.get("structural_pairing_passed") is True and pairing.get("errors") == [], "需已通过的相同attempt RGB/EXR/native配对")
    _require(calibration.get("schema") == "futsalmot_depth_encoding_calibration_v1" and calibration.get("rows")
             and all(r["summary"]["R_cm_to_view_z_error_m"]["max_abs"]<=TOLERANCE_M for r in calibration["rows"]),
             "需既有真实深度通道平面标定通过5mm")
    ue_dir = Path(__file__).resolve().parents[3]/"ue"
    if str(ue_dir) not in sys.path:
        sys.path.insert(0,str(ue_dir))
    from pose17_native_receipt import normalize_native_receipt
    models = {}
    for asset, source in surface["skin_sources"].items():
        raw, sha = read_json(source["path"])
        _require(sha==source["file_sha256"], "静态skin文件相对离线收据已变化")
        skin = decode_skin_snapshot(raw)
        _require(skin["identity"]["asset_path"] == asset, "静态skin资产错配")
        models[asset] = {"skin": skin}
    selected = [r for r in surface["rows"] if r.get("component_surface_file")]
    expected_entities = {s+str(i) for s in "LR" for i in range(5)}
    expected_cameras = {"CineCam_%02d"%i for i in range(1,5)}
    _require(len(selected)==40 and {r["camera_id"] for r in selected}==expected_cameras
             and all(r["output_frame"]==5 for r in selected)
             and {(r["camera_id"],r["entity_id"]) for r in selected}=={(c,e) for c in expected_cameras for e in expected_entities},
             "需frame5四camera×十唯一实体NPZ，不自动取其他帧")
    native_sources = {str(Path(s["path"]).resolve()):s["file_sha256"] for s in surface["native_sources"]}
    native_cache = {}
    for item in selected:
        asset = item["mesh_identity"]["asset_path"]
        model = models[asset]
        if "triangles" not in model:
            ref = item["reference_geometry_ref"]
            raw, sha = read_json(ref["path"])
            _require(sha==ref["file_sha256"], "参考geometry文件相对离线收据已变化")
            positions, triangles = decode_reference_geometry(raw, model["skin"])
            model.update(positions=positions,triangles=triangles,
                         regions=build_regions(model["skin"],positions,triangles,radius_m,min_mean_weight))
        path = str(Path(item["native_ref"]).resolve())
        if path not in native_cache:
            raw, sha = read_json(path)
            _require(sha==native_sources[path] and Path(pairing["attempt"]).resolve()==Path(path).parent, "native/配对/离线表面不属于同一未变化attempt")
            receipt = normalize_native_receipt(raw)
            plan,_ = read_json(Path(path).parent/"plan.json")
            context = sampling_context(plan,receipt)
            native_cache[path] = ({(r["output_frame"],r["entity_id"]):r for r in receipt["records"]},context)
    report = {"schema":"futsalmot_joint_surface_support_candidate_v1","visibility_state":"unknown","COCO_written":False,
              "surface_report_ref":str(surface_path),"surface_report_sha256":surface_sha,
              "pairing_report_ref":str(Path(paired_report_path).resolve()),"pairing_report_sha256":pairing_sha,
              "depth_encoding_calibration_ref":str(Path(calibration_path).resolve()),"depth_encoding_calibration_sha256":calibration_sha,
              "candidate_parameters":{"reference_radius_m":radius_m,"min_mean_chain_weight":min_mean_weight,"tolerance_m":TOLERANCE_M,"pixel_radius":1},
              "parameters_fitted_to_rendered_pixels":False,"anatomical_support_regions_semantically_validated":False,
              "render_surface_equivalence_verified":False,"pixel_rule":"nine integer pixel centres; nearest=floor(uv+.5); calibrated cx=(w-1)/2",
              "depth_semantics":"FinalImageWorldDepth.R float32 linear_camera_forward_z_cm / 100",
              "limitations":["关节3D/2D仍是native骨中心，表面支持只作未审查解剖代理", "没有用骨中心Z判自遮挡或mask命中判可见",
                             "三角region语义、独立定位精度与最终GPU表面等价仍未正式验收", "纯coverage/同实例/10mm邻域深度spread门禁保守丢点；不为增加候选而改阈值"],
              "cameras":{},"errors":[]}
    output.mkdir(parents=True)
    region_payload = {a:m["regions"] for a,m in models.items()}
    (output/"support_regions.json").write_text(json.dumps(region_payload,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    for name in sorted(expected_cameras):
        camera_rows = []
        report["cameras"][name] = {"points":camera_rows,"exr_open_count":0}
        try:
            paired = [r for r in pairing["cameras"][name]["frames"] if r["output_frame"]==5]
            exr_paths = {r["EXR"] for r in paired}
            _require(len(paired)==10 and len(exr_paths)==1, "配对报告frame5没有唯一EXR和十实体")
            exr_path = next(iter(exr_paths))
            exr = OpenEXR.File(exr_path)
            report["cameras"][name].update(exr_ref=exr_path,exr_open_count=1)
            header,channels = exr.header(),exr.channels()
            camera = camera_from_exr_header(header)
            depth = [v.pixels for k,v in channels.items() if k in ("FinalImageWorldDepth","FinalImageWorldDepth.RGBA")]
            _require(len(depth)==1 and depth[0].dtype==np.float32,"实际FinalImageWorldDepth层不唯一或不是float32")
            depth_m = depth[0][:,:,0]*np.float32(.01)
            crypto = [v.pixels for k,v in sorted(channels.items()) if k.startswith("ActorHitProxyMask")]
            _require(len(crypto)==3 and all(layer.dtype==np.float32 for layer in crypto),"实际Cryptomatte三rank缺失/非float32")
            keys = [k for k in header if k.startswith("cryptomatte/") and k.endswith("/manifest")]
            _require(len(keys)==1,"实际Cryptomatte manifest不唯一")
            manifest = json.loads(header[keys[0]])
            actor_by_token = {int(token,16):actor for actor,token in manifest.items()}
            for item in [r for r in selected if r["camera_id"]==name]:
                records,context = native_cache[str(Path(item["native_ref"]).resolve())]
                row = records[(5,item["entity_id"])]
                check_pair(row,header,name,5,context)
                model = models[row["mesh_identity"]["asset_path"]]
                with np.load(surface_path.parent/item["component_surface_file"],allow_pickle=False) as data:
                    _require(data["positions_cm"].dtype==np.float32 and data["positions_cm"].shape==(row["mesh_identity"]["vertex_count"],3)
                             and str(data["coordinate_space"])=="mesh_component" and str(data["unit"])=="cm"
                             and canonical_identity(json.loads(str(data["mesh_identity_json"])))==canonical_identity(row["mesh_identity"])
                             and str(data["native_ref_to_local_sha256"])==row["ref_to_local_matrices_sha256"]
                             and str(data["skin_weight_sha256"])==row["skin_weight_data_sha256"]
                             and str(data["camera_id"])==name and str(data["entity_id"])==row["entity_id"]
                             and int(data["output_frame"])==5 and int(data["source_frame"])==row["sequence_source_frame"]
                             and json.loads(str(data["visual_world_transform_json"]))==row["visual_world_transform_ue_cm"], "NPZ身份/帧/单位/变换与真实native不符")
                    world_vertices = component_points_to_world_m(data["positions_cm"],row["visual_world_transform_ue_cm"])
                token = int(manifest["Player_"+row["entity_id"]],16)
                projected = project_landmarks([{"world_m":p} for p in row["world_m"][5:]],camera)
                for index,joint in enumerate(COCO17_NAMES[5:]):
                    _require(joint in JOINT_CHAINS,"不是十二COCO关节")
                    point = observe_joint_support(projected[index].get("uv_px"),camera,world_vertices,model["triangles"],model["regions"][joint],depth_m,crypto,token,actor_by_token)
                    point.update(camera_id=name,entity_id=row["entity_id"],keypoint_name=joint,output_frame=5,source_frame=row["sequence_source_frame"],
                                 joint_world_m=row["world_m"][5+index],joint_uv_px=projected[index].get("uv_px"),
                                 mesh_asset_path=row["mesh_identity"]["asset_path"],native_ref_to_local_sha256=row["ref_to_local_matrices_sha256"],
                                 component_surface_ref=item["component_surface_file"])
                    camera_rows.append(point)
            # 每相机一次全EXR解码；不把4K数组保留到下一机位。
            del channels,exr,depth,depth_m,crypto
        except Exception as error:
            report["errors"].append(name+": "+repr(error))
    points = [r for cam in report["cameras"].values() for r in cam["points"]]
    report["point_count"] = len(points)
    report["candidate_counts"] = dict(Counter(r["candidate_state"] for r in points))
    report["reason_counts"] = dict(Counter(r["reason"] for r in points))
    report["support_ray_hit_count_histogram"] = dict(Counter(str(r.get("support_ray_hit_count",0)) for r in points))
    report["by_joint"] = {joint:{"candidate_counts":dict(Counter(r["candidate_state"] for r in points if r["keypoint_name"]==joint)),
        "reason_counts":dict(Counter(r["reason"] for r in points if r["keypoint_name"]==joint))} for joint in JOINT_CHAINS}
    report["completed_candidate_evaluation"] = len(points)==480 and not report["errors"]
    (output/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("surface_report","paired_report","depth_calibration","output"):
        parser.add_argument("--"+name.replace("_","-"),type=Path,required=True)
    parser.add_argument("--support-radius-m",type=float,default=.08)
    parser.add_argument("--min-mean-weight",type=float,default=.25)
    args = parser.parse_args()
    report = inspect(args.surface_report,args.paired_report,args.depth_calibration,args.output,args.support_radius_m,args.min_mean_weight)
    print(json.dumps({k:report[k] for k in ("completed_candidate_evaluation","point_count","candidate_counts","reason_counts","errors")},ensure_ascii=False))
    return 0 if report["completed_candidate_evaluation"] else 2


if __name__=="__main__":
    raise SystemExit(main())
