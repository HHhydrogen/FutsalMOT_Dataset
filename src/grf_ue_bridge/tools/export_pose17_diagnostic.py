"""只将已验联合小样导出为独立诊断组；保留逐相机原始世界点，不生成训练准入。"""

import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import shutil
import stat
import sys

import numpy as np
from PIL import Image

from grf_ue_bridge.pose17_export import COCO17_NAMES, project_landmarks
from grf_ue_bridge.tools.inspect_pose17_smoke import media_index

REPO = Path(__file__).resolve().parents[3]
WORKSPACE_SAVED = REPO.parents[2] / "Saved"
if str(REPO / "ue") not in sys.path:
    sys.path.insert(0,str(REPO / "ue"))
from pose17_native_receipt import normalize_native_receipt
from pose17_override_receipt import validate_joint_bundle
from pose17_location_pilot_contract import validate_location_pilot_contract


def require(ok,message):
    if not ok: raise ValueError(message)


def guard(path):
    """只读写当前workspace Saved内普通路径，不跟随junction或symlink。"""
    path = Path(os.path.abspath(os.fspath(path)))
    root = Path(os.path.abspath(os.fspath(WORKSPACE_SAVED)))
    require(path!=root and root in path.parents,"路径必须位于当前workspace Saved内")
    for part in reversed((path,)+tuple(path.parents)):
        if not part.exists() and not part.is_symlink(): continue
        info = part.lstat()
        require(not stat.S_ISLNK(info.st_mode) and not(getattr(info,"st_file_attributes",0)&0x400),"拒绝联接/符号链接")
    return path


def read_json(path):
    return json.loads(guard(path).read_text(encoding="utf-8-sig"))


def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x",encoding="utf-8",newline="\n") as handle:
        json.dump(value,handle,ensure_ascii=False,indent=2,allow_nan=False)
        handle.write("\n")


def write_jsonl(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x",encoding="utf-8",newline="\n") as handle:
        for row in rows: handle.write(json.dumps(row,ensure_ascii=False,allow_nan=False,separators=(",",":"))+"\n")


def decode_mask(exr,mapping):
    """每份EXR调用一次现有精确Cryptomatte解码，不读取深度或生成额外审计图。"""
    from grf_ue_bridge.cryptomatte import load_cryptomatte, build_mask
    manifest,ids = load_cryptomatte(exr)
    require(all(label in manifest for label in mapping.values()),"EXR缺少已验映射Actor")
    return build_mask(manifest,ids,mapping)


def mask_geometry(mask,mask_id):
    ys,xs = np.where(mask==mask_id)
    return (None,0) if len(xs)==0 else ([int(xs.min()),int(ys.min()),int(xs.max()-xs.min()+1),int(ys.max()-ys.min()+1)],int(len(xs)))


def _export_mask_frame(job):
    """独立进程一次解码/保存一帧，仅返回尺寸和十实体几何小结果。"""
    frame,exr,rgb,destination,mapping = job
    mask,counts = decode_mask(exr,mapping)
    require(isinstance(mask,np.ndarray) and mask.ndim==2 and mask.dtype==np.uint8
        and set(np.unique(mask)).issubset(set(range(11))),"需要0背景/非球员及1..10球员uint8 mask")
    width,height = mask.shape[1],mask.shape[0]
    with Image.open(rgb) as image:
        require(image.size==(width,height),"RGB/mask尺寸不同")
    geometry = {}
    for entity in mapping:
        track = int(entity[1])+(1 if entity[0]=="L" else 6)
        bbox,area = mask_geometry(mask,track)
        require(counts[entity]==area,"mask计数与bbox解码不同")
        geometry[entity] = (bbox,area)
    with Path(destination).open("xb") as handle:
        Image.fromarray(mask).save(handle,format="PNG")
    return frame,width,height,geometry


def _mask_results(jobs,workers):
    """Windows spawn；最多workers个在途帧，按输入顺序汇总，不传像素数组。"""
    require(type(workers) is int and 1<=workers<=6,"CPU workers必须为1..6整数")
    if workers==1:
        for job in jobs:
            yield _export_mask_frame(job)
        return
    jobs = iter(jobs)
    pending = deque()
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context("spawn")) as pool:
        try:
            for _ in range(workers):
                job = next(jobs,None)
                if job is not None: pending.append(pool.submit(_export_mask_frame,job))
            while pending:
                result = pending.popleft().result()
                job = next(jobs,None)
                if job is not None: pending.append(pool.submit(_export_mask_frame,job))
                yield result
        finally:
            for future in pending: future.cancel()


def load_inputs(attempt,alignment,native_report):
    attempt,alignment,native_report = guard(attempt),guard(alignment),guard(native_report)
    plan,state = read_json(attempt/"plan.json"),read_json(attempt/"state.json")
    pilot = validate_location_pilot_contract(plan)
    require(type(plan.get("frame_count")) is int and plan["frame_count"]==(120 if pilot else 10)
        and len(plan["cameras"])==4,"只接收四相机十帧联合诊断或显式冻结120帧位置试产")
    aligned,accepted = read_json(alignment),read_json(native_report)
    require(aligned.get("structural_pairing_passed") is True and not aligned.get("errors"),"结构配对报告未通过")
    require(accepted.get("acceptance_passed") is True and accepted.get("component_transaction_passed") is True
        and not accepted.get("errors"),"原生事务/恢复报告未通过")
    require(Path(aligned.get("attempt","")).resolve()==attempt and Path(accepted.get("attempt","")).resolve()==attempt,"报告不属于当前attempt")
    require(aligned.get("MRQ_sync_verified") is False and accepted.get("MRQ_sync_verified") is False,"诊断报告不得升级正式同步")
    require(hashlib.sha256((attempt/"plan.json").read_bytes()).hexdigest()==state["plan_sha256"],"冻结plan SHA不一致")
    receipts,manifests,hashes = {},{},{}
    for target in plan["targets"]:
        entity = target["entity_id"]
        raw = guard(attempt/(entity+"_landmark_snapshot.json")).read_bytes()
        manifests[entity],hashes[entity] = json.loads(raw),hashlib.sha256(raw).hexdigest()
    for camera in plan["cameras"]:
        camera_id = camera["camera_id"]
        receipts[camera_id] = normalize_native_receipt(read_json(attempt/(camera_id+"_native_pose_restore_receipt.json")))
    checked = validate_joint_bundle(plan,state,read_json(attempt/"preflight.json"),read_json(attempt/"editor_after.json"),manifests,hashes,receipts)
    require(checked["acceptance_passed"] and not checked["errors"],"原始native/manifest/恢复当前校验失败: "+repr(checked["errors"]))
    require(set(aligned["cameras"])=={c["camera_id"] for c in plan["cameras"]},"配对机位集合不同")
    for camera_id,receipt in receipts.items():
        paired_rows = aligned["cameras"][camera_id]["frames"]
        lookup = {(r["output_frame"],r["entity_id"]):r for r in paired_rows}
        require(len(lookup)==len(paired_rows)==plan["frame_count"]*len(plan["targets"]),"已验报告的实体帧重复或缺失")
        for row in receipt["records"]:
            paired = lookup[(row["output_frame"],row["entity_id"])]
            require(paired["source_frame"]==row["sequence_source_frame"]
                and [point["world_m"] for point in paired["keypoints"]]==row["world_m"],"原始native世界点/源帧与已验报告不同")
    return attempt,plan,receipts,aligned,hashes


def export_attempt(attempt,alignment,native_report,output,copy_rgb=False,workers=1):
    """全新Saved诊断目录；实际世界点逐相机保留，不平均、不容差去重。"""
    output = guard(output)
    require(type(workers) is int and 1<=workers<=6,"CPU workers必须为1..6整数")
    require(output.name.startswith("diagnostic_") and not output.exists(),"输出须为全新diagnostic_目录")
    attempt,plan,receipts,aligned,hashes = load_inputs(attempt,alignment,native_report)
    require(attempt not in output.parents,"输出不能写入源渲染attempt内部")
    source = plan["source_motion"]
    frame_count = plan["frame_count"]
    seed,fps = source["root_seed"],source["fps"]
    require(type(seed) is int and seed>=0 and type(fps) in (float,int) and math.isfinite(fps) and fps>0,"真实GRF来源seed/FPS无效")
    require(source["sequence_frame_equals_source_step"] is True,"本适配器不猜测Sequence/source重映射")
    group_rel = "groups/seed_%08d"%seed
    group = output/group_rel
    mapping = {target["entity_id"]:"Player_"+target["entity_id"] for target in plan["targets"]}
    index = {}
    for camera in plan["cameras"]:
        camera_id = camera["camera_id"]
        rgb,exr = media_index(attempt/camera_id,"rgb"),media_index(attempt/camera_id,"exr")
        require(set(rgb)==set(exr)==set(range(frame_count)),"实际RGB/EXR相对帧集合不完整或越界")
        for path in list(rgb.values())+list(exr.values()): guard(path)
        rows = aligned["cameras"][camera_id]["frames"]
        require(len(rows)==frame_count*len(mapping) and {(r["output_frame"],r["entity_id"]) for r in rows}=={(f,e) for f in range(frame_count) for e in mapping},"配对报告帧/实体不完整")
        index[camera_id] = (rgb,exr,{(r["output_frame"],r["entity_id"]):r for r in rows})
    output.mkdir(parents=True)
    write_json(output/"export_started.json",{"status":"started","diagnostic_only":True,"not_training":True})
    provenance = group/"provenance"
    provenance.mkdir(parents=True)
    for name in ("plan.json","state.json","preflight.json","editor_after.json"):
        shutil.copyfile(attempt/name,provenance/name)
    shutil.copyfile(guard(alignment),provenance/"alignment_report.json")
    shutil.copyfile(guard(native_report),provenance/"native_acceptance_report.json")
    landmark_refs = {}
    for entity,digest in hashes.items():
        relative = "provenance/landmarks/"+digest+".json"
        target = group/relative
        if not target.exists():
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(attempt/(entity+"_landmark_snapshot.json"),target)
        landmark_refs[entity] = relative
    source_ref = source.get("source_build_receipt")
    if source_ref:
        shutil.copyfile(guard(source_ref),provenance/"source_build_receipt.json")
    timeline = [{"frame_index":frame,"source_frame_index":plan["start_frame"]+frame,
        "sequence_frame_index":plan["start_frame"]+frame,"source_time_seconds":(plan["start_frame"]+frame)/fps} for frame in range(frame_count)]
    write_jsonl(group/"timeline.jsonl",timeline)
    cameras = {}
    for camera_id,receipt in receipts.items():
        rgb,exr,views = index[camera_id]
        camera_dir = group/"cameras"/camera_id
        camera_dir.mkdir(parents=True)
        native_ref = "provenance/"+camera_id+"_native.json"
        shutil.copyfile(attempt/(camera_id+"_native_pose_restore_receipt.json"),group/native_ref)
        worlds,images = [],[]
        fixed_calibration = None
        (camera_dir/"mask").mkdir()
        jobs = ((frame,exr[frame],rgb[frame],camera_dir/("mask/%06d.png"%frame),mapping)
                for frame in range(frame_count))
        for frame,width,height,geometry in _mask_results(jobs,workers):
            mask_relative = "mask/%06d.png"%frame
            rgb_relative = None
            if copy_rgb:
                (camera_dir/"rgb").mkdir(exist_ok=True)
                rgb_relative = "rgb/%06d.png"%frame
                shutil.copyfile(rgb[frame],camera_dir/rgb_relative)
            world_objects,image_objects = [],[]
            selected = [r for r in receipt["records"] if r["output_frame"]==frame]
            for row in sorted(selected,key=lambda r:r["entity_id"]):
                entity = row["entity_id"]
                track = int(entity[1])+(1 if entity[0]=="L" else 6)
                paired = views[(frame,entity)]
                require(Path(paired["RGB"]).resolve()==rgb[frame].resolve() and Path(paired["EXR"]).resolve()==exr[frame].resolve(),"配对报告媒体路径不同")
                camera = paired["camera"]
                require((camera["intrinsics"]["width"],camera["intrinsics"]["height"])==(width,height),"已验相机尺寸不同")
                points = project_landmarks([{"world_m":p,"source_method":row["sampling_method"]} for p in row["world_m"]],camera)
                calibration = {key:camera[key] for key in ("intrinsics","extrinsics")}
                if fixed_calibration is None: fixed_calibration = calibration
                require(calibration==fixed_calibration,"此最小导出仅接收实际静态相机标定，不静默丢弃变化")
                valid = [p.get("uv_px") is not None and p["visibility_state"]=="unknown" for p in points]
                bbox,area = geometry[entity]
                world_objects.append(dict(entity_id=entity,track_id=track,world_m=row["world_m"],world_location_valid=[True]*17,
                    source_method=row["sampling_method"],mesh_identity=row["mesh_identity"],landmark_manifest_ref=landmark_refs[entity],
                    native_record_ref=native_ref+"#output_frame=%d,entity_id=%s"%(frame,entity),
                    engine_frame=row["hook_engine_frame"],pose_revision=row["pose_revision_after"],
                    capture_world_time_seconds=row["world_time_seconds"],mrq_cached_world_seconds=row.get("m_rq_world_seconds")))
                image_objects.append(dict(entity_id=entity,track_id=track,mask_id=track,world_frame_index=frame,
                    world_ref="pose3d_world.jsonl",uv_px=[p.get("uv_px") for p in points],location_valid=valid,
                    camera_forward_depth_m=[p.get("camera_forward_depth_m") for p in points],
                    visibility_state=[p["visibility_state"] for p in points],visibility_method="not_measured",
                    bbox_visible_xywh=bbox,visible_mask_area=area))
            worlds.append(dict(frame_index=frame,source_frame_index=plan["start_frame"]+frame,sequence_frame_index=plan["start_frame"]+frame,objects=world_objects))
            images.append(dict(frame_index=frame,world_frame_index=frame,rgb_ref=rgb_relative,
                rgb_source_ref=None if copy_rgb else str(rgb[frame]),mask_ref=mask_relative,camera_calibration_ref="view.json",objects=image_objects))
        write_json(camera_dir/"view.json",fixed_calibration)
        write_jsonl(camera_dir/"pose3d_world.jsonl",worlds)
        write_jsonl(camera_dir/"pose2d.jsonl",images)
        cameras[camera_id] = {"view_ref":"cameras/"+camera_id+"/view.json","pose3d_world_ref":"cameras/"+camera_id+"/pose3d_world.jsonl","pose2d_ref":"cameras/"+camera_id+"/pose2d.jsonl"}
    group_index = dict(schema="futsalmot_pose17_per_camera_diagnostic_group_v1",seed=seed,seed_role="motion_source_only",
        source_scope=plan["source_scope"],diagnostic_only=True,not_training=True,production_accepted=False,MRQ_sync_verified=False,
        world_coordinates="ue_left_handed_x_forward_y_right_z_up_m",keypoint_names=list(COCO17_NAMES),
        world_storage="per_camera_native",cross_camera_world_shared=False,world_values_modified=False,
        timeline_ref="timeline.jsonl",timeline_time_base="GRF_source_frame_divided_by_source_fps",cameras=cameras,
        landmark_manifest_refs=landmark_refs,roster=[{k:t[k] for k in ("entity_id","calibration_player_id")} for t in plan["targets"]],
        source_motion=source,location_pilot=plan.get("location_pilot"),rgb_mode="copy" if copy_rgb else "external_Saved_reference",
        mask_semantics="0_background_or_nonplayer_1_to_10_players",bbox_semantics="visible_mask_tight_half_open_xywh")
    write_json(group/"group.json",group_index)
    image_count = frame_count*len(cameras)
    report = dict(schema="futsalmot_pose17_diagnostic_release_v1",status="diagnostic_export_complete_not_training",
        diagnostic_only=True,not_training=True,production_accepted=False,MRQ_sync_verified=False,world_storage="per_camera_native",
        group_ref=group_rel+"/group.json",rgb_frames=image_count,mask_frames=image_count,native_pose_records=image_count*len(mapping),
        exr_decode_count=image_count,rgb_files_copied=image_count if copy_rgb else 0,visibility_upgrade_performed=False)
    write_json(output/"dataset.json",report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("attempt",type=Path)
    parser.add_argument("alignment_report",type=Path)
    parser.add_argument("native_acceptance_report",type=Path)
    parser.add_argument("output",type=Path)
    parser.add_argument("--copy-rgb",action="store_true")
    parser.add_argument("--workers",type=int,choices=range(1,7),default=6,help="CPU EXR后处理进程数，默认6；最多6个在途帧")
    args = parser.parse_args()
    print(json.dumps(export_attempt(args.attempt,args.alignment_report,args.native_acceptance_report,args.output,args.copy_rgb,args.workers),ensure_ascii=False))


if __name__=="__main__": main()
