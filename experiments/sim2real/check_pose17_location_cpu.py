"""真实诊断导出→位置JSONL→单个CPU batch/loss检查；无优化器、GPU或训练准入。"""

import argparse
from copy import copy
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"))
from grf_ue_bridge.tools.export_pose17_diagnostic import guard, load_inputs
from grf_ue_bridge.pose17_export import project_landmarks
from grf_ue_bridge.pose_location_supervision_v2 import synthetic_person_location, collate_location_contract

SOURCE_SHA = "869e83fcdffdc7371fa4e34cd8e51c838cc729571d1635e5141e3075e9319dc0"


def require(ok,message):
    if not ok: raise ValueError(message)


def read(path):return json.loads(guard(path).read_text(encoding="utf-8"))
def lines(path):return [json.loads(line) for line in guard(path).read_text(encoding="utf-8").splitlines() if line]
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def adapt_records(release,acceptance):
    """读取实际证据内容及原native，不凭非空引用或diagnostic标志升级准入。"""
    release,acceptance = guard(release),guard(acceptance)
    dataset,receipt = read(release/"dataset.json"),read(acceptance)
    require(dataset.get("schema")=="futsalmot_pose17_diagnostic_release_v1" and dataset.get("status")=="diagnostic_export_complete_not_training"
        and dataset.get("diagnostic_only") is True and dataset.get("not_training") is True
        and dataset.get("production_accepted") is False and dataset.get("MRQ_sync_verified") is False,"源必须保持诊断/not_training原状态")
    require(receipt.get("schema")=="pose17_real_diagnostic_export_cpu_acceptance_v1" and receipt.get("passed") is True
        and not receipt.get("errors") and Path(receipt.get("dataset","")).resolve()==release
        and receipt.get("world_objects_exact")==400 and receipt.get("independent_bool_location_masks")==400,"真实导出CPU验收内容缺失/未通过")
    group_dir = guard(release/Path(dataset["group_ref"]).parent)
    group = read(group_dir/"group.json")
    require(group.get("world_storage")=="per_camera_native" and group.get("cross_camera_world_shared") is False
        and group.get("world_values_modified") is False and group.get("not_training") is True,"不能共享/改写原始世界点或训练状态")
    provenance = group_dir/"provenance"
    attempt,plan,native,alignment,hashes = load_inputs(receipt["source_attempt"],provenance/"alignment_report.json",provenance/"native_acceptance_report.json")
    require(group.get("source_motion")==plan["source_motion"] and group.get("source_scope")==plan["source_scope"]
        and group.get("seed")==plan["source_motion"]["root_seed"],"源运动provenance发生变化")
    frames=[];nonvisible=0
    for camera_id in sorted(group["cameras"]):
        camera_dir = group_dir/"cameras"/camera_id
        world_rows,image_rows,view = lines(camera_dir/"pose3d_world.jsonl"),lines(camera_dir/"pose2d.jsonl"),read(camera_dir/"view.json")
        require(len(world_rows)==len(image_rows)==plan["frame_count"],"完整相机帧数不同")
        native_index = {(r["output_frame"],r["entity_id"]):r for r in native[camera_id]["records"]}
        expected_entities = {t["entity_id"] for t in plan["targets"]}
        for index,(world,image) in enumerate(zip(world_rows,image_rows)):
            require(world["frame_index"]==image["frame_index"]==index and world["source_frame_index"]==world["sequence_frame_index"]==plan["start_frame"]+index,"相对帧/真实源帧错配")
            require(len(world["objects"])==len(image["objects"])==10
                and {o["entity_id"] for o in world["objects"]}=={o["entity_id"] for o in image["objects"]}==expected_entities,"有人物遗漏/重复")
            world_objects={o["entity_id"]:o for o in world["objects"]}
            width,height=view["intrinsics"]["width"],view["intrinsics"]["height"]
            instances=[];nonvisible_entities=[]
            for obj in image["objects"]:
                entity=obj["entity_id"];native_row=native_index[(index,entity)]
                expected_id=int(entity[1])+(1 if entity[0]=="L" else 6)
                require(obj["track_id"]==obj["mask_id"]==expected_id,"实体track/mask ID被改变")
                require(world_objects[entity]["world_m"]==native_row["world_m"] and world_objects[entity]["mesh_identity"]==native_row["mesh_identity"],"导出世界点/mesh与已验native不同")
                points=project_landmarks([{"world_m":p,"source_method":native_row["sampling_method"]} for p in native_row["world_m"]],view)
                expected_valid=[p.get("uv_px") is not None and p["visibility_state"]=="unknown" for p in points]
                require(len(obj["location_valid"])==17 and all(type(v) is bool for v in obj["location_valid"])
                    and obj["location_valid"]==expected_valid and obj["uv_px"]==[p.get("uv_px") for p in points]
                    and obj["visibility_state"]==[p["visibility_state"] for p in points]
                    and obj["visibility_method"]=="not_measured","位置有效性/UV/unknown与实际投影证据不同")
                bbox,area=obj["bbox_visible_xywh"],obj["visible_mask_area"]
                if area==0:
                    require(bbox is None,"不可见实例不能有可见bbox")
                    nonvisible_entities.append(entity);nonvisible+=1;continue
                require(type(area) is int and area>0 and isinstance(bbox,list) and len(bbox)==4
                    and all(type(v) in (int,float) and math.isfinite(v) for v in bbox),"可见人物bbox/area无效")
                x,y,w,h=bbox
                require(w>0 and h>0 and x>=0 and y>=0 and x+w<=width and y+h<=height and area<=w*h,"可见bbox/area不能裁剪或越界")
                annotation_id=len(frames)*10+obj["track_id"]
                target=synthetic_person_location(dict(id=annotation_id,bbox=bbox,xy_px=[p if p is not None else [0.,0.] for p in obj["uv_px"]],
                    location_valid=obj["location_valid"],visibility_state=obj["visibility_state"]),width,height)
                target.update(entity_id=entity,source_world_ref=str(camera_dir/"pose3d_world.jsonl"),source_native_record_ref=world_objects[entity]["native_record_ref"])
                instances.append(target)
            image_path=guard(camera_dir/image["rgb_ref"]) if image["rgb_ref"] is not None else guard(image["rgb_source_ref"])
            frame=dict(schema="separated_pose_location_frame_v1",frame_key="seed_%08d:%s:%06d"%(group["seed"],camera_id,index),
                image_file=str(image_path),width=width,height=height,bbox_coverage="all_noncrowd_person_instances",crowd_present=False,
                expected_person_annotation_ids=[o["annotation_id"] for o in instances],instances=instances,
                nonvisible_entities_retained_in_world=nonvisible_entities,source="synthetic_diagnostic_export_cpu_engineering_only",
                source_frame_index=world["source_frame_index"],source_dataset_ref=str(release/"dataset.json"),source_acceptance_ref=str(acceptance),
                source_alignment_ref=str(provenance/"alignment_report.json"),not_training=True,MRQ_sync_verified=False)
            collate_location_contract([frame]);frames.append(frame)
    require(len(frames)==40,"只接收当前40图工程检查")
    evidence={"dataset_json_sha256":sha(release/"dataset.json"),"group_json_sha256":sha(group_dir/"group.json"),
        "acceptance_sha256":sha(acceptance),"alignment_report_sha256":sha(provenance/"alignment_report.json"),
        "native_acceptance_report_sha256":sha(provenance/"native_acceptance_report.json"),"plan_sha256":sha(attempt/"plan.json"),
        "manifest_sha256":hashes,"world_and_projection_rechecked":True,"omitted_nonvisible_bbox_entities":nonvisible,
        "source_not_training_preserved":True,"location_training_admission":False}
    return frames,evidence


def invalid_gradient_probe(torch,criterion):
    """独立反事实mask探针；不篡改真实JSONL，不冒称源数据存在缺点。"""
    from grf_ue_bridge.pose_location_supervision_v2 import pack_location_batch
    mask=torch.zeros((2,17),dtype=torch.bool);mask[0,0]=True
    packed=pack_location_batch(dict(keypoints=torch.full((2,17,2),.4),location_valid=mask))
    predictions=torch.full((1,2,17,2),.6,requires_grad=True)
    position,visibility=criterion.calculate_keypoints_loss(torch.ones((1,2),dtype=torch.bool),torch.tensor([[0,1]]),
        packed["keypoints"],torch.tensor([0,0]),torch.ones((2,1)),torch.tensor([[[0.,0.,1.,1.],[0.,0.,1.,1.]]]),predictions)
    position.backward()
    require(float(position)>0 and float(visibility)==0 and predictions.grad[0,0,0].abs().sum().item()>0
        and predictions.grad[0,0,1:].abs().sum().item()==0 and predictions.grad[0,1].abs().sum().item()==0,"独立无效位置梯度探针失败")
    return {"counterfactual_probe_not_source_GT":True,"valid_gradient_positive":True,"invalid_gradient_abs_sum":0.,"fully_missing_person_pose_gradient_abs_sum":0.,"visibility_loss":0.}


def check_batch(jsonl,checkpoint,imgsz=128):
    """已发布Loader/严格xy映射/同一loss路径；单batch backward，无optimizer/Trainer训练。"""
    import torch
    import ultralytics
    from ultralytics.utils import DEFAULT_CFG
    from grf_ue_bridge.pose_location_dataset_v2 import LocationPoseDataset
    from grf_ue_bridge.pose_location_supervision_v2 import separated_location_loss_class
    from grf_ue_bridge.pose_xy_head_transfer import map_pose_xy,XY_ROWS
    require(ultralytics.__version__=="8.4.60" and sha(checkpoint)==SOURCE_SHA,"仅已核验8.4.60/官方本地Pose权重")
    torch.set_num_threads(1);torch.manual_seed(20261024)
    dataset=LocationPoseDataset(jsonl,imgsz,verify_image_sha256=True)
    batch=dataset.collate_fn([dataset[i] for i in range(len(dataset))])
    require(len(batch["target_ids"])==sum(len(r["instances"]) for r in dataset.records),"Loader遗漏可见bbox人物")
    require(batch["location_valid"].dtype==torch.bool and tuple(batch["keypoints"].shape[1:])==(17,2),"batch位置契约不同")
    require(all(state=="unknown" for states in batch["visibility_state"] for state in states if state not in ("outside","behind_camera","missing")),"源unknown被升级")
    checkpoint_data=torch.load(checkpoint,map_location="cpu",weights_only=False)
    model=map_pose_xy(checkpoint_data["model"].float().cpu()).cpu().train()
    model.requires_grad_(True);model.args=copy(DEFAULT_CFG)
    criterion=separated_location_loss_class()(model)
    criterion.bce_pose.register_forward_pre_hook(lambda *a:(_ for _ in ()).throw(AssertionError("禁止visibility BCE")))
    image=batch["img"].float()/255.
    predictions=model(image)
    loss,items=criterion(predictions,batch)
    require(torch.isfinite(loss).all().item() and items[2].item()==0.,"实际CPU batch loss非有限或有visibility监督")
    loss.sum().backward()
    pose_weights=[block[-1][0].weight for block in model.model[-1].cv4]
    visibility_gradient=sum(weight.grad[2::3].abs().sum().item() for weight in pose_weights)
    xy_gradient=sum(weight.grad[list(XY_ROWS)].abs().sum().item() for weight in pose_weights)
    require(visibility_gradient==0.,"未输出visibility行有梯度")
    require(items[1].item()>0 and xy_gradient>0,"真实batch未产生有效位置监督，不能冒称loss通路已实际执行")
    probe=invalid_gradient_probe(torch,criterion)
    return dict(images=len(dataset),visible_bbox_instances=len(batch["target_ids"]),image_batch_shape=list(image.shape),
        keypoint_shape=list(batch["keypoints"].shape),location_mask_dtype=str(batch["location_valid"].dtype),
        location_valid_true=int(batch["location_valid"].sum()),loss_items=items.tolist(),visibility_loss=items[2].item(),
        physical_visibility_row_gradient_abs_sum=visibility_gradient,invalid_gradient_probe=probe,
        xy_row_gradient_abs_sum=xy_gradient,
        physical_pose_output_channels=[block[-1][0].out_channels for block in model.model[-1].cv4],
        hidden_pose_channels=[block[-1][0].in_channels for block in model.model[-1].cv4],
        source_checkpoint_sha256=SOURCE_SHA,torch=torch.__version__,ultralytics=ultralytics.__version__,
        optimizer_updates=0,gpu_used=False,training_admission=False)


def run(release,acceptance,output,checkpoint,imgsz=128):
    output=guard(output)
    require(not output.exists() and output.name.startswith("diagnostic_"),"需要全新Saved diagnostic_输出")
    frames,evidence=adapt_records(release,acceptance)
    output.mkdir(parents=True)
    # 40图工程检查首次明确计算图像SHA，Loader按其已发布默认契约独立核验。
    for frame in frames:frame["image_sha256"]=sha(frame["image_file"])
    jsonl=output/"location_cpu_check.jsonl"
    jsonl.write_text("".join(json.dumps(frame,ensure_ascii=False,allow_nan=False)+"\n" for frame in frames),encoding="utf-8")
    result=check_batch(jsonl,checkpoint,imgsz)
    receipt=dict(schema="pose17_real_location_jsonl_cpu_engineering_check_v1",passed=True,not_training=True,
        source_not_training_preserved=True,MRQ_sync_verified=False,real_test_read=False,gpu_used=False,
        jsonl=str(jsonl),source_evidence=evidence,cpu_batch=result,optimizer_updates=0,AP_evaluated=False,
        note="仅真实诊断数据到已发布Loader/严格xy loss的CPU工程检查，不运行toy Trainer或升级研究训练准入")
    (output/"receipt.json").write_text(json.dumps(receipt,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release",type=Path,required=True)
    parser.add_argument("--acceptance",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--checkpoint",type=Path,default=Path("D:/projects/FutsalMOT_Research/weights/yolo11n-pose.pt"))
    parser.add_argument("--imgsz",type=int,default=128)
    args=parser.parse_args()
    require(args.imgsz in (128,256),"CPU工程检查只允许128/256，不自动扩大计算")
    os.environ.update(CUDA_VISIBLE_DEVICES="-1",YOLO_AUTOINSTALL="false",YOLO_OFFLINE="true",WANDB_MODE="disabled",COMET_MODE="DISABLED")
    socket.socket.connect=lambda *a,**k:(_ for _ in ()).throw(RuntimeError("CPU工程检查禁止网络"))
    result=run(args.release,args.acceptance,args.output,args.checkpoint,args.imgsz)
    print(json.dumps({"passed":result["passed"],"receipt":str(args.output/"receipt.json"),"not_training":True},ensure_ascii=False))


if __name__=="__main__":main()
