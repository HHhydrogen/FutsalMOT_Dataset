"""仅CPU消费固定dev检测/Pose附件，输出ByteTrack候选MOT；不推理、不读GT。"""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"));sys.path.insert(0,str(Path(__file__).resolve().parent))
from prepare_pose_association_dev import load_fixed_cache,output_path,plain
from grf_ue_bridge import pose_bytetrack_v2 as tracker
from grf_ue_bridge.fixed_detection_pose_attach_v2 import validate_attachment_frame,validate_configuration

SEQUENCES=("SNMOT-151","SNMOT-152","SNMOT-153")


def read(path):return json.loads(Path(path).read_text(encoding="utf-8"))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def checked(binding):
    path=plain(binding["path"])
    tracker.require(path.is_file() and sha(path)==binding["sha256"],"绑定文件SHA错误")
    return path


def run(cache_path,cache_sha,protocol_path,protocol_sha,output,pose_lambda=0.,attachment_path=None,attachment_sha=None,lambda0_receipt=None):
    manifest,frames=load_fixed_cache(cache_path,cache_sha)
    protocol=read(checked(dict(path=protocol_path,sha256=protocol_sha)))
    tracker.require(protocol.get("schema")=="futsalmot_pose_association_protocol_candidate_v1"
        and protocol["source_manifest"]==manifest["source_manifest"] and protocol.get("test_split_used") is False,"关联协议/来源范围不同")
    cost=protocol["pose_cost_candidate"]
    tracker.require(pose_lambda in cost["lambda_candidates"],"lambda不在预登记有限候选内")
    from ultralytics.utils import YAML
    from ultralytics.trackers.byte_tracker import BYTETracker
    import ultralytics
    import lap
    config=YAML.load(checked(protocol["tracking"]["tracker_config"]))
    tracker.require(config==dict(tracker_type="bytetrack",**tracker.FIXED_ARGS),"固定ByteTrack配置变化")
    identity=dict(fixed_detection_set_sha256=cache_sha,implementation_sha256=sha(tracker.__file__),
        upstream_bytetrack_sha256=sha(inspect.getfile(BYTETracker)),tracker_config_sha256=protocol["tracking"]["tracker_config"]["sha256"],
        ultralytics=ultralytics.__version__,lap=lap.__version__)
    attachment=None;poses={};model_sha=None;margin=.25
    if pose_lambda:
        tracker.require(attachment_path is not None and attachment_sha and lambda0_receipt,"lambda>0需实际附件及一次已通过lambda0收据")
        baseline=read(lambda0_receipt)
        tracker.require(baseline.get("schema")=="futsalmot_fixed_pose_association_candidate_v1"
            and baseline.get("identity")==identity and baseline.get("pose_lambda")==0
            and set(baseline.get("sequences",{}))==set(SEQUENCES)
            and all(row.get("frames")==150 and row.get("lambda0_equivalence",{}).get("passed") is True
                    for row in baseline["sequences"].values()),"lambda0基线证据未通过或实现/检测来源不同")
        attachment=read(checked(dict(path=attachment_path,sha256=attachment_sha)))
        tracker.require(attachment.get("schema")=="futsalmot_fixed_detection_pose_attachment_manifest_v1"
            and attachment.get("status")=="complete_pose_attachment_not_tracking"
            and attachment["fixed_detection_set_sha256"]==cache_sha
            and Path(attachment["fixed_detection_cache"]["path"]).resolve()==Path(cache_path).resolve()
            and attachment["fixed_detection_cache"]["sha256"]==cache_sha,"Pose附件不属于当前原始检测集")
        model_sha=attachment["checkpoint"]["sha256"]
        configuration=attachment["configuration"]
        attached_config=read(checked(configuration));validate_configuration(attached_config)
        tracker.require(attached_config==configuration["values"],"附件几何配置与实际文件不同")
        margin=attached_config["bbox_margin_fraction"]
        source=attachment["attachments_jsonl"]
        attached=[json.loads(line) for line in checked(source).read_text(encoding="utf-8").splitlines() if line]
        tracker.require(len(attached)==source["frames"]==len(frames)==450,"Pose与固定检测完整帧数不同")
        for frame,row in zip(frames,attached):
            validate_attachment_frame(frame,row,model_sha256=model_sha,fixed_detection_set_sha256=cache_sha,config=attached_config)
            poses.setdefault(frame["sequence_id"],{})[frame["frame_index"]]={p["detection_id"]:p for p in row["attachments"]}
    output=output_path(output);output.mkdir(parents=True);(output/"data").mkdir()
    report=dict(schema="futsalmot_fixed_pose_association_candidate_v1",status="candidate_cpu_tracking_complete_not_metric_evaluated",
        identity=identity,protocol_sha256=protocol_sha,pose_lambda=pose_lambda,pose_model_sha256=model_sha,
        attachment_manifest_sha256=attachment_sha if pose_lambda else None,sequences={},GT_read=False,test_read=False,
        physical_visibility_used=False,gpu_used=False,metrics_evaluated=False,
        limitations=["game9_development_same_match_not_independent_matches","no_AssA_HOTA_improvement_claim_before_TrackEval",
                      "lambda_and_quality_configuration_candidate_until_root_freeze"])
    for sequence in SEQUENCES:
        selected=[frame for frame in frames if frame["sequence_id"]==sequence]
        if not pose_lambda:
            equivalence,outputs=tracker.verify_lambda_zero(selected,cache_sha,return_outputs=True);audit=equivalence["audit"]
        else:
            outputs,_,audit=tracker.run_sequence(selected,fixed_sha=cache_sha,pose_lambda=pose_lambda,model_sha=model_sha,
                poses_by_frame=poses[sequence],bbox_margin_fraction=margin,joint_indices=cost["joint_indices"],
                min_joints=cost["min_common_geometry_joints"],sigma=cost["sigma"],max_frame_gap=cost["max_frame_gap"])
            equivalence=None
        lines=[]
        for frame,values in zip(selected,outputs):
            for row in values:
                x,y,r,b,track_id,score,_,_=map(float,row)
                lines.append(",".join([str(frame["frame_number"]),str(int(track_id))]+[format(v,".17g") for v in (x,y,r-x,b-y,score)]+["-1"]*3))
        (output/"data"/(sequence+".txt")).write_text("\n".join(lines)+("\n" if lines else ""),encoding="utf-8")
        report["sequences"][sequence]=dict(frames=len(selected),tracked_rows=len(lines),audit=audit,lambda0_equivalence=equivalence)
    with (output/"receipt.json").open("x",encoding="utf-8") as handle:json.dump(report,handle,ensure_ascii=False,indent=2,allow_nan=False)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ("cache-manifest","protocol","output"):parser.add_argument("--"+key,type=Path,required=True)
    for key in ("cache-sha256","protocol-sha256"):parser.add_argument("--"+key,required=True)
    parser.add_argument("--pose-lambda",type=float,default=0.)
    parser.add_argument("--attachment-manifest",type=Path)
    parser.add_argument("--attachment-sha256")
    parser.add_argument("--lambda0-receipt",type=Path)
    args=parser.parse_args();os.environ.update(CUDA_VISIBLE_DEVICES="-1",YOLO_AUTOINSTALL="false")
    # 与既有pipeline相同的本地LAP依赖，不安装或热修改共享训练环境。
    sys.path.append("D:/projects/FutsalMOT_Research/runtime_dependencies")
    print(json.dumps(run(args.cache_manifest,args.cache_sha256,args.protocol,args.protocol_sha256,args.output,args.pose_lambda,
        args.attachment_manifest,args.attachment_sha256,args.lambda0_receipt),ensure_ascii=False))


if __name__=="__main__":main()
