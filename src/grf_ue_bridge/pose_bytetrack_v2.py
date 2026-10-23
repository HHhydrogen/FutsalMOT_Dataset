"""固定原始检测的最小Pose ByteTrack候选；lambda0严格走原bbox数值路径。"""
from copy import deepcopy
import math
import re
from types import SimpleNamespace

import numpy as np

from .pose_association_contract_v2 import _validate,pose_pair_cost_candidate

VERSION="fixed_detection_bytetrack_pose_cost_candidate_v1"
FIXED_ARGS=dict(track_high_thresh=.25,track_low_thresh=.1,new_track_thresh=.25,
                track_buffer=30,match_thresh=.8,fuse_score=True)
_TRACKER=None


def require(ok,message):
    if not ok:raise ValueError(message)


def numeric_boxes(frame):
    """只重建同一固定float32检测输入，不读图片、不推理、不替换框或分数。"""
    from ultralytics.engine.results import Boxes
    values=[]
    for detection in frame["detections"]:
        x,y,w,h=detection["bbox_xywh_px"]
        require(w>0 and h>0 and all(math.isfinite(v) for v in (x,y,w,h)),"固定bbox无效")
        require(detection["class_id"]==0 and math.isfinite(detection["score"]),"只允许固定player检测")
        values.append([x,y,x+w,y+h,detection["score"],0.])
    return Boxes(np.asarray(values,dtype=np.float32).reshape(-1,6),(frame["height"],frame["width"]))


def validate_attachment(record,detection,frame,fixed_sha,bbox_margin_fraction):
    _validate(record)
    require(record["fixed_detection_set_sha256"]==fixed_sha
        and re.fullmatch("[0-9a-f]{64}",record["model_sha256"]) is not None,"附件模型/检测SHA错误")
    require(record.get("detection_id")==detection["detection_id"]
        and record["bbox_xywh_px"]==detection["bbox_xywh_px"]
        and record["sequence_id"]==frame["sequence_id"] and record["camera_id"]==frame["camera_id"]
        and record["frame_index"]==frame["frame_index"],"附件没有精确绑定原始固定检测")
    require(type(record.get("prediction_valid")) is bool,"缺明确模型附件是否成功")
    require(record["prediction_valid"] or not any(record["geometry_valid"]),"附着失败不能有有效位置")
    x,y,w,h=detection["bbox_xywh_px"];margin=bbox_margin_fraction
    for (u,v),valid in zip(record["xy_px"],record["geometry_valid"]):
        if valid:
            require(record["prediction_valid"] and 0<=u<frame["width"] and 0<=v<frame["height"]
                and x-margin*w<=u<=x+(1+margin)*w and y-margin*h<=v<=y+(1+margin)*h,
                "有效预测点不符合冻结图像/相对bbox几何范围")


class _PoseBoxes:
    def __init__(self,boxes,poses):self.boxes=boxes;self.poses=poses
    def __len__(self):return len(self.boxes)
    def __getitem__(self,index):
        indices=np.arange(len(self))[index]
        return _PoseBoxes(self.boxes[index],[self.poses[int(i)] for i in np.atleast_1d(indices)])
    @property
    def conf(self):return self.boxes.conf
    @property
    def cls(self):return self.boxes.cls
    @property
    def xywh(self):return self.boxes.xywh


def pose_bytetrack_class():
    """只覆盖初始化元数据和既有两次get_dists；不复制上游update状态机。"""
    global _TRACKER
    if _TRACKER is not None:return _TRACKER
    import ultralytics
    from ultralytics.trackers.byte_tracker import BYTETracker,STrack
    require(ultralytics.__version__=="8.4.60","仅核验8.4.60 ByteTrack调用顺序")

    class PoseSTrack(STrack):
        def __init__(self,*args,pose=None,**kwargs):
            super().__init__(*args,**kwargs);self.pose_record=deepcopy(pose)
        def update(self,new_track,frame_id):
            super().update(new_track,frame_id);self.pose_record=deepcopy(new_track.pose_record)
        def re_activate(self,new_track,frame_id,new_id=False):
            super().re_activate(new_track,frame_id,new_id);self.pose_record=deepcopy(new_track.pose_record)

    class PoseBYTETracker(BYTETracker):
        def __init__(self,*,pose_lambda=0.,fixed_detection_set_sha256,model_sha256=None,bbox_margin_fraction=.25,
                     joint_indices=tuple(range(5,17)),min_joints=6,sigma=.1,max_frame_gap=30):
            require(type(pose_lambda) in (int,float) and math.isfinite(pose_lambda) and 0<=pose_lambda<=1,"lambda必须0..1")
            require(type(bbox_margin_fraction) in (int,float) and math.isfinite(bbox_margin_fraction) and 0<=bbox_margin_fraction<=1,"bbox margin无效")
            require(type(fixed_detection_set_sha256) is str and re.fullmatch("[0-9a-f]{64}",fixed_detection_set_sha256),"固定检测集SHA无效")
            if pose_lambda:
                require(isinstance(model_sha256,str) and re.fullmatch("[0-9a-f]{64}",model_sha256),"lambda>0需统一真实模型SHA")
            self.pose_lambda=pose_lambda;self.fixed_sha=fixed_detection_set_sha256;self.model_sha=model_sha256
            self.margin=bbox_margin_fraction;self.pose_parameters=dict(joint_indices=tuple(joint_indices),min_joints=min_joints,sigma=sigma,max_frame_gap=max_frame_gap)
            self.sequence_id=self.camera_id=None;self.previous_frame=-1;self._distance_stage=0
            self.audit=dict(pose_pairs_used=0,bbox_fallback_pairs=0,bbox_gated_out_pairs=0,low_score_stage="unchanged_upstream_bbox_only")
            super().__init__(SimpleNamespace(**FIXED_ARGS))

        def update_frame(self,frame,poses=None):
            require(type(frame.get("frame_index")) is int and frame["frame_index"]==self.previous_frame+1,"必须逐帧完整输入，包括空帧")
            if self.sequence_id is None:self.sequence_id,self.camera_id=frame["sequence_id"],frame["camera_id"]
            require((self.sequence_id,self.camera_id)==(frame["sequence_id"],frame["camera_id"]),"每个序列/相机须独立重置Tracker")
            boxes=numeric_boxes(frame);self._distance_stage=0
            if self.pose_lambda==0:
                # 不读取附件，不做混合算术，也不换STrack类型。
                result=super().update(boxes)
            else:
                poses=poses or {};ids=[d["detection_id"] for d in frame["detections"]]
                require(len(set(ids))==len(ids) and set(poses).issubset(ids),"附件ID不属于当前固定检测")
                ordered=[]
                for detection in frame["detections"]:
                    record=poses.get(detection["detection_id"])
                    if record is not None:
                        validate_attachment(record,detection,frame,self.fixed_sha,self.margin)
                        require(record["model_sha256"]==self.model_sha,"不能在一条轨迹内混用Pose模型")
                    ordered.append(record)
                result=super().update(_PoseBoxes(boxes,ordered))
            require(self._distance_stage==2,"上游get_dists调用顺序变化，停止而不是猜测阶段")
            self.previous_frame=frame["frame_index"]
            return result

        def init_track(self,results,img=None):
            if self.pose_lambda==0:return super().init_track(results,img)
            if not len(results):return []
            bboxes=np.concatenate([results.xywh,np.arange(len(results)).reshape(-1,1)],axis=-1)
            return [PoseSTrack(box,score,cls,pose=pose) for box,score,cls,pose in zip(bboxes,results.conf,results.cls,results.poses)]

        def get_dists(self,tracks,detections):
            stage=self._distance_stage;self._distance_stage+=1
            require(stage in (0,1),"未知ByteTrack关联阶段")
            base=super().get_dists(tracks,detections)
            if self.pose_lambda==0 or not base.size:return base
            threshold=self.args.match_thresh if stage==0 else .7
            result=base.copy()
            for i,track in enumerate(tracks):
                for j,detection in enumerate(detections):
                    if base[i,j]>threshold:
                        self.audit["bbox_gated_out_pairs"]+=1;continue
                    a,b=track.pose_record,detection.pose_record
                    gap=b["frame_index"]-a["frame_index"] if a is not None and b is not None else None
                    if a is None or b is None or not 1<=gap<=self.pose_parameters["max_frame_gap"]:
                        self.audit["bbox_fallback_pairs"]+=1;continue
                    box=a["bbox_xywh_px"];center=[box[0]+box[2]/2,box[1]+box[3]/2]
                    delta=[float(track.mean[k])-center[k] for k in range(2)]
                    cost=pose_pair_cost_candidate(a,b,delta,**self.pose_parameters)
                    if cost["pose_cost"] is None:
                        self.audit["bbox_fallback_pairs"]+=1;continue
                    result[i,j]=(1-self.pose_lambda)*base[i,j]+self.pose_lambda*cost["pose_cost"]
                    self.audit["pose_pairs_used"]+=1
            return result
    _TRACKER=PoseBYTETracker
    return _TRACKER


def tracker_state(tracker):
    """精确数值快照用于lambda0基线对照，不读Pose或GT。"""
    return {name:[dict(track_id=t.track_id,state=int(t.state),frame_id=t.frame_id,start_frame=t.start_frame,
        is_activated=t.is_activated,score=float(t.score),idx=float(t.idx),
        mean=None if t.mean is None else t.mean.tolist(),covariance=None if t.covariance is None else t.covariance.tolist())
        for t in getattr(tracker,name)] for name in ("tracked_stracks","lost_stracks","removed_stracks")}


def run_sequence(frames,*,fixed_sha,pose_lambda=0.,model_sha=None,poses_by_frame=None,reference=False,**parameters):
    from ultralytics.trackers.byte_tracker import BYTETracker
    tracker=BYTETracker(SimpleNamespace(**FIXED_ARGS)) if reference else pose_bytetrack_class()(
        pose_lambda=pose_lambda,fixed_detection_set_sha256=fixed_sha,model_sha256=model_sha,**parameters)
    outputs=[];states=[]
    for frame in frames:
        output=tracker.update(numeric_boxes(frame)) if reference else tracker.update_frame(frame,(poses_by_frame or {}).get(frame["frame_index"]))
        outputs.append(output.copy());states.append(tracker_state(tracker))
    return outputs,states,{} if reference else tracker.audit


def verify_lambda_zero(frames,fixed_sha,return_outputs=False):
    reference,reference_states,_=run_sequence(frames,fixed_sha=fixed_sha,reference=True)
    candidate,candidate_states,audit=run_sequence(frames,fixed_sha=fixed_sha,pose_lambda=0.)
    require(all(np.array_equal(a,b) for a,b in zip(reference,candidate)) and reference_states==candidate_states,
            "lambda0未逐值复现原ByteTrack轨迹/均值/协方差/生命周期")
    receipt=dict(passed=True,frames=len(frames),output_arrays_exact=True,track_states_exact=True,audit=audit)
    return (receipt,candidate) if return_outputs else receipt
