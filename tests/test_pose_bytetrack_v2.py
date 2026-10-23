"""真实ByteTrack CPU状态机夹具：lambda0等价与预测缺点回退，不读真实数据。"""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from grf_ue_bridge.pose_bytetrack_v2 import FIXED_ARGS,pose_bytetrack_class,run_sequence,verify_lambda_zero

FIXED="a"*64
MODEL="b"*64


@pytest.fixture(autouse=True)
def runtime():
    pytest.importorskip("ultralytics")


def frames():
    output=[]
    for frame in range(10):
        detections=[]
        if frame not in (3,7):
            for index,x in enumerate((20.+frame*2,120.-frame)):
                score=.16 if frame==2 and index==0 else .9
                detections.append(dict(detection_id="toy:%d:%d"%(frame,index),bbox_xywh_px=[x,10.,40.,80.],score=score,class_id=0))
        if frame==5:detections.append(dict(detection_id="toy:5:new",bbox_xywh_px=[220.,10.,40.,80.],score=.8,class_id=0))
        output.append(dict(sequence_id="toy",camera_id="cam",frame_index=frame,width=320,height=180,detections=detections))
    return output


def pose(frame,detection,valid=True):
    x,y,w,h=detection["bbox_xywh_px"]
    return dict(source="model_prediction_xy",coordinate_space="image_pixels",validity_method="finite_bbox_relative_geometry",
        model_sha256=MODEL,fixed_detection_set_sha256=FIXED,sequence_id=frame["sequence_id"],camera_id=frame["camera_id"],
        frame_index=frame["frame_index"],detection_id=detection["detection_id"],bbox_xywh_px=detection["bbox_xywh_px"],
        prediction_valid=valid,xy_px=[[x+w*.5,y+h*.5]]*17 if valid else [[0.,0.]]*17,geometry_valid=[valid]*17)


def test_lambda_zero_exact_upstream_output_and_kalman_lifecycle():
    result=verify_lambda_zero(frames(),FIXED)
    assert result["passed"] and result["output_arrays_exact"] and result["track_states_exact"]
    assert result["audit"]["pose_pairs_used"]==0


def test_lambda_zero_never_reads_pose_even_if_gt_like_metadata_present():
    tracker=pose_bytetrack_class()(pose_lambda=0.,fixed_detection_set_sha256=FIXED)
    for frame in frames(): tracker.update_frame(frame,{"forbidden_GT":object()})
    assert tracker.audit["pose_pairs_used"]==0


@pytest.mark.parametrize("missing",["absent","all_invalid"])
def test_missing_pose_keeps_bbox_path_exact(missing):
    data=frames();attachments={}
    if missing=="all_invalid":
        attachments={f["frame_index"]:{d["detection_id"]:pose(f,d,False) for d in f["detections"]} for f in data}
    baseline,states,_=run_sequence(data,fixed_sha=FIXED,reference=True)
    result,candidate_states,audit=run_sequence(data,fixed_sha=FIXED,pose_lambda=.2,model_sha=MODEL,poses_by_frame=attachments)
    assert all(np.array_equal(a,b) for a,b in zip(baseline,result)) and states==candidate_states
    assert audit["pose_pairs_used"]==0 and audit["bbox_fallback_pairs"]>0


def test_prediction_pose_used_only_within_existing_bbox_gate():
    data=frames();attachments={f["frame_index"]:{d["detection_id"]:pose(f,d) for d in f["detections"]} for f in data}
    _,_,audit=run_sequence(data,fixed_sha=FIXED,pose_lambda=.2,model_sha=MODEL,poses_by_frame=attachments)
    assert audit["pose_pairs_used"]>0 and audit["bbox_gated_out_pairs"]>0
    assert audit["low_score_stage"]=="unchanged_upstream_bbox_only"


@pytest.mark.parametrize("damage",["GT","fake_visibility","identity","outside","prediction_failed"])
def test_reject_gt_masks_wrong_detection_or_invalid_geometry(damage):
    frame=frames()[0];detection=frame["detections"][0];record=deepcopy(pose(frame,detection))
    if damage=="GT":record["source"]="ground_truth"
    elif damage=="fake_visibility":record["validity_method"]="GT_visibility"
    elif damage=="identity":record["detection_id"]="other"
    elif damage=="outside":record["xy_px"][0]=[10000.,1.]
    else:record["prediction_valid"]=False
    tracker=pose_bytetrack_class()(pose_lambda=.2,fixed_detection_set_sha256=FIXED,model_sha256=MODEL)
    with pytest.raises(ValueError):tracker.update_frame(frame,{detection["detection_id"]:record})


def test_lambda_zero_cost_matrix_exact_and_outside_gate_not_resurrected():
    from ultralytics.trackers.byte_tracker import BYTETracker
    a=SimpleNamespace(angle=None,xyxy=np.array([0.,0.,10.,20.]),score=.9)
    b=SimpleNamespace(angle=None,xyxy=np.array([200.,0.,210.,20.]),score=.9)
    baseline=BYTETracker(SimpleNamespace(**FIXED_ARGS))
    expected=baseline.get_dists([a],[b])
    zero=pose_bytetrack_class()(pose_lambda=0.,fixed_detection_set_sha256=FIXED)
    assert np.array_equal(expected,zero.get_dists([a],[b]))
    candidate=pose_bytetrack_class()(pose_lambda=.9,fixed_detection_set_sha256=FIXED,model_sha256=MODEL)
    # 无需读取pose_record：基线不可行边必须原样保留。
    assert np.array_equal(expected,candidate.get_dists([a],[b]))
    assert candidate.audit["bbox_gated_out_pairs"]==1
