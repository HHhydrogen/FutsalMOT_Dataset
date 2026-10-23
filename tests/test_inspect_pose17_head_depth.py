"""头表面深度诊断反例；未知可见性不因残差变小自动升级。"""

import numpy as np
import pytest
import json

from grf_ue_bridge.tools.inspect_pose17_head_depth import TOLERANCE_M, pixel_observation, summarize


def inputs():
    token = 0x12345678
    depth = np.full((7,7),2.,dtype=np.float32)
    layers = [np.zeros((7,7,4),dtype=np.float32) for _ in range(3)]
    layers[0][:,:,0] = np.asarray(token,dtype=np.uint32).view(np.float32)
    layers[0][:,:,1] = 1
    return token,depth,layers


def test_exact_surface_match_remains_unknown():
    token,depth,layers = inputs()
    result = pixel_observation([3.1,3.2],2.002,depth,layers,token)
    assert result["candidate"] == "surface_agreement_candidate"
    assert result["visibility_state"] == "unknown"
    assert result["target_coverage"] == 1
    assert result["absolute_residual_m"] == pytest.approx(.002)
    assert result["target_interior_support"]
    assert result["tolerance_m"] == TOLERANCE_M == .005


@pytest.mark.parametrize("case,reason",[("mixed_id","instance_boundary"),("coverage","mixed_or_partial_coverage"),
    ("depth_step","depth_spread_exceeds_10mm_conservative_gate"),("invalid","invalid_depth")])
def test_boundaries_cannot_generate_candidate(case,reason):
    token,depth,layers = inputs()
    if case=="mixed_id": layers[0][2,2,0] = np.asarray(token+1,dtype=np.uint32).view(np.float32)
    if case=="coverage": layers[0][2,2,1] = .75
    if case=="depth_step": depth[2,2] += .02
    if case=="invalid": depth[2,2] = 0
    result = pixel_observation([3,3],2.,depth,layers,token)
    assert reason in result["unknown_reasons"]
    assert result["candidate"] == result["visibility_state"] == "unknown"


def test_image_edge_and_signed_self_occlusion_candidates():
    token,depth,layers = inputs()
    assert pixel_observation([0,0],2.,depth,layers,token)["candidate"] == "unknown"
    assert pixel_observation([3,3],2.03,depth,layers,token)["candidate"] == "nearer_first_hit_self_candidate"
    assert pixel_observation([3,3],1.97,depth,layers,token)["candidate"] == "point_in_front_of_rendered_surface_inconsistent"


def test_invalid_center_depth_and_coverage_serialize_as_unknown_not_nan():
    token,depth,layers = inputs()
    depth[3,3] = float("nan")
    layers[0][3,3,1] = float("nan")
    result = pixel_observation([3,3],2.,depth,layers,token)
    assert result["observed_forward_depth_m"] is None
    assert result["absolute_residual_m"] is None
    assert result["candidate"] == "unknown"
    json.dumps(result,allow_nan=False)


def test_cross_phase_summary_uses_common_support_not_low_error_dropout():
    rows = []
    for name in ("nose","left_eye"):
        for offset in (-1,0,1):
            rows.append({"image_frame":1,"entity_id":"L0","keypoint_name":name,"pose_frame_offset":offset,
                "target_instance_hit":True,"target_interior_support":not(name=="left_eye" and offset==1),
                "absolute_residual_m":.001 if offset==0 else .02,"unknown_reasons":[],"candidate":"unknown"})
    result = summarize(rows)
    assert result["common_triplet_target_interior"]["points"] == 1
    assert result["common_triplet_target_interior"]["offsets"]["0"]["within_5mm"] == 1
    assert result["1"]["target_interior_before_depth_spread_gate"]["count"] == 1
