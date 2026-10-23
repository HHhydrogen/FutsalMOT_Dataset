"""120帧显式试产边界；纯CPU反例不代表生产验收。"""
import copy
import pytest

from pose17_location_pilot_contract import PILOT_WINDOWS, validate_location_pilot_contract
from pose17_smoke_runner import validate_plan
from test_pose17_joint_receipt import joint_plan


def pilot_plan(seed=1042):
    start, end, split = PILOT_WINDOWS[seed]
    plan = joint_plan()
    plan.update(start_frame=start, frame_count=120, review_frame_indices=[60],
                location_pilot_contract=True, source_scope="diagnostic_location_pilot_fixed_roster",
                location_pilot=dict(schema="futsalmot_pose17_location_pilot_v1", batch_id="location_pilot_4seed_a01",
                    seed=seed, split=split, source_frame_range=[start,end], location_supervision=True,
                    visibility_supervision=False, formal_pose_release=False),
                source_motion=dict(root_seed=seed, fps=10, context_source_frame_range=[0,end],
                    selected_source_frame_range=[start,end], sequence_frame_equals_source_step=True,
                    source_frames_sha256="a"*64, source_meta_sha256="b"*64))
    return plan


@pytest.mark.parametrize("seed", list(PILOT_WINDOWS))
def test_only_frozen_four_seed_windows_and_whole_group_split(seed):
    plan = pilot_plan(seed)
    before = copy.deepcopy(plan)
    assert validate_location_pilot_contract(plan)
    validate_plan(plan)
    assert plan == before


@pytest.mark.parametrize("case", ["optin", "seed", "split", "visibility", "formal", "count", "review", "window", "source", "sha", "scope"])
def test_no_implicit_expansion_or_visibility_training_admission(case):
    plan = pilot_plan()
    if case == "optin": plan["location_pilot_contract"] = 1
    elif case == "seed": plan["location_pilot"]["seed"] = True
    elif case == "split": plan["location_pilot"]["split"] = "test"
    elif case == "visibility": plan["location_pilot"]["visibility_supervision"] = True
    elif case == "formal": plan["location_pilot"]["formal_pose_release"] = True
    elif case == "count": plan["frame_count"] = 119
    elif case == "review": plan["review_frame_indices"] = [5]
    elif case == "window": plan["source_motion"]["selected_source_frame_range"] = [101,221]
    elif case == "source": plan["source_motion"]["sequence_frame_equals_source_step"] = False
    elif case == "sha": plan["source_motion"]["source_meta_sha256"] = "unknown"
    else: plan["diagnostic_scope"] = "pose17_gpu_render_cpu_lbs"
    with pytest.raises(ValueError): validate_plan(plan)


def test_old_joint_30_frame_cap_unchanged_without_explicit_contract():
    plan = joint_plan()
    assert not validate_location_pilot_contract(plan)
    plan["frame_count"] = 120
    with pytest.raises(ValueError): validate_plan(plan)
