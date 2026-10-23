"""联合快速通路的纯CPU toy收据；不代表已渲染十名真实角色。"""

import copy

import pytest

from pose17_calibration_assets import JOINT_VISUAL_IDS, ORIGINAL_VISUAL_MESH, production_visual_mesh
from pose17_override_receipt import validate_joint_bundle
from pose17_smoke_runner import validate_plan
from test_pose17_override_receipt import gpu_bundle
from test_pose17_smoke_runner import gpu_override_plan


def joint_plan():
    data = gpu_override_plan()
    data.update(diagnostic_scope="pose17_gpu_render_cpu_lbs_joint_diagnostic",allow_joint_pose17_diagnostic=True,
        start_frame=100,frame_count=10,review_frame_indices=[5])
    data.pop("calibration_player_id")
    data["cameras"] = [dict(camera_id="CineCam_%02d"%i,source_sequence="/Game/Test/GRF_%d"%i,
        camera_actor_object_name="Camera_%d"%i) for i in range(1,5)]
    data["targets"] = []
    for entity,player_id in JOINT_VISUAL_IDS.items():
        target = dict(entity_id=entity,actor_object_name="Actor"+entity,component_object_name="MPFBVisual",
            calibration_player_id=player_id,sequence_binding_name="Player_"+entity,landmark_manifest="toy_%d.json"%player_id)
        if player_id!=3: target["diagnostic_visual_mesh_override"] = production_visual_mesh(player_id)
        data["targets"].append(target)
    return data


def joint_bundle():
    base = gpu_bundle()
    plan = joint_plan()
    state,before,after = copy.deepcopy(base[1:4])
    state["completed_jobs"] = [dict(job=c["camera_id"],sampling_passed=True,restore_passed=True,
        streaming_state_restart_required=False) for c in plan["cameras"]]
    before["players"] = []
    manifests,hashes,receipts = {},{},{}
    for target in plan["targets"]:
        entity = target["entity_id"]
        identity = dict(base[4]["ue_identity"],asset_path=production_visual_mesh(target["calibration_player_id"]))
        manifests[entity] = {"ue_identity":identity}
        hashes[entity] = "c"*64
        actor = copy.deepcopy(base[2]["players"][0])
        actor["actor_name"] = target["actor_object_name"]
        actor["meshes"][0]["asset"] = ORIGINAL_VISUAL_MESH
        before["players"].append(actor)
    after = copy.deepcopy(before)
    template = base[-1]["CineCam_01"]
    for camera in plan["cameras"]:
        receipt = copy.deepcopy(template)
        receipt.update(diagnostic_scope=plan["diagnostic_scope"],records=[],visual_mesh_overrides=[])
        for target in plan["targets"]:
            entity = target["entity_id"]
            component = "/Game/FutsalMOT/Maps/UEDPIE_0_L_FutsalCourt.L_FutsalCourt:PersistentLevel."+target["actor_object_name"]+".MPFBVisual"
            if target["calibration_player_id"]!=3:
                evidence = copy.deepcopy(template["visual_mesh_overrides"][0])
                evidence.update(component_path=component,original_animation_instance_path=component+".AnimInstance_0",
                    applied_asset_path=manifests[entity]["ue_identity"]["asset_path"])
                receipt["visual_mesh_overrides"].append(evidence)
            for frame in range(10):
                row = copy.deepcopy(template["records"][0])
                row.update(output_frame=frame,sequence_source_frame=100+frame,entity_id=entity,component_path=component,
                    camera_name=camera["camera_id"],mesh_identity=manifests[entity]["ue_identity"],hook_engine_frame=110+frame,
                    engine_frame_before=110+frame,engine_frame_after=110+frame,gpu_lbs_warmup_engine_frames_observed=9+frame,
                    override_warmup_engine_frames_observed=10+frame)
                receipt["records"].append(row)
        receipts[camera["camera_id"]] = receipt
    return [plan,state,before,after,manifests,hashes,receipts]


def test_joint_four_cameras_ten_entities_reuses_existing_native_contract():
    data = joint_bundle()
    validate_plan(data[0])
    result = validate_joint_bundle(*data)
    assert result["acceptance_passed"] and result["records_per_camera"]==100
    assert result["visibility_state"]=="unknown" and not result["MRQ_sync_verified"]


@pytest.mark.parametrize("case",["optin","role","actor","mesh","camera","warmup"])
def test_joint_plan_rejects_incomplete_or_ambiguous_roster(case):
    plan = joint_plan()
    if case=="optin": plan["allow_joint_pose17_diagnostic"] = False
    if case=="role": plan["targets"][1]["calibration_player_id"] = 5
    if case=="actor": plan["targets"][1]["actor_object_name"] = plan["targets"][0]["actor_object_name"]
    if case=="mesh": plan["targets"][1]["diagnostic_visual_mesh_override"] = production_visual_mesh(4).replace("/Mesh/","/")
    if case=="camera": plan["cameras"].pop()
    if case=="warmup": plan["override_warmup_frames"] = 9
    with pytest.raises(ValueError): validate_plan(plan)


@pytest.mark.parametrize("case",["late_restore_failure","identity","missing_frame","component"])
def test_joint_receipt_rejects_partial_result_without_per_role_rerender(case):
    data = joint_bundle()
    receipt = data[-1]["CineCam_04"]
    if case=="late_restore_failure": receipt["visual_mesh_overrides"][-1]["restore_succeeded"] = False
    if case=="identity": receipt["records"][-1]["manifest_sha256"] = "d"*64
    if case=="missing_frame": receipt["records"].pop()
    if case=="component": receipt["visual_mesh_overrides"][-1]["component_path"] = receipt["visual_mesh_overrides"][0]["component_path"]
    assert not validate_joint_bundle(*data)["acceptance_passed"]
