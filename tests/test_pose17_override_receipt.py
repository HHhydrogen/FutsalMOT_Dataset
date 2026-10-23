"""独立Player004原生收据验收；反例不允许用角色名/CPU模式替代完整事务。"""

import copy

import pytest

from pose17_override_receipt import ANIM_CLASS, NAMES, NEW_MESH, OLD_MESH, SKELETON, validate_bundle


def bundle():
    actor = "ActualL0"
    component = "/Game/FutsalMOT/Maps/UEDPIE_0_L_FutsalCourt.L_FutsalCourt:PersistentLevel."+actor+".MPFBVisual"
    identity = {"asset_path":NEW_MESH,"signature_scheme":"ue_lod_le_u32_f32_v1","lod":0,"vertex_count":18623,
        "topology_sha256":"a"*64,"geometry_sha256":"b"*64}
    target = {"entity_id":"L0","actor_object_name":actor,"component_object_name":"MPFBVisual","diagnostic_visual_mesh_override":NEW_MESH}
    plan = {"allow_pie_visual_mesh_override":True,"diagnostic_scope":"pose17_character_calibration",
        "allow_pie_cpu_preparation":True,"allow_pie_driver_hide":True,"override_warmup_frames":8,
        "targets":[target],"cameras":[{"camera_id":"CineCam_01"},{"camera_id":"CineCam_02"}],"frame_count":3,"start_frame":0}
    mesh = {"component_name":"MPFBVisual","asset":OLD_MESH,"override_material_paths":["/Game/Original.Material"],"animation_class_path":ANIM_CLASS}
    editor = {"level":"/Game/FutsalMOT/Maps/L_FutsalCourt.L_FutsalCourt","dirty_packages":[],"pie_worlds":[],"mrq_running":False,
        "players":[{"actor_name":actor,"meshes":[mesh]}]}
    state = {"executor_success":True,"errors":[],"ready_to_render":False,"editor_scene_state_matched":True,
        "editor_mesh_state_matched":True,"editor_restore_probe_pending":False,"streaming_restart_required":False,
        "completed_jobs":[{"job":c["camera_id"],"sampling_passed":True,"restore_passed":True,"streaming_state_restart_required":False} for c in plan["cameras"]]}
    evidence = {"original_asset_path":OLD_MESH,"applied_asset_path":NEW_MESH,"shared_skeleton_path":SKELETON,
        "animation_class_path":ANIM_CLASS,"component_path":component,"original_animation_instance_path":component+".AnimInstance_0",
        "identity_verified":True,"apply_succeeded":True,"animation_binding_preserved":True,"new_asset_materials_verified":True,
        "restore_attempted":True,"restore_succeeded":True,"apply_error":"","restore_error":"","streaming_state_matched":True,
        "original_override_material_paths":["/Game/Original.Material"],"applied_asset_material_paths":["/Game/Player004.Material"],"applied_engine_frame":100}
    receipts = {}
    for camera in plan["cameras"]:
        rows = [{"output_frame":i,"sequence_source_frame":i,"entity_id":"L0","component_path":component,
            "camera_name":camera["camera_id"],"sample_succeeded":True,"error":"","keypoint_names":NAMES,
            "mesh_identity":dict(identity),"manifest_sha256":"c"*64,"cine_camera_look_at_state_observed":True,
            "cine_camera_look_at_enabled":False,"override_warmup_engine_frames_observed":10+i,"hook_engine_frame":110+i,
            "cache_fence_completed_this_engine_frame":True,"local_read_stable":True,"engine_frame_before":110+i,"engine_frame_after":110+i,
            "pose_revision_before":3+i,"pose_revision_after":3+i,"temporal_sample_count":1,"spatial_sample_count":1,
            "motion_blur_fraction":0,"mrq_synchronization_verified":False,"world_m":[[i,j,1] for j in range(17)],
            "actor_world_transform_ue_cm":{"translation":[i,0,0]},"visual_world_transform_ue_cm":{"translation":[i,0,0]}} for i in range(3)]
        receipts[camera["camera_id"]] = {"schema":"futsalmot_pose17_native_receipt_v1","lifecycle_stage":"pipeline_teardown_before_PIE_cleanup",
            "visual_mesh_override_requested":True,"diagnostic_scope":"pose17_character_calibration","setup_error":"","pipeline_setup_error":"",
            "restore_errors":[],"record_limit_reached":False,"camera_state_restored":True,"visual_mesh_overrides":[copy.deepcopy(evidence)],
            "streaming_state_restart_required":False,"records":rows}
    return [plan,state,editor,copy.deepcopy(editor),{"ue_identity":identity},"c"*64,receipts]


def test_complete_bundle_accepts_native_component_scope_only():
    result = validate_bundle(*bundle())
    assert result["acceptance_passed"] and result["component_transaction_passed"]
    assert result["editor_postflight_passed"]
    assert result["cross_view_world_max_delta_m"] == 0
    assert not result["MRQ_sync_verified"]
    assert result["visibility_state"] == "unknown"
    assert result["evidence_gaps"]


@pytest.mark.parametrize("field,value",[("original_asset_path","/Game/Not003.Not003"),("applied_asset_path",OLD_MESH),
    ("animation_binding_preserved",False),("new_asset_materials_verified",False),("restore_succeeded",False),
    ("animation_class_path","/Game/Fake.ABP_MPFB_RuntimeRetarget_C"),("applied_asset_material_paths",[])])
def test_reject_wrong_asset_animation_material_or_restore(field,value):
    data = bundle()
    data[-1]["CineCam_01"]["visual_mesh_overrides"][0][field] = value
    result = validate_bundle(*data)
    assert not result["acceptance_passed"] and result["errors"]


@pytest.mark.parametrize("field,value",[("cine_camera_look_at_enabled",True),("cine_camera_look_at_state_observed",False),
    ("override_warmup_engine_frames_observed",7),("sample_succeeded",False),("engine_frame_after",999),
    ("actor_world_transform_ue_cm",None),("manifest_sha256","d"*64)])
def test_every_record_requires_actual_runtime_evidence(field,value):
    data = bundle()
    data[-1]["CineCam_02"]["records"][2][field] = value
    result = validate_bundle(*data)
    assert not result["acceptance_passed"] and result["errors"]


def test_role_name_does_not_replace_complete_geometry_signature():
    data = bundle()
    data[-1]["CineCam_01"]["records"][0]["mesh_identity"]["topology_sha256"] = "d"*64
    assert not validate_bundle(*data)["acceptance_passed"]


def test_all_cameras_and_frames_and_editor_restoration_required():
    for case in ("camera","frame","editor"):
        data = bundle()
        if case=="camera": del data[-1]["CineCam_02"]
        if case=="frame": data[-1]["CineCam_02"]["records"].pop()
        if case=="editor": data[3]["players"][0]["meshes"][0]["asset"] = NEW_MESH
        assert not validate_bundle(*data)["acceptance_passed"]


def test_streaming_requires_honest_restart_without_faking_component_failure():
    data = bundle()
    data[-1]["CineCam_01"]["visual_mesh_overrides"][0]["streaming_state_matched"] = False
    assert not validate_bundle(*data)["component_transaction_passed"]
    data[-1]["CineCam_01"]["streaming_state_restart_required"] = True
    data[1]["completed_jobs"][0]["streaming_state_restart_required"] = True
    data[1]["streaming_restart_required"] = True
    result = validate_bundle(*data)
    assert result["component_transaction_passed"]
    assert result["streaming_restart_required"] and not result["acceptance_passed"]
    assert result["status"] == "native_component_switch_passed_streaming_restart_required"


def test_cross_view_changed_pose_requires_diagnosis():
    data = bundle()
    data[-1]["CineCam_02"]["records"][0]["world_m"][0][0] += .01
    result = validate_bundle(*data)
    assert not result["component_transaction_passed"]
    assert result["cross_view_world_max_delta_m"] == pytest.approx(.01)


def gpu_bundle():
    """纯CPU toy；并非实际Player004渲染收据。"""
    from test_pose17_smoke_runner import gpu_lbs_receipt
    data = bundle()
    data[0].update(pose_sampling_path="gpu_render_cpu_lbs",allow_gpu_render_cpu_lbs_diagnostic=True,
        diagnostic_scope="pose17_gpu_render_cpu_lbs_character_calibration",allow_pie_cpu_preparation=False,gpu_lbs_warmup_frames=8,
        calibration_player_id=4)
    toy = gpu_lbs_receipt()["records"][0]
    for receipt in data[-1].values():
        receipt.update(pose_sampling_path="gpu_render_cpu_lbs",gpu_lbs_diagnostic_requested=True,
            diagnostic_scope=data[0]["diagnostic_scope"],gpu_lbs_prepared_engine_frame=101)
        eligibility = copy.deepcopy(toy["gpu_lbs_eligibility"])
        eligibility.update(reference_vertex_count=18623,weight_vertex_count=18623)
        evidence = receipt["visual_mesh_overrides"][0]
        evidence.update(gpu_lbs_validation_attempted=True,gpu_lbs_validation_succeeded=True,gpu_lbs_validation_error="",
            gpu_lbs_eligibility_after_override=copy.deepcopy(eligibility))
        for row in receipt["records"]:
            row.update({key:copy.deepcopy(toy[key]) for key in ("render_cpu_enabled_at_read","render_static_at_read",
                "predicted_lod_at_read","sampling_method","pose_fence_method","end_frame_update_tasks_finished")})
            row.update(gpu_lbs_eligibility=copy.deepcopy(eligibility),gpu_lbs_warmup_engine_frames_observed=row["hook_engine_frame"]-101)
    return data


def test_combined_gpu_override_complete_transaction_stays_diagnostic():
    result = validate_bundle(*gpu_bundle())
    assert result["acceptance_passed"] and not result["MRQ_sync_verified"]
    assert result["visibility_state"] == "unknown"
    assert not any("sample未显式序列化CPU" in gap for gap in result["evidence_gaps"])


@pytest.mark.parametrize("field,value",[("gpu_lbs_validation_attempted",False),("gpu_lbs_validation_succeeded",False),
    ("gpu_lbs_validation_error","WPO rejected"),("gpu_lbs_eligibility_after_override",{})])
def test_combined_gpu_override_rejects_post_switch_partial_failure(field,value):
    data = gpu_bundle()
    data[-1]["CineCam_01"]["visual_mesh_overrides"][0][field] = value
    assert not validate_bundle(*data)["component_transaction_passed"]


@pytest.mark.parametrize("player_id",range(4,10))
def test_single_role_validator_requires_plan_manifest_actual_asset_agreement(player_id):
    from pose17_calibration_assets import calibration_visual_mesh
    data = gpu_bundle()
    asset = calibration_visual_mesh(player_id)
    data[0]["calibration_player_id"] = player_id
    data[0]["targets"][0]["diagnostic_visual_mesh_override"] = asset
    data[4]["ue_identity"]["asset_path"] = asset
    for receipt in data[-1].values():
        receipt["visual_mesh_overrides"][0]["applied_asset_path"] = asset
        for row in receipt["records"]: row["mesh_identity"]["asset_path"] = asset
    result = validate_bundle(*data)
    assert result["acceptance_passed"] and result["calibration_player_id"]==player_id
    assert result["expected_new_mesh"]==asset
    data[0]["calibration_player_id"] = 4 if player_id!=4 else 5
    assert not validate_bundle(*data)["acceptance_passed"]
