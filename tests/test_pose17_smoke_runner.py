"""只测试独立 smoke 计划约束，不执行 Unreal 或 MRQ。"""

import copy
import json
import sys
import types

import pytest

import pose17_smoke_runner as runner
from pose17_smoke_runner import _assert_prepare_gate, _assert_release_gate, validate_plan


def plan():
    return {"schema": "futsalmot_pose17_smoke_plan_v1", "attempt": "a01", "start_frame": 10,
            "frame_count": 3, "cameras": [{"camera_id": "CineCam_01", "source_sequence": "/Game/Test/First", "camera_actor_object_name": "CameraOne"},
                                         {"camera_id": "CineCam_02", "source_sequence": "/Game/Test/Second", "camera_actor_object_name": "CameraTwo"}],
            "targets": [{"actor_object_name": "ActualActor", "component_object_name": "MPFBVisual"}]}


def test_explicit_three_frame_two_camera_plan():
    validate_plan(plan())


@pytest.mark.parametrize("field,value", [("attempt", "../v1"), ("frame_count", 2), ("frame_count", 100),
                                        ("targets", []), ("start_frame", 1.5)])
def test_invalid_plan_rejected(field, value):
    data = plan()
    data[field] = value
    with pytest.raises(ValueError):
        validate_plan(data)


def test_media_path_escape_and_duplicate_views_rejected():
    data = plan()
    data["cameras"][0]["camera_id"] = "../../formal"
    with pytest.raises(ValueError):
        validate_plan(data)
    data = plan()
    data["cameras"][1] = copy.deepcopy(data["cameras"][0])
    with pytest.raises(ValueError):
        validate_plan(data)


@pytest.mark.parametrize("case", ["changes_selected_frames", "constant", "changes_outside_window", "wrong_actor"])
def test_dynamic_gate_checks_target_and_selected_window(case):
    data = plan()
    data["start_frame"] = 0
    data["targets"][0]["entity_id"] = "L0"
    values = [{"frame": 0, "value": 0}, {"frame": 1, "value": 1}]
    if case == "constant":
        values[1]["value"] = 0
    if case == "changes_outside_window":
        values = [{"frame": 10, "value": 0}, {"frame": 11, "value": 1}]
    report = {"level": "/Game/FutsalMOT/Maps/L_FutsalCourt.L_FutsalCourt", "dirty_packages": [], "pie_worlds": [],
              "mrq_running": False, "pose_library": True, "pose_pass": True, "filtered_png_output": True,
              "players": [{"actor_name": "ActualActor", "meshes": [{"component_name": "MPFBVisual", "visible": True, "hidden_in_game": False}]}],
              "source_sequences": [{"top_tracks": ["MovieSceneCameraCutTrack"], "playback_start": 0, "playback_end": 4,
                "bindings": [{"name": "OtherActor" if case == "wrong_actor" else "Player_L0", "tracks": [
                  {"class": "MovieScene3DTransformTrack", "numeric_channels": [{"name": "Location.X", "keys": values}]}]}]}]}
    if case == "changes_selected_frames":
        _assert_prepare_gate(data, report)
    else:
        with pytest.raises(RuntimeError):
            _assert_prepare_gate(data, report)


def finished_session(tmp_path):
    data = plan()
    data["targets"][0]["entity_id"] = "L0"
    state = {"status":"rendered_requires_RGB_mask_pose_alignment_acceptance", "executor_success":True,
        "errors":[],"ready_to_render":False,"editor_restore_probe_pending":False,
        "editor_scene_state_matched":True,"editor_mesh_state_matched":True,"streaming_restart_required":False,
        "completed_jobs":[{"job":c["camera_id"],"sampling_passed":True,"restore_passed":True} for c in data["cameras"]]}
    return {"attempt":tmp_path,"plan":data,"state":state}


def editor_report():
    return {"level":runner._LEVEL+".L_FutsalCourt","players":[],"mrq_running":False,"pie_worlds":[],"dirty_packages":[]}


@pytest.mark.parametrize("key,value", [("status","submitted_async"),("executor_success",False),
    ("editor_restore_probe_pending",True),("editor_mesh_state_matched",False),("streaming_restart_required",True),
    ("errors",["failure"]),("completed_jobs",[])])
def test_release_requires_finished_successful_postflight(tmp_path,key,value):
    session = finished_session(tmp_path)
    session["state"][key] = value
    with pytest.raises(RuntimeError):
        _assert_release_gate(session,editor_report(),False)


@pytest.mark.parametrize("key,value", [("mrq_running",True),("pie_worlds",["PIE"]),("dirty_packages",["/Game/UserEdit"]),
    ("level","/Game/Wrong.Wrong")])
def test_release_rechecks_live_editor(tmp_path,key,value):
    report = editor_report()
    report[key] = value
    with pytest.raises(RuntimeError):
        _assert_release_gate(finished_session(tmp_path),report,False)
    with pytest.raises(RuntimeError):
        _assert_release_gate(finished_session(tmp_path),editor_report(),True)


def test_release_only_detaches_owned_callback_and_keeps_files(tmp_path,monkeypatch):
    session = finished_session(tmp_path)
    owned, other = lambda:None, lambda:None
    class Delegate:
        bindings = [owned,other]
        def remove_callable(self,callback):
            self.bindings.remove(callback)
        def contains_callable(self,callback):
            return callback in self.bindings
    class Executor:
        on_executor_finished_delegate = Delegate()
        def is_rendering(self):
            return False
    executor = Executor()
    session.update(executor=executor,callbacks=[owned],queue=object(),settings=[object()])
    holder = types.ModuleType(runner._SESSION)
    holder.prepared = session
    monkeypatch.setitem(sys.modules,runner._SESSION,holder)
    monkeypatch.setattr(runner,"_PREPARED",session)
    monkeypatch.setattr(runner,"preflight",editor_report)
    (tmp_path/"state.json").write_text(json.dumps(session["state"]),encoding="utf-8")
    (tmp_path/"editor_after.json").write_text(json.dumps(editor_report()),encoding="utf-8")
    native = {"schema":"futsalmot_pose17_native_receipt_v1","lifecycle_stage":"pipeline_teardown_before_PIE_cleanup",
        "setup_error":"","record_limit_reached":False,"restore_errors":[],"camera_state_restored":True,
        "streaming_state_restart_required":False,"records":[{"outputFrame":i,"entityId":"L0","sampleSucceeded":True,
        "sample":{"worldPointsM":[[1,2,3]]*17,"keypointNames":[str(i) for i in range(17)],
        "mrqSynchronizationVerified":False}} for i in range(3)]}
    for camera in session["plan"]["cameras"]:
        (tmp_path/(camera["camera_id"]+"_native_pose_restore_receipt.json")).write_text(json.dumps(native),encoding="utf-8")
    media = tmp_path/"keep.exr"
    media.write_bytes(b"unchanged")
    result = runner.release()
    assert result["owned_callbacks_removed"] == 1
    assert executor.on_executor_finished_delegate.bindings == [other]
    assert holder.prepared is None and runner._PREPARED is None and session == {}
    assert media.read_bytes() == b"unchanged"
    assert json.loads((tmp_path/"state.json").read_text(encoding="utf-8"))["session_released"] is True
    assert (tmp_path/"session_release.json").exists()
    with pytest.raises(RuntimeError,match="没有待释放"):
        runner.release()


def override_plan():
    data = plan()
    data.update(allow_pie_visual_mesh_override=True,diagnostic_scope="pose17_character_calibration",
        allow_pie_cpu_preparation=True,allow_pie_driver_hide=True,override_warmup_frames=8)
    data["targets"][0].update(entity_id="L0",
        diagnostic_visual_mesh_override="/Game/FutsalMOT/Characters/MPFB/Players/Player_004/Mesh/MPFB_Player_004.MPFB_Player_004")
    return data


def test_override_requires_diagnostic_scope_optin_warmup_and_actual_visual():
    data = override_plan()
    validate_plan(data)
    for field,value in (("allow_pie_visual_mesh_override",False),("diagnostic_scope","production"),
        ("allow_pie_cpu_preparation",False),("allow_pie_driver_hide",False),("override_warmup_frames",7),
        ("override_warmup_frames",121),("override_warmup_frames",8.0)):
        changed = copy.deepcopy(data)
        changed[field] = value
        with pytest.raises(ValueError): validate_plan(changed)
    changed = copy.deepcopy(data)
    changed["targets"][0]["component_object_name"] = "CharacterMesh0"
    with pytest.raises(ValueError): validate_plan(changed)
    changed = copy.deepcopy(data)
    changed["targets"][0]["diagnostic_visual_mesh_override"] = "/Game/FutsalMOT/Characters/MPFB/Players/Player_001/MPFB_Player_001.MPFB_Player_001"
    with pytest.raises(ValueError): validate_plan(changed)


def override_receipt():
    data = override_plan()
    path = data["targets"][0]["diagnostic_visual_mesh_override"]
    return {"visual_mesh_override_requested":True,"diagnostic_scope":"pose17_character_calibration","pipeline_setup_error":"",
        "visual_mesh_overrides":[dict(applied_asset_path=path,identity_verified=True,apply_succeeded=True,
            animation_binding_preserved=True,new_asset_materials_verified=True,restore_attempted=True,restore_succeeded=True,
            apply_error="",restore_error="")],
        "records":[{"entity_id":"L0","mesh_identity":{"asset_path":path},"override_warmup_engine_frames_observed":10}]}


def test_override_receipt_cannot_reuse_003_identity_or_skip_material_animation_restore():
    data,receipt = override_plan(),override_receipt()
    runner._assert_override_receipt(data,receipt)
    for field in ("identity_verified","apply_succeeded","animation_binding_preserved","new_asset_materials_verified",
                  "restore_attempted","restore_succeeded"):
        changed = copy.deepcopy(receipt)
        changed["visual_mesh_overrides"][0][field] = False
        with pytest.raises(RuntimeError): runner._assert_override_receipt(data,changed)
    changed = copy.deepcopy(receipt)
    changed["records"][0]["mesh_identity"]["asset_path"] = "/Game/Player_003.Player_003"
    with pytest.raises(RuntimeError): runner._assert_override_receipt(data,changed)
    changed = copy.deepcopy(receipt)
    changed["records"][0]["override_warmup_engine_frames_observed"] = 7
    with pytest.raises(RuntimeError): runner._assert_override_receipt(data,changed)
    with pytest.raises(RuntimeError): runner._assert_override_receipt(plan(),receipt)


def gpu_lbs_plan():
    """纯CPU契约例；不代表任何UE捕获。"""
    data = plan()
    data.update(pose_sampling_path="gpu_render_cpu_lbs",allow_gpu_render_cpu_lbs_diagnostic=True,
        diagnostic_scope="pose17_gpu_render_cpu_lbs",allow_pie_cpu_preparation=False,
        allow_pie_driver_hide=True,gpu_lbs_warmup_frames=8)
    return data


def test_gpu_lbs_requires_explicit_separate_scope_and_warmup():
    data = gpu_lbs_plan()
    validate_plan(data)
    for key,value in (("pose_sampling_path","unknown"),("allow_gpu_render_cpu_lbs_diagnostic",False),
        ("diagnostic_scope","production"),("allow_pie_cpu_preparation",True),("allow_pie_driver_hide",False),
        ("allow_pie_visual_mesh_override",True),("gpu_lbs_warmup_frames",7),("gpu_lbs_warmup_frames",121),
        ("gpu_lbs_warmup_frames",8.0),("gpu_lbs_warmup_frames",True)):
        changed = copy.deepcopy(data)
        changed[key] = value
        with pytest.raises(ValueError): validate_plan(changed)
    data["targets"][0]["diagnostic_visual_mesh_override"] = "/Game/Other.Other"
    with pytest.raises(ValueError): validate_plan(data)


def gpu_lbs_receipt():
    """明确toy，检查native收据门禁失败路径。"""
    eligibility = dict.fromkeys(("checks_executed","eligible","pie_world","weights_readable"),True)
    eligibility.update(dict.fromkeys(("render_cpu_enabled","render_static","has_nanite_data","parallel_evaluation_running",
        "leader_parallel_evaluation_running","non_finite_morph_weights","lod0_has_cloth_data","variable_influence_weights"),False))
    eligibility.update(dict.fromkeys(("active_morph_weight_count","manual_morph_curve_count","external_morph_set_count",
        "cloth_simulation_count","lod0_cloth_section_count","mesh_deformer_asset_count","mesh_deformer_instance_count",
        "material_wpo_slot_count","null_material_slot_count"),0))
    eligibility.update(weight_vertex_count=12,reference_vertex_count=12,max_bone_influences=4)
    return {"pose_sampling_path":"gpu_render_cpu_lbs","gpu_lbs_diagnostic_requested":True,
        "diagnostic_scope":"pose17_gpu_render_cpu_lbs","pipeline_setup_error":"","gpu_lbs_prepared_engine_frame":100,
        "records":[dict(sample_succeeded=True,render_cpu_enabled_at_read=False,render_static_at_read=False,
            predicted_lod_at_read=0,sampling_method="gpu_render_component_pose_cpu_lbs_no_morph_cloth_deformer_wpo_diagnostic",
            pose_fence_method="gpu_dynamic_data_endframe_tasks_finished_render_commands_flush",
            end_frame_update_tasks_finished=True,gpu_lbs_warmup_engine_frames_observed=8,hook_engine_frame=108,
            mesh_identity={"vertex_count":12},gpu_lbs_eligibility=eligibility)]}


def test_gpu_lbs_receipt_rejects_cpu_render_missing_checks_or_unfenced_samples():
    receipt = gpu_lbs_receipt()
    runner._assert_gpu_lbs_receipt(gpu_lbs_plan(),receipt)
    for key,value in (("sample_succeeded",False),("render_cpu_enabled_at_read",True),("render_static_at_read",True),
        ("predicted_lod_at_read",1),("sampling_method","cached_cpu"),("pose_fence_method","none"),
        ("end_frame_update_tasks_finished",False),("gpu_lbs_warmup_engine_frames_observed",7),("hook_engine_frame",109)):
        changed = copy.deepcopy(receipt)
        changed["records"][0][key] = value
        with pytest.raises(RuntimeError): runner._assert_gpu_lbs_receipt(gpu_lbs_plan(),changed)
    for field,value in receipt["records"][0]["gpu_lbs_eligibility"].items():
        changed = copy.deepcopy(receipt)
        changed["records"][0]["gpu_lbs_eligibility"].pop(field)
        with pytest.raises(RuntimeError): runner._assert_gpu_lbs_receipt(gpu_lbs_plan(),changed)
        changed = copy.deepcopy(receipt)
        changed["records"][0]["gpu_lbs_eligibility"][field] = not value if type(value) is bool else value+1
        if field=="max_bone_influences": changed["records"][0]["gpu_lbs_eligibility"][field]=0
        with pytest.raises(RuntimeError): runner._assert_gpu_lbs_receipt(gpu_lbs_plan(),changed)
    with pytest.raises(RuntimeError): runner._assert_gpu_lbs_receipt(plan(),receipt)


def gpu_override_plan():
    data = gpu_lbs_plan()
    data.update(allow_pie_visual_mesh_override=True,diagnostic_scope="pose17_gpu_render_cpu_lbs_character_calibration",
        override_warmup_frames=8,calibration_player_id=4)
    data["targets"][0].update(entity_id="L0",component_object_name="MPFBVisual",
        diagnostic_visual_mesh_override=override_plan()["targets"][0]["diagnostic_visual_mesh_override"])
    return data


def test_gpu_override_only_single_004_dual_authorized_equal_warmup():
    data = gpu_override_plan()
    validate_plan(data)
    for field,value in (("allow_pie_visual_mesh_override",False),("allow_gpu_render_cpu_lbs_diagnostic",False),
        ("allow_pie_cpu_preparation",True),("diagnostic_scope","pose17_gpu_render_cpu_lbs"),
        ("override_warmup_frames",9),("override_warmup_frames",8.0),("pose_sampling_path","cpu_skin_cache")):
        changed = copy.deepcopy(data)
        changed[field] = value
        with pytest.raises(ValueError): validate_plan(changed)
    for target in ({},dict(diagnostic_visual_mesh_override="/Game/FutsalMOT/Characters/MPFB/Players/Player_005/MPFB_Player_005.MPFB_Player_005")):
        changed = copy.deepcopy(data)
        if target: changed["targets"][0].update(target)
        else: changed["targets"].append(copy.deepcopy(changed["targets"][0]))
        with pytest.raises(ValueError): validate_plan(changed)


def test_gpu_override_requires_post_swap_eligibility_and_real_frame_order():
    data = gpu_override_plan()
    receipt = gpu_lbs_receipt()
    receipt.update(visual_mesh_override_requested=True,diagnostic_scope=data["diagnostic_scope"],
        visual_mesh_overrides=override_receipt()["visual_mesh_overrides"])
    evidence = receipt["visual_mesh_overrides"][0]
    evidence.update(gpu_lbs_validation_attempted=True,gpu_lbs_validation_succeeded=True,gpu_lbs_validation_error="",
        gpu_lbs_eligibility_after_override=copy.deepcopy(receipt["records"][0]["gpu_lbs_eligibility"]),applied_engine_frame=99,
        original_asset_path=runner.ORIGINAL_VISUAL_MESH)
    row = receipt["records"][0]
    row.update(entity_id="L0",override_warmup_engine_frames_observed=9)
    row["mesh_identity"]["asset_path"] = data["targets"][0]["diagnostic_visual_mesh_override"]
    runner._assert_override_receipt(data,receipt)
    runner._assert_gpu_lbs_receipt(data,receipt)
    for field,value in (("gpu_lbs_validation_attempted",False),("gpu_lbs_validation_succeeded",False),
        ("gpu_lbs_validation_error","partial"),("gpu_lbs_eligibility_after_override",{}),("applied_engine_frame",101)):
        changed = copy.deepcopy(receipt)
        changed["visual_mesh_overrides"][0][field] = value
        with pytest.raises(RuntimeError): runner._assert_override_receipt(data,changed)
    changed = copy.deepcopy(receipt)
    changed["records"][0]["override_warmup_engine_frames_observed"] = 8
    with pytest.raises(RuntimeError): runner._assert_override_receipt(data,changed)


@pytest.mark.parametrize("player_id",range(4,10))
def test_single_gpu_calibration_id_binds_exact_asset_whitelist(player_id):
    from pose17_calibration_assets import calibration_visual_mesh
    data = gpu_override_plan()
    data["calibration_player_id"] = player_id
    data["targets"][0]["diagnostic_visual_mesh_override"] = calibration_visual_mesh(player_id)
    validate_plan(data)
    data["targets"][0]["diagnostic_visual_mesh_override"] = calibration_visual_mesh(player_id).replace("/Mesh/","/")
    with pytest.raises(ValueError): validate_plan(data)


@pytest.mark.parametrize("player_id",[None,True,4.0,"4",3,10])
def test_calibration_id_requires_explicit_supported_integer(player_id):
    data = gpu_override_plan()
    data["calibration_player_id"] = player_id
    with pytest.raises(ValueError): validate_plan(data)
