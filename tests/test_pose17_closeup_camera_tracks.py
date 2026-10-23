"""近景诊断的无UE数学、命名通道与变更边界。"""

import copy
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest

M = runpy.run_path(str(Path(__file__).resolve().parents[1] / "ue/pose17_closeup_camera_tracks.py"))


@pytest.mark.parametrize("target,yaw,pitch", [([1, 0, 0], 0, 0), ([0, 1, 0], 90, 0),
                                            ([-1, 0, 0], 180, 0), ([1, 0, 1], 0, 45)])
def test_camera_axes(target, yaw, pitch):
    value = M["camera_values"]([0, 0, 0], target)
    assert value["Rotation.Z"] == pytest.approx(yaw)
    assert value["Rotation.Y"] == pytest.approx(pitch)
    assert value["Rotation.X"] == 0


@pytest.mark.parametrize("target", [[0, 0, 0], [float("nan"), 1, 1], [1, 2]])
def test_bad_camera_geometry(target):
    with pytest.raises(ValueError):
        M["camera_values"]([0, 0, 0], target)


def test_named_channels_only_no_positional_fallback():
    expected = [kind + "." + axis for kind in ("Location", "Rotation", "Scale") for axis in "XYZ"]
    section = SimpleNamespace(get_all_channels=lambda: [SimpleNamespace(channel_name=n + "_31") for n in expected])
    assert set(M["normalized_channels"](section)) == set(expected)
    wrong = SimpleNamespace(get_all_channels=lambda: [SimpleNamespace(channel_name="wrong" + str(i)) for i in range(9)])
    with pytest.raises(ValueError):
        M["normalized_channels"](wrong)


def test_only_camera_transform_excluded_from_integrity_hash():
    before = {"path": "/Game/Development/closeup/CineCam_01", "bindings": [
        {"name": "CineCam_01", "tracks": [{"class": "MovieScene3DTransformTrack", "value": 1}]},
        {"name": "Player_L0", "tracks": [{"class": "MovieScene3DTransformTrack", "value": 2}]}],
        "top_tracks": ["MovieSceneCameraCutTrack"]}
    after = copy.deepcopy(before)
    after["bindings"][0]["tracks"][0]["value"] = 999
    assert M["non_camera_signature"](before, "CineCam_01") == M["non_camera_signature"](after, "CineCam_01")
    after["bindings"][1]["tracks"][0]["value"] = 999
    assert M["non_camera_signature"](before, "CineCam_01") != M["non_camera_signature"](after, "CineCam_01")


def test_queue_assignment_enforces_exact_owned_target_without_conversion_api():
    job = SimpleNamespace(job_name="CineCam_01", sequence=None)
    path = "/Game/FutsalMOT/Development/Pose17/player003_closeup_a03/CineCam_01"
    factory = lambda p: {"constructed_exact_path": p}
    M["assign_owned_job_sequence"](job, path, path, factory)
    assert job.sequence == {"constructed_exact_path": path}


@pytest.mark.parametrize("wrong", ["/Game/FutsalMOT/Sequences/LS_Cam_01",
    "/Game/FutsalMOT/Development/Pose17/another_attempt/CineCam_01",
    "/Game/FutsalMOT/Development/Pose17/player003_closeup_a03/CineCam_02"])
def test_wrong_queue_assignment_rejected_before_mutation(wrong):
    job = SimpleNamespace(job_name="CineCam_01", sequence="untouched")
    expected = "/Game/FutsalMOT/Development/Pose17/player003_closeup_a03/CineCam_01"
    with pytest.raises(ValueError):
        M["assign_owned_job_sequence"](job, wrong, expected, lambda p: p)
    assert job.sequence == "untouched"


def test_explicit_float32_assignment_is_identical_for_float_and_double_channels():
    requested = -5.9468630539735
    assert M["assigned_key_value"]("MovieSceneScriptingFloatChannel", requested) == -5.946863174438477
    assert M["assigned_key_value"]("MovieSceneScriptingDoubleChannel", requested) == -5.946863174438477
    assert abs(M["assigned_key_value"]("MovieSceneScriptingDoubleChannel", requested) - requested) > 1e-7


def test_unknown_precision_rejected_instead_of_relaxing_tolerance():
    with pytest.raises(ValueError):
        M["assigned_key_value"]("UnverifiedChannel", 1.2345)


@pytest.mark.parametrize("damage", [None, "wrong_attempt", "wrong_plan", "saved_copy", "missing_camera", "success"])
def test_resume_evidence_scope(damage):
    plan = {"attempt": "player003_closeup_a03", "cameras": [{"camera_id": c} for c in ("a", "b", "c")]}
    record = {"attempt": plan["attempt"], "plan_path": "plan.json", "status": "failed_partial_copied_tracks_preserved",
              "cameras": [{"camera_id": c, "modified_and_saved": False} for c in ("a", "b", "c")]}
    if damage == "wrong_attempt":
        record["attempt"] = "another_attempt"
    elif damage == "wrong_plan":
        record["plan_path"] = "other.json"
    elif damage == "saved_copy":
        record["cameras"][0]["modified_and_saved"] = True
    elif damage == "missing_camera":
        record["cameras"].pop()
    elif damage == "success":
        record["status"] = "copied_camera_tracks_saved_not_rendered"
    if damage is None:
        M["validate_failed_partial"](record, plan, "plan.json")
    else:
        with pytest.raises(ValueError):
            M["validate_failed_partial"](record, plan, "plan.json")


def test_tracking_exception_only_allows_exact_property_on_target_camera():
    row = {"class": "MovieSceneBoolTrack", "property_name": "bEnableLookAtTracking",
           "property_path": "LookatTrackingSettings.bEnableLookAtTracking", "bool_sections": [{"keys": [False]}]}
    before = {"bindings": [{"name": "CineCam_01", "tracks": []}, {"name": "Player_L0", "tracks": []}]}
    after = copy.deepcopy(before)
    after["bindings"][0]["tracks"].append(row)
    assert M["non_camera_signature"](before, "CineCam_01", True) == M["non_camera_signature"](after, "CineCam_01", True)
    wrong = copy.deepcopy(after)
    wrong["bindings"][0]["tracks"][0]["property_path"] = "LookatTrackingSettings.bAllowRoll"
    assert M["non_camera_signature"](before, "CineCam_01", True) != M["non_camera_signature"](wrong, "CineCam_01", True)
    wrong = copy.deepcopy(before)
    wrong["bindings"][1]["tracks"].append(row)
    assert M["non_camera_signature"](before, "CineCam_01", True) != M["non_camera_signature"](wrong, "CineCam_01", True)


@pytest.mark.parametrize("pie,enabled,forward,passed", [(True, False, [1, 0, 0], True),
    (False, False, [1, 0, 0], False), (True, True, [1, 0, 0], False), (True, False, [-1, 0, 0], False)])
def test_runtime_gate_needs_actual_pie_actor_and_disabled_tracking(pie, enabled, forward, passed):
    settings = SimpleNamespace(get_editor_property=lambda n: enabled)
    actor = SimpleNamespace(get_path_name=lambda: "/Game/UEDPIE_0_Map.Camera" if pie else "/Game/Map.Camera",
        get_editor_property=lambda n: settings,
        get_actor_forward_vector=lambda: SimpleNamespace(x=forward[0], y=forward[1], z=forward[2]))
    if passed:
        assert M["read_runtime_tracking_gate"](actor, [1, 0, 0])["runtime_gate_passed"]
    else:
        with pytest.raises(ValueError):
            M["read_runtime_tracking_gate"](actor, [1, 0, 0])


def calibration_fixture(player=4):
    asset = M["production_player_asset"](player)
    plan = {"calibration_player_id": player, "attempt": f"player{player:03d}_closeup_a01",
        "allow_pie_visual_mesh_override": True, "diagnostic_scope": "pose17_character_calibration",
        "allow_pie_cpu_preparation": True, "allow_pie_driver_hide": True,
        "override_warmup_frames": 8, "disable_look_at_tracking_in_copies": True,
        "targets": [{"entity_id": "L0", "actor_object_name": M["L0_ACTOR"],
            "component_object_name": "MPFBVisual", "sequence_binding_name": "Player_L0",
            "landmark_manifest": f"mpfb_player{player:03d}_ue_surface_head_candidates.json",
            "diagnostic_visual_mesh_override": asset}], "cameras": []}
    for camera_id, (source, actor, location) in M["PRODUCTION_CAMERA_SCOPE"].items():
        plan["cameras"].append({"camera_id": camera_id, "source_sequence": source,
            "camera_actor_object_name": actor, "closeup_location_cm": location[:],
            "closeup_look_at_cm": [-1980., 6., 75.]})
    return plan, {"ue_identity": {"asset_path": asset}}


@pytest.mark.parametrize("player", list(range(3, 10)))
def test_production_role_scope_all_seven_models(player):
    plan, manifest = calibration_fixture(player)
    result = M["validate_calibration_scope"](plan, manifest)
    assert result["calibration_player_id"] == player


@pytest.mark.parametrize("damage", ["wrong_role", "wrong_asset", "wrong_manifest", "no_optin",
    "short_warmup", "bool_warmup", "wrong_actor", "driver_mesh", "wrong_camera", "tracking_enabled"])
def test_calibration_scope_rejects_unapproved_changes(damage):
    plan, manifest = calibration_fixture()
    if damage == "wrong_role":
        plan["calibration_player_id"] = 2
    elif damage == "wrong_asset":
        plan["targets"][0]["diagnostic_visual_mesh_override"] = M["production_player_asset"](5)
    elif damage == "wrong_manifest":
        manifest["ue_identity"]["asset_path"] = M["production_player_asset"](5)
    elif damage == "no_optin":
        plan["allow_pie_visual_mesh_override"] = False
    elif damage == "short_warmup":
        plan["override_warmup_frames"] = 7
    elif damage == "bool_warmup":
        plan["override_warmup_frames"] = True
    elif damage == "wrong_actor":
        plan["targets"][0]["actor_object_name"] = "SomeOtherActor"
    elif damage == "driver_mesh":
        plan["targets"][0]["component_object_name"] = "CharacterMesh0"
    elif damage == "wrong_camera":
        plan["cameras"][0]["closeup_location_cm"][0] += 1
    else:
        plan["disable_look_at_tracking_in_copies"] = False
    with pytest.raises(ValueError):
        M["validate_calibration_scope"](plan, manifest)


def gpu_lbs_calibration_fixture(player=4):
    plan, manifest = calibration_fixture(player)
    plan.update(attempt=f"player{player:03d}_closeup_a01_gpu_lbs", start_frame=0, frame_count=3,
                pose_sampling_path="gpu_render_cpu_lbs", allow_gpu_render_cpu_lbs_diagnostic=True,
                allow_pie_cpu_preparation=False, gpu_lbs_warmup_frames=8,
                diagnostic_scope="pose17_gpu_render_cpu_lbs_character_calibration")
    return plan, manifest


@pytest.mark.parametrize("player", list(range(4, 10)))
def test_gpu_lbs_override_scope_is_independent_and_exact_single_production_role(player):
    plan, manifest = gpu_lbs_calibration_fixture(player)
    result = M["validate_calibration_scope"](plan, manifest)
    assert result["pose_sampling_path"] == "gpu_render_cpu_lbs"
    assert result["calibration_player_id"] == player and result["override_is_pie_only"]
    assert result["expected_visual_mesh"] == M["production_player_asset"](player)


@pytest.mark.parametrize("field,value", [
    ("allow_pie_visual_mesh_override", False), ("allow_gpu_render_cpu_lbs_diagnostic", False),
    ("allow_gpu_render_cpu_lbs_diagnostic", 1), ("allow_pie_cpu_preparation", True),
    ("allow_pie_cpu_preparation", None), ("allow_pie_driver_hide", False),
    ("diagnostic_scope", "pose17_character_calibration"), ("diagnostic_scope", "pose17_gpu_render_cpu_lbs"),
    ("pose_sampling_path", "cpu_skin_cache"), ("pose_sampling_path", "unknown"),
    ("gpu_lbs_warmup_frames", 7), ("gpu_lbs_warmup_frames", True),
    ("gpu_lbs_warmup_frames", 8.0), ("gpu_lbs_warmup_frames", 121),
    ("gpu_lbs_warmup_frames", 9), ("override_warmup_frames", 9),
    ("override_warmup_frames", 7), ("override_warmup_frames", True),
    ("start_frame", 1), ("frame_count", 4),
])
def test_gpu_lbs_override_rejects_scope_optin_path_and_warmup_changes(field, value):
    plan, manifest = gpu_lbs_calibration_fixture()
    plan[field] = value
    with pytest.raises(ValueError):
        M["validate_calibration_scope"](plan, manifest)


def test_gpu_lbs_override_scope_does_not_include_source_player003_as_override():
    plan, manifest = calibration_fixture(3)
    plan.update(start_frame=0, frame_count=3, pose_sampling_path="gpu_render_cpu_lbs",
                allow_gpu_render_cpu_lbs_diagnostic=True, allow_pie_cpu_preparation=False,
                gpu_lbs_warmup_frames=8, diagnostic_scope="pose17_gpu_render_cpu_lbs_character_calibration")
    with pytest.raises(ValueError, match="仅允许004–009"):
        M["validate_calibration_scope"](plan, manifest)


@pytest.mark.parametrize("player", list(range(4, 10)))
@pytest.mark.parametrize("mismatch", ["asset", "manifest", "manifest_filename", "role_id"])
def test_gpu_lbs_each_role_rejects_other_model_identity(player, mismatch):
    plan, manifest = gpu_lbs_calibration_fixture(player)
    other = 4 if player != 4 else 5
    if mismatch == "asset":
        plan["targets"][0]["diagnostic_visual_mesh_override"] = M["production_player_asset"](other)
    elif mismatch == "manifest":
        manifest["ue_identity"]["asset_path"] = M["production_player_asset"](other)
    elif mismatch == "manifest_filename":
        plan["targets"][0]["landmark_manifest"] = f"mpfb_player{other:03d}_ue_surface_head_candidates.json"
    else:
        plan["calibration_player_id"] = other
    with pytest.raises(ValueError):
        M["validate_calibration_scope"](plan, manifest)


def test_gpu_lbs_override_rejects_second_target_and_wrong_role_identity():
    plan, manifest = gpu_lbs_calibration_fixture()
    plan["targets"].append(copy.deepcopy(plan["targets"][0]))
    with pytest.raises(ValueError, match="禁止批量"):
        M["validate_calibration_scope"](plan, manifest)
    plan, manifest = gpu_lbs_calibration_fixture()
    manifest["ue_identity"]["asset_path"] = M["production_player_asset"](5)
    with pytest.raises(ValueError, match="身份不一致"):
        M["validate_calibration_scope"](plan, manifest)
