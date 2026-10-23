"""CPU 配对工具的相机/身份/动态反例；不启动 UE 或 GPU。"""
import copy
import math
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from grf_ue_bridge.pose17_export import COCO17_NAMES, project_landmarks
from grf_ue_bridge.tools.inspect_pose17_smoke import (
    camera_from_exr_header, check_pair, check_expected_camera_plan, media_index, pose_motion, ue_rotation_basis,
    sampling_context, GPU_LBS_METHOD, GPU_LBS_FENCE, TRANSFORM_ENCODING, LIMB_BONES, inspect,
    GPU_LBS_SCOPE, GPU_LBS_CHARACTER_SCOPE, PLAYER003_ASSET, PLAYER004_ASSET,
    CALIBRATION_ASSETS, GPU_LBS_JOINT_SCOPE, JOINT_ROLE_IDS, cryptomatte_manifest_from_header,
)


def header():
    result = {"dataWindow": (np.array([0,0]),np.array([159,89])),
        "unreal/jobName": "CineCam_01", "unreal/sequenceFrameNumber": "000000",
        "unreal/aa/spatialSampleCount": "1", "unreal/aa/temporalSampleCount": "1",
        "unreal/camera/shutterAngle": "0", "unreal/camera/ActorHitProxyMask/fov":90}
    for key,value in {"fov":90,"sensorAspectRatio":16/9,"overscanPercent":0,"squeezeFactor":1}.items():
        result["unreal/camera/FinalImage/"+key] = value
    for key in ("x","y","z"):
        result["unreal/camera/curPos/"+key] = 0
    for key in ("pitch","yaw","roll"):
        result["unreal/camera/curRot/"+key] = 0
    return result


def row():
    return {"sample_succeeded":True,"keypoint_names":list(COCO17_NAMES),"output_frame":0,
        "sequence_source_frame":0,"temporal_sample_count":1,"spatial_sample_count":1,
        "motion_blur_fraction":0,"cache_fence_completed_this_engine_frame":True,"local_read_stable":True,
        "engine_frame_before":15,"engine_frame_after":15,"hook_engine_frame":15,
        "pose_revision_before":2,"pose_revision_after":2,"projection_jitter":[0,0],
        "off_center_projection_offset":[0,0],"image_size":[160,90],"camera_world_location_m":[0,0,0],
        "camera_forward":[1,0,0],"camera_right":[0,1,0],"camera_up":[0,0,1],"camera_horizontal_fov_degrees":90}


def test_axes_and_projection_keep_unknown():
    axes = ue_rotation_basis(-29, -140, 17)
    for i,a in enumerate(axes):
        for j,b in enumerate(axes):
            assert sum(x*y for x,y in zip(a,b)) == pytest.approx(int(i==j))
    camera = camera_from_exr_header(header())
    points = project_landmarks([{"world_m":[10,0,0]}],camera)
    assert points[0]["uv_px"] == pytest.approx([79.5,44.5])
    assert points[0]["visibility_state"] == "unknown"


def test_exact_media_frames_ignore_debug_png(tmp_path):
    for name in ("FinalImageCineCam_01.000000.png","FinalImageWorldDepthCineCam_01.000000.png","ActorHitProxyMask00main_camera.000000.png","000000.exr"):
        (tmp_path/name).touch()
    assert list(media_index(tmp_path,"rgb","CineCam_01")) == [0]
    assert list(media_index(tmp_path,"exr")) == [0]
    (tmp_path/"FinalImageCineCam_01.0000.png").touch()
    with pytest.raises(ValueError,match="重复"):
        media_index(tmp_path,"rgb","CineCam_01")


def test_filtered_finalimage_names_do_not_accept_debug_or_other_camera(tmp_path):
    for name in ("FinalImage000000.png", "FinalImage000001.png", "FinalImageWorldDepth000002.png",
                 "FinalImageCineCam_02.000002.png", "ActorHitProxyMask000002.png"):
        (tmp_path/name).touch()
    assert list(media_index(tmp_path, "rgb", "CineCam_01")) == [0, 1]


def test_mixed_finalimage_naming_cannot_duplicate_a_frame(tmp_path):
    (tmp_path/"FinalImage000000.png").touch()
    (tmp_path/"FinalImageCineCam_01.000000.png").touch()
    with pytest.raises(ValueError, match="同帧媒体重复"):
        media_index(tmp_path, "rgb", "CineCam_01")


@pytest.mark.parametrize("key,value", [("sequence_source_frame",1),("engine_frame_after",16),
    ("pose_revision_after",3),("cache_fence_completed_this_engine_frame",False),("projection_jitter",[.01,0])])
def test_reject_mismatched_frame_or_cache(key,value):
    sample = row()
    sample[key] = value
    with pytest.raises(ValueError):
        check_pair(sample,header(),"CineCam_01",0)


def test_pair_and_aspect_rejection():
    assert check_pair(row(),header(),"CineCam_01",0)["intrinsics"]["width"] == 160
    metadata = header()
    metadata["unreal/camera/FinalImage/sensorAspectRatio"] = 1
    with pytest.raises(ValueError,match="宽高比"):
        camera_from_exr_header(metadata)


def test_translation_is_not_articulation():
    a = {"sequence_source_frame":0,"world_m":[[i,0,1] for i in range(17)]}
    b = copy.deepcopy(a)
    b["sequence_source_frame"] = 1
    b["world_m"] = [[p[0]+3,p[1]+4,p[2]] for p in a["world_m"]]
    motion = pose_motion([b,a])[0]
    assert motion["hip_midpoint_translation_m"] == 5
    assert motion["max_translation_removed_delta_m"] == 0
    b["world_m"][9][2] += .1
    assert pose_motion([a,b])[0]["max_translation_removed_delta_m"] == pytest.approx(.1)


def test_expected_plan_detects_lookat_override_even_if_native_and_EXR_agree():
    camera = camera_from_exr_header(header())
    plan = {"closeup_location_cm":[0,0,0],"closeup_look_at_cm":[-100,0,0]}
    result = check_expected_camera_plan(camera,plan)
    assert result["position_max_error_m"] == 0
    assert result["forward_cos_similarity"] == -1
    assert not result["plan_camera_match_passed"]
    camera["extrinsics"]["forward"] = [-1,0,0]
    assert check_expected_camera_plan(camera,plan)["plan_camera_match_passed"]
    assert check_expected_camera_plan(camera,{}) == {"applicable":False}


def gpu_toy_case(tmp_path):
    """仅契约假例，绝不代表任何实际GPU捕获或MRQ已通过。"""
    identity = {"asset_path":"/Game/Toy/Mesh.Mesh","signature_scheme":"ue_lod_le_u32_f32_v1",
                "lod":"0","vertex_count":"3","topology_sha256":"a"*64,"geometry_sha256":"b"*64}
    manifest = tmp_path/"toy_manifest.json"
    manifest.write_text(json.dumps({"ue_identity":identity}),encoding="utf-8")
    target = {"entity_id":"L0","actor_object_name":"ToyActor","component_object_name":"MPFBVisual",
              "landmark_manifest":str(manifest)}
    plan = {"pose_sampling_path":"gpu_render_cpu_lbs","allow_gpu_render_cpu_lbs_diagnostic":True,
            "allow_pie_cpu_preparation":False,"diagnostic_scope":"pose17_gpu_render_cpu_lbs",
            "gpu_lbs_warmup_frames":8,"targets":[target]}
    receipt = {"pose_sampling_path":"gpu_render_cpu_lbs","gpu_lbs_diagnostic_requested":True,
               "diagnostic_scope":"pose17_gpu_render_cpu_lbs","gpu_lbs_prepared_engine_frame":7,
               "visual_mesh_override_requested":False,"visual_mesh_overrides":[],"restore_errors":[],
               "camera_state_restored":True,"streaming_state_restart_required":False,"pipeline_setup_error":""}
    sample = row()
    transform = {"translation_cm":[0.,0.,0.],"rotation_xyzw":[0.,0.,0.,1.],"scale_xyz":[1.,1.,1.]}
    sample.update(entity_id="L0",camera_name="CineCam_01",temporal_sample_index=0,spatial_sample_index=0,
        mrq_synchronization_verified=False,sampling_method=GPU_LBS_METHOD,render_cpu_enabled_at_read=False,
        render_static_at_read=False,predicted_lod_at_read=0,pose_fence_method=GPU_LBS_FENCE,
        end_frame_update_tasks_finished=True,gpu_lbs_warmup_engine_frames_observed=8,
        ref_to_local_signature_scheme="ue_ref_to_local_rowmajor_f32_le_v1",ref_to_local_matrices_sha256="c"*64,
        skin_weight_signature_scheme="ue_component_lod0_fixed_skinweight_raw_v1",skin_weight_data_sha256="d"*64,
        manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        mesh_identity={**{k:v for k,v in identity.items() if k not in ("lod","vertex_count")},"l_od":0,"vertex_count":3},
        component_path="/Game/Toy/UEDPIE_0_Toy.Toy:PersistentLevel.ToyActor.MPFBVisual",
        bone_space_base_component_path="/Game/Toy/UEDPIE_0_Toy.Toy:PersistentLevel.ToyActor.DriverMesh",
        bone_space_base_names=list(LIMB_BONES),bone_space_base_transforms_ue_cm=[copy.deepcopy(transform) for _ in LIMB_BONES],
        bone_pose_transform_encoding=TRANSFORM_ENCODING,world_transform_encoding=TRANSFORM_ENCODING,
        actor_world_transform_ue_cm=copy.deepcopy(transform),visual_world_transform_ue_cm=copy.deepcopy(transform),
        world_m=[[10.,.001*i,0.] for i in range(17)])
    eligibility = dict.fromkeys(("checks_executed","eligible","pie_world","weights_readable"),True)
    eligibility.update(dict.fromkeys(("render_cpu_enabled","render_static","has_nanite_data","parallel_evaluation_running",
        "leader_parallel_evaluation_running","non_finite_morph_weights","lod0_has_cloth_data","variable_influence_weights"),False))
    eligibility.update(dict.fromkeys(("active_morph_weight_count","manual_morph_curve_count","external_morph_set_count",
        "cloth_simulation_count","lod0_cloth_section_count","mesh_deformer_asset_count","mesh_deformer_instance_count",
        "material_wpo_slot_count","null_material_slot_count"),0))
    eligibility.update(weight_vertex_count=3,reference_vertex_count=3,max_bone_influences=4)
    sample["gpu_lbs_eligibility"] = eligibility
    return plan,receipt,sample


def test_gpu_contract_toy_requires_optin_and_retains_unknown_projection(tmp_path):
    plan,receipt,sample = gpu_toy_case(tmp_path)
    context = sampling_context(plan,receipt)
    camera = check_pair(sample,header(),"CineCam_01",0,sampling=context)
    assert camera["native_sampling_evidence"]["actual_gpu_render_cpu_lbs_checks_passed"]
    assert not camera["native_sampling_evidence"]["final_rendered_surface_equivalence_verified"]
    assert project_landmarks([{"world_m":sample["world_m"][0]}],camera)[0]["visibility_state"] == "unknown"
    with pytest.raises(ValueError,match="显式"):
        check_pair(sample,header(),"CineCam_01",0)
    with pytest.raises(ValueError,match="CPU计划"):
        sampling_context({},receipt)
    assert sampling_context({}, {}) == {"path":"cpu_skin_cache"}
    assert check_pair(row(),header(),"CineCam_01",0)["intrinsics"]["width"] == 160


def test_gpu_native_writer_key_encoding_roundtrip_is_still_toy(tmp_path):
    ue_dir = Path(__file__).resolve().parents[1]/"ue"
    if str(ue_dir) not in sys.path:sys.path.insert(0,str(ue_dir))
    from pose17_native_receipt import normalize_native_receipt
    plan,receipt,sample = gpu_toy_case(tmp_path)
    record = copy.deepcopy(sample)
    points = record.pop("world_m")
    identity = record.pop("mesh_identity")
    identity["lOD"] = identity.pop("l_od")  # 与实际UStruct writer的LOD字段编码一致。
    record["sample"] = {"worldPointsM":[dict(zip("xyz",point)) for point in points],
                        "meshIdentity":identity,"keypointNames":record.pop("keypoint_names"),
                        "gpuLbsEligibility":record.pop("gpu_lbs_eligibility")}
    normalized = normalize_native_receipt({**receipt,"schema":"futsalmot_pose17_native_receipt_v1","records":[record]})
    row_value = normalized["records"][0]
    assert row_value["mesh_identity"]["l_od"] == 0
    assert row_value["visibility_state"] == ["unknown"]*17
    assert check_pair(row_value,header(),"CineCam_01",0,sampling=sampling_context(plan,normalized))


@pytest.mark.parametrize("where,key,value", [
    ("plan","allow_gpu_render_cpu_lbs_diagnostic",False),("plan","allow_pie_cpu_preparation",True),
    ("plan","diagnostic_scope","other"),("plan","gpu_lbs_warmup_frames",7),
    ("plan","gpu_lbs_warmup_frames",True),("plan","allow_pie_visual_mesh_override",True),
    ("receipt","gpu_lbs_diagnostic_requested",False),("receipt","gpu_lbs_prepared_engine_frame",-1),
    ("receipt","visual_mesh_override_requested",True),("receipt","camera_state_restored",False),
    ("receipt","restore_errors",["failed"]),("receipt","streaming_state_restart_required",True),
    ("receipt","pipeline_setup_error","failed"),])
def test_gpu_plan_and_native_top_fail_closed(tmp_path,where,key,value):
    plan,receipt,_ = gpu_toy_case(tmp_path)
    (plan if where=="plan" else receipt)[key] = value
    with pytest.raises(ValueError):sampling_context(plan,receipt)


@pytest.mark.parametrize("key,value", [
    ("render_cpu_enabled_at_read",True),("render_static_at_read",True),("predicted_lod_at_read",1),
    ("sampling_method","existing_cpu_skin_cache_unverified_render_sample"),
    ("pose_fence_method","cpu_skin_cache_dynamic_data_render_commands_flush"),
    ("end_frame_update_tasks_finished",False),("gpu_lbs_warmup_engine_frames_observed",7),
    ("gpu_lbs_warmup_engine_frames_observed",9),("ref_to_local_signature_scheme","unknown"),
    ("ref_to_local_matrices_sha256","not-a-hash"),("skin_weight_signature_scheme","unknown"),
    ("skin_weight_data_sha256",""),("manifest_sha256","f"*64),
    ("component_path","/Game/Toy/Toy:PersistentLevel.ToyActor.MPFBVisual"),
    ("bone_space_base_component_path","/Game/Toy/UEDPIE_0_Toy.Toy:PersistentLevel.Other.DriverMesh"),
    ("bone_pose_transform_encoding","unverified_text"),("world_transform_encoding","unverified_text"),
    ("camera_name","OtherCamera"),("temporal_sample_index",1),("spatial_sample_index",1),
    ("mrq_synchronization_verified",True),("local_read_stable",1),
    # 新GPU路径仍必须通过原有fence/frame/revision/EXR与相机门禁。
    ("cache_fence_completed_this_engine_frame",False),("local_read_stable",False),
    ("engine_frame_after",16),("pose_revision_after",3),("sequence_source_frame",1),
    ("projection_jitter",[.01,0]),("camera_world_location_m",[1,0,0]),])
def test_gpu_sample_does_not_bypass_identity_fence_geometry(tmp_path,key,value):
    plan,receipt,sample = gpu_toy_case(tmp_path);sample[key] = value
    with pytest.raises(ValueError):check_pair(sample,header(),"CineCam_01",0,sampling=sampling_context(plan,receipt))


@pytest.mark.parametrize("key", ["checks_executed","eligible","pie_world","weights_readable",
    "render_cpu_enabled","render_static","has_nanite_data","parallel_evaluation_running",
    "leader_parallel_evaluation_running","non_finite_morph_weights","lod0_has_cloth_data","variable_influence_weights",
    "active_morph_weight_count","manual_morph_curve_count","external_morph_set_count","cloth_simulation_count",
    "lod0_cloth_section_count","mesh_deformer_asset_count","mesh_deformer_instance_count","material_wpo_slot_count",
    "null_material_slot_count","weight_vertex_count","reference_vertex_count","max_bone_influences"])
@pytest.mark.parametrize("missing", [False,True])
def test_gpu_eligibility_each_actual_rejection_field_required(tmp_path,key,missing):
    plan,receipt,sample = gpu_toy_case(tmp_path)
    evidence = sample["gpu_lbs_eligibility"]
    old = evidence.pop(key)
    if not missing:evidence[key] = not old if type(old) is bool else (0 if key=="max_bone_influences" else old+1)
    with pytest.raises(ValueError):check_pair(sample,header(),"CineCam_01",0,sampling=sampling_context(plan,receipt))


@pytest.mark.parametrize("case", ["bone_count","bone_name","bone_nonfinite","world_quat","lod","vertex_count","topology","nonfinite_point"])
def test_gpu_current_pose_and_reference_identity_required(tmp_path,case):
    plan,receipt,sample = gpu_toy_case(tmp_path)
    if case=="bone_count":sample["bone_space_base_transforms_ue_cm"].pop()
    if case=="bone_name":sample["bone_space_base_names"][0] = "wrong_bone"
    if case=="bone_nonfinite":sample["bone_space_base_transforms_ue_cm"][0]["translation_cm"][0] = float("nan")
    if case=="world_quat":sample["visual_world_transform_ue_cm"]["rotation_xyzw"] = [0,0,0,0]
    if case=="lod":sample["mesh_identity"]["l_od"] = 1
    if case=="vertex_count":sample["mesh_identity"]["vertex_count"] = 4
    if case=="topology":sample["mesh_identity"]["topology_sha256"] = "e"*64
    if case=="nonfinite_point":sample["world_m"][0][0] = float("nan")
    with pytest.raises(ValueError):check_pair(sample,header(),"CineCam_01",0,sampling=sampling_context(plan,receipt))


def test_gpu_plan_alone_without_native_output_cannot_pass(tmp_path):
    plan,_,_ = gpu_toy_case(tmp_path)
    plan["cameras"] = [{"camera_id":"CineCam_01"}]
    attempt = tmp_path/"attempt";attempt.mkdir()
    (attempt/"plan.json").write_text(json.dumps(plan),encoding="utf-8")
    with pytest.raises(ValueError,match="缺少 native 收据"):
        inspect(attempt,tmp_path/"analysis_should_not_exist")
    assert not (tmp_path/"analysis_should_not_exist").exists()


def gpu_character_toy_case(tmp_path, player_id=4):
    """单角色组合契约假例；不代表该角色已渲染或材质已在UE中验收。"""
    plan,receipt,sample = gpu_toy_case(tmp_path)
    asset=CALIBRATION_ASSETS[player_id]
    plan.update(diagnostic_scope=GPU_LBS_CHARACTER_SCOPE,allow_pie_visual_mesh_override=True,
                allow_pie_driver_hide=True,calibration_player_id=player_id,override_warmup_frames=8)
    target=plan['targets'][0]
    target.update(sequence_binding_name='Player_L0',diagnostic_visual_mesh_override=asset)
    manifest=Path(target['landmark_manifest'])
    content=json.loads(manifest.read_text(encoding='utf-8'));content['ue_identity']['asset_path']=asset
    manifest.write_text(json.dumps(content),encoding='utf-8')
    sample['manifest_sha256']=hashlib.sha256(manifest.read_bytes()).hexdigest()
    sample['mesh_identity']['asset_path']=asset
    sample['override_warmup_engine_frames_observed']=9
    evidence=dict(component_path=sample['component_path'],original_asset_path=PLAYER003_ASSET,applied_asset_path=asset,
        shared_skeleton_path='/Game/Toy/SKEL_MPFB_Base.SKEL_MPFB_Base',
        animation_class_path='/Game/Toy/ABP_MPFB_RuntimeRetarget.ABP_MPFB_RuntimeRetarget_C',
        original_animation_instance_path=sample['component_path']+'.ABP_MPFB_RuntimeRetarget_C_0',
        original_override_material_paths=[''],applied_asset_material_paths=['/Game/Toy/M_Body.M_Body'],applied_engine_frame=6,
        apply_error='',restore_error='',gpu_lbs_validation_error='',
        gpu_lbs_eligibility_after_override=copy.deepcopy(sample['gpu_lbs_eligibility']))
    for name in ('identity_verified','apply_succeeded','animation_binding_preserved','new_asset_materials_verified',
                 'restore_attempted','restore_succeeded','streaming_state_matched','gpu_lbs_validation_attempted','gpu_lbs_validation_succeeded'):
        evidence[name]=True
    receipt.update(diagnostic_scope=GPU_LBS_CHARACTER_SCOPE,visual_mesh_override_requested=True,visual_mesh_overrides=[evidence])
    return plan,receipt,sample


def test_gpu_character_toy_scope_is_strict_and_does_not_relax_original_scope(tmp_path):
    plan,receipt,sample=gpu_character_toy_case(tmp_path)
    context=sampling_context(plan,receipt)
    camera=check_pair(sample,header(),'CineCam_01',0,sampling=context)
    result=camera['native_sampling_evidence']
    assert result['diagnostic_scope']==GPU_LBS_CHARACTER_SCOPE
    assert result['mesh_identity']['asset_path']==PLAYER004_ASSET
    assert result['character_override_evidence']['restore_succeeded']
    assert result['override_warmup_engine_frames_observed']==9
    assert not result['final_rendered_surface_equivalence_verified']
    # CPU=false本身不能让有override的收据进入原003 scope。
    old_plan=copy.deepcopy(plan);old_receipt=copy.deepcopy(receipt)
    old_plan['diagnostic_scope']=old_receipt['diagnostic_scope']=GPU_LBS_SCOPE
    with pytest.raises(ValueError,match='仍拒绝'):sampling_context(old_plan,old_receipt)
    plan['pose_sampling_path']='cpu_skin_cache'
    with pytest.raises(ValueError,match='伪装'):sampling_context(plan,receipt)


@pytest.mark.parametrize('key,value',[
    ('allow_pie_visual_mesh_override',False),('allow_pie_driver_hide',False),('allow_pie_cpu_preparation',True),
    ('allow_gpu_render_cpu_lbs_diagnostic',False),('calibration_player_id',3),('calibration_player_id',True),
    ('override_warmup_frames',9),('override_warmup_frames',True)])
def test_gpu_character_plan_combination_not_implicit(tmp_path,key,value):
    plan,receipt,_=gpu_character_toy_case(tmp_path);plan[key]=value
    with pytest.raises(ValueError):sampling_context(plan,receipt)


@pytest.mark.parametrize('key',[
    'identity_verified','apply_succeeded','animation_binding_preserved','new_asset_materials_verified',
    'restore_attempted','restore_succeeded','streaming_state_matched','gpu_lbs_validation_attempted','gpu_lbs_validation_succeeded'])
@pytest.mark.parametrize('missing',[False,True])
def test_gpu_character_native_apply_restore_material_checks_each_required(tmp_path,key,missing):
    plan,receipt,_=gpu_character_toy_case(tmp_path)
    evidence=receipt['visual_mesh_overrides'][0];evidence.pop(key)
    if not missing:evidence[key]=False
    with pytest.raises(ValueError):sampling_context(plan,receipt)


@pytest.mark.parametrize('key,value',[
    ('apply_error','failed'),('restore_error','failed'),('gpu_lbs_validation_error','failed'),
    ('original_asset_path',PLAYER004_ASSET),('applied_asset_path',PLAYER003_ASSET),
    ('component_path','/Game/Toy/UEDPIE_0_Toy.Toy:PersistentLevel.Other.MPFBVisual'),
    ('shared_skeleton_path','/Game/Toy/Other.Other'),('animation_class_path','/Game/Toy/Other.Other_C'),
    ('original_animation_instance_path','/Game/Toy/NotThePIEInstance'),
    ('applied_asset_material_paths',[]),('applied_asset_material_paths',['']),
    ('original_override_material_paths',None),('applied_engine_frame',8),('applied_engine_frame',-1)])
def test_gpu_character_native_identity_restore_and_material_paths_fail_closed(tmp_path,key,value):
    plan,receipt,_=gpu_character_toy_case(tmp_path);receipt['visual_mesh_overrides'][0][key]=value
    with pytest.raises(ValueError):sampling_context(plan,receipt)


@pytest.mark.parametrize('key',[
    'checks_executed','eligible','pie_world','weights_readable','render_cpu_enabled','render_static',
    'has_nanite_data','parallel_evaluation_running','leader_parallel_evaluation_running','non_finite_morph_weights',
    'lod0_has_cloth_data','variable_influence_weights','active_morph_weight_count','manual_morph_curve_count',
    'external_morph_set_count','cloth_simulation_count','lod0_cloth_section_count','mesh_deformer_asset_count',
    'mesh_deformer_instance_count','material_wpo_slot_count','null_material_slot_count',
    'weight_vertex_count','reference_vertex_count','max_bone_influences'])
def test_gpu_character_swap_eligibility_not_inferred_from_cpu_false(tmp_path,key):
    plan,receipt,_=gpu_character_toy_case(tmp_path)
    eligibility=receipt['visual_mesh_overrides'][0]['gpu_lbs_eligibility_after_override']
    old=eligibility[key]
    eligibility[key]=not old if type(old) is bool else (0 if key=='max_bone_influences' else old+1)
    with pytest.raises(ValueError):sampling_context(plan,receipt)


@pytest.mark.parametrize('case',['wrong_target','wrong_entity','old_manifest','two_targets','two_overrides','elapsed','elapsed_missing','row_component'])
def test_gpu_character_manifest_target_and_two_warmup_identities_bound(tmp_path,case):
    plan,receipt,sample=gpu_character_toy_case(tmp_path)
    target=plan['targets'][0]
    if case=='wrong_target':target['diagnostic_visual_mesh_override']=PLAYER003_ASSET
    if case=='wrong_entity':target['entity_id']='R0'
    if case=='old_manifest':
        path=Path(target['landmark_manifest']);data=json.loads(path.read_text());data['ue_identity']['asset_path']=PLAYER003_ASSET
        path.write_text(json.dumps(data))
    if case=='two_targets':plan['targets'].append(copy.deepcopy(target))
    if case=='two_overrides':receipt['visual_mesh_overrides'].append(copy.deepcopy(receipt['visual_mesh_overrides'][0]))
    if case=='elapsed':sample['override_warmup_engine_frames_observed']=8
    if case=='elapsed_missing':sample.pop('override_warmup_engine_frames_observed')
    if case=='row_component':receipt['visual_mesh_overrides'][0]['component_path']=sample['component_path'].replace('UEDPIE_0','UEDPIE_1')
    with pytest.raises(ValueError):check_pair(sample,header(),'CineCam_01',0,sampling=sampling_context(plan,receipt))


@pytest.mark.parametrize('player_id',[4,5,6,7,8,9])
def test_gpu_whitelist_all_single_roles_use_exact_asset_without_duplicate_native_id(tmp_path,player_id):
    plan,receipt,sample=gpu_character_toy_case(tmp_path,player_id)
    assert 'calibration_player_id' not in receipt
    result=check_pair(sample,header(),'CineCam_01',0,sampling=sampling_context(plan,receipt))['native_sampling_evidence']
    assert result['calibration_player_id']==player_id
    assert result['mesh_identity']['asset_path']==CALIBRATION_ASSETS[player_id]
    assert result['character_override_evidence']['original_asset_path']==PLAYER003_ASSET


@pytest.mark.parametrize('case',['cross_role','same_fname_other_package','bad_id'])
def test_gpu_whitelist_rejects_cross_role_prefix_and_nonint_plan(tmp_path,case):
    plan,receipt,sample=gpu_character_toy_case(tmp_path,5)
    if case=='bad_id':plan['calibration_player_id']='5'
    else:
        bad=CALIBRATION_ASSETS[6] if case=='cross_role' else '/Game/Other/Player_005/MPFB_Player_005.MPFB_Player_005'
        target=plan['targets'][0];target['diagnostic_visual_mesh_override']=bad
        path=Path(target['landmark_manifest']);content=json.loads(path.read_text());content['ue_identity']['asset_path']=bad
        path.write_text(json.dumps(content));sample['manifest_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
        sample['mesh_identity']['asset_path']=bad;receipt['visual_mesh_overrides'][0]['applied_asset_path']=bad
    with pytest.raises(ValueError):sampling_context(plan,receipt)


def gpu_joint_toy_case(tmp_path):
    """固定十实体/八事务的CPU假例；不代表真实联合捕获。"""
    plan,receipt,template=gpu_character_toy_case(tmp_path)
    target_template=copy.deepcopy(plan['targets'][0])
    evidence_template=copy.deepcopy(receipt['visual_mesh_overrides'][0])
    identity_template=json.loads(Path(target_template['landmark_manifest']).read_text())
    plan.update(diagnostic_scope=GPU_LBS_JOINT_SCOPE,allow_joint_pose17_diagnostic=True,
                frame_count=10,start_frame=100,review_frame_indices=[5],source_scope='toy_cpu_only',
                cameras=[dict(camera_id='CineCam_%02d'%i,source_sequence='/Game/Toy/LS_%02d'%i) for i in range(1,5)])
    plan.pop('calibration_player_id');plan['targets']=[]
    receipt.update(diagnostic_scope=GPU_LBS_JOINT_SCOPE,visual_mesh_overrides=[])
    samples=[]
    for entity,role in JOINT_ROLE_IDS.items():
        asset=PLAYER003_ASSET if role==3 else CALIBRATION_ASSETS[role]
        manifest=tmp_path/('manifest_%s.json'%entity)
        content=copy.deepcopy(identity_template);content['ue_identity']['asset_path']=asset
        manifest.write_text(json.dumps(content),encoding='utf-8')
        target=dict(target_template,entity_id=entity,actor_object_name='ToyActor_'+entity,
                    sequence_binding_name='Player_'+entity,calibration_player_id=role,landmark_manifest=str(manifest))
        if role==3:target.pop('diagnostic_visual_mesh_override')
        else:target['diagnostic_visual_mesh_override']=asset
        plan['targets'].append(target)
        sample=copy.deepcopy(template);sample.update(entity_id=entity,sequence_source_frame=100)
        sample['component_path']=template['component_path'].replace('ToyActor.',target['actor_object_name']+'.')
        sample['bone_space_base_component_path']=template['bone_space_base_component_path'].replace('ToyActor.',target['actor_object_name']+'.')
        sample['mesh_identity']['asset_path']=asset
        sample['manifest_sha256']=hashlib.sha256(manifest.read_bytes()).hexdigest()
        if role==3:sample.pop('override_warmup_engine_frames_observed')
        else:
            evidence=copy.deepcopy(evidence_template)
            evidence.update(component_path=sample['component_path'],applied_asset_path=asset,
                            original_asset_path=CALIBRATION_ASSETS[9],
                            original_animation_instance_path=sample['component_path']+'.ABP_MPFB_RuntimeRetarget_C_0')
            receipt['visual_mesh_overrides'].append(evidence)
        samples.append(sample)
    return plan,receipt,samples


def test_joint_toy_exact_roles_components_source_and_relative_frames(tmp_path):
    plan,receipt,samples=gpu_joint_toy_case(tmp_path)
    metadata=header();metadata.update({'unreal/sequenceFrameNumber':100,'unreal/sequenceFrameNumberRelative':0})
    context=sampling_context(plan,receipt)
    assert len(context['overrides_by_entity'])==8
    for sample in samples:
        result=check_pair(sample,metadata,'CineCam_01',0,sampling=context)['native_sampling_evidence']
        assert result['calibration_player_id']==JOINT_ROLE_IDS[sample['entity_id']]
        assert bool(result['character_override_evidence'])==(sample['entity_id'] not in ('L0','R2'))
        assert not result['final_rendered_surface_equivalence_verified']
    for key,value in [('unreal/sequenceFrameNumberRelative',1),('unreal/sequenceFrameNumber',0)]:
        bad=dict(metadata);bad[key]=value
        with pytest.raises(ValueError):check_pair(samples[0],bad,'CineCam_01',0,sampling=context)


@pytest.mark.parametrize('case',['optin','review','mapping','duplicate_component','wrong_component','003_override','original_asset','cross_manifest'])
def test_joint_toy_rejects_misbound_or_implicit_transactions(tmp_path,case):
    plan,receipt,samples=gpu_joint_toy_case(tmp_path)
    if case=='optin':plan['allow_joint_pose17_diagnostic']=False
    if case=='review':plan['review_frame_indices']=[0]
    if case=='mapping':plan['targets'][1]['calibration_player_id']=5
    if case=='duplicate_component':receipt['visual_mesh_overrides'][-1]['component_path']=receipt['visual_mesh_overrides'][0]['component_path']
    if case=='wrong_component':receipt['visual_mesh_overrides'][0]['component_path']=receipt['visual_mesh_overrides'][0]['component_path'].replace('ToyActor_L1.','Wrong.')
    if case=='003_override':plan['targets'][0]['diagnostic_visual_mesh_override']=PLAYER003_ASSET
    if case=='original_asset':receipt['visual_mesh_overrides'][0]['original_asset_path']='/Game/Other/MPFB_Player_009.MPFB_Player_009'
    if case=='cross_manifest':plan['targets'][1]['landmark_manifest']=plan['targets'][2]['landmark_manifest']
    with pytest.raises(ValueError):sampling_context(plan,receipt)


def test_joint_header_manifest_only_and_missing_rejection():
    assert cryptomatte_manifest_from_header({'cryptomatte/a/manifest':'{"Player_L0":"abcdef01"}'})=={'Player_L0':'abcdef01'}
    for bad in ({},{'cryptomatte/a/manifest':'{}'},{'cryptomatte/a/manifest':'{"Player_L0":"bad"}'}):
        with pytest.raises(ValueError):cryptomatte_manifest_from_header(bad)


def test_joint_three_frame_floor_plan_requires_exact_middle_review(tmp_path):
    plan,receipt,_=gpu_joint_toy_case(tmp_path)
    plan.update(frame_count=3,review_frame_indices=[1])
    assert sampling_context(plan,receipt)['review_frame_indices']==[1]
    for invalid in ([5],[0],[True],[1,2]):
        plan['review_frame_indices']=invalid
        with pytest.raises(ValueError):sampling_context(plan,receipt)
    for invalid in (True,1,4,9):
        plan.update(frame_count=invalid,review_frame_indices=[1])
        with pytest.raises(ValueError):sampling_context(plan,receipt)


@pytest.mark.parametrize('frame_count,missing_rgb',[(10,False),(3,False),(3,True)])
def test_joint_toy_inspect_only_decodes_four_middle_frames(tmp_path,monkeypatch,frame_count,missing_rgb):
    """tiny假媒体走实际normalizer与analyzer，禁止把它称为MRQ证据。"""
    from PIL import Image
    import OpenEXR
    import grf_ue_bridge.cryptomatte as crypto
    plan,receipt,samples=gpu_joint_toy_case(tmp_path)
    review=frame_count//2
    plan.update(frame_count=frame_count,review_frame_indices=[review])
    attempt=tmp_path/'attempt';attempt.mkdir()
    (attempt/'plan.json').write_text(json.dumps(plan),encoding='utf-8')
    manifest={'Player_'+entity:('%08x'%i) for i,entity in enumerate(JOINT_ROLE_IDS,1)}
    opened=[];decoded=[]
    class HeaderOnly:
        def __init__(self,path,**kwargs):
            assert kwargs=={'header_only':True};self.path=Path(path);opened.append(self.path)
        def header(self):
            frame=int(self.path.stem);value=header()
            value.update({'unreal/jobName':self.path.parent.name,'unreal/sequenceFrameNumber':100+frame,
                          'unreal/sequenceFrameNumberRelative':frame,'cryptomatte/toy/manifest':json.dumps(manifest)})
            return value
    def load_pixels(path):
        decoded.append(Path(path));return manifest,object()
    mask=np.zeros((90,160),dtype=np.uint8)
    for index in range(10):mask[30:60,index*16:(index+1)*16]=index+1
    monkeypatch.setattr(OpenEXR,'File',HeaderOnly)
    monkeypatch.setattr(crypto,'load_cryptomatte',load_pixels)
    monkeypatch.setattr(crypto,'build_mask',lambda *args:(mask,{e:480 for e in JOINT_ROLE_IDS}))
    for camera in plan['cameras']:
        name=camera['camera_id'];directory=attempt/name;directory.mkdir();records=[]
        for frame in range(frame_count):
            if not (missing_rgb and name=='CineCam_04' and frame==2):
                Image.new('RGB',(160,90),(20,30,40)).save(directory/('FinalImage%06d.png'%frame))
            (directory/('%06d.exr'%frame)).touch()
            for sample in samples:
                value=copy.deepcopy(sample);value.update(camera_name=name,output_frame=frame,
                    sequence_source_frame=100+frame,world_time_seconds=frame/30)
                value['world_points_m']=value.pop('world_m')
                records.append({'sample':value})
        native=dict(receipt,schema='futsalmot_pose17_native_receipt_v1',records=records,setup_error='',record_limit_reached=False)
        (attempt/(name+'_native_pose_restore_receipt.json')).write_text(json.dumps(native),encoding='utf-8')
    output=tmp_path/'analysis'
    report=inspect(attempt,output)
    if missing_rgb:
        assert not report['structural_pairing_passed']
        assert any('帧集合' in error for error in report['errors'])
        return
    assert report['structural_pairing_passed'],report['errors']
    assert len(opened)==4*frame_count and len(decoded)==4 and {p.stem for p in decoded}=={'%06d'%review}
    assert len(list(output.rglob('*_all_players_overlay.png')))==4
    assert len(list(output.rglob('*_mask.png')))==4
    assert not list(output.rglob('*_crop.png'))
    assert len(list(output.rglob('*.png')))==8
    rows=[r for c in report['cameras'].values() for r in c['frames']]
    assert len(rows)==40*frame_count
    for item in rows:
        assert item['source_frame']==100+item['output_frame']
        if item['output_frame']!=review:
            assert item['pixel_evaluation_status']=='not_evaluated'
            assert item['visible_instance_area_pixels'] is item['visible_instance_bbox_xywh'] is None
            assert all(p['instance_mask_hit_at_nearest_pixel'] is None for p in item['keypoints'])
        assert all(p['visibility_state']=='unknown' for p in item['keypoints'])
