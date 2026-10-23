"""只读元数据交互候选的CPU夹具；不表示真实遮挡或模型恢复成功。"""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from grf_ue_bridge.interaction_window_candidate import DEFAULTS,analyze_group,contiguous_runs,pose_coordinates
from grf_ue_bridge.tools import analyze_interaction_windows as tool


def fixture():
    timeline=[dict(frame_index=i,source_frame_index=100+i) for i in range(50)]
    cameras={}
    for camera_id in ("CineCam_01","CineCam_02"):
        images=[];worlds=[]
        for frame in range(50):
            image_objects=[];world_objects=[]
            for entity,index in (("L0",0),("L1",1)):
                root=np.array([frame*.2,index*.6,1.])
                p=np.stack([root+[.01*j,0.,.02*j] for j in range(17)])
                p[11]=root+[-.1,0.,0.];p[12]=root+[.1,0.,0.]
                p[5]=root+[-.1,0.,.5];p[6]=root+[.1,0.,.5]
                if camera_id=="CineCam_02":p+=.000001
                x=100. if index==0 else (88. if frame==18 else 112.) if 15<=frame<21 else 300.
                image_objects.append(dict(entity_id=entity,bbox_visible_xywh=[x-20.,80.,40.,80.],
                    uv_px=[[x,120.]]*17,location_valid=[True]*17,visibility_state=["unknown"]*17))
                world_objects.append(dict(entity_id=entity,world_m=p.tolist()))
            images.append(dict(frame_index=frame,objects=image_objects))
            worlds.append(dict(frame_index=frame,source_frame_index=100+frame,objects=world_objects))
        cameras[camera_id]=dict(images=images,worlds=worlds,view={"intrinsics":{"width":640,"height":360}})
    return timeline,cameras,dict(DEFAULTS,window_frames=40,stride_frames=10)


def test_proxy_event_context_duration_and_shared_source_window():
    timeline,cameras,config=fixture()
    result=analyze_group(7,"train",timeline,cameras,10,config)
    assert result["nominal_sample_duration_seconds"]==5. and result["first_to_last_timestamp_span_seconds"]==4.9
    for data in result["per_camera"].values():
        assert data["frames_with_any_bbox_overlap"]==6 and len(data["overlap_events"])==1
        assert data["overlap_events"][0]["full_separated_pre_post_context"] is True
        assert data["overlap_events"][0]["start_frame"]==15 and data["overlap_events"][0]["end_frame_exclusive"]==21
        assert len(data["image_horizontal_order_swap_proxies"])==2
        assert data["height_at_640_px"]["median"]==80.
    selected=result["selected_candidate"]
    assert selected["source_frame_range"]==[100,140] and len(selected["cameras_shared_same_source_window"])==2
    assert selected["same_team_context_complete_view_events"]==2
    assert result["physical_occlusion_measured"] is False and result["identity_recovery_success_measured"] is False


def test_pose_repeat_separated_from_root_motion_without_world_averaging():
    timeline,cameras,config=fixture();result=analyze_group(7,"train",timeline,cameras,10,config)
    assert result["world_motion_reference_camera"]=="CineCam_01"
    motion=result["motion"]
    assert motion["world_proximity_pair_frames"]==50
    for entity in motion["entities"].values():
        assert entity["path_length_ground_m"]==pytest.approx(9.8)
        assert entity["adjacent_near_repeat_fraction"]==1.
        assert entity["locomotion_proxy_steps"]==entity["locomotion_with_near_repeated_pose_steps"]==49
        assert entity["longest_adjacent_near_repeat_run_frames"]==50
    source=np.asarray(cameras["CineCam_01"]["worlds"][0]["objects"][0]["world_m"])
    angle=.8;rotation=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
    _,_,a=pose_coordinates(source);_,_,b=pose_coordinates(source@rotation.T+[2.,3.,0.])
    assert np.allclose(a,b,atol=1e-12)


def test_selection_ignores_model_outcomes_and_rejects_discontinuous_gt():
    timeline,cameras,config=fixture()
    before=analyze_group(7,"train",timeline,cameras,10,config)["selected_candidate"]
    for camera in cameras.values():
        for frame in camera["images"]:frame["model_tracking_error"]=999999
    assert analyze_group(7,"train",timeline,cameras,10,config)["selected_candidate"]==before
    timeline[20]["source_frame_index"]+=1
    with pytest.raises(ValueError):analyze_group(7,"train",timeline,cameras,10,config)
    assert contiguous_runs([0,1,4,6,7])==[[0,2],[4,5],[6,8]]


def test_metadata_cli_reads_no_media_and_preserves_split(tmp_path,monkeypatch):
    timeline,cameras,config=fixture();source=tmp_path/"published";source.mkdir();saved=tmp_path/"Saved";saved.mkdir()
    monkeypatch.setattr(tool,"WORKSPACE",tmp_path)
    def dump(path,value):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value),encoding="utf-8")
    group=source/"groups/seed_00000007";refs={}
    for name,data in cameras.items():
        base=group/"cameras"/name
        for key,value in (("pose2d",data["images"]),("pose3d_world",data["worlds"])):
            path=base/(key+".jsonl");path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text("".join(json.dumps(row)+"\n" for row in value),encoding="utf-8")
        dump(base/"view.json",data["view"])
        refs[name]={"view_ref":"cameras/"+name+"/view.json","pose2d_ref":"cameras/"+name+"/pose2d.jsonl",
            "pose3d_world_ref":"cameras/"+name+"/pose3d_world.jsonl"}
    (group/"timeline.jsonl").write_text("".join(json.dumps(row)+"\n" for row in timeline),encoding="utf-8")
    dump(group/"group.json",dict(seed=7,world_storage="per_camera_native",world_values_modified=False,
        source_motion={"fps":10},timeline_ref="timeline.jsonl",cameras=refs))
    dump(source/"dataset.json",dict(schema="futsalmot_pose17_location_pilot_dataset_v1",fps=10,
        groups=[dict(seed=7,split="synthetic_dev",group_ref="groups/seed_00000007/group.json")]))
    output=saved/"candidate_a01";result=tool.analyze_dataset(source/"dataset.json",output,config)
    assert result["coverage"]["source_time_samples"]==50 and result["coverage"]["view_frames"]==100
    assert result["coverage"]["source_nominal_seconds"]==5 and result["coverage"]["view_nominal_seconds"]==10
    assert result["selected_candidates"][0]["split"]=="synthetic_dev"
    assert all(Path(path).suffix in (".json",".jsonl") for path in result["inputs_text_sha256"])
    assert result["media_pixels_read"] is False and result["media_hashes_read"] is False
    assert (output/"coverage.md").exists() and (output/"candidate_windows.jsonl").exists()
    with pytest.raises(ValueError):tool.analyze_dataset(source/"dataset.json",output,config)
