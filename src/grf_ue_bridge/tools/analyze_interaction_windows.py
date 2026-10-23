"""只读发布的Pose JSON元数据，生成独立交互窗口候选与coverage摘要。"""
import argparse
import hashlib
import json
from pathlib import Path,PurePosixPath
import stat

from grf_ue_bridge.interaction_window_candidate import DEFAULTS,VERSION,analyze_group,require,summary

WORKSPACE=Path(__file__).resolve().parents[6]


def metadata(path,inputs,jsonl=False):
    path=Path(path).absolute()
    require(path.suffix in (".json",".jsonl"),"只允许JSON元数据，禁止媒体扫描")
    for parent in (path,*path.parents):
        if parent.exists():
            info=parent.lstat()
            require(not stat.S_ISLNK(info.st_mode) and not(getattr(info,"st_file_attributes",0)&0x400),"不跟随junction或symlink")
    raw=path.read_bytes();inputs[str(path)]=hashlib.sha256(raw).hexdigest()
    text=raw.decode("utf-8-sig")
    return [json.loads(line) for line in text.splitlines() if line] if jsonl else json.loads(text)


def relative(root,value):
    require(isinstance(value,str) and ":" not in value and "\\" not in value,"需要组内相对POSIX引用")
    path=PurePosixPath(value)
    require(not path.is_absolute() and ".." not in path.parts,"元数据引用不得越界")
    return root/Path(value)


def analyze_dataset(index_path,output,config=None):
    config=dict(DEFAULTS if config is None else config);index_path=Path(index_path).absolute();output=Path(output).absolute()
    require(not output.exists() and (WORKSPACE/"Saved") in output.parents,"只写全新workspace Saved诊断目录")
    inputs={};index=metadata(index_path,inputs)
    require(index.get("schema")=="futsalmot_pose17_location_pilot_dataset_v1","只接收发布的位置试产索引")
    root=index_path.parent;groups=[];heights=[]
    for row in index["groups"]:
        group_path=relative(root,row["group_ref"]);group=metadata(group_path,inputs);group_root=group_path.parent
        require(group["seed"]==row["seed"] and group["world_storage"]=="per_camera_native" and group["world_values_modified"] is False,"组身份或world原值契约不同")
        fps=group["source_motion"]["fps"];require(fps==index["fps"],"组与索引FPS不同")
        timeline=metadata(relative(group_root,group["timeline_ref"]),inputs,True);cameras={}
        for camera,refs in sorted(group["cameras"].items()):
            view=metadata(relative(group_root,refs["view_ref"]),inputs)
            images=metadata(relative(group_root,refs["pose2d_ref"]),inputs,True)
            worlds=metadata(relative(group_root,refs["pose3d_world_ref"]),inputs,True)
            cameras[camera]=dict(view=view,images=images,worlds=worlds)
            scale=640/max(view["intrinsics"]["width"],view["intrinsics"]["height"])
            heights.extend(o["bbox_visible_xywh"][3]*scale for frame in images for o in frame["objects"] if o["bbox_visible_xywh"])
        groups.append(analyze_group(row["seed"],row["split"],timeline,cameras,fps,config))
    cameras=[camera for group in groups for camera in group["per_camera"].values()]
    motions=[entity for group in groups for entity in group["motion"]["entities"].values()]
    comparisons=sum(e["yaw_and_torso_normalized_pose_step_rms"]["count"] for e in motions)
    repeats=sum(e["adjacent_near_repeat_steps"] for e in motions)
    contexts=[event for camera in cameras for event in camera["overlap_events"] if event["full_separated_pre_post_context"]]
    longest=max(group["nominal_sample_duration_seconds"] for group in groups)
    coverage=dict(independent_seed_groups=len(groups),source_time_samples=sum(g["source_frames"] for g in groups),
        camera_sequences=len(cameras),view_frames=sum(c["frames"] for c in cameras),
        source_nominal_seconds=sum(g["nominal_sample_duration_seconds"] for g in groups),
        view_nominal_seconds=sum(g["nominal_sample_duration_seconds"]*g["camera_sequences"] for g in groups),
        longest_continuous_nominal_seconds=longest,image_person_height_at_640_px=summary(heights),
        frames_with_bbox_overlap_proxy=sum(c["frames_with_any_bbox_overlap"] for c in cameras),
        source_frames_with_overlap_proxy_in_any_view=sum(len({p["frame_index"] for c in g["per_camera"].values() for p in c["overlap_pair_frames"]}) for g in groups),
        bbox_overlap_view_pair_frames=sum(len(c["overlap_pair_frames"]) for c in cameras),
        overlap_view_events=sum(len(c["overlap_events"]) for c in cameras),
        overlap_separation_with_full_context_view_events=len(contexts),same_team_context_view_events=sum(e["same_team"] for e in contexts),
        image_order_swap_view_events=sum(len(c["image_horizontal_order_swap_proxies"]) for c in cameras),
        same_team_order_swap_view_events=sum(e["same_team"] for c in cameras for e in c["image_horizontal_order_swap_proxies"]),
        world_near_pair_frames_reference_camera_only=sum(g["motion"]["world_proximity_pair_frames"] for g in groups),
        adjacent_near_repeated_pose_comparisons=repeats,adjacent_pose_comparisons=comparisons,
        adjacent_near_repeat_fraction=repeats/comparisons if comparisons else None,
        locomotion_proxy_steps=sum(e["locomotion_proxy_steps"] for e in motions),
        locomotion_with_near_repeated_pose_steps=sum(e["locomotion_with_near_repeated_pose_steps"] for e in motions))
    report=dict(schema=VERSION,candidate_only=True,config=config,coverage=coverage,groups=groups,
        selected_candidates=[g["selected_candidate"] for g in groups if g["selected_candidate"] is not None],
        selection_depends_on_model_outputs=False,physical_occlusion_GT_used=False,visibility_upgraded=False,identity_recovery_performance_measured=False,
        feasibility=dict(short_metadata_proxy_events_available=bool(contexts or coverage["image_order_swap_view_events"]),
            at_least_30_seconds_continuous_available=longest>=30,physical_occlusion_recovery_ground_truth_available=False,
            short_pilot_not_sufficient_for_long_term_identity_recovery_benchmark=True),
        inputs_text_sha256=inputs,media_pixels_read=False,media_hashes_read=False,gpu_used=False,ue_used=False,
        limitations=["overlap_nearness_order_swap_are_proxies_not_physical_occlusion_or_successful_identity_recovery",
            "hip_midpoint_is_root_motion_proxy_not_actor_transform","pose_similarity_is_normalized_kinematics_not_action_labels",
            "reference_camera_world_used_without_cross_camera_averaging","same_seed_views_not_independent_episodes",
            "event_sampling_not_claimed_novel_or_beneficial_without_budget_matched_experiments"])
    output.mkdir(parents=True)
    with (output/"report.json").open("x",encoding="utf-8") as handle:json.dump(report,handle,ensure_ascii=False,indent=2,allow_nan=False)
    with (output/"candidate_windows.jsonl").open("x",encoding="utf-8") as handle:
        for group in groups:
            for window in group["candidate_windows"]:handle.write(json.dumps(window,ensure_ascii=False,allow_nan=False)+"\n")
    lines=["# 交互窗口元数据覆盖候选","","仅GT元数据；bbox重叠/邻近/换序都是proxy，未评价模型或身份恢复。","",
        "|seed|split|连续秒数|640框高中位(px)|视角重叠帧|完整前后分离context事件|图像换序事件|", "|---|---|---:|---:|---:|---:|---:|"]
    for g in groups:
        cs=list(g["per_camera"].values())
        lines.append("|%d|%s|%.1f|%s|%d|%d|%d|"%(g["seed"],g["split"],g["nominal_sample_duration_seconds"],
            ",".join("%.2f"%c["height_at_640_px"]["median"] for c in cs),sum(c["frames_with_any_bbox_overlap"] for c in cs),
            sum(e["full_separated_pre_post_context"] for c in cs for e in c["overlap_events"]),sum(len(c["image_horizontal_order_swap_proxies"]) for c in cs)))
    lines.extend(["","独立seed=%d；源时间样本=%d；view图像=%d。最长连续%.1f秒，不足30秒。"%(len(groups),coverage["source_time_samples"],coverage["view_frames"],longest),
        "相邻根/航向/躯干归一姿态近重复比例=%.4f（阈值为躯干单位RMS %.3f），不等于动作类别重复率。"%(coverage["adjacent_near_repeat_fraction"] or 0,config["pose_near_repeat_torso_rms"]),
        "候选选择同seed全部机位共享源窗口，split保持不变；未重采集或改原发布。"])
    (output/"coverage.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset",type=Path,default=Path("G:/FutsalMOT_Dataset/v2_pose17_location_pilot_a01/dataset.json"))
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--window-frames",type=int,default=60);parser.add_argument("--stride-frames",type=int,default=10)
    args=parser.parse_args();config=dict(DEFAULTS,window_frames=args.window_frames,stride_frames=args.stride_frames)
    result=analyze_dataset(args.dataset,args.output,config)
    print(json.dumps(dict(output=str(args.output),coverage=result["coverage"],feasibility=result["feasibility"]),ensure_ascii=False))


if __name__=="__main__":main()
