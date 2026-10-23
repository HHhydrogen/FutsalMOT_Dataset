"""只用已有GT元数据的交互窗口候选；重叠/邻近/换序均为proxy。"""
from itertools import combinations
import math

import numpy as np

VERSION="gt_metadata_interaction_windows_candidate_v1"
DEFAULTS=dict(window_frames=60,stride_frames=10,overlap_smaller_box_fraction=.2,
    minimum_overlap_run_frames=2,separation_context_frames=10,world_near_distance_m=1.5,
    order_swap_vertical_height_fraction=.75,order_swap_distance_height_fraction=1.5,
    pose_near_repeat_torso_rms=.01,root_step_for_locomotion_proxy_m=.05,pose_return_lag_frames=[5,10,20,30])


def require(ok,message):
    if not ok:raise ValueError(message)


def summary(values):
    a=np.asarray(values,dtype=float)
    require(np.isfinite(a).all(),"统计量非有限")
    if not len(a):return dict(count=0,mean=None,min=None,p05=None,p25=None,median=None,p75=None,p95=None,max=None)
    return dict(count=len(a),mean=float(a.mean()),min=float(a.min()),p05=float(np.quantile(a,.05)),p25=float(np.quantile(a,.25)),
        median=float(np.median(a)),p75=float(np.quantile(a,.75)),p95=float(np.quantile(a,.95)),max=float(a.max()))


def contiguous_runs(indices):
    out=[]
    for index in sorted(indices):
        if out and out[-1][1]==index:out[-1][1]=index+1
        else:out.append([index,index+1])
    return out


def overlap_fraction(a,b):
    if a is None or b is None:return 0.
    x,y=max(a[0],b[0]),max(a[1],b[1]);r,d=min(a[0]+a[2],b[0]+b[2]),min(a[1]+a[3],b[1]+b[3])
    area=max(0.,r-x)*max(0.,d-y)
    return area/min(a[2]*a[3],b[2]*b[3]) if min(a[2],a[3],b[2],b[3])>0 else 0.


def pose_coordinates(points):
    """骨盆中点/髋横轴/肩盆长度归一；不是Actor根骨或动作类别。"""
    p=np.asarray(points,dtype=float);require(p.shape==(17,3) and np.isfinite(p).all(),"world17无效")
    root=(p[11]+p[12])/2;right=p[12]-p[11];right[2]=0
    length=np.linalg.norm(right);torso=np.linalg.norm((p[5]+p[6])/2-root)
    if length<1e-8 or torso<1e-8:return root,None,None
    right/=length;forward=np.array([right[1],-right[0],0.])
    centered=p[5:]-root
    aligned=np.stack((centered@forward,centered@right,centered[:,2]),axis=1)
    return root,centered,aligned/torso


def image_root(obj):
    points=obj["uv_px"];valid=obj["location_valid"]
    if valid[11] and valid[12]:return [(points[11][k]+points[12][k])/2 for k in range(2)],"projected_hip_midpoint"
    if obj["bbox_visible_xywh"]:
        x,y,w,h=obj["bbox_visible_xywh"];return [x+w/2,y+h/2],"visible_bbox_center_fallback"
    return None,"missing"


def analyze_camera(images,worlds,view,source_frames,fps,config):
    count=len(images);require(count==len(worlds)==len(source_frames),"相机帧数不齐")
    ids=sorted(o["entity_id"] for o in worlds[0]["objects"]);pairs=list(combinations(ids,2))
    lookup=[];heights=[];centers=[];root_fallbacks=0;visibility={};positive=0
    for frame,(image,world) in enumerate(zip(images,worlds)):
        require(image["frame_index"]==world["frame_index"]==frame and world["source_frame_index"]==source_frames[frame],"相对/源帧不连续或错配")
        objects={o["entity_id"]:o for o in image["objects"]}
        require(len(objects)==len(image["objects"])==len(ids) and set(objects)==set(ids),"人物身份缺失/重复")
        lookup.append(objects);frame_centers={}
        for entity,obj in objects.items():
            box=obj["bbox_visible_xywh"]
            if box is not None:
                require(len(box)==4 and all(math.isfinite(v) for v in box) and min(box[2:])>0,"bbox无效")
                heights.append(box[3]*640/max(view["intrinsics"]["width"],view["intrinsics"]["height"]));positive+=1
            center,method=image_root(obj);frame_centers[entity]=center
            root_fallbacks+=method=="visible_bbox_center_fallback"
            for state in obj["visibility_state"]:visibility[state]=visibility.get(state,0)+1
        centers.append(frame_centers)
    pair_frames=[];overlap_events=[];swaps=[]
    for a,b in pairs:
        hit=[]
        for frame in range(count):
            boxes=[lookup[frame][e]["bbox_visible_xywh"] for e in (a,b)]
            value=overlap_fraction(*boxes)
            if value>=config["overlap_smaller_box_fraction"]:
                hit.append(frame);pair_frames.append(dict(frame_index=frame,pair=[a,b],same_team=a[0]==b[0],smaller_box_overlap_fraction=value))
            if frame==0 or any(box is None for box in boxes):continue
            p,q=centers[frame][a],centers[frame][b];previous_p,previous_q=centers[frame-1][a],centers[frame-1][b]
            if any(c is None for c in (p,q,previous_p,previous_q)):continue
            h=(boxes[0][3]+boxes[1][3])/2
            if ((previous_p[0]-previous_q[0])*(p[0]-q[0])<0
                    and abs(p[1]-q[1])<=config["order_swap_vertical_height_fraction"]*h
                    and math.dist(p,q)<=config["order_swap_distance_height_fraction"]*h):
                swaps.append(dict(frame_index=frame,source_frame=source_frames[frame],pair=[a,b],same_team=a[0]==b[0]))
        hit_set=set(hit);context=config["separation_context_frames"]
        for start,end in contiguous_runs(hit):
            if end-start<config["minimum_overlap_run_frames"]:continue
            full_context=start>=context and end+context<=count
            separated=(full_context and not any(i in hit_set for i in range(start-context,start))
                       and not any(i in hit_set for i in range(end,end+context)))
            overlap_events.append(dict(pair=[a,b],same_team=a[0]==b[0],start_frame=start,end_frame_exclusive=end,
                source_frame_range=[source_frames[start],source_frames[end-1]+1],nominal_duration_seconds=(end-start)/fps,
                full_separated_pre_post_context=bool(separated),context_frames=context,
                boundary_censored=not full_context))
    heights=np.asarray(heights)
    return dict(frames=count,positive_bbox_people=positive,height_at_640_px=summary(heights),
        scale_counts=dict(lt32=int((heights<32).sum()),from32_to64=int(((heights>=32)&(heights<64)).sum()),ge64=int((heights>=64).sum())),
        visibility_state_counts=visibility,root_center_bbox_fallback_count=root_fallbacks,
        overlap_pair_frames=pair_frames,overlap_events=overlap_events,image_horizontal_order_swap_proxies=swaps,
        frames_with_any_bbox_overlap=len({r["frame_index"] for r in pair_frames}))


def motion_coverage(worlds,source_frames,fps,config):
    """只选明确参考相机计算源时序覆盖，不平均四视角world或重复计数。"""
    entities={};near_pairs={};count=len(worlds)
    for frame in range(count):
        objects=worlds[frame]["objects"]
        for obj in objects:
            root,centered,pose=pose_coordinates(obj["world_m"])
            entities.setdefault(obj["entity_id"],[]).append((root,centered,pose))
        for a,b in combinations(objects,2):
            ar=(np.asarray(a["world_m"][11])+a["world_m"][12])/2
            br=(np.asarray(b["world_m"][11])+b["world_m"][12])/2
            distance=float(np.linalg.norm((ar-br)[:2]))
            if distance<=config["world_near_distance_m"]:
                near_pairs.setdefault(tuple(sorted((a["entity_id"],b["entity_id"]))),[]).append(frame)
    reports={}
    for entity,data in sorted(entities.items()):
        require(len(data)==count,"参考相机world人物时间覆盖不完整")
        roots=np.stack([d[0] for d in data]);steps=np.linalg.norm(np.diff(roots[:,:2],axis=0),axis=1)
        raw=[];aligned=[];near_repeat_indices=[];lags={}
        for index in range(1,count):
            if data[index][1] is not None and data[index-1][1] is not None:
                raw.append(float(np.sqrt(np.mean((data[index][1]-data[index-1][1])**2))))
                difference=float(np.sqrt(np.mean((data[index][2]-data[index-1][2])**2)))
                aligned.append(difference)
                if difference<=config["pose_near_repeat_torso_rms"]:near_repeat_indices.append(index)
        for lag in config["pose_return_lag_frames"]:
            distances=[float(np.sqrt(np.mean((data[i][2]-data[i-lag][2])**2))) for i in range(lag,count)
                       if data[i][2] is not None and data[i-lag][2] is not None]
            lags[str(lag)]=dict(comparisons=len(distances),near_repeat_fraction=sum(d<=config["pose_near_repeat_torso_rms"] for d in distances)/len(distances) if distances else None,
                               rms_torso_units=summary(distances))
        runs=contiguous_runs(near_repeat_indices)
        moving=[i+1 for i,value in enumerate(steps) if value>=config["root_step_for_locomotion_proxy_m"]]
        moving_repeats=len(set(moving)&set(near_repeat_indices))
        reports[entity]=dict(root_definition="midpoint_of_world_hip_keypoints_proxy_not_actor_root",
            path_length_ground_m=float(steps.sum()),net_displacement_ground_m=float(np.linalg.norm(roots[-1,:2]-roots[0,:2])),
            step_ground_m=summary(steps),speed_ground_mps=summary(steps*fps),
            root_centered_pose_step_rms_m=summary(raw),yaw_and_torso_normalized_pose_step_rms=summary(aligned),
            adjacent_near_repeat_fraction=len(near_repeat_indices)/len(aligned) if aligned else None,
            adjacent_near_repeat_steps=len(near_repeat_indices),
            locomotion_proxy_steps=len(moving),locomotion_with_near_repeated_pose_steps=moving_repeats,
            locomotion_with_near_repeated_pose_fraction=moving_repeats/len(moving) if moving else None,
            longest_adjacent_near_repeat_run_frames=max((end-start+1 for start,end in runs),default=0),
            pose_returns_at_lags=lags)
    proximity=[dict(pair=list(pair),same_team=pair[0][0]==pair[1][0],start_frame=start,end_frame_exclusive=end,
                    source_frame_range=[source_frames[start],source_frames[end-1]+1])
               for pair,indices in sorted(near_pairs.items()) for start,end in contiguous_runs(indices)
               if end-start>=config["minimum_overlap_run_frames"]]
    return dict(reference_camera_only=True,entities=reports,world_proximity_proxy_events=proximity,
        world_proximity_pair_frames=sum(len(indices) for indices in near_pairs.values()))


def analyze_group(seed,split,timeline,cameras,fps,config=None):
    config=dict(DEFAULTS if config is None else config);count=len(timeline)
    require(type(fps) in (float,int) and math.isfinite(fps) and fps>0 and count>1,"帧率/时间长度无效")
    require(type(config["window_frames"]) is int and config["window_frames"]>=2
        and type(config["stride_frames"]) is int and config["stride_frames"]>=1,"候选窗口长度/步幅无效")
    require(0<config["overlap_smaller_box_fraction"]<=1
        and all(type(config[k]) is int and config[k]>=1 for k in ("minimum_overlap_run_frames","separation_context_frames"))
        and all(math.isfinite(config[k]) and config[k]>0 for k in ("world_near_distance_m","order_swap_vertical_height_fraction",
            "order_swap_distance_height_fraction","pose_near_repeat_torso_rms","root_step_for_locomotion_proxy_m")),"事件候选阈值无效")
    require(all(type(v) is int and v>0 for v in config["pose_return_lag_frames"]),"姿态回返lag必须为正整数")
    source=[t["source_frame_index"] for t in timeline]
    require([t["frame_index"] for t in timeline]==list(range(count)) and source==list(range(source[0],source[0]+count)),"源窗口必须连续，不拼接或补静止帧")
    per_camera={name:analyze_camera(data["images"],data["worlds"],data["view"],source,fps,config) for name,data in sorted(cameras.items())}
    reference=sorted(cameras)[0];motion=motion_coverage(cameras[reference]["worlds"],source,fps,config)
    windows=[];length=config["window_frames"];context=config["separation_context_frames"]
    for start in range(0,max(0,count-length+1),config["stride_frames"]):
        end=start+length;events=[];swaps=[];overlap_frames=0
        for name,data in per_camera.items():
            events.extend(dict(e,camera_id=name) for e in data["overlap_events"] if e["full_separated_pre_post_context"]
                          and start<=e["start_frame"]-context and e["end_frame_exclusive"]+context<=end)
            swaps.extend(dict(e,camera_id=name) for e in data["image_horizontal_order_swap_proxies"] if start<=e["frame_index"]<end)
            overlap_frames+=sum(start<=e["frame_index"]<end for e in data["overlap_pair_frames"])
        score=[sum(e["same_team"] for e in events),sum(e["same_team"] for e in swaps),len(events),len(swaps),overlap_frames]
        windows.append(dict(seed=seed,split=split,relative_frame_range=[start,end],source_frame_range=[source[start],source[end-1]+1],
            nominal_seconds=length/fps,cameras_shared_same_source_window=sorted(cameras),selection_rank_tuple=score,
            context_complete_overlap_separation_view_events=len(events),same_team_context_complete_view_events=score[0],
            image_order_swap_view_events=len(swaps),overlap_view_pair_frames=overlap_frames,
            selection_evidence="GT_metadata_proxies_only_no_model_results"))
    chosen=max(windows,key=lambda w:(w["selection_rank_tuple"],-w["relative_frame_range"][0])) if windows else None
    return dict(seed=seed,split=split,source_frame_range=[source[0],source[-1]+1],source_frames=count,fps=fps,
        nominal_sample_duration_seconds=count/fps,first_to_last_timestamp_span_seconds=(source[-1]-source[0])/fps,
        camera_sequences=len(cameras),world_motion_reference_camera=reference,per_camera=per_camera,motion=motion,
        candidate_windows=windows,selected_candidate=chosen,
        selection_rule="lexicographic_same_team_context_overlap_then_same_team_order_swap_then_all_context_overlap_then_all_swaps_then_overlap_pairframes_tie_earlier",
        candidate_only=True,physical_occlusion_measured=False,identity_recovery_success_measured=False)
