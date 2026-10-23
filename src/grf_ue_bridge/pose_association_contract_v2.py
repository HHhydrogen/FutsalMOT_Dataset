"""固定检测输入的Pose关联成本候选；仅CPU接口，不是已集成Tracker。"""
import math


def _validate(record):
    if record.get('coordinate_space')!='image_pixels' or record.get('source')!='model_prediction_xy':
        raise ValueError('关联只接收图像像素空间模型预测，不接收GT Pose')
    if record.get('validity_method')!='finite_bbox_relative_geometry':
        raise ValueError('仅允许预先固定的预测几何有效性，不可借用GT/visibility mask')
    if not record.get('model_sha256') or not record.get('fixed_detection_set_sha256'):
        raise ValueError('缺模型/固定检测输入身份')
    points=record.get('xy_px');valid=record.get('geometry_valid')
    if len(points or [])!=17 or len(valid or [])!=17 or any(type(v) is not bool for v in valid):
        raise ValueError('需要COCO17预测xy和独立bool几何mask')
    if any(len(p)!=2 or any(not isinstance(v,(int,float)) or isinstance(v,bool) or not math.isfinite(v) for v in p) for p in points):
        raise ValueError('预测点须为有限xy')
    box=record.get('bbox_xywh_px')
    if len(box or [])!=4 or any(not isinstance(v,(int,float)) or isinstance(v,bool) or not math.isfinite(v) for v in box) or min(box[2:])<=0:
        raise ValueError('固定检测bbox无效')
    if type(record.get('frame_index')) is not int or not record.get('sequence_id') or not record.get('camera_id'):
        raise ValueError('缺帧/序列/相机身份')


def pose_pair_cost_candidate(track,detection,kalman_center_delta_xy,*,joint_indices=tuple(range(5,17)),min_joints=6,sigma=.1,max_frame_gap=30):
    """Kalman预测中心位移补偿后1-平均OKS式成本；参数尚未冻结。"""
    _validate(track);_validate(detection)
    for key in ('sequence_id','camera_id','model_sha256','fixed_detection_set_sha256'):
        if track[key]!=detection[key]:raise ValueError('Pose配对身份错配：'+key)
    gap=detection['frame_index']-track['frame_index']
    if not 1<=gap<=max_frame_gap:raise ValueError('帧顺序或间距不合规')
    if len(kalman_center_delta_xy)!=2 or any(not math.isfinite(v) for v in kalman_center_delta_xy):
        raise ValueError('缺有效Kalman中心位移')
    if not joint_indices or len(set(joint_indices))!=len(joint_indices) or any(type(j) is not int or not 0<=j<17 for j in joint_indices):
        raise ValueError('关节子集无效')
    if type(min_joints) is not int or min_joints<1 or min_joints>len(joint_indices) or not math.isfinite(sigma) or sigma<=0:
        raise ValueError('成本参数无效')
    joints=[j for j in joint_indices if track['geometry_valid'][j] and detection['geometry_valid'][j]]
    if len(joints)<min_joints:
        return {'pose_cost':None,'reason':'insufficient_predicted_geometry','fallback':'bbox_only','joint_count':len(joints),'candidate_only':True}
    a=track['bbox_xywh_px'];b=detection['bbox_xywh_px'];area=(a[2]*a[3]+b[2]*b[3])/2
    dx,dy=kalman_center_delta_xy
    similarities=[]
    for j in joints:
        p=track['xy_px'][j];q=detection['xy_px'][j]
        d2=(p[0]+dx-q[0])**2+(p[1]+dy-q[1])**2
        similarities.append(math.exp(-d2/(2*(2*sigma)**2*area)))
    return {'pose_cost':1-sum(similarities)/len(similarities),'joint_count':len(joints),
            'reason':'predicted_geometry_only','candidate_only':True,'used_in_tracker':False}
