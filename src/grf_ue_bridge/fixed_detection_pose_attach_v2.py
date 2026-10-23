"""固定检测输入的整帧17xy附件候选；不修改检测、不推断物理可见性。

几何规则只是预测质量代理，不能证明点准确或无遮挡。模型/图像依赖均延迟加载。
"""
from copy import deepcopy
import hashlib
import math

VERSION='futsalmot_fixed_detection_pose_attachment_v1'
STRATEGY='full_frame_640_letterbox_one_to_one_bbox_iou'
SEQUENCES=('SNMOT-151','SNMOT-152','SNMOT-153')
LIMBS=tuple(range(5,17))


def require(ok,message):
    if not ok:raise ValueError(message)


def finite(values,length):
    return isinstance(values,(list,tuple)) and len(values)==length and all(
        type(v) in (int,float) and math.isfinite(v) for v in values)


def is_sha(value):
    return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)


def candidate_configuration():
    """可登记的显式候选值；不是已经冻结或经开发结果挑选的参数。"""
    return dict(schema='futsalmot_fixed_detection_pose_configuration_v1',strategy=STRATEGY,imgsz=640,
        pose_confidence_threshold=.001,pose_nms_iou=.7,max_pose_detections=100,min_matched_iou=.3,
        bbox_margin_fraction=.25,minimum_geometry_limb_joints=6,joint_indices=list(LIMBS),
        minimum_bbox_height_input_px=0.,minimum_limb_spread_bbox_diagonal_fraction=0.,
        quality_gate_status='candidate_not_frozen',physical_visibility='unknown')


def validate_configuration(config):
    require(config.get('schema')=='futsalmot_fixed_detection_pose_configuration_v1'
        and config.get('strategy')==STRATEGY and config.get('imgsz')==640
        and config.get('joint_indices')==list(LIMBS) and config.get('physical_visibility')=='unknown',
        '仅支持显式整帧640/十二肢体候选，不接受GT crop/visibility')
    for key in ('pose_confidence_threshold','pose_nms_iou','min_matched_iou'):
        require(type(config.get(key)) in (int,float) and math.isfinite(config[key]) and 0<config[key]<=1,
                '无效阈值: '+key)
    for key in ('bbox_margin_fraction','minimum_bbox_height_input_px','minimum_limb_spread_bbox_diagonal_fraction'):
        require(type(config.get(key)) in (int,float) and math.isfinite(config[key]) and config[key]>=0,
                '无效几何门槛: '+key)
    require(config['bbox_margin_fraction']<=1 and config['minimum_limb_spread_bbox_diagonal_fraction']<=1,
            '拒绝无限放宽几何规则')
    require(type(config.get('max_pose_detections')) is int and 1<=config['max_pose_detections']<=300,
            'Pose检测上限必须1..300')
    require(type(config.get('minimum_geometry_limb_joints')) is int
        and 1<=config['minimum_geometry_limb_joints']<=12,'肢体有效点门槛必须1..12')


def validate_fixed_frame(frame):
    require(frame.get('schema')=='futsalmot_fixed_detection_frame_v1'
        and frame.get('sequence_id') in SEQUENCES and frame.get('camera_id')=='broadcast'
        and frame.get('coordinate_space')=='image_pixels','不接受非既定开发域或GT检测替代品')
    index=frame.get('frame_index');number=frame.get('frame_number')
    require(type(index) is int and type(number) is int and 0<=index<150 and number==index+1
        and frame.get('frame_key')==f"{frame['sequence_id']}:{number:06d}",'帧身份错配')
    require(type(frame.get('width')) is int and type(frame.get('height')) is int
        and min(frame['width'],frame['height'])>0,'图像尺寸无效')
    for i,det in enumerate(frame['detections']):
        require(type(det.get('source_index')) is int and det['source_index']==i
            and type(det.get('class_id')) is int and det['class_id']==0
            and det.get('detection_id')==f"{frame['frame_key']}:{i:06d}",'固定检测键/顺序/类别错配')
        require(finite(det.get('bbox_xywh_px'),4) and min(det['bbox_xywh_px'][2:])>0
            and type(det.get('score')) in (int,float) and math.isfinite(det['score']) and 0<=det['score']<=1,
            '固定检测数值无效')
        x,y,w,h=det['bbox_xywh_px']
        require(0<=x<x+w<=frame['width'] and 0<=y<y+h<=frame['height'],'固定框不是原图像素框')


def bbox_iou(a,b):
    x=max(a[0],b[0]);y=max(a[1],b[1]);r=min(a[0]+a[2],b[0]+b[2]);d=min(a[1]+a[3],b[1]+b[3])
    intersection=max(0.,r-x)*max(0.,d-y)
    return intersection/(a[2]*a[3]+b[2]*b[3]-intersection) if intersection>0 else 0.


def match_predictions(detections,predictions,minimum_iou):
    """按IoU降序贪心一对一；同分按固定检测ID/模型输出序号，完全不用GT。"""
    pairs=[];rejected=[]
    for pi,pred in enumerate(predictions):
        require(isinstance(pred.get('xy_px'),(list,tuple)) and len(pred['xy_px'])==17
            and all(isinstance(p,(list,tuple)) and len(p)==2 for p in pred['xy_px']),'Pose输出不是17xy')
        if not finite(pred.get('bbox_xywh_px'),4) or min(pred['bbox_xywh_px'][2:])<=0:
            rejected.append(pi);continue
        require(type(pred.get('score')) in (int,float) and math.isfinite(pred['score']) and 0<=pred['score']<=1,
                '模型person score无效，不接受关节confidence替代')
        for di,det in enumerate(detections):
            iou=bbox_iou(det['bbox_xywh_px'],pred['bbox_xywh_px'])
            if iou>=minimum_iou:pairs.append((-iou,det['detection_id'],pi,di))
    selected={};used=set()
    for negative_iou,_,pi,di in sorted(pairs):
        if di not in selected and pi not in used:
            selected[di]=(pi,-negative_iou);used.add(pi)
    return selected,rejected


def geometry_mask(points,box,image_wh,margin):
    """仅数值/画内/固定框相对几何；false点置零，不能当visible。"""
    width,height=image_wh;x,y,w,h=box;xy=[];valid=[];reasons=[];raw=[]
    for point in points:
        numeric=finite(point,2)
        inside=numeric and 0<=point[0]<width and 0<=point[1]<height
        relative=inside and x-margin*w<=point[0]<=x+(1+margin)*w and y-margin*h<=point[1]<=y+(1+margin)*h
        ok=bool(relative);raw.append(list(point) if numeric else None)
        xy.append(list(point) if ok else [0.,0.]);valid.append(ok)
        reasons.append(None if ok else 'nonfinite_prediction' if not numeric else
                       'outside_image' if not inside else 'outside_fixed_bbox_margin')
    return xy,valid,reasons,raw


def attach_frame(frame,predictions,*,model_sha256,fixed_detection_set_sha256,config):
    """每个固定检测都输出附件；缺失/差匹配明示false，既有bbox与ID逐值保留。"""
    validate_configuration(config);validate_fixed_frame(frame)
    require(is_sha(model_sha256) and is_sha(fixed_detection_set_sha256),'缺模型/完整固定cache SHA')
    selected,rejected=match_predictions(frame['detections'],predictions,config['min_matched_iou'])
    from .pose_location_dataset_v2 import letterbox_geometry
    geometry=letterbox_geometry(frame['width'],frame['height'],640)
    attachments=[]
    for di,det in enumerate(frame['detections']):
        match=selected.get(di);xy=[[0.,0.] for _ in range(17)];valid=[False]*17;raw=[None]*17
        reasons=['no_model_bbox_match']*17;global_reason='no_model_bbox_match';before=[False]*17
        pred=None;iou=None;pi=None
        if match is not None:
            pi,iou=match;pred=predictions[pi]
            xy,valid,reasons,raw=geometry_mask(pred['xy_px'],det['bbox_xywh_px'],
                (frame['width'],frame['height']),config['bbox_margin_fraction'])
            before=list(valid);limb_points=[xy[j] for j in LIMBS if valid[j]]
            h=det['bbox_xywh_px'][3]*geometry['ratio']
            spread=max((math.dist(a,b) for a in limb_points for b in limb_points),default=0.)
            diagonal=math.hypot(*det['bbox_xywh_px'][2:])
            global_reason=('too_few_geometry_valid_limbs' if len(limb_points)<config['minimum_geometry_limb_joints']
                else 'bbox_too_small_at_model_input' if h<config['minimum_bbox_height_input_px']
                else 'degenerate_limb_spread' if spread<config['minimum_limb_spread_bbox_diagonal_fraction']*diagonal
                else None)
            if global_reason is not None:
                xy=[[0.,0.] for _ in range(17)];valid=[False]*17
        row=dict(schema=VERSION,detection_id=det['detection_id'],source_index=det['source_index'],
            sequence_id=frame['sequence_id'],camera_id=frame['camera_id'],frame_index=frame['frame_index'],
            frame_number=frame['frame_number'],frame_key=frame['frame_key'],image_wh=[frame['width'],frame['height']],
            bbox_xywh_px=list(det['bbox_xywh_px']),fixed_detection_score=det['score'],class_id=det['class_id'],
            source='model_prediction_xy',coordinate_space='image_pixels',validity_method='finite_bbox_relative_geometry',
            model_sha256=model_sha256,fixed_detection_set_sha256=fixed_detection_set_sha256,
            xy_px=xy,geometry_valid=valid,prediction_valid=global_reason is None,
            prediction_rejection=global_reason,raw_prediction_xy_px=raw,
            geometry_mask_before_global_gate=before,point_geometry_rejection=reasons,
            model_prediction_index=pi,model_prediction_bbox_xywh_px=deepcopy(pred['bbox_xywh_px']) if pred else None,
            model_person_score=pred['score'] if pred else None,matched_bbox_iou=iou,
            physical_visibility='unknown',visibility_state=['unknown']*17,joint_confidence=None,
            transform=dict(strategy=STRATEGY,original_to_input_matrix=geometry['matrix'],
                           scale=geometry['ratio'],left=geometry['left'],top=geometry['top'],input_wh=[640,640]))
        attachments.append(row)
    return dict(schema='futsalmot_fixed_detection_pose_frame_v1',sequence_id=frame['sequence_id'],
        camera_id=frame['camera_id'],frame_key=frame['frame_key'],frame_index=frame['frame_index'],
        frame_number=frame['frame_number'],fixed_detection_count=len(frame['detections']),
        pose_prediction_count=len(predictions),rejected_model_bbox_indices=rejected,
        attachments=attachments)


def validate_attachment_frame(fixed,attached,*,model_sha256,fixed_detection_set_sha256,config):
    """附件读取门禁：缺Pose也不能丢失固定检测、换框或恢复为假有效点。"""
    validate_fixed_frame(fixed);validate_configuration(config)
    require(attached.get('schema')=='futsalmot_fixed_detection_pose_frame_v1'
        and attached.get('fixed_detection_count')==len(fixed['detections'])
        and len(attached.get('attachments',[]))==len(fixed['detections']),'附件改变固定检测数量')
    for key in ('sequence_id','camera_id','frame_key','frame_index','frame_number'):
        require(attached.get(key)==fixed[key],'附件帧身份错配: '+key)
    for det,row in zip(fixed['detections'],attached['attachments']):
        for key in ('detection_id','source_index','class_id','bbox_xywh_px'):
            require(row.get(key)==det[key],'附件修改原固定检测: '+key)
        for key in ('sequence_id','camera_id','frame_key','frame_index','frame_number'):
            require(row.get(key)==fixed[key],'附件记录帧身份错配: '+key)
        require(row.get('model_sha256')==model_sha256 and row.get('fixed_detection_set_sha256')==fixed_detection_set_sha256
            and row.get('source')=='model_prediction_xy' and row.get('coordinate_space')=='image_pixels'
            and row.get('validity_method')=='finite_bbox_relative_geometry','附件模型/缓存/语义错配')
        xy=row.get('xy_px');valid=row.get('geometry_valid')
        require(isinstance(xy,list) and len(xy)==17 and all(finite(p,2) for p in xy)
            and isinstance(valid,list) and len(valid)==17 and all(type(v) is bool for v in valid)
            and type(row.get('prediction_valid')) is bool,'附件须有限17xy和独立bool mask')
        require(row.get('physical_visibility')=='unknown' and row.get('visibility_state')==['unknown']*17
            and row.get('joint_confidence') is None,'xy头不能生成伪visible或关节confidence')
        _,recomputed,_,_=geometry_mask(xy,det['bbox_xywh_px'],(fixed['width'],fixed['height']),config['bbox_margin_fraction'])
        require(all(not ok or recomputed[j] for j,ok in enumerate(valid)),'true几何mask不满足冻结画内/扩框规则')
        require(all(ok or xy[j]==[0.,0.] for j,ok in enumerate(valid)),'false点必须0占位')
        if not row['prediction_valid']:
            require(not any(valid),'无效人仍含可用于关联的true点')
        else:
            require(sum(valid[j] for j in LIMBS)>=config['minimum_geometry_limb_joints']
                and finite(row.get('model_prediction_bbox_xywh_px'),4)
                and type(row.get('matched_bbox_iou')) in (int,float)
                and row['matched_bbox_iou']==bbox_iou(det['bbox_xywh_px'],row['model_prediction_bbox_xywh_px'])
                and row['matched_bbox_iou']>=config['min_matched_iou'],'有效附件缺匹配/有效肢体证据')
            from .pose_location_dataset_v2 import letterbox_geometry
            ratio=letterbox_geometry(fixed['width'],fixed['height'],640)['ratio']
            points=[xy[j] for j in LIMBS if valid[j]]
            spread=max((math.dist(a,b) for a in points for b in points),default=0.)
            require(det['bbox_xywh_px'][3]*ratio>=config['minimum_bbox_height_input_px']
                and spread>=config['minimum_limb_spread_bbox_diagonal_fraction']*math.hypot(*det['bbox_xywh_px'][2:]),
                '附件绕过显式尺度/退化门槛')


def original_prediction(row,image_wh):
    """本机普通Pose NMS行[xyxy,score,class,17xy]，逆中心LetterBox，不裁关键点。"""
    require(len(row)==40 and row[5]==0,'非单类17xy NMS布局')
    from .pose_location_dataset_v2 import letterbox_geometry
    width,height=image_wh;geometry=letterbox_geometry(width,height,640)
    ratio=geometry['ratio'];left=geometry['left'];top=geometry['top']
    x1,y1,x2,y2=row[:4]
    if not finite([x1,y1,x2,y2],4):return None
    x1=max(0.,min(float(width),(x1-left)/ratio));x2=max(0.,min(float(width),(x2-left)/ratio))
    y1=max(0.,min(float(height),(y1-top)/ratio));y2=max(0.,min(float(height),(y2-top)/ratio))
    if x2<=x1 or y2<=y1:return None
    return dict(bbox_xywh_px=[x1,y1,x2-x1,y2-y1],score=row[4],
        xy_px=[[(row[6+2*j]-left)/ratio,(row[7+2*j]-top)/ratio] for j in range(17)])


def infer_image(image,model,device,config):
    """仅调用者显式执行时推理；纯整帧，不读取GT、不用固定框裁图。"""
    validate_configuration(config)
    require(device in ('cpu','cuda:0'),'禁止自动选设备')
    import cv2
    import numpy as np
    import torch
    import ultralytics
    from ultralytics.utils.nms import non_max_suppression
    from .pose_location_dataset_v2 import letterbox_geometry
    require(ultralytics.__version__=='8.4.60','仅验证Ultralytics8.4.60')
    require(isinstance(image,np.ndarray) and image.dtype==np.uint8 and image.ndim==3 and image.shape[2]==3,
            '图像必须原始BGR uint8三通道')
    height,width=image.shape[:2];geometry=letterbox_geometry(width,height,640)
    resized=cv2.resize(image,geometry['resize_wh'],interpolation=cv2.INTER_LINEAR)
    padded=cv2.copyMakeBorder(resized,geometry['top'],geometry['bottom'],geometry['left'],geometry['right'],
                             cv2.BORDER_CONSTANT,value=(114,114,114))
    tensor=torch.from_numpy(np.ascontiguousarray(padded[:,:,::-1].transpose(2,0,1))).unsqueeze(0).to(device).float()/255.
    with torch.inference_mode():
        decoded=model(tensor)
        rows=non_max_suppression(decoded,conf_thres=config['pose_confidence_threshold'],iou_thres=config['pose_nms_iou'],
            classes=[0],agnostic=False,multi_label=False,max_det=config['max_pose_detections'],nc=1,max_time_img=10.)[0]
    predictions=[original_prediction(row,(width,height)) for row in rows.detach().cpu().tolist()]
    return [p for p in predictions if p is not None]


def load_checkpoint(path,expected_sha256,member='ema'):
    """读取明确SHA的可信本地checkpoint，不重建YAML、不隐式选择model/best。"""
    from pathlib import Path
    import torch
    from . import pose_xy_head_transfer as transfer
    require(member in ('model','ema') and is_sha(expected_sha256),'模型成员/SHA无效')
    require(hashlib.sha256(Path(path).read_bytes()).hexdigest()==expected_sha256,'实际checkpoint SHA不符')
    checkpoint=torch.load(path,map_location='cpu',weights_only=False)
    require(isinstance(checkpoint,dict) and isinstance(checkpoint.get(member),torch.nn.Module),'缺显式实际模型成员')
    if checkpoint.get('schema','').startswith('futsalmot_pose_location_control_checkpoint'):
        require(member=='ema' and type(checkpoint.get('completed_updates')) is int
            and checkpoint['completed_updates']==checkpoint.get('planned_updates')==checkpoint.get('ema_updates'),
            '正式位置对照只接受完成预算的EMA')
    model=checkpoint[member].float().eval();head=transfer._validate(model)
    require(tuple(head.kpt_shape)==(17,2) and head.pose_xy_transfer.get('mode')=='exact_full_conv_then_select'
        and head.nc==1,'只支持实际严格xy普通Pose单类模型，不允许重新随机建塔')
    metadata={k:checkpoint[k] for k in ('schema','arm','seed','completed_updates','planned_updates','bindings') if k in checkpoint}
    return model,dict(path=str(Path(path).resolve()),sha256=expected_sha256,member=member,
                      checkpoint_metadata=metadata,pose_xy_transfer=deepcopy(head.pose_xy_transfer))
