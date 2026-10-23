"""候选真实/合成位置监督桥接：位置有效性与可见性完全分开。

本模块只做纯CPU目标转换/损失桥接，不读图；独立pose_location_dataset_v2
已调用本桥接完成JSONL图像/LetterBox/严格xy Trainer的toy更新与续载。
正式研究训练、COCO评价和crowd-ignore仍需独立门槛。
"""
from copy import deepcopy
import math

VERSION='separated_pose_location_supervision_candidate_v1'
_LOSS_CLASS=None


def _number(value):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
        raise ValueError('要求有限数值')
    return float(value)


def _box(annotation,width,height):
    width=_number(width);height=_number(height)
    if width<=0 or height<=0:raise ValueError('图像尺寸必须大于0')
    box=annotation.get('bbox')
    if not isinstance(box,(list,tuple)) or len(box)!=4:raise ValueError('需要完整person bbox')
    x,y,w,h=map(_number,box)
    if w<=0 or h<=0:raise ValueError('bbox为空')
    left=max(0.,x);top=max(0.,y);right=min(width,x+w);bottom=min(height,y+h)
    if right<=left or bottom<=top:raise ValueError('bbox与图像无交集')
    return [left,top,right-left,bottom-top]


def coco_person_location(annotation,width,height):
    """保留缺点/全缺点person；COCO v0不是(0,0)位置真值。"""
    if annotation.get('category_id')!=1:raise ValueError('只处理COCO person category=1')
    if annotation.get('iscrowd',0)!=0:raise ValueError('crowd需要独立ignore区域实现，不静默删除')
    box=_box(annotation,width,height)
    raw=annotation.get('keypoints',[0]*51)
    if not isinstance(raw,(list,tuple)) or len(raw)!=51:raise ValueError('需要COCO17原始51值')
    xy=[];valid=[];visibility=[];flags=[]
    for i in range(17):
        x,y,v=raw[3*i:3*i+3]
        if isinstance(v,bool) or v not in (0,1,2):raise ValueError('COCO v仅允许0/1/2')
        x=_number(x);y=_number(y)
        if v==0 and (x!=0 or y!=0):raise ValueError('COCO v0的原始坐标必须为0')
        in_frame=0<=x<width and 0<=y<height
        ok=v>0 and in_frame
        xy.append([x/width,y/height] if ok else [0.,0.])
        valid.append(ok)
        visibility.append('visible' if v==2 else 'occluded' if v==1 else 'unlabeled')
        flags.append(v)
    if 'num_keypoints' in annotation and annotation['num_keypoints']!=sum(v>0 for v in flags):
        raise ValueError('num_keypoints与原始v不一致')
    return {'annotation_id':annotation['id'],'source':'coco_real','bbox_xywh_px':box,
            'xy_normalized':xy,'location_valid':valid,'visibility_state':visibility,
            'source_coco_v':flags,'source_keypoints':list(raw),
            'pose_labelled_count':sum(valid),'detection_target_retained':True}


def synthetic_person_location(annotation,width,height):
    """已投影合成坐标可有效而visibility未知，绝不转换成COCO v1。"""
    box=_box(annotation,width,height)
    points=annotation.get('xy_px');mask=annotation.get('location_valid');vis=annotation.get('visibility_state')
    if any(not isinstance(v,(list,tuple)) or len(v)!=17 for v in (points,mask,vis)):
        raise ValueError('要求独立17点xy、bool位置mask及可见性元数据')
    xy=[]
    for point,ok,state in zip(points,mask,vis):
        if type(ok) is not bool:raise ValueError('location_valid必须bool，不能借用v数字')
        if state not in ('unknown','visible','occluded','outside','behind_camera','missing'):
            raise ValueError('可见性状态不合法')
        if not isinstance(point,(list,tuple)) or len(point)!=2:raise ValueError('xy必须2维')
        x,y=map(_number,point)
        if ok and not (0<=x<width and 0<=y<height):raise ValueError('有效位置必须画内')
        if ok and state in ('outside','behind_camera','missing'):raise ValueError('位置有效与状态冲突')
        xy.append([x/width,y/height] if ok else [0.,0.])
    return {'annotation_id':annotation['id'],'source':'synthetic','bbox_xywh_px':box,
            'xy_normalized':xy,'location_valid':list(mask),'visibility_state':list(vis),
            'source_coco_v':None,'pose_labelled_count':sum(mask),'detection_target_retained':True}


def collate_location_contract(frames):
    """仅整理目标，不读图/不增强；要求框覆盖清单完整且无crowd。"""
    result={k:[] for k in ('keypoints','location_valid','visibility_state','bboxes','cls','batch_idx','target_ids')}
    for batch_idx,frame in enumerate(frames):
        if frame.get('bbox_coverage')!='all_noncrowd_person_instances' or frame.get('crowd_present') is not False:
            raise ValueError('完整person框覆盖和crowd门禁必须显式提供')
        instances=frame['instances'];ids=[i['annotation_id'] for i in instances]
        if len(set(ids))!=len(ids) or set(ids)!=set(frame['expected_person_annotation_ids']):
            raise ValueError('person实例遗漏或重复，不能把未标姿态的人当背景')
        width=_number(frame['width']);height=_number(frame['height'])
        for instance in instances:
            x,y,w,h=instance['bbox_xywh_px']
            result['bboxes'].append([(x+w/2)/width,(y+h/2)/height,w/width,h/height])
            result['keypoints'].append(deepcopy(instance['xy_normalized']))
            result['location_valid'].append(list(instance['location_valid']))
            result['visibility_state'].append(list(instance['visibility_state']))
            result['cls'].append([0]);result['batch_idx'].append(batch_idx)
            result['target_ids'].append(instance['annotation_id'])
    return result


def pack_location_batch(batch):
    """为本机loss创建临时(x,y,location_valid)，第三维绝非COCO可见性。"""
    import torch
    xy=batch['keypoints'];valid=batch['location_valid']
    if not isinstance(xy,torch.Tensor) or xy.ndim!=3 or tuple(xy.shape[1:])!=(17,2):
        raise ValueError('keypoints必须[N,17,2]tensor')
    if not isinstance(valid,torch.Tensor) or valid.dtype!=torch.bool or valid.shape!=xy.shape[:2]:
        raise ValueError('location_valid必须独立[N,17]bool tensor')
    if valid.device!=xy.device or not xy.is_floating_point() or not torch.isfinite(xy).all():
        raise ValueError('位置必须有限浮点且mask同设备')
    if valid.any() and ((xy[valid]<0).any() or (xy[valid]>=1).any()):
        raise ValueError('有效坐标不在归一化画内范围')
    packed=torch.cat((torch.where(valid[...,None],xy,torch.zeros_like(xy)),valid[...,None].to(xy.dtype)),dim=-1)
    result=dict(batch);result['keypoints']=packed
    return result


def _loss_init(self,model,*args,**kwargs):
    import ultralytics
    from ultralytics.utils.loss import v8PoseLoss
    if ultralytics.__version__!='8.4.60' or tuple(model.model[-1].kpt_shape)!=(17,2):
        raise ValueError('仅核验Ultralytics8.4.60的17×2位置模型')
    v8PoseLoss.__init__(self,model,*args,**kwargs)


def _loss_forward(self,preds,batch):
    from ultralytics.utils.loss import v8PoseLoss
    return v8PoseLoss.loss(self,preds,pack_location_batch(batch))


def separated_location_loss_class():
    """候选桥接复用分配器/位置loss，输出头仍17×2且不训练visibility。"""
    global _LOSS_CLASS
    if _LOSS_CLASS is None:
        from ultralytics.utils.loss import v8PoseLoss
        _LOSS_CLASS=type('SeparatedLocationPoseLoss',(v8PoseLoss,),
                         {'__module__':__name__,'__init__':_loss_init,'loss':_loss_forward})
        globals()['SeparatedLocationPoseLoss']=_LOSS_CLASS
    return _LOSS_CLASS


def __getattr__(name):
    if name=='SeparatedLocationPoseLoss':return separated_location_loss_class()
    raise AttributeError(name)
