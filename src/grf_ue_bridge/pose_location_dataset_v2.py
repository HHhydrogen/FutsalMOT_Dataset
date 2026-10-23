"""独立JSONL位置Dataset、LetterBox与严格xy CPU兼容Trainer。

当前Trainer只允许自建toy兼容配置；不替代正式评测器或已冻结研究协议。
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

from .pose_location_supervision_v2 import collate_location_contract,separated_location_loss_class

VERSION='jsonl_separated_location_letterbox_cpu_candidate_v1'
_TRAINER=None


def letterbox_geometry(width,height,size):
    """与本机Ultralytics中心LetterBox一致的比例、取整和整数边框。"""
    if any(type(v) is not int or v<=0 for v in (width,height,size)):
        raise ValueError('尺寸必须正整数')
    ratio=min(size/height,size/width)
    resized=(round(width*ratio),round(height*ratio))
    dw=size-resized[0];dh=size-resized[1]
    left=round(dw/2-.1);top=round(dh/2-.1)
    return {'ratio':ratio,'resize_wh':resized,'left':left,'top':top,'right':dw-left,'bottom':dh-top,
            'matrix':[[ratio,0.,left],[0.,ratio,top],[0.,0.,1.]],'output_wh':[size,size]}


def transform_location_instance(instance,input_wh,matrix,output_wh,flip_idx=None):
    """同步仿射xy/bbox/mask；原false永不因变换恢复，画外新false规范置零。"""
    if len(matrix)!=3 or any(len(row)!=3 for row in matrix) or matrix[2]!=[0.,0.,1.]:
        raise ValueError('只支持明确二维仿射，不猜测透视除法')
    if any(not math.isfinite(v) for row in matrix for v in row):raise ValueError('仿射非有限')
    width,height=input_wh;ow,oh=output_wh
    if min(width,height,ow,oh)<=0:raise ValueError('图像尺寸无效')
    result=deepcopy(instance)
    xy=instance['xy_normalized'];valid=instance['location_valid']
    if len(xy)!=17 or len(valid)!=17 or any(type(v) is not bool for v in valid):raise ValueError('位置契约不完整')
    def apply(x,y):return [matrix[0][0]*x+matrix[0][1]*y+matrix[0][2],matrix[1][0]*x+matrix[1][1]*y+matrix[1][2]]
    new_xy=[];new_valid=[];reasons=[]
    for point,ok in zip(xy,valid):
        if len(point)!=2 or any(not math.isfinite(v) for v in point):raise ValueError('坐标须有限xy')
        x,y=apply(point[0]*width,point[1]*height)
        keep=ok and 0<=x<ow and 0<=y<oh
        new_xy.append([x/ow,y/oh] if keep else [0.,0.]);new_valid.append(keep)
        reasons.append(None if keep else 'source_invalid' if not ok else 'transformed_outside')
    if flip_idx is not None:
        if sorted(flip_idx)!=list(range(17)):raise ValueError('镜像必须给出完整COCO17置换')
        new_xy=[new_xy[i] for i in flip_idx];new_valid=[new_valid[i] for i in flip_idx];reasons=[reasons[i] for i in flip_idx]
        for key in ('visibility_state','source_coco_v'):
            if result.get(key) is not None:result[key]=[result[key][i] for i in flip_idx]
        if result.get('source_keypoints') is not None:
            original=result['source_keypoints']
            if len(original)!=51:raise ValueError('原始COCO证据不完整')
            result['source_keypoints']=[v for i in flip_idx for v in original[3*i:3*i+3]]
    x,y,w,h=instance['bbox_xywh_px']
    corners=[apply(px,py) for px,py in ((x,y),(x+w,y),(x,y+h),(x+w,y+h))]
    left=max(0.,min(p[0] for p in corners));right=min(float(ow),max(p[0] for p in corners))
    top=max(0.,min(p[1] for p in corners));bottom=min(float(oh),max(p[1] for p in corners))
    if right<=left or bottom<=top:return None
    result.update(xy_normalized=new_xy,location_valid=new_valid,geometry_rejection=reasons,
                  bbox_xywh_px=[left,top,right-left,bottom-top],pose_labelled_count=sum(new_valid))
    return result


def _validate_instance(instance,width,height):
    xy=instance['xy_normalized'];valid=instance['location_valid'];vis=instance['visibility_state']
    if any(len(v)!=17 for v in (xy,valid,vis)) or any(type(v) is not bool for v in valid):
        raise ValueError('要求独立17点xy/bool mask/source可见性')
    for p,ok in zip(xy,valid):
        if len(p)!=2 or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in p):
            raise ValueError('xy必须有限数值')
        if ok and not all(0<=v<1 for v in p):raise ValueError('有效点不在画内')
        if not ok and p!=[0.,0.] and p!=[0,0]:raise ValueError('无效点必须显式置零占位')
    source=instance.get('source');raw_v=instance.get('source_coco_v')
    if source=='coco_real':
        if not isinstance(raw_v,list) or len(raw_v)!=17:raise ValueError('COCO需原始v证据')
        for v,ok,state in zip(raw_v,valid,vis):
            if type(v) is not int or v not in (0,1,2) or (v==0 and ok):raise ValueError('COCO v0不能变有效点')
            if state!=('unlabeled' if v==0 else 'occluded' if v==1 else 'visible'):raise ValueError('真实源可见性改变')
    elif source=='synthetic':
        if raw_v is not None:raise ValueError('合成不能伪造COCO v')
        if any(v not in ('unknown','visible','occluded','outside','behind_camera','missing') for v in vis):raise ValueError('合成状态非法')
        if any(ok and state in ('outside','behind_camera','missing') for ok,state in zip(valid,vis)):
            raise ValueError('合成位置有效性与画外/缺失状态冲突')
    else:raise ValueError('来源必须真实COCO或合成位置记录')
    x,y,w,h=instance['bbox_xywh_px']
    if not all(math.isfinite(v) for v in (x,y,w,h)) or min(w,h)<=0 or min(x,y)<0 or x+w>width+1e-6 or y+h>height+1e-6:
        raise ValueError('人物bbox必须已裁到图像有效范围')


class LocationPoseDataset:
    """map-style图像Dataset；不经YOLO txt，不自动下载或丢弃缺点人物。"""
    rect=False
    mosaic=False

    def __init__(self,jsonl_path,imgsz=128,verify_image_sha256=True):
        import numpy as np
        self.path=Path(jsonl_path).resolve();self.imgsz=imgsz;self.verify_image_sha256=verify_image_sha256
        self.records=[];self.labels=[];self.im_files=[]
        with self.path.open(encoding='utf-8') as handle:
            for line in handle:
                if not line.strip():continue
                frame=json.loads(line)
                if frame.get('schema')!='separated_pose_location_frame_v1':raise ValueError('未知JSONL图像记录')
                if any(type(frame.get(k)) is not int or frame[k]<=0 for k in ('width','height')):raise ValueError('原始图像尺寸非法')
                collate_location_contract([frame])
                for instance in frame['instances']:_validate_instance(instance,frame['width'],frame['height'])
                raw=Path(frame['image_file'])
                image=(self.path.parent/raw).resolve() if not raw.is_absolute() else raw.resolve()
                if not image.is_file():raise FileNotFoundError(image)
                digest=frame.get('image_sha256')
                if verify_image_sha256 and (not isinstance(digest,str) or len(digest)!=64):raise ValueError('缺图像SHA')
                if verify_image_sha256 and hashlib.sha256(image.read_bytes()).hexdigest()!=digest:raise ValueError('图像SHA不一致')
                self.records.append(frame);self.im_files.append(str(image))
                boxes=[]
                for instance in frame['instances']:
                    x,y,w,h=instance['bbox_xywh_px'];boxes.append([(x+w/2)/frame['width'],(y+h/2)/frame['height'],w/frame['width'],h/frame['height']])
                self.labels.append({'cls':np.zeros((len(boxes),1),dtype=np.float32),'bboxes':np.array(boxes,dtype=np.float32).reshape(-1,4)})
        if not self.records:raise ValueError('JSONL为空')

    def __len__(self):return len(self.records)

    def __getitem__(self,index):
        import cv2
        import numpy as np
        import torch
        image=cv2.imread(self.im_files[index],cv2.IMREAD_COLOR)
        frame=self.records[index]
        if image is None or image.shape[:2]!=(frame['height'],frame['width']):raise ValueError('实际图像尺寸不符')
        geometry=letterbox_geometry(frame['width'],frame['height'],self.imgsz)
        if (image.shape[1],image.shape[0])!=tuple(geometry['resize_wh']):image=cv2.resize(image,geometry['resize_wh'],interpolation=cv2.INTER_LINEAR)
        image=cv2.copyMakeBorder(image,geometry['top'],geometry['bottom'],geometry['left'],geometry['right'],cv2.BORDER_CONSTANT,value=(114,114,114))
        instances=[transform_location_instance(v,(frame['width'],frame['height']),geometry['matrix'],geometry['output_wh']) for v in frame['instances']]
        if any(v is None for v in instances):raise AssertionError('LetterBox不应移除有效原图人物bbox')
        transformed={**frame,'width':self.imgsz,'height':self.imgsz,'instances':instances}
        packed=collate_location_contract([transformed]);count=len(instances)
        return {'img':torch.from_numpy(np.ascontiguousarray(image[:,:,::-1].transpose(2,0,1))),
                'keypoints':torch.tensor(packed['keypoints'],dtype=torch.float32).reshape(count,17,2),
                'location_valid':torch.tensor(packed['location_valid'],dtype=torch.bool).reshape(count,17),
                'bboxes':torch.tensor(packed['bboxes'],dtype=torch.float32).reshape(count,4),
                'cls':torch.zeros((count,1),dtype=torch.float32),'batch_idx':torch.zeros(count,dtype=torch.int64),
                'visibility_state':packed['visibility_state'],'target_ids':packed['target_ids'],
                'sources':[v['source'] for v in instances],'im_file':self.im_files[index],
                'ori_shape':(frame['height'],frame['width']),'resized_shape':(self.imgsz,self.imgsz),
                'ratio_pad':((geometry['ratio'],geometry['ratio']),(geometry['left'],geometry['top']))}

    @staticmethod
    def collate_fn(samples):
        import torch
        result={'img':torch.stack([s['img'] for s in samples])}
        for key in ('keypoints','location_valid','bboxes','cls'):
            result[key]=torch.cat([s[key] for s in samples],dim=0)
        result['batch_idx']=torch.cat([s['batch_idx']+i for i,s in enumerate(samples)])
        for key in ('visibility_state','target_ids','sources'):
            result[key]=[v for s in samples for v in s[key]]
        for key in ('im_file','ori_shape','resized_shape','ratio_pad'):result[key]=[s[key] for s in samples]
        return result


class LocationLossOnlyValidator:
    """只计算独立mask位置/检测loss；绝不把内部mask当COCO AP标签。"""
    metrics=SimpleNamespace(keys=[])

    def __call__(self,trainer):
        import torch
        model=trainer.ema.ema if trainer.ema else trainer.model
        model.eval();total=torch.zeros(5,device=trainer.device);batches=0
        # 本验证器只与真实Trainer绑定，不支持独立AP推理入口。
        with torch.no_grad():
            for batch in trainer.test_loader:
                batch=trainer.preprocess_batch(batch)
                predictions=model(batch['img'])
                criterion=separated_location_loss_class()(model)
                _,items=criterion(predictions,batch)
                total+=items;batches+=1
        if not batches:raise ValueError('验证loader为空')
        values=total/batches
        result=trainer.label_loss_items(values,prefix='val')
        result['fitness']=-float(values.sum())
        return result


def location_pose_trainer_class():
    """组合既有exact_xy Trainer，不修改其已验函数；仅CPU toy范围。"""
    global _TRAINER
    if _TRAINER is None:
        import ultralytics
        from .pose_xy_head_transfer import pose_xy_trainer_class
        if ultralytics.__version__!='8.4.60':raise ValueError('只核验8.4.60')
        class LocationPoseCPUTrainer(pose_xy_trainer_class()):
            def __init__(self,*args,overrides=None,**kwargs):
                if not isinstance(overrides,dict) or overrides.get('device')!='cpu':
                    raise ValueError('必须在Trainer初始化前显式device=cpu，禁止自动设备选择')
                super().__init__(*args,overrides=overrides,**kwargs)

            def get_dataset(self):
                if self.device.type!='cpu' or self.args.workers!=0 or self.args.amp or self.args.plots:
                    raise ValueError('本候选Trainer仅允许CPU/workers0/ampfalse/plotsfalse')
                for key in ('mosaic','mixup','cutmix','copy_paste','degrees','translate','scale','shear','perspective','fliplr','flipud','hsv_h','hsv_s','hsv_v','bgr','erasing','multi_scale'):
                    if getattr(self.args,key,0)!=0:raise ValueError('首轮只支持确定性LetterBox，拒绝未实现增强：'+key)
                if getattr(self.args,'augmentations',None) is not None:raise ValueError('自定义增强未核验，不静默忽略')
                if self.args.rect or self.args.cache or self.args.compile or self.args.fraction!=1 or self.args.cls_pw!=0:
                    raise ValueError('rect/cache/compile/fraction/类别权重尚未核验')
                config_path=Path(self.args.data).resolve()
                data=json.loads(config_path.read_text(encoding='utf-8'))
                if data.get('schema')!='pose_location_jsonl_cpu_toy_v1' or data.get('usage_scope')!='cpu_toy_compatibility_only':
                    raise ValueError('只允许明确toy兼容配置，不能启动研究训练')
                for split in ('train','val'):
                    p=Path(data[split]);data[split]=str((config_path.parent/p).resolve() if not p.is_absolute() else p.resolve())
                return {**data,'nc':1,'names':{0:'player'},'kpt_shape':[17,2],'channels':3}

            def build_dataset(self,img_path,mode='train',batch=None):
                return LocationPoseDataset(img_path,self.args.imgsz,verify_image_sha256=True)

            def set_model_attributes(self):
                super().set_model_attributes()
                self.model.criterion=separated_location_loss_class()(self.model)

            def get_validator(self):
                self.loss_names=('box_loss','pose_loss','kobj_loss','cls_loss','dfl_loss')
                return LocationLossOnlyValidator()

            def final_eval(self):
                # 正式AP需要另行核验COCO缺点/ignore语义；不能调用默认PoseValidator冒充。
                self.location_final_eval_scope='loss_only_cpu_compatibility_no_AP_no_optimizer_strip'
        _TRAINER=LocationPoseCPUTrainer
    return _TRAINER
