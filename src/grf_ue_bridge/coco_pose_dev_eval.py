"""固定COCO公开开发集的位置头评价；不接受任意测试集或伪可见性。

依赖延迟加载：元数据、坐标适配和预测验证不需要Torch；AP只由官方COCOeval计算。
"""
from collections import defaultdict
from contextlib import redirect_stdout
from copy import deepcopy
import hashlib
import io
import json
import math
from pathlib import Path
import re
import statistics

VERSION = 'futsalmot_coco_pose_dev_evaluation_v1'
SCORE_POLICY = 'person_detection_score_broadcast_to_17_xy_points_not_physical_visibility'
COCO17 = ('nose','left_eye','right_eye','left_ear','right_ear','left_shoulder','right_shoulder',
          'left_elbow','right_elbow','left_wrist','right_wrist','left_hip','right_hip',
          'left_knee','right_knee','left_ankle','right_ankle')
POSTPROCESS = dict(confidence=.001, nms_iou=.7, max_detections_per_image=20,
                   batch_size=1, imgsz=640, augment=False, half=False,
                   letterbox='center_integer_padding_fill114', keypoint_clipping=False)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    def invalid(value):
        raise ValueError('JSON包含非有限常数: '+value)
    return json.loads(Path(path).read_text(encoding='utf-8-sig'), parse_constant=invalid)


def digest(value):
    return isinstance(value,str) and re.fullmatch('[0-9a-f]{64}',value) is not None


def safe_path(path):
    """在打开文件前拒绝封存SoccerNet及测试目录；具体图片另锁到val2017。"""
    value=Path(path).resolve()
    pieces=re.split(r'[/\\]',str(value).lower())
    require(not any(p in {'test','test2017','test2014','test-dev2017','test-dev2014'}
                    or re.search(r'snmot[-_]?0*(116|187)(?:\D|$)',p) for p in pieces),
            '禁止读取测试集或封存SNMOT116/187: '+str(value))
    return value


def checked_json(path, expected):
    path=safe_path(path)
    require(digest(expected) and sha256(path)==expected,'文件SHA不符: '+str(path))
    return read_json(path)


def _ids(values, count, label):
    require(isinstance(values,list) and len(values)==count and len(set(values))==count
            and all(type(v) is int and v>0 for v in values),label+' ID数量/唯一性错误')
    return values


def load_dev_context(protocol_path, protocol_sha256):
    """只读准备收据和val子集；不读取train图像/标注或官方全val。"""
    protocol_path=safe_path(protocol_path)
    protocol=checked_json(protocol_path,protocol_sha256)
    require(protocol.get('schema')=='futsalmot_pose_location_coco_control_candidate_v1','协议schema不支持')
    candidate_path=protocol_path;candidate_sha256=protocol_sha256;candidate=protocol
    frozen='root_freeze' in protocol
    if frozen:
        freeze=protocol['root_freeze']
        require(isinstance(freeze,dict) and freeze.get('approved_by')=='root'
                and protocol.get('training_authorized') is True,'冻结协议缺root明确准入')
        reference=freeze.get('candidate_protocol',{})
        candidate_path=safe_path(reference['path']);candidate_sha256=reference['sha256']
        require(candidate_path!=protocol_path,'冻结协议不能循环引用自身')
        candidate=checked_json(candidate_path,candidate_sha256)
        require(candidate.get('schema')==protocol['schema'] and 'root_freeze' not in candidate,
                '冻结来源必须是原始候选')
        for key in ('real_selection','data_root','initialization','paired_training_candidate','arms',
                    'development_evaluation_candidate'):
            require(protocol.get(key)==candidate.get(key),'冻结协议改变已准备的开发/初始化/配对配置: '+key)
    settings=protocol['real_selection']
    require(settings.get('train_split')=='train2017' and settings.get('train_images')==1000
            and settings.get('pose_dev_split')=='val2017' and settings.get('pose_dev_images')==200,
            '本入口仅支持锁定1000train/200公开val的候选')
    require(protocol['paired_training_candidate'].get('imgsz')==640,'评价输入须固定640')
    root=safe_path(protocol['data_root'])
    binding=read_json(safe_path(root/'candidate_protocol_manifest.json'))
    require(binding.get('schema')=='futsalmot_coco_pose_candidate_binding_v1'
            and binding.get('protocol_sha256')==candidate_sha256
            and safe_path(binding['protocol_path'])==candidate_path
            and binding.get('test_SN116_SN187_read') is False,'准备收据未绑定当前协议或测试封存状态')
    selection_path=safe_path(root/'subset'/'selection_manifest.json')
    require(safe_path(settings['selection_manifest'])==selection_path
            and safe_path(binding['selection_manifest_path'])==selection_path,'选择清单路径不是准备目录')
    selection=checked_json(selection_path,binding['selection_manifest_sha256'])
    require(selection.get('schema')=='official_coco2017_location_subset_candidate_v1'
            and selection.get('val_role')=='public_pose_development_only_not_SoccerNet_final_test',
            '未明确公开Pose开发集角色')
    train=_ids(selection['splits']['train2017']['image_ids_in_sha_rank_order'],1000,'train')
    val=selection['splits']['val2017']
    ids=_ids(val['image_ids_in_sha_rank_order'],200,'val')
    require(not set(train)&set(ids) and val.get('role')=='pose_dev_only','train/val泄漏或角色错误')
    annotation_path=safe_path(root/'subset'/'val2017_person_annotations.json')
    require(safe_path(val['annotation_subset_path'])==annotation_path,'仅允许准备好的val标注子集')
    gt=checked_json(annotation_path,val['annotation_subset_sha256'])
    validate_ground_truth(gt,ids)
    receipt_path=safe_path(root/'images_receipt.json')
    require(safe_path(binding['images_receipt_path'])==receipt_path,'图片收据路径错误')
    images_receipt=checked_json(receipt_path,binding['images_receipt_sha256'])
    require(images_receipt.get('selection_manifest_sha256')==binding['selection_manifest_sha256'],
            '图片收据与选择清单不同')
    entries=[row for row in images_receipt['images'] if row.get('split')=='val2017']
    require(len(entries)==200 and len({r['image_id'] for r in entries})==200
            and {r['image_id'] for r in entries}==set(ids),'val图片收据不完整')
    image_info={r['id']:r for r in gt['images']}
    for row in entries:
        info=image_info[row['image_id']]
        expected=safe_path(root/'images'/'val2017'/info['file_name'])
        require(safe_path(row['path'])==expected and row.get('object')=='val2017/'+info['file_name']
                and row.get('width')==info['width'] and row.get('height')==info['height']
                and digest(row.get('sha256')),'val图片身份/尺寸/SHA收据错误')
    return dict(protocol=protocol,protocol_sha256=protocol_sha256,frozen_protocol=frozen,
                candidate_protocol_path=candidate_path,candidate_protocol_sha256=candidate_sha256,
                root=root,gt=gt,image_ids=ids,
                images={r['image_id']:r for r in entries},annotation_path=annotation_path,
                binding=dict(protocol_sha256=protocol_sha256,
                    candidate_protocol_sha256=candidate_sha256,
                    selection_manifest_sha256=binding['selection_manifest_sha256'],
                    annotation_sha256=val['annotation_subset_sha256'],
                    images_receipt_sha256=binding['images_receipt_sha256']))


def _numbers(values, length):
    return (isinstance(values,(list,tuple)) and len(values)==length
            and all(type(v) in (int,float) and math.isfinite(v) for v in values))


def validate_ground_truth(gt, ids):
    images=gt['images'];annotations=gt['annotations']
    require(len(images)==len(ids) and {v['id'] for v in images}==set(ids),'GT不是精确dev IDs')
    require(len(gt['categories'])==1 and gt['categories'][0]['id']==1
            and tuple(gt['categories'][0]['keypoints'])==COCO17,'GT类别或COCO17顺序不符')
    require(len({a['id'] for a in annotations})==len(annotations),'重复标注ID')
    for row in images:
        require(type(row['width']) is int and type(row['height']) is int and min(row['width'],row['height'])>0
                and row['file_name']==f"{row['id']:012d}.jpg",'图片文件名或尺寸错误')
    for ann in annotations:
        require(ann['image_id'] in ids and ann['category_id']==1 and ann.get('iscrowd')==0
                and _numbers(ann['bbox'],4) and min(ann['bbox'][2:])>0
                and type(ann.get('area')) in (int,float) and math.isfinite(ann['area']) and ann['area']>0,
                '完整非crowd人物框/原始area无效')
        k=ann['keypoints']
        require(_numbers(k,51) and all(type(v) is int and v in (0,1,2) for v in k[2::3])
                and ann['num_keypoints']==sum(v>0 for v in k[2::3]),'真实COCO关键点v/数量无效')
    # num_keypoints=0人物完整保留；标准COCOeval自己处理关键点评价ignore。


def inverse_prediction(image_id, bbox_xyxy, xy, score, image_wh, size=640):
    """回到原图像素；点不截断，框截断。分数不是物理可见性。"""
    from .pose_location_dataset_v2 import letterbox_geometry
    require(type(image_id) is int and _numbers(bbox_xyxy,4) and len(xy)==17
            and all(_numbers(p,2) for p in xy) and type(score) in (int,float)
            and math.isfinite(score) and 0<=score<=1,'预测坐标/分数无效')
    width,height=image_wh;geometry=letterbox_geometry(width,height,size)
    ratio=geometry['ratio'];left=geometry['left'];top=geometry['top']
    x1,y1,x2,y2=bbox_xyxy
    x1=max(0.,min(float(width),(x1-left)/ratio));x2=max(0.,min(float(width),(x2-left)/ratio))
    y1=max(0.,min(float(height),(y1-top)/ratio));y2=max(0.,min(float(height),(y2-top)/ratio))
    if x2<=x1 or y2<=y1:
        return None
    return dict(image_id=image_id,category_id=1,bbox=[x1,y1,x2-x1,y2-y1],score=float(score),
                keypoints=[v for x,y in xy for v in ((x-left)/ratio,(y-top)/ratio,float(score))])


def validate_predictions(predictions, processed_ids, ids):
    require(isinstance(processed_ids,list) and len(processed_ids)==len(ids)
            and len(set(processed_ids))==len(ids) and set(processed_ids)==set(ids),
            '必须记录全部dev图像（包括零检测）')
    require(isinstance(predictions,list),'预测须为COCO结果列表')
    counts=defaultdict(int)
    for row in predictions:
        require(type(row.get('image_id')) is int and row['image_id'] in ids and row.get('category_id')==1,
                '预测含未准入的图像/类别')
        require(_numbers(row.get('bbox'),4) and min(row['bbox'][2:])>0
                and type(row.get('score')) in (int,float) and math.isfinite(row['score'])
                and POSTPROCESS['confidence']<=row['score']<=1 and _numbers(row.get('keypoints'),51),'预测格式/数值无效')
        require(all(s==row['score'] for s in row['keypoints'][2::3]),
                'xy头必须统一广播检测分数，拒绝伪COCO v/GT可见性')
        counts[row['image_id']]+=1
        require(counts[row['image_id']]<=POSTPROCESS['max_detections_per_image'],'预测超固定20人上限')


def validate_checkpoint_metadata(checkpoint, run, context, schedule_sha256):
    """完成的last与运行收据精确绑定；未完整1000步不得伪称最终结果。"""
    protocol=context['protocol'];bindings=checkpoint.get('bindings',{})
    budget=protocol['paired_training_candidate']['actual_optimizer_updates']
    required_bindings={'protocol_sha256','schedule_manifest_sha256','admission_sha256','schedule_sha256',
                       'trainer_source_sha256','real_jsonl_sha256','synthetic_jsonl_sha256'}
    require(type(budget) is int and budget==1000 and required_bindings<=set(bindings)
            and all(digest(bindings[k]) for k in required_bindings),'固定预算或完整来源bindings缺失')
    require(checkpoint.get('schema')=='futsalmot_pose_location_control_checkpoint_v1'
            and type(checkpoint.get('completed_updates')) is int and checkpoint['completed_updates']==budget
            and checkpoint.get('planned_updates')==budget and checkpoint.get('ema_updates')==budget,
            'checkpoint不是完成固定预算的last/EMA')
    require(bindings.get('protocol_sha256')==context['protocol_sha256']
            and digest(schedule_sha256) and bindings.get('schedule_manifest_sha256')==schedule_sha256,
            'checkpoint协议/采样表SHA不符')
    require(checkpoint.get('arm') in protocol['arms'] and type(checkpoint.get('seed')) is int
            and checkpoint['seed'] in protocol['paired_training_candidate']['seeds']
            and digest(checkpoint.get('exposure_chain_sha256')),'checkpoint arm/seed/曝光链错误')
    require(run.get('schema')=='futsalmot_pose_location_control_run_v1' and run.get('complete') is True
            and run.get('completed_updates')==budget and run.get('planned_updates')==budget
            and run.get('checkpoint_choice')=='last_exact_budget_EMA_no_best'
            and run.get('visibility_supervised') is False and run.get('real_test_read') is False
            and run.get('bindings')==bindings
            and digest(checkpoint.get('initial_state_sha256'))
            and all(run.get(k)==checkpoint.get(k) for k in ('arm','seed','exposure_chain_sha256','initial_state_sha256')),
            '运行收据不完整或与checkpoint不一致')
    return dict(arm=checkpoint['arm'],seed=checkpoint['seed'],completed_updates=budget,
                bindings=bindings,exposure_chain_sha256=checkpoint['exposure_chain_sha256'])


def bbox_iou(a,b):
    x=max(a[0],b[0]);y=max(a[1],b[1]);r=min(a[0]+a[2],b[0]+b[2]);d=min(a[1]+a[3],b[1]+b[3])
    intersection=max(0.,r-x)*max(0.,d-y)
    return intersection/(a[2]*a[3]+b[2]*b[3]-intersection) if intersection>0 else 0.


def location_metrics(gt,predictions):
    """辅助指标按检测框IoU贪心配对；明确漏检分母，不用GT点挑配对。"""
    by_image=defaultdict(list);dets=defaultdict(list)
    for a in gt['annotations']:by_image[a['image_id']].append(a)
    for p in predictions:dets[p['image_id']].append(p)
    errors=[];normalized=[];matched_people=0;covered=0;inside=0
    widths={r['id']:(r['width'],r['height']) for r in gt['images']}
    for image_id,anns in by_image.items():
        remaining={a['id']:a for a in anns}
        for prediction in sorted(dets[image_id],key=lambda r:-r['score']):
            ordered=sorted(remaining.values(),key=lambda a:(-bbox_iou(a['bbox'],prediction['bbox']),a['id']))
            if not ordered or bbox_iou(ordered[0]['bbox'],prediction['bbox'])<.5:continue
            ann=remaining.pop(ordered[0]['id']);matched_people+=1
            width,height=widths[image_id]
            for i in range(17):
                x,y,v=ann['keypoints'][3*i:3*i+3]
                if v==0:continue
                px,py,_=prediction['keypoints'][3*i:3*i+3]
                error=math.hypot(px-x,py-y);errors.append(error);normalized.append(error/math.sqrt(ann['area']))
                covered+=1;inside+=int(0<=px<width and 0<=py<height)
    total=sum(a['num_keypoints'] for a in gt['annotations']);people=len(gt['annotations'])
    areas=[a['area'] for a in gt['annotations']]
    return dict(matching='score_ordered_bbox_IoU_greedy_threshold_0.5_not_GT_pose_matching',
                person_boxes=people,all_missing_pose_person_boxes=sum(a['num_keypoints']==0 for a in gt['annotations']),
                matched_person_boxes=matched_people,unmatched_person_boxes=people-matched_people,
                labeled_GT_points=total,matched_labeled_points=covered,
                matched_labeled_coverage=covered/total if total else None,
                matched_prediction_points_inside_image=inside,
                conditional_mean_L2_px=statistics.mean(errors) if errors else None,
                conditional_median_L2_px=statistics.median(errors) if errors else None,
                conditional_mean_L2_over_sqrt_original_COCO_area=statistics.mean(normalized) if normalized else None,
                conditional_PCK_0p05_sqrt_area=sum(v<=.05 for v in normalized)/len(normalized) if normalized else None,
                original_COCO_area=dict(min=min(areas) if areas else None,median=statistics.median(areas) if areas else None,
                    max=max(areas) if areas else None,small=sum(a<32**2 for a in areas),
                    medium=sum(32**2<=a<96**2 for a in areas),large=sum(a>=96**2 for a in areas)))


def official_metrics(gt,predictions,ids):
    """只调用官方实现；空预测也由官方COCOeval计算，缺依赖绝不伪造AP。"""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    output=io.StringIO();results={}
    with redirect_stdout(output):
        truth=COCO();truth.dataset=deepcopy(gt);truth.createIndex()
        if predictions:
            detected=truth.loadRes(deepcopy(predictions))
        else:
            detected=COCO();detected.dataset=dict(images=deepcopy(gt['images']),
                categories=deepcopy(gt['categories']),annotations=[]);detected.createIndex()
        for kind in ('keypoints','bbox'):
            kind_truth=truth
            if kind=='bbox' and 'source_instance_area_by_annotation_id' in gt:
                # 关键点原area不改；bbox辅助评价另用严格join保留的instances原area。
                bbox_gt=deepcopy(gt);areas=gt['source_instance_area_by_annotation_id']
                require(set(areas)=={str(a['id']) for a in gt['annotations']},'instances area映射不完整')
                for ann in bbox_gt['annotations']:
                    area=areas[str(ann['id'])]
                    require(type(area) in (int,float) and math.isfinite(area) and area>0,'instances原area无效')
                    ann['area']=area
                kind_truth=COCO();kind_truth.dataset=bbox_gt;kind_truth.createIndex()
            evaluator=COCOeval(kind_truth,detected,kind);evaluator.params.imgIds=list(ids);evaluator.params.catIds=[1]
            evaluator.evaluate();evaluator.accumulate();evaluator.summarize()
            names=(('AP','AP50','AP75','AP_medium','AP_large','AR','AR50','AR75','AR_medium','AR_large')
                   if kind=='keypoints' else ('AP','AP50','AP75','AP_small','AP_medium','AP_large',
                                              'AR1','AR10','AR100','AR_small','AR_medium','AR_large'))
            results[kind]=dict(zip(names,map(float,evaluator.stats)))
            results[kind]['maxDets']=list(evaluator.params.maxDets)
            results[kind]['area_source']=('original_joined_instances_area' if kind=='bbox'
                and 'source_instance_area_by_annotation_id' in gt else 'original_keypoint_annotation_area')
    return results,output.getvalue()


def evaluate(context,predictions,processed_ids):
    validate_predictions(predictions,processed_ids,context['image_ids'])
    import importlib.metadata
    metrics,log=official_metrics(context['gt'],predictions,context['image_ids'])
    return dict(schema=VERSION,status='official_public_dev_evaluated',binding=context['binding'],
        processed_image_ids=list(processed_ids),image_count=len(processed_ids),prediction_count=len(predictions),
        official_COCOeval=metrics,pycocotools_version=importlib.metadata.version('pycocotools'),
        auxiliary_location=location_metrics(context['gt'],predictions),postprocess=POSTPROCESS,
        keypoint_score_policy=SCORE_POLICY,physical_visibility='unknown_not_predicted',
        test_read=False,protocol_status=context['protocol']['status'],
        limitations=['public_200_image_crowd_excluded_development_subset_not_full_COCO_val',
                     'COCO_pretrained_initialization_continued_training',
                     'pose_AP_includes_detector_quality_not_MOT_AssA_or_HOTA',
                     'location_error_conditional_on_IoU_matched_people_read_coverage_together',
                     'official_minus_one_metric_means_no_applicable_area_targets']),log


def load_evaluation_model(checkpoint_path, expected_sha256, role, context,
                          run_receipt_path=None, run_receipt_sha256=None, schedule_sha256=None):
    """只反序列化调用者明确钉住SHA的可信本地模型，不重建YAML或选择best。"""
    import torch
    from . import pose_xy_head_transfer as transfer
    checkpoint_path=safe_path(checkpoint_path)
    require(digest(expected_sha256) and sha256(checkpoint_path)==expected_sha256,'checkpoint实际SHA变化')
    require(role in ('last_ema','pretrained_reference'),'不支持checkpoint角色')
    initialization=context['protocol']['initialization']
    if role=='pretrained_reference':
        allowed={initialization['official_sha256'],initialization['derived_sha256']}
        require(expected_sha256 in allowed,'预训练参考不是协议的官方/严格xy适配权重')
    else:
        require(checkpoint_path.name=='last.pt','正式对照仅允许last.pt，拒绝best/中途选择')
        require(context.get('frozen_protocol') is True,'正式last评价必须使用root冻结协议，不接受原候选冒充冻结')
    loaded=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
    require(isinstance(loaded,dict),'checkpoint不是已知字典')
    metadata=dict(role=role,path=str(checkpoint_path),sha256=expected_sha256)
    if role=='last_ema':
        require(run_receipt_path is not None,'缺完成运行收据')
        run=checked_json(run_receipt_path,run_receipt_sha256)
        metadata.update(validate_checkpoint_metadata(loaded,run,context,schedule_sha256))
        require(isinstance(loaded.get('ema'),torch.nn.Module),'缺实际EMA模型，不允许回退model')
        model=loaded['ema']
        metadata.update(run_receipt_path=str(safe_path(run_receipt_path)),run_receipt_sha256=run_receipt_sha256)
    else:
        model=loaded.get('ema') if isinstance(loaded.get('ema'),torch.nn.Module) else loaded.get('model')
        model=transfer.map_pose_xy(model.float(),mode='exact_full_conv_then_select')
    head=transfer._validate(model)
    require(tuple(head.kpt_shape)==(17,2) and head.pose_xy_transfer.get('mode')=='exact_full_conv_then_select'
            and head.nc==1,'评价必须使用保留物理51通道的严格xy普通单人类Pose头')
    return model.float().eval(),metadata


def inference(context,model,device='cpu'):
    """逐张开发图推理；不开启AutoBackend/fuse/下载/半精度/自动选GPU。"""
    import csv
    import os
    import subprocess
    require(device in ('cpu','cuda:0'),'设备必须显式cpu或cuda:0')
    if device=='cuda:0':
        require(os.name=='nt','当前GPU门禁仅验证Windows，其他平台拒绝自动推断UE状态')
        proc=subprocess.run(['tasklist','/FI','IMAGENAME eq UnrealEditor.exe','/FO','CSV','/NH'],
                            check=True,capture_output=True,text=True)
        require(not any(row and row[0].lower()=='unrealeditor.exe' for row in csv.reader(proc.stdout.splitlines())),
                'UnrealEditor仍在运行，禁止GPU评价')
    import cv2
    import numpy as np
    import torch
    import ultralytics
    from ultralytics.utils.nms import non_max_suppression
    from .pose_location_dataset_v2 import letterbox_geometry
    require(ultralytics.__version__=='8.4.60','只验证Ultralytics8.4.60推理输出/NMS')
    if device.startswith('cuda'):require(torch.cuda.is_available(),'显式CUDA不可用，不回退')
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    model=model.to(device).float().eval();predictions=[];processed=[];discarded_empty_boxes=0
    with torch.inference_mode():
        for image_id in context['image_ids']:
            entry=context['images'][image_id];path=safe_path(entry['path']);raw=path.read_bytes()
            require(hashlib.sha256(raw).hexdigest()==entry['sha256'],'开发图片SHA不符: '+str(path))
            image=cv2.imdecode(np.frombuffer(raw,dtype=np.uint8),cv2.IMREAD_COLOR)
            require(image is not None and image.shape[:2]==(entry['height'],entry['width']),'开发图片尺寸/解码失败')
            geometry=letterbox_geometry(entry['width'],entry['height'],640)
            resized=cv2.resize(image,geometry['resize_wh'],interpolation=cv2.INTER_LINEAR)
            padded=cv2.copyMakeBorder(resized,geometry['top'],geometry['bottom'],geometry['left'],geometry['right'],
                                      cv2.BORDER_CONSTANT,value=(114,114,114))
            tensor=torch.from_numpy(np.ascontiguousarray(padded[:,:,::-1].transpose(2,0,1))).unsqueeze(0)
            tensor=tensor.to(device).float()/255.
            output=model(tensor)
            detections=non_max_suppression(output,conf_thres=.001,iou_thres=.7,classes=[0],
                agnostic=False,multi_label=False,max_det=20,nc=1,max_time_img=10.)[0].detach().cpu().tolist()
            for detection in detections:
                require(len(detection)==40 and detection[5]==0,'NMS输出不是单类17xy布局')
                point_pairs=[detection[6+2*i:8+2*i] for i in range(17)]
                row=inverse_prediction(image_id,detection[:4],point_pairs,detection[4],
                                       (entry['width'],entry['height']))
                if row is not None:predictions.append(row)
                else:discarded_empty_boxes+=1
            processed.append(image_id)
    return predictions,processed,dict(device=device,torch_version=torch.__version__,
        ultralytics_version=ultralytics.__version__,numpy_version=np.__version__,opencv_version=cv2.__version__,
        discarded_degenerate_clipped_boxes=discarded_empty_boxes,deterministic_algorithms=True)


def validate_prediction_receipt(receipt, context, prediction_path):
    require(receipt.get('schema')=='futsalmot_coco_pose_dev_predictions_v1'
            and receipt.get('binding')==context['binding']
            and receipt.get('prediction_sha256')==sha256(prediction_path)
            and receipt.get('postprocess')==POSTPROCESS and receipt.get('score_policy')==SCORE_POLICY
            and receipt.get('physical_visibility')=='unknown_not_predicted'
            and receipt.get('test_read') is False,'预测收据未绑定当前开发集/文件/固定后处理')
    require(digest(receipt.get('checkpoint',{}).get('sha256')),'预测来源缺模型SHA')
    return receipt['processed_image_ids']
