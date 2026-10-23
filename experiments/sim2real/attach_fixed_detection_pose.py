"""给固定dev151/152/153 raw detections附加整帧17xy；不运行Tracker。"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import sys

CODE=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(CODE/'src'))
from grf_ue_bridge import fixed_detection_pose_attach_v2 as attachment
import prepare_pose_association_dev as fixed_cache


def safe(path):
    path=Path(path).absolute()
    require=attachment.require
    require(not any(p.lower() in ('test','test2017','test2014')
        or re.search(r'snmot[-_]?0*(116|187)(?:\D|$)',p.lower()) for p in path.parts),
        '禁止封存test或未登记测试目录')
    require('..' not in path.parts,'拒绝父目录逃逸')
    return fixed_cache.plain(path)


def check_file(path,expected):
    path=safe(path)
    attachment.require(attachment.is_sha(expected) and fixed_cache.sha(path)==expected,'文件SHA不符: '+str(path))
    return path


def validate_cache_source(manifest,frames):
    """只读源登记JSON；不读其中GT文件，不把文本GT用于选Pose或匹配。"""
    source=manifest['source_manifest'];path=check_file(source['path'],source['sha256']);source=fixed_cache.read(path)
    attachment.require(source.get('schema')=='futsalmot_pose_association_dev_source_candidate_v1'
        and source.get('frames')==450 and source.get('test_split_used') is False
        and [r['sequence_id'] for r in source['sequences']]==list(attachment.SEQUENCES),'非登记的固定开发源')
    expected={}
    for row in source['sequences']:
        attachment.require(row['game_id']=='9' and row['role']=='development_only' and row['camera_id']=='broadcast',
                           '来源不是game9公开开发')
        for frame in row['frames']:
            image=safe(fixed_cache.source_paths(row['sequence_id'])/'img1'/f"{frame['frame_number']:06d}.jpg")
            attachment.require(safe(frame['image_file'])==image,'源图像离开固定151/152/153路径')
            expected[frame['frame_key']]=dict(frame,width=row['width'],height=row['height'],image_file=str(image))
    attachment.require(len(expected)==450 and len(frames)==450,'来源帧集合错误')
    for frame in frames:
        attachment.validate_fixed_frame(frame);reference=expected.get(frame['frame_key'])
        attachment.require(reference is not None and safe(frame['image_file'])==safe(reference['image_file'])
            and all(frame[k]==reference[k] for k in ('frame_index','frame_number','width','height')),
            '检测缓存与原图帧/尺寸错配')
    return source


def write(path,value):
    with Path(path).open('x',encoding='utf-8') as handle:
        json.dump(value,handle,indent=2,ensure_ascii=False,allow_nan=False);handle.write('\n')


def execute(cache_path,cache_sha,configuration_path,configuration_sha,checkpoint_path,checkpoint_sha,
            member,output,device,explicit=False):
    attachment.require(explicit,'本入口仅显式--execute-inference时才读图/推理')
    cache_path=check_file(cache_path,cache_sha)
    manifest,frames=fixed_cache.load_fixed_cache(cache_path,cache_sha)
    validate_cache_source(manifest,frames)
    configuration_path=check_file(configuration_path,configuration_sha);config=fixed_cache.read(configuration_path)
    attachment.validate_configuration(config)
    checkpoint_path=check_file(checkpoint_path,checkpoint_sha)
    attachment.require(device in ('cpu','cuda:0'),'只允许显式cpu/cuda:0')
    os.environ.update(YOLO_AUTOINSTALL='false',YOLO_OFFLINE='true',WANDB_MODE='disabled',COMET_MODE='DISABLED',
                      CUBLAS_WORKSPACE_CONFIG=':4096:8')
    if device=='cuda:0':
        import csv
        import subprocess
        attachment.require(os.name=='nt','当前GPU串行进程门禁只验证Windows')
        state=subprocess.run(['tasklist','/FI','IMAGENAME eq UnrealEditor.exe','/FO','CSV','/NH'],
            check=True,capture_output=True,text=True)
        attachment.require(not any(row and row[0].lower()=='unrealeditor.exe' for row in csv.reader(state.stdout.splitlines())),
                           'UE仍运行，禁止GPU附件推理')
    output=safe(output);attachment.require(not output.exists(),'保留已有输出，必须使用新目录')
    output.mkdir(parents=True)
    write(output/'started.json',dict(status='started_not_complete',fixed_detection_set_sha256=cache_sha,
        checkpoint_sha256=checkpoint_sha,configuration_sha256=configuration_sha,device=device,test_read=False))
    try:
        import cv2
        import numpy as np
        import torch
        import ultralytics
        model,model_identity=attachment.load_checkpoint(checkpoint_path,checkpoint_sha,member)
        model=model.to(device).float().eval()
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        torch.use_deterministic_algorithms(True)
        counts=Counter();partial=output/'attachments.partial.jsonl'
        with partial.open('x',encoding='utf-8') as handle:
            for frame in frames:
                raw=safe(frame['image_file']).read_bytes()
                image=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
                attachment.require(image is not None and image.shape[:2]==(frame['height'],frame['width']),
                                   '开发原图解码/尺寸错误')
                predictions=attachment.infer_image(image,model,device,config)
                record=attachment.attach_frame(frame,predictions,model_sha256=checkpoint_sha,
                    fixed_detection_set_sha256=cache_sha,config=config)
                attachment.validate_attachment_frame(frame,record,model_sha256=checkpoint_sha,
                    fixed_detection_set_sha256=cache_sha,config=config)
                record['image_content_sha256']=hashlib.sha256(raw).hexdigest()
                handle.write(json.dumps(record,ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n')
                counts['frames']+=1;counts['fixed_detections']+=len(frame['detections'])
                for row in record['attachments']:
                    counts['prediction_valid' if row['prediction_valid'] else 'bbox_fallback']+=1
                    if row['prediction_rejection']:counts['reason:'+row['prediction_rejection']]+=1
        final=output/'attachments.jsonl';partial.rename(final)
        result=dict(schema='futsalmot_fixed_detection_pose_attachment_manifest_v1',status='complete_pose_attachment_not_tracking',
            fixed_detection_cache=dict(path=str(cache_path),sha256=cache_sha),fixed_detection_set_sha256=cache_sha,
            source_manifest=manifest['source_manifest'],checkpoint=model_identity,
            configuration=dict(path=str(configuration_path),sha256=configuration_sha,values=config),
            attachments_jsonl=dict(path=str(final),sha256=fixed_cache.sha(final),frames=450),counts=dict(counts),
            runtime=dict(device=device,torch=torch.__version__,ultralytics=ultralytics.__version__,opencv=cv2.__version__),
            source_code_sha256={str(path):fixed_cache.sha(path) for path in
                (Path(__file__),Path(attachment.__file__),Path(fixed_cache.__file__))},
            frames_order='151,152,153; each frame_number1..150 including empty detections',
            images_hashed_during_the_same_required_decode_read_only=True,detections_modified=False,
            fixed_boxes_used_for_crops=False,GT_used_for_selection=False,physical_visibility='unknown',
            joint_confidence_available=False,tracking_executed=False,test_read=False,
            limitations=['geometry_valid_is_not_accuracy_or_visibility','all_sequences_are_development_game9',
                         'matching_is_greedy_bbox_IoU_not_global_optimum_or_GT_pose_match',
                         'quality_configuration_candidate_until_separately_registered'])
        write(output/'attachment_manifest.json',result)
        return result
    except Exception as error:
        write(output/'failure.json',dict(status='failed_no_complete_attachment_claim',
            error_type=type(error).__name__,error=str(error)))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('cache-manifest','configuration','checkpoint','output'):parser.add_argument('--'+name,type=Path,required=True)
    for name in ('cache-sha256','configuration-sha256','checkpoint-sha256'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--model-member',choices=('ema','model'),default='ema')
    parser.add_argument('--device',choices=('cpu','cuda:0'),default='cpu')
    parser.add_argument('--execute-inference',action='store_true')
    args=parser.parse_args()
    result=execute(args.cache_manifest,args.cache_sha256,args.configuration,args.configuration_sha256,
        args.checkpoint,args.checkpoint_sha256,args.model_member,args.output,args.device,args.execute_inference)
    print(json.dumps(dict(status=result['status'],counts=result['counts']),ensure_ascii=False))


if __name__=='__main__':main()
