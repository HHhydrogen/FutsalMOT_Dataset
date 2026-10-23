"""独立v3 Pose-only评价身份适配；复用冻结v2数值，不改写任何旧收据。"""
from pathlib import Path

from . import coco_pose_dev_eval as numeric

VERSION='futsalmot_coco_pose_dev_evaluation_v3'
NUMERICAL_SOURCE_SHA256='c5f2cf886cd6549385b82504e0a5cdfab2fa16242a1ba8315fb984daef84e7ee'
PROTOCOL_SCHEMA='futsalmot_pose_location_poseonly_protocol_v3'
ADMISSION_SCHEMA='futsalmot_pose_location_poseonly_admission_v3'
CHECKPOINT_SCHEMA='futsalmot_pose_location_poseonly_checkpoint_v3'
RUN_SCHEMA='futsalmot_pose_location_poseonly_run_v3'
POLICY=dict(trainable='pose_cv4_only',model_mode='eval_with_autograd',all_bn_running_stats_frozen=True,
    backbone_neck_detector_frozen=True,visibility_supervision=False,physical_channels=51,
    output_keypoint_shape=[17,2],sigma='explicit_COCO17_OKS_SIGMA',ema_dtype='float32',
    checkpoint_choice='last1000_actual_FP32_EMA_no_best',hard_crash_budget_replay=False)
SETTINGS=dict(actual_optimizer_updates=1000,batch=4,nbs=4,gradient_accumulation=1,imgsz=640,
    optimizer='SGD',lr0=.0001,lrf=1.,momentum=.9,weight_decay=0.,warmup_epochs=0.,amp=False,
    multi_scale=False,early_stopping=False,augmentation=False,checkpoint_interval=50,
    seeds=[20261024,20261025,20261026])
require=numeric.require


def bound(reference):
    require(isinstance(reference,dict) and numeric.digest(reference.get('sha256')),'缺文件路径/SHA绑定')
    path=numeric.safe_path(reference['path'])
    require(numeric.sha256(path)==reference['sha256'],'绑定文件SHA变化: '+str(path))
    return path


def same_reference(first,second):
    require(bound(first)==bound(second) and first['sha256']==second['sha256'],'文件绑定链不一致')


def load_context(protocol_path,protocol_sha256,admission_path,admission_sha256):
    """验证v3独立授权与来源；只读JSON/源码/JSONL，不读图片或原GT文件。"""
    require(numeric.sha256(numeric.__file__)==NUMERICAL_SOURCE_SHA256,'冻结的旧评价数值实现已改变')
    protocol_path=numeric.safe_path(protocol_path);admission_path=numeric.safe_path(admission_path)
    protocol=numeric.checked_json(protocol_path,protocol_sha256)
    admission=numeric.checked_json(admission_path,admission_sha256)
    require(protocol.get('schema')==PROTOCOL_SCHEMA and protocol.get('status')=='frozen'
        and protocol.get('training_authorized') is True and protocol.get('test_split_used') is False,
        '必须使用独立冻结v3协议，不能伪装v2协议SHA')
    require(admission.get('schema')==ADMISSION_SCHEMA and admission.get('status')=='approved_poseonly_location_v3'
        and admission.get('approved_by')=='root' and admission.get('training_authorized') is True
        and admission.get('location_only') is True and admission.get('visibility_supervision') is False
        and admission.get('test_split_used') is False,'缺v3独立root准入')
    require(bound(admission['protocol'])==protocol_path and admission['protocol']['sha256']==protocol_sha256,
        'v3准入不属于实际新协议')
    require(protocol.get('policy')==admission.get('policy')==POLICY,'v3 Pose-only/BN/sigma/EMA策略错配')
    settings=protocol.get('paired_training',{})
    require(all(type(settings.get(k)) is type(v) and settings[k]==v for k,v in SETTINGS.items()),
            'v3固定预算/学习率/尺度/种子发生变化')
    require(admission.get('UE_serial_gate')==dict(all_rendering_finished=True,editor_safely_exited=True),
            '缺独立v3 UE串行准入')
    for key in ('source_v2_protocol','source_v2_admission','schedule_manifest'):
        same_reference(protocol[key],admission[key])
    source_path=bound(protocol['source_v2_protocol']);source_sha=protocol['source_v2_protocol']['sha256']
    old=numeric.load_dev_context(source_path,source_sha)
    require(old.get('frozen_protocol') is True,'v3来源必须是已冻结v2协议，不能直接冒用原候选')
    require(protocol.get('initialization')==old['protocol']['initialization']
        and protocol.get('arms')==old['protocol']['arms'],'v3更换了共同初始化或原R/RS曝光预算')
    old_admission=numeric.read_json(bound(protocol['source_v2_admission']))
    require(old_admission.get('schema')=='futsalmot_pose_location_training_admission_v1'
        and old_admission.get('status')=='approved_location_only' and old_admission.get('approved_by')=='root'
        and old_admission.get('training_authorized') is True,'v2来源准入未通过')
    same_reference(old_admission['protocol'],protocol['source_v2_protocol'])
    same_reference(old_admission['schedule_manifest'],protocol['schedule_manifest'])
    source_files=protocol.get('source_files')
    require(isinstance(source_files,list) and source_files and source_files==admission.get('source_files'),
            'v3源码绑定不完整或不一致')
    source_shas={}
    for reference in source_files:
        path=bound(reference)
        require(path.name not in source_shas,'源码basename重复，拒绝猜测Trainer身份')
        source_shas[path.name]=reference['sha256']
    for name in ('train_pose_location_control.py','train_pose_location_poseonly_v3.py'):
        require(name in source_shas,'缺训练源码绑定: '+name)
    manifest_path=bound(protocol['schedule_manifest']);manifest=numeric.read_json(manifest_path)
    require(manifest.get('schema')=='futsalmot_pose_location_paired_schedule_manifest_v1'
        and manifest.get('candidate_protocol_sha256')==old['candidate_protocol_sha256'],
        '采样表不是原固定R/S选择')
    for key in ('real_jsonl','synthetic_jsonl'):
        same_reference(manifest[key],old_admission[key])
    by_seed={}
    for seed in SETTINGS['seeds']:
        entries=[entry for entry in manifest['schedules'] if entry['seed']==seed]
        require(len(entries)==2 and {entry['arm'] for entry in entries}=={'R','RS25'}
            and len({(entry['path'],entry['sha256']) for entry in entries})==1
            and all(entry.get('updates')==1000 and entry.get('batch')==4 for entry in entries),
            'R/RS采样表不是同一完整1000更新预算')
        entry=entries[0];bound(entry)
        by_seed[seed]=dict(protocol_sha256=source_sha,
            admission_sha256=protocol['source_v2_admission']['sha256'],
            schedule_manifest_sha256=protocol['schedule_manifest']['sha256'],schedule_sha256=entry['sha256'],
            trainer_source_sha256=source_shas['train_pose_location_control.py'],
            real_jsonl_sha256=manifest['real_jsonl']['sha256'],synthetic_jsonl_sha256=manifest['synthetic_jsonl']['sha256'])
    context=dict(old)
    context.update(protocol=protocol,protocol_sha256=protocol_sha256,v3_admission_sha256=admission_sha256,
        v3_source_files=source_files,source_v2_bindings_by_seed=by_seed,
        v3_trainer_source_sha256=source_shas['train_pose_location_poseonly_v3.py'])
    context['binding']=dict(old['binding'],protocol_sha256=protocol_sha256,
        source_v2_protocol_sha256=source_sha,source_v2_admission_sha256=protocol['source_v2_admission']['sha256'],
        admission_sha256=admission_sha256,schedule_manifest_sha256=protocol['schedule_manifest']['sha256'])
    return context


def validate_checkpoint(checkpoint,run,context):
    require(checkpoint.get('schema')==CHECKPOINT_SCHEMA and run.get('schema')==RUN_SCHEMA,
            'v3入口拒绝旧checkpoint/run schema')
    for key in ('completed_updates','planned_updates','ema_updates'):
        require(type(checkpoint.get(key)) is int and checkpoint[key]==1000,'v3必须完成last1000/EMA1000')
    require(run.get('complete') is True and run.get('completed_updates')==run.get('planned_updates')==1000,
            'v3运行未完整结束')
    require(checkpoint.get('policy')==run.get('policy')==context['protocol']['policy']==POLICY,
            'v3策略没有匹配实际协议/运行')
    require(checkpoint.get('checkpoint_choice')=='last_actual_FP32_EMA_no_best'
        and run.get('checkpoint_choice')=='last1000_actual_FP32_EMA_no_best','v3不接受best/非最终EMA')
    for record in (checkpoint,run):
        require(record.get('actual_ema_fp32') is True and record.get('keypoint_shape')==[17,2]
            and record.get('physical_output_channels')==51 and record.get('visibility_supervised') is False,
            'v3数值/输出语义声明不符')
    seed=checkpoint.get('seed');arm=checkpoint.get('arm')
    require(type(seed) is int and seed in SETTINGS['seeds'] and arm in ('R','RS25')
        and run.get('seed')==seed and run.get('arm')==arm,'v3 arm/seed错配')
    require(run.get('actual_exposures')==dict(real=4000 if arm=='R' else 3000,synthetic=0 if arm=='R' else 1000),
            'v3实际R/S曝光预算不符')
    bindings=checkpoint.get('bindings',{})
    require(bindings==run.get('bindings'),'v3运行与checkpoint绑定不一致')
    expected=context['source_v2_bindings_by_seed'][seed]
    require(bindings.get('source_v2_bindings')==expected,'v3原R/S/采样表/v2来源不一致')
    for key in ('schedule_manifest_sha256','schedule_sha256','real_jsonl_sha256','synthetic_jsonl_sha256'):
        require(bindings.get(key)==expected[key],'v3更换原数据/采样表: '+key)
    for key,value in dict(protocol_sha256=context['protocol_sha256'],admission_sha256=context['v3_admission_sha256'],
                          trainer_source_sha256=context['v3_trainer_source_sha256']).items():
        require(bindings.get(key)==value and bindings.get('v3_'+key)==value,'v3新身份/别名错配: '+key)
    require(bindings.get('source_files')==context['v3_source_files'],'v3源码清单不同')
    require(run.get('real_test_read') is False and all(numeric.digest(checkpoint.get(k))
        and checkpoint[k]==run.get(k) for k in ('initial_state_sha256','exposure_chain_sha256')),
        'v3初始化/曝光链或测试封存错配')
    return dict(role='last_ema_v3',arm=arm,seed=seed,completed_updates=1000,bindings=bindings,
        initial_state_sha256=checkpoint['initial_state_sha256'],exposure_chain_sha256=checkpoint['exposure_chain_sha256'])


def validate_model(model):
    """先核实际FP32，再检查实际模型；不以float()转换掩盖FP16 EMA。"""
    import torch
    from . import pose_xy_head_transfer as transfer
    require(isinstance(model,torch.nn.Module),'缺实际EMA模型')
    require(all(not value.is_floating_point() or value.dtype==torch.float32 for value in model.state_dict().values()),
            '实际EMA含非FP32浮点状态')
    head=transfer._validate(model)
    require(tuple(head.kpt_shape)==(17,2) and head.nc==1
        and head.pose_xy_transfer.get('mode')=='exact_full_conv_then_select','v3不是共同严格xy单类Pose头')
    return model.eval()


def load_model(path,expected_sha256,role,context,run_path=None,run_sha256=None):
    import torch
    from . import pose_xy_head_transfer
    path=numeric.safe_path(path)
    require(numeric.digest(expected_sha256) and numeric.sha256(path)==expected_sha256,'v3 checkpoint SHA错误')
    require(role in ('last_ema','pretrained_reference'),'v3模型角色无效')
    if role=='pretrained_reference':
        model,metadata=numeric.load_evaluation_model(path,expected_sha256,role,context)
        return validate_model(model),metadata
    require(path.name=='last.pt','v3只能评价last.pt，不选择best')
    require(run_path is not None,'v3 last缺完成运行收据')
    checkpoint=torch.load(path,map_location='cpu',weights_only=False)
    run=numeric.checked_json(run_path,run_sha256)
    metadata=validate_checkpoint(checkpoint,run,context)
    model=validate_model(checkpoint.get('ema'))
    metadata.update(path=str(path),sha256=expected_sha256,run_receipt_path=str(numeric.safe_path(run_path)),
                    run_receipt_sha256=run_sha256)
    return model,metadata


def evaluate(context,predictions,processed_ids):
    report,log=numeric.evaluate(context,predictions,processed_ids)
    report.update(schema=VERSION,numerical_evaluator_source_sha256=NUMERICAL_SOURCE_SHA256,
                  v3_policy=POLICY,v3_evaluator_source_sha256=numeric.sha256(__file__))
    return report,log


def validate_prediction_receipt(receipt,context,prediction_path):
    require(receipt.get('schema')=='futsalmot_coco_pose_dev_predictions_v3'
        and receipt.get('binding')==context['binding']
        and receipt.get('prediction_sha256')==numeric.sha256(prediction_path)
        and receipt.get('numerical_evaluator_source_sha256')==NUMERICAL_SOURCE_SHA256
        and receipt.get('postprocess')==numeric.POSTPROCESS and receipt.get('score_policy')==numeric.SCORE_POLICY
        and receipt.get('physical_visibility')=='unknown_not_predicted' and receipt.get('test_read') is False,
        'v3预测收据来源/数值策略/身份错误')
    require(numeric.digest(receipt.get('checkpoint',{}).get('sha256')),'v3预测缺明确模型SHA')
    return receipt['processed_image_ids']
