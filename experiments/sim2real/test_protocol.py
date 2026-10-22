# -*- coding: utf-8 -*-
"""训练数据准入和坐标边界回归，避免未验收数据静默进入实验。"""
import importlib.util
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('sim2real_pipeline',Path(__file__).with_name('pipeline.py'))
pipeline=importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)


def make_episode(tmp_path):
    payloads={'audit/soak_audit_report.json':{'passed':True,'exit_code':0,'validation':{'level':'full','passed':True,'exit_code':0}},'render_summary.json':{'status':'success'},'provenance/appearance_application.json':{'state':'restored'},'provenance/environment_application.json':{'state':'restored'},'meta.json':{'randomness':{'root_seed':1001}}}
    payloads['pilot_ready.json']={'root_seed':1001,'task_id':'test_task','status':'audit_passed','appearance_restored':True,'environment_restored':True}
    for path,payload in payloads.items():pipeline.write_json(tmp_path/path,payload)
    return {'episode':str(tmp_path),'root_seed':1001,'task_id':'test_task'}


def test_quick_audit_cannot_enter_training_even_with_top_level_passed(tmp_path):
    entry=make_episode(tmp_path)
    path=tmp_path/'audit/soak_audit_report.json'
    report=pipeline.read_json(path);report['validation']['level']='quick'
    pipeline.write_json(path,report)
    with pytest.raises(ValueError,match='full'):pipeline.verify_episode(entry)


def test_failed_restore_cannot_enter_training(tmp_path):
    entry=make_episode(tmp_path)
    pipeline.write_json(tmp_path/'provenance/appearance_application.json',{'state':'restore_failed'})
    with pytest.raises(ValueError,match='未恢复'):pipeline.verify_episode(entry)


def test_manifest_seed_must_match_grouping_key(tmp_path):
    entry=make_episode(tmp_path);entry['root_seed']=1002
    with pytest.raises(ValueError,match='seed|entry'):pipeline.verify_episode(entry)


def test_full_audited_restored_episode_is_accepted(tmp_path):
    entry=make_episode(tmp_path)
    assert pipeline.verify_episode(entry)==tmp_path


def test_full_audit_without_ready_is_not_accepted(tmp_path):
    entry=make_episode(tmp_path)
    payload=pipeline.read_json(tmp_path/'pilot_ready.json');payload['environment_restored']=False
    pipeline.write_json(tmp_path/'pilot_ready.json',payload)
    with pytest.raises(ValueError,match='pilot_ready'):pipeline.verify_episode(entry)


def test_bbox_clips_to_image_without_negative_training_coordinates():
    assert pipeline.bbox_to_yolo((-5,-5,15,15),100,100)=='0 0.05000000 0.05000000 0.10000000 0.10000000'
    assert pipeline.bbox_to_yolo((105,105,10,10),100,100) is None
    with pytest.raises(ValueError):pipeline.bbox_to_yolo((1,1,float('nan'),10),100,100)


def test_same_seed_cannot_leak_across_camera_or_clip_splits(tmp_path):
    entries=[{'episode':str(tmp_path/'clip_0'),'root_seed':42,'split':'train','frames_per_camera':50},
             {'episode':str(tmp_path/'clip_1'),'root_seed':42,'split':'validation','frames_per_camera':100}]
    with pytest.raises(ValueError,match='跨 split'):pipeline.validate_entries(entries)


def test_variable_clip_lengths_share_seed_group_without_leak(tmp_path):
    pipeline.validate_entries([{'episode':str(tmp_path/'clip_0'),'root_seed':42,'split':'train','frames_per_camera':50},
                               {'episode':str(tmp_path/'clip_1'),'root_seed':42,'split':'train','frames_per_camera':100}])


def test_temporal_sampling_spans_long_clips_and_balances_cameras():
    images=[f'/images/train/seed_{seed:08d}_clip_000_CineCam_{camera:02d}_{frame:06d}.png'
            for seed in (1001,1002,1003) for camera in range(1,5) for frame in range(1,1801)]
    selected,receipt=pipeline.temporal_sample(images,300,20261021)
    assert len(selected)==len(set(selected))==300
    assert len(receipt)==12 and {value['selected'] for value in receipt.values()}=={25}
    for value in receipt.values():
        assert value['frame_indices'][0]<100 and value['frame_indices'][-1]>1700
    assert pipeline.temporal_sample(list(reversed(images)),300,20261021)==(selected,receipt)


def test_temporal_sampling_redistributes_short_sequence_budget():
    images=['/a/CineCam_01_000001.png']+[f'/a/CineCam_02_{frame:06d}.png' for frame in range(1,10)]
    selected,receipt=pipeline.temporal_sample(images,6,10)
    assert len(selected)==len(set(selected))==6
    assert receipt['CineCam_01']['selected']==1 and receipt['CineCam_02']['selected']==5
