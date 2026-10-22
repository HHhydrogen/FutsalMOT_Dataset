"""用临时结果验证跨组设备与 GT 契约，不启动推理。"""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

spec=importlib.util.spec_from_file_location('sim2real_compare_pipeline',Path(__file__).with_name('pipeline.py'))
pipeline=importlib.util.module_from_spec(spec);spec.loader.exec_module(pipeline)


def fixture(tmp_path,monkeypatch):
    monkeypatch.setattr(pipeline,'ROOT',tmp_path)
    for name,score in [('a',0.6),('b',0.65)]:
        pipeline.write_json(tmp_path/'evaluations'/name/'detection_metrics.json',{
            'split':'val','weights_sha256':name*64,'dataset_provenance_sha256':'d'*64,
            'inference':{'device':'0','imgsz':640,'batch':4},
            'mAP50_95':score,'mAP50':0.9,'precision':0.8,'recall':0.9})
        pipeline.write_json(tmp_path/'tracking_metrics'/name/'metrics.json',{'split':'val',
            'metrics':{'HOTA':score,'DetA':score,'AssA':score,'IDF1':score,'MOTA':score,'IDSW':4 if name=='a' else 5}})
        pipeline.write_json(tmp_path/'tracking'/name/'tracking_receipt.json',{
            'split':'val','weights_sha256':name*64,'device':'0','imgsz':640,
            'dataset_provenance_sha256':'d'*64,'tracker_config_sha256':'f'*64})
    return SimpleNamespace(names=['a','b'],split='val')


def test_same_gpu_comparison_keeps_negative_id_switch_change(tmp_path,monkeypatch):
    pipeline.compare(fixture(tmp_path,monkeypatch))
    rows=pipeline.read_json(tmp_path/'comparison/comparison_val.json')['rows']
    assert rows[1]['delta_vs_first']['IDSW']==1
    assert rows[1]['inference_device']=='0'


@pytest.mark.parametrize('kind,key,value',[
    ('tracking','device','cpu'),('tracking','imgsz',1280),
    ('tracking','dataset_provenance_sha256','e'*64),('tracking','tracker_config_sha256','e'*64)])
def test_mixed_device_resolution_gt_or_tracker_rejects(tmp_path,monkeypatch,kind,key,value):
    args=fixture(tmp_path,monkeypatch)
    path=tmp_path/'tracking/b/tracking_receipt.json'
    receipt=pipeline.read_json(path);receipt[key]=value;pipeline.write_json(path,receipt)
    with pytest.raises(ValueError):pipeline.compare(args)
    assert not (tmp_path/'comparison/comparison_val.json').exists()


def test_two_internally_consistent_but_different_devices_reject(tmp_path,monkeypatch):
    args=fixture(tmp_path,monkeypatch)
    path=tmp_path/'tracking/b/tracking_receipt.json';receipt=pipeline.read_json(path)
    receipt['device']='cpu';pipeline.write_json(path,receipt)
    path=tmp_path/'evaluations/b/detection_metrics.json';receipt=pipeline.read_json(path)
    receipt['inference']['device']='cpu';pipeline.write_json(path,receipt)
    with pytest.raises(ValueError,match='各组'):pipeline.compare(args)


def test_legacy_missing_device_metadata_requires_fresh_named_reevaluation(tmp_path,monkeypatch):
    args=fixture(tmp_path,monkeypatch)
    path=tmp_path/'evaluations/a/detection_metrics.json';receipt=pipeline.read_json(path)
    del receipt['inference'];pipeline.write_json(path,receipt)
    with pytest.raises(ValueError,match='新名称重评'):pipeline.compare(args)
