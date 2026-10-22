"""只用临时元数据验证完整天气协议，不读取真实数据或调用训练框架。"""
import importlib.util
from pathlib import Path

import pytest

spec=importlib.util.spec_from_file_location('sim2real_coverage_pipeline',Path(__file__).with_name('pipeline.py'))
pipeline=importlib.util.module_from_spec(spec);spec.loader.exec_module(pipeline)


def expected_protocol():
    return {'schema':'sim2real_selection_v1','groups':[
        {'root_seed':seed,'preset':f'weather_{seed}','split':split,
         'frame_count':100,'source_frame_range':[10,110],
         'cameras':['CineCam_01','CineCam_02'],'resolution':[3840,2160]}
        for seed,split in [(1,'train'),(2,'train'),(3,'val'),(4,'test')]]}


def record(entry):
    return {**entry,'images':entry['frame_count']*len(entry['cameras']),
            'camera_resolutions':{camera:entry['resolution'] for camera in entry['cameras']}}


def test_many_images_from_one_weather_cannot_freeze_declared_multiweather_c():
    protocol=expected_protocol()
    with pytest.raises(ValueError,match='缺少.*2'):
        pipeline.verify_protocol_coverage([record(protocol['groups'][0])],protocol,{'train'})


def test_train_coverage_does_not_unlock_gpu_until_holdouts_are_complete():
    protocol=expected_protocol();records=[record(entry) for entry in protocol['groups'][:2]]
    assert pipeline.verify_protocol_coverage(records,protocol,{'train'})['groups']==2
    with pytest.raises(ValueError,match='缺少'):
        pipeline.verify_protocol_coverage(records,protocol,{'train','val','test'})


@pytest.mark.parametrize('key,value',[('preset','wrong'),('frame_count',99),
    ('source_frame_range',[0,100]),('cameras',['CineCam_01']),('split','val')])
def test_seed_alone_cannot_substitute_wrong_weather_clip_or_split(key,value):
    protocol=expected_protocol();records=[record(entry) for entry in protocol['groups']]
    records[0][key]=value
    with pytest.raises(ValueError):pipeline.verify_protocol_coverage(records,protocol,{'train','val','test'})


def test_each_camera_must_have_declared_native_resolution():
    protocol=expected_protocol();records=[record(entry) for entry in protocol['groups']]
    records[0]['camera_resolutions']['CineCam_02']=[1920,1080]
    with pytest.raises(ValueError,match='分辨率'):
        pipeline.verify_protocol_coverage(records,protocol,{'train'})


def test_complete_protocol_is_order_independent_but_duplicates_reject():
    protocol=expected_protocol();records=[record(entry) for entry in protocol['groups']]
    assert pipeline.verify_protocol_coverage(list(reversed(records)),protocol,{'train','val','test'})=={
        'groups':4,'root_seeds':[1,2,3,4],'images':800}
    with pytest.raises(ValueError,match='重复'):
        pipeline.verify_protocol_coverage(records+[records[0]],protocol,{'train'})


def test_undeclared_training_group_cannot_change_first_freeze_pool():
    protocol=expected_protocol();records=[record(entry) for entry in protocol['groups']]
    extra=record(protocol['groups'][0]);extra['root_seed']=99;records.append(extra)
    with pytest.raises(ValueError,match='未声明'):
        pipeline.verify_protocol_coverage(records,protocol,{'train'})
