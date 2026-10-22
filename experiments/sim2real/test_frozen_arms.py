"""用已完成 B 的真实格式 receipt 验证准备 C 不会改写基线，也不会追加新发布组。"""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

spec=importlib.util.spec_from_file_location('sim2real_frozen_pipeline',Path(__file__).with_name('pipeline.py'))
pipeline=importlib.util.module_from_spec(spec);spec.loader.exec_module(pipeline)


def fixture(root,monkeypatch):
    monkeypatch.setattr(pipeline,'ROOT',root)
    real=root/'datasets/real_player_pilot_v1';real.mkdir(parents=True)
    train=[str((real/'images/train'/f'{index}.jpg').resolve()) for index in range(2)]
    pipeline.write_lists(real,{'train':train,'val':[real/'images/val/a.jpg'],'test':[real/'images/test/a.jpg']})
    pipeline.write_json(real/'provenance.json',{'protocol':'real_fixture','games':{'train':[4,6],'val':[9],'test':[7,11]}})
    protocol=root/'selection_protocol.json'
    entry={'root_seed':1001,'preset':'fixture_weather','split':'train','frame_count':6,
           'source_frame_range':[0,6],'cameras':['CineCam_01'],'resolution':[3840,2160]}
    pipeline.write_json(protocol,{'schema':'sim2real_selection_v1','groups':[entry]})
    args=SimpleNamespace(seed=20261021,protocol=protocol)
    pipeline.prepare_real_arm(args)
    b=root/'datasets/arm_real_only'
    receipt={'arm':'real_only','training_seed':args.seed,'epochs':5,'batch':1,'optimizer_steps':20,
             'expected_optimizer_steps':20,'budget':pipeline.read_json(b/'budget.json'),
             'input_sha256':{str((b/name).resolve()):pipeline.digest(b/name) for name in ('dataset.yaml','budget.json','train.txt','val.txt')}}
    run=root/'runs/real_only_seed20261021_e5'
    checkpoint=run/'weights/last.pt';checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b'fixture checkpoint, never loaded')
    receipt.update({'evaluation_checkpoint':'weights/last.pt','last_checkpoint_sha256':pipeline.digest(checkpoint)})
    pipeline.write_json(run/'training_receipt.json',receipt)
    synthetic=root/'datasets/synthetic_pilot_v1';synthetic.mkdir()
    images=[synthetic/'images/train'/f'seed_00001001_clip_000_CineCam_01_{index:06d}.png' for index in range(1,7)]
    pipeline.write_lists(synthetic,{'train':images,'val':[],'test':[]})
    pipeline.write_json(synthetic/'provenance.json',{'protocol':'canonical_seed_grouped_visible_player_v1',
                         'canonical_root':'G:/FutsalMOT_Dataset','dataset_index_sha256':'a'*64,
                         'clips':[{**entry,'images':6,'camera_resolutions':{'CineCam_01':[3840,2160]}}],
                         'root_seed_split':{'1001':'train'},'counts':{'train':6,'val':0,'test':0}})
    return args,b,synthetic,run


def bytes_in(directory):
    return {path.relative_to(directory).as_posix():path.read_bytes() for path in directory.rglob('*') if path.is_file()}


def test_completed_b_inputs_and_receipt_remain_byte_identical(tmp_path,monkeypatch):
    args,b,synthetic,run=fixture(tmp_path,monkeypatch)
    before=bytes_in(b);receipt_before=bytes_in(run)
    pipeline.prepare_arms(args)
    pipeline.prepare_real_arm(args)
    assert bytes_in(b)==before and bytes_in(run)==receipt_before
    c=tmp_path/'datasets/arm_real_plus_synthetic'
    budget=pipeline.read_json(c/'budget.json')
    assert budget['epoch_samples']==4 and budget['unique_real_annotation_budget']==2
    assert budget['synthetic_dataset_sha256']==pipeline.digest(synthetic/'provenance.json')
    assert (c/'source_snapshot/synthetic_provenance.json').read_bytes()==(synthetic/'provenance.json').read_bytes()
    assert (c/'source_snapshot/baseline_receipts/real_only_seed20261021_e5.json').read_bytes()==(run/'training_receipt.json').read_bytes()
    pipeline.verify_frozen_arm(c)


def test_new_canonical_groups_do_not_silently_enter_existing_c(tmp_path,monkeypatch,capsys):
    args,b,synthetic,run=fixture(tmp_path,monkeypatch)
    pipeline.prepare_arms(args)
    c=tmp_path/'datasets/arm_real_plus_synthetic';before=bytes_in(c)
    with (synthetic/'train.txt').open('a',encoding='utf-8') as stream:
        stream.write(str(synthetic/'images/train/seed_00001002_clip_000_CineCam_01_000001.png')+'\n')
    provenance=pipeline.read_json(synthetic/'provenance.json');provenance['counts']['train']=7
    provenance['root_seed_split']['1002']='train'
    pipeline.write_json(synthetic/'provenance.json',provenance)
    pipeline.prepare_arms(args)
    assert bytes_in(c)==before
    assert 'ARMS_FROZEN_REUSED' in capsys.readouterr().out
    assert '1002' not in (c/'train.txt').read_text(encoding='utf-8')


@pytest.mark.parametrize('name',['dataset.yaml','budget.json','train.txt','val.txt'])
def test_changed_b_frozen_input_rejects_before_c_creation(tmp_path,monkeypatch,name):
    args,b,synthetic,run=fixture(tmp_path,monkeypatch)
    # 仅改变 JSON 空白或列表换行风格，也不能因语义相同而静默接受。
    path=b/name;data=path.read_bytes()
    if name.endswith('.txt'):
        newline=b'\n' if b'\r\n' in data else b'\r\n'
        path.write_bytes(newline.join(data.splitlines())+newline)
    else:path.write_bytes(data+b'\n')
    before=path.read_bytes()
    with pytest.raises(ValueError,match='SHA256'):pipeline.prepare_arms(args)
    assert path.read_bytes()==before
    assert not (tmp_path/'datasets/arm_real_plus_synthetic').exists()


def test_missing_completed_receipt_rejects_without_rebuilding_b(tmp_path,monkeypatch):
    args,b,synthetic,run=fixture(tmp_path,monkeypatch)
    (run/'training_receipt.json').unlink();before=bytes_in(b)
    with pytest.raises(ValueError,match='完成 receipt'):pipeline.prepare_arms(args)
    assert bytes_in(b)==before and not (tmp_path/'datasets/arm_real_plus_synthetic').exists()


def test_frozen_c_conflict_and_new_sampling_seed_are_explicit_errors(tmp_path,monkeypatch):
    args,b,synthetic,run=fixture(tmp_path,monkeypatch)
    pipeline.prepare_arms(args)
    with pytest.raises(ValueError,match='不同采样 seed'):
        pipeline.prepare_arms(SimpleNamespace(seed=20261022,protocol=args.protocol))
    c=tmp_path/'datasets/arm_real_plus_synthetic'
    (c/'source_snapshot/synthetic_provenance.json').write_bytes(b'changed')
    with pytest.raises(ValueError,match='冻结实验输入已改变'):pipeline.prepare_arms(args)


def test_legacy_synthetic_provenance_is_not_used_for_c(tmp_path,monkeypatch):
    args,b,synthetic,run=fixture(tmp_path,monkeypatch)
    value=pipeline.read_json(synthetic/'provenance.json');value['protocol']='synthetic_episode_seed_grouped_v1'
    pipeline.write_json(synthetic/'provenance.json',value)
    with pytest.raises(ValueError,match='canonical'):pipeline.prepare_arms(args)
    assert not (tmp_path/'datasets/arm_real_plus_synthetic').exists()


def test_missing_selected_weather_rejects_first_c_without_touching_b(tmp_path,monkeypatch):
    args,b,synthetic,run=fixture(tmp_path,monkeypatch);before=bytes_in(b)
    protocol=pipeline.read_json(args.protocol)
    protocol['groups'].append({**protocol['groups'][0],'root_seed':1002,'preset':'second_weather'})
    pipeline.write_json(args.protocol,protocol)
    with pytest.raises(ValueError,match='缺少.*1002'):pipeline.prepare_arms(args)
    assert bytes_in(b)==before and not (tmp_path/'datasets/arm_real_plus_synthetic').exists()


def test_frozen_selection_protocol_is_an_immutable_source_input(tmp_path,monkeypatch):
    args,b,synthetic,run=fixture(tmp_path,monkeypatch)
    pipeline.prepare_arms(args)
    c=tmp_path/'datasets/arm_real_plus_synthetic'
    assert (c/'source_snapshot/selection_protocol.json').read_bytes()==args.protocol.read_bytes()
    args.protocol.write_bytes(args.protocol.read_bytes()+b'\n')
    with pytest.raises(ValueError,match='不同选择协议'):pipeline.prepare_arms(args)


def test_gpu_window_preflight_checks_b_checkpoint_and_c_without_writes(tmp_path,monkeypatch):
    args,b,synthetic,run=fixture(tmp_path,monkeypatch)
    pipeline.prepare_arms(args);before=bytes_in(tmp_path)
    pipeline.verify_arms(args)
    assert bytes_in(tmp_path)==before
    (run/'weights/last.pt').write_bytes(b'changed checkpoint')
    with pytest.raises(ValueError,match='checkpoint'):pipeline.verify_arms(args)
