"""迁移只修复可执行输入，历史证据与原始字节保持不变。"""
import importlib.util
import json
from pathlib import Path

import pytest

spec=importlib.util.spec_from_file_location('sim2real_migrate',Path(__file__).with_name('migrate_paths.py'))
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def fixture(root):
    dataset=root/'datasets/real_player_pilot_v1';dataset.mkdir(parents=True)
    image=dataset/'images/train/a.jpg';image.parent.mkdir(parents=True);image.write_bytes(b'image')
    label=dataset/'labels/train/a.txt';label.parent.mkdir(parents=True);label.write_text('0 0.5 0.5 0.2 0.2\n')
    old=module.OLD+'/datasets/real_player_pilot_v1'
    (dataset/'dataset.yaml').write_text(json.dumps({'path':old,'train':'train.txt','val':'val.txt','test':'test.txt','names':{'0':'player'}}))
    (dataset/'train.txt').write_text((old+'/images/train/a.jpg').replace('/','\\')+'\n')
    (dataset/'val.txt').write_text(old+'/images/train/a.jpg\n')
    (dataset/'test.txt').write_text('')
    (dataset/'labels/train.cache').write_bytes(b'cache with G:/FutsalMOT_Research')
    history=root/'evaluations/receipt.json';history.parent.mkdir();history.write_text(json.dumps({'weights':module.OLD+'/weights/yolo11n.pt'}))
    return dataset,history,label


def test_migration_validates_then_backs_up_only_mutable_paths(tmp_path):
    dataset,history,label=fixture(tmp_path)
    before_history=history.read_bytes();before_list=(dataset/'train.txt').read_bytes()
    dry=module.migrate(tmp_path)
    assert len(dry['changes'])==3 and (dataset/'train.txt').read_bytes()==before_list
    result=module.migrate(tmp_path,apply=True)
    assert history.read_bytes()==before_history
    assert Path((dataset/'train.txt').read_text().strip()).is_file()
    assert not (dataset/'labels/train.cache').exists()
    archive=Path(result['archive'])
    assert (archive/'backup/datasets/real_player_pilot_v1/train.txt').read_bytes()==before_list
    assert (archive/'label_caches/datasets/real_player_pilot_v1/labels/train.cache').is_file()
    assert not module.migrate(tmp_path)['changes']


def test_missing_label_rejects_before_any_config_change(tmp_path):
    dataset,history,label=fixture(tmp_path)
    before=(dataset/'dataset.yaml').read_bytes();label.unlink()
    with pytest.raises(ValueError,match='标签不存在'):module.migrate(tmp_path,apply=True)
    assert (dataset/'dataset.yaml').read_bytes()==before
    assert not (tmp_path/'path_migrations').exists()
