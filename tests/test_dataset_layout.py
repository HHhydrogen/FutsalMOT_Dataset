"""平铺只改目录和引用；不能合并多片段或破坏full审计基准。"""
from pathlib import Path
import pytest
from grf_ue_bridge import dataset_layout, seed_dataset
from test_seed_dataset import make_episode, quick_policy_fixture


def test_flatten_preserves_media_and_repairs_quick_anchor(tmp_path):
    root,entry,policy=quick_policy_fixture(tmp_path)
    second=seed_dataset.promote_episode(entry,root,audit_policy=policy)
    old=Path(second['clip'])/'cameras/CineCam_01/rgb/000001.png'
    identity=seed_dataset._file_identity(old.stat())
    report=dataset_layout.flatten_dataset(root,tmp_path/'backup',apply=True)
    flat=root/'groups/seed_00001002'
    assert report['stage']=='completed'
    assert seed_dataset._file_identity((flat/'cameras/CineCam_01/rgb/000001.png').stat())==identity
    assert not (flat/'clips').exists()
    clip=seed_dataset.verify_clip(flat)
    assert clip['admission']['full_audit_anchor']['clip_ref']=='groups/seed_00001001/clip.json'
    assert dataset_layout.flatten_dataset(root,tmp_path/'other')['already_flat']


def test_flat_publisher_keeps_same_seed_idempotent(tmp_path):
    root=tmp_path/'published'
    entry=make_episode(tmp_path)
    result=seed_dataset.promote_episode(entry,root,layout='seed_flat_v1')
    assert Path(result['clip'])==root/'groups/seed_00001001'
    assert seed_dataset.promote_episode(entry,root)['idempotent']
    with pytest.raises(ValueError,match='一个连续片段'):
        seed_dataset.promote_episode(make_episode(tmp_path,task_id='another'),root)


def test_flatten_refuses_multiple_clips_without_mutation(tmp_path):
    root=tmp_path/'published'
    seed_dataset.promote_episode(make_episode(tmp_path),root)
    seed_dataset.promote_episode(make_episode(tmp_path,task_id='another'),root)
    before=(root/'dataset.json').read_bytes()
    with pytest.raises(ValueError,match='恰好一个'):
        dataset_layout.flatten_dataset(root,tmp_path/'backup',apply=True)
    assert (root/'dataset.json').read_bytes()==before
    assert not (tmp_path/'backup').exists()
