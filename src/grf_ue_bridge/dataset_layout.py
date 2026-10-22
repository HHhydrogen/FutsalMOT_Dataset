"""把每seed单片段的数据集原位平铺；保留媒体身份和变更前文本。"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil

from .seed_dataset import _atomic_json, _hash, _read, _safe_path, verify_clip


def flatten_dataset(root, backup, *, apply=False):
    """只接受每组恰好一个片段；所有冲突先检查，再移动目录。"""
    root, backup = Path(root).resolve(), Path(backup).resolve()
    index = _read(root / 'dataset.json')
    if index.get('layout') == 'seed_flat_v1':
        return {'layout':'seed_flat_v1','already_flat':True,'groups':len(index['groups'])}
    plans = []
    mapping = {}
    for reference in index['groups']:
        group_file = _safe_path(root, reference['path'])
        group = _read(group_file)
        if len(group['clips']) != 1:
            raise ValueError('每seed必须恰好一个片段，不能合并不同轨迹')
        source = _safe_path(group_file.parent, group['clips'][0]['path']).parent
        if source.parent.name != 'clips' or source.parent.parent != group_file.parent:
            raise ValueError('只迁移规范旧多片段目录')
        if {p.name for p in source.iterdir()} != {'clip.json','checksums.json','trajectory.jsonl','cameras'}:
            raise ValueError('clip目录包含未声明产物')
        for child in source.iterdir():
            if (group_file.parent / child.name).exists():
                raise ValueError('目标已存在，拒绝覆盖：{}'.format(child.name))
        mapping[(source / 'clip.json').relative_to(root).as_posix()] = (group_file.parent / 'clip.json').relative_to(root).as_posix()
        plans.append((group_file,group,source))
    if backup == root or root in backup.parents:
        raise ValueError('备份必须位于数据集之外')
    report = {'schema':'futsalmot_layout_migration','version':1,'from_layout':'seed_clips_v1',
        'to_layout':'seed_flat_v1','counts':index['counts'],'clip_path_mapping':mapping,
        'media_reencoded':False,'media_rehashed':False}
    if not apply:
        return report
    backup.mkdir(parents=True, exist_ok=False)
    lock = root / '.seed_dataset.lock'
    descriptor = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.close(descriptor)
    try:
        # 留存所有会改写的文本；媒体在同卷移动，不制作第二份。
        for path in [root/'dataset.json'] + [p for p in (root/'shared').rglob('*.json')]:
            destination = backup / path.relative_to(root)
            destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(path,destination)
        for group_file,group,source in plans:
            for path in (group_file,source/'clip.json',source/'checksums.json'):
                destination = backup / path.relative_to(root)
                destination.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(path,destination)
        _atomic_json(backup/'migration.json',dict(report,stage='moving'))
        for group_file,group,source in plans:
            for child in list(source.iterdir()):
                identity = child.stat().st_ino
                target = group_file.parent / child.name
                child.rename(target)
                if target.stat().st_ino != identity:
                    raise ValueError('同卷移动改变了文件身份')
            source.rmdir()
            source.parent.rmdir()
            group['clips'][0]['path'] = 'clip.json'
            _atomic_json(group_file,group)
        # 先更新未依赖anchor的完整基准，再更新其他clip指向它的引用。
        digest_by_ref = {}
        for group_file,_,_ in plans:
            path = group_file.parent / 'clip.json'
            clip = _read(path)
            if not clip.get('admission',{}).get('full_audit_anchor'):
                digest_by_ref[path.relative_to(root).as_posix()] = _hash(path)
        for group_file,_,_ in plans:
            path = group_file.parent / 'clip.json'
            clip = _read(path)
            anchor = clip.get('admission',{}).get('full_audit_anchor')
            if anchor:
                anchor['clip_ref'] = mapping.get(anchor['clip_ref'],anchor['clip_ref'])
                anchor['clip_sha256'] = digest_by_ref[anchor['clip_ref']]
                _atomic_json(path,clip)
            sums_path = group_file.parent / 'checksums.json'
            sums = _read(sums_path)
            sums['clip.json'] = _hash(path)
            _atomic_json(sums_path,sums)
        policy_path = root / 'shared/quality_policy.json'
        if policy_path.exists():
            policy = _read(policy_path)
            reference = policy.get('full_audit_anchor_ref')
            if reference in mapping:
                policy['full_audit_anchor_ref'] = mapping[reference]
                _atomic_json(policy_path,policy)
        index['layout'] = 'seed_flat_v1'
        _atomic_json(root/'dataset.json',index)
        for group_file,_,_ in plans:
            verify_clip(group_file.parent,verification_level='structure')
        report.update(finished_utc=datetime.now(timezone.utc).isoformat(),stage='completed')
        _atomic_json(backup/'migration.json',report)
        _atomic_json(root/'shared/layout_migration.json',report)
        return report
    finally:
        lock.unlink()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    parser.add_argument('--backup',type=Path,required=True)
    parser.add_argument('--apply',action='store_true')
    args=parser.parse_args()
    print(json.dumps(flatten_dataset(args.root,args.backup,apply=args.apply),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
