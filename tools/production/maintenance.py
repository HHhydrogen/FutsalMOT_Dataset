"""清理明确属于测试的媒体，并将已验收的两个组原子迁入新版本目录。"""
import argparse
import os
from pathlib import Path
import shutil
import sys
from common import dump, load_settings, now, read, require_child, sha
from p1 import archive_text, is_reparse, remove_tree, use_runtime

PROTECTED = {'episode_paper_4k_s1021_c000_q2', 'episode_paper_4k_s1025_c000_ssd'}


def linked_scratch(directory, project):
    targets = []
    allowed = Path(project) / 'Saved/PaperPilot'
    for parent, folders, files in os.walk(str(directory), followlinks=False):
        for name in list(folders):
            child = Path(parent) / name
            if is_reparse(child):
                if name not in ('render','render_mask'):
                    raise ValueError('未识别的测试链接，停止清理：{}'.format(child))
                target = require_child(child.resolve(), allowed)
                targets.append(target)
                folders.remove(name)
        if any(is_reparse(Path(parent) / name) for name in files):
            raise ValueError('测试目录包含未识别文件链接')
    return targets


def clean_tests(settings, apply=False):
    private = Path(settings['private_root'])
    root = Path('G:/FutsalMOT_Dataset/_staging')
    targets = []
    for name in ('active_v1','long_v1'):
        path = root / name
        if path.exists():
            targets.append(path)
    if (root / '4k_v1').exists():
        targets.extend(path for path in (root / '4k_v1').iterdir() if path.is_dir() and path.name not in PROTECTED)
    plan = []
    for target in targets:
        require_child(target, root)
        if is_reparse(target):
            raise ValueError('测试顶层目录不允许为链接')
        scratch = linked_scratch(target, settings['project'])
        # 任何候选源都不得位于清理范围内。
        reports = read(settings['legacy_source_reports'])
        for report in reports:
            source = Path(report['source_episode']).resolve()
            if source == target.resolve() or target.resolve() in source.parents:
                raise ValueError('候选轨迹仍引用此目录')
        item = {'target':str(target), 'scratch':[str(path) for path in scratch]}
        if apply:
            item['archive'] = archive_text(target, private / 'archives/retired_tests' / (target.parent.name+'_'+target.name+'.zip'))
            remove_tree(target, root)
            for path in scratch:
                remove_tree(path, Path(settings['project']) / 'Saved/PaperPilot')
            item['deleted'] = True
        plan.append(item)
    receipt = {'updated_utc':now(), 'applied':apply, 'protected':sorted(PROTECTED), 'targets':plan}
    dump(private / ('test_cleanup_applied.json' if apply else 'test_cleanup_plan.json'), receipt)
    return receipt


def adopt(settings):
    use_runtime(settings)
    from grf_ue_bridge.seed_dataset import verify_clip, _index_groups
    private, root, old = Path(settings['private_root']), Path(settings['dataset_root']), Path('G:/FutsalMOT_Dataset')
    root.mkdir(parents=True, exist_ok=True)
    legacy_entries = read(settings['legacy_index'])
    entries = [next(e for e in legacy_entries if e['root_seed']==seed and e['split']!='gate') for seed in settings['adopt_seeds']]
    for entry in entries:
        status = read(private.parent.parent / 'PaperPilot/20261021_4k' / (entry['task_id']+'_continuation.json'))
        if status['status'] != 'published_and_verified':
            raise ValueError('旧批次仍在验收/发布，不允许迁移：{}'.format(entry['task_id']))
    index_path = root / 'dataset.json'
    if index_path.exists():
        index = read(index_path)
    else:
        index = read(old / 'dataset.json')
        dump(private / 'archives/legacy_dataset_index.json', index)
    # 所有共享引用均为根目录相对路径，迁移不会修改clip校验和。
    for directory, folders, files in os.walk(str(old / 'shared')):
        for name in files:
            source = Path(directory) / name
            relative = source.relative_to(old)
            if relative.as_posix() == 'shared/generation_recipe.json':
                relative = Path('shared/legacy_generation_recipe.json')
            target = require_child(root / relative, root)
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if sha(target) != sha(source):
                    raise ValueError('迁入的共享配置冲突：{}'.format(relative))
            else:
                shutil.copy2(source,target)
    for entry in entries:
        group_name = 'seed_{:08d}'.format(entry['root_seed'])
        source, target = old / 'groups' / group_name, root / 'groups' / group_name
        receipt_path = private / 'adopted' / (group_name+'.json')
        if receipt_path.exists():
            continue
        if source.exists() and target.exists():
            raise ValueError('迁移源和目标同时存在，拒绝猜测')
        candidate = source if source.exists() else target
        clip = candidate / 'clips/clip_000'
        # 两个旧批次已经独立完成全量SHA验证；迁移仅核对文本/帧集/图像头。
        verified = verify_clip(clip, verification_level='metadata')
        assert verified['frame_count'] == entry['frames_per_camera']
        group = read(candidate / 'group.json')
        assert group['root_seed'] == entry['root_seed'] and group['split']=='train'
        before = {str(p.relative_to(candidate)): (p.stat().st_ino,p.stat().st_size,p.stat().st_mtime_ns)
            for p in candidate.rglob('*') if p.is_file()}
        if source.exists():
            require_child(source, old / 'groups')
            require_child(target, root / 'groups')
            target.parent.mkdir(parents=True,exist_ok=True)
            dump(private / 'adoption_journal.json', {'source':str(source),'target':str(target),'phase':'verified_before_atomic_rename'})
            source.rename(target)
        after = {str(p.relative_to(target)): (p.stat().st_ino,p.stat().st_size,p.stat().st_mtime_ns)
            for p in target.rglob('*') if p.is_file()}
        assert before == after, '同卷迁移文件身份发生变化'
        published = read(private.parent.parent / 'PaperPilot/20261021_4k' / (entry['task_id']+'_published.json'))
        assert published['verified'] and sha(published['evidence_archive'])==published['evidence_sha256']
        dump(receipt_path, {'adopted_utc':now(), 'entry':entry, 'canonical_clip':str(target/'clips/clip_000'),
            'verified':True, 'file_identity_preserved':True, 'legacy_archive':published['evidence_archive']})
    index.update(release_id=settings['release_id'], generation_ref='shared/generation_recipe.json', resolution=[3840,2160])
    _index_groups(root,index)
    dump(old / 'dataset.json', {'schema':'futsalmot_dataset_redirect','version':1,
        'dataset_ref':'v1_no_numbers_4k/dataset.json','migration_evidence':str(private/'adopted')})
    return {'groups_adopted':len(entries),'dataset_root':str(root)}


def clean_adopted(settings):
    private = Path(settings['private_root'])
    records = sorted((private / 'adopted').glob('seed_*.json'))
    assert len(records)==len(settings['adopt_seeds'])
    done=[]
    for path in records:
        record=read(path)
        assert record['verified'] and record['file_identity_preserved']
        entry=record['entry']
        episode=Path(entry['episode'])
        if not episode.exists():
            continue
        canonical=Path(record['canonical_clip'])
        # 清理前核对正式全部文本/mask哈希和RGB硬链接，防止档案/索引被改动。
        for relative,expected in read(canonical/'checksums.json').items():
            target=require_child(canonical/relative,canonical)
            if '/rgb/' in relative:
                parts=Path(relative).parts
                assert os.path.samefile(episode/parts[1]/'img1'/parts[-1],target)
            else:
                assert sha(target)==expected, relative
        scratch=linked_scratch(episode,settings['project'])
        # 旧晴天批次的render也位于SSD；只处理收据中精确目标。
        archive=archive_text(episode,private/'archives'/('adopted_'+entry['task_id']+'.zip'))
        remove_tree(episode,'G:/FutsalMOT_Dataset/_staging/4k_v1')
        for target in scratch:
            remove_tree(target,Path(settings['project'])/'Saved/PaperPilot')
        done.append({'task_id':entry['task_id'],'archive':archive})
    dump(private/'adopted_scratch_cleaned.json',{'updated_utc':now(),'tasks':done})
    return done


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=('clean-tests','adopt','clean-adopted'))
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--apply',action='store_true')
    args=parser.parse_args()
    settings=load_settings(args.config)
    result=clean_tests(settings,args.apply) if args.action=='clean-tests' else (adopt(settings) if args.action=='adopt' else clean_adopted(settings))
    print('PRODUCTION_MAINTENANCE '+str(result),flush=True)
