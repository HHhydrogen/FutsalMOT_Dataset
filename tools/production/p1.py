"""正式批次的P1准备、完整后处理、发布和有证据的临时文件清理。"""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import zipfile
from common import canonical_sha, dump, load_settings, now, read, require_child, sha, task_id_for_row


def archive_text(source, destination):
    """不跟随链接，只归档可审计文本；媒体保留至正式校验通过。"""
    source = Path(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for directory, folders, files in os.walk(str(source), followlinks=False):
            folders[:] = [name for name in folders if not is_reparse(Path(directory) / name)]
            for name in files:
                path = Path(directory) / name
                if path.suffix.lower() in ('.json','.jsonl','.ini','.txt','.md','.yaml','.yml','.log','.csv') and not is_reparse(path):
                    archive.write(path, path.relative_to(source).as_posix())
    with zipfile.ZipFile(destination) as archive:
        if archive.testzip() is not None:
            raise ValueError('文本归档CRC验证失败')
    return {'path':str(destination), 'sha256':sha(destination)}


def is_reparse(path):
    return bool(path.lstat().st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT) if os.name=='nt' else path.is_symlink()


def remove_tree(path, parent):
    """先验证物理目标边界，链接本身单独移除，绝不递归跟随junction。"""
    path, parent = Path(path), Path(parent).resolve()
    absolute = Path(os.path.abspath(str(path)))
    if absolute == parent or parent not in absolute.parents:
        raise ValueError('删除目标超出明确边界：{}'.format(path))
    if not path.exists() and not path.is_symlink():
        return
    if is_reparse(path):
        os.rmdir(str(path)) if path.is_dir() else path.unlink()
        return
    require_child(path, parent)
    for child in path.iterdir():
        if child.is_dir() or is_reparse(child):
            remove_tree(child, parent)
        else:
            require_child(child, parent).unlink()
    path.rmdir()


def use_runtime(settings):
    private = Path(settings['private_root'])
    receipt = read(private / 'runtime_freeze.json') if (private / 'runtime_freeze.json').exists() else {}
    runtime = require_child(receipt.get('runtime_root', private / 'runtime'), private)
    sys.path.insert(0, str(runtime / 'src'))
    return runtime


def quality_policy(settings):
    """选样清单保持冻结，用户授权的质量策略单独版本化。"""
    path = Path(settings['private_root']) / 'quality_policy.json'
    if not path.exists():
        return None
    policy = read(path)
    if policy.get('schema') != 'futsalmot_quality_policy' or policy.get('mode') not in ('quick_after_full', 'basic_after_full'):
        raise ValueError('不支持的正式质量策略')
    return policy


def prepare(settings, row):
    runtime = use_runtime(settings)
    from grf_ue_bridge.active_episode import prepare_active_episode
    from grf_ue_bridge.config.resolver import resolve_task, save_resolved_task, validate_task
    private = Path(settings['private_root'])
    task_id = task_id_for_row(row)
    entry_path = private / 'entries' / (task_id + '.json')
    if entry_path.exists():
        return read(entry_path)
    task = copy.deepcopy(read(settings['template']))
    task.update(task_id=task_id, episode_name='episode_' + task_id)
    task['export'].update(seed=row['root_seed'], num_steps=row['frames_per_camera'])
    task['audit']['expected_frames_per_camera'] = row['frames_per_camera']
    task['ue']['environment']['preset'] = row['preset']
    task['ue']['sequence_package_path'] = '/Game/FutsalMOT/Generated/DatasetV1/' + task_id
    policy = quality_policy(settings)
    task['postprocess'].update(workers=settings['cpu_workers'], validation_level='quick' if policy else 'full')
    if policy:
        task['postprocess']['formats'] = ['json']
        task['ue']['annotation_export']['export_mot'] = False
    config = private / 'configs' / (task_id + '.json')
    dump(config, task)
    kwargs = dict(dataset_root=settings['work_root'], ue_project_root=settings['project'])
    errors = validate_task(config, **kwargs)
    if errors:
        raise ValueError(errors)
    resolved = resolve_task(config, **kwargs)
    if Path(resolved.repo_root).resolve() != runtime.resolve():
        raise ValueError('P1解析器并非来自冻结runtime')
    resolved_path = save_resolved_task(resolved, runtime)
    report = next(e for e in read(private / 'candidate_reports.json') if e['root_seed']==row['root_seed'])
    if sha(Path(row['source_episode']) / 'frames.jsonl') != row['source_frames_sha256']:
        raise ValueError('源轨迹指纹变化')
    crop = prepare_active_episode(resolved, row['source_episode'], report)
    assert crop['frame_count'] == row['frames_per_camera']
    entry = dict(row, task_id=task_id, config=str(config), resolved=str(resolved_path),
        episode=resolved.dataset_episode_dir, resolution=settings['resolution'], camera_count=4,
        source_capture_steps=settings['source_steps'], active_policy_sha256=report['policy_sha256'],
        active_qc=report['selected_windows'][0])
    # 持久会话逐机位收尾：SSD保留全组EXR，RGB临时副本只同时保留一个机位。
    execution_path = private / 'execution_policy.json'
    execution = read(execution_path) if execution_path.exists() else {}
    retained_session = execution.get('scene_lifecycle') == 'retain'
    upper = row['frames_per_camera'] * (4 * 50000000 + (20000000 if retained_session else 0))
    free = shutil.disk_usage('D:/').free
    route = []
    if free >= upper + settings['d_free_reserve_gib'] * 1024**3:
        for index in range(1,5):
            camera = 'CineCam_{:02d}'.format(index)
            folders = [('render_mask', private / 'exr_scratch' / 'masks' / task_id / camera),
                ('render', private / 'exr_scratch' / 'raw' / task_id / camera)] if retained_session else [
                ('render_mask', private / 'exr_scratch' / task_id / camera)]
            for folder, target in folders:
                link = Path(entry['episode']) / camera / folder
                if link.exists() or target.exists():
                    raise ValueError('scratch目标已存在，拒绝覆盖')
                require_child(target, private)
                target.mkdir(parents=True)
                link.parent.mkdir(parents=True, exist_ok=True)
                def quote(value):
                    return "'" + str(value).replace("'", "''") + "'"
                subprocess.run(['powershell','-NoProfile','-Command',
                    "$ErrorActionPreference='Stop'; New-Item -ItemType Junction -Path " + quote(link) + ' -Target ' + quote(target) + ' | Out-Null'], check=True)
                assert link.resolve() == target.resolve()
                route.append({'link':str(link), 'target':str(target)})
    entry['scratch'] = {'mode':('RAW_SSD_SERIAL_CAMERAS' if retained_session else 'EXR_SSD_RGB_G') if route else 'all_G', 'paths':route,
        'ssd_upper_bytes':upper, 'ssd_free_before_bytes':free}
    if retained_session:
        entry.update(raw_disk='D' if route else 'G', raw_root=str(private / 'exr_scratch/raw') if route else None,
            queue=execution['image_write_queue_size'], scene_lifecycle='retain')
    dump(entry_path, entry)
    return entry


def postprocess(settings, entry):
    runtime = use_runtime(settings)
    from grf_ue_bridge.config.resolver import load_resolved_task
    from grf_ue_bridge.workflows.task_postprocess import run_postprocess
    from grf_ue_bridge.workflows.task_audit import main as audit
    from PIL import Image, ImageDraw
    episode = Path(entry['episode'])
    summary = read(episode / 'render_summary.json')
    if summary['status'] != 'success':
        raise ValueError('渲染未成功，不得启动正式后处理')
    from grf_ue_bridge.scene_lifecycle import retained_completion
    retained = retained_completion(episode)
    for kind in ('appearance','environment'):
        report = read(episode / ('provenance/' + kind + '_application.json'))
        if report['state'] != ('applied' if retained else 'restored') or report.get('errors'):
            raise ValueError('{}恢复未通过'.format(kind))
    resolved = load_resolved_task(Path(entry['resolved']))
    policy = quality_policy(settings)
    level = 'quick' if policy else 'full'
    basic = bool(policy and policy['mode'] == 'basic_after_full')
    execution_path = Path(settings['private_root']) / 'execution_policy.json'
    execution = read(execution_path) if execution_path.exists() else {}
    workers = int(execution.get('postprocess_workers', settings['cpu_workers']))
    resolved.postprocess = dict(resolved.postprocess, workers=workers)
    if policy:
        resolved.postprocess = dict(resolved.postprocess, formats=['json'], validation_level='quick')
    dump(episode / 'provenance/postprocess_execution.json', {'workers':workers,
        'cryptomatte_sha256':sha(runtime / 'src/grf_ue_bridge/cryptomatte.py'), 'editor_closed':not bool(retained), 'scene_lifecycle':'retain' if retained else 'restore',
        'effective_formats':resolved.postprocess.get('formats'), 'audit_level':'none' if basic else level,
        'quality_policy_sha256':sha(Path(settings['private_root']) / 'quality_policy.json') if policy else None})
    assert run_postprocess(resolved, skip_validate=True, print_fn=lambda line:print(line,flush=True)) == 0
    if basic:
        # 标注生成必需，逐组像素抽审和预览不再运行；发布器只做结构准入。
        dump(episode / 'pilot_ready.json', dict(entry, status='postprocess_complete',
            rgb_frames=entry['frames_per_camera']*4, mask_frames=entry['frames_per_camera']*4,
            appearance_restored=not bool(retained), environment_restored=not bool(retained),
            scene_retained=bool(retained), audit_level='none', full_audit_passed=False))
        return
    args = ['--input',str(episode),'--episode',str(episode),'--expected-cameras','4',
        '--expected-frames-per-camera',str(entry['frames_per_camera']),'--render-fps','10',
        '--pose-skip','true','--environment-required','true','--mot-required','false' if policy else 'true',
        '--yolo-det-required','false' if policy else 'true','--yolo-seg-required','false' if policy else 'true','--validation-level',level]
    assert audit(args, appearance_catalog=resolved.ue_profile['appearance'],
        environment_profile=resolved.ue_profile['environment_profile'], episode_dir=episode) == 0
    sheet = Image.new('RGB',(1280,780),(24,24,30))
    for index in range(4):
        camera = 'CineCam_{:02d}'.format(index+1)
        frame = episode / camera / 'img1' / '{:06d}.png'.format(entry['frames_per_camera']//2+1)
        with Image.open(frame) as image:
            assert image.size == (3840,2160)
            image.thumbnail((640,360))
            sheet.paste(image.convert('RGB'),((index%2)*640,(index//2)*390+25))
        ImageDraw.Draw(sheet).text(((index%2)*640+8,(index//2)*390+6),camera,fill='white')
    sheet.save(Path(settings['private_root']) / 'evidence' / entry['task_id'] / 'preview.png')
    dump(episode / 'pilot_ready.json', dict(entry, status='audit_passed', rgb_frames=entry['frames_per_camera']*4,
        mask_frames=entry['frames_per_camera']*4, appearance_restored=not bool(retained), environment_restored=not bool(retained), scene_retained=bool(retained),
        audit_level=level, full_audit_passed=(level=='full')))


def publish(settings, entry):
    use_runtime(settings)
    from grf_ue_bridge.seed_dataset import promote_episode, verify_clip
    private, root = Path(settings['private_root']), Path(settings['dataset_root'])
    archive = archive_text(Path(entry['episode']), private / 'archives' / (entry['task_id'] + '.zip'))
    policy = quality_policy(settings)
    result = promote_episode(entry, root, audit_policy=policy)
    basic = bool(policy and policy['mode'] == 'basic_after_full')
    verification = 'publication_structure' if basic else ('metadata' if policy else 'full')
    clip = read(Path(result['clip']) / 'clip.json') if basic else verify_clip(Path(result['clip']), verification_level=verification)
    assert clip['frame_count'] == entry['frames_per_camera']
    execution = read(Path(entry['episode']) / 'provenance/render_execution.json')
    tuning = Path(entry['episode']) / 'provenance/write_queue_tuning.json'
    if tuning.exists():
        execution['live_queue_tuning'] = read(tuning)
    queue_runtime = Path(entry['episode']) / 'provenance/write_queue_runtime.json'
    if queue_runtime.exists():
        execution['write_queue_runtime'] = read(queue_runtime)
    if execution.pop('task_id') != entry['task_id']:
        raise ValueError('实际渲染记录与当前任务不一致')
    jobs = execution.pop('jobs')
    assert len(jobs)==4 and all('FutsalMoviePipelinePNGOutput' in job['settings'] for job in jobs)
    execution['job_setting_classes'] = sorted({name for job in jobs for name in job['settings']})
    execution['evidence_mode'] = 'recorded_before_mrq'
    execution_ref = 'shared/runtime_profiles/' + canonical_sha(execution) + '.json'
    profile = root / execution_ref
    if profile.exists():
        if read(profile)!=execution:
            raise ValueError('共享runtime摘要冲突')
    else:
        dump(profile,execution)
    index = read(root / 'dataset.json')
    index.update(release_id=settings['release_id'], generation_ref='shared/generation_recipe.json', resolution=[3840,2160])
    index.setdefault('execution_refs',{})[entry['task_id']] = {'runtime_profile_ref':execution_ref}
    if (private / 'execution_policy.json').exists():
        index['execution_policy_ref'] = 'shared/execution_policy.json'
    dump(root / 'dataset.json', index)
    record = dict(result, archive=archive, verified=True, verified_utc=now(), resolution=[3840,2160],
        verification_level=verification, audit_level='none' if basic else ('quick' if policy else 'full'))
    dump(private / 'published' / (entry['task_id'] + '.json'), record)
    return record


def clean_episode(settings, entry):
    private = Path(settings['private_root'])
    record = read(private / 'published' / (entry['task_id'] + '.json'))
    assert record['verified'] and record['root_seed'] == entry['root_seed']
    assert sha(record['archive']['path']) == record['archive']['sha256']
    # 正式发布已读取一次SHA；激进策略不重复读取媒体，仅确认交付文件身份。
    canonical = require_child(record['clip'], Path(settings['dataset_root']) / 'groups')
    checksums = read(canonical / 'checksums.json')
    episode = require_child(entry['episode'], settings['work_root'])
    policy = quality_policy(settings)
    basic = bool(policy and policy['mode'] == 'basic_after_full')
    for relative, expected in checksums.items():
        target = require_child(canonical / relative, canonical)
        if '/rgb/' in relative or (basic and '/mask/' in relative):
            parts = Path(relative).parts
            source = episode / parts[1] / ('img1' if '/rgb/' in relative else 'mask') / parts[-1]
            if not os.path.samefile(source,target):
                raise ValueError('RGB移交后不再是同一文件，停止清理')
        elif sha(target) != expected:
            raise ValueError('正式文件发生变化，停止清理')
    remove_tree(episode, settings['work_root'])
    for pair in entry['scratch']['paths']:
        remove_tree(pair['target'], private / 'exr_scratch')
    dump(private / 'cleaned' / (entry['task_id'] + '.json'), {'cleaned_utc':now(),
        'archive':record['archive'], 'canonical_clip':str(canonical)})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('prepare','postprocess','publish','clean'))
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--row', type=Path)
    parser.add_argument('--entry', type=Path)
    args = parser.parse_args()
    settings = load_settings(args.config)
    if args.action == 'prepare':
        result = prepare(settings, read(args.row))
    else:
        entry = read(args.entry)
        result = {'postprocess':postprocess,'publish':publish,'clean':clean_episode}[args.action](settings, entry)
    print('PRODUCTION_P1_' + args.action.upper() + ' ' + json.dumps(result,ensure_ascii=False), flush=True)
