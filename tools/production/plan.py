"""采集固定候选源池，检查真实运动，再一次冻结正式清单。"""
import argparse
import copy
import json
from pathlib import Path
import shutil
import sys
from common import PRESETS, canonical_sha, dump, load_settings, now, read, select_release, sha


def build(config):
    settings = load_settings(config)
    private = Path(settings['private_root'])
    private.mkdir(parents=True, exist_ok=True)
    target = private / 'production_plan.json'
    if target.exists():
        plan = read(target)
        if plan['settings_sha256'] != canonical_sha(settings):
            raise ValueError('已有正式清单对应不同配置，拒绝静默改动')
        return plan
    # 候选源采集使用已冻结且验证过的P1，不导入UE Python。
    sys.path.insert(0, str(Path(settings['source_runtime']) / 'src'))
    from grf_ue_bridge.config.resolver import resolve_task, save_resolved_task
    from grf_ue_bridge.workflows.task_export import run_export
    from grf_ue_bridge.active_trajectory import ActiveWindowPolicy, inspect_episode_active_windows
    from grf_ue_bridge.validator import validate_episode
    old_reports = read(settings['legacy_source_reports'])
    by_seed = {row['root_seed']: row for row in old_reports}
    cache = private / 'candidate_reports.json'
    if cache.exists():
        by_seed.update({row['root_seed']: row for row in read(cache)})
    template = read(settings['template'])
    policy = ActiveWindowPolicy(**old_reports[0]['policy'])
    wanted = range(settings['candidate_seed_first'], settings['candidate_seed_last'] + 1)
    for seed in wanted:
        if (private / 'STOP_AFTER_CURRENT').exists():
            raise RuntimeError('已收到停止标记，未冻结清单')
        if seed in by_seed:
            old = by_seed[seed]
            source = Path(old['source_episode'])
            if sha(source / 'frames.jsonl') != old['source_frames_sha256']:
                raise ValueError('候选源轨迹已变化：{}'.format(seed))
            continue
        preset = PRESETS[(seed - 1001) % len(PRESETS)]
        task = copy.deepcopy(template)
        task_id = 'fmot_source_s{:08d}'.format(seed)
        task.update(task_id=task_id, episode_name='source_s{:08d}'.format(seed))
        task['export'].update(seed=seed, num_steps=settings['source_steps'])
        task['audit']['expected_frames_per_camera'] = settings['source_steps']
        task['ue']['environment']['preset'] = preset
        task['ue']['sequence_package_path'] = '/Game/FutsalMOT/Generated/DatasetV1/' + task_id
        task_path = private / 'source_configs' / (task_id + '.json')
        dump(task_path, task)
        resolved = resolve_task(task_path, dataset_root=str(private / 'sources'), ue_project_root=settings['project'])
        save_resolved_task(resolved, private / 'source_resolved')
        source = Path(resolved.dataset_episode_dir)
        if not source.exists():
            if run_export(resolved) != 0:
                raise RuntimeError('GRF候选源导出失败：{}'.format(seed))
        if validate_episode(source) != 0:
            raise ValueError('候选源验证失败：{}'.format(seed))
        report = inspect_episode_active_windows(source, policy=policy)
        report.update(root_seed=seed, preset=preset, source_episode=str(source),
            source_frames_sha256=sha(source / 'frames.jsonl'))
        by_seed[seed] = report
        dump(cache, [by_seed[k] for k in sorted(by_seed) if k in wanted])
        dump(private / 'state.json', {'stage': 'collecting_candidates', 'updated_utc': now(),
            'last_seed': seed, 'candidate_count': sum(k in wanted for k in by_seed), 'target_candidates': len(wanted),
            'gpu_training': False})
        print('CANDIDATE_READY {} frames={}'.format(seed, report['selected_windows'][0]['frame_count'] if report['selected_windows'] else 0), flush=True)
    reports = [by_seed[seed] for seed in wanted]
    dump(cache, reports)
    free = shutil.disk_usage('G:/').free
    budget = min(settings['dataset_budget_gib'] * 1024**3, free - settings['g_free_reserve_gib'] * 1024**3)
    selected = select_release(reports, settings, budget)
    plan = {'schema': 'futsalmot_production_plan', 'version': 1, 'created_utc': now(),
        'settings': settings, 'settings_sha256': canonical_sha(settings),
        'candidate_count': len(reports), 'candidate_reports_sha256': sha(cache),
        'selection': '逐seed最长连续活跃窗；天气内长度降序/seed升序；固定哈希划分；已完成组保留train',
        'minimum_clip_frames': settings['minimum_clip_frames'], 'capacity_budget_bytes': budget,
        'source_motion_policy': old_reports[0]['policy'], **selected}
    dump(target, plan)
    dump(private / 'state.json', {'stage': 'plan_frozen', 'updated_utc': now(), 'plan_sha256': sha(target),
        'seed_groups': len(plan['entries']), 'rgb_frames': sum(e['frames_per_camera'] * 4 for e in plan['entries']),
        'estimated_public_bytes': plan['estimated_public_bytes'], 'gpu_training': False})
    return plan


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    result = build(args.config)
    print('PRODUCTION_PLAN_FROZEN ' + json.dumps({'groups': len(result['entries']),
        'rgb_frames': sum(e['frames_per_camera'] * 4 for e in result['entries']),
        'estimated_public_gib': round(result['estimated_public_bytes'] / 1024**3, 1)}, ensure_ascii=False), flush=True)
