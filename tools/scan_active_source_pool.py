"""增量扫描源池，按 seed 固定天气/划分并按连续活跃窗长度排序。"""

import argparse
from collections import Counter
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

from grf_ue_bridge.active_trajectory import ActiveWindowPolicy, inspect_episode_active_windows


PRESETS = ('daylight_clear_v1', 'overcast_soft_v1', 'warm_evening_v1', 'cool_evening_v1',
           'light_haze_v1', 'cloudy_variation_v1', 'dawn_cool_v1', 'night_floodlit_v1')
POLICY = ActiveWindowPolicy(min_duration_s=5, max_duration_s=180,
    min_active_outfields=4, min_team_active_coverage=.8, max_low_motion_streak_s=3,
    local_motion_window_s=3, min_local_spatial_extent_m=1, max_local_low_motion_streak_s=3)


def _write(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(path)


def _counts(values):
    return dict(sorted(Counter(json.dumps(value, ensure_ascii=False, sort_keys=True) for value in values).items()))


def motion_diagnosis(meta, frames):
    """只报告可见事实；JSONL 未保存 done/info 时明确不可判定回合状态。"""
    dt = meta['timing']['source_step_seconds']
    ids = [e['id'] for e in meta['entities'] if e['id'] != 'BALL' and not e.get('is_goalkeeper')]
    active_counts = [0]
    all_speeds = [0.]
    balls = [0.]
    balls_3d = [0.]
    for before, after in zip(frames, frames[1:]):
        a = {p['id']: p['position_m'] for p in before['players']}
        b = {p['id']: p['position_m'] for p in after['players']}
        speeds = {pid: math.dist(a[pid][:2], b[pid][:2]) / dt for pid in a}
        active_counts.append(sum(speeds[pid] >= .5 for pid in ids))
        all_speeds.append(max(speeds.values()))
        balls.append(math.dist(before['ball']['position_m'][:2], after['ball']['position_m'][:2]) / dt)
        balls_3d.append(math.dist(before['ball']['position_m'], after['ball']['position_m']) / dt)
    tail = len(frames) - 1
    while tail > 0 and active_counts[tail] < 4:
        tail -= 1
    # tail 是最后一个≥4外场运动的帧；后续帧构成低运动尾段。
    start = min(len(frames), tail + 1)
    sample = frames[start:]
    half_x, half_y = meta['field']['length_m'] / 2, meta['field']['width_m'] / 2
    if not sample:
        return {'tail_frame_count': 0, 'done_information_available': any('done' in f for f in frames)}
    points = [f['ball']['position_m'] for f in sample]
    return {
        'tail_start_index': start, 'tail_end_index_exclusive': len(frames),
        'tail_frame_count': len(sample), 'tail_sample_duration_s': round(len(sample) * dt, 6),
        'tail_all_player_max_speed_mps': round(max(all_speeds[start:]), 6),
        'tail_all_player_near_static_ratio': round(sum(v <= 1e-5 for v in all_speeds[start:]) / len(sample), 6),
        'tail_ball_planar_speed_mean_mps': round(statistics.mean(balls[start:]), 6),
        'tail_ball_planar_speed_max_mps': round(max(balls[start:]), 6),
        'tail_ball_3d_speed_max_mps': round(max(balls_3d[start:]), 6),
        'tail_ball_planar_extent_m': round(math.hypot(max(p[0] for p in points)-min(p[0] for p in points), max(p[1] for p in points)-min(p[1] for p in points)), 6),
        'tail_ball_position_first_m': points[0], 'tail_ball_position_last_m': points[-1],
        'tail_ball_outside_nominal_field_ratio': round(sum(abs(p[0]) > half_x or abs(p[1]) > half_y for p in points) / len(points), 6),
        'tail_game_mode_counts': _counts(f.get('game_mode') for f in sample),
        'tail_score_counts': _counts(f.get('score') for f in sample),
        'tail_ball_ownership_counts': _counts([f.get('ball_owned_team'), f.get('ball_owned_player')] for f in sample),
        'source_game_mode_counts': _counts(f.get('game_mode') for f in frames),
        'source_score_changes': sum(a.get('score') != b.get('score') for a,b in zip(frames,frames[1:])),
        'done_information_available': any('done' in f for f in frames),
        'steps_left_information_available': any('steps_left' in f for f in frames),
        'limitation': 'frames.jsonl 通常不保存 StepSnapshot.done/info/steps_left；不能由运动、mode或球出界单独判定引擎停止原因。',
    }


def scan(index_path, output_dir):
    index = json.loads(index_path.read_text(encoding='utf-8'))
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / 'active_pool_selection.json'
    cache = json.loads(cache_path.read_text(encoding='utf-8')) if cache_path.is_file() else []
    by_seed = {entry['root_seed']: entry for entry in cache}
    policy_dict = dict(POLICY.__dict__)
    for entry in index:
        seed = entry['root_seed']
        source = Path(entry['episode'])
        if not (source / 'frames.jsonl').is_file() or not (source / 'meta.json').is_file():
            continue
        meta = json.loads((source / 'meta.json').read_text(encoding='utf-8'))
        if meta['source']['seed'] != seed:
            raise ValueError('源 meta seed 与索引不一致: {}'.format(seed))
        policy = replace(POLICY, max_duration_s=meta['timing']['num_steps'] * meta['timing']['source_step_seconds'])
        source_policy = dict(policy.__dict__)
        source_sha = hashlib.sha256((source / 'frames.jsonl').read_bytes()).hexdigest()
        old = by_seed.get(seed)
        if old and old.get('source_frames_sha256') == source_sha and old.get('policy') == source_policy:
            continue
        preset_index = (seed - 1001) % 8
        preset = PRESETS[preset_index]
        split = 'train' if preset_index < 6 else ('validation' if preset_index == 6 else 'synthetic_test')
        if entry.get('preset', preset) != preset or entry.get('split', split) != split:
            raise ValueError('seed {} 的天气/划分与固定根seed分配不一致'.format(seed))
        started = time.perf_counter()
        report = inspect_episode_active_windows(source, policy=policy)
        frames = [json.loads(line) for line in (source / 'frames.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
        report.update(root_seed=seed, preset_index=preset_index, preset=preset, split=split,
            source_episode=str(source), source_frames_sha256=source_sha,
            motion_diagnosis=motion_diagnosis(meta, frames), elapsed_s=round(time.perf_counter()-started, 6))
        by_seed[seed] = report
        _write(cache_path, [by_seed[k] for k in sorted(by_seed)])
        windows = report['selected_windows']
        print(seed, preset, [(w['start_index'], w['end_index_exclusive'], w['frame_count']) for w in windows], flush=True)
    reports = [by_seed[k] for k in sorted(by_seed)]
    buckets = []
    for i, preset in enumerate(PRESETS):
        candidates = []
        for report in reports:
            if report['preset_index'] != i:
                continue
            window = report['selected_windows'][0] if report['selected_windows'] else None
            candidates.append({'root_seed': report['root_seed'], 'frame_count': window['frame_count'] if window else 0,
                'source_range': [window['start_index'], window['end_index_exclusive']] if window else None,
                'sample_duration_s': window['sample_duration_s'] if window else 0,
                'team_active_coverage': window['team_active_coverage'] if window else None,
                'split': report['split'], 'source_episode': report['source_episode']})
        candidates.sort(key=lambda entry: (-entry['frame_count'], entry['root_seed']))
        buckets.append({'preset_index': i, 'preset': preset, 'ranked_candidates': candidates})
    summary = {'schema': 'futsalmot_active_source_pool', 'version': 1,
        'indexed_sources': len(index), 'scanned_sources': len(reports),
        'selection_basis': '仅连续运动 QC 与帧数，长度降序、根seed升序；不使用模型训练/验证指标',
        'policy': policy_dict, 'max_duration_rule': '每个源 meta.num_steps × dt，不固定180秒', 'buckets': buckets}
    _write(output_dir / 'active_pool_ranking.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--watch-seconds', type=float, default=0)
    args = parser.parse_args()
    deadline = time.monotonic() + args.watch_seconds
    while True:
        result = scan(args.index, args.output_dir)
        print('已扫描 {} / {} 个索引源'.format(result['scanned_sources'], result['indexed_sources']), flush=True)
        if time.monotonic() >= deadline:
            break
        time.sleep(min(5, max(0, deadline - time.monotonic())))


if __name__ == '__main__':
    main()
