"""从真实位置差分中确定性选择连续活跃轨迹，不拼接或改写原始帧。

阈值属于数据筛选 QC，不能作为 GRF 比赛状态的真值。默认复用 motion_quality
中的活跃速度和团队覆盖阈值，但不强迫门将或每位外场球员持续运动。
"""

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from .motion_quality import (
    ACTIVE_MPS,
    STATIONARY_MPS,
    TEAM_ACTIVE_COVERAGE_REQ,
    TEAM_ACTIVE_MIN,
    analyze_frames,
)


@dataclass(frozen=True)
class ActiveWindowPolicy:
    """按采样帧数计片段时长；另记录首末观测之间的真实时间跨度。"""

    min_duration_s: float = 5.0
    max_duration_s: float = 10.0
    active_speed_mps: float = ACTIVE_MPS
    stationary_speed_mps: float = STATIONARY_MPS
    min_active_outfields: int = TEAM_ACTIVE_MIN
    min_team_active_coverage: float = TEAM_ACTIVE_COVERAGE_REQ
    max_low_motion_streak_s: float = 1.0
    min_spatial_extent_m: float = 1.5
    min_moving_outfields: int = 4
    max_plausible_speed_mps: float = 12.0
    max_windows: int = 1
    local_motion_window_s: float = 0.0
    min_local_spatial_extent_m: float = 1.0
    max_local_low_motion_streak_s: float = 3.0

    def validate(self) -> None:
        """拒绝无意义的阈值，避免筛选被静默放宽。"""
        positive = (
            'min_duration_s', 'max_duration_s', 'active_speed_mps',
            'stationary_speed_mps', 'min_spatial_extent_m',
            'max_plausible_speed_mps',
            'min_local_spatial_extent_m',
        )
        for name in positive:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError('{} 必须是正有限数'.format(name))
        if self.max_duration_s < self.min_duration_s:
            raise ValueError('max_duration_s 不得小于 min_duration_s')
        if not 0 < self.min_team_active_coverage <= 1:
            raise ValueError('min_team_active_coverage 必须属于 (0,1]')
        if not math.isfinite(self.max_low_motion_streak_s) or self.max_low_motion_streak_s < 0:
            raise ValueError('max_low_motion_streak_s 必须为非负有限数')
        for name in ('local_motion_window_s', 'max_local_low_motion_streak_s'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError('{} 必须为非负有限数'.format(name))
        for name in ('min_active_outfields', 'min_moving_outfields', 'max_windows'):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError('{} 必须是正整数'.format(name))


def _prefix(values):
    result = [0]
    for value in values:
        result.append(result[-1] + value)
    return result


def _longest_true(values):
    best = current = 0
    for value in values:
        current = current + 1 if value else 0
        best = max(best, current)
    return best


class _RangeExtrema:
    """静态稀疏表：预处理 O(n log n)，查询任意连续区间极值 O(1)。"""

    def __init__(self, values):
        self.minimum = [list(values)]
        self.maximum = [list(values)]
        width = 2
        while width <= len(values):
            half = width // 2
            lower, upper = self.minimum[-1], self.maximum[-1]
            self.minimum.append([min(lower[i], lower[i + half]) for i in range(len(values) - width + 1)])
            self.maximum.append([max(upper[i], upper[i + half]) for i in range(len(values) - width + 1)])
            width *= 2

    def extent(self, start, end):
        level = (end - start).bit_length() - 1
        other = end - (1 << level)
        return max(self.maximum[level][start], self.maximum[level][other]) - min(self.minimum[level][start], self.minimum[level][other])


def _position_series(frames, dt_s, gk_ids):
    """严格检查连续 roster 和时间；速度只由平面坐标差分计算。"""
    if isinstance(dt_s, bool) or not isinstance(dt_s, (int, float)) or not math.isfinite(dt_s) or dt_s <= 0:
        raise ValueError('dt_s 必须为正有限数，不能隐式猜测 FPS')
    if not frames:
        return {}, [], set(gk_ids or [])
    first = frames[0].get('players')
    if not isinstance(first, list) or not first:
        raise ValueError('每帧必须包含非空 players')
    ids = [p.get('id') for p in first if isinstance(p, Mapping)]
    if len(ids) != len(first) or any(not isinstance(pid, str) or not pid for pid in ids) or len(set(ids)) != len(ids):
        raise ValueError('首帧 player id 缺失或重复')
    positions = {pid: [] for pid in sorted(ids)}
    goalkeeper_ids = set(gk_ids or [])
    for index, frame in enumerate(frames):
        players = frame.get('players')
        if not isinstance(players, list) or len(players) != len(ids):
            raise ValueError('frame {} 的 roster 不完整'.format(index))
        lookup = {}
        for player in players:
            pid = player.get('id')
            if pid in lookup:
                raise ValueError('frame {} player id 重复'.format(index))
            lookup[pid] = player
        if set(lookup) != set(ids):
            raise ValueError('frame {} 的 roster 发生变化'.format(index))
        for pid, player in lookup.items():
            point = player.get('position_m')
            if not isinstance(point, (list, tuple)) or len(point) < 2 or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in point[:2]):
                raise ValueError('frame {} {} position_m 无效'.format(index, pid))
            positions[pid].append((float(point[0]), float(point[1])))
            if not gk_ids and (player.get('is_goalkeeper') or player.get('role') == 'goalkeeper'):
                goalkeeper_ids.add(pid)
        if index and 'time_seconds' in frame and 'time_seconds' in frames[index - 1]:
            delta = float(frame['time_seconds']) - float(frames[index - 1]['time_seconds'])
            if not math.isfinite(delta) or not math.isclose(delta, dt_s, rel_tol=1e-6, abs_tol=1e-6):
                raise ValueError('frame {} 时间间隔不连续或与 dt_s 不符'.format(index))
    if not goalkeeper_ids <= set(ids):
        raise ValueError('gk_ids 包含 roster 之外的球员')
    return positions, sorted(ids), goalkeeper_ids


def _window_metrics(start, end, positions, speeds, outfields, dt_s, policy):
    players = {}
    for pid, series in positions.items():
        points = series[start:end]
        steps = speeds[pid][start + 1:end]
        path = sum(steps) * dt_s
        dx = max(p[0] for p in points) - min(p[0] for p in points)
        dy = max(p[1] for p in points) - min(p[1] for p in points)
        extent = math.hypot(dx, dy)
        players[pid] = {
            'path_length_m': round(path, 6),
            'net_displacement_m': round(math.dist(points[0], points[-1]), 6),
            'spatial_extent_m': round(extent, 6),
            'active_ratio': round(sum(s >= policy.active_speed_mps for s in steps) / len(steps), 6),
            'stationary_ratio': round(sum(s < policy.stationary_speed_mps for s in steps) / len(steps), 6),
            'max_speed_mps': round(max(steps), 6),
            'is_goalkeeper': pid not in outfields,
        }
    moving = [pid for pid in outfields if players[pid]['spatial_extent_m'] >= policy.min_spatial_extent_m]
    return players, moving


def select_active_windows(
    frames: Sequence[Mapping[str, Any]],
    *,
    dt_s: float,
    gk_ids: Optional[Sequence[str]] = None,
    policy: Optional[ActiveWindowPolicy] = None,
) -> Dict[str, Any]:
    """返回不重叠的 [start_index,end_index_exclusive) 窗及筛选证据。

    先选最长窗，再按团队覆盖和真实空间运动排序，最终以较早起点打破平局。
    不循环轨迹、不重采样、不消费 RNG；多个窗仍是分别保存的连续片段。
    """
    policy = policy or ActiveWindowPolicy()
    policy.validate()
    positions, ids, goalkeepers = _position_series(frames, dt_s, gk_ids)
    outfields = sorted(set(ids) - goalkeepers)
    n = len(frames)
    policy_json = json.dumps(asdict(policy), sort_keys=True, separators=(',', ':'))
    report = {
        'schema': 'futsalmot_active_trajectory_qc', 'version': 1,
        'policy': asdict(policy),
        'policy_sha256': hashlib.sha256(policy_json.encode('utf-8')).hexdigest(),
        'source_frame_count': n, 'dt_s': dt_s,
        'goalkeeper_ids': sorted(goalkeepers), 'outfield_ids': outfields,
        'range_convention': '[start_index,end_index_exclusive), original frame order',
        'selected_windows': [], 'candidate_rejection_counts': {},
        'status': 'rejected', 'rejection_reason': None,
    }
    min_frames = max(2, math.ceil(policy.min_duration_s / dt_s - 1e-9))
    max_frames = max(min_frames, math.floor(policy.max_duration_s / dt_s + 1e-9))
    if n < min_frames:
        report['rejection_reason'] = 'insufficient_frames'
        return report
    if len(outfields) < max(policy.min_active_outfields, policy.min_moving_outfields):
        report['rejection_reason'] = 'insufficient_outfield_roster'
        return report
    speeds = {pid: [0.0] + [math.dist(a, b) / dt_s for a, b in zip(points, points[1:])] for pid, points in positions.items()}
    counts = [sum(speeds[pid][i] >= policy.active_speed_mps for pid in outfields) for i in range(n)]
    active_prefix = _prefix([int(c >= policy.min_active_outfields) for c in counts])
    bad_prefix = _prefix([int(any(speeds[pid][i] > policy.max_plausible_speed_mps for pid in ids)) for i in range(n)])
    allowed_low_intervals = math.floor(policy.max_low_motion_streak_s / dt_s + 1e-9)
    low_run = 0
    forbidden_low_ends = []
    for count in counts:
        low_run = low_run + 1 if count < policy.min_active_outfields else 0
        forbidden_low_ends.append(int(low_run > allowed_low_intervals))
    low_prefix = _prefix(forbidden_low_ends)
    spatial = {
        pid: (_RangeExtrema([p[0] for p in positions[pid]]), _RangeExtrema([p[1] for p in positions[pid]]))
        for pid in outfields
    }
    local_good = None
    local_active_prefix = local_bad_prefix = None
    allowed_local_intervals = math.floor(policy.max_local_low_motion_streak_s / dt_s + 1e-9)
    if policy.local_motion_window_s > 0:
        width = min(n, max(2, math.ceil(policy.local_motion_window_s / dt_s - 1e-9)))
        local_good = []
        local_run = 0
        forbidden_local_ends = []
        for index in range(n):
            # 居中的固定宽度邻域在两端平移，避免仅因上下文不足误判开场。
            left = max(0, min(n - width, index - width // 2))
            right = left + width
            moving_count = sum(math.hypot(spatial[pid][0].extent(left, right), spatial[pid][1].extent(left, right)) >= policy.min_local_spatial_extent_m for pid in outfields)
            good = moving_count >= policy.min_moving_outfields
            local_good.append(good)
            local_run = 0 if good else local_run + 1
            forbidden_local_ends.append(int(local_run > allowed_local_intervals))
        local_active_prefix = _prefix([int(good) for good in local_good])
        local_bad_prefix = _prefix(forbidden_local_ends)
        report['source_local_motion'] = {
            'window_s': policy.local_motion_window_s,
            'min_extent_m': policy.min_local_spatial_extent_m,
            'coverage': round(sum(local_good[1:]) / (n - 1), 6),
            'max_low_motion_streak_s': round(_longest_true(not good for good in local_good[1:]) * dt_s, 6),
            'context': '完整源轨迹中的居中邻域，端点平移；依赖完整源而非裁剪后重新分类',
        }
    report['source_metrics'] = analyze_frames(frames, dt_s=dt_s, gk_ids=sorted(goalkeepers))
    ranges = [(0, n)]
    rejects = report['candidate_rejection_counts']
    for _ in range(policy.max_windows):
        selected = None
        selected_key = None
        for length in range(min(max_frames, n), min_frames - 1, -1):
            for range_start, range_end in ranges:
                for start in range(range_start, range_end - length + 1):
                    end = start + length
                    intervals = length - 1
                    coverage = (active_prefix[end] - active_prefix[start + 1]) / intervals
                    reason = None
                    if bad_prefix[end] - bad_prefix[start + 1]:
                        reason = 'implausible_transition_speed'
                    elif coverage + 1e-12 < policy.min_team_active_coverage:
                        reason = 'low_team_active_coverage'
                    elif start + allowed_low_intervals + 1 < end and low_prefix[end] - low_prefix[start + allowed_low_intervals + 1]:
                        reason = 'long_low_motion_streak'
                    elif local_good is not None and start + allowed_local_intervals + 1 < end and local_bad_prefix[end] - local_bad_prefix[start + allowed_local_intervals + 1]:
                        reason = 'long_local_small_range_streak'
                    if reason:
                        rejects[reason] = rejects.get(reason, 0) + 1
                        continue
                    extents = {pid: round(math.hypot(spatial[pid][0].extent(start, end), spatial[pid][1].extent(start, end)), 6) for pid in outfields}
                    moving = [pid for pid in outfields if extents[pid] >= policy.min_spatial_extent_m]
                    if len(moving) < policy.min_moving_outfields:
                        reason = 'small_spatial_extent_or_jitter'
                        rejects[reason] = rejects.get(reason, 0) + 1
                        continue
                    mean_extent = sum(extents.values()) / len(outfields)
                    key = (length, coverage, len(moving), mean_extent, -start)
                    if selected_key is None or key > selected_key:
                        selected_key = key
                        selected = {
                            'start_index': start, 'end_index_exclusive': end,
                            'frame_count': length,
                            'sample_duration_s': round(length * dt_s, 6),
                            'trajectory_span_s': round(intervals * dt_s, 6),
                            'source_time_start_s': frames[start].get('time_seconds', round(start * dt_s, 6)),
                            'source_time_end_s': frames[end - 1].get('time_seconds', round((end - 1) * dt_s, 6)),
                            'source_step_start': frames[start].get('step', start),
                            'source_step_last': frames[end - 1].get('step', end - 1),
                            'team_active_coverage': round(coverage, 6),
                            'moving_outfield_ids': moving,
                        }
            if selected is not None:
                break
        if selected is None:
            break
        # 只对最终窗口计算每人路径、净位移等 O(窗长) 的报告指标。
        selected['players'], selected['moving_outfield_ids'] = _window_metrics(
            selected['start_index'], selected['end_index_exclusive'], positions, speeds, outfields, dt_s, policy,
        )
        if local_good is not None:
            left, right = selected['start_index'], selected['end_index_exclusive']
            selected['local_motion'] = {
                'coverage': round((local_active_prefix[right] - local_active_prefix[left + 1]) / (right - left - 1), 6),
                'max_low_motion_streak_s': round(_longest_true(not good for good in local_good[left + 1:right]) * dt_s, 6),
            }
        report['selected_windows'].append(selected)
        a, b = selected['start_index'], selected['end_index_exclusive']
        ranges = [(left, min(right, a)) for left, right in ranges if left < a] + [(max(left, b), right) for left, right in ranges if right > b]
        ranges = [(left, right) for left, right in ranges if right - left >= min_frames]
    if report['selected_windows']:
        report['status'] = 'accepted'
    else:
        report['rejection_reason'] = 'no_contiguous_active_window'
    return report


def inspect_episode_active_windows(episode_dir, policy=None):
    """只读 episode 的 meta/frames，明确使用其时间步与门将元数据。"""
    episode_dir = Path(episode_dir)
    meta = json.loads((episode_dir / 'meta.json').read_text(encoding='utf-8'))
    frames = [json.loads(line) for line in (episode_dir / 'frames.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    dt_s = meta['timing']['source_step_seconds']
    gks = [entity['id'] for entity in meta['entities'] if entity.get('is_goalkeeper')]
    return select_active_windows(frames, dt_s=dt_s, gk_ids=gks, policy=policy)
