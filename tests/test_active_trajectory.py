"""活跃选段应排除停滞/抖动并保留连续真实运动。"""
import copy
import math
import random

import pytest

from grf_ue_bridge.active_trajectory import ActiveWindowPolicy, select_active_windows

OUTFIELDS = ('L1', 'L2', 'L3', 'L4', 'R1', 'R2', 'R3', 'R4')
GKS = ('L0', 'R0')


def _frames(n, position=None, paused=(), static=()):
    result = []
    x = 0.0
    for index in range(n):
        if index and not any(a <= index < b for a, b in paused):
            x += .12
        players = []
        for j, pid in enumerate(OUTFIELDS + GKS):
            xy = (0.0, float(j)) if pid in GKS or pid in static else (x, float(j))
            if position is not None and pid not in GKS:
                xy = position(index, j)
            players.append({'id': pid, 'position_m': [*xy, 0.0], 'active': True})
        result.append({'step': index, 'time_seconds': index * .1, 'players': players})
    return result


def _select(frames, **policy):
    return select_active_windows(frames, dt_s=.1, gk_ids=GKS, policy=ActiveWindowPolicy(**policy))


def test_static_players_active_flag_does_not_make_valid_data():
    result = _select(_frames(150, position=lambda i, j: (0.0, float(j))))
    assert result['status'] == 'rejected'
    assert not result['selected_windows']
    assert result['candidate_rejection_counts']['low_team_active_coverage'] > 0


def test_fast_small_range_jitter_is_not_active_trajectory():
    result = _select(_frames(120, position=lambda i, j: ((i % 2) * .08, float(j))))
    assert result['status'] == 'rejected'
    assert result['candidate_rejection_counts']['small_spatial_extent_or_jitter'] > 0


def test_real_team_motion_allows_stationary_goalkeepers_and_one_outfield():
    result = _select(_frames(150, static=('L1',)))
    window = result['selected_windows'][0]
    assert result['status'] == 'accepted'
    assert window['frame_count'] == 100
    assert window['team_active_coverage'] == 1.0
    assert window['players']['L0']['path_length_m'] == 0.0
    assert window['players']['L1']['path_length_m'] == 0.0
    assert len(window['moving_outfield_ids']) == 7


def test_short_normal_pause_inside_clip_is_allowed():
    result = _select(_frames(100, paused=((45, 48),)))
    assert result['status'] == 'accepted'
    assert result['selected_windows'][0]['start_index'] == 0
    assert result['selected_windows'][0]['team_active_coverage'] > .95


def test_long_stationary_gap_is_not_removed_and_stitched():
    result = _select(_frames(210, paused=((80, 130),)), max_windows=3)
    windows = result['selected_windows']
    assert len(windows) >= 2
    for window in windows:
        a, b = window['start_index'], window['end_index_exclusive']
        assert not (a < 80 and b > 130)
        assert window['frame_count'] == b - a
        assert window['source_step_last'] == b - 1
    ranges = sorted((w['start_index'], w['end_index_exclusive']) for w in windows)
    assert all(left[1] <= right[0] for left, right in zip(ranges, ranges[1:]))


def test_returning_to_start_with_real_excursion_is_not_jitter():
    frames = _frames(100, position=lambda i, j: (2 * math.cos(i * .1), j + 2 * math.sin(i * .1)))
    result = _select(frames)
    assert result['status'] == 'accepted'
    assert result['selected_windows'][0]['players']['L1']['path_length_m'] > 15


def test_teleport_transition_cannot_qualify_an_active_window():
    frames = _frames(110)
    for index in range(55, 110):
        for player in frames[index]['players']:
            player['position_m'][0] += 30
    result = _select(frames, min_duration_s=8, max_duration_s=10)
    assert result['status'] == 'rejected'
    assert result['candidate_rejection_counts']['implausible_transition_speed'] > 0


def test_deterministic_does_not_mutate_inputs_or_random_state():
    frames = _frames(160)
    before = copy.deepcopy(frames)
    state = random.getstate()
    assert _select(frames) == _select(frames)
    assert frames == before
    assert random.getstate() == state


def test_long_real_motion_followed_by_fast_local_jitter_is_cut():
    frames = _frames(600, position=lambda i, j: (min(i, 80) * .12 + (.08 * (i % 2) if i > 80 else 0), float(j)))
    original = _select(frames, max_duration_s=60)
    assert original['selected_windows'][0]['frame_count'] == 600
    guarded = _select(frames, max_duration_s=60, local_motion_window_s=3)
    window = guarded['selected_windows'][0]
    assert window['end_index_exclusive'] < 150
    assert window['local_motion']['max_low_motion_streak_s'] <= 3.
    assert guarded['candidate_rejection_counts']['long_local_small_range_streak'] > 0


def test_local_motion_guard_allows_stationary_gk_and_small_outfield_pause():
    result = _select(_frames(160, static=('L1',), paused=((70, 73),)), max_duration_s=16, local_motion_window_s=3)
    assert result['selected_windows'][0]['frame_count'] == 160
    assert result['selected_windows'][0]['local_motion']['coverage'] == 1.


def test_sparse_extrema_matches_direct_range_queries():
    from grf_ue_bridge.active_trajectory import _RangeExtrema
    values = [math.sin(i * 1.73) for i in range(35)]
    table = _RangeExtrema(values)
    for left in range(len(values)):
        for right in range(left + 1, len(values) + 1):
            assert table.extent(left, right) == max(values[left:right]) - min(values[left:right])


def test_long_policy_local_motion_deterministic_without_duration_cap():
    frames = _frames(1800, position=lambda i, j: (2 * math.cos(i * .06), j + 2 * math.sin(i * .06)))
    kwargs = dict(min_duration_s=20, max_duration_s=180, local_motion_window_s=3,
                  min_active_outfields=4, min_team_active_coverage=.8, max_low_motion_streak_s=3)
    first = _select(frames, **kwargs)
    assert first == _select(frames, **kwargs)
    assert first['selected_windows'][0]['frame_count'] == 1800


def test_variable_length_window_when_source_is_shorter_than_cap():
    result = _select(_frames(63))
    assert result['selected_windows'][0]['frame_count'] == 63
    assert result['selected_windows'][0]['sample_duration_s'] == 6.3
    assert result['selected_windows'][0]['trajectory_span_s'] == 6.2


@pytest.mark.parametrize('bad_dt', [0, -.1, float('nan'), True])
def test_invalid_dt_fails_closed(bad_dt):
    with pytest.raises(ValueError):
        select_active_windows(_frames(100), dt_s=bad_dt, gk_ids=GKS)


def test_missing_roster_or_discontinuous_time_fails_closed():
    frames = _frames(100)
    frames[10]['players'].pop()
    with pytest.raises(ValueError, match='roster'):
        _select(frames)
    frames = _frames(100)
    frames[10]['time_seconds'] += .1
    with pytest.raises(ValueError, match='时间间隔'):
        _select(frames)


def test_too_short_sequence_is_rejected_not_extended():
    result = _select(_frames(20))
    assert result['rejection_reason'] == 'insufficient_frames'
    assert not result['selected_windows']
