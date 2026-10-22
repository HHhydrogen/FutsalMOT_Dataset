"""源池分桶固定且停滞诊断仅使用位置差分，不猜测未导出的 done 状态。"""

import importlib.util
from pathlib import Path


def _module():
    path = Path(__file__).resolve().parents[1] / 'tools/scan_active_source_pool.py'
    spec = importlib.util.spec_from_file_location('active_pool_tool', str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_tail_diagnosis_distinguishes_static_held_ball_from_out_of_field():
    module = _module()
    meta = {'timing': {'source_step_seconds': .1}, 'field': {'length_m':40, 'width_m':20},
            'entities':[{'id': '{}{}'.format(t,j), 'is_goalkeeper': j == 0} for t in 'LR' for j in range(5)]}
    frames = [{'players':[{'id': e['id'], 'position_m': [0.,0.,0.], 'active':True} for e in meta['entities']],
               'ball': {'position_m':[2.,3.,.11]}, 'game_mode':0, 'score':[0,0],
               'ball_owned_team':0, 'ball_owned_player':2} for _ in range(100)]
    result = module.motion_diagnosis(meta, frames)
    assert result['tail_all_player_near_static_ratio'] == 1.
    assert result['tail_ball_planar_speed_max_mps'] == 0.
    assert result['tail_ball_outside_nominal_field_ratio'] == 0.
    assert result['tail_game_mode_counts'] == {'0':99}
    assert result['tail_ball_ownership_counts'] == {'[0, 2]':99}
    assert result['done_information_available'] is False
    assert result['steps_left_information_available'] is False
    for frame in frames:
        frame['ball']['position_m'][1] = 12.
    assert module.motion_diagnosis(meta, frames)['tail_ball_outside_nominal_field_ratio'] == 1.


def test_preset_bucket_is_seed_deterministic():
    module = _module()
    assert len(module.PRESETS) == 8
    for seed in (1001, 1009, 1017, 1025, 1033):
        assert module.PRESETS[(seed-1001)%8] == 'daylight_clear_v1'
    assert module.PRESETS[(1007-1001)%8] == 'dawn_cool_v1'
    assert module.PRESETS[(1008-1001)%8] == 'night_floodlit_v1'
