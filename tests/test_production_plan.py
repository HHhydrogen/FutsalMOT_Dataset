"""正式清单必须按seed隔离、天气均衡并守住容量边界。"""
import importlib.util
from pathlib import Path
import pytest

path = Path(__file__).resolve().parents[1] / 'tools/production/common.py'
spec = importlib.util.spec_from_file_location('production_common', path)
common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common)


def fixture():
    settings = dict(adopt_seeds=[1025, 1021], minimum_clip_frames=300,
        train_per_weather=[3, 2], val_per_weather=1, test_per_weather=1,
        split_seed=20261022, camera_count=4, estimated_bytes_per_rgb_mask=1)
    rows = []
    for i, weather in enumerate(common.PRESETS):
        for j in range(8):
            seed = 1001 + i + j * 8
            rows.append(dict(root_seed=seed, preset=weather, status='accepted',
                selected_windows=[dict(frame_count=350+j, start_index=20, end_index_exclusive=370+j)],
                source_episode=str(seed), source_frames_sha256=str(seed)))
    return rows, settings


def test_deterministic_group_split_and_balance():
    rows, settings = fixture()
    plan = common.select_release(rows, settings, 10**9)
    assert plan == common.select_release(list(reversed(rows)), settings, 10**9)
    assert len(plan['entries']) == 40
    assert {e['root_seed'] for e in plan['entries'] if e['root_seed'] in (1021, 1025) and e['split']=='train'} == {1021, 1025}
    for weather in common.PRESETS:
        assert sorted(e['split'] for e in plan['entries'] if e['preset']==weather) == ['test','train','train','train','val']


def test_capacity_can_choose_smaller_balanced_release():
    rows, settings = fixture()
    big = common.select_release(rows, settings, 10**9)
    smaller = common.select_release(rows, settings, big['estimated_public_bytes'] - 1)
    assert len(smaller['entries']) == 32
    assert smaller['estimated_public_bytes'] < big['estimated_public_bytes']


def test_reject_duplicate_source_seed_and_impossible_capacity():
    rows, settings = fixture()
    with pytest.raises(ValueError, match='重复'):
        common.select_release(rows + [rows[0]], settings, 10**9)
    with pytest.raises(ValueError, match='容量'):
        common.select_release(rows, settings, 1)


def test_no_short_or_inactive_source_is_admitted():
    rows, settings = fixture()
    for row in rows:
        if row['preset'] == common.PRESETS[1]:
            row['selected_windows'][0]['frame_count'] = 299
    with pytest.raises(ValueError, match='不足'):
        common.select_release(rows, settings, 10**9)
