"""裁剪必须连续、保留源证据并重新构建绑定轨迹指纹的 manifest。"""

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from grf_ue_bridge.active_episode import prepare_active_episode
from grf_ue_bridge.active_trajectory import ActiveWindowPolicy, inspect_episode_active_windows
from grf_ue_bridge.config.models import ResolvedTask
from grf_ue_bridge.environment import ENVIRONMENT_POLICY, ENVIRONMENT_CATALOG_ID, environment_catalog_fingerprint
from grf_ue_bridge.seeds import derive_episode_seeds
from grf_ue_bridge.validator import validate_episode


@pytest.fixture
def inputs(tmp_path):
    source = tmp_path / 'archive'
    source.mkdir()
    entities = [{'id': '{}{}'.format(t, i), 'team': 'left' if t == 'L' else 'right', 'is_goalkeeper': i == 0} for t in 'LR' for i in range(5)]
    meta = {'schema': 'grf_ue_episode', 'version': 1, 'episode_id': 'original300',
            'source': {'seed': 1001, 'scenario': '5_vs_5'},
            'randomness': derive_episode_seeds(1001).model_dump(),
            'timing': {'num_steps': 300, 'source_step_seconds': .1, 'playback_fps': 10},
            'field': {'length_m': 40., 'width_m': 20.}, 'entities': entities}
    frames = []
    for i in range(300):
        distance = max(0, min(i - 10, 90)) * .12
        players = [{'id': e['id'], 'position_m': [distance if not e['is_goalkeeper'] else 0., j * .5, 0.], 'active': True} for j, e in enumerate(entities)]
        frames.append({'step': i, 'time_seconds': round(i * .1, 9), 'players': players, 'score': [0, 0], 'ball': {'position_m': [0., 0., 0.]}})
    (source / 'meta.json').write_text(json.dumps(meta), encoding='utf-8')
    (source / 'frames.jsonl').write_text(''.join(json.dumps(f) + '\n' for f in frames), encoding='utf-8')
    report = inspect_episode_active_windows(source, policy=ActiveWindowPolicy(max_duration_s=6))
    count = report['selected_windows'][0]['frame_count']
    target = tmp_path / 'dataset' / 'active1001'
    resolved = ResolvedTask(task_id='active1001', episode_name=target.name,
        source_task_file=str(tmp_path / 'task.json'), repo_root=str(tmp_path), ue_project_root=str(tmp_path),
        dataset_root=str(target.parent), trajectory_output=str(target), dataset_episode_dir=str(target),
        export_profile={'scenario': '5_vs_5', 'seed': 1001, 'num_steps': count, 'target_fps': 10, 'playback_fps': 10},
        ue_profile={'environment_profile': {'mode': 'seeded', 'policy': ENVIRONMENT_POLICY,
                    'catalog_id': ENVIRONMENT_CATALOG_ID, 'environment_seed': 1001,
                    'preset': 'daylight_clear_v1', 'catalog_fingerprint': environment_catalog_fingerprint()}})
    return resolved, source, report, frames


def test_crop_resets_time_without_splicing_and_preserves_source(inputs):
    resolved, source, report, frames = inputs
    originals = {name: (source / name).read_bytes() for name in ('meta.json', 'frames.jsonl')}
    entry = prepare_active_episode(resolved, source, report)
    target = Path(entry['episode_dir'])
    cropped = [json.loads(line) for line in (target / 'frames.jsonl').read_text(encoding='utf-8').splitlines()]
    win = report['selected_windows'][0]
    assert win['start_index'] > 0
    assert [f['players'] for f in cropped] == [f['players'] for f in frames[win['start_index']:win['end_index_exclusive']]]
    assert [f['step'] for f in cropped] == list(range(len(cropped)))
    assert cropped[0]['time_seconds'] == 0
    assert cropped[-1]['time_seconds'] == pytest.approx((len(cropped) - 1) * .1)
    provenance = json.loads((target / 'provenance/active_window.json').read_text(encoding='utf-8'))
    assert provenance['source_capture_steps'] == 300
    assert provenance['source_frames_sha256'] == hashlib.sha256(originals['frames.jsonl']).hexdigest()
    assert provenance['source_step_range'] == [win['start_index'], win['end_index_exclusive']]
    assert provenance['reproduction_requires_full_source_capture'] is True
    assert provenance['source_export_profile_available'] is False
    environment = json.loads((target / 'provenance/environment.json').read_text(encoding='utf-8'))
    assert environment['trajectory_sha256'] == hashlib.sha256((target / 'frames.jsonl').read_bytes()).hexdigest()
    assert environment['trajectory_sha256'] != provenance['source_frames_sha256']
    assert validate_episode(target) == 0
    assert {name: (source / name).read_bytes() for name in originals} == originals


def test_repeated_call_is_identical_and_different_output_is_refused(inputs):
    resolved, source, report, _ = inputs
    first = prepare_active_episode(resolved, source, report)
    assert prepare_active_episode(resolved, source, report) == first
    changed = resolved.model_copy(deep=True)
    changed.ue_profile['environment_profile']['preset'] = 'overcast_soft_v1'
    with pytest.raises(FileExistsError, match='不同输出'):
        prepare_active_episode(changed, source, report)
    path = Path(first['episode_dir']) / 'frames.jsonl'
    original = path.read_bytes()
    path.write_bytes(original + b'\n')
    with pytest.raises(FileExistsError):
        prepare_active_episode(resolved, source, report)
    assert path.read_bytes() == original + b'\n'


@pytest.mark.parametrize('field,value', [('num_steps', 300), ('seed', 2), ('target_fps', 30), ('trajectory_time_scale', 2.)])
def test_config_does_not_match_crop_is_rejected(inputs, field, value):
    resolved, source, report, _ = inputs
    resolved.export_profile[field] = value
    with pytest.raises(ValueError):
        prepare_active_episode(resolved, source, report)
    assert not Path(resolved.trajectory_output).exists()


def test_path_escape_and_source_overlap_are_rejected(inputs):
    resolved, source, report, _ = inputs
    resolved.trajectory_output = resolved.dataset_episode_dir = str(source)
    resolved.episode_name = source.name
    with pytest.raises(ValueError, match='逃逸'):
        prepare_active_episode(resolved, source, report)
    resolved.dataset_root = str(source.parent)
    with pytest.raises(ValueError, match='重叠'):
        prepare_active_episode(resolved, source, report)


@pytest.mark.parametrize('change', ['range', 'policy', 'count', 'seed'])
def test_stale_or_fabricated_selection_fails_closed(inputs, change):
    resolved, source, report, _ = inputs
    report = copy.deepcopy(report)
    if change == 'range':
        report['selected_windows'][0]['start_index'] += 1
    elif change == 'policy':
        report['policy']['active_speed_mps'] = .01
    elif change == 'count':
        report['source_frame_count'] = 60
    else:
        report['root_seed'] = 1
    with pytest.raises(ValueError):
        prepare_active_episode(resolved, source, report)


def test_manifest_tamper_is_not_silently_reused(inputs):
    resolved, source, report, _ = inputs
    entry = prepare_active_episode(resolved, source, report)
    path = Path(entry['episode_dir']) / 'provenance/environment.json'
    manifest = json.loads(path.read_text(encoding='utf-8'))
    manifest['trajectory_sha256'] = '0' * 64
    path.write_text(json.dumps(manifest), encoding='utf-8')
    with pytest.raises(ValueError, match='environment.json'):
        prepare_active_episode(resolved, source, report)


def test_appearance_manifest_rebuilt_and_verified(inputs):
    from test_appearance import appearance_catalog
    resolved, source, report, _ = inputs
    resolved.ue_profile['appearance'] = appearance_catalog()
    entry = prepare_active_episode(resolved, source, report)
    folder = Path(entry['episode_dir'])
    manifest = json.loads((folder / 'provenance/appearance.json').read_text(encoding='utf-8'))
    assert manifest['trajectory_sha256'] == hashlib.sha256((folder / 'frames.jsonl').read_bytes()).hexdigest()
    assert prepare_active_episode(resolved, source, report) == entry


def test_cli_selects_seed_and_writes_entry_json(inputs, tmp_path):
    resolved, source, report, _ = inputs
    task_path = tmp_path / 'resolved.json'
    task_path.write_text(resolved.model_dump_json(by_alias=True), encoding='utf-8')
    report['root_seed'] = 1001
    selection_path = tmp_path / 'selection.json'
    selection_path.write_text(json.dumps([report]), encoding='utf-8')
    output = tmp_path / 'entry.json'
    command = [sys.executable, '-X', 'utf8', str(Path(__file__).resolve().parents[1] / 'tools/prepare_active_episode.py'),
               '--resolved', str(task_path), '--source', str(source), '--selection-json', str(selection_path),
               '--seed', '1001', '--output-json', str(output)]
    result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8')
    assert result.returncode == 0, result.stdout + result.stderr
    entry = json.loads(output.read_text(encoding='utf-8'))
    assert entry['root_seed'] == 1001
    assert entry['source_capture_steps'] == 300
    assert validate_episode(Path(entry['episode_dir'])) == 0
