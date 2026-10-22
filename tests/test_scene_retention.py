"""持久场景收据必须绑定当前输入、成功终态和真实应用状态。"""
import json
from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'ue'))
from scene_retention import retained_completion, admission_has_completed_scene, file_sha


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


@pytest.fixture
def retained(tmp_path):
    episode = tmp_path / 'episode'
    write(episode / 'render_summary.json', {'status': 'success', 'episode_id': 'episode'})
    write(episode / 'frames.jsonl', {'step': 0})
    files = ['render_summary.json', 'frames.jsonl']
    for kind, field in [('appearance', 'manifest_fingerprint'), ('environment', 'environment_manifest_fingerprint')]:
        write(episode / ('provenance/' + kind + '.json'), {'fingerprint': kind})
        write(episode / ('provenance/' + kind + '_application.json'), {'state': 'applied', 'scene_lifecycle': 'retain',
            'session_id': 'session', 'mrq_pending': False, 'errors': [], field: kind, 'render_result': {'status': 'success'}})
        files.extend(['provenance/' + kind + '.json', 'provenance/' + kind + '_application.json'])
    receipt = {'schema': 'futsalmot_scene_completion', 'version': 1, 'lifecycle': 'retain', 'state': 'retained',
        'live_readback_verified': True, 'session_id': 'session', 'episode_id': 'episode', 'checkpoint_sha256': 'a' * 64,
        'files_sha256': {relative: file_sha(episode / relative) for relative in files}}
    write(episode / 'provenance/scene_completion.json', receipt)
    return episode


def test_absent_receipt_uses_legacy(tmp_path):
    assert retained_completion(tmp_path) is None


def test_valid_retained_completion(retained):
    assert retained_completion(retained)['lifecycle'] == 'retain'


@pytest.mark.parametrize('relative', ['frames.jsonl', 'render_summary.json', 'provenance/appearance_application.json',
    'provenance/environment_application.json', 'provenance/appearance.json', 'provenance/environment.json'])
def test_changed_capture_evidence_is_rejected(retained, relative):
    with (retained / relative).open('a') as stream:
        stream.write(' ')
    with pytest.raises(ValueError, match='内容已改变'):
        retained_completion(retained)


@pytest.mark.parametrize('field,value', [('live_readback_verified', False), ('session_id', ''), ('state', 'restored'),
    ('episode_id', 'another'), ('files_sha256', {})])
def test_invalid_receipt_is_rejected(retained, field, value):
    path = retained / 'provenance/scene_completion.json'
    receipt = json.loads(path.read_text())
    receipt[field] = value
    write(path, receipt)
    with pytest.raises(ValueError):
        retained_completion(retained)


def test_retained_admission_never_claims_restored():
    record = {'scene_lifecycle': 'retain', 'scene_retained': True, 'appearance_restored': False,
        'environment_restored': False, 'scene_completion_sha256': 'a' * 64}
    assert admission_has_completed_scene(record)
    assert not admission_has_completed_scene(dict(record, appearance_restored=True))
    assert not admission_has_completed_scene(dict(record, scene_completion_sha256=None))
    assert admission_has_completed_scene({'appearance_restored': True, 'environment_restored': True})
