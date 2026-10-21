import json
from grf_ue_bridge.environment import ENVIRONMENT_CATALOG_ID, build_environment_manifest, environment_catalog_fingerprint
from grf_ue_bridge.workflows.task_audit import check_environment
import episode_environment

def test_audit_rejects_environment_manifest_trajectory_mismatch(tmp_path):
    (trajectory, dataset, profile, _manifest) = _environment_fixture(tmp_path)
    (trajectory / 'frames.jsonl').write_text('{"step":1}\n', encoding='utf-8')
    errors = []
    state = check_environment(trajectory, dataset, errors, environment_profile=profile, render_required=False)
    assert state['ok'] is False
    assert any(('trajectory' in error or 'environment' in error for error in errors))

def test_audit_rejects_environment_render_metadata_drift(tmp_path):
    (trajectory, dataset, profile, manifest) = _environment_fixture(tmp_path)
    (dataset / 'render_summary.json').write_text(json.dumps({'status': 'success', 'cameras': {'Camera_01': {'ok': True}}, 'environment_policy': profile['policy'], 'environment_manifest_fingerprint': 'f' * 64, 'environment_preset': manifest['preset'], 'environment_restore_ready': True}), encoding='utf-8')
    errors = []
    state = check_environment(trajectory, dataset, errors, environment_profile=profile, render_required=True)
    assert state['ok'] is False
    assert any(('render_summary' in error for error in errors))

def _environment_profile():
    return {'mode': 'seeded', 'policy': episode_environment.ENVIRONMENT_POLICY, 'catalog_id': ENVIRONMENT_CATALOG_ID, 'environment_seed': 42, 'preset': 'daylight_clear_v1', 'catalog_fingerprint': environment_catalog_fingerprint()}

def test_audit_skips_environment_checks_when_only_residual_files_exist(tmp_path):
    (trajectory, dataset, _profile, _manifest) = _environment_fixture(tmp_path)
    errors = []
    state = check_environment(trajectory, dataset, errors, environment_profile=None, render_required=False)
    assert state == {'ok': True, 'required': False, 'enabled': False, 'skipped': True}
    assert errors == []

def _environment_fixture(tmp_path, *, report_state='restored'):
    trajectory = tmp_path / 'trajectory'
    dataset = tmp_path / 'dataset'
    provenance = trajectory / 'provenance'
    provenance.mkdir(parents=True)
    dataset.mkdir()
    meta = {'episode_id': 'environment-audit'}
    frames = trajectory / 'frames.jsonl'
    frames.write_text('{"step":0}\n', encoding='utf-8')
    (trajectory / 'meta.json').write_text(json.dumps(meta), encoding='utf-8')
    profile = _environment_profile()
    manifest = build_environment_manifest(meta, frames, profile)
    (provenance / 'environment.json').write_text(json.dumps(manifest), encoding='utf-8')
    actors = [{'actor_label': label, 'actor_path': '/Game/FutsalMOT/Maps/L_FutsalCourt.{}'.format(label), 'class': class_name, 'package_path': '/Game/__ExternalActors__/{}'.format(label), 'apply_status': 'not_started', 'restore_status': 'not_started'} for (label, class_name) in episode_environment.ACTOR_CLASSES.items()]
    report = episode_environment.build_environment_application_report(manifest, 'env_tx_001', report_state, actors, [], [])
    (provenance / 'environment_application.json').write_text(json.dumps(report), encoding='utf-8')
    return (trajectory, dataset, profile, manifest)
