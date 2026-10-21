import json
from grf_ue_bridge.workflows.artifact_cleanup import apply_cleanup

def test_environment_cleanup_accepts_retired_local_report_with_restored_public(tmp_path):
    (episode, _camera) = _episode(tmp_path, application_state='restored')
    resolved = _resolved(tmp_path, policy='futsalmot_appearance_v1')
    resolved['ue_profile']['environment_profile'] = {'policy': 'futsalmot_environment_v1_5', 'catalog_id': 'futsalmot_indoor_environment_v1_5', 'environment_seed': 42, 'preset': 'daylight_clear_v1', 'catalog_fingerprint': 'a' * 64}
    (episode / 'provenance' / 'environment.json').write_text(json.dumps({'schema': 'futsalmot_episode_environment'}), encoding='utf-8')
    (episode / 'provenance' / 'environment_application.json').write_text(json.dumps({'state': 'restored'}), encoding='utf-8')
    result = apply_cleanup(episode, ['Camera_01'], resolved=resolved)
    assert not any(('缺少本机 application report' in problem for problem in result['gate_problems']))

def test_environment_cleanup_blocks_unresolved_report_and_preserves_provenance(tmp_path):
    (episode, camera) = _episode(tmp_path, application_state='restored')
    resolved = _resolved(tmp_path, policy='futsalmot_appearance_v1')
    resolved['ue_profile']['environment_profile'] = {'policy': 'futsalmot_environment_v1_5', 'catalog_id': 'futsalmot_indoor_environment_v1_5', 'environment_seed': 42, 'preset': 'daylight_clear_v1', 'catalog_fingerprint': 'a' * 64}
    (episode / 'provenance' / 'environment.json').write_text(json.dumps({'schema': 'futsalmot_episode_environment'}), encoding='utf-8')
    (episode / 'provenance' / 'environment_application.json').write_text(json.dumps({'state': 'rendering'}), encoding='utf-8')
    result = apply_cleanup(episode, ['Camera_01'], resolved=resolved)
    assert result['ok'] is False
    assert any(('environment' in problem for problem in result['gate_problems']))
    assert any(("state actual='rendering'" in problem for problem in result['gate_problems']))
    assert (episode / 'provenance' / 'environment.json').exists()
    assert (camera / 'mask' / '000001.png').exists()

def _resolved(tmp_path, policy='futsalmot_appearance_v1'):
    return {'task_id': 'environment-cleanup-test', 'repo_root': str(tmp_path), 'ue_profile': {'appearance': {'policy': policy}}, 'postprocess': {'yolo_pose': {'enabled': False}}}

def _episode(tmp_path, application_state='restored', audit_passed=True):
    episode = tmp_path / 'episode'
    camera = episode / 'Camera_01'
    (camera / 'mask').mkdir(parents=True)
    (camera / 'render_mask').mkdir()
    (episode / 'provenance').mkdir(parents=True)
    (episode / 'audit').mkdir()
    (camera / 'mask' / '000001.png').write_bytes(b'number')
    (camera / 'render_mask' / '000001.png').write_bytes(b'render-number')
    (camera / 'annotations.jsonl').write_text(json.dumps({'frame_index': 1, 'objects': [{'entity_id': 'L0'}]}) + '\n', encoding='utf-8')
    (episode / 'provenance' / 'appearance_application.json').write_text(json.dumps({'state': application_state}), encoding='utf-8')
    (episode / 'audit' / 'soak_audit_report.json').write_text(json.dumps({'passed': audit_passed, 'exit_code': 0 if audit_passed else 1, 'errors': [] if audit_passed else ['audit failed'], 'warnings': [], 'checks': {}}), encoding='utf-8')
    return (episode, camera)
