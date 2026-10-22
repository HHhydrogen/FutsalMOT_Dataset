"""持有场景的完成证据；供 UE 写入、P1 只读校验，禁止伪装为 restored。"""
import hashlib
import json
from pathlib import Path


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def retained_completion(episode):
    """仅完整且与实际文件绑定的 retained 收据可以替代恢复门禁。"""
    episode = Path(episode)
    path = episode / 'provenance/scene_completion.json'
    if not path.exists():
        return None
    receipt = json.loads(path.read_text(encoding='utf-8'))
    if receipt.get('schema') != 'futsalmot_scene_completion' or receipt.get('version') != 1:
        raise ValueError('场景完成收据 schema 不受支持')
    if receipt.get('lifecycle') != 'retain' or receipt.get('state') != 'retained' or receipt.get('live_readback_verified') is not True:
        raise ValueError('场景未验证为 retained')
    if not receipt.get('session_id') or not receipt.get('checkpoint_sha256'):
        raise ValueError('缺少场景会话或 checkpoint 身份')
    expected_files = {'render_summary.json', 'provenance/appearance_application.json',
        'provenance/environment_application.json', 'frames.jsonl', 'provenance/appearance.json', 'provenance/environment.json'}
    if set(receipt.get('files_sha256', {})) != expected_files:
        raise ValueError('场景完成收据文件集合不完整')
    for relative, expected in receipt['files_sha256'].items():
        if file_sha(episode / relative) != expected:
            raise ValueError('场景完成收据内容已改变：' + relative)
    summary = json.loads((episode / 'render_summary.json').read_text(encoding='utf-8'))
    if summary.get('status') != 'success' or summary.get('episode_id') != receipt.get('episode_id'):
        raise ValueError('保留场景的渲染未成功或 episode 身份不一致')
    for kind, field in [('appearance', 'manifest_fingerprint'), ('environment', 'environment_manifest_fingerprint')]:
        report = json.loads((episode / ('provenance/' + kind + '_application.json')).read_text(encoding='utf-8'))
        manifest = json.loads((episode / ('provenance/' + kind + '.json')).read_text(encoding='utf-8'))
        if report.get('state') != 'applied' or report.get('scene_lifecycle') != 'retain' or report.get('mrq_pending') is not False or report.get('errors'):
            raise ValueError(kind + ' 场景保留状态不完整')
        if report.get('session_id') != receipt['session_id'] or report.get(field) != manifest.get('fingerprint'):
            raise ValueError(kind + ' 场景会话/manifest 不一致')
        if (report.get('render_result') or {}).get('status') != 'success':
            raise ValueError(kind + ' 缺少成功渲染终态')
    return receipt


def admission_has_completed_scene(admission):
    """兼容历史恢复收据，同时严格区分实际保留状态。"""
    if admission.get('scene_lifecycle') == 'retain':
        return (admission.get('scene_retained') is True and admission.get('appearance_restored') is False
            and admission.get('environment_restored') is False
            and isinstance(admission.get('scene_completion_sha256'), str)
            and len(admission['scene_completion_sha256']) == 64)
    return admission.get('appearance_restored') is True and admission.get('environment_restored') is True
