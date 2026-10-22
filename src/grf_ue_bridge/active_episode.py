"""从完整源轨迹确定性裁剪活跃 episode，保留重新生成所需的裁剪证据。"""

import copy
import hashlib
import json
import math
from pathlib import Path
import tempfile

from .active_trajectory import ActiveWindowPolicy, select_active_windows
from .config.models import ExportConfig, ResolvedTask
from .validator import validate_episode
from .workflows.task_export import _write_provenance


def _json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n').encode('utf-8')


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _policy_sha(policy):
    return _sha(json.dumps(policy, sort_keys=True, separators=(',', ':')).encode('utf-8'))


def _contained(path, root):
    try:
        path.relative_to(root)
    except ValueError:
        raise ValueError('输出路径不得逃逸 dataset_root: {}'.format(path))
    if path == root:
        raise ValueError('episode 输出不得等于 dataset_root')


def _build_manifests(resolved, folder, meta):
    """与 task export 使用相同的构建与验证入口，不能沿用源指纹。"""
    from .appearance import build_appearance_manifest, validate_appearance_manifest, write_appearance_manifest
    from .environment import build_environment_manifest, validate_environment_manifest, write_environment_manifest

    frames_path = folder / 'frames.jsonl'
    for key, filename, build, validate, write, argument in (
        ('appearance', 'appearance.json', build_appearance_manifest, validate_appearance_manifest, write_appearance_manifest, 'catalog'),
        ('environment_profile', 'environment.json', build_environment_manifest, validate_environment_manifest, write_environment_manifest, 'profile'),
    ):
        profile = resolved.ue_profile.get(key)
        if profile is None:
            continue
        manifest = build(meta, frames_path, profile)
        write(folder, manifest)
        written = json.loads((folder / 'provenance' / filename).read_text(encoding='utf-8'))
        errors = validate(written, meta=meta, frames_path=frames_path, **{argument: profile})
        if errors:
            raise ValueError('{} 校验失败: {}'.format(filename, '; '.join(errors)))


def prepare_active_episode(resolved, source_episode, selection_report, window_index=0):
    """新建裁剪结果并返回 index 条目；源目录只需 frames.jsonl 和 meta.json。

    resolved 必须为针对裁剪帧数生成的 ResolvedTask。只复制连续帧，不调用 GRF，
    不修改原文件；同输入可重复调用，已有不同内容或已有产物损坏时拒绝覆盖。
    """
    if not isinstance(resolved, ResolvedTask):
        raise TypeError('resolved 必须是 ResolvedTask')
    if type(window_index) is not int or window_index < 0:
        raise ValueError('window_index 必须是非负整数')
    source = Path(source_episode).resolve()
    root = Path(resolved.dataset_root).resolve()
    target = Path(resolved.trajectory_output).resolve()
    _contained(target, root)
    if target != Path(resolved.dataset_episode_dir).resolve():
        raise ValueError('trajectory_output 与 dataset_episode_dir 必须一致')
    if target.name != resolved.episode_name or any(c in resolved.episode_name for c in ('/', '\\')):
        raise ValueError('episode_name 必须与输出目录名一致且不能包含路径')
    if target == source or source in target.parents or target in source.parents:
        raise ValueError('源目录与输出目录不得重叠')

    source_meta_bytes = (source / 'meta.json').read_bytes()
    source_frame_bytes = (source / 'frames.jsonl').read_bytes()
    source_meta = json.loads(source_meta_bytes.decode('utf-8'))
    frames = [json.loads(line) for line in source_frame_bytes.decode('utf-8').splitlines() if line.strip()]
    if validate_episode(source) != 0:
        raise ValueError('原始 episode 验证失败')
    report = copy.deepcopy(selection_report)
    if report.get('schema') != 'futsalmot_active_trajectory_qc' or report.get('version') != 1 or report.get('status') != 'accepted':
        raise ValueError('必须提供 accepted 的活跃筛选报告 v1')
    if report.get('source_frame_count') != len(frames):
        raise ValueError('筛选报告源帧数不匹配')
    if report.get('policy_sha256') != _policy_sha(report.get('policy', {})):
        raise ValueError('筛选 policy SHA256 不匹配')
    if report.get('root_seed', source_meta['source']['seed']) != source_meta['source']['seed']:
        raise ValueError('筛选报告 seed 不匹配')
    if window_index >= len(report.get('selected_windows', [])):
        raise ValueError('window_index 超出报告窗口范围')
    dt = float(source_meta['timing']['source_step_seconds'])
    if not math.isclose(float(report['dt_s']), dt, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError('筛选报告 dt 与源 meta 不一致')
    fresh = select_active_windows(frames, dt_s=dt, gk_ids=report.get('goalkeeper_ids'), policy=ActiveWindowPolicy(**report['policy']))
    if fresh['selected_windows'] != report['selected_windows']:
        raise ValueError('筛选报告与当前源轨迹重新计算的窗口不一致')
    window = fresh['selected_windows'][window_index]
    start, end = window['start_index'], window['end_index_exclusive']
    count = end - start
    cfg = ExportConfig(**resolved.export_profile)
    fps = cfg.target_fps or 10
    if cfg.num_steps * max(1, fps // 10) != count or not math.isclose(1.0 / fps, dt, rel_tol=1e-8):
        raise ValueError('渲染 config.num_steps/target_fps 与裁剪窗帧数或 dt 不一致')
    if cfg.seed != source_meta['source']['seed'] or cfg.scenario != source_meta['source']['scenario']:
        raise ValueError('渲染 config seed/scenario 与源轨迹不一致')
    if cfg.trajectory_time_scale != 1.0:
        raise ValueError('裁剪任务不允许再次配置 trajectory_time_scale')
    if cfg.playback_fps != source_meta['timing']['playback_fps']:
        raise ValueError('裁剪任务 playback_fps 与源 meta 不一致')

    cropped = copy.deepcopy(frames[start:end])
    for index, frame in enumerate(cropped):
        frame['step'] = index
        frame['time_seconds'] = round(index * dt, 12)
    frame_bytes = b''.join((json.dumps(frame, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n').encode('utf-8') for frame in cropped)
    meta = copy.deepcopy(source_meta)
    meta['episode_id'] = resolved.episode_name
    meta['timing']['num_steps'] = count
    meta_bytes = _json_bytes(meta)
    capture_profile_path = source / 'provenance' / 'export-profile.json'
    capture_profile = json.loads(capture_profile_path.read_text(encoding='utf-8')) if capture_profile_path.is_file() else None
    capture = {
        'source_episode_id': source_meta['episode_id'],
        'source_frames_sha256': _sha(source_frame_bytes),
        'source_meta_sha256': _sha(source_meta_bytes),
        'source_capture_steps': len(frames),
        'source_capture_steps_semantics': '完整源 frames.jsonl 的采样帧数；不是裁剪后的任务步数，也不等同插值前的 GRF 引擎步数',
        'source_export_profile': capture_profile,
        'source_export_profile_available': capture_profile is not None,
        'reproduction_requires_full_source_capture': True,
        'reproduction': '先重新生成完整源轨迹并核验 frames SHA256，再按本记录连续 [start,end) 裁剪。禁止直接用裁剪任务的 num_steps 导出并声称复现。源目录仅存 frames/meta 时，不能据此声称已恢复全部 GRF 导出配置。',
    }
    provenance = {
        'schema': 'futsalmot_active_episode', 'version': 1,
        'root_seed': cfg.seed, 'window_index': window_index,
        **capture,
        'source_window': window,
        'policy': fresh['policy'], 'policy_sha256': fresh['policy_sha256'],
        'source_time_range_s': [frames[start]['time_seconds'], frames[end - 1]['time_seconds'] + dt],
        'source_step_range': [frames[start]['step'], frames[end - 1]['step'] + 1],
        'dt_s': dt, 'frame_count': count,
        'frames_sha256': _sha(frame_bytes), 'meta_sha256': _sha(meta_bytes),
    }
    fingerprint = _sha(_json_bytes({'crop': provenance, 'export': resolved.export_profile, 'ue': resolved.ue_profile}))
    provenance['input_fingerprint'] = fingerprint
    entry = {
        'root_seed': cfg.seed, 'episode_name': resolved.episode_name,
        'episode_dir': str(target), 'trajectory_output': str(target),
        'frame_count': count, 'num_steps': cfg.num_steps, 'dt_s': dt,
        'source_window': window, 'source_capture_steps': len(frames),
        'source_frames_sha256': capture['source_frames_sha256'],
        'policy_sha256': fresh['policy_sha256'], 'input_fingerprint': fingerprint,
        'active_provenance': str(target / 'provenance' / 'active_window.json'),
    }
    if target.exists():
        old_path = target / 'provenance' / 'active_window.json'
        old = json.loads(old_path.read_text(encoding='utf-8')) if old_path.is_file() else {}
        if old != provenance or (target / 'frames.jsonl').read_bytes() != frame_bytes or (target / 'meta.json').read_bytes() != meta_bytes:
            raise FileExistsError('拒绝覆盖已有不同输出: {}'.format(target))
        if validate_episode(target) != 0:
            raise ValueError('已有输出验证失败，不覆盖')
        # 重建到临时目录后比较 manifest，避免同名产物被修改后静默复用。
        with tempfile.TemporaryDirectory(prefix='.active-verify-', dir=str(root)) as temp:
            check = Path(temp)
            (check / 'frames.jsonl').write_bytes(frame_bytes)
            _build_manifests(resolved, check, meta)
            for name in ('appearance.json', 'environment.json'):
                expected = check / 'provenance' / name
                actual = target / 'provenance' / name
                if expected.is_file() and (not actual.is_file() or json.loads(expected.read_text(encoding='utf-8')) != json.loads(actual.read_text(encoding='utf-8'))):
                    raise ValueError('已有 {} 与当前输入不符，不覆盖'.format(name))
        return entry

    target.parent.mkdir(parents=True, exist_ok=True)
    # 临时目录与目标同盘；仅验证完成后发布，失败目录保留以便诊断。
    temp = Path(tempfile.mkdtemp(prefix='.' + target.name + '.active-', dir=str(target.parent)))
    (temp / 'frames.jsonl').write_bytes(frame_bytes)
    (temp / 'meta.json').write_bytes(meta_bytes)
    _write_provenance(resolved, temp)
    (temp / 'provenance' / 'active_window.json').write_bytes(_json_bytes(provenance))
    (temp / 'provenance' / 'source-meta.json').write_bytes(source_meta_bytes)
    (temp / 'provenance' / 'source-capture.json').write_bytes(_json_bytes(capture))
    (temp / 'provenance' / 'active-selection.json').write_bytes(_json_bytes(fresh))
    _build_manifests(resolved, temp, meta)
    if validate_episode(temp) != 0:
        raise ValueError('裁剪 episode 验证失败，诊断目录: {}'.format(temp))
    temp.rename(target)
    return entry
