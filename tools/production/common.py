"""正式生成的原子记录、路径边界、天气分桶和确定性选样。"""
import hashlib
import json
import os
from pathlib import Path
import tempfile
from datetime import datetime, timezone

PRESETS = ('daylight_clear_v1', 'overcast_soft_v1', 'warm_evening_v1', 'cool_evening_v1',
           'light_haze_v1', 'cloudy_variation_v1', 'dawn_cool_v1', 'night_floodlit_v1')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def now():
    return datetime.now(timezone.utc).isoformat()


def task_id_for_row(row):
    """同seed重新开始使用独立attempt，避免覆盖此前失败产物。"""
    attempt = row.get('attempt', 1)
    if type(attempt) is not int or attempt < 1:
        raise ValueError('attempt必须是正整数')
    return 'fmot_v1_s{:08d}_c000_a{:02d}'.format(row['root_seed'], attempt)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name, dir=str(path.parent))
    try:
        with os.fdopen(fd, 'w', encoding='utf-8', newline='\n') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
        os.replace(temporary, str(path))
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def contained(path, parent):
    try:
        Path(path).resolve().relative_to(Path(parent).resolve())
        return True
    except ValueError:
        return False


def require_child(path, parent):
    path, parent = Path(path).resolve(), Path(parent).resolve()
    if path == parent or not contained(path, parent):
        raise ValueError('路径超出明确工作目录：{}'.format(path))
    return path


def load_settings(path):
    value = read(path)
    if value.get('schema') != 'futsalmot_production_release' or value.get('version') != 1:
        raise ValueError('不支持的正式生成配置')
    if value['resolution'] != [3840, 2160] or value['fps'] != 10 or value['camera_count'] != 4:
        raise ValueError('本次正式契约固定为原生4K、10FPS和四机位')
    if value['numbers_enabled'] or value['pose_enabled'] or value['auto_gpu_training']:
        raise ValueError('本次仅生成无号码RGB/实例mask/轨迹，不自动启动训练')
    require_child(value['private_root'], Path(value['project']) / 'Saved/Production')
    require_child(value['dataset_root'], 'G:/FutsalMOT_Dataset')
    require_child(value['work_root'], 'G:/FutsalMOT_Dataset/_work')
    if contained(value['work_root'], value['dataset_root']) or contained(value['dataset_root'], value['work_root']):
        raise ValueError('正式目录和工作目录不能重叠')
    if value['mcp_url'] != 'http://127.0.0.1:8000/mcp':
        raise ValueError('仅允许当前已配置的本机 Unreal MCP')
    return value


def select_release(reports, settings, budget_bytes):
    """优先最长连续窗；按天气均衡，并保留已渲染的两个训练组。"""
    pins = set(settings['adopt_seeds'])
    minimum = settings['minimum_clip_frames']
    if len({row['root_seed'] for row in reports}) != len(reports):
        raise ValueError('候选源seed重复')
    for train_count in settings['train_per_weather']:
        selected = []
        per_weather = train_count + settings['val_per_weather'] + settings['test_per_weather']
        for preset in PRESETS:
            candidates = [row for row in reports if row['preset'] == preset and row.get('status') == 'accepted'
                and row.get('selected_windows') and row['selected_windows'][0]['frame_count'] >= minimum]
            candidates.sort(key=lambda row: (-row['selected_windows'][0]['frame_count'], row['root_seed']))
            pinned = [row for row in candidates if row['root_seed'] in pins]
            if len(pinned) > train_count or len(candidates) < per_weather:
                raise ValueError('{}的合格长片段不足{}组；不缩短或伪造数据'.format(preset, per_weather))
            chosen = pinned + [row for row in candidates if row not in pinned][:per_weather - len(pinned)]
            ordered = sorted(chosen, key=lambda row: (canonical_sha([settings['split_seed'], preset, row['root_seed']]), row['root_seed']))
            trains = pinned + [row for row in ordered if row not in pinned][:train_count - len(pinned)]
            holdout = [row for row in ordered if row not in trains]
            for row in chosen:
                split = 'train' if row in trains else ('val' if row in holdout[:settings['val_per_weather']] else 'test')
                selected.append({'root_seed': row['root_seed'], 'preset': preset, 'split': split,
                    'frames_per_camera': row['selected_windows'][0]['frame_count'],
                    'source_frame_range': [row['selected_windows'][0]['start_index'], row['selected_windows'][0]['end_index_exclusive']],
                    'source_episode': row['source_episode'], 'source_frames_sha256': row['source_frames_sha256']})
        if not pins <= {row['root_seed'] for row in selected}:
            raise ValueError('已渲染组不满足正式选样条件，需单独检查')
        estimated = sum(row['frames_per_camera'] * settings['camera_count'] for row in selected) * settings['estimated_bytes_per_rgb_mask']
        if estimated <= budget_bytes:
            return {'entries': sorted(selected, key=lambda row: (row['root_seed'] not in pins, row['split'] != 'train', row['root_seed'])),
                    'train_per_weather': train_count, 'estimated_public_bytes': estimated}
    raise ValueError('当前容量不足以保留完整连续片段及每种天气的独立划分')
