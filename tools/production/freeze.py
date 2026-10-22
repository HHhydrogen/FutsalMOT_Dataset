"""隔离已验收v1运行时与开发中的号码代码，写入逐文件摘要。"""
import argparse
from pathlib import Path
import shutil
from common import canonical_sha, dump, load_settings, now, read, sha


def freeze(config):
    settings=load_settings(config)
    private=Path(settings['private_root'])
    runtime=private/'runtime'
    receipt_path=private/'runtime_freeze.json'
    if receipt_path.exists():
        receipt=read(receipt_path)
        for relative,expected in receipt['files'].items():
            if sha(runtime/relative)!=expected:
                raise ValueError('已冻结runtime发生变化：{}'.format(relative))
        return receipt
    if runtime.exists():
        raise ValueError('存在没有摘要的runtime，保留现场不覆盖')
    source=Path(settings['source_runtime'])
    code=Path(__file__).resolve().parents[2]
    png_gate=Path(settings['legacy_index']).parent/'paper_4k_png_filter_gate_20261022_retry1_preflight.json'
    expected_dll=read(png_gate)['module_dll_sha256']
    dll=Path(settings['project'])/'Binaries/Win64/UnrealEditor-FutsalMOT.dll'
    if sha(dll)!=expected_dll:
        raise ValueError('当前DLL与4K完整PNG门禁实际验收版本不一致')
    for folder in ('src','ue','configs'):
        shutil.copytree(source/folder,runtime/folder,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for name in ('pyproject.toml','uv.lock','external_sources.lock.json','.python-version'):
        shutil.copy2(source/name,runtime/name)
    shutil.copy2(code/'src/grf_ue_bridge/seed_dataset.py',runtime/'src/grf_ue_bridge/seed_dataset.py')
    shutil.copytree(code/'tools/production',runtime/'tools/production',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copy2(code/'ue/production_control.py',runtime/'ue/production_control.py')
    baseline=runtime/'baselines';baseline.mkdir()
    shutil.copy2(Path(settings['project'])/'Saved/EnvironmentPreviews/weather_baseline_20261021.json',baseline/'weather.json')
    shutil.copy2(Path(settings['project'])/'Saved/EnvironmentPreviews/weather_20261021/manual_lights_baseline.json',baseline/'manual_lights.json')
    receipt={'schema':'futsalmot_production_runtime_freeze','version':1,'created_utc':now(),
        'settings_sha256':canonical_sha(settings),'module_dll_sha256':expected_dll,
        'base_runtime':str(source),'files':{p.relative_to(runtime).as_posix():sha(p) for p in runtime.rglob('*') if p.is_file()}}
    dump(receipt_path,receipt)
    return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args()
    print('PRODUCTION_RUNTIME_FROZEN files='+str(len(freeze(args.config)['files'])),flush=True)
