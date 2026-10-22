# -*- coding: utf-8 -*-
"""把小模型权重及评测依赖放到独立研究目录，不修改两个既有虚拟环境。"""
import hashlib
import io
import json
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path('D:/projects/FutsalMOT_Research')


def fetch(url):
    with urllib.request.urlopen(url, timeout=120) as response:
        return response.read()


def main():
    overlay = ROOT / 'runtime_dependencies'
    overlay.mkdir(parents=True, exist_ok=True)
    records = []
    versions={'lap':'0.5.13','trackeval':'1.3.0','pycocotools':'2.0.11','tabulate':'0.9.0'}
    for package,version in versions.items():
        metadata = json.loads(fetch(f'https://pypi.org/pypi/{package}/{version}/json'))
        wheels = [x for x in metadata['urls'] if x['filename'].endswith('.whl') and ('cp310-cp310-win_amd64' in x['filename'] or 'py3-none-any' in x['filename'])]
        if not wheels:
            raise RuntimeError(f'{package} 没有匹配 Python3.10/Windows 的 wheel')
        wheel = wheels[0]
        data = fetch(wheel['url'])
        digest = hashlib.sha256(data).hexdigest()
        assert digest == wheel['digests']['sha256']
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            for info in archive.infolist():
                path = PurePosixPath(info.filename)
                if path.is_absolute() or '..' in path.parts:
                    raise ValueError('wheel 内部路径不安全')
                destination = overlay / Path(*path.parts)
                destination.resolve().relative_to(overlay.resolve())
                if not info.is_dir():
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(archive.read(info))
        records.append({'package': package, 'version': metadata['info']['version'], 'wheel': wheel['filename'], 'url': wheel['url'], 'sha256': digest})
        print('PREPARED_DEPENDENCY', package, metadata['info']['version'], flush=True)
    weight = ROOT / 'weights/yolo11n.pt'
    weight.parent.mkdir(parents=True, exist_ok=True)
    url = 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt'
    if not weight.exists():
        weight.write_bytes(fetch(url))
    records.append({'model': 'yolo11n.pt', 'url': url, 'path': str(weight), 'size': weight.stat().st_size, 'sha256': hashlib.sha256(weight.read_bytes()).hexdigest()})
    (ROOT / 'runtime_receipt.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
    print('PREPARED_RUNTIME', overlay, weight)


if __name__ == '__main__':
    main()
