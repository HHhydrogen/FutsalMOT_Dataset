"""用独立临时文件比较 copy2、同卷硬链接和可选的真实跨卷移交。

只计文件移交，不计准备文件和 SHA256 核验。小文件集会命中系统缓存，结果不能
当作磁盘持续吞吐；字节/设备号/inode 核验才是本工具的正确性证据。
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ue.render_episode import _link_or_copy_frame_atomic


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def make_sources(root, count, frame_bytes):
    root.mkdir()
    chunk = bytes(range(256)) * 4096
    paths = []
    for index in range(count):
        path = root / ('frame_%06d.bin' % index)
        with path.open('wb') as stream:
            remaining = frame_bytes
            while remaining:
                data = chunk[:min(remaining, len(chunk))]
                stream.write(data)
                remaining -= len(data)
        paths.append(path)
    return paths


def measure(sources, target_root, operation, repeats, expect_link):
    records = []
    for repeat in range(repeats):
        target = target_root / ('repeat_%d' % repeat)
        target.mkdir(parents=True)
        started = time.perf_counter()
        for source in sources:
            operation(source, target / source.name)
        elapsed = time.perf_counter() - started
        linked = 0
        for source in sources:
            destination = target / source.name
            assert sha256(source) == sha256(destination)
            same = os.path.samefile(source, destination)
            assert same == expect_link
            linked += int(same)
        records.append({'seconds': elapsed, 'same_inode_files': linked,
                        'all_sha256_equal': True})
    return {'repeats': records, 'median_seconds': statistics.median(
        record['seconds'] for record in records)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scratch-parent', required=True, type=Path)
    parser.add_argument('--cross-volume-parent', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--files', type=int, default=8)
    parser.add_argument('--frame-mib', type=int, default=8)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    if min(args.files, args.frame_mib, args.repeats) < 1:
        parser.error('文件数、每帧 MiB 和重复次数必须为正整数')
    if args.files * args.frame_mib * args.repeats > 1024:
        parser.error('本工具仅用于小样本，文件总 MiB × 重复次数不得超过 1024')
    if args.output.exists():
        parser.error('证据文件已存在，必须使用新的输出路径')
    args.scratch_parent.mkdir(parents=True, exist_ok=True)
    frame_bytes = args.frame_mib * 1024 * 1024
    with tempfile.TemporaryDirectory(prefix='handoff_bench_', dir=str(args.scratch_parent)) as scratch:
        root = Path(scratch)
        sources = make_sources(root / 'source', args.files, frame_bytes)
        result = {'files': args.files, 'frame_bytes': frame_bytes,
                  'payload_bytes_per_repeat': args.files * frame_bytes,
                  'scratch_device': root.stat().st_dev,
                  'scope': '临时文件、缓存可命中、不含准备和校验；不代表持续磁盘吞吐'}
        result['copy2'] = measure(sources, root / 'copy2', shutil.copy2, args.repeats, False)
        result['atomic_same_volume'] = measure(
            sources, root / 'atomic', _link_or_copy_frame_atomic, args.repeats, True)
        if args.cross_volume_parent:
            args.cross_volume_parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix='handoff_bench_', dir=str(args.cross_volume_parent)) as cross:
                cross_root = Path(cross)
                if cross_root.stat().st_dev == root.stat().st_dev:
                    parser.error('跨卷目录与 scratch 实际为同一设备，不能声称跨卷回退已验收')
                result['cross_device'] = cross_root.stat().st_dev
                result['atomic_cross_volume'] = measure(
                    sources, cross_root, _link_or_copy_frame_atomic, args.repeats, False)
        result['same_volume_payload_reads_avoided_per_repeat'] = args.files * frame_bytes
        result['same_volume_payload_writes_avoided_per_repeat'] = args.files * frame_bytes
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
