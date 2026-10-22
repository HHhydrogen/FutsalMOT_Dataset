"""在独立证据目录比较现有 File 读取和 planar 读取、有限线程或进程并发。

只读 episode 的 EXR/golden mask，不更新生产模块或原数据。planar 仍整帧解压，
只是避免交错 RGBA 和再次提取 R；每帧须与 grouped 基线及 golden 逐像素一致。
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import gc
import hashlib
import json
from pathlib import Path
import statistics
import time

import numpy as np
import OpenEXR
from PIL import Image

from grf_ue_bridge import cryptomatte
from instance_mask import compute_instance_stats


def digest(array):
    return hashlib.sha256(array.tobytes()).hexdigest()


def bboxes(mask):
    return {str(key): list(value.bbox_xyxy)
            for key, value in compute_instance_stats(mask).items()}


def planar_proxies(channels, specification):
    """复用 grouped 的层序与 dtype，只把对应 R 暴露为 H×W×1，无数值近似。"""
    layers = {}
    for name, spec in specification.items():
        red_name = 'R' if name in ('RGB', 'RGBA') and 'R' in channels else name + '.R'
        if red_name not in channels:
            raise ValueError('planar 实验不支持无对应 R 的层: ' + name)
        red = channels[red_name].pixels
        if list(red.shape) != spec['shape']:
            raise ValueError('planar 与 grouped 尺寸不同: ' + name)
        if str(red.dtype) != spec['dtype']:
            raise ValueError('planar 与 grouped dtype 不同，禁止隐式精度转换: ' + name)
        layers[name] = red[:, :, None]
    return layers


def decode(task):
    path, output, reader, mapping, specification = task
    start = time.perf_counter()
    with OpenEXR.File(str(path), separate_channels=reader == 'planar') as file:
        read_at = time.perf_counter()
        manifest = next(json.loads(value) for key, value in file.header().items()
                        if key.startswith('cryptomatte/') and key.endswith('/manifest'))
        channels = file.channels()
        if reader == 'planar':
            layers = planar_proxies(channels, specification)
        else:
            layers = channels
            specification = {name: {'shape': list(channel.pixels.shape[:2]),
                                    'dtype': str(channel.pixels.dtype)}
                             for name, channel in channels.items()}
        ids, selected = cryptomatte.select_cryptomatte_id_channel(manifest, layers)
        select_at = time.perf_counter()
        mask, counts = cryptomatte.build_mask(manifest, ids, mapping)
        build_at = time.perf_counter()
        cryptomatte.save_mask_png(mask, output)
        save_at = time.perf_counter()
        result = {'exr': str(path), 'selected_layer': selected,
                  'id_uint32_sha256': digest(ids.view(np.uint32)),
                  'mask_sha256': digest(mask), 'pixel_counts': counts,
                  'bbox_xyxy': bboxes(mask),
                  'png_sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
                  'read_seconds': read_at - start,
                  'select_seconds': select_at - read_at,
                  'build_seconds': build_at - select_at,
                  'save_seconds': save_at - build_at,
                  'decode_seconds': save_at - start}
        golden_path = path.parent.parent / 'mask' / ('%06d.png' % (int(path.stem) + 1))
        with Image.open(golden_path) as image:
            golden = np.array(image)
        if not np.array_equal(mask, golden):
            raise AssertionError('与 golden 的 mask 像素不同: ' + str(path))
        if result['bbox_xyxy'] != bboxes(golden):
            raise AssertionError('与 golden 的可见 bbox 不同: ' + str(path))
        result['all_golden_pixels_and_bboxes_equal'] = True
    return result, specification


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frames-per-camera', type=int, default=2)
    parser.add_argument('--workers', type=int, nargs='+', default=[1, 2])
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--executor', choices=['thread', 'process'], default='thread')
    parser.add_argument('--readers', choices=['grouped', 'planar'], nargs='+', default=['grouped', 'planar'])
    args = parser.parse_args()
    if args.frames_per_camera < 1 or args.repeats < 1 or not args.workers or any(w not in (1, 2) for w in args.workers):
        parser.error('帧数和轮数须为正数，并发数仅允许 1 或 2，限制整帧 EXR 的内存峰值')
    try:
        args.output.resolve().relative_to(args.episode.resolve())
    except ValueError:
        pass
    else:
        parser.error('实验输出目录必须在既有 episode 之外')
    args.output.mkdir(parents=True, exist_ok=False)
    mapping = json.loads((Path(__file__).resolve().parents[1] / 'ue/actor_mapping.example.json').read_text(encoding='utf-8'))
    frames = []
    for camera in sorted(args.episode.glob('CineCam_*')):
        frames.extend(sorted((camera / 'render_mask').glob('*.exr'))[:args.frames_per_camera])
    if not frames:
        parser.error('未找到可用 EXR')
    all_records = {}
    reference = {}
    specifications = {}
    variants = [('grouped', 1)] + [(reader, workers) for reader in args.readers
                                  for workers in args.workers if (reader, workers) != ('grouped', 1)]
    invariant_fields = ('selected_layer', 'id_uint32_sha256', 'mask_sha256',
                        'pixel_counts', 'bbox_xyxy', 'png_sha256')
    run_plan = []
    for repeat in range(args.repeats):
        offset = repeat % len(variants)
        run_plan.extend((reader, workers, repeat)
                        for reader, workers in variants[offset:] + variants[:offset])
    for reader, workers, repeat in run_plan:
        variant = '%s_w%d_r%d' % (reader, workers, repeat + 1)
        tasks = [(path, args.output / variant / path.parent.parent.name / ('%06d.png' % (int(path.stem) + 1)),
                  reader, mapping, specifications.get(str(path))) for path in frames]
        started = time.perf_counter()
        if workers == 1:
            decoded = [decode(task) for task in tasks]
        else:
            executor_class = ThreadPoolExecutor if args.executor == 'thread' else ProcessPoolExecutor
            with executor_class(max_workers=workers) as executor:
                decoded = list(executor.map(decode, tasks))
        elapsed = time.perf_counter() - started
        records = []
        for (row, spec), path in zip(decoded, frames):
            if variant == 'grouped_w1_r1':
                reference[str(path)] = row
                specifications[str(path)] = spec
            else:
                for field in invariant_fields:
                    if row[field] != reference[str(path)][field]:
                        raise AssertionError('实验输出与 grouped 基线不同: %s %s' % (field, path))
            records.append(row)
        all_records[variant] = {'wall_seconds_including_verification': elapsed,
                                'sum_decode_seconds': sum(row['decode_seconds'] for row in records),
                                'mean_stage_seconds': {stage: statistics.mean(row[stage] for row in records)
                                                       for stage in ('read_seconds', 'select_seconds', 'build_seconds', 'save_seconds')},
                                'frames': records}
        print(variant, 'wall_seconds', elapsed, flush=True)
        gc.collect()
    result = {'scope': '只读真实4K golden对照；线程并发墙钟包含逐像素/可见bbox验证，缓存与机器负载会影响计时',
              'openexr_version': OpenEXR.__version__, 'numpy_version': np.__version__,
              'executor': args.executor,
              'production_module_sha256': hashlib.sha256(Path(cryptomatte.__file__).read_bytes()).hexdigest(),
              'frames': len(frames), 'variants': all_records,
              'all_id_bits_mask_png_counts_bboxes_equal': True}
    (args.output / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('ALL_VARIANTS_EXACT', len(frames), flush=True)


if __name__ == '__main__':
    main()
