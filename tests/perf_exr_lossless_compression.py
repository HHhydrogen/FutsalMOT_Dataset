"""只读 2–3 张独立 golden EXR，比较 P1 的 PIZ/ZIP/ZIPS 无损重编码。

每个输出逐通道按 uint32/uint16 位模式、采样属性和完整 metadata 回读核验。
写入墙钟包含 OpenEXR.write 的编码与缓存写入，不含 fsync，也不代表 UE encoder。
"""

import argparse
import gc
import hashlib
import json
from pathlib import Path
import statistics
import time

import numpy as np
import OpenEXR


def normalize(value):
    if isinstance(value, np.ndarray):
        return {'dtype': str(value.dtype), 'shape': list(value.shape), 'values': value.tolist()}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (tuple, list)):
        return [normalize(item) for item in value]
    if isinstance(value, dict):
        return {key: normalize(item) for key, item in sorted(value.items())}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if type(value).__module__ == 'OpenEXR':
        return {'type': type(value).__name__, 'value': str(value)}
    raise TypeError('不支持的 metadata 类型，禁止跳过: ' + type(value).__name__)


def header_contract(header):
    return normalize({key: value for key, value in header.items()
                      if key not in ('channels', 'compression')})


def channel_contract(channel):
    pixels = channel.pixels
    if pixels.dtype not in (np.dtype('float16'), np.dtype('float32'), np.dtype('uint32')):
        raise TypeError('只支持 HALF/FLOAT/UINT channel')
    bits = pixels.view(np.uint16 if pixels.dtype.itemsize == 2 else np.uint32)
    return {'dtype': str(pixels.dtype), 'shape': list(pixels.shape),
            'xSampling': channel.xSampling, 'ySampling': channel.ySampling,
            'pLinear': channel.pLinear,
            'pixel_bits_sha256': hashlib.sha256(bits.tobytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exr', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not 2 <= len(args.exr) <= 3:
        parser.error('只接受 2–3 张小样本，不做大吞吐压测')
    for source in args.exr:
        if not source.is_file():
            parser.error('源文件不存在: ' + str(source))
        try:
            args.output.resolve().relative_to(source.resolve().parent.parent.parent)
        except ValueError:
            pass
        else:
            parser.error('实验输出不得放在源 episode 数据目录内')
    args.output.mkdir(parents=True, exist_ok=False)
    compressions = [('PIZ', OpenEXR.PIZ_COMPRESSION),
                    ('ZIP', OpenEXR.ZIP_COMPRESSION),
                    ('ZIPS', OpenEXR.ZIPS_COMPRESSION)]
    rows = []
    for frame, source in enumerate(args.exr):
        started = time.perf_counter()
        with OpenEXR.File(str(source), separate_channels=True) as original:
            source_read_seconds = time.perf_counter() - started
            if len(original.parts) != 1 or original.header().get('type') != OpenEXR.scanlineimage:
                raise ValueError('本实验仅支持 single-part scanline EXR')
            channels = original.channels()
            header = dict(original.header())
            expected_header = header_contract(header)
            expected_channels = {name: channel_contract(channel) for name, channel in channels.items()}
            original_compression = str(header['compression'])
            # 每张图轮换顺序，避免始终由同一种编码先测。
            ordered = compressions[frame:] + compressions[:frame]
            for name, compression in ordered:
                output = args.output / ('%02d_%s_%s.exr' % (frame, source.parent.parent.name, name))
                candidate_header = dict(header)
                candidate_header['compression'] = compression
                write_start = time.perf_counter()
                # File.__exit__ 会清空传入的字典，必须复制容器，保留原通道和缓冲区。
                with OpenEXR.File(candidate_header, dict(channels)) as encoded:
                    encoded.write(str(output))
                write_seconds = time.perf_counter() - write_start
                read_start = time.perf_counter()
                with OpenEXR.File(str(output), separate_channels=True) as decoded:
                    read_seconds = time.perf_counter() - read_start
                    actual_header = header_contract(decoded.header())
                    actual_channels = {key: channel_contract(channel) for key, channel in decoded.channels().items()}
                    if actual_header != expected_header:
                        changed = sorted(key for key in set(actual_header) | set(expected_header)
                                         if actual_header.get(key) != expected_header.get(key))
                        raise AssertionError('metadata 改变: ' + ', '.join(changed))
                    if actual_channels != expected_channels:
                        raise AssertionError('channel 名称/采样/dtype/像素位模式改变')
                    if decoded.header()['compression'] != compression:
                        raise AssertionError('compression 不符合请求')
                    # SHA 为证据摘要，同时实际逐位比较，不能用浮点近似相等。
                    for key, channel in channels.items():
                        dtype = np.uint16 if channel.pixels.dtype.itemsize == 2 else np.uint32
                        if not np.array_equal(channel.pixels.view(dtype), decoded.channels()[key].pixels.view(dtype)):
                            raise AssertionError('位模式不相等: ' + key)
                row = {'source': str(source), 'source_compression': original_compression,
                       'source_bytes': source.stat().st_size, 'source_read_seconds': source_read_seconds,
                       'compression': name, 'output': str(output), 'output_bytes': output.stat().st_size,
                       'write_seconds': write_seconds, 'read_seconds': read_seconds,
                       'channel_count': len(channels),
                       'uncompressed_channel_bytes': sum(c.pixels.nbytes for c in channels.values()),
                       'metadata_keys': len(expected_header),
                       'crypto_metadata': {key: value for key, value in expected_header.items() if key.startswith('cryptomatte/')},
                       'all_metadata_except_compression_equal': True,
                       'all_channel_bits_and_sampling_equal': True,
                       'channels': expected_channels}
                rows.append(row)
                print(json.dumps({key: row[key] for key in ('compression', 'output_bytes', 'write_seconds', 'read_seconds')}) , flush=True)
                del decoded, actual_channels, actual_header
                gc.collect()
        del original, channels, header, candidate_header, encoded
        gc.collect()
    aggregate = {}
    for name, _ in compressions:
        subset = [row for row in rows if row['compression'] == name]
        aggregate[name] = {'median_write_seconds': statistics.median(row['write_seconds'] for row in subset),
                           'mean_write_seconds': statistics.mean(row['write_seconds'] for row in subset),
                           'mean_read_seconds': statistics.mean(row['read_seconds'] for row in subset),
                           'total_bytes': sum(row['output_bytes'] for row in subset)}
    result = {'scope': 'P1 OpenEXR 小样本、单进程串行、D盘缓存写入；同时有UE MRQ负载；不等于UE实际编码器表现',
              'openexr_version': OpenEXR.__version__, 'numpy_version': np.__version__,
              'source_files': len(args.exr), 'all_channels_bits_and_metadata_equal': True,
              'aggregate': aggregate, 'frames': rows}
    (args.output / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('ALL_LOSSLESS_CHANNELS_EXACT', json.dumps(aggregate), flush=True)


if __name__ == '__main__':
    main()
