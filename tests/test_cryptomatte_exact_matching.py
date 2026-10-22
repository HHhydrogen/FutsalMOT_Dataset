"""优化后的分块整数查找与原始 isin/稳定排序语义逐位一致。"""
import numpy as np
import pytest

from grf_ue_bridge.cryptomatte import _count_manifest_hits, select_cryptomatte_id_channel


def reference_select(manifest, layers):
    targets = np.asarray([int(value, 16) for value in manifest.values()], dtype=np.uint32)
    candidates = []
    for name, pixels in layers.items():
        array = np.asarray(pixels)
        if array.ndim != 3 or array.shape[2] < 1:continue
        if array.dtype != np.float32:array = np.asarray(array, dtype=np.float32)
        red = np.ascontiguousarray(array[:, :, 0], dtype=np.float32)
        hits = int(np.isin(red.view(np.uint32), targets).sum())
        candidates.append((hits, int(name.upper() == 'RGBA'), name, red))
    candidates.sort(key=lambda item: item[:2], reverse=True)
    return candidates[0][3], candidates[0][2]


def test_chunk_boundaries_duplicates_and_all_uint32_bits_match_isin():
    rng = np.random.default_rng(2091)
    bits = rng.integers(0, 2**32, size=(601, 2000), dtype=np.uint32)[:, ::2]
    targets = np.array([0, 2**32-1, 0x7FC00001, 0xFFC00001, 0x80000000, 17, 17], dtype=np.uint32)
    bits[523:526, 1:8] = targets
    assert _count_manifest_hits(bits, np.unique(targets)) == int(np.isin(bits, targets).sum())


@pytest.mark.parametrize('dtype', [np.float16, np.float32, np.float64])
def test_layer_and_red_bits_match_reference_for_all_source_dtypes(dtype):
    rng = np.random.default_rng(1021)
    layers = {'FinalImage': rng.random((17, 29, 4)).astype(dtype),
              'ActorHitProxyMask00': rng.random((17, 29, 4)).astype(np.float32),
              'ActorHitProxyMask01': rng.random((17, 29, 4)).astype(np.float32)}
    # 目标含 NaN 的不同 payload，以及带符号零，不能改用 float 近似比较。
    targets = np.array([0x7FC00001, 0xFFC00001, 0x80000000, 0, 0x1F21A2E4], dtype=np.uint32)
    layers['ActorHitProxyMask01'][:5, 0, 0] = targets.view(np.float32)
    manifest = {str(index):format(int(value), '08x') for index,value in enumerate(targets)}
    actual,name = select_cryptomatte_id_channel(manifest,layers)
    expected,expected_name = reference_select(manifest,layers)
    assert name == expected_name == 'ActorHitProxyMask01'
    assert np.array_equal(actual.view(np.uint32),expected.view(np.uint32))


def test_ties_keep_rgba_priority_and_stable_non_rgba_order():
    pixels = np.ones((2, 2, 4), dtype=np.float32)
    manifest = {'actor':'3f800000'}
    assert select_cryptomatte_id_channel(manifest,{'first':pixels,'second':pixels})[1] == 'first'
    assert select_cryptomatte_id_channel(manifest,{'first':pixels,'RGBA':pixels,'last':pixels})[1] == 'RGBA'


@pytest.mark.parametrize('count', [3, 12])
def test_lookup_hash_collisions_use_full_bits_without_double_count(count):
    targets = (np.arange(count,dtype=np.uint32) << np.uint32(16)) | np.uint32(17)
    bits = np.array([[17, 18, 0, 65535, 0xFFFFFFFF], list(targets[:3]) + [17, 18]],dtype=np.uint32)
    assert _count_manifest_hits(bits,targets) == int(np.isin(bits,targets).sum())


def test_larger_later_layer_can_beat_fully_matching_smaller_layer():
    pixels = np.ones((2, 2, 4),dtype=np.float32)
    larger = np.ones((3, 3, 4),dtype=np.float32)
    assert select_cryptomatte_id_channel({'actor':'3f800000'},{'first':pixels,'later':larger})[1] == 'later'
