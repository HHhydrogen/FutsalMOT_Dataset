"""cryptomatte.py 解码函数的单元测试。"""

import struct

import numpy as np
import pytest

from grf_ue_bridge.cryptomatte import (
    entity_names_from_mapping,
    hex_id_to_float,
    select_cryptomatte_id_channel,
)


def _float_to_hex_be(v):
    return struct.pack(">f", v).hex()


class TestHexIdToFloat:
    def test_roundtrip(self):
        for hx in ("1f21a2e4", "2154a9bc", "2d158ff6", "00000000"):
            v = hex_id_to_float(hx)
            assert _float_to_hex_be(v) == hx

    def test_known_player_l0(self):
        # UE 5.8 实测：Player_L0 = "1f21a2e4"，帧内 R 通道 float 值 ≈ 3.4228e-20
        v = hex_id_to_float("1f21a2e4")
        assert abs(v - 3.4228e-20) / 3.4228e-20 < 1e-4

    def test_known_ball(self):
        # Ball_01 = "2d158ff6" → ≈ 8.5016e-12
        v = hex_id_to_float("2d158ff6")
        assert abs(v - 8.5016e-12) / 8.5016e-12 < 1e-4


class TestEntityNamesFromMapping:
    def test_basic(self):
        m = entity_names_from_mapping({"L0": "Player_L0", "BALL": "Ball_01"})
        assert m == {"L0": "Player_L0", "BALL": "Ball_01"}


def _id_float(hex_id):
    return np.array([int(hex_id, 16)], dtype=np.uint32).view(np.float32)[0]


def test_selects_object_id_layer_when_rgba_is_final_image():
    player = _id_float("1f21a2e4")
    layers = {
        "RGBA": np.zeros((1, 2, 4), dtype=np.float32),
        "ActorHitProxyMask00": np.array(
            [[[player, 0, 0, 0], [0, 0, 0, 0]]], dtype=np.float32
        ),
        "ActorHitProxyMask01": np.zeros((1, 2, 4), dtype=np.float32),
    }
    selected, name = select_cryptomatte_id_channel(
        {"Player_L0": "1f21a2e4"}, layers
    )
    assert name == "ActorHitProxyMask00"
    assert selected.shape == (1, 2)
    assert selected[0, 0].view(np.uint32) == np.uint32(int("1f21a2e4", 16))


def test_legacy_rgba_layer_remains_supported():
    player = _id_float("1f21a2e4")
    layers = {"RGBA": np.array([[[player, 0, 0, 0]]], dtype=np.float32)}
    selected, name = select_cryptomatte_id_channel(
        {"Player_L0": "1f21a2e4"}, layers
    )
    assert name == "RGBA"
    assert selected.shape == (1, 1)


def test_selects_layer_with_most_manifest_hash_matches():
    player = _id_float("1f21a2e4")
    ball = _id_float("2d158ff6")
    layers = {
        "RGBA": np.array([[[player, 0, 0, 0], [0, 0, 0, 0]]], dtype=np.float32),
        "ActorHitProxyMask00": np.array(
            [[[player, 0, 0, 0], [ball, 0, 0, 0]]], dtype=np.float32
        ),
    }
    selected, name = select_cryptomatte_id_channel(
        {"Player_L0": "1f21a2e4", "Ball_01": "2d158ff6"}, layers
    )
    assert name == "ActorHitProxyMask00"
    assert selected[0, 1].view(np.uint32) == np.uint32(int("2d158ff6", 16))


def test_no_manifest_match_fails_with_candidate_names():
    with pytest.raises(ValueError, match="ActorHitProxyMask00"):
        select_cryptomatte_id_channel(
            {"Player_L0": "1f21a2e4"},
            {
                "RGBA": np.zeros((1, 1, 4), dtype=np.float32),
                "ActorHitProxyMask00": np.zeros((1, 1, 4), dtype=np.float32),
            },
        )
