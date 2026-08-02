"""cryptomatte.py 解码函数的单元测试。"""

import struct

from grf_ue_bridge.cryptomatte import (
    entity_names_from_mapping,
    hex_id_to_float,
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
