"""import_grf_episode.py 的 Rotation.Z yaw 展开（_unwind_angle）测试。

验证写入 Sequencer 前的连续 yaw 在跨 ±180° 边界时保持最短路径旋转。
"""

import import_grf_episode as ig


def _unwrap_sequence(facings):
    """模拟 create_sequence 里对 Rotation.Z 的展开：首帧取原始值，后续累加最短角度差。"""
    continuous = None
    out = []
    for f in facings:
        if continuous is None:
            continuous = f
        else:
            continuous = ig._unwind_angle(continuous, f)
        out.append(continuous)
    return out


def test_unwrap_forward_crossing_180():
    facings = [170.0, 179.0, -176.0, -165.0]
    assert _unwrap_sequence(facings) == [170.0, 179.0, 184.0, 195.0]


def test_unwrap_backward_crossing_neg180():
    facings = [-170.0, -179.0, 176.0, 165.0]
    assert _unwrap_sequence(facings) == [-170.0, -179.0, -184.0, -195.0]


def test_unwrap_multiple_crossings():
    # 单调增方向连续两次跨 ±180°：160→174→183→195→200→210
    facings = [160.0, 174.0, -177.0, -165.0, -160.0, -150.0]
    assert _unwrap_sequence(facings) == [160.0, 174.0, 183.0, 195.0, 200.0, 210.0]


def test_unwrap_follows_shortest_reversal():
    # 跨 180° 后反向（最短路径是转回而非继续绕圈）
    facings = [174.0, -177.0, 178.0]
    assert _unwrap_sequence(facings) == [174.0, 183.0, 178.0]


def test_unwrap_keeps_non_crossing_unchanged():
    facings = [10.0, 30.0, 45.0, -10.0, -60.0]
    assert _unwrap_sequence(facings) == [10.0, 30.0, 45.0, -10.0, -60.0]