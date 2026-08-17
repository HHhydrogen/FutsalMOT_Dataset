"""interpolate.py 纯函数测试：GRF 10fps 位置 → 30fps velocity-aware Hermite 插值。"""

import math

import pytest

from grf_ue_bridge.interpolate import interpolate_frames
from render_episode import select_rendered_frame_indices  # noqa: E402


def _frame(step, ball, players, score=(0, 0)):
    return {
        "step": step,
        "time_seconds": round(step * 0.1, 6),
        "score": list(score),
        "ball": {"position_m": list(ball), "source_grf_position": list(ball)},
        "players": [{"id": f"L{i}", "position_m": list(players[i])} for i in range(len(players))],
    }


def _frame_with_velocity(step, ball, ball_vel, players, player_vels):
    """构建带 velocity_mps 的帧（ball 3 维、player 2 维速度）。"""
    return {
        "step": step,
        "time_seconds": round(step * 0.1, 6),
        "score": [0, 0],
        "ball": {"position_m": list(ball), "source_grf_position": list(ball),
                 "velocity_mps": list(ball_vel)},
        "players": [
            {"id": f"L{i}", "position_m": list(players[i]), "velocity_mps": list(player_vels[i])}
            for i in range(len(players))
        ],
    }


def _ten_players(p):
    return [p] * 10


class TestInterpolateFrames:
    def test_length_factor(self):
        frames = [_frame(i, [i, i, 0.1], _ten_players([i, i, 0])) for i in range(3)]
        out = interpolate_frames(frames, 3, 1 / 30)
        assert len(out) == 9

    def test_exact_frames_preserved(self):
        """k%3==0 的下标帧 = 原 GRF 真值，位置原样保留。"""
        frames = [_frame(i, [i * 2.0, i, 0.1], _ten_players([i * 3.0, i * 4.0, 0]))
                  for i in range(3)]
        out = interpolate_frames(frames, 3, 1 / 30)
        for k in (0, 3, 6):
            src = frames[k // 3]
            assert out[k]["ball"]["position_m"] == src["ball"]["position_m"]
            assert out[k]["players"][0]["position_m"] == src["players"][0]["position_m"]

    def test_lerp_half(self):
        """frac=1/3、2/3 的位置 = 线性插值。"""
        f0 = _frame(0, [0.0, 0.0, 0.1], _ten_players([0.0, 0.0, 0]))
        f1 = _frame(1, [3.0, 3.0, 0.1], _ten_players([3.0, 6.0, 0]))
        out = interpolate_frames([f0, f1], 3, 1 / 30)
        assert out[1]["ball"]["position_m"] == [1.0, 1.0, 0.1]
        assert out[1]["players"][0]["position_m"] == [1.0, 2.0, 0.0]
        assert out[2]["ball"]["position_m"] == [2.0, 2.0, 0.1]
        assert out[2]["players"][0]["position_m"] == [2.0, 4.0, 0.0]

    def test_last_step_held(self):
        """最后一个 GRF 步之后无 next → 末尾保持该位置。"""
        f0 = _frame(0, [0.0, 0.0, 0.1], _ten_players([0.0, 0.0, 0]))
        f1 = _frame(1, [3.0, 3.0, 0.1], _ten_players([3.0, 3.0, 0]))
        out = interpolate_frames([f0, f1], 3, 1 / 30)
        assert out[4]["ball"]["position_m"] == [3.0, 3.0, 0.1]
        assert out[5]["ball"]["position_m"] == [3.0, 3.0, 0.1]

    def test_step_and_time(self):
        f0 = _frame(0, [0.0, 0.0, 0.1], _ten_players([0.0, 0.0, 0]))
        f1 = _frame(1, [1.0, 1.0, 0.1], _ten_players([1.0, 1.0, 0]))
        out = interpolate_frames([f0, f1], 3, 1 / 30)
        assert [f["step"] for f in out] == [0, 1, 2, 3, 4, 5]
        assert out[5]["time_seconds"] == round(5 / 30, 6)

    def test_score_carried(self):
        f0 = _frame(0, [0.0, 0.0, 0.1], _ten_players([0.0, 0.0, 0]), score=(1, 0))
        f1 = _frame(1, [1.0, 1.0, 0.1], _ten_players([1.0, 1.0, 0]), score=(1, 0))
        out = interpolate_frames([f0, f1], 3, 1 / 30)
        assert out[1]["score"] == [1, 0]

    def test_factor_one_is_identity(self):
        frames = [_frame(i, [i, i, 0.1], _ten_players([i, i, 0])) for i in range(2)]
        out = interpolate_frames(frames, 1, 0.1)
        assert len(out) == 2
        assert out[0]["ball"]["position_m"] == frames[0]["ball"]["position_m"]

    def test_empty(self):
        assert interpolate_frames([], 3, 1 / 30) == []


class TestFrameMapping30fps:
    def test_1to1_all_frames_kept(self):
        """900 步 30fps episode：渲染取帧 = 全部 → 渲 900 标 900（1:1）。"""
        idx = select_rendered_frame_indices(900, 1 / 30, 30)
        assert idx == list(range(900))
        assert len(idx) == 900


# ── velocity-aware Hermite 插值 ───────────────────────────────────────────

class TestHermiteInterpolation:
    def test_velocity_preserved_at_sample_points(self):
        """整数倍下标帧（真值帧）速度 = 原始 GRF 速度，位置不变。"""
        f0 = _frame_with_velocity(0, [0.0, 0.0, 0.1], [1.0, 0.0, 0.0],
                                  [[0.0, 0.0, 0.0]] * 10, [[2.0, 0.0]] * 10)
        f1 = _frame_with_velocity(1, [0.2, 0.1, 0.11], [1.0, 0.5, 0.0],
                                  [[0.1, 0.0, 0.0]] * 10, [[2.0, 0.0]] * 10)
        out = interpolate_frames([f0, f1], 3, 1 / 30)
        # k=0, 3 是真值帧
        assert out[0]["players"][0]["velocity_mps"] == [2.0, 0.0]
        assert out[3]["players"][0]["velocity_mps"] == [2.0, 0.0]
        assert out[0]["ball"]["velocity_mps"] == [1.0, 0.0, 0.0]
        assert out[0]["players"][0]["position_m"] == [0.0, 0.0, 0.0]
        assert out[3]["players"][0]["position_m"] == [0.1, 0.0, 0.0]

    def test_hermite_differs_from_linear_on_accel(self):
        """带速度切线时 Hermite 与线性不同（加减速场景）。"""
        # 减速：P0=0, V0=2 m/s；P1=0.1, V1=0（急停），h=0.1
        f0 = _frame_with_velocity(0, [0.0, 0.0, 0.1], [0.0, 0.0, 0.0],
                                  [[0.0, 0.0, 0.0]] * 10, [[2.0, 0.0]] * 10)
        f1 = _frame_with_velocity(1, [0.0, 0.0, 0.1], [0.0, 0.0, 0.0],
                                  [[0.1, 0.0, 0.0]] * 10, [[0.0, 0.0]] * 10)
        out = interpolate_frames([f0, f1], 3, 1 / 30)
        # u=1/3：Hermite 减速曲线，位置应大于线性（仍 < 0.1，被 clamp 在 [0,0.1] 内）
        x = out[1]["players"][0]["position_m"][0]
        assert 0.0 <= x <= 0.1
        assert out[2]["players"][0]["position_m"][0] >= out[1]["players"][0]["position_m"][0]

    def test_no_overshoot_with_opposite_tangents(self):
        """P0==P1 但切线相反（抖动）：Hermite 会 overshoot，clamp 后仍为 0。"""
        f0 = _frame_with_velocity(0, [0.0, 0.0, 0.1], [0.0, 0.0, 0.0],
                                  [[0.0, 0.0, 0.0]] * 10, [[10.0, 0.0]] * 10)
        f1 = _frame_with_velocity(1, [0.0, 0.0, 0.1], [0.0, 0.0, 0.0],
                                  [[0.0, 0.0, 0.0]] * 10, [[-10.0, 0.0]] * 10)
        out = interpolate_frames([f0, f1], 3, 1 / 30)
        for k in (1, 2):
            x = out[k]["players"][0]["position_m"][0]
            assert x == 0.0  # clamp 到 [0,0]，绝不 overshoot

    def test_no_overshoot_with_sudden_stop(self):
        """急停：V0 高、V1=0，插值位置不超过段端点。"""
        f0 = _frame_with_velocity(0, [0.0, 0.0, 0.1], [0.0, 0.0, 0.0],
                                  [[0.0, 0.0, 0.0]] * 10, [[5.0, 0.0]] * 10)
        f1 = _frame_with_velocity(1, [0.0, 0.0, 0.1], [0.0, 0.0, 0.0],
                                  [[0.5, 0.0, 0.0]] * 10, [[0.0, 0.0]] * 10)
        out = interpolate_frames([f0, f1], 3, 1 / 30)
        for k in (1, 2):
            assert 0.0 <= out[k]["players"][0]["position_m"][0] <= 0.5

    def test_position_continuity_and_finite(self):
        """位置连续（单调递增轨迹）且无 NaN。"""
        frames = [
            _frame_with_velocity(i, [0.1 * i, 0.0, 0.11], [1.0, 0.0, 0.0],
                                 [[0.2 * i, 0.0, 0.0]] * 10, [[2.0, 0.0]] * 10)
            for i in range(5)
        ]
        out = interpolate_frames(frames, 3, 1 / 30)
        assert len(out) == 15
        prev = out[0]["players"][0]["position_m"][0]
        for f in out[1:]:
            x = f["players"][0]["position_m"][0]
            assert x >= prev  # 单调不减
            assert math.isfinite(x)
            prev = x
        for f in out:
            for p in f["players"]:
                for v in p["velocity_mps"]:
                    assert math.isfinite(v)
            for v in f["ball"]["velocity_mps"]:
                assert math.isfinite(v)

    def test_velocity_units_are_mps(self):
        """速度单位为 m/s：匀速 2 m/s 跑 0.1s 位移 0.2m，插值速度恒为 2。"""
        n = 3
        frames = [
            _frame_with_velocity(i, [0.2 * i, 0.0, 0.11], [2.0, 0.0, 0.0],
                                 [[0.2 * i, 0.0, 0.0]] * 10, [[2.0, 0.0]] * 10)
            for i in range(n)
        ]
        out = interpolate_frames(frames, 3, 1 / 30)
        # 匀速 + 恒速度切线 → Hermite 退化为线性，速度恒为 2（非 hold 尾帧）
        last_sample_k = (n - 1) * 3
        for k in range(last_sample_k + 1):
            f = out[k]
            assert f["players"][0]["speed_mps"] == 2.0
            assert f["players"][0]["velocity_mps"] == [2.0, 0.0]
        # 中间帧位置 = 线性（0.2 * k/3，6 位小数舍入）
        for k in range(last_sample_k + 1):
            assert out[k]["players"][0]["position_m"][0] == pytest.approx(0.2 * k / 3, abs=1e-5)

    def test_speed_and_heading_recomputed(self):
        """speed_mps / movement_heading_deg 从插值速度重算。"""
        n = 2
        frames = [
            _frame_with_velocity(0, [0.0, 0.0, 0.1], [1.0, 0.0, 0.0],
                                 [[0.0, 0.0, 0.0]] * 10, [[0.0, 3.0]] * 10),
            _frame_with_velocity(1, [0.2, 0.0, 0.11], [2.0, 0.0, 0.0],
                                 [[0.0, 0.3, 0.0]] * 10, [[0.0, 3.0]] * 10),
        ]
        out = interpolate_frames(frames, 3, 1 / 30)
        last_sample_k = (n - 1) * 3
        for k in range(last_sample_k + 1):
            p = out[k]["players"][0]
            assert p["speed_mps"] == pytest.approx(3.0)
            assert p["movement_heading_deg"] == pytest.approx(90.0)

    def test_finite_difference_fallback_without_velocity(self):
        """无 velocity_mps 的旧帧：用有限差分切线，仍无 NaN、位置不变。"""
        frames = [_frame(i, [i, i, 0.1], _ten_players([i, i, 0])) for i in range(3)]
        out = interpolate_frames(frames, 3, 1 / 30)
        for f in out:
            for p in f["players"]:
                assert math.isfinite(p["speed_mps"])
            assert math.isfinite(f["ball"]["velocity_mps"][0])
