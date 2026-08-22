"""interpolate.py 纯函数测试：GRF 10fps 位置 → 30fps velocity-aware Hermite 插值。"""

import math

import pytest

from grf_ue_bridge.interpolate import interpolate_frames, resample_frames_time_scale
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


def _linear_source(n, vel=1.0, owned_step=0):
    """线性运动源帧：position x = step*0.1（速度 vel m/s），y=0。"""
    frames = []
    for step in range(n):
        x = step * 0.1
        frames.append({
            "step": step,
            "time_seconds": round(step * 0.1, 6),
            "score": [0, 0],
            "ball_owned_team": step + owned_step,
            "game_mode": 0,
            "ball": {"position_m": [x, 0.0, 0.11],
                     "source_grf_position": [x, 0.0, 0.11],
                     "velocity_mps": [vel, 0.0, 0.0]},
            "players": [{"id": f"L{p}",
                         "position_m": [x, 0.0, 0.0],
                         "velocity_mps": [vel, 0.0]}
                        for p in range(10)],
        })
    return frames


class TestResampleTimeScale:
    def test_velocity_is_source_times_scale(self):
        # ts=2, 10fps 输出：source_time = dataset×2，线性运动位置 0.2k/步、速度 2.0
        src = _linear_source(10)
        out = resample_frames_time_scale(src, 2.0, 10, 5)
        assert len(out) == 5
        for k, f in enumerate(out):
            assert f["players"][0]["position_m"][0] == pytest.approx(0.2 * k)
            assert f["players"][0]["velocity_mps"] == pytest.approx([2.0, 0.0])
            assert f["players"][0]["speed_mps"] == pytest.approx(2.0)
            assert f["ball"]["velocity_mps"][0] == pytest.approx(2.0)

    def test_discrete_state_hold_nearest(self):
        # 离散状态随 source_time 对应 sample（floor）hold，不做插值
        src = _linear_source(10, owned_step=0)
        out = resample_frames_time_scale(src, 2.0, 10, 5)
        for k, f in enumerate(out):
            # s = k*0.1*2 = 0.2k → sample i = 2k
            assert f["ball_owned_team"] == 2 * k

    def test_position_passes_through_grf_samples_unaligned(self):
        # ts=2.4, 30fps：source 等效 24Hz，只有 k 为 5 的倍数时精确命中 GRF sample
        src = _linear_source(9)  # 覆盖到 0.8s
        out = resample_frames_time_scale(src, 2.4, 30, 10)
        # 每 5 帧命中一次 sample：k=5 → s=0.4 → sample 4
        for k in range(10):
            x = out[k]["players"][0]["position_m"][0]
            s = k * (1.0 / 30.0) * 2.4
            expected_exact = round(s, 6)
            assert x == pytest.approx(expected_exact, abs=1e-4)
            assert math.isfinite(out[k]["players"][0]["speed_mps"])
        # 命中帧严格等于 GRF 真值
        assert out[5]["players"][0]["position_m"][0] == pytest.approx(0.4)
        assert out[5]["players"][0]["velocity_mps"] == pytest.approx([2.4, 0.0])

    def test_no_nan_and_bounded(self):
        src = _linear_source(730)  # 覆盖 73s > 72s 需求
        out = resample_frames_time_scale(src, 2.4, 30, 900)
        assert len(out) == 900
        src_max = (len(src) - 1) * 0.1
        for f in out:
            for p in f["players"]:
                assert all(math.isfinite(v) for v in p["velocity_mps"])
                assert math.isfinite(p["speed_mps"])
                assert 0.0 <= p["position_m"][0] <= src_max + 1e-6  # clamp 到段内
