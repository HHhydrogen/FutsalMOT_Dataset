"""interpolate.py 纯函数测试：GRF 10fps 位置 → 30fps 线性插值。"""

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
