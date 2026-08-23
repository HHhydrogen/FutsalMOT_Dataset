"""player_motion.py 纯函数测试：角度工具 / facing / 步态 / 运动状态。"""

import math

import pytest

from player_motion import (
    Gait,
    MotionConfig,
    MotionState,
    PlayerMotionTracker,
    compute_facing_yaw,
    compute_gait,
    compute_motion_state,
    compute_player_motion_sequence,
    diagnose_trajectory,
    heading_from_velocity_deg,
    move_towards_angle_deg,
    normalize_angle_deg,
    shortest_angle_delta_deg,
    write_motion_debug_jsonl,
)


# ── 角度工具 ─────────────────────────────────────────────────────────────

class TestAngleHelpers:
    def test_normalize_wrap(self):
        assert normalize_angle_deg(179.0) == 179.0
        assert normalize_angle_deg(181.0) == -179.0
        assert normalize_angle_deg(-181.0) == 179.0
        assert normalize_angle_deg(0.0) == 0.0
        assert abs(normalize_angle_deg(540.0)) == 180.0

    def test_shortest_delta_wrap(self):
        assert shortest_angle_delta_deg(179.0, -179.0) == pytest.approx(2.0)
        assert shortest_angle_delta_deg(-179.0, 179.0) == pytest.approx(-2.0)
        assert shortest_angle_delta_deg(90.0, 180.0) == pytest.approx(90.0)
        assert shortest_angle_delta_deg(0.0, -90.0) == pytest.approx(-90.0)

    def test_move_towards_clamps(self):
        assert move_towards_angle_deg(0.0, 180.0, 45.0) == pytest.approx(45.0)
        assert move_towards_angle_deg(0.0, -90.0, 45.0) == pytest.approx(-45.0)
        # 179 → -179 最短差 2 度：限速 1 度 → 180；限速 3 度 → 到达 -179（wrap 正确）
        assert move_towards_angle_deg(179.0, -179.0, 1.0) == pytest.approx(180.0)
        assert move_towards_angle_deg(179.0, -179.0, 3.0) == pytest.approx(-179.0)
        assert move_towards_angle_deg(0.0, 30.0, 100.0) == pytest.approx(30.0)

    def test_heading(self):
        assert heading_from_velocity_deg([1.0, 0.0]) == pytest.approx(0.0)
        assert heading_from_velocity_deg([0.0, 1.0]) == pytest.approx(90.0)
        assert heading_from_velocity_deg([-1.0, 0.0]) == pytest.approx(180.0)
        assert heading_from_velocity_deg([0.0, 0.0]) is None
        assert heading_from_velocity_deg([float("nan"), 0.0]) is None


# ── Facing / Yaw ─────────────────────────────────────────────────────────

class TestComputeFacingYaw:
    def test_idle_holds_previous(self):
        yaw = compute_facing_yaw([0.0, 0.0], 45.0, 0.1, 0.3, 360.0, 0.1)
        assert yaw == 45.0

    def test_slow_holds_previous(self):
        yaw = compute_facing_yaw([0.1, 0.0], 45.0, 0.1, 0.3, 360.0, 0.1)
        assert yaw == 45.0

    def test_already_aligned_keeps_heading(self):
        # 上一帧朝 +x，速度仍朝 +x → 朝向不变
        yaw = compute_facing_yaw([5.0, 0.0], 0.0, 0.1, 0.3, 360.0, 0.1)
        assert yaw == 0.0

    def test_normal_motion_turns_toward_velocity(self):
        # 上一帧朝 0，速度朝 +y（90 度）→ 朝向朝 90 收敛
        yaw = compute_facing_yaw([0.0, 5.0], 0.0, 0.1, 0.3, 360.0, 0.1)
        assert yaw > 0.0
        # 上一帧朝 0，速度朝 -y（-90 度）→ 朝向朝 -90 收敛
        yaw = compute_facing_yaw([0.0, -5.0], 0.0, 0.1, 0.3, 360.0, 0.1)
        assert yaw < 0.0

    def test_90_degree_turn_is_limited(self):
        yaw = compute_facing_yaw([0.0, 3.0], 0.0, 1 / 30, 0.3, 180.0, 0.0)
        assert 0.0 < yaw <= 6.0 + 1e-9

    def test_180_degree_turn_shortest_path(self):
        # 无平滑无上限时，180 度转身直接到目标（+180 或 -180 均可，已归一化）
        yaw = compute_facing_yaw([-5.0, 0.0], 0.0, 0.1, 0.3, 1e9, 0.0)
        assert abs(abs(yaw) - 180.0) < 1e-6
        # 限速下，一帧最多转 max_yaw_rate*dt 度，且方向正确（朝 +180 侧）
        yaw = compute_facing_yaw([-5.0, 0.0], 0.0, 0.1, 0.3, 360.0, 0.0)
        assert 0.0 < yaw <= 36.0 + 1e-9

    def test_wrap_179_to_neg179(self):
        # 直接验证最短差：179 → -179 为 +2 度（而非 -358）
        assert shortest_angle_delta_deg(179.0, -179.0) == pytest.approx(2.0)
        # 从 179 朝 -90 转向：跨过 180 边界，结果正确归一化到 (-180,180]
        yaw = compute_facing_yaw([0.0, -5.0], 179.0, 0.1, 0.3, 360.0, 0.0)
        assert -180.0 < yaw < 0.0  # 已跨过 +180 进入负半轴，无 358° 跳变

    def test_repeated_position_no_crash(self):
        # 重复坐标（速度 0）保持上一帧朝向，不 NaN 不崩溃
        yaw = compute_facing_yaw([0.0, 0.0], 30.0, 0.1, 0.3, 360.0, 0.1)
        assert yaw == 30.0
        yaw2 = compute_facing_yaw([0.0, 0.0], yaw, 0.1, 0.3, 360.0, 0.1)
        assert yaw2 == 30.0

    def test_tiny_dt_no_nan(self):
        yaw = compute_facing_yaw([5.0, 0.0], 0.0, 1e-6, 0.3, 360.0, 0.1)
        assert math.isfinite(yaw)

    def test_nonfinite_velocity_safe(self):
        assert compute_facing_yaw([float("nan"), 0.0], 45.0, 0.1, 0.3, 360.0, 0.1) == 45.0
        assert compute_facing_yaw([float("inf"), 0.0], 45.0, 0.1, 0.3, 360.0, 0.1) == 45.0

    def test_zero_or_negative_dt_safe(self):
        assert compute_facing_yaw([5.0, 0.0], 10.0, 0.0, 0.3, 360.0, 0.1) == 10.0
        assert compute_facing_yaw([5.0, 0.0], 10.0, -0.1, 0.3, 360.0, 0.1) == 10.0

    def test_no_smoothing_converges_fast(self):
        yaw = compute_facing_yaw([0.0, 3.0], 0.0, 0.1, 0.3, 10000.0, 0.0)
        assert yaw == pytest.approx(90.0)


# ── 步态 ─────────────────────────────────────────────────────────────────

class TestGait:
    def _cfg(self):
        return MotionConfig(
            idle_max_speed_mps=0.3, walk_max_speed_mps=1.0,
            jog_max_speed_mps=2.0, run_max_speed_mps=3.5,
        )

    def test_bands(self):
        cfg = self._cfg()
        assert compute_gait(0.0, Gait.IDLE, cfg) == Gait.IDLE
        assert compute_gait(0.3, Gait.IDLE, cfg) == Gait.IDLE
        assert compute_gait(0.5, Gait.IDLE, cfg) == Gait.WALK
        assert compute_gait(1.5, Gait.IDLE, cfg) == Gait.JOG
        assert compute_gait(2.5, Gait.IDLE, cfg) == Gait.RUN
        assert compute_gait(4.0, Gait.IDLE, cfg) == Gait.SPRINT

    def test_hysteresis(self):
        cfg = MotionConfig(
            idle_max_speed_mps=0.3, walk_max_speed_mps=1.0,
            jog_max_speed_mps=2.0, run_max_speed_mps=3.5,
            gait_hysteresis_mps=0.15,
        )
        assert compute_gait(2.5, Gait.RUN, cfg) == Gait.RUN
        assert compute_gait(1.9, Gait.RUN, cfg) == Gait.RUN
        assert compute_gait(1.9, Gait.IDLE, cfg) == Gait.JOG


# ── 运动状态 ─────────────────────────────────────────────────────────────

class TestMotionState:
    def _cfg(self):
        return MotionConfig(
            start_accel_mps2=1.2, decel_accel_mps2=-1.2, stop_speed_mps=0.4,
            start_max_speed_mps=1.5, decel_min_speed_mps=1.0,
            pivot_min_speed_mps=2.0, pivot_min_turn_rate_deg_s=90.0,
            speed_hysteresis_mps=0.15,
        )

    def test_idle(self):
        s = compute_motion_state(0.0, 0.0, 0.0, 0.0, MotionState.IDLE, self._cfg())
        assert s == MotionState.IDLE

    def test_start(self):
        s = compute_motion_state(1.0, 2.0, 0.0, 0.0, MotionState.IDLE, self._cfg())
        assert s == MotionState.START

    def test_locomotion(self):
        s = compute_motion_state(3.0, 0.5, 10.0, 3.0, MotionState.LOCOMOTION, self._cfg())
        assert s == MotionState.LOCOMOTION

    def test_decelerate(self):
        s = compute_motion_state(4.0, -2.0, 0.0, 4.0, MotionState.LOCOMOTION, self._cfg())
        assert s == MotionState.DECELERATE

    def test_stop(self):
        s = compute_motion_state(0.1, -1.0, 0.0, 4.0, MotionState.DECELERATE, self._cfg())
        assert s == MotionState.STOP

    def test_pivot(self):
        s = compute_motion_state(3.0, 0.0, 120.0, 3.0, MotionState.LOCOMOTION, self._cfg())
        assert s == MotionState.PIVOT

    def test_straight_run_then_pivot(self):
        cfg = self._cfg()
        # 直线跑（小角速度）→ locomotion
        s1 = compute_motion_state(3.0, 0.0, 10.0, 3.0, MotionState.LOCOMOTION, cfg)
        assert s1 == MotionState.LOCOMOTION
        # 急转（大角速度）→ pivot
        s2 = compute_motion_state(3.0, 0.0, 120.0, 3.0, s1, cfg)
        assert s2 == MotionState.PIVOT
        # 转完恢复直线 → locomotion
        s3 = compute_motion_state(3.0, 0.0, 10.0, 3.0, s2, cfg)
        assert s3 == MotionState.LOCOMOTION


# ── 追踪器 ───────────────────────────────────────────────────────────────

class TestPlayerMotionTracker:
    def _cfg(self):
        return MotionConfig(
            idle_max_speed_mps=0.3, walk_max_speed_mps=1.0,
            jog_max_speed_mps=2.0, run_max_speed_mps=3.5,
            min_facing_speed_mps=0.3, max_yaw_rate_deg_s=360.0,
            yaw_smoothing_time_s=0.0,
        )

    def test_idle_start_run_decel_stop_sequence(self):
        tracker = PlayerMotionTracker(config=self._cfg())
        # 静止
        p = tracker.update([0.0, 0.0, 0.0], [0.0, 0.0], 0.0)
        assert p["gait"] == Gait.IDLE
        assert p["motion_state"] == MotionState.IDLE
        # 起步加速
        p = tracker.update([0.05, 0.0, 0.0], [0.5, 0.0], 0.1)
        assert p["gait"] == Gait.WALK
        # 跑起来
        p = tracker.update([0.3, 0.0, 0.0], [2.5, 0.0], 0.2)
        assert p["gait"] == Gait.RUN
        assert p["motion_state"] == MotionState.LOCOMOTION
        # 减速
        p = tracker.update([0.45, 0.0, 0.0], [1.5, 0.0], 0.3)
        assert p["motion_state"] == MotionState.DECELERATE
        # 停止
        p = tracker.update([0.5, 0.0, 0.0], [0.0, 0.0], 0.4)
        assert p["gait"] == Gait.IDLE

    def test_facing_smooth_turn(self):
        tracker = PlayerMotionTracker(config=self._cfg())
        # 先朝 +x 跑
        tracker.update([0.0, 0.0, 0.0], [3.0, 0.0], 0.0)
        assert tracker.previous_facing_yaw_deg == pytest.approx(0.0, abs=1e-6)
        # 突然转向 +y：受 max_yaw_rate 限制，不会瞬间 90 度
        p = tracker.update([0.1, 0.1, 0.0], [0.0, 3.0], 1 / 30)
        assert 0.0 < p["facing_deg"] < 90.0

    def test_deterministic(self):
        cfg = self._cfg()
        def run():
            t = PlayerMotionTracker(config=cfg)
            out = []
            for i in range(10):
                out.append(t.update([i * 0.1, 0.0, 0.0], [1.0, 0.0], i * 0.1))
            return out
        a = run()
        b = run()
        assert a == b

    def test_fps_independence_gait_and_state(self):
        # 同一物理运动（匀速 2.5 m/s 跑 1 秒）在 10fps 与 30fps 下步态/状态一致
        cfg = self._cfg()
        def run_fps(fps):
            t = PlayerMotionTracker(config=cfg)
            dt = 1.0 / fps
            gaits = []
            states = []
            for i in range(fps):
                time_s = i * dt
                dist = 2.5 * time_s
                p = t.update([dist, 0.0, 0.0], [2.5, 0.0], time_s)
                gaits.append(p["gait"])
                states.append(p["motion_state"])
            return gaits, states
        g10, s10 = run_fps(10)
        g30, s30 = run_fps(30)
        assert set(g10) == set(g30) == {Gait.RUN}
        assert set(s10) == set(s30)


# ── 守门员 ball-aware facing ─────────────────────────────────────────────

class TestGKFaceBall:
    def _cfg(self):
        # 关闭平滑、高 yaw rate，便于验证"朝向收敛到球/保持运动朝向"的判定本身
        return MotionConfig(
            idle_max_speed_mps=0.3, walk_max_speed_mps=1.0,
            jog_max_speed_mps=2.0, run_max_speed_mps=3.5,
            min_facing_speed_mps=0.3, max_yaw_rate_deg_s=360.0,
            yaw_smoothing_time_s=0.0,
            gk_face_ball_max_speed_mps=2.0,
            gk_face_ball_max_angle_deg=90.0,
        )

    def test_normal_player_ignores_ball_facing(self):
        # 普通球员（face_ball=False）：传入球位置也不改变 movement-facing
        cfg = self._cfg()
        t = PlayerMotionTracker(config=cfg)
        p = t.update([0.0, 0.0, 0.0], [3.0, 0.0], 0.0,
                     ball_position_m=[-5.0, 0.0, 0.0], face_ball=False)
        # 应保持 movement heading（0°，朝 +x），而非面向球（180°）
        assert p["facing_deg"] == pytest.approx(0.0, abs=1e-6)

    def test_gk_low_speed_faces_ball_not_movement(self):
        # 低速（<2.0 m/s）：面向球（≈0°，球员上移 0.02 后球向角 -0.115°），
        # 而非 movement heading +y（90°）
        cfg = self._cfg()
        t = PlayerMotionTracker(config=cfg)
        t.update([0.0, 0.0, 0.0], [0.0, 0.0], 0.0,
                 ball_position_m=[10.0, 0.0, 0.0], face_ball=True)
        p = t.update([0.0, 0.02, 0.0], [0.0, 1.0], 0.1,
                     ball_position_m=[10.0, 0.0, 0.0], face_ball=True)
        ball_heading = math.degrees(math.atan2(-0.02, 10.0))
        assert abs(shortest_angle_delta_deg(p["facing_deg"], ball_heading)) <= 1e-6
        # 远小于 movement heading（90°），证明是面向球而非朝运动方向
        assert abs(shortest_angle_delta_deg(p["facing_deg"], 0.0)) < 0.2

    def test_gk_fast_faces_ball_within_90deg(self):
        # 高速（>=2.0 m/s）+ movement 与 ball 夹角 <=90°：面向球
        cfg = self._cfg()
        t = PlayerMotionTracker(config=cfg)
        p = t.update([0.0, 0.0, 0.0], [3.0, 0.0], 0.0,
                     ball_position_m=[5.0, 2.0, 0.0], face_ball=True)
        ball_heading = math.degrees(math.atan2(2.0, 5.0))  # ≈21.8°
        assert abs(shortest_angle_delta_deg(p["facing_deg"], ball_heading)) <= 1e-6

    def test_gk_fast_keeps_movement_heading_when_ball_behind(self):
        # 高速 + 球在正后方（夹角 180° > 90°）：保持 movement heading，不高速倒跑
        cfg = self._cfg()
        t = PlayerMotionTracker(config=cfg)
        p = t.update([0.0, 0.0, 0.0], [3.0, 0.0], 0.0,
                     ball_position_m=[-5.0, 0.0, 0.0], face_ball=True)
        assert abs(shortest_angle_delta_deg(p["facing_deg"], 0.0)) <= 1e-6

    def test_gk_not_stuck_facing_away(self):
        # 静止 GK：球从 +x 移到 -x，应在限速内转身面向球（不会长期背对）
        cfg = self._cfg()
        t = PlayerMotionTracker(config=cfg)
        t.update([0.0, 0.0, 0.0], [0.0, 0.0], 0.0,
                 ball_position_m=[10.0, 0.0, 0.0], face_ball=True)
        for i in range(1, 31):
            t.update([0.0, 0.0, 0.0], [0.0, 0.0], i / 30.0,
                     ball_position_m=[-10.0, 0.0, 0.0], face_ball=True)
        p = t.update([0.0, 0.0, 0.0], [0.0, 0.0], 31 / 30.0,
                     ball_position_m=[-10.0, 0.0, 0.0], face_ball=True)
        # 30 帧（1s，yaw rate 360°/s 足够转 180°）后应面向球（180°）
        assert abs(shortest_angle_delta_deg(p["facing_deg"], 180.0)) <= 1.0

    def test_gk_yaw_continuous_across_180(self):
        # 球跨越 ±180°（从 (0,-10) 前移到 (0,+10)），unwrap 后朝向应连续、无整圈
        def _unwind(prev, cur):
            while cur - prev > 180.0:
                cur -= 360.0
            while cur - prev < -180.0:
                cur += 360.0
            return cur

        cfg = self._cfg()
        t = PlayerMotionTracker(config=cfg)
        # 球沿 y 轴从 -10 移到 +10，heading 从 -90 → +90（经过 0 与 ±180 附近）
        ball_ys = [-10.0, -5.0, -1.0, -0.2, 0.2, 1.0, 5.0, 10.0]
        unwrapped = None
        prev = None
        max_jump = 0.0
        for i, by in enumerate(ball_ys):
            p = t.update([0.0, 0.0, 0.0], [0.0, 0.0], i / 10.0,
                         ball_position_m=[0.0, by, 0.0], face_ball=True)
            f = p["facing_deg"]
            unwrapped = f if unwrapped is None else _unwind(unwrapped, f)
            if prev is not None:
                max_jump = max(max_jump, abs(unwrapped - prev))
            prev = unwrapped
        # 每帧最多转 360/10=36°（yaw rate 限速），unwrap 后无 ~350° 整圈跳变
        assert max_jump <= 36.0 + 1e-6


# ── 批量工具 / 诊断 ───────────────────────────────────────────────────────

def _one_player_frame(step, time_s, pos, vel):
    return {
        "step": step,
        "time_seconds": time_s,
        "players": [{"id": "L0", "position_m": pos, "velocity_mps": vel}],
    }


class TestDiagnostics:
    def test_compute_motion_sequence_shape(self):
        frames = [_one_player_frame(i, i * 0.1, [i * 0.1, 0.0, 0.0], [1.0, 0.0])
                  for i in range(5)]
        seq = compute_player_motion_sequence(frames)
        assert len(seq) == 5
        assert "L0" in seq[0]["players"]
        assert "speed_mps" in seq[0]["players"]["L0"]
        assert "gait" in seq[0]["players"]["L0"]

    def test_position_discontinuity_detected(self):
        # 瞬移（位置跳跃 100m）→ 检测到 position_discontinuity
        frames = [
            _one_player_frame(0, 0.0, [0.0, 0.0, 0.0], [0.0, 0.0]),
            _one_player_frame(1, 0.1, [100.0, 0.0, 0.0], [0.0, 0.0]),
        ]
        warnings = diagnose_trajectory(frames)
        kinds = {w["kind"] for w in warnings}
        assert "position_discontinuity" in kinds

    def test_speed_spike_detected(self):
        # 速度超过 max_speed_mps → speed_spike
        frames = [
            _one_player_frame(0, 0.0, [0.0, 0.0, 0.0], [100.0, 0.0]),
            _one_player_frame(1, 0.1, [0.1, 0.0, 0.0], [100.0, 0.0]),
        ]
        warnings = diagnose_trajectory(frames)
        kinds = {w["kind"] for w in warnings}
        assert "speed_spike" in kinds

    def test_clean_trajectory_no_warnings(self):
        frames = [_one_player_frame(i, i * 0.1, [i * 0.1, 0.0, 0.0], [1.0, 0.0])
                  for i in range(5)]
        assert diagnose_trajectory(frames) == []

    def test_diagnose_deterministic(self):
        frames = [_one_player_frame(i, i * 0.1, [i * 0.1, 0.0, 0.0], [1.0, 0.0])
                  for i in range(5)]
        assert diagnose_trajectory(frames) == diagnose_trajectory(frames)

    def test_write_motion_debug(self, tmp_path):
        import json

        frames = [_one_player_frame(i, i * 0.1, [i * 0.1, 0.0, 0.0], [1.0, 0.0])
                  for i in range(5)]
        motion_path, warnings = write_motion_debug_jsonl(str(tmp_path / "dbg"), frames)
        assert warnings == []
        # motion sidecar 写成功且每行可解析
        with open(motion_path, encoding="utf-8") as f:
            lines = [json.loads(l) for l in f if l.strip()]
        assert len(lines) == 5
        assert "players" in lines[0]
