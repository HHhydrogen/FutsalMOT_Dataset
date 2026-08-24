"""player_motion.py 纯函数测试：角度工具 / facing / 步态 / 运动状态。"""

import math

import pytest

from player_motion import (
    AnimationClass,
    Gait,
    MotionConfig,
    MotionState,
    PlayerMotionTracker,
    compute_facing_yaw,
    compute_gait,
    compute_motion_state,
    compute_player_motion_sequence,
    diagnose_trajectory,
    gk_entity_ids_from_meta,
    heading_from_velocity_deg,
    move_towards_angle_deg,
    normalize_angle_deg,
    select_animation_class,
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

class TestRelativeMovementHeading:
    def _cfg(self):
        return MotionConfig(
            idle_max_speed_mps=0.3, walk_max_speed_mps=1.0,
            jog_max_speed_mps=2.0, run_max_speed_mps=3.5,
            min_facing_speed_mps=0.3, max_yaw_rate_deg_s=360.0,
            yaw_smoothing_time_s=0.0,
            gk_face_ball_max_speed_mps=2.0,
            gk_face_ball_max_angle_deg=90.0,
        )

    def _gk_frame(self, vel, ball_pos, speed=1.0):
        """GK 低速帧：facing 强制 = ball heading，movement heading = vel 方向。"""
        # 归一化 vel 到 speed=1（<2.0 → GK 面向球）
        h = math.atan2(vel[1], vel[0])
        v = [speed * math.cos(h), speed * math.sin(h)]
        cfg = self._cfg()
        t = PlayerMotionTracker(config=cfg)
        return t.update([0.0, 0.0, 0.0], v, 0.0,
                        ball_position_m=ball_pos, face_ball=True)

    def test_forward_zero(self):
        p = self._gk_frame([1.0, 0.0], [10.0, 0.0])
        assert p["facing_deg"] == pytest.approx(0.0, abs=1e-6)
        assert p["movement_heading_deg"] == pytest.approx(0.0, abs=1e-6)
        assert p["relative_movement_heading_deg"] == pytest.approx(0.0, abs=1e-6)

    def test_left_lateral_plus_90(self):
        p = self._gk_frame([0.0, 1.0], [10.0, 0.0])
        assert p["facing_deg"] == pytest.approx(0.0, abs=1e-6)
        assert p["movement_heading_deg"] == pytest.approx(90.0, abs=1e-6)
        assert p["relative_movement_heading_deg"] == pytest.approx(90.0, abs=1e-6)

    def test_right_lateral_minus_90(self):
        p = self._gk_frame([0.0, -1.0], [10.0, 0.0])
        assert p["facing_deg"] == pytest.approx(0.0, abs=1e-6)
        assert p["movement_heading_deg"] == pytest.approx(-90.0, abs=1e-6)
        assert p["relative_movement_heading_deg"] == pytest.approx(-90.0, abs=1e-6)

    def test_backward_180(self):
        p = self._gk_frame([-1.0, 0.0], [10.0, 0.0])
        assert p["facing_deg"] == pytest.approx(0.0, abs=1e-6)
        assert abs(shortest_angle_delta_deg(p["movement_heading_deg"], 180.0)) <= 1e-6
        # ±180（符号由 normalize 决定，此处为 +180）
        assert abs(abs(p["relative_movement_heading_deg"]) - 180.0) <= 1e-6

    def test_wrap_179_minus_179(self):
        # facing≈179、movement≈-179 → 相对角 ≈ +2（最短角），绝非 ±358
        ball_deg = math.radians(179.0)
        ball = [10.0 * math.cos(ball_deg), 10.0 * math.sin(ball_deg)]
        p = self._gk_frame([-1.0, -0.02], ball)  # movement ≈ -179°
        assert abs(p["facing_deg"] - 179.0) < 0.01
        assert abs(shortest_angle_delta_deg(p["movement_heading_deg"], -179.0)) < 0.5
        rel = p["relative_movement_heading_deg"]
        assert rel is not None and 1.0 < rel < 3.0  # 最短角 +2，无 ±358 环绕

    def test_gk_facing_differs_from_movement(self):
        # GK 面向球（+y），movement 为 +x → 相对角 -90（非零）
        p = self._gk_frame([1.0, 0.0], [0.0, 10.0])
        assert p["facing_deg"] == pytest.approx(90.0, abs=1e-6)
        assert p["relative_movement_heading_deg"] == pytest.approx(-90.0, abs=1e-6)

    def test_normal_player_relative_zero_when_following(self):
        # 普通球员 facing 跟随 movement → 相对角 0
        cfg = self._cfg()
        t = PlayerMotionTracker(config=cfg)
        p = t.update([0.0, 0.0, 0.0], [3.0, 0.0], 0.0)
        assert p["relative_movement_heading_deg"] == pytest.approx(0.0, abs=1e-6)


# ── Temporal Stabilization（raw_motion_state → animation_motion_state）─────

class TestTemporalStabilization:
    def _cfg(self):
        return MotionConfig(
            idle_max_speed_mps=0.3, walk_max_speed_mps=1.0,
            jog_max_speed_mps=2.0, run_max_speed_mps=3.5,
            min_facing_speed_mps=0.3, max_yaw_rate_deg_s=360.0,
            yaw_smoothing_time_s=0.0,
            stop_speed_mps=0.4, decel_min_speed_mps=1.0,
            decel_accel_mps2=-1.2, start_max_speed_mps=1.5,
            start_accel_mps2=1.2, pivot_min_speed_mps=2.0,
            pivot_min_turn_rate_deg_s=90.0,
        )

    # 通过 update 构造原始状态序列（控制速度/加速度/角速度）
    def _run_raw_states(self, states):
        """直接喂 raw 状态序列（经 _stabilize），返回 [(raw, anim, time)]。
        用独立 tracker + 手动调用，隔离 compute_motion_state 的影响。
        """
        t = PlayerMotionTracker(config=self._cfg())
        out = []
        for i, s in enumerate(states):
            time_s = i / 30.0
            anim = t._stabilize(s, time_s)
            out.append((s, anim, time_s))
        return out

    def _run_velocity(self, vels, fps=30):
        """按速度序列驱动 update()（位置 = 累计位移），返回 [(raw, anim, time)]。"""
        cfg = self._cfg()
        t = PlayerMotionTracker(config=cfg)
        dt = 1.0 / fps
        out = []
        px = py = 0.0
        for i, (vx, vy) in enumerate(vels):
            time_s = i * dt
            px += vx * dt
            py += vy * dt
            p = t.update([px, py, 0.0], [vx, vy], time_s)
            out.append((p["motion_state"], p["animation_motion_state"], time_s))
        return out

    def test_single_frame_start_filtered(self):
        # IDLE → START(1帧,0.033s<0.05确认) → LOCOMOTION：animation 不出现 START
        out = self._run_raw_states(["idle", "start", "locomotion"])
        anims = [a for _, a, _ in out]
        assert anims[0] == "idle"
        assert anims[1] == "idle"      # START 未确认，保持 idle
        assert anims[2] == "locomotion"
        assert "start" not in anims
        # raw 不变
        assert [r for r, _, _ in out] == ["idle", "start", "locomotion"]

    def test_sustained_start_enters(self):
        # START 持续 ≥0.05s → animation 进入 START
        out = self._run_raw_states(["idle", "start", "start", "start",
                                    "start", "start", "start", "locomotion"])
        anims = [a for _, a, _ in out]
        assert "start" in anims
        # 进入时刻：从首个 start 起需 ≥0.05s（30fps 下第 2 帧后）
        idx = anims.index("start")
        assert out[idx][2] - out[1][2] >= 0.05 - 1e-9

    def test_stop_short_event_triggers_and_holds(self):
        # LOCOMOTION → STOP 持续 0.05s 确认后进入，min_visible 0.2s 内不切走
        out = self._run_raw_states(
            ["locomotion", "stop", "stop", "stop", "locomotion",
             "locomotion", "locomotion", "locomotion", "locomotion",
             "locomotion", "locomotion"])
        anims = [a for _, a, _ in out]
        # STOP 需连续 0.05s（2 帧）确认：3 帧 stop 足够 → 进入
        assert "stop" in anims
        stop_idx = anims.index("stop")
        # min_visible 0.2s = 6 帧：进入后即使 raw 回 locomotion 也保持 stop
        stop_run = 0
        for a in anims[stop_idx:]:
            if a == "stop":
                stop_run += 1
            else:
                break
        assert stop_run >= 6  # 0.2s @30fps = 6 帧

    def test_single_frame_pivot_filtered(self):
        out = self._run_raw_states(["locomotion", "pivot", "locomotion",
                                    "locomotion", "locomotion"])
        anims = [a for _, a, _ in out]
        assert "pivot" not in anims

    def test_sustained_pivot_enters(self):
        # PIVOT 持续 ≥0.12s（5 帧@30fps）→ animation 进入 PIVOT
        out = self._run_raw_states(
            ["locomotion", "pivot", "pivot", "pivot", "pivot", "pivot",
             "pivot", "locomotion"])
        anims = [a for _, a, _ in out]
        assert "pivot" in anims
        idx = anims.index("pivot")
        assert out[idx][2] - out[1][2] >= 0.12 - 1e-9

    def test_no_oscillation_within_min_visible(self):
        # START 进入后，min_visible 内 raw 反复切 locomotion/start 不导致 anim 反复跳
        out = self._run_raw_states(
            ["idle", "start", "start", "start", "locomotion", "start",
             "locomotion", "start", "locomotion", "start", "locomotion",
             "locomotion", "locomotion", "locomotion", "locomotion"])
        anims = [a for _, a, _ in out]
        # START 只进入一次，且在 min_visible 内不被来回切换
        transitions = sum(1 for i in range(1, len(anims)) if anims[i] != anims[i - 1])
        assert anims.count("start") >= 4  # 进入后保持至少 ~0.2s
        assert transitions <= 3           # 不反复跳（idle→start→locomotion 最多 2-3 次）

    def test_fps_equivalence(self):
        # 同一 raw 状态时序（按秒定义）在 10fps 与 30fps 下 animation 状态序列等价
        t10 = PlayerMotionTracker(config=self._cfg())
        t30 = PlayerMotionTracker(config=self._cfg())
        # raw 时序（秒）：idle 0.2s → start 0.3s → locomotion 0.4s → pivot 0.3s → locomotion 0.3s
        events = [
            ("idle", 0.0, 0.2), ("start", 0.2, 0.5), ("locomotion", 0.5, 0.9),
            ("pivot", 0.9, 1.2), ("locomotion", 1.2, 1.5),
        ]
        anim_times = []
        for fps in (10, 30):
            t = t10 if fps == 10 else t30
            seen = []
            for state, t0, t1 in events:
                n = max(1, int(round((t1 - t0) * fps)))
                for j in range(n):
                    time_s = t0 + j / fps
                    anim = t._stabilize(state, time_s)
                    seen.append((state, anim, round(time_s, 4)))
            anim_times.append(seen)
        # 两种 fps 下 animation 状态与切换时刻一致（进入确认/最小可见基于秒）
        a10, a30 = anim_times
        # 比较 animation 状态序列（状态名序列，允许帧采样差异——取切换时序）
        def anim_series(rows):
            out = []
            for _, anim, t in rows:
                if not out or out[-1][1] != anim:
                    out.append((anim, t))
            return out
        s10, s30 = anim_series(a10), anim_series(a30)
        # 至少：起始与结束状态一致，且 PIVOT 正确进入（未被过度过滤）
        assert s10[0][0] == s30[0][0] == "idle"
        assert s10[-1][0] == s30[-1][0]
        assert "pivot" in [s for s, _ in s10] and "pivot" in [s for s, _ in s30]
        # 切换时刻（秒）两种 fps 相差不超过一个粗采样间隔
        assert abs(s10[-1][1] - s30[-1][1]) <= 0.1

    def test_raw_motion_state_unchanged(self):
        # animation 层存在不影响 raw motion_state（与 compute_motion_state 独立）
        cfg = self._cfg()
        t = PlayerMotionTracker(config=cfg)
        t.update([0.0, 0.0, 0.0], [0.0, 0.0], 0.0)  # 建立历史（speed 0）
        p = t.update([0.05, 0.0, 0.0], [0.5, 0.0], 0.1)  # accel=(0.5-0)/0.1=5 → START
        assert p["motion_state"] == MotionState.START
        assert p["animation_motion_state"] in (MotionState.IDLE, MotionState.START)
        # 再跑 compute_motion_state 直接比对 raw（prev_speed=0，首帧前速度）
        r = compute_motion_state(0.5, 5.0, 0.0, 0.0, MotionState.IDLE, cfg)
        assert p["motion_state"] == r


# ── Animation Selector（第一版）───────────────────────────────────────────

class TestAnimationSelector:
    def _cfg(self):
        return MotionConfig(
            idle_max_speed_mps=0.3, walk_max_speed_mps=1.0,
            jog_max_speed_mps=2.0, run_max_speed_mps=3.5,
            min_facing_speed_mps=0.3, max_yaw_rate_deg_s=360.0,
            yaw_smoothing_time_s=0.0,
        )

    # 1. pivot 正负 turn_rate → L/R
    def test_pivot_sign(self):
        assert select_animation_class("pivot", 120.0, None, False) == AnimationClass.PIVOT_L
        assert select_animation_class("pivot", -120.0, None, False) == AnimationClass.PIVOT_R
        assert select_animation_class("pivot", 0.0, None, False) == AnimationClass.PIVOT_R

    # 2. GK relative > 0 → gk_shuffle_r
    def test_gk_shuffle_r_when_relative_positive(self):
        assert select_animation_class("locomotion", 0.0, 90.0, True) == AnimationClass.GK_SHUFFLE_R
        assert select_animation_class("locomotion", 0.0, 45.0, True) == AnimationClass.GK_SHUFFLE_R

    # 3. GK relative < 0 → gk_shuffle_l
    def test_gk_shuffle_l_when_relative_negative(self):
        assert select_animation_class("locomotion", 0.0, -90.0, True) == AnimationClass.GK_SHUFFLE_L
        assert select_animation_class("locomotion", 0.0, -134.9, True) == AnimationClass.GK_SHUFFLE_L

    # 4. |relative| >= 135 → backpedal
    def test_gk_backpedal(self):
        assert select_animation_class("locomotion", 0.0, 135.0, True) == AnimationClass.GK_BACKPEDAL
        assert select_animation_class("locomotion", 0.0, -135.0, True) == AnimationClass.GK_BACKPEDAL
        assert select_animation_class("locomotion", 0.0, 180.0, True) == AnimationClass.GK_BACKPEDAL

    # 5. 非 GK 不触发 shuffle/backpedal
    def test_non_gk_no_directional(self):
        assert select_animation_class("locomotion", 0.0, 90.0, False) == AnimationClass.LOCOMOTION
        assert select_animation_class("locomotion", 0.0, -179.0, False) == AnimationClass.LOCOMOTION
        assert select_animation_class("locomotion", 0.0, 135.0, False) == AnimationClass.LOCOMOTION

    # 6. Pivot 优先于 GK Backpedal/Shuffle
    def test_pivot_priority(self):
        assert select_animation_class("pivot", 50.0, 170.0, True) == AnimationClass.PIVOT_L
        assert select_animation_class("pivot", -50.0, 90.0, True) == AnimationClass.PIVOT_R

    # 7-10. Selector temporal stabilization（仅 GK 方向类）
    def _stab(self, raws, fps=30):
        t = PlayerMotionTracker(config=self._cfg())
        out = []
        for i, rc in enumerate(raws):
            out.append((rc, t._stabilize_animation_class(rc, i / fps)))
        return out

    def test_single_frame_gk_shuffle_filtered(self):
        out = self._stab(["locomotion", "gk_shuffle_l", "locomotion"])
        anims = [a for _, a in out]
        assert anims == ["locomotion", "locomotion", "locomotion"]

    def test_sustained_gk_shuffle_enters_after_confirm(self):
        # 0.10s = 4 帧@30fps 后进入
        out = self._stab(["locomotion"] + ["gk_shuffle_r"] * 7 + ["locomotion"])
        anims = [a for _, a in out]
        idx = anims.index("gk_shuffle_r")
        assert idx >= 4  # enter_confirm 0.10s @30fps = 第4帧
        assert (idx - 1) / 30.0 >= 0.10 - 1e-9

    def test_min_visible_holds_gk_class(self):
        # 进入后 min_visible 0.15s 内不因 raw 回 locomotion 而立即切走
        out = self._stab(["locomotion"] + ["gk_shuffle_l"] * 5 + ["locomotion"] * 8)
        anims = [a for _, a in out]
        idx = anims.index("gk_shuffle_l")
        run = 0
        for a in anims[idx:]:
            if a == "gk_shuffle_l":
                run += 1
            else:
                break
        assert run >= 4  # 0.15s = 4.5 帧，至少保持 4 帧

    def test_selector_fps_equivalence(self):
        t10 = PlayerMotionTracker(config=self._cfg())
        t30 = PlayerMotionTracker(config=self._cfg())
        seqs = []
        for fps in (10, 30):
            t = t10 if fps == 10 else t30
            seen = []
            for i in range(int(1.0 * fps)):
                time_s = i / fps
                raw = "gk_shuffle_r" if 0.3 <= time_s < 0.6 else "locomotion"
                anim = t._stabilize_animation_class(raw, time_s)
                seen.append((raw, anim, round(time_s, 4)))
            seqs.append(seen)
        def series(rows):
            out = []
            for _, anim, t in rows:
                if not out or out[-1][1] != anim:
                    out.append((anim, t))
            return out
        s10, s30 = series(seqs[0]), series(seqs[1])
        assert s10[0][0] == s30[0][0] == "locomotion"
        assert s10[-1][0] == s30[-1][0]
        assert "gk_shuffle_r" in [s for s, _ in s10]
        assert "gk_shuffle_r" in [s for s, _ in s30]
        assert abs(s10[-1][1] - s30[-1][1]) <= 0.1

    # 11. frames.jsonl 完全不变（animation 字段不进 Frame schema）
    def test_animation_class_not_in_frame_schema(self):
        from grf_ue_bridge.schema import BallFrame, Frame, PlayerFrame
        players = [PlayerFrame(id=f"L{i}", position_m=[0.0, 0.0, 0.0]) for i in range(5)] \
            + [PlayerFrame(id=f"R{i}", position_m=[0.0, 0.0, 0.0]) for i in range(5)]
        f = Frame(
            step=0, time_seconds=0.0, score=[0, 0],
            ball=BallFrame(position_m=[0.0, 0.0, 0.11]),
            players=players,
        )
        d = f.model_dump()
        assert "animation_class" not in d
        assert "raw_animation_class" not in d
        assert "animation_motion_state" not in d


# ── GK 身份来源迁移（meta.entities[].is_goalkeeper）────────────────────────

class TestGKIdentityFromMeta:
    def _cfg(self):
        return MotionConfig(
            idle_max_speed_mps=0.3, walk_max_speed_mps=1.0,
            jog_max_speed_mps=2.0, run_max_speed_mps=3.5,
            min_facing_speed_mps=0.3, max_yaw_rate_deg_s=360.0,
            yaw_smoothing_time_s=0.0,
            gk_face_ball_max_speed_mps=2.0,
            gk_face_ball_max_angle_deg=90.0,
        )

    # 1. GK 在 index0：现有行为不变
    def test_gk_index0_unchanged(self):
        meta = {"entities": [
            {"id": "L0", "is_goalkeeper": True}, {"id": "L1", "is_goalkeeper": False},
            {"id": "R0", "is_goalkeeper": True}, {"id": "R1", "is_goalkeeper": False},
        ]}
        assert gk_entity_ids_from_meta(meta) == frozenset({"L0", "R0"})
        # L0 用 GK ball-aware facing：球在 +x、moving +y → rel +90 → raw gk_shuffle_r
        cfg = self._cfg()
        t = PlayerMotionTracker(config=cfg)
        p = t.update([0.0, 0.0, 0.0], [0.0, 1.0], 0.0,
                     ball_position_m=[10.0, 0.0, 0.0], face_ball=True, is_goalkeeper=True)
        assert p["facing_deg"] == pytest.approx(0.0, abs=1e-6)
        assert p["raw_animation_class"] == AnimationClass.GK_SHUFFLE_R

    # 2. GK 在 index1：GK 行为正确转移到 L1/R1
    def test_gk_index1_transfers(self):
        meta = {"entities": [
            {"id": "L0", "is_goalkeeper": False}, {"id": "L1", "is_goalkeeper": True},
            {"id": "R0", "is_goalkeeper": False}, {"id": "R1", "is_goalkeeper": True},
        ]}
        assert gk_entity_ids_from_meta(meta) == frozenset({"L1", "R1"})
        # L1 是 GK → 触发 GK 朝向 + selector
        cfg = self._cfg()
        t = PlayerMotionTracker(config=cfg)
        p = t.update([0.0, 0.0, 0.0], [0.0, 1.0], 0.0,
                     ball_position_m=[10.0, 0.0, 0.0], face_ball=True, is_goalkeeper=True)
        assert p["facing_deg"] == pytest.approx(0.0, abs=1e-6)
        assert p["raw_animation_class"] == AnimationClass.GK_SHUFFLE_R

    # 3. 非 GK L0 不再误触发 GK selector / facing
    def test_non_gk_l0_no_gk_behavior(self):
        cfg = self._cfg()
        t = PlayerMotionTracker(config=cfg)
        # L0 非 GK：即使传入球位置 + face_ball=False → 面向运动方向，rel≈0，无 shuffle
        p = t.update([0.0, 0.0, 0.0], [0.0, 1.0], 0.0,
                     ball_position_m=[10.0, 0.0, 0.0], face_ball=False, is_goalkeeper=False)
        assert p["facing_deg"] == pytest.approx(90.0, abs=1e-6)  # movement-facing
        assert p["raw_animation_class"] == AnimationClass.LOCOMOTION
        assert p["animation_class"] == AnimationClass.LOCOMOTION

    # 4. GK Facing 行为仍通过（is_goalkeeper=True 触发 ball-aware）
    def test_gk_facing_still_works_via_is_goalkeeper(self):
        cfg = self._cfg()
        t = PlayerMotionTracker(config=cfg)
        t.update([0.0, 0.0, 0.0], [0.0, 0.0], 0.0,
                 ball_position_m=[10.0, 0.0, 0.0], face_ball=True, is_goalkeeper=True)
        p = t.update([0.0, 0.02, 0.0], [0.0, 1.0], 0.1,
                     ball_position_m=[10.0, 0.0, 0.0], face_ball=True, is_goalkeeper=True)
        ball_heading = math.degrees(math.atan2(-0.02, 10.0))
        assert abs(shortest_angle_delta_deg(p["facing_deg"], ball_heading)) <= 1e-6

    # 5. Animation Selector GK 类仍通过（L1 GK 的 shuffle/backpedal）
    def test_selector_gk_class_via_is_goalkeeper(self):
        cfg = self._cfg()
        t = PlayerMotionTracker(config=cfg)
        p = t.update([0.0, 0.0, 0.0], [0.0, 1.0], 0.0,
                     ball_position_m=[10.0, 0.0, 0.0], face_ball=True, is_goalkeeper=True)
        assert p["raw_animation_class"] == AnimationClass.GK_SHUFFLE_R

    # 6. 回退：meta 缺失/无 GK 标记 → 用 GK_ENTITY_IDS
    def test_fallback_when_meta_missing(self):
        assert gk_entity_ids_from_meta(None) == frozenset({"L0", "R0"})
        assert gk_entity_ids_from_meta({}) == frozenset({"L0", "R0"})
        assert gk_entity_ids_from_meta({"entities": []}) == frozenset({"L0", "R0"})
        assert gk_entity_ids_from_meta({"entities": [
            {"id": "L0", "is_goalkeeper": False}]}) == frozenset({"L0", "R0"})

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
