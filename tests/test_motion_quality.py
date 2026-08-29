"""Motion Quality Audit 单元测试（C6-P1.5）。"""
import json
import math

import pytest

from grf_ue_bridge.motion_quality import analyze_frames, find_active_window


def _mk_frames(n_frames, speed, dt=0.1, outfield_ids=("L1", "L2", "L3", "L4", "R1", "R2", "R3", "R4"),
               gk_ids=("L0", "R0")):
    """构造 n_frames 帧，每帧每个球员以 speed(m/s) 沿 x 前进。"""
    frames = []
    for k in range(n_frames):
        players = []
        for i, pid in enumerate(outfield_ids + gk_ids):
            players.append({
                "id": pid,
                "position_m": [k * speed * dt, i, 0.0],
                "role": "goalkeeper" if pid in gk_ids else "midfielder",
                "is_goalkeeper": pid in gk_ids,
            })
        frames.append({"players": players})
    return frames


def test_analyze_frames_high_motion():
    frames = _mk_frames(600, speed=2.0)
    m = analyze_frames(frames, dt_s=0.1, gk_ids=["L0", "R0"])
    # 全部球员持续 2 m/s 移动（首帧无速度，用容差）
    for pid, pm in m["players"].items():
        assert pm["active_ratio"] >= 0.99
        assert pm["longest_stationary_streak_s"] <= 0.1
    assert m["team_active_outfield_coverage"] >= 0.99
    assert m["longest_global_low_motion_plateau_s"] <= 0.1


def test_analyze_frames_stationary():
    # 一半时间静止，一半时间移动
    frames = _mk_frames(300, speed=0.0)
    m = analyze_frames(frames, dt_s=0.1, gk_ids=["L0", "R0"])
    for pid, pm in m["players"].items():
        assert pm["active_ratio"] == 0.0
        assert pm["longest_stationary_streak_s"] >= 29.0  # 300 帧全静止 = 29.9s


def test_find_active_window_high_motion():
    frames = _mk_frames(900, speed=1.5)  # 90s, 全程活跃
    win = find_active_window(frames, 60.0, dt_s=0.1, gk_ids=["L0", "R0"])
    assert win is not None
    start, end = win
    assert (end - start) >= 600  # >= 60s


def test_find_active_window_mixed():
    # 前 30s 活跃，后 60s 静止
    frames = _mk_frames(300, speed=1.5) + _mk_frames(600, speed=0.0)
    # 活跃 window 应落在前 300 帧
    win = find_active_window(frames, 30.0, dt_s=0.1, gk_ids=["L0", "R0"])
    assert win is not None
    start, _ = win
    assert start < 300
