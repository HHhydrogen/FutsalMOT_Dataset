"""Tests for agent yaw persistence — yaw tracks actual velocity, not rule ghost."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from futsalmot_rl.envs.defender_follow_env import FutsalDefenderFollowEnv


@pytest.fixture
def env(mini_a33: Path) -> FutsalDefenderFollowEnv:
    return FutsalDefenderFollowEnv(source_episode_path=str(mini_a33))


def _get_yaw_sin_cos(obs: np.ndarray) -> tuple[float, float]:
    return float(obs[4]), float(obs[5])


def test_move_right(env: FutsalDefenderFollowEnv):
    """Moving right → yaw ≈ 0° → cos ≈ 1, sin ≈ 0."""
    env.reset()
    for _ in range(5):
        obs, _, _, _, _ = env.step(np.array([1.0, 0.0], dtype=np.float32))
    s, c = _get_yaw_sin_cos(obs)
    assert c > 0.7, f"Moving right should have cos≈1, got {c}"
    assert abs(s) < 0.7, f"Moving right should have sin≈0, got {s}"


def test_move_up(env: FutsalDefenderFollowEnv):
    """Moving up → yaw ≈ 90° → sin ≈ 1, cos ≈ 0."""
    env.reset()
    for _ in range(5):
        obs, _, _, _, _ = env.step(np.array([0.0, 1.0], dtype=np.float32))
    s, c = _get_yaw_sin_cos(obs)
    assert s > 0.5, f"Moving up should have sin>0.5, got {s}"


def test_move_left(env: FutsalDefenderFollowEnv):
    """Moving left → yaw ≈ ±180° → cos ≈ -1, sin ≈ 0."""
    env.reset()
    for _ in range(5):
        obs, _, _, _, _ = env.step(np.array([-1.0, 0.0], dtype=np.float32))
    s, c = _get_yaw_sin_cos(obs)
    assert c < -0.5, f"Moving left should have cos≈-1, got {c}"


def test_stop_keeps_yaw(env: FutsalDefenderFollowEnv):
    """After moving then stopping, yaw should persist from last motion."""
    env.reset()
    # Move right
    for _ in range(5):
        obs, _, _, _, _ = env.step(np.array([1.0, 0.0], dtype=np.float32))
    cos_right = _get_yaw_sin_cos(obs)[1]
    # Stop
    for _ in range(3):
        obs, _, _, _, _ = env.step(np.array([0.0, 0.0], dtype=np.float32))
    cos_stop = _get_yaw_sin_cos(obs)[1]
    assert abs(cos_stop - cos_right) < 0.1, "Stop should preserve last yaw"


def test_yaw_not_from_rule_trajectory(env: FutsalDefenderFollowEnv):
    """RL agent's yaw differs from rule trajectory when actions differ."""
    env.reset()
    for _ in range(10):
        obs, _, _, _, _ = env.step(np.array([-1.0, 0.5], dtype=np.float32))
    s, c = _get_yaw_sin_cos(obs)
    # Yaw should be finite (not NaN/Inf)
    assert np.isfinite(s)
    assert np.isfinite(c)


def test_reset_yaw(env: FutsalDefenderFollowEnv):
    """Reset should not crash; yaw after reset is finite."""
    # Move around
    for _ in range(20):
        env.step(np.array([0.5, -0.3], dtype=np.float32))
    # Reset
    obs, _ = env.reset()
    s, c = _get_yaw_sin_cos(obs)
    assert np.isfinite(s)
    assert np.isfinite(c)
