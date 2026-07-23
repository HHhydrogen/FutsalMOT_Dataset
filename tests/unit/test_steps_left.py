"""Tests for steps_left: reset=1.0, terminal=0.0."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from futsalmot_rl.envs.defender_follow_env import FutsalDefenderFollowEnv


@pytest.fixture
def env(mini_a33: Path) -> FutsalDefenderFollowEnv:
    return FutsalDefenderFollowEnv(source_episode_path=str(mini_a33))


def test_steps_left_at_reset(env: FutsalDefenderFollowEnv):
    """Reset → steps_left ≈ 1.0 (299/299 transitions remaining)."""
    obs, _ = env.reset()
    # obs[30] = steps_left_norm
    assert abs(obs[30] - 1.0) < 0.01, f"Expected ~1.0, got {obs[30]}"


def test_steps_left_terminal_exactly_zero(env: FutsalDefenderFollowEnv):
    """Terminal observation → steps_left == 0.0 (strict)."""
    env.reset()
    done = False
    while not done:
        obs, _, term, trunc, _ = env.step(np.array([0.0, 0.0], dtype=np.float32))
        done = term or trunc
    assert obs[30] == 0.0, f"Expected 0.0 at terminal, got {obs[30]}"


def test_steps_left_monotonic(env: FutsalDefenderFollowEnv):
    """steps_left never increases during an episode."""
    env.reset()
    prev = 1.0
    done = False
    while not done:
        obs, _, term, trunc, _ = env.step(np.array([0.0, 0.0], dtype=np.float32))
        done = term or trunc
        sl = obs[30]
        assert sl <= prev + 0.001, f"steps_left increased: {prev} → {sl}"
        prev = sl


def test_steps_left_mid_episode(env: FutsalDefenderFollowEnv):
    """Mid-episode steps_left is in (0, 1)."""
    env.reset()
    for _ in range(150):
        obs, _, term, trunc, _ = env.step(np.array([0.5, 0.0], dtype=np.float32))
        if term or trunc:
            break
    sl = obs[30]
    assert 0.4 < sl < 0.6, f"Expected ~0.5 mid-episode, got {sl}"
