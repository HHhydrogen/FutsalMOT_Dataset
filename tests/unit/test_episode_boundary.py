"""Tests for frame/transition semantics and episode boundaries."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from futsalmot_rl.envs.defender_follow_env import FutsalDefenderFollowEnv


@pytest.fixture
def env(mini_a33: Path) -> FutsalDefenderFollowEnv:
    return FutsalDefenderFollowEnv(source_episode_path=str(mini_a33))


def test_reset(env: FutsalDefenderFollowEnv):
    """Reset returns valid obs at frame 0."""
    obs, info = env.reset()
    assert env.current_frame == 0
    assert np.all(np.isfinite(obs))
    assert obs.dtype == np.float32
    # steps_left should be at or near max
    assert obs[30] > 0.99  # steps_left_norm ≈ 1.0 at start


def test_first_step(env: FutsalDefenderFollowEnv):
    """After first step, frame advances."""
    env.reset()
    obs, _, _, _, _ = env.step(np.array([0.0, 0.0], dtype=np.float32))
    assert env.current_frame == 1
    assert np.all(np.isfinite(obs))


def test_penultimate_step(env: FutsalDefenderFollowEnv):
    """Frame 298 should have steps_left ≈ 0."""
    env.reset()
    done = False
    while env.current_frame < 298 and not done:
        _, _, term, trunc, _ = env.step(np.array([0.0, 0.0], dtype=np.float32))
        done = term or trunc
    obs, _, _, _, _ = env.step(np.array([0.0, 0.0], dtype=np.float32))
    assert env.current_frame == 299
    assert obs[30] < 0.01, f"steps_left should be ~0, got {obs[30]}"


def test_terminal_step(env: FutsalDefenderFollowEnv):
    """Last action terminates the episode."""
    env.reset()
    for _ in range(298):
        _, _, term, trunc, _ = env.step(np.array([0.0, 0.0], dtype=np.float32))
        if term or trunc:
            break
    obs, reward, terminated, truncated, info = env.step(np.array([0.0, 0.0], dtype=np.float32))
    assert terminated, "Last step should terminate episode"
    assert np.all(np.isfinite(obs))


def test_no_out_of_bounds_after_terminal(env: FutsalDefenderFollowEnv):
    """After terminal, reset works fine."""
    env.reset()
    done = False
    while not done:
        _, _, term, trunc, _ = env.step(np.array([0.0, 0.0], dtype=np.float32))
        done = term or trunc
    obs, _ = env.reset()
    assert env.current_frame == 0
    assert np.all(np.isfinite(obs))


def test_steps_left_monotonic(env: FutsalDefenderFollowEnv):
    """steps_left never increases during an episode."""
    env.reset()
    prev_steps = env.total_frames
    done = False
    while not done:
        obs, _, term, trunc, _ = env.step(np.array([0.0, 0.0], dtype=np.float32))
        done = term or trunc
        steps = int(round(obs[30] * env.total_frames))
        assert steps <= prev_steps, f"steps_left increased from {prev_steps} to {steps}"
        prev_steps = steps
