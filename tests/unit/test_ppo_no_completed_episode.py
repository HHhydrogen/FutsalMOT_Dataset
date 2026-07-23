"""Tests for PPO with fewer steps than a full episode — mean_ep_reward should be None."""

from __future__ import annotations

from pathlib import Path

import pytest

from futsalmot_rl.envs.defender_follow_env import FutsalDefenderFollowEnv
from futsalmot_rl.training.train_ppo import PPOTrainer


@pytest.fixture
def source(mini_a33: Path) -> str:
    return str(mini_a33)


def test_total_1_step(source: str):
    """total_timesteps=1 should complete without crash and have no completed episodes."""
    env = FutsalDefenderFollowEnv(source_episode_path=source)
    trainer = PPOTrainer(env, config={"total_timesteps": 1, "n_steps": 2048, "n_epochs": 1, "batch_size": 64})
    summary = trainer.train(total_timesteps=1, log_dir=source.rsplit("\\", 1)[0] + "/logs", model_dir=source.rsplit("\\", 1)[0] + "/models")
    assert summary.get("best_mean_reward") is None, "Should have no best reward with 0 completed episodes"
    assert summary.get("total_steps", 0) <= 1, "global_step should not exceed total_timesteps"
    env.close()


def test_total_32_steps(source: str):
    """total_timesteps=32 should not crash, no completed episodes."""
    env = FutsalDefenderFollowEnv(source_episode_path=source)
    trainer = PPOTrainer(env, config={"total_timesteps": 32, "n_steps": 2048, "n_epochs": 1, "batch_size": 64})
    summary = trainer.train(total_timesteps=32, log_dir=source.rsplit("\\", 1)[0] + "/logs", model_dir=source.rsplit("\\", 1)[0] + "/models")
    assert summary.get("best_mean_reward") is None
    assert summary.get("total_steps", 0) <= 32
    env.close()


def test_total_less_than_episode(source: str):
    """total_timesteps < 299 transitions should not crash, no completed episodes."""
    env = FutsalDefenderFollowEnv(source_episode_path=source)
    trainer = PPOTrainer(env, config={"total_timesteps": 250, "n_steps": 2048, "n_epochs": 1, "batch_size": 64})
    summary = trainer.train(total_timesteps=250, log_dir=source.rsplit("\\", 1)[0] + "/logs", model_dir=source.rsplit("\\", 1)[0] + "/models")
    assert summary.get("best_mean_reward") is None
    env.close()
