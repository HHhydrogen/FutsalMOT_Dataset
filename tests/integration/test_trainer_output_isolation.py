"""Verify Trainer creates directories only where told."""

from __future__ import annotations

from pathlib import Path

import pytest

from futsalmot_rl.envs.defender_follow_env import FutsalDefenderFollowEnv
from futsalmot_rl.training.train_ppo import PPOTrainer


@pytest.mark.parametrize("total_steps", [1, 32, 250])
def test_trainer_creates_only_given_dirs(mini_a33: Path, tmp_path: Path, total_steps: int):
    """Trainer only creates log_dir and model_dir; no side effects on formal dirs."""
    source = str(mini_a33)
    env = FutsalDefenderFollowEnv(source_episode_path=source)

    log_dir = tmp_path / "logs"
    model_dir = tmp_path / "models"
    trainer = PPOTrainer(env, config={
        "total_timesteps": total_steps,
        "n_steps": 2048,
        "n_epochs": 1,
        "batch_size": 64,
    })

    # Track dirs that should NOT be created
    formal_models = Path("Saved/FutsalMOT_RL/models")
    formal_logs = Path("Saved/FutsalMOT_RL/train_logs/ppo")
    formal_existed_before = formal_models.is_dir() or formal_logs.is_dir()

    summary = trainer.train(
        total_timesteps=total_steps,
        log_dir=log_dir,
        model_dir=model_dir,
        run_name="test_isolation",
    )

    # Verify only given dirs created
    assert log_dir.is_dir(), "log_dir should exist"
    assert model_dir.is_dir(), "model_dir should exist"
    assert (log_dir / "test_isolation_log.jsonl").is_file(), "log file missing"
    assert (model_dir / "test_isolation_latest.pt").is_file(), "latest checkpoint missing"

    # Verify formal dirs not created by us
    if not formal_existed_before:
        assert not formal_models.is_dir(), "Trainer must not create formal model dir"
        assert not formal_logs.is_dir(), "Trainer must not create formal log dir"

    # best model only if episodes completed
    best_path = model_dir / "test_isolation_best.pt"
    if total_steps < 250:
        assert summary.get("best_mean_reward") is None
        assert not best_path.is_file(), "No best model for incomplete episode"
    else:
        # May or may not have completed an episode — just check no crash
        pass

    env.close()
