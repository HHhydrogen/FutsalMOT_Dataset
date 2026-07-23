"""Tests for the unified evaluate command."""

from __future__ import annotations

from pathlib import Path

import pytest

from futsalmot_rl.evaluation.evaluator import evaluate_policy, save_evaluation


def test_n_episodes_parameter(mini_a33: Path, tmp_path: Path):
    """n_episodes=3 must produce exactly 3 episode results."""
    from futsalmot_rl.envs.defender_follow_env import FutsalDefenderFollowEnv
    env = FutsalDefenderFollowEnv(source_episode_path=str(mini_a33))

    def dummy_action(obs):
        return env.action_space.sample()

    result = evaluate_policy(env, dummy_action, n_episodes=3, seed=42, algorithm="test")
    assert len(result.episode_rewards) == 3
    assert len(result.episode_lengths) == 3

    save_evaluation(result, tmp_path)
    assert (tmp_path / "evaluation_summary.json").is_file()
    assert (tmp_path / "episodes.jsonl").is_file()

    summary = result.to_summary()
    assert summary["n_episodes"] == 3
    assert summary["mean_episode_reward"] is not None
    assert summary["std_episode_reward"] is not None
    env.close()


def test_missing_source_returns_error():
    """Non-existent source file should cause evaluate to fail."""
    from futsalmot_rl.commands.evaluate import run as eval_run
    import argparse
    args = argparse.Namespace(
        eval_cmd="bc",
        source="/nonexistent/file.json",
        model="/nonexistent/model.pt",
        output_dir="/tmp",
        n_episodes=3,
        device="cpu",
        seed=42,
    )
    rc = eval_run(args, "")
    assert rc != 0, "Missing source should return non-zero"


def test_missing_model_returns_error():
    """Non-existent model file should cause evaluate to fail."""
    from futsalmot_rl.commands.evaluate import run as eval_run
    import argparse
    args = argparse.Namespace(
        eval_cmd="ppo",
        source=__file__,  # exists but is not a valid model
        model="/nonexistent/model.pt",
        output_dir="/tmp",
        n_episodes=3,
        device="cpu",
        seed=42,
    )
    rc = eval_run(args, "")
    assert rc != 0, "Missing model should return non-zero"
