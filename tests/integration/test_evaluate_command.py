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
    assert summary["completed_episode_count"] == 3
    assert summary["requested_episode_count"] == 3
    assert summary["failed_episode_count"] == 0
    assert summary["mean_episode_reward"] is not None
    assert summary["std_episode_reward"] is not None
    env.close()


def test_n_episodes_positive():
    """n_episodes must be >= 1."""
    from futsalmot_rl.commands.evaluate import positive_int
    assert positive_int("5") == 5
    with pytest.raises(Exception):
        positive_int("0")
    with pytest.raises(Exception):
        positive_int("-1")


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
        source=__file__,
        model="/nonexistent/model.pt",
        output_dir="/tmp",
        n_episodes=3,
        device="cpu",
        seed=42,
    )
    rc = eval_run(args, "")
    assert rc != 0, "Missing model should return non-zero"


def test_success_count_none():
    """When no success info, summary success_count is None."""
    from futsalmot_rl.evaluation.evaluator import EvaluationResult, EpisodeEvaluation

    episodes = tuple(
        EpisodeEvaluation(episode_index=i, seed=42 + i, status="completed",
                          reward=float(i + 1), length=10, success=None)
        for i in range(3)
    )
    r = EvaluationResult(
        algorithm="test", source_path="", model_path="",
        n_episodes=3, seed=42, device="cpu",
        episodes=episodes,
    )
    summary = r.to_summary()
    assert summary["success_count"] is None
    assert summary["failure_count"] is None


def test_success_count_mixed():
    """Mixed True/False/None: only known values counted."""
    from futsalmot_rl.evaluation.evaluator import EvaluationResult, EpisodeEvaluation

    episodes = (
        EpisodeEvaluation(0, 42, "completed", reward=1.0, length=10, success=True),
        EpisodeEvaluation(1, 43, "completed", reward=2.0, length=10, success=False),
        EpisodeEvaluation(2, 44, "completed", reward=3.0, length=10, success=True),
        EpisodeEvaluation(3, 45, "completed", reward=4.0, length=10, success=None),
    )
    r = EvaluationResult(
        algorithm="test", source_path="", model_path="",
        n_episodes=4, seed=42, device="cpu", episodes=episodes,
    )
    summary = r.to_summary()
    assert summary["success_count"] == 2
    assert summary["failure_count"] == 1


def test_summary_has_errors():
    """Summary includes error details."""
    from futsalmot_rl.evaluation.evaluator import EvaluationResult, EpisodeEvaluation

    episodes = (
        EpisodeEvaluation(0, 42, "error", error_type="ValueError", error_message="test"),
    )
    r = EvaluationResult(
        algorithm="test", source_path="", model_path="",
        n_episodes=1, seed=42, device="cpu", episodes=episodes,
    )
    summary = r.to_summary()
    assert summary["failed_episode_count"] == 1
    assert len(summary["errors"]) == 1
    assert summary["errors"][0]["message"] == "test"
