"""Unified evaluation for BC and PPO policies."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable

import numpy as np


@dataclass(frozen=True)
class EvaluationResult:
    """Immutable evaluation result for one policy run."""

    algorithm: str
    source_path: str
    model_path: str
    n_episodes: int
    seed: int
    device: str
    episode_rewards: tuple[float, ...] = ()
    episode_lengths: tuple[int, ...] = ()
    successes: tuple[bool | None, ...] = ()
    errors: tuple[str, ...] = ()

    def to_summary(self) -> dict[str, Any]:
        eps_rewards = list(self.episode_rewards)
        eps_lengths = list(self.episode_lengths)
        successes_list = list(self.successes)

        summary: dict[str, Any] = {
            "algorithm": self.algorithm,
            "source_path": self.source_path,
            "model_path": self.model_path,
            "n_episodes": self.n_episodes,
            "seed": self.seed,
            "device": self.device,
            "timestamp": __import__("datetime").datetime.now().isoformat(),
        }

        if eps_rewards:
            summary["mean_episode_reward"] = float(np.mean(eps_rewards))
            summary["std_episode_reward"] = float(np.std(eps_rewards))
            summary["min_episode_reward"] = float(np.min(eps_rewards))
            summary["max_episode_reward"] = float(np.max(eps_rewards))
            summary["mean_episode_length"] = float(np.mean(eps_lengths))
            summary["min_episode_length"] = int(np.min(eps_lengths))
            summary["max_episode_length"] = int(np.max(eps_lengths))
        else:
            summary["mean_episode_reward"] = None
            for k in ("std", "min", "max"):
                summary[f"{k}_episode_reward"] = None
            summary["mean_episode_length"] = None
            summary["min_episode_length"] = None
            summary["max_episode_length"] = None

        n_success = sum(1 for s in successes_list if s is True)
        n_fail = sum(1 for s in successes_list if s is False)
        summary["success_count"] = n_success if successes_list else None
        summary["failure_count"] = n_fail if successes_list else None

        return summary

    def to_jsonl_lines(self) -> list[dict[str, Any]]:
        lines: list[dict[str, Any]] = []
        for i in range(len(self.episode_rewards)):
            lines.append({
                "episode_index": i,
                "reward": float(self.episode_rewards[i]),
                "length": int(self.episode_lengths[i]),
                "success": self.successes[i] if i < len(self.successes) else None,
            })
        return lines


def evaluate_policy(
    env: Any,
    action_fn: Callable[[np.ndarray], np.ndarray],
    n_episodes: int = 5,
    seed: int = 42,
    algorithm: str = "unknown",
    source_path: str = "",
    model_path: str = "",
    device: str = "cpu",
) -> EvaluationResult:
    """Run multiple evaluation episodes using a common action function.

    Args:
        env: Gymnasium-like environment.
        action_fn: Callable taking (obs) → action array.
        n_episodes: Number of episodes to run.
        seed: Base RNG seed (incremented per episode).
        algorithm: Policy name for the result.
        source_path: Source episode path for provenance.
        model_path: Model path for provenance.
        device: Device string.

    Returns:
        EvaluationResult with per-episode metrics.
    """
    episode_rewards: list[float] = []
    episode_lengths: list[int] = []
    successes: list[bool | None] = []
    errors: list[str] = []

    for ep_idx in range(n_episodes):
        try:
            obs, info = env.reset(seed=seed + ep_idx)
            done = False
            ep_reward = 0.0
            ep_length = 0

            while not done:
                action = action_fn(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                ep_reward += float(reward)
                ep_length += 1
                done = terminated or truncated

            episode_rewards.append(ep_reward)
            episode_lengths.append(ep_length)
            successes.append(info.get("success", None))

        except Exception as exc:
            errors.append(f"episode {ep_idx}: {exc}")

    return EvaluationResult(
        algorithm=algorithm,
        source_path=source_path,
        model_path=model_path,
        n_episodes=n_episodes,
        seed=seed,
        device=device,
        episode_rewards=tuple(episode_rewards),
        episode_lengths=tuple(episode_lengths),
        successes=tuple(successes),
        errors=tuple(errors),
    )


def save_evaluation(
    result: EvaluationResult,
    output_dir: str | Path,
) -> Path:
    """Save evaluation summary JSON and per-episode JSONL.

    Returns:
        Path to the summary JSON.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = result.to_summary()
    summary_path = output_dir / "evaluation_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    episodes_path = output_dir / "episodes.jsonl"
    with open(episodes_path, "w", encoding="utf-8") as f:
        for line in result.to_jsonl_lines():
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

    return summary_path
