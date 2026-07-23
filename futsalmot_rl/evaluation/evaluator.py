"""Unified evaluation for BC and PPO policies."""

from __future__ import annotations

import json
import traceback
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable

import numpy as np


@dataclass(frozen=True)
class EpisodeEvaluation:
    """One episode's evaluation result, preserving original index."""

    episode_index: int
    seed: int
    status: str  # "completed" or "error"
    reward: float | None = None
    length: int | None = None
    success: bool | None = None
    terminated: bool | None = None
    truncated: bool | None = None
    error_type: str | None = None
    error_message: str | None = None
    traceback: str | None = None


@dataclass(frozen=True)
class EvaluationResult:
    """Immutable evaluation result with per-episode records."""

    algorithm: str
    source_path: str
    model_path: str
    n_episodes: int
    seed: int
    device: str
    episodes: tuple[EpisodeEvaluation, ...] = ()

    @property
    def episode_rewards(self) -> tuple[float, ...]:
        return tuple(e.reward for e in self.episodes if e.status == "completed" and e.reward is not None)

    @property
    def episode_lengths(self) -> tuple[int, ...]:
        return tuple(e.length for e in self.episodes if e.status == "completed" and e.length is not None)

    @property
    def successes(self) -> tuple[bool | None, ...]:
        return tuple(e.success for e in self.episodes if e.status == "completed")

    @property
    def errors(self) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for e in self.episodes:
            if e.status == "error":
                result.append({
                    "episode_index": e.episode_index,
                    "error_type": e.error_type,
                    "message": e.error_message,
                    "traceback": e.traceback,
                })
        return result

    def to_summary(self) -> dict[str, Any]:
        completed = [e for e in self.episodes if e.status == "completed"]
        failed = [e for e in self.episodes if e.status == "error"]
        eps_rewards = [e.reward for e in completed if e.reward is not None]
        eps_lengths = [e.length for e in completed if e.length is not None]
        successes_list = [e.success for e in completed]

        summary: dict[str, Any] = {
            "algorithm": self.algorithm,
            "source_path": self.source_path,
            "model_path": self.model_path,
            "n_episodes": self.n_episodes,
            "requested_episode_count": len(self.episodes),
            "completed_episode_count": len(completed),
            "failed_episode_count": len(failed),
            "seed": self.seed,
            "device": self.device,
            "timestamp": __import__("datetime").datetime.now().isoformat(),
            "errors": self.errors,
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
            for k in ("mean", "std", "min", "max"):
                summary[f"{k}_episode_reward"] = None
            summary["mean_episode_length"] = None
            summary["min_episode_length"] = None
            summary["max_episode_length"] = None

        known = [s for s in successes_list if s is not None]
        if known:
            summary["success_count"] = sum(1 for s in known if s is True)
            summary["failure_count"] = sum(1 for s in known if s is False)
        else:
            summary["success_count"] = None
            summary["failure_count"] = None

        return summary

    def to_jsonl_lines(self) -> list[dict[str, Any]]:
        lines: list[dict[str, Any]] = []
        for e in self.episodes:
            d: dict[str, Any] = {
                "episode_index": e.episode_index,
                "seed": e.seed,
                "status": e.status,
            }
            if e.status == "completed":
                d["reward"] = e.reward
                d["length"] = e.length
                d["success"] = e.success
                d["terminated"] = e.terminated
                d["truncated"] = e.truncated
            else:
                d["error_type"] = e.error_type
                d["error_message"] = e.error_message
            lines.append(d)
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
    """Run multiple evaluation episodes.

    Each episode is recorded at its original index regardless of success/failure.
    """
    if n_episodes < 1:
        raise ValueError("n_episodes must be at least 1")

    records: list[EpisodeEvaluation] = []

    for ep_idx in range(n_episodes):
        try:
            obs, info = env.reset(seed=seed + ep_idx)
            done = False
            ep_reward = 0.0
            ep_length = 0
            terminated = False
            truncated = False

            while not done:
                action = action_fn(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                ep_reward += float(reward)
                ep_length += 1
                done = terminated or truncated

            records.append(EpisodeEvaluation(
                episode_index=ep_idx,
                seed=seed + ep_idx,
                status="completed",
                reward=ep_reward,
                length=ep_length,
                success=info.get("success"),
                terminated=bool(terminated),
                truncated=bool(truncated),
            ))

        except Exception as exc:
            records.append(EpisodeEvaluation(
                episode_index=ep_idx,
                seed=seed + ep_idx,
                status="error",
                error_type=type(exc).__name__,
                error_message=str(exc),
                traceback=traceback.format_exc(),
            ))

    return EvaluationResult(
        algorithm=algorithm,
        source_path=source_path,
        model_path=model_path,
        n_episodes=n_episodes,
        seed=seed,
        device=device,
        episodes=tuple(records),
    )


def save_evaluation(
    result: EvaluationResult,
    output_dir: str | Path,
) -> Path:
    """Save evaluation summary JSON and per-episode JSONL."""
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
