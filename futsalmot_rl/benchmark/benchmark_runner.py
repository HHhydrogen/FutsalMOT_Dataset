"""Run policy evaluation across multiple episodes and collect metrics."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from futsalmot_rl.benchmark.metrics import compute_policy_metrics
from futsalmot_rl.data.a33_reader import (
    get_ball_positions_2d,
    get_player_positions_2d,
    load_a33_config,
)
from futsalmot_rl.envs.defender_follow_env import FutsalDefenderFollowEnv


def benchmark_policy(
    policy_fn: Callable,
    source_paths: list[Path],
    policy_name: str,
    n_episodes: int = 5,
) -> list[dict[str, Any]]:
    """Run a policy on multiple episodes and collect metrics.

    Args:
        policy_fn: Callable taking (obs, deterministic) -> action.
        source_paths: List of A3.3 source file paths.
        policy_name: Name for this policy.
        n_episodes: Max episodes to evaluate.

    Returns:
        List of per-episode metric dicts.
    """
    results: list[dict[str, Any]] = []
    n = min(n_episodes, len(source_paths))

    for i in range(n):
        source = source_paths[i]
        cfg = load_a33_config(source)
        all_pos = get_player_positions_2d(cfg)
        ball_pos = np.array(get_ball_positions_2d(cfg), dtype=np.float32)
        target_pos = np.array(all_pos.get("Player_01", []), dtype=np.float32)
        rule_pos = np.array(all_pos.get("Player_05", []), dtype=np.float32)

        # Other player positions for collision detection
        other_pos = {}
        for pid, pos in all_pos.items():
            if pid != "Player_05":
                other_pos[pid] = np.array(pos, dtype=np.float32)

        # Rollout policy
        env = FutsalDefenderFollowEnv(source_episode_path=str(source))
        collected: list[tuple[float, float]] = []
        obs, info = env.reset()

        # Capture initial position
        init_pos = info.get("all_positions", {}).get("Player_05", (0, 0))
        collected.append((float(init_pos[0]), float(init_pos[1])))

        done = False
        while not done:
            action = policy_fn(obs)
            obs_next, _, terminated, truncated, info = env.step(action)
            pos = info.get("all_positions", {}).get("Player_05", (0, 0))
            collected.append((float(pos[0]), float(pos[1])))
            done = terminated or truncated
            obs = obs_next
        env.close()

        agent_positions = np.array(collected, dtype=np.float32)

        metrics = compute_policy_metrics(
            positions=agent_positions,
            target_positions=target_pos,
            ball_positions=ball_pos,
            all_player_positions=other_pos,
            agent_id="Player_05",
        )
        metrics["seq_id"] = cfg.get("seq_id", cfg.get("episode_id", "unknown"))
        metrics["template_id"] = _extract_template(metrics["seq_id"])
        metrics["seed"] = _extract_seed(metrics["seq_id"])
        metrics["policy_type"] = policy_name

        results.append(metrics)
        print("  [{}] episode {}: mark_dist={:.1f}cm oob={} coll={}".format(
            policy_name, i + 1,
            metrics["mean_marking_distance_cm"],
            metrics["out_of_bounds_count"],
            metrics["collision_count"],
        ))

    return results


def benchmark_rule(source_paths: list[Path], n_episodes: int = 5) -> list[dict[str, Any]]:
    """Benchmark rule baseline (read from source A3.3 directly)."""
    results: list[dict[str, Any]] = []
    n = min(n_episodes, len(source_paths))

    for i in range(n):
        source = source_paths[i]
        cfg = load_a33_config(source)
        all_pos = get_player_positions_2d(cfg)
        ball_pos = np.array(get_ball_positions_2d(cfg), dtype=np.float32)
        target_pos = np.array(all_pos.get("Player_01", []), dtype=np.float32)
        rule_pos = np.array(all_pos.get("Player_05", []), dtype=np.float32)

        other_pos = {}
        for pid, pos in all_pos.items():
            if pid != "Player_05":
                other_pos[pid] = np.array(pos, dtype=np.float32)

        metrics = compute_policy_metrics(
            positions=rule_pos,
            target_positions=target_pos,
            ball_positions=ball_pos,
            all_player_positions=other_pos,
            agent_id="Player_05",
        )
        metrics["seq_id"] = cfg.get("seq_id", cfg.get("episode_id", "unknown"))
        metrics["template_id"] = _extract_template(metrics["seq_id"])
        metrics["seed"] = _extract_seed(metrics["seq_id"])
        metrics["policy_type"] = "rule"

        results.append(metrics)
        print("  [rule] episode {}: mark_dist={:.1f}cm oob={} coll={}".format(
            i + 1,
            metrics["mean_marking_distance_cm"],
            metrics["out_of_bounds_count"],
            metrics["collision_count"],
        ))

    return results


def _extract_template(seq_id: str) -> int:
    """Extract template_id from seq_id like 'episode_random_0001_t1'."""
    import re
    m = re.search(r"_t(\d+)", seq_id)
    return int(m.group(1)) if m else 0


def _extract_seed(seq_id: str) -> int:
    """Extract seed from seq_id like 'episode_random_0001_t1'."""
    import re
    m = re.search(r"_(\d{4})_t", seq_id)
    return int(m.group(1)) if m else 0
