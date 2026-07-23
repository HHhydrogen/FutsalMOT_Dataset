"""Compare rule, BC, and RL policies on the same evaluation episodes."""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

from futsalmot_rl.core.rl_io import write_json_atomic
from futsalmot_rl.core.rl_paths import (
    REPORTS_DIR,
    ensure_dirs,
)
from futsalmot_rl.data.a33_reader import get_player_positions_2d, load_a33_config
from futsalmot_rl.envs.defender_follow_env import FutsalDefenderFollowEnv
from futsalmot_rl.features.action_builder import (
    PLAYER_05_MAX_SPEED_CM_S,
)


def _compute_trajectory_metrics(
    positions: list[tuple[float, float]],
    other_positions: dict[str, list[tuple[float, float]]],
    court_bounds: tuple[float, float, float, float],
    collision_dist: float = 50.0,
) -> dict[str, Any]:
    """Compute trajectory quality metrics for a position sequence."""
    out_of_bounds = 0
    collisions = 0
    min_player_dist = float("inf")
    total_dist = 0.0
    max_speed = 0.0
    speed_warnings = 0
    turn_warnings = 0

    x_min, x_max, y_min, y_max = court_bounds
    margin = 10.0

    for i in range(len(positions)):
        x, y = positions[i]

        # Out of bounds
        if not (x_min + margin <= x <= x_max - margin and y_min + margin <= y <= y_max - margin):
            out_of_bounds += 1

        # Collisions
        for pid, other_pos in other_positions.items():
            if i < len(other_pos):
                ox, oy = other_pos[i]
                dist = math.hypot(x - ox, y - oy)
                if dist < collision_dist:
                    collisions += 1
                if dist < min_player_dist:
                    min_player_dist = dist

        # Distance / speed
        if i > 0:
            dx = x - positions[i - 1][0]
            dy = y - positions[i - 1][1]
            seg_dist = math.hypot(dx, dy)
            total_dist += seg_dist
            speed = seg_dist * 30  # 30 FPS
            if speed > PLAYER_05_MAX_SPEED_CM_S:
                speed_warnings += 1
            if speed > max_speed:
                max_speed = speed

        # Turn angle
        if i > 1:
            v1 = (
                positions[i - 1][0] - positions[i - 2][0],
                positions[i - 1][1] - positions[i - 2][1],
            )
            v2 = (x - positions[i - 1][0], y - positions[i - 1][1])
            dot = v1[0] * v2[0] + v1[1] * v2[1]
            cross = v1[0] * v2[1] - v1[1] * v2[0]
            angle = abs(math.degrees(math.atan2(cross, dot)))
            if angle > 100:
                turn_warnings += 1

    return {
        "out_of_bounds_count": out_of_bounds,
        "collision_count": collisions,
        "minimum_player_distance_cm": float(min_player_dist)
        if min_player_dist != float("inf")
        else None,
        "total_distance_cm": total_dist,
        "max_speed_cm_s": max_speed,
        "speed_warning_count": speed_warnings,
        "turn_warning_count": turn_warnings,
    }


def compare_policies(
    source_a33_path: str | Path,
    bc_policy: Callable | None = None,
    rl_policy: Callable | None = None,
    n_episodes: int = 3,
) -> dict[str, Any]:
    """Compare rule, BC, and RL policies on the same source episodes.

    Args:
        source_a33_path: Path to the source (rule) A3.3 config.
        bc_policy: BC policy callable. If None, compare only rule vs RL.
        rl_policy: RL policy callable. If None, compare only rule vs BC.
        n_episodes: Number of evaluation episodes.

    Returns:
        Dict with per-policy metrics.
    """

    ensure_dirs()
    results: dict[str, Any] = {}
    cfg = load_a33_config(source_a33_path)
    source_all_pos = get_player_positions_2d(cfg)
    court_bounds = (-1950.0, 1950.0, -950.0, 950.0)

    # ── Rule baseline ──────────────────────────────────────────
    rule_positions = source_all_pos.get("Player_05", [])
    other_positions = {pid: pos for pid, pos in source_all_pos.items() if pid != "Player_05"}
    rule_metrics = _compute_trajectory_metrics(
        [(float(x), float(y)) for x, y in rule_positions],
        other_positions,
        court_bounds,
    )
    rule_metrics["method"] = "rule"
    results["rule"] = rule_metrics

    # ── BC policy ──────────────────────────────────────────────
    if bc_policy is not None:
        env = FutsalDefenderFollowEnv(source_episode_path=source_a33_path)
        bc_positions: list[tuple[float, float]] = []
        for ep in range(n_episodes):
            obs, _ = env.reset()
            done = False
            while not done:
                action = bc_policy(obs)
                obs_next, reward, term, trunc, info = env.step(action)
                bc_positions.append(info.get("all_positions", {}).get("Player_05", (0.0, 0.0)))
                done = term or trunc
                obs = obs_next

        bc_metrics = _compute_trajectory_metrics(bc_positions, other_positions, court_bounds)
        bc_metrics["method"] = "bc"
        results["bc"] = bc_metrics
        env.close()

    # ── RL policy ──────────────────────────────────────────────
    if rl_policy is not None:
        env = FutsalDefenderFollowEnv(source_episode_path=source_a33_path)
        rl_positions: list[tuple[float, float]] = []
        for ep in range(n_episodes):
            obs, _ = env.reset()
            done = False
            while not done:
                action = rl_policy(obs)
                obs_next, reward, term, trunc, info = env.step(action)
                rl_positions.append(info.get("all_positions", {}).get("Player_05", (0.0, 0.0)))
                done = term or trunc
                obs = obs_next

        rl_metrics = _compute_trajectory_metrics(rl_positions, other_positions, court_bounds)
        rl_metrics["method"] = "rl"
        results["rl"] = rl_metrics
        env.close()

    return results


def save_comparison_report(
    results: dict[str, Any],
    seq_id: str = "comparison",
) -> dict[str, Any]:
    """Save comparison report as JSON and CSV."""
    report_path = REPORTS_DIR / f"compare_rule_bc_rl_{seq_id}.json"
    write_json_atomic(report_path, results)

    # CSV summary
    csv_path = REPORTS_DIR / "compare_rule_bc_rl_summary.csv"
    import csv

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        # Collect all metric keys
        all_keys = set()
        for policy_data in results.values():
            if isinstance(policy_data, dict):
                all_keys.update(policy_data.keys())
        all_keys.discard("method")

        fieldnames = ["method"] + sorted(all_keys)
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for method_name, policy_data in results.items():
            if isinstance(policy_data, dict):
                row = {"method": method_name}
                row.update(policy_data)
                writer.writerow(row)

    return results
