"""Core metrics computation for FutsalMOT-RL policy evaluation."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from futsalmot_rl.core.rl_paths import (
    COURT_X_MAX,
    COURT_X_MIN,
    COURT_Y_MAX,
    COURT_Y_MIN,
    FPS,
)
from futsalmot_rl.features.action_builder import PLAYER_05_MAX_SPEED_CM_S


def compute_policy_metrics(
    positions: np.ndarray,
    target_positions: np.ndarray,
    ball_positions: np.ndarray | None = None,
    all_player_positions: dict[str, np.ndarray] | None = None,
    agent_id: str = "Player_05",
    own_goal_pos: tuple[float, float] = (COURT_X_MIN, 0.0),
    ideal_mark_distance: float = 180.0,
    collision_distance: float = 50.0,
    fps: int = FPS,
) -> dict[str, Any]:
    """Compute comprehensive metrics for a policy's trajectory.

    Args:
        positions: (T, 2) array of agent positions.
        target_positions: (T, 2) array of target (Player_01) positions.
        ball_positions: Optional (T, 2) array of ball positions.
        all_player_positions: {player_id: (T, 2) array} for collision detection.
        agent_id: Which player is the agent.
        own_goal_pos: (x, y) of own goal center.
        ideal_mark_distance: Ideal distance to maintain from target.
        collision_distance: Minimum allowed distance to other players.
        fps: Frames per second.

    Returns:
        Dict of metrics.
    """
    T = len(positions)
    n_target = len(target_positions)
    n = min(T, n_target)

    # ── Distance to target ────────────────────────────────────
    distances = np.sqrt(
        (positions[:n, 0] - target_positions[:n, 0]) ** 2
        + (positions[:n, 1] - target_positions[:n, 1]) ** 2
    )
    mean_marking_distance = float(np.mean(distances))
    std_marking_distance = float(np.std(distances))

    # ── Goal-side success rate ─────────────────────────────────
    goal_side_count = 0
    for i in range(n):
        px, py = positions[i]
        tx, ty = target_positions[i]
        gx, gy = own_goal_pos
        dist_to_goal = math.hypot(px - gx, py - gy)
        dist_target_to_goal = math.hypot(tx - gx, ty - gy)
        vec_to_target = (tx - px, ty - py)
        vec_to_goal = (gx - px, gy - py)
        dot = vec_to_target[0] * vec_to_goal[0] + vec_to_target[1] * vec_to_goal[1]
        if dot > 0 and dist_to_goal < dist_target_to_goal:
            goal_side_count += 1
    goal_side_success_rate = goal_side_count / max(1, n)

    # ── Out of bounds ──────────────────────────────────────────
    oob_count = 0
    margin = 10.0
    for i in range(T):
        x, y = positions[i]
        if not (COURT_X_MIN + margin <= x <= COURT_X_MAX - margin
                and COURT_Y_MIN + margin <= y <= COURT_Y_MAX - margin):
            oob_count += 1

    # ── Collisions ─────────────────────────────────────────────
    collision_count = 0
    if all_player_positions:
        for pid, other_pos in all_player_positions.items():
            if pid == agent_id:
                continue
            m = min(T, len(other_pos))
            for i in range(m):
                dist = math.hypot(
                    positions[i, 0] - other_pos[i, 0],
                    positions[i, 1] - other_pos[i, 1],
                )
                if dist < collision_distance:
                    collision_count += 1

    # ── Speed ──────────────────────────────────────────────────
    speeds = np.zeros(T)
    for i in range(1, T):
        dx = positions[i, 0] - positions[i - 1, 0]
        dy = positions[i, 1] - positions[i - 1, 1]
        speeds[i] = math.hypot(dx, dy) * fps
    max_speed = float(np.max(speeds))
    mean_speed = float(np.mean(speeds[1:]))
    speed_warning_count = int(np.sum(speeds > PLAYER_05_MAX_SPEED_CM_S))

    # ── Turn angles ────────────────────────────────────────────
    turn_angles = []
    for i in range(2, T):
        v1 = positions[i - 1] - positions[i - 2]
        v2 = positions[i] - positions[i - 1]
        dot = float(v1[0] * v2[0] + v1[1] * v2[1])
        cross = float(v1[0] * v2[1] - v1[1] * v2[0])
        angle = abs(math.degrees(math.atan2(cross, dot)))
        turn_angles.append(angle)
    turn_warning_count = sum(1 for a in turn_angles if a > 100)

    # ── Total distance ─────────────────────────────────────────
    total_dist = float(np.sum(speeds[1:] / fps))

    # ── Time behind attacker ratio ─────────────────────────────
    behind_count = 0
    for i in range(n):
        # Agent is "behind" if their x is less than attacker's x
        # (Team A attacks +X, Team B defends -X)
        if positions[i, 0] < target_positions[i, 0]:
            behind_count += 1
    time_behind_ratio = behind_count / max(1, n)

    # ── Min player distance ────────────────────────────────────
    min_player_dist = float("inf")
    if all_player_positions:
        for pid, other_pos in all_player_positions.items():
            if pid == agent_id:
                continue
            m = min(T, len(other_pos))
            for i in range(m):
                dist = math.hypot(
                    positions[i, 0] - other_pos[i, 0],
                    positions[i, 1] - other_pos[i, 1],
                )
                if dist < min_player_dist:
                    min_player_dist = dist
    if math.isinf(min_player_dist):
        min_player_dist = None

    return {
        "mean_marking_distance_cm": mean_marking_distance,
        "std_marking_distance_cm": std_marking_distance,
        "goal_side_success_rate": goal_side_success_rate,
        "time_behind_attacker_ratio": time_behind_ratio,
        "out_of_bounds_count": oob_count,
        "collision_count": collision_count,
        "max_speed_cm_s": max_speed,
        "mean_speed_cm_s": mean_speed,
        "speed_warning_count": speed_warning_count,
        "turn_angle_warning_count": turn_warning_count,
        "total_distance_cm": total_dist,
        "min_player_distance_cm": min_player_dist,
        "n_frames": T,
    }
