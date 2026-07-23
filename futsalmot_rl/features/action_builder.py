"""Action extraction and conversion utilities.

Extracts continuous velocity actions from rule trajectories,
and defines the action specification for the RL environment.
"""

from __future__ import annotations

import math

import numpy as np

# Action spec
ACT_DIM = 2

# Player_05 limits (from the template defend_event parameters)
PLAYER_05_MAX_SPEED_CM_S = 540.0
PLAYER_05_MAX_ACCEL_CM_S2 = 950.0


def extract_action_from_trajectory(
    positions: list[tuple[float, float]],
    frame: int,
    fps: int = 30,
    max_speed: float = PLAYER_05_MAX_SPEED_CM_S,
) -> np.ndarray:
    """Extract a normalized action from a rule trajectory at a given frame.

    Action = [desired_vx_norm, desired_vy_norm] in [-1, 1].
    Computed from the displacement between frame and frame+1.
    """
    if frame >= len(positions) - 1:
        return np.array([0.0, 0.0], dtype=np.float32)

    vx = (positions[frame + 1][0] - positions[frame][0]) * fps
    vy = (positions[frame + 1][1] - positions[frame][1]) * fps

    # Normalize by max_speed and clip to [-1, 1]
    ax = max(-1.0, min(1.0, vx / max_speed))
    ay = max(-1.0, min(1.0, vy / max_speed))
    return np.array([ax, ay], dtype=np.float32)


def extract_all_actions(
    positions: list[tuple[float, float]],
    fps: int = 30,
    max_speed: float = PLAYER_05_MAX_SPEED_CM_S,
) -> np.ndarray:
    """Extract actions for all frames from a position sequence.

    Returns shape (T-1, 2) array.
    """
    n = len(positions)
    actions = np.zeros((max(0, n - 1), 2), dtype=np.float32)
    for i in range(n - 1):
        actions[i] = extract_action_from_trajectory(positions, i, fps, max_speed)
    return actions


def apply_motion_constraints(
    action: np.ndarray,
    current_vel: tuple[float, float],
    max_speed: float = PLAYER_05_MAX_SPEED_CM_S,
    max_accel: float = PLAYER_05_MAX_ACCEL_CM_S2,
    fps: int = 30,
) -> tuple[float, float]:
    """Apply acceleration and speed limits to a desired action.

    Args:
        action: Normalized [-1, 1] desired velocity action.
        current_vel: Current (vx, vy) velocity in cm/s.
        max_speed: Maximum speed in cm/s.
        max_accel: Maximum acceleration in cm/s².
        fps: Frames per second.

    Returns:
        (new_vx, new_vy) in cm/s.
    """
    # Denormalize desired velocity
    desired_vx = float(action[0]) * max_speed
    desired_vy = float(action[1]) * max_speed

    cvx, cvy = current_vel
    dvx = desired_vx - cvx
    dvy = desired_vy - cvy
    dv_mag = math.hypot(dvx, dvy)
    max_delta = max_accel / fps

    if dv_mag > max_delta:
        dvx = dvx / dv_mag * max_delta
        dvy = dvy / dv_mag * max_delta

    new_vx = cvx + dvx
    new_vy = cvy + dvy
    speed = math.hypot(new_vx, new_vy)

    if speed > max_speed:
        new_vx = new_vx / speed * max_speed
        new_vy = new_vy / speed * max_speed

    return (new_vx, new_vy)


def action_spec() -> tuple[int, float, float]:
    """Return (act_dim, low, high) for gym.Box."""
    return (ACT_DIM, -1.0, 1.0)
