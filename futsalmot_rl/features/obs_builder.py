"""Observation builder for the FutsalDefenderFollowEnv.

Produces a 39-dim observation vector for Player_05 (defender) targeting Player_01.
"""

from __future__ import annotations

import math

import numpy as np

from futsalmot_rl.core.rl_paths import COURT_X_MAX, COURT_X_MIN, COURT_Y_MAX, COURT_Y_MIN
from futsalmot_rl.features.normalization import Normalizer

# Supported event types for one-hot encoding
EVENT_TYPES = ["hold", "move", "dribble", "pass", "receive", "defend_follow", "shot"]
EVENT_TYPE_TO_IDX = {t: i for i, t in enumerate(EVENT_TYPES)}

OBS_DIM = 38


def build_observation(
    *,
    # Self (Player_05)
    self_pos: tuple[float, float],
    self_vel: tuple[float, float],
    self_yaw_deg: float,
    # Target (Player_01)
    target_pos: tuple[float, float],
    target_vel: tuple[float, float],
    # Ball
    ball_pos: tuple[float, float],
    ball_vel: tuple[float, float],
    # Own goal center (Team B defends -X goal)
    own_goal_pos: tuple[float, float] = (COURT_X_MIN, 0.0),
    # Possession info
    possession_owner: str | None = None,
    current_event_type: str | None = None,
    steps_left: int = 300,
    total_steps: int = 300,
) -> np.ndarray:
    """Build a 39-dim observation vector for the defender.

    All spatial values are normalized to roughly [-1, 1].
    """
    # ── Self ────────────────────────────────────────────────────
    sx, sy = self_pos
    svx, svy = self_vel
    yaw_rad = math.radians(float(self_yaw_deg))

    features: list[float] = [
        Normalizer.x(sx),  # self_x_norm
        Normalizer.y(sy),  # self_y_norm
        Normalizer.player_speed(svx),  # self_vx_norm
        Normalizer.player_speed(svy),  # self_vy_norm
        math.sin(yaw_rad),  # self_yaw_sin
        math.cos(yaw_rad),  # self_yaw_cos
    ]

    # ── Target (Player_01) ──────────────────────────────────────
    tx, ty = target_pos
    tvx, tvy = target_vel
    features.extend(
        [
            Normalizer.x(tx),  # target_x_norm
            Normalizer.y(ty),  # target_y_norm
            Normalizer.player_speed(tvx),  # target_vx_norm
            Normalizer.player_speed(tvy),  # target_vy_norm
        ]
    )

    # ── Ball ────────────────────────────────────────────────────
    bx, by = ball_pos
    bvx, bvy = ball_vel
    features.extend(
        [
            Normalizer.x(bx),  # ball_x_norm
            Normalizer.y(by),  # ball_y_norm
            Normalizer.ball_speed(bvx),  # ball_vx_norm
            Normalizer.ball_speed(bvy),  # ball_vy_norm
        ]
    )

    # ── Own goal ─────────────────────────────────────────────────
    ogx, ogy = own_goal_pos
    features.extend(
        [
            Normalizer.x(ogx),  # own_goal_x_norm
            Normalizer.y(ogy),  # own_goal_y_norm
        ]
    )

    # ── Distances ────────────────────────────────────────────────
    dist_to_target = math.hypot(tx - sx, ty - sy)
    dist_to_ball = math.hypot(bx - sx, by - sy)
    dist_to_goal = math.hypot(ogx - sx, ogy - sy)
    features.extend(
        [
            Normalizer.distance(dist_to_target),
            Normalizer.distance(dist_to_ball),
            Normalizer.distance(dist_to_goal),
        ]
    )

    # ── Angles ──────────────────────────────────────────────────
    a_target_sin, a_target_cos = Normalizer.angle_sin_cos(sx, sy, tx, ty)
    a_ball_sin, a_ball_cos = Normalizer.angle_sin_cos(sx, sy, bx, by)
    features.extend([a_target_sin, a_target_cos, a_ball_sin, a_ball_cos])

    # ── Boundary distances ──────────────────────────────────────
    features.extend(
        [
            Normalizer.x(sx - COURT_X_MIN),  # boundary_left
            Normalizer.x(COURT_X_MAX - sx),  # boundary_right
            Normalizer.y(sy - COURT_Y_MIN),  # boundary_top
            Normalizer.y(COURT_Y_MAX - sy),  # boundary_bottom
        ]
    )

    # ── Possession ──────────────────────────────────────────────
    pos_target = 1.0 if possession_owner == "Player_01" else 0.0
    pos_teammate = (
        1.0 if possession_owner in ("Player_05", "Player_06", "Player_07", "Player_08") else 0.0
    )
    pos_free = 1.0 if possession_owner is None else 0.0
    features.extend([pos_target, pos_teammate, pos_free])

    # ── Steps left ──────────────────────────────────────────────
    features.append(Normalizer.steps_left(steps_left, total_steps))

    # ── Event type one-hot ──────────────────────────────────────
    evt_idx = EVENT_TYPE_TO_IDX.get(current_event_type, -1)
    for i in range(len(EVENT_TYPES)):
        features.append(1.0 if i == evt_idx else 0.0)

    arr = np.array(features, dtype=np.float32)
    assert arr.shape == (OBS_DIM,), f"Expected obs dim {OBS_DIM} got {arr.shape}"
    return arr


def get_obs_dim() -> int:
    """Return the observation dimension."""
    return OBS_DIM
