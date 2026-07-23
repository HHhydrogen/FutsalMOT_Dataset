"""Coordinate and speed normalization for FutsalMOT-RL."""

from __future__ import annotations

import math

from futsalmot_rl.core.rl_paths import (
    BALL_MAX_SPEED_CM_S,
    COURT_X_MAX,
    COURT_X_MIN,
    COURT_Y_MAX,
    COURT_Y_MIN,
    PLAYER_MAX_SPEED_CM_S,
)


class Normalizer:
    """Normalize / denormalize coordinates and speeds.

    All methods are static — no state needed for the first version.
    """

    # ── Court coordinates ───────────────────────────────────────

    @staticmethod
    def x(x_cm: float) -> float:
        return (float(x_cm) - COURT_X_MIN) / (COURT_X_MAX - COURT_X_MIN) * 2.0 - 1.0

    @staticmethod
    def y(y_cm: float) -> float:
        return (float(y_cm) - COURT_Y_MIN) / (COURT_Y_MAX - COURT_Y_MIN) * 2.0 - 1.0

    @staticmethod
    def denormalize_x(x_norm: float) -> float:
        return (float(x_norm) + 1.0) / 2.0 * (COURT_X_MAX - COURT_X_MIN) + COURT_X_MIN

    @staticmethod
    def denormalize_y(y_norm: float) -> float:
        return (float(y_norm) + 1.0) / 2.0 * (COURT_Y_MAX - COURT_Y_MIN) + COURT_Y_MIN

    # ── Speed ────────────────────────────────────────────────────

    @staticmethod
    def player_speed(v_cm_s: float) -> float:
        return float(v_cm_s) / PLAYER_MAX_SPEED_CM_S

    @staticmethod
    def ball_speed(v_cm_s: float) -> float:
        return float(v_cm_s) / BALL_MAX_SPEED_CM_S

    @staticmethod
    def denormalize_player_speed(v_norm: float) -> float:
        return float(v_norm) * PLAYER_MAX_SPEED_CM_S

    # ── Distance ─────────────────────────────────────────────────

    @staticmethod
    def distance(d_cm: float, scale: float = 2200.0) -> float:
        return float(d_cm) / scale

    @staticmethod
    def denormalize_distance(d_norm: float, scale: float = 2200.0) -> float:
        return float(d_norm) * scale

    # ── Steps ────────────────────────────────────────────────────

    @staticmethod
    def steps_left(steps: int, max_steps: int = 300) -> float:
        return float(steps) / float(max_steps)

    # ── Angle (auto) ─────────────────────────────────────────────

    @staticmethod
    def angle_sin_cos(
        origin_x: float, origin_y: float, target_x: float, target_y: float
    ) -> tuple[float, float]:
        """Return (sin, cos) of the angle from origin to target."""
        dx = float(target_x) - float(origin_x)
        dy = float(target_y) - float(origin_y)
        dist = math.hypot(dx, dy)
        if dist < 1e-8:
            return (0.0, 1.0)
        return (dy / dist, dx / dist)
