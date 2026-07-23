"""Tests for evaluation metrics: consistency check for BC and PPO."""

from __future__ import annotations

import numpy as np
import pytest

from futsalmot_rl.benchmark.metrics import compute_policy_metrics


def test_metrics_have_required_keys():
    positions = np.zeros((10, 2), dtype=np.float32)
    target_positions = np.zeros((10, 2), dtype=np.float32)
    ball_positions = np.zeros((10, 2), dtype=np.float32)
    other = {"Player_02": np.zeros((10, 2), dtype=np.float32)}

    metrics = compute_policy_metrics(
        positions=positions,
        target_positions=target_positions,
        ball_positions=ball_positions,
        all_player_positions=other,
    )

    required = [
        "mean_marking_distance_cm",
        "std_marking_distance_cm",
        "out_of_bounds_count",
        "collision_count",
        "max_speed_cm_s",
        "goal_side_success_rate",
        "time_behind_attacker_ratio",
    ]
    for key in required:
        assert key in metrics, f"Missing required metric: {key}"


def test_no_nan_in_metrics():
    rng = np.random.RandomState(42)
    positions = rng.randn(100, 2).cumsum(0).astype(np.float32)
    target_positions = rng.randn(100, 2).cumsum(0).astype(np.float32)

    metrics = compute_policy_metrics(
        positions=positions,
        target_positions=target_positions,
    )
    for key, value in metrics.items():
        if isinstance(value, float):
            assert np.isfinite(value), f"Non-finite metric: {key}={value}"
