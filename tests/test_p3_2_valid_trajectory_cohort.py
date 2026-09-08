"""P3-2 valid trajectory cohort selection tests."""

from grf_ue_bridge.tools.p3_2_valid_trajectory_cohort import (
    assess_motion_quality,
    select_cohort,
)


def _metrics(**overrides):
    metrics = {
        "players": {
            "L0": {"is_gk": True, "longest_stationary_streak_s": 1.0},
            "R0": {"is_gk": True, "longest_stationary_streak_s": 1.0},
            **{
                pid: {
                    "is_gk": False,
                    "active_ratio": 0.80,
                    "longest_stationary_streak_s": 1.0,
                }
                for pid in ("L1", "L2", "L3", "L4", "R1", "R2", "R3", "R4")
            },
        },
        "team_active_outfield_coverage": 0.95,
        "longest_global_low_motion_plateau_s": 1.0,
    }
    metrics.update(overrides)
    return metrics


def test_assess_motion_quality_uses_existing_gate_without_conditional_upgrade():
    assert assess_motion_quality(_metrics())["valid"] is True
    invalid = _metrics(team_active_outfield_coverage=0.899)
    assert assess_motion_quality(invalid)["valid"] is False


def test_goalkeeper_streak_is_reported_but_does_not_change_existing_cohort_gate():
    metrics = _metrics()
    metrics["players"]["L0"]["longest_stationary_streak_s"] = 12.2
    metrics["players"]["R0"]["longest_stationary_streak_s"] = 12.2
    decision = assess_motion_quality(metrics)
    assert decision["valid"] is True
    assert "goalkeeper_stationary_streak" not in decision["failures"]


def test_select_cohort_uses_numeric_order_and_reports_insufficient():
    results = [
        {"seed": 45, "valid": True},
        {"seed": 42, "valid": True},
        {"seed": 44, "valid": False},
        {"seed": 43, "valid": True},
    ]
    selected = select_cohort(results, target_size=3)
    assert selected == {
        "status": "PASS",
        "valid_seeds": [42, 43, 45],
        "selected_seeds": [42, 43, 45],
        "target_size": 3,
    }

    insufficient = select_cohort(results, target_size=4)
    assert insufficient["status"] == "INSUFFICIENT"
    assert insufficient["valid_seeds"] == [42, 43, 45]
    assert insufficient["selected_seeds"] == []


def test_selection_does_not_consume_camera_fields():
    results = [
        {"seed": 42, "valid": True, "p01_events": 999},
        {"seed": 43, "valid": True, "camera_score": -1},
    ]
    assert select_cohort(results, target_size=2)["selected_seeds"] == [42, 43]
