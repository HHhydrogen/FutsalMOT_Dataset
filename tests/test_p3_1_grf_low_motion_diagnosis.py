"""P3-1.2 GRF 低运动诊断工具测试。"""

from grf_ue_bridge.tools.p3_1_grf_low_motion_diagnosis import analyze_snapshots


def _snapshot(step, owner, positions, directions, ball=(0.0, 0.0, 0.11), mode=0):
    return {
        "step": step,
        "observation": {
            "left_team": positions[:5],
            "right_team": positions[5:],
            "left_team_direction": directions[:5],
            "right_team_direction": directions[5:],
            "left_team_active": [True] * 5,
            "right_team_active": [True] * 5,
            "ball": list(ball),
            "ball_direction": [0.0, 0.0, 0.0],
            "ball_owned_team": owner[0],
            "ball_owned_player": owner[1],
            "game_mode": mode,
            "score": [0, 0],
            "steps_left": 3000 - step,
        },
        "done": False,
    }


def test_analyze_snapshots_reports_possession_runs_and_first_stops():
    positions = [[float(i), 0.0] for i in range(10)]
    moving = [[0.01, 0.0]] * 10
    stopped = [[0.0, 0.0]] * 10
    snapshots = [
        _snapshot(0, (-1, -1), positions, moving),
        _snapshot(1, (0, 2), positions, moving),
        _snapshot(2, (0, 2), positions, stopped),
        _snapshot(3, (0, 2), positions, stopped),
    ]

    report = analyze_snapshots(snapshots, fps=10.0, low_motion_mps=0.2)

    assert report["possession"]["longest_run"]["value"] == (0, 2)
    assert report["possession"]["longest_run"]["duration_s"] == 0.3
    assert report["players"]["L0"]["first_low_motion_frame"] == 2
    assert report["players"]["R4"]["first_low_motion_frame"] == 2
    assert report["game_state"]["steps_left"]["start"] == 3000
    assert report["game_state"]["steps_left"]["end"] == 2997


def test_analyze_snapshots_records_external_action_as_constant_and_no_internal_distribution():
    positions = [[0.0, 0.0] for _ in range(10)]
    directions = [[0.0, 0.0] for _ in range(10)]
    snapshots = [_snapshot(i, (-1, -1), positions, directions) for i in range(2)]

    report = analyze_snapshots(
        snapshots,
        fps=10.0,
        external_action_name="builtin_ai",
        external_action_index=19,
    )

    assert report["actions"] == {
        "external_action_name": "builtin_ai",
        "external_action_index": 19,
        "external_action_count_per_step": 1,
        "external_action_distribution": {"builtin_ai": 2},
        "internal_builtin_ai_distribution": "not_exposed_by_grf_observation",
    }
