"""P2-8 多 seed 相机分布稳定性分析测试。"""

from grf_ue_bridge.tools.p2_8_camera_distribution_stability import (
    analyze_visibility_events,
    trajectory_is_valid,
)


def _frame(step, visible=True):
    return {
        "step": step,
        "players": [
            {"id": "L0", "position_m": [float(step), 0.0, 0.0]},
            {"id": "R0", "position_m": [0.0, 0.0, 0.0]},
        ],
        "visible": {"R0": visible},
    }


def test_trajectory_is_invalid_when_player_is_stationary_too_long():
    frames = [_frame(i) for i in range(20)]
    assert trajectory_is_valid(frames, fps=10, max_stationary_s=1.0) is False


def test_visibility_event_counts_p01_difference_from_anchor_pattern():
    anchor = [{"frame_index": i + 1, "objects": [{"entity_id": "R0", "class": "player", "in_frame": True}]} for i in range(3)]
    p01 = [{"frame_index": 1, "objects": [{"entity_id": "R0", "class": "player", "in_frame": False}]}, {"frame_index": 2, "objects": [{"entity_id": "R0", "class": "player", "in_frame": True}]}, {"frame_index": 3, "objects": [{"entity_id": "R0", "class": "player", "in_frame": False}]}]

    result = analyze_visibility_events({"CineCam_01": anchor, "CineCam_P01": p01}, "CineCam_P01")

    assert result["unique_event_count"] == 2
    assert result["players"] == ["R0"]
    assert result["event_durations_frames"] == [1, 1]
