"""用解析轨迹核验物理单位、时间边界、转角及邻近代理。"""

import math
from pathlib import Path

import pytest

from grf_ue_bridge.motion_coverage import CoverageThresholds, analyze_frames, bouts
from grf_ue_bridge.tools.motion_coverage_report import scoped_reference

ENTITIES = [{"entity_id": "L1", "team": "left", "is_goalkeeper": False},
            {"entity_id": "R1", "team": "right", "is_goalkeeper": False}]


def frames(positions, times=None, separation=0.5):
    times = times or list(range(len(positions)))
    return [{"step": i, "time_seconds": t, "players": [
        {"id": "L1", "position_m": [p[0], p[1], 0]},
        {"id": "R1", "position_m": [p[0], p[1] + separation, 0]}]}
        for i, (t, p) in enumerate(zip(times, positions))]


def test_actual_time_not_frame_count_and_irregular_spacing():
    data = frames([[0, 0], [1, 0], [3, 0], [4, 0]], [0, 1, 3, 4])
    result = analyze_frames(data, ENTITIES, CoverageThresholds(min_active_outfields=2))
    assert result["trajectory_span_s"] == 4
    assert result["player_seconds"] == 8
    assert result["players"]["L1"]["path_length_m"] == 4
    assert result["players"]["L1"]["speed_mps"]["p50"] == 1
    assert result["team_active"]["longest_s"] == 4
    assert result["players"]["L1"]["acceleration_mps2"]["max"] == 0
    assert result["proximity"]["1.0"]["cross_team_pair_seconds"] == 4


def test_proximity_is_pair_time_not_extra_unique_motion_time():
    result = analyze_frames(frames([[0, 0], [1, 0], [2, 0]], separation=1.5), ENTITIES)
    assert result["trajectory_span_s"] == 2
    assert result["proximity"]["1.0"]["pair_seconds"] == 0
    assert result["proximity"]["2.0"]["cross_team_pair_seconds"] == 2


def test_group_ball_entity_is_not_counted_as_player():
    result = analyze_frames(frames([[0, 0], [1, 0], [2, 0]]),
        ENTITIES + [{"entity_id": "BALL", "team": None, "is_goalkeeper": False}])
    assert result["entity_count"] == 2
    assert result["player_seconds"] == 4


def test_turn_wrap_uses_small_angle():
    a, b = math.radians(179), math.radians(-179)
    result = analyze_frames(frames([[0, 0], [math.cos(a), math.sin(a)],
        [math.cos(a) + math.cos(b), math.sin(a) + math.sin(b)]]), ENTITIES)
    assert result["players"]["L1"]["turn_rate_abs_deg_s"]["max"] == pytest.approx(2)
    assert result["players"]["L1"]["sharp_turn"]["total_s"] == 0


def test_stationary_heading_does_not_invent_turn():
    result = analyze_frames(frames([[0, 0], [0, 0], [1, 0]]), ENTITIES)
    assert result["players"]["L1"]["turn_valid_support_s"] == 0
    assert result["players"]["L1"]["turn_rate_abs_deg_s"]["max"] is None


def test_speed_bins_and_acceleration_units():
    result = analyze_frames(frames([[0, 0], [0, 0], [4, 0]]), ENTITIES)
    assert result["players"]["L1"]["gait_seconds"]["sprint"] == 1
    assert result["players"]["L1"]["acceleration_mps2"]["max"] == 4


def test_bouts_never_bridge_inactive_gap():
    result = bouts([True, True, False, True, True], [.2] * 5, .5)
    assert result["total_s"] == pytest.approx(.8)
    assert result["longest_s"] == pytest.approx(.4)
    assert result["episode_count"] == 2
    assert result["persistent_episode_count"] == 0


@pytest.mark.parametrize("damage", ["duplicate_player", "gap", "bad_time", "nan"])
def test_rejects_invalid_input(damage):
    data = frames([[0, 0], [1, 0], [2, 0]])
    if damage == "duplicate_player":
        data[1]["players"][1]["id"] = "L1"
    elif damage == "gap":
        data[1]["step"] = 5
    elif damage == "bad_time":
        data[1]["time_seconds"] = 0
    else:
        data[1]["players"][0]["position_m"][0] = float("nan")
    with pytest.raises(ValueError):
        analyze_frames(data, ENTITIES)


def test_manifest_reference_cannot_escape_root(tmp_path):
    with pytest.raises(ValueError):
        scoped_reference(tmp_path, "../outside.json", tmp_path)
    assert scoped_reference(tmp_path, "inside.json", tmp_path) == tmp_path / "inside.json"
