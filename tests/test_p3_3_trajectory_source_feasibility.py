"""P3-3 trajectory source failure characterization tests."""

import json

from grf_ue_bridge.tools.p3_3_trajectory_source_feasibility import (
    assess_ue_native_feasibility,
    build_project_owned_prototype,
    characterize_failures,
    contract_compatibility,
    evaluate_frames_candidate,
    inventory_grf_sources,
    sha256_file,
    validate_bounded_settings,
    write_grf_native_infeasibility,
)


def test_ue_native_assessment_reports_playback_support_separately_from_simulation_support():
    assessment = assess_ue_native_feasibility()
    assert assessment["playback_contract_supported"] is True
    assert assessment["complete_5v5_simulation_available"] is False
    assert assessment["disposition"] == "ARCHITECTURALLY_POSSIBLE_NOT_READY"


def test_project_owned_prototype_is_deterministic_and_contract_shaped():
    first = build_project_owned_prototype(seed=42, frames=10, fps=10.0)
    second = build_project_owned_prototype(seed=42, frames=10, fps=10.0)
    assert first == second
    assert len(first) == 10
    assert {player["id"] for player in first[0]["players"]} == {
        *(f"L{i}" for i in range(5)),
        *(f"R{i}" for i in range(5)),
    }


def test_project_owned_prototype_enforces_motion_and_game_state_acceptance_criteria():
    frames = build_project_owned_prototype(seed=42, frames=37, fps=10.0)
    for frame in frames:
        assert frame["score"] == [0, 0]
        assert frame["game_mode"] == 0
        assert len(frame["ball"]["position_m"]) == 3
        assert len(frame["ball"]["velocity_mps"]) == 3
        assert frame["ball"]["position_m"][0] >= -20.0
        assert frame["ball"]["position_m"][0] <= 20.0
        assert frame["ball"]["position_m"][1] >= -10.0
        assert frame["ball"]["position_m"][1] <= 10.0
        owners = []
        for player in frame["players"]:
            assert len(player["position_m"]) == 3
            assert len(player["velocity_mps"]) == 2
            assert player["active"] is True
            assert isinstance(player["has_ball"], bool)
            assert -20.0 <= player["position_m"][0] <= 20.0
            assert -10.0 <= player["position_m"][1] <= 10.0
            if player["has_ball"]:
                owners.append(player["id"])
        assert len(owners) == 1
        owner = owners[0]
        assert frame["ball_owned_team"] == (0 if owner.startswith("L") else 1)
        assert frame["ball_owned_player"] == int(owner[1:])
        left_x = [p["position_m"][0] for p in frame["players"] if p["id"].startswith("L")]
        right_x = [p["position_m"][0] for p in frame["players"] if p["id"].startswith("R")]
        assert max(left_x) < 0.0
        assert min(right_x) > 0.0


def test_project_owned_prototype_uses_deterministic_spatial_possession_handoffs():
    frames = build_project_owned_prototype(seed=42, frames=37, fps=10.0)
    owners = [
        next(player["id"] for player in frame["players"] if player["has_ball"])
        for frame in frames
    ]
    assert owners[0] == "L0"
    assert len(set(owners)) > 1
    assert any(previous != current for previous, current in zip(owners, owners[1:]))
    assert all(
        frame["ball"]["position_m"][:2] == [
            player["position_m"][0], player["position_m"][1]
        ]
        for frame in frames
        for player in frame["players"]
        if player["has_ball"]
    )


def test_project_owned_prototype_keeps_ball_and_owner_consistent_for_300_frames():
    frames = build_project_owned_prototype(seed=42, frames=300, fps=10.0)

    assert len(frames) == 300
    for frame in frames:
        owners = [player for player in frame["players"] if player["has_ball"]]
        assert len(owners) == 1
        owner = owners[0]
        assert frame["ball"]["position_m"][:2] == owner["position_m"][:2]
        assert frame["ball_owned_team"] == (0 if owner["id"].startswith("L") else 1)
        assert frame["ball_owned_player"] == int(owner["id"][1:])


def test_project_owned_prototype_is_not_an_independent_random_walk():
    first = build_project_owned_prototype(seed=42, frames=10, fps=10.0)
    second = build_project_owned_prototype(seed=43, frames=10, fps=10.0)
    first_positions = [
        player["position_m"]
        for frame in first
        for player in frame["players"]
    ]
    second_positions = [
        player["position_m"]
        for frame in second
        for player in frame["players"]
    ]
    assert first_positions != second_positions
    assert first[0]["players"][0]["position_m"] != first[1]["players"][0]["position_m"]
    assert first[1]["players"][0]["velocity_mps"] != [0.0, 0.0]


def test_contract_compatibility_accepts_legacy_frames_without_optional_fields():
    assert contract_compatibility([_frame(0)]) == {"compatible": True, "errors": []}


def test_contract_compatibility_validates_declared_source_neutral_fields():
    frame = _frame(0)
    frame["players"][0].update({
        "velocity_mps": [0.0, 0.0],
        "active": True,
        "has_ball": False,
    })
    frame["ball"]["velocity_mps"] = [0.0, 0.0, 0.0]
    frame["ball_owned_team"] = -1
    frame["ball_owned_player"] = -1
    result = contract_compatibility([frame])
    assert result["compatible"] is True


def test_contract_compatibility_reports_structured_errors_for_malformed_declared_fields():
    frame = _frame(0)
    frame["players"][0]["velocity_mps"] = [0.0]
    frame["players"][1]["active"] = "yes"
    frame["players"][2]["has_ball"] = 1
    frame["ball"]["velocity_mps"] = [0.0, 0.0]
    frame["ball_owned_team"] = 2
    frame["ball_owned_player"] = "0"
    frame["score"] = [0]
    frame["game_mode"] = "normal"
    result = contract_compatibility([frame])
    assert result["compatible"] is False
    assert {error["code"] for error in result["errors"]} == {
        "player_velocity_invalid",
        "player_active_invalid",
        "player_has_ball_invalid",
        "ball_velocity_invalid",
        "ownership_invalid",
        "score_invalid",
        "game_mode_invalid",
    }


def test_contract_compatibility_rejects_malformed_player_positions():
    invalid_positions = [None, [0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, float("nan"), 0.0]]
    for position in invalid_positions:
        frame = _frame(0)
        frame["players"][0]["position_m"] = position
        result = contract_compatibility([frame])
        assert result["compatible"] is False
        assert result["errors"][0]["code"] == "player_position_invalid"


def _frame(step):
    players = []
    for team in ("L", "R"):
        for index in range(5):
            players.append({
                "id": f"{team}{index}",
                "position_m": [step * 0.1 + index, float(index), 0.0],
                "velocity_mps": [1.0, 0.0],
                "active": True,
            })
    return {
        "step": step,
        "time_seconds": step / 10.0,
        "players": players,
        "ball": {"position_m": [0.0, 0.0, 0.11]},
        "score": [0, 0],
        "game_mode": 0,
    }


def test_characterization_counts_failure_criteria_without_using_gk_as_a_gate():
    results = [
        {
            "seed": 42,
            "valid": False,
            "gate_failures": ["outfield_active_ratio", "team_active_coverage"],
            "min_outfield_active_ratio": 0.39,
            "max_outfield_stationary_streak_s": 13.5,
            "max_gk_stationary_streak_s": 22.0,
            "team_active_outfield_coverage": 0.39,
            "longest_global_low_motion_plateau_s": 15.3,
        },
        {
            "seed": 44,
            "valid": True,
            "gate_failures": [],
            "min_outfield_active_ratio": 0.763,
            "max_outfield_stationary_streak_s": 1.1,
            "max_gk_stationary_streak_s": 12.2,
            "team_active_outfield_coverage": 0.913,
            "longest_global_low_motion_plateau_s": 0.7,
        },
    ]
    report = characterize_failures(results)
    assert report["criterion_failure_counts"] == {
        "outfield_active_ratio": 1,
        "outfield_stationary_streak": 0,
        "team_active_coverage": 1,
    }
    assert report["gk_stationary_streak_is_gate"] is False
    assert report["severe_global_low_motion_seeds"] == [42]


def test_characterization_classifies_single_near_miss():
    results = [{
        "seed": 45,
        "valid": False,
        "gate_failures": ["team_active_coverage"],
        "min_outfield_active_ratio": 0.727,
        "max_outfield_stationary_streak_s": 1.3,
        "max_gk_stationary_streak_s": 11.6,
        "team_active_outfield_coverage": 0.89,
        "longest_global_low_motion_plateau_s": 0.8,
    }]
    report = characterize_failures(results)
    assert report["single_criterion_failure_seeds"] == [45]
    assert report["classification_by_seed"]["45"] == "single_criterion_near_miss"


def test_characterization_counts_failure_combinations():
    results = [
        {"seed": 42, "gate_failures": ["outfield_active_ratio", "team_active_coverage"]},
        {"seed": 43, "gate_failures": ["outfield_active_ratio", "team_active_coverage"]},
        {"seed": 47, "gate_failures": ["outfield_active_ratio", "outfield_stationary_streak", "team_active_coverage"]},
        {"seed": 44, "gate_failures": []},
    ]
    report = characterize_failures(results)
    assert report["failure_combinations"] == {
        "outfield_active_ratio+team_active_coverage": 2,
        "outfield_active_ratio+outfield_stationary_streak+team_active_coverage": 1,
    }


def test_characterization_classifies_multi_criterion_failures():
    report = characterize_failures([{
        "seed": 47,
        "valid": False,
        "gate_failures": ["outfield_active_ratio", "team_active_coverage"],
        "longest_global_low_motion_plateau_s": 0.0,
    }])
    assert report["classification_by_seed"]["47"] == "multi_criterion_failure"


def test_severe_classification_uses_inclusive_boundary_and_takes_precedence():
    report = characterize_failures([
        {
            "seed": 46,
            "valid": False,
            "gate_failures": ["team_active_coverage"],
            "longest_global_low_motion_plateau_s": 5.0,
        },
        {
            "seed": 47,
            "valid": False,
            "gate_failures": ["outfield_active_ratio", "team_active_coverage"],
            "longest_global_low_motion_plateau_s": 5.1,
        },
    ])
    assert report["severe_global_low_motion_seeds"] == [46, 47]
    assert report["classification_by_seed"] == {
        "46": "severe_global_low_motion",
        "47": "severe_global_low_motion",
    }


def test_gk_metrics_are_preserved_without_becoming_strict_failures():
    report = characterize_failures([{
        "seed": 44,
        "valid": True,
        "gate_failures": [],
        "max_gk_stationary_streak_s": 12.2,
        "longest_global_low_motion_plateau_s": 0.7,
    }])
    assert report["gk_stationary_streak_is_gate"] is False
    assert report["gk_metrics_by_seed"]["44"] == {
        "max_gk_stationary_streak_s": 12.2,
    }
    assert report["criterion_failure_counts"] == {
        "outfield_active_ratio": 0,
        "outfield_stationary_streak": 0,
        "team_active_coverage": 0,
    }


def test_unrecognized_and_duplicate_failures_are_not_strict_counts():
    report = characterize_failures([
        {
            "seed": 48,
            "valid": False,
            "gate_failures": ["unknown_failure"],
            "longest_global_low_motion_plateau_s": 0.0,
        },
        {
            "seed": 49,
            "valid": False,
            "gate_failures": ["team_active_coverage", "team_active_coverage"],
            "longest_global_low_motion_plateau_s": 0.0,
        },
    ])
    assert report["criterion_failure_counts"] == {
        "outfield_active_ratio": 0,
        "outfield_stationary_streak": 0,
        "team_active_coverage": 1,
    }
    assert report["failure_combinations"] == {
        "team_active_coverage": 1,
    }
    assert report["classification_by_seed"]["48"] == "single_criterion_near_miss"
    assert report["classification_by_seed"]["49"] == "multi_criterion_failure"


def test_single_near_miss_uses_raw_failure_length_and_ignores_valid_flag():
    report = characterize_failures([{
        "seed": 50,
        "valid": True,
        "gate_failures": ["team_active_coverage"],
        "longest_global_low_motion_plateau_s": 0.0,
    }])
    assert report["single_criterion_failure_seeds"] == [50]
    assert report["classification_by_seed"]["50"] == "single_criterion_near_miss"


def test_grf_inventory_distinguishes_available_api_from_unavailable_policy_resources():
    inventory = inventory_grf_sources()
    by_name = {item["name"]: item for item in inventory}
    assert by_name["builtin_ai"]["api_evidence"] == "action_set_v2_index_19"
    assert by_name["bot"]["api_evidence"] == "sample_bot_player"
    assert by_name["replay"]["api_evidence"] == "replay_player_requires_trace"
    assert by_name["ppo_checkpoint"]["disposition"] in {
        "INFEASIBLE_IN_CURRENT_ENVIRONMENT",
        "AVAILABLE_FOR_PROBE",
    }


def test_contract_compatibility_requires_fixed_entities_and_frame_count(tmp_path):
    frames = [_frame(step) for step in range(3)]
    path = tmp_path / "frames.jsonl"
    path.write_text("\n".join(json.dumps(frame) for frame in frames) + "\n", encoding="utf-8")
    evaluation = evaluate_frames_candidate("prototype", path, fps=10.0)
    assert evaluation["frame_count"] == 3
    assert evaluation["contract"]["compatible"] is True


def test_contract_compatibility_rejects_missing_ball(tmp_path):
    frame = _frame(0)
    frame.pop("ball")
    assert contract_compatibility([frame])["compatible"] is False


def test_contract_compatibility_rejects_non_mapping_frame_and_player():
    frame = _frame(0)
    frame["players"][0] = None
    result = contract_compatibility([None, frame])
    assert result["compatible"] is False
    assert {error["code"] for error in result["errors"]} == {
        "frame_not_mapping",
        "player_not_mapping",
    }


def test_contract_compatibility_rejects_duplicate_player_ids():
    frame = _frame(0)
    frame["players"][1]["id"] = frame["players"][0]["id"]
    result = contract_compatibility([frame])
    assert result["compatible"] is False
    assert result["errors"][0]["code"] == "player_ids_invalid"


def test_contract_compatibility_rejects_malformed_ball_positions():
    invalid_positions = [None, "0,0,0", [0.0, 0.0], [0.0, 0.0, "0.11"]]
    for position in invalid_positions:
        frame = _frame(0)
        frame["ball"]["position_m"] = position
        result = contract_compatibility([frame])
        assert result["compatible"] is False
        assert result["errors"][0]["code"] == "ball_position_invalid"


def test_evaluate_candidate_contract_failure_is_authoritative(tmp_path):
    frames = [_frame(0), _frame(1)]
    frames[0].pop("ball")
    path = tmp_path / "frames.jsonl"
    path.write_text("\n".join(json.dumps(frame) for frame in frames) + "\n", encoding="utf-8")
    evaluation = evaluate_frames_candidate("prototype", path, fps=10.0)
    assert evaluation["contract"]["compatible"] is False
    assert evaluation["valid"] is False
    assert evaluation["motion_quality"]["total_frames"] == 2


def test_evaluate_candidate_malformed_frame_keeps_safe_error_result(tmp_path):
    path = tmp_path / "frames.jsonl"
    path.write_text(json.dumps(None) + "\n", encoding="utf-8")
    evaluation = evaluate_frames_candidate("prototype", path, fps=10.0)
    assert evaluation["valid"] is False
    assert evaluation["motion_quality"] is None
    assert evaluation["motion_quality_error"]


def test_sha256_file_matches_standard_digest(tmp_path):
    path = tmp_path / "payload.bin"
    path.write_bytes(b"futsal-mot")
    assert sha256_file(path) == "6483bd04436f3ef5774edbc92fa903f4d7f5981774d48aa30b8f97417f0ba973"


def test_evaluate_candidate_uses_fixed_goalkeeper_ids(tmp_path):
    frames = [_frame(0), _frame(1)]
    path = tmp_path / "frames.jsonl"
    path.write_text("\n".join(json.dumps(frame) for frame in frames) + "\n", encoding="utf-8")
    evaluation = evaluate_frames_candidate("prototype", path, fps=10.0)
    assert evaluation["motion_quality"]["gks"] == ["L0", "R0"]


def test_bounded_settings_reject_out_of_scope_values():
    validate_bounded_settings(scenario="5_vs_5", frames=300, fps=10.0, seeds=[42, 43, 44, 45])
    for kwargs in (
        {"scenario": "11_vs_11", "frames": 300, "fps": 10.0, "seeds": [42, 43, 44, 45]},
        {"scenario": "5_vs_5", "frames": 301, "fps": 10.0, "seeds": [42, 43, 44, 45]},
        {"scenario": "5_vs_5", "frames": 300, "fps": 30.0, "seeds": [42, 43, 44, 45]},
        {"scenario": "5_vs_5", "frames": 300, "fps": 10.0, "seeds": [42, 43, 46, 45]},
    ):
        try:
            validate_bounded_settings(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("out-of-scope settings must be rejected")


def test_candidate_result_labels_project_owned_prototype(tmp_path):
    frames_path = tmp_path / "frames.jsonl"
    frames_path.write_text(
        "\n".join(json.dumps(frame) for frame in build_project_owned_prototype(seed=42, frames=2, fps=10.0)) + "\n",
        encoding="utf-8",
    )
    result = evaluate_frames_candidate("project_owned_prototype", frames_path, fps=10.0)
    assert result["prototype"] is True
    assert result["production"] is False


def test_infeasibility_artifact_contains_concrete_evidence(tmp_path):
    path = write_grf_native_infeasibility(tmp_path)
    artifact = json.loads(path.read_text(encoding="utf-8"))
    assert artifact["candidate"] == "grf_native_alternative"
    assert artifact["disposition"] == "INFEASIBLE_IN_CURRENT_ENVIRONMENT"
    assert "no local checkpoint/trace" in artifact["evidence"]
    assert "GRF_MARL runtime dependencies/policies unavailable" in artifact["evidence"]
