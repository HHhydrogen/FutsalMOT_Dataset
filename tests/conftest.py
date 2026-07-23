"""Shared test fixtures and helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture
def tmp_project(tmp_path: Path) -> Path:
    """Create a temporary directory tree mimicking a FutsalMOT project.

    Structure:
        <tmp>/
          FutsalMOT.uproject          ← UE project root marker
          Content/
            FutsalMOT/
              code/
                pyproject.toml        ← repo root marker
                futsalmot_rl/
                  core/
                    paths.py
                configs/
                  runs/
                  pipeline_config.json
    """
    root = tmp_path / "project"
    root.mkdir()

    # UE project root marker
    (root / "FutsalMOT.uproject").write_text("{}", encoding="utf-8")

    # Repo root marker
    code_dir = root / "Content" / "FutsalMOT" / "code"
    code_dir.mkdir(parents=True)
    (code_dir / "pyproject.toml").write_text("", encoding="utf-8")

    # Configs
    (code_dir / "configs").mkdir()
    (code_dir / "configs" / "runs").mkdir()
    (code_dir / "configs" / "pipeline_config.json").write_text("{}", encoding="utf-8")

    # RL package marker
    rl_core = code_dir / "futsalmot_rl" / "core"
    rl_core.mkdir(parents=True)

    return root


@pytest.fixture
def mini_a33(tmp_path: Path) -> Path:
    """Create a minimal valid A3.3 fixture file."""
    data = {
        "schema_version": "3.1",
        "seq_id": "episode_test_fixture",
        "episode_id": "episode_test_fixture",
        "timeline": {"frame_start": 0, "frame_end": 299, "display_rate": 30.0},
        "objects": {
            "Player_01": {
                "category": "player", "class_id": 0, "track_id": 1,
                "team": "A", "role": "ball_carrier",
                "keyframes": [{"frame": i, "loc": [float(-1000 + i), float(100 - i), 90.0], "yaw_deg": 0.0} for i in range(300)],
                "action_timeline": [{"start_frame": 0, "end_frame": 299, "action": "jog"}],
            },
            "Player_05": {
                "category": "player", "class_id": 0, "track_id": 5,
                "team": "B", "role": "primary_marker",
                "keyframes": [{"frame": i, "loc": [float(-800 + i), float(50 - i), 90.0], "yaw_deg": 180.0} for i in range(300)],
                "action_timeline": [{"start_frame": 0, "end_frame": 299, "action": "defend"}],
            },
            "Ball_01": {
                "category": "ball", "class_id": 1, "track_id": 101,
                "keyframes": [{"frame": i, "loc": [float(-900 + i), float(75 - i), 11.0]} for i in range(300)],
            },
        },
        "possession_timeline": [
            {"state": "owned", "owner": "Player_01", "start_frame": 0, "end_frame": 299},
        ],
        "event_timeline": [
            {"event_id": "event_001", "type": "dribble", "actor": "Player_01",
             "start_frame": 0, "end_frame_exclusive": 300, "last_frame": 299},
            {"event_id": "event_002", "type": "defend_follow", "actor": "Player_05",
             "start_frame": 0, "end_frame_exclusive": 300, "last_frame": 299, "target": "Player_01"},
        ],
        "event_frame_map": {
            "event_001": {"type": "dribble", "start_frame": 0, "end_frame_exclusive": 300, "actor": "Player_01"},
            "event_002": {"type": "defend_follow", "start_frame": 0, "end_frame_exclusive": 300, "actor": "Player_05", "target": "Player_01"},
        },
        "contact_frames": [],
        "ball_state_timeline": [
            {"start_frame": 0, "end_frame": 299, "state": "controlled", "owner": "Player_01"},
        ],
        "episode_metadata": {
            "event_count": 2, "player_count": 2,
            "object_stats": {
                "Player_01": {"keyframe_count": 300, "total_distance_xy_cm": 0.0, "max_speed_xy_cm_s": 0.0, "category": "player"},
                "Player_05": {"keyframe_count": 300, "total_distance_xy_cm": 0.0, "max_speed_xy_cm_s": 0.0, "category": "player"},
                "Ball_01": {"keyframe_count": 300, "total_distance_xy_cm": 0.0, "max_speed_xy_cm_s": 0.0, "category": "ball"},
            },
        },
        "track_id_map": {"Player_01": 1, "Player_05": 5, "Ball_01": 101},
        "class_id_map": {"player": 0, "ball": 1},
        "image": {"width": 1920, "height": 1080},
    }
    path = tmp_path / "mini_a33.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return path
