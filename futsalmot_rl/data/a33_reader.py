"""Read-only A3.3 trajectory config and event annotation parser.

All functions are pure reads — never modify source data.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from futsalmot_rl.core.rl_io import read_json
from futsalmot_rl.core.rl_paths import RUNS_DIR

# ── Run discovery ───────────────────────────────────────────────

def find_rule_runs(
    runs_dir: str | Path = RUNS_DIR,
    template_ids: list[int] | None = None,
    seeds: list[int] | None = None,
) -> list[Path]:
    """Scan runs_dir for run directories containing A3.3 configs.

    Returns a list of A3.3 file paths matching the optional filter.
    """
    runs_dir = Path(runs_dir)
    if not runs_dir.is_dir():
        return []

    candidates: list[Path] = []
    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        for a33_path in run_dir.glob("*_a33.json"):
            if a33_path.is_file() and a33_path.stat().st_size > 0:
                candidates.append(a33_path)

    if template_ids is not None or seeds is not None:
        filtered: list[Path] = []
        for path in candidates:
            cfg = load_a33_config(path)
            ep_id = cfg.get("episode_id", "")
            for tid in (template_ids or []):
                if f"_t{tid:d}" in ep_id:
                    filtered.append(path)
                    break
            else:
                if seeds is not None:
                    for seed in seeds:
                        if f"_{seed:04d}_" in ep_id:
                            filtered.append(path)
                            break
                else:
                    filtered.append(path)
        candidates = filtered

    return candidates


def find_latest_successful_run(runs_dir: str | Path = RUNS_DIR) -> Path | None:
    """Find the most recent run directory with a pipeline_run_report.json."""
    runs_dir = Path(runs_dir)
    if not runs_dir.is_dir():
        return None

    latest: tuple[float, Path] | None = None
    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        report_path = run_dir / "pipeline_run_report.json"
        if report_path.is_file() and report_path.stat().st_size > 0:
            mtime = report_path.stat().st_mtime
            if latest is None or mtime > latest[0]:
                latest = (mtime, run_dir)

    if latest is None:
        return None
    # Return the A3.3 config in the latest run
    a33_files = list(latest[1].glob("*_a33.json"))
    return a33_files[0] if a33_files else None


# ── A3.3 config parsing ─────────────────────────────────────────

def load_a33_config(path: str | Path) -> dict[str, Any]:
    """Load and validate an A3.3 trajectory config JSON."""
    data = read_json(path)
    if not isinstance(data, dict):
        raise ValueError(f"A3.3 config must be a JSON object: {path}")
    if data.get("schema_version", "").startswith("3."):
        return data
    raise ValueError("Unsupported schema version: {}".format(data.get("schema_version")))


def get_player_positions(a33_cfg: dict[str, Any]) -> dict[str, list[tuple[float, float, float]]]:
    """Extract per-player position arrays from an A3.3 config.

    Returns {player_id: [(x, y, z), ...]} — one entry per frame.
    """
    objects = a33_cfg.get("objects", {})
    result: dict[str, list[tuple[float, float, float]]] = {}
    for obj_id, obj_data in objects.items():
        if isinstance(obj_data, dict) and obj_data.get("category") == "player":
            kfs = obj_data.get("keyframes", [])
            positions: list[tuple[float, float, float]] = []
            for kf in kfs:
                loc = kf.get("loc", [0, 0, 0])
                positions.append((float(loc[0]), float(loc[1]), float(loc[2])))
            result[obj_id] = positions
    return result


def get_player_positions_2d(a33_cfg: dict[str, Any]) -> dict[str, list[tuple[float, float]]]:
    """Extract per-player (x, y) positions — dropping z."""
    full = get_player_positions(a33_cfg)
    result: dict[str, list[tuple[float, float]]] = {}
    for pid, positions in full.items():
        result[pid] = [(x, y) for x, y, _ in positions]
    return result


def get_ball_positions(a33_cfg: dict[str, Any]) -> list[tuple[float, float, float]]:
    """Extract ball position array from an A3.3 config."""
    objects = a33_cfg.get("objects", {})
    ball = objects.get("Ball_01")
    if not isinstance(ball, dict):
        return []
    kfs = ball.get("keyframes", [])
    return [(float(kf["loc"][0]), float(kf["loc"][1]), float(kf["loc"][2])) for kf in kfs]


def get_ball_positions_2d(a33_cfg: dict[str, Any]) -> list[tuple[float, float]]:
    """Extract ball (x, y) positions."""
    full = get_ball_positions(a33_cfg)
    return [(x, y) for x, y, _ in full]


def get_possession_timeline(a33_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Get the possession timeline (compressed segments)."""
    return list(a33_cfg.get("possession_timeline", []))


def get_player_yaws(a33_cfg: dict[str, Any]) -> dict[str, list[float]]:
    """Extract per-player yaw values."""
    objects = a33_cfg.get("objects", {})
    result: dict[str, list[float]] = {}
    for obj_id, obj_data in objects.items():
        if isinstance(obj_data, dict) and obj_data.get("category") == "player":
            kfs = obj_data.get("keyframes", [])
            result[obj_id] = [float(kf.get("yaw_deg", 0.0)) for kf in kfs]
    return result


def get_player_velocities(
    positions: dict[str, list[tuple[float, float, float]]],
    fps: int = 30,
) -> dict[str, list[tuple[float, float]]]:
    """Compute per-player velocities (cm/s) from positions.

    Returns {player_id: [(vx, vy), ...]} with len = T-1.
    """
    result: dict[str, list[tuple[float, float]]] = {}
    for pid, pos_list in positions.items():
        vels: list[tuple[float, float]] = []
        for i in range(len(pos_list) - 1):
            vx = (pos_list[i + 1][0] - pos_list[i][0]) * fps
            vy = (pos_list[i + 1][1] - pos_list[i][1]) * fps
            vels.append((vx, vy))
        result[pid] = vels
    return result


def get_ball_velocities(
    positions: list[tuple[float, float, float]],
    fps: int = 30,
) -> list[tuple[float, float]]:
    """Compute ball velocities from positions."""
    vels: list[tuple[float, float]] = []
    for i in range(len(positions) - 1):
        vx = (positions[i + 1][0] - positions[i][0]) * fps
        vy = (positions[i + 1][1] - positions[i][1]) * fps
        vels.append((vx, vy))
    return vels


def get_event_frame_map(a33_cfg: dict[str, Any]) -> dict[str, Any]:
    """Get the event-to-frame mapping dict."""
    return dict(a33_cfg.get("event_frame_map", {}))


def get_contact_frames(a33_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Get contact frame metadata."""
    return list(a33_cfg.get("contact_frames", []))


def get_ball_state_timeline(a33_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Get ball state segments."""
    return list(a33_cfg.get("ball_state_timeline", []))


def get_action_timeline(a33_cfg: dict[str, Any], player_id: str) -> list[dict[str, Any]]:
    """Get a specific player's action timeline."""
    obj = a33_cfg.get("objects", {}).get(player_id, {})
    if isinstance(obj, dict):
        return list(obj.get("action_timeline", []))
    return []


def get_object_stats(a33_cfg: dict[str, Any]) -> dict[str, Any]:
    """Get the object_stats dict from episode_metadata."""
    meta = a33_cfg.get("episode_metadata", {})
    return dict(meta.get("object_stats", {}))


def get_seq_id(a33_cfg: dict[str, Any]) -> str:
    """Get the seq_id from the config."""
    return str(a33_cfg.get("seq_id", "unknown"))


def get_episode_id(a33_cfg: dict[str, Any]) -> str:
    """Get the episode_id from the config."""
    return str(a33_cfg.get("episode_id", "unknown"))


# ── Event annotation parsing ────────────────────────────────────

def load_events(path: str | Path) -> list[dict[str, Any]]:
    """Load events from an events_*.json annotation file."""
    data = read_json(path)
    return list(data.get("events", []))


def load_frame_states(path: str | Path) -> list[dict[str, Any]]:
    """Load frame states from a frame_states_*.jsonl file.

    Returns a list of dicts, one per line.
    """
    path = Path(path)
    states: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                states.append(json.loads(line))
    return states
