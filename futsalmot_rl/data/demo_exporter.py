"""Export rule-based A3.3 trajectories as demonstration data for imitation learning."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from futsalmot_rl.core.rl_io import save_npz, write_json_atomic
from futsalmot_rl.core.rl_paths import DEMOS_DIR, FPS
from futsalmot_rl.data.a33_reader import (
    get_ball_positions_2d,
    get_ball_velocities,
    get_event_frame_map,
    get_player_positions_2d,
    get_player_velocities,
    get_player_yaws,
    get_seq_id,
    load_a33_config,
)
from futsalmot_rl.features.action_builder import PLAYER_05_MAX_SPEED_CM_S, extract_all_actions
from futsalmot_rl.features.obs_builder import build_observation


def export_demo_from_a33(
    a33_path: str | Path,
    agent_id: str = "Player_05",
    target_id: str = "Player_01",
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Export a single demo episode from an A3.3 config file.

    Returns a dict with metadata about the export.
    """
    a33_path = Path(a33_path)
    cfg = load_a33_config(a33_path)
    seq_id = get_seq_id(cfg)

    # Extract positions (2D)
    all_positions_2d = get_player_positions_2d(cfg)
    ball_positions_2d = get_ball_positions_2d(cfg)

    # Get agent and target positions
    agent_positions = all_positions_2d.get(agent_id)
    target_positions = all_positions_2d.get(target_id)
    if agent_positions is None or target_positions is None:
        raise ValueError(f"Missing {agent_id} or {target_id} in config {a33_path}")

    # Get yaw for the agent
    all_yaws = get_player_yaws(cfg)
    agent_yaw = all_yaws.get(agent_id, [0.0] * len(agent_positions))

    # Get velocities
    all_positions_3d = {pid: [(x, y, 0.0) for x, y in pts] for pid, pts in all_positions_2d.items()}
    all_velocities = get_player_velocities(all_positions_3d, FPS)
    ball_velocities = get_ball_velocities(
        [(x, y, 0.0) for x, y in ball_positions_2d], FPS
    )

    agent_velocities = all_velocities.get(agent_id, [(0.0, 0.0)] * (len(agent_positions) - 1))
    target_velocities = all_velocities.get(target_id, [(0.0, 0.0)] * (len(target_positions) - 1))

    # Build possession timeline lookup
    possession_segments = cfg.get("possession_timeline", [])
    possession_by_frame: dict[int, str | None] = {}
    for seg in possession_segments:
        start = int(seg.get("start_frame", 0))
        end = int(seg.get("end_frame", 0))
        owner = seg.get("owner")
        for f in range(start, end + 1):
            possession_by_frame[f] = owner

    # Event lookup per frame (from event_frame_map)
    event_map = get_event_frame_map(cfg)
    # Build a reverse lookup: frame -> event types
    frame_event_type: dict[int, str] = {}
    for evt_id, evt_data in event_map.items():
        evt_type = evt_data.get("type", "")
        actor = evt_data.get("actor", "")
        start = int(evt_data.get("start_frame", 0))
        end_exc = int(evt_data.get("end_frame_exclusive", 0))
        if actor == agent_id:
            for f in range(start, end_exc):
                frame_event_type[f] = evt_type

    total_frames = len(agent_positions)
    n_transitions = max(0, total_frames - 1)
    from futsalmot_rl.features.obs_builder import OBS_DIM as OBS_DIM_VAL

    obs_dim = OBS_DIM_VAL

    obs = np.zeros((n_transitions, obs_dim), dtype=np.float32)
    actions = np.zeros((n_transitions, 2), dtype=np.float32)
    next_obs = np.zeros((n_transitions, obs_dim), dtype=np.float32)
    dones = np.zeros(n_transitions, dtype=bool)
    frames = np.arange(n_transitions, dtype=np.int32)

    # Pre-compute all actions from rule trajectory
    rule_actions = extract_all_actions(agent_positions, FPS, PLAYER_05_MAX_SPEED_CM_S)

    for t in range(n_transitions):
        # Observation at frame t
        owner_t = possession_by_frame.get(t)
        evt_type_t = frame_event_type.get(t)

        obs[t] = build_observation(
            self_pos=agent_positions[t],
            self_vel=agent_velocities[t] if t < len(agent_velocities) else (0.0, 0.0),
            self_yaw_deg=agent_yaw[t],
            target_pos=target_positions[t],
            target_vel=target_velocities[t] if t < len(target_velocities) else (0.0, 0.0),
            ball_pos=ball_positions_2d[t] if t < len(ball_positions_2d) else (0.0, 0.0),
            ball_vel=ball_velocities[t] if t < len(ball_velocities) else (0.0, 0.0),
            possession_owner=owner_t,
            current_event_type=evt_type_t,
            steps_left=total_frames - t,
            total_steps=total_frames,
        )

        action_t = rule_actions[t] if t < len(rule_actions) else np.zeros(2, dtype=np.float32)
        actions[t] = action_t

        # Next observation at frame t+1
        owner_t1 = possession_by_frame.get(t + 1)
        evt_type_t1 = frame_event_type.get(t + 1)

        next_obs[t] = build_observation(
            self_pos=agent_positions[t + 1],
            self_vel=agent_velocities[t] if t < len(agent_velocities) else (0.0, 0.0),
            self_yaw_deg=agent_yaw[t + 1],
            target_pos=target_positions[t + 1],
            target_vel=target_velocities[t] if t < len(target_velocities) else (0.0, 0.0),
            ball_pos=ball_positions_2d[t + 1] if t + 1 < len(ball_positions_2d) else (0.0, 0.0),
            ball_vel=ball_velocities[t] if t < len(ball_velocities) else (0.0, 0.0),
            possession_owner=owner_t1,
            current_event_type=evt_type_t1,
            steps_left=total_frames - t - 1,
            total_steps=total_frames,
        )

    dones[n_transitions - 1] = True

    # Save demo
    output_dir = Path(output_dir) if output_dir else DEMOS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    demo_filename = f"demo_{agent_id}_{seq_id}.npz"
    demo_path = output_dir / demo_filename

    save_npz(
        demo_path,
        obs=obs,
        actions=actions,
        next_obs=next_obs,
        dones=dones,
        frames=frames,
        positions_rule=np.array(agent_positions, dtype=np.float32),
        target_positions=np.array(target_positions, dtype=np.float32),
        ball_positions=np.array(ball_positions_2d, dtype=np.float32),
        seq_id=np.array(seq_id),
        agent_id=np.array(agent_id),
    )

    return {
        "seq_id": seq_id,
        "agent_id": agent_id,
        "target_id": target_id,
        "demo_path": str(demo_path.resolve()),
        "frames": total_frames,
        "transitions": n_transitions,
        "source_config": str(a33_path.resolve()),
    }


def build_demo_index(
    demo_entries: list[dict[str, Any]],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Build a demo_index.json from a list of demo export entries."""
    index = {
        "schema_version": "RL_DEMO_INDEX_V1",
        "agent_id": "Player_05",
        "target_id": "Player_01",
        "demos": [],
        "total_episodes": 0,
        "total_transitions": 0,
    }

    for entry in demo_entries:
        index["demos"].append({
            "seq_id": entry["seq_id"],
            "path": entry["demo_path"],
            "frames": entry["transitions"],
            "source_run": entry["source_config"],
        })

    index["total_episodes"] = len(demo_entries)
    index["total_transitions"] = sum(d["frames"] for d in index["demos"])

    index_path = Path(output_dir) / "demo_index.json"
    write_json_atomic(index_path, index)

    return index
