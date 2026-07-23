"""Export RL-controlled trajectories to A3.3-compatible JSON format.

Reads a source A3.3 file, replaces Player_05's trajectory with an RL rollout,
while preserving all other objects and metadata unchanged.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

import numpy as np

from futsalmot_rl.core.rl_io import write_json_atomic
from futsalmot_rl.core.rl_paths import EXPORTED_A33_DIR, REPORTS_DIR, FPS, ensure_dirs
from futsalmot_rl.data.a33_reader import get_seq_id, load_a33_config
from futsalmot_rl.envs.defender_follow_env import FutsalDefenderFollowEnv
from futsalmot_rl.features.action_builder import PLAYER_05_MAX_SPEED_CM_S
from futsalmot_rl.rollout.policy_rollout import rollout_episode


def export_rl_a33(
    source_a33_path: str | Path,
    policy: Any,
    output_dir: str | Path | None = None,
    agent_id: str = "Player_05",
) -> dict[str, Any]:
    """Export an RL-controlled trajectory as an A3.3-compatible JSON file.

    Args:
        source_a33_path: Path to the source (rule) A3.3 config.
        policy: Policy callable taking (obs, deterministic) → action.
        output_dir: Output directory (default: Saved/FutsalMOT_RL/exported_a33/).
        agent_id: The RL-controlled agent ID.

    Returns:
        Dict with export metadata.
    """
    source_path = Path(source_a33_path)
    source_cfg = load_a33_config(source_path)
    seq_id = get_seq_id(source_cfg)

    # Create environment and rollout
    env = FutsalDefenderFollowEnv(source_episode_path=source_path, agent_id=agent_id)
    rollout_data = rollout_episode(env, policy, deterministic=True, collect_all_info=True)

    # Build output config — deep copy of source
    output_cfg = copy.deepcopy(source_cfg)
    output_seq_id = "rl_{}_p05".format(seq_id)
    output_cfg["seq_id"] = output_seq_id
    output_cfg["description"] = output_cfg.get("description", "") + " [RL Player_05]"
    output_cfg["episode_id"] = output_seq_id

    # Replace Player_05 trajectory
    rl_positions = rollout_data["agent_positions"]  # (T, 2)
    n_frames = len(rl_positions)

    # Get original Player_05 object
    objects = output_cfg.get("objects", {})
    if agent_id not in objects:
        raise ValueError("Agent {} not found in source config".format(agent_id))

    player_obj = objects[agent_id]

    # Replace keyframes
    old_keyframes = player_obj.get("keyframes", [])
    new_keyframes: list[dict[str, Any]] = []

    # Compute yaw from RL trajectory positions
    yaws = _compute_yaw_from_positions(rl_positions, fps=FPS)

    for i in range(n_frames):
        frame_data = old_keyframes[i] if i < len(old_keyframes) else {"frame": i}
        new_keyframe = {
            "frame": i,
            "loc": [
                round(float(rl_positions[i, 0]), 6),
                round(float(rl_positions[i, 1]), 6),
                float(frame_data.get("loc", [0, 0, 90.0])[2]),
            ],
            "yaw_deg": round(float(yaws[i]), 6) if i < len(yaws) else 0.0,
        }
        new_keyframes.append(new_keyframe)

    player_obj["keyframes"] = new_keyframes

    # Update action timeline to reflect RL control
    player_obj["action_timeline"] = [
        {
            "start_frame": 0,
            "end_frame": n_frames - 1,
            "action": "defend",
            "source_events": ["rl_control"],
        }
    ]
    player_obj["yaw_source"] = "rl_computed_from_trajectory"

    # Update object_stats in episode_metadata
    meta = output_cfg.get("episode_metadata", {})
    obj_stats = meta.get("object_stats", {})
    if agent_id in obj_stats:
        total_dist = 0.0
        max_speed = 0.0
        max_speed_frame = 0
        for i in range(1, n_frames):
            dx = rl_positions[i, 0] - rl_positions[i - 1, 0]
            dy = rl_positions[i, 1] - rl_positions[i - 1, 1]
            dist = math.hypot(dx, dy)
            total_dist += dist
            speed = dist * FPS
            if speed > max_speed:
                max_speed = speed
                max_speed_frame = i

        obj_stats[agent_id] = {
            "keyframe_count": n_frames,
            "total_distance_xy_cm": total_dist,
            "max_speed_xy_cm_s": max_speed,
            "max_speed_3d_cm_s": max_speed,
            "max_speed_frame": max_speed_frame,
            "category": "player",
        }
        meta["object_stats"] = obj_stats

    meta["rl_export"] = {
        "source_config": str(source_path.resolve()),
        "agent_id": agent_id,
        "policy_type": "ppo_rl_v1",
        "total_reward": float(rollout_data["total_reward"]),
        "n_frames": n_frames,
    }
    output_cfg["episode_metadata"] = meta

    # Update event_frame_map for agent events
    event_map = output_cfg.get("event_frame_map", {})
    for evt_id, evt_data in event_map.items():
        if isinstance(evt_data, dict) and evt_data.get("actor") == agent_id:
            evt_data["note"] = "rl_controlled"
    output_cfg["event_frame_map"] = event_map

    # Write output
    output_dir = Path(output_dir) if output_dir else EXPORTED_A33_DIR
    ensure_dirs()

    output_path = output_dir / "rl_{}_{}_a33.json".format(seq_id, agent_id)
    write_json_atomic(output_path, output_cfg)

    # Write export report
    report = {
        "schema_version": "RL_A33_EXPORT_V1",
        "source_config": str(source_path.resolve()),
        "output_path": str(output_path.resolve()),
        "output_seq_id": output_seq_id,
        "agent_id": agent_id,
        "n_frames": n_frames,
        "total_reward": float(rollout_data["total_reward"]),
        "total_distance_cm": float(
            obj_stats.get(agent_id, {}).get("total_distance_xy_cm", 0.0)
        ),
        "max_speed_cm_s": float(
            obj_stats.get(agent_id, {}).get("max_speed_xy_cm_s", 0.0)
        ),
    }
    report_path = output_dir / "rl_{}_{}_export_report.json".format(seq_id, agent_id)
    write_json_atomic(report_path, report)

    # Run trajectory validation
    try:
        _run_validation(output_path)
    except Exception as exc:
        report["validation_error"] = str(exc)
        write_json_atomic(report_path, report)

    env.close()
    return report


def _compute_yaw_from_positions(
    positions: np.ndarray, fps: int = 30
) -> np.ndarray:
    """Compute yaw angles from a trajectory of (x, y) positions.

    Uses central differences for smooth yaw, with forward fill for gaps.
    """
    n = len(positions)
    yaws = np.zeros(n, dtype=np.float32)

    for i in range(n):
        if n <= 1:
            yaws[i] = 0.0
        elif i == 0:
            dx = positions[1, 0] - positions[0, 0]
            dy = positions[1, 1] - positions[0, 1]
        elif i == n - 1:
            dx = positions[-1, 0] - positions[-2, 0]
            dy = positions[-1, 1] - positions[-2, 1]
        else:
            dx = positions[i + 1, 0] - positions[i - 1, 0]
            dy = positions[i + 1, 1] - positions[i - 1, 1]

        if math.hypot(dx, dy) < 1e-6:
            yaws[i] = yaws[i - 1] if i > 0 else 0.0
        else:
            yaws[i] = math.degrees(math.atan2(dy, dx))

    # Simple smoothing
    smoothed = np.copy(yaws)
    window = 3
    for i in range(1, n - 1):
        smoothed[i] = np.mean(yaws[max(0, i - window // 2): min(n, i + window // 2 + 1)])

    return smoothed


def _run_validation(a33_path: Path) -> dict[str, Any]:
    """Run trajectory validation on the exported A3.3 file.

    Uses the existing validate_trajectory.py via subprocess (read-only).
    """
    import subprocess
    import json

    from futsalmot_rl.core.rl_paths import CODE_DIR

    validator_path = CODE_DIR / "futsalmot" / "scripts" / "validate_trajectory.py"
    if not validator_path.is_file():
        return {"skipped": "validator not found"}

    result = subprocess.run(
        [
            "python",
            str(validator_path),
            "--config",
            str(a33_path),
            "--output-dir",
            str(REPORTS_DIR / "validation"),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )

    report = {
        "returncode": result.returncode,
        "stdout_tail": result.stdout[-500:],
        "stderr_tail": result.stderr[-500:],
    }
    report_path = REPORTS_DIR / "rl_a33_validation_{}.json".format(
        get_seq_id(load_a33_config(a33_path))
    )
    write_json_atomic(report_path, report)
    return report
