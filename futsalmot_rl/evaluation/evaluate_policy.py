"""Policy evaluation utilities for FutsalMOT-RL."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from futsalmot_rl.core.rl_io import read_json
from futsalmot_rl.features.action_builder import PLAYER_05_MAX_SPEED_CM_S, apply_motion_constraints


def evaluate_bc_policy(
    policy: Any,
    demo_index_path: str | Path,
    device: str = "cpu",
    max_episodes: int = 5,
) -> dict[str, Any]:
    """Evaluate a BC policy on held-out test episodes.

    Args:
        policy: An MLPPolicy or MLPActorCritic instance.
        demo_index_path: Path to demo_index.json.
        device: Device for inference.
        max_episodes: Max test episodes to evaluate.

    Returns:
        Dict of evaluation metrics.
    """
    from futsalmot_rl.data.demo_dataset import DemoDataset

    if isinstance(device, str):
        device = torch.device(device)

    test_dataset = DemoDataset(demo_index_path, split="test")
    policy.eval()

    total_mse = 0.0
    total_position_error = 0.0
    n_samples = 0

    # Sample a few episodes for position error
    episode_infos = test_dataset.get_episode_info()[:max_episodes]

    for ep_info in episode_infos:
        start = ep_info["start"]
        end = ep_info["end"]

        for i in range(start, min(end, start + 299)):  # limit per episode
            obs = test_dataset.obs[i]
            true_action = test_dataset.actions[i]

            with torch.no_grad():
                obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(device)
                pred_action = policy(obs_tensor).squeeze(0).cpu().numpy()

            total_mse += float(np.mean((pred_action - true_action) ** 2))

            # Estimate position error from action difference
            true_vel = np.array(
                [
                    true_action[0] * PLAYER_05_MAX_SPEED_CM_S,
                    true_action[1] * PLAYER_05_MAX_SPEED_CM_S,
                ]
            )
            pred_vel = np.array(
                [
                    pred_action[0] * PLAYER_05_MAX_SPEED_CM_S,
                    pred_action[1] * PLAYER_05_MAX_SPEED_CM_S,
                ]
            )
            # Position error over one frame (1/30 s)
            pos_error = np.linalg.norm(true_vel - pred_vel) / 30.0
            total_position_error += pos_error
            n_samples += 1

    avg_mse = total_mse / max(1, n_samples)
    avg_pos_error = total_position_error / max(1, n_samples)

    return {
        "test_action_mse": float(avg_mse),
        "mean_position_error_cm": float(avg_pos_error),
        "n_samples": n_samples,
        "n_episodes": len(episode_infos),
    }


def make_bc_video_callback(demo_dir: str | Path):
    """Create a video callback for BC training.

    Returns a callable that records a video of the BC policy at a given epoch.
    """

    def callback(policy, epoch, video_output_dir):
        """Record a video using the BC policy."""
        try:
            import imageio
            import matplotlib.pyplot as _plt_for_close
            import numpy as _np_for_cb

            from futsalmot_rl.viz.pitch_drawer import create_pitch_figure, draw_pitch_frame

            # Find a demo file to use for replay
            demo_index_path = Path(demo_dir) / "demo_index.json"
            if not demo_index_path.is_file():
                print("    [video] No demo_index.json found")
                return

            index = read_json(demo_index_path)
            demos = index.get("demos", [])
            if not demos:
                return

            # Use the first demo as the eval episode
            demo_entry = demos[0]
            seq_id = demo_entry.get("seq_id", "unknown")
            demo_path = Path(demo_entry["path"])
            if not demo_path.is_absolute():
                demo_path = Path(demo_dir) / demo_path.name

            if not demo_path.is_file():
                return

            data = _np_for_cb.load(str(demo_path), allow_pickle=True)

            n_frames = len(data["positions_rule"])
            fig, ax = create_pitch_figure()

            output_path = Path(video_output_dir) / f"bc_epoch_{epoch:04d}_{seq_id}.mp4"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            writer = imageio.get_writer(str(output_path), fps=15, codec="h264")

            # Run BC policy for each frame
            obs_sequence = data["obs"]
            positions_rule = data["positions_rule"]
            target_positions = data["target_positions"]
            ball_positions = data["ball_positions"]

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            policy.to(device)
            policy.eval()

            for t in range(n_frames - 1):
                obs = obs_sequence[t]
                with torch.no_grad():
                    obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(device)
                    if hasattr(policy, "get_action"):
                        action = policy.get_action(obs, deterministic=True)
                    else:
                        action = policy(obs_tensor).squeeze(0).cpu().numpy()

                # Apply motion constraints to get velocity
                if t == 0:
                    vel = (0.0, 0.0)
                vel = apply_motion_constraints(action, vel, PLAYER_05_MAX_SPEED_CM_S)

                # Get predicted position
                pred_pos = (
                    positions_rule[t, 0] + vel[0] / 30.0,
                    positions_rule[t, 1] + vel[1] / 30.0,
                )

                all_positions = {
                    "Player_05": (float(pred_pos[0]), float(pred_pos[1])),
                    "Player_01": (float(target_positions[t, 0]), float(target_positions[t, 1])),
                }
                # Add approximate positions for other players (use rule as ghost)
                ghost_positions = {
                    "Player_02": (
                        float(target_positions[t, 0]) + 200,
                        float(target_positions[t, 1]) + 100,
                    ),
                    "Player_03": (
                        float(target_positions[t, 0]) + 150,
                        float(target_positions[t, 1]) - 200,
                    ),
                    "Player_04": (
                        float(target_positions[t, 0]) - 300,
                        float(target_positions[t, 1]),
                    ),
                    "Player_06": (
                        float(target_positions[t, 0]) + 300,
                        float(target_positions[t, 1]) + 200,
                    ),
                    "Player_07": (
                        float(target_positions[t, 0]) + 250,
                        float(target_positions[t, 1]) - 100,
                    ),
                    "Player_08": (
                        float(target_positions[t, 0]) + 450,
                        float(target_positions[t, 1]),
                    ),
                }
                ball_pos = (
                    (
                        float(ball_positions[t, 0]),
                        float(ball_positions[t, 1]),
                    )
                    if t < len(ball_positions)
                    else None
                )

                dist_to_target = math.hypot(
                    pred_pos[0] - float(target_positions[t, 0]),
                    pred_pos[1] - float(target_positions[t, 1]),
                )

                ax.clear()
                draw_pitch_frame(
                    ax,
                    all_positions=all_positions,
                    ball_pos=ball_pos,
                    agent_id="Player_05",
                    target_id="Player_01",
                    agent_velocity=vel,
                    ghost_positions=ghost_positions,
                    frame=t,
                    distance_to_target=dist_to_target,
                    title=f"BC Epoch {epoch} - {seq_id}",
                )
                fig.canvas.draw()
                try:
                    buf = fig.canvas.buffer_rgba()
                    writer.append_data(_np_for_cb.asarray(buf)[:, :, :3])
                except AttributeError:
                    buf = _np_for_cb.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                    writer.append_data(buf)

            writer.close()
            _plt_for_close.close(fig)
            # data.close() may trigger a benign NameError in some Python versions
            try:
                data.close()
            except Exception:
                pass
            print(f"    [video] Saved {output_path}")

        except Exception as exc:
            print(f"    [video] Failed: {exc}")

    return callback
