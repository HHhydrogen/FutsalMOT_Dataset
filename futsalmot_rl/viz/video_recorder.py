"""Video recording utilities for FutsalMOT-RL 2D visualization."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from futsalmot_rl.core.rl_io import ensure_dir

matplotlib.use("Agg")

# Try to import imageio for video writing
try:
    import imageio

    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False


def _fig_to_rgb(fig: matplotlib.figure.Figure) -> np.ndarray:
    """Convert a matplotlib figure to an (H, W, 3) uint8 RGB array.

    Works across matplotlib versions (buffer_rgba for mpl>=3.8).
    """
    fig.canvas.draw()
    try:
        # matplotlib >= 3.8
        buf = fig.canvas.buffer_rgba()
        arr = np.asarray(buf)
        return arr[:, :, :3].copy()  # drop alpha
    except AttributeError:
        # fallback for older versions
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        return buf.reshape((*fig.canvas.get_width_height()[::-1], 3))


def _extract_frames(
    frame_callback: Callable[[matplotlib.axes.Axes, int], None],
    n_frames: int,
    figsize: tuple[int, int] = (12, 7),
    fps: int = 15,
) -> list[np.ndarray]:
    """Render frames via a callback and return as RGB arrays.

    The callback is called as callback(ax, frame_index) for each frame.
    Returns a list of (H, W, 3) uint8 arrays.
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    fig.patch.set_facecolor("#0D0D0D")

    frames: list[np.ndarray] = []
    for i in range(n_frames):
        frame_callback(ax, i)
        frames.append(_fig_to_rgb(fig))

    plt.close(fig)
    return frames


def record_video_from_callback(
    frame_callback: Callable[[matplotlib.axes.Axes, int], None],
    n_frames: int,
    output_path: str | Path,
    fps: int = 15,
    figsize: tuple[int, int] = (12, 7),
) -> str | None:
    """Record a video from a frame-drawing callback.

    The callback is called as callback(ax, frame_index) for each frame.
    Draws onto the provided axes; must NOT call ax.clear() if cumulative
    drawing is intended. For fresh drawing each frame, clear inside callback.

    Falls back to saving PNG frames if video writing fails.
    """
    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    try:
        frames = _extract_frames(frame_callback, n_frames, figsize, fps)

        if HAS_IMAGEIO:
            writer = imageio.get_writer(str(output_path), fps=fps, codec="h264")
            for frame in frames:
                writer.append_data(frame)
            writer.close()
            return str(output_path)
        else:
            # Fallback: save PNG frames
            fallback_dir = output_path.parent / f"{output_path.stem}_frames"
            ensure_dir(fallback_dir)
            for i, frame in enumerate(frames):
                import imageio as iio

                iio.imwrite(str(fallback_dir / f"frame_{i:04d}.png"), frame)
            return str(fallback_dir)

    except Exception as exc:
        print(f"[WARNING] Video recording failed ({exc}), trying fallback...")
        try:
            frames = _extract_frames(frame_callback, n_frames, figsize, fps)
            fallback_dir = output_path.parent / f"{output_path.stem}_frames"
            ensure_dir(fallback_dir)
            for i, frame in enumerate(frames):
                import imageio as iio

                iio.imwrite(str(fallback_dir / f"frame_{i:04d}.png"), frame)
            return str(fallback_dir)
        except Exception as fallback_exc:
            print(f"[ERROR] Fallback also failed: {fallback_exc}")
            return None


def record_episode_video(
    env: Any,
    policy: Any,
    output_path: str | Path,
    fps: int = 15,
    figsize: tuple[int, int] = (12, 7),
    title: str = "",
    render_every_n: int = 1,
) -> str | None:
    """Record a video of an episode by stepping through an environment.

    Args:
        env: Gymnasium-like environment with render() and step().
        policy: Callable taking (obs, deterministic=True) returning actions.
        output_path: Output video path.
        fps: Video frame rate (default 15).
        figsize: Figure size.
        title: Title overlay text.
        render_every_n: Record every Nth frame for faster videos.

    Returns:
        Path to the video file, or None on failure.
    """
    from futsalmot_rl.viz.pitch_drawer import create_pitch_figure, draw_pitch_frame

    fig, ax = create_pitch_figure(figsize=figsize)
    fig.patch.set_facecolor("#0D0D0D")

    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    # Reset environment
    obs, info = env.reset()

    frames: list[np.ndarray] = []
    done = False
    step = 0
    total_reward = 0.0

    while not done:
        # Get action from policy
        action = policy(obs, deterministic=True)
        next_obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        done = terminated or truncated

        if step % render_every_n == 0:
            ax.clear()
            draw_pitch_frame(
                ax,
                all_positions=info.get("all_positions", {}),
                ball_pos=info.get("ball_pos"),
                agent_id=env.agent_id if hasattr(env, "agent_id") else "Player_05",
                target_id=env.target_id if hasattr(env, "target_id") else "Player_01",
                agent_velocity=info.get("agent_velocity"),
                agent_trail=info.get("agent_trail"),
                ghost_positions=info.get("ghost_positions"),
                frame=step,
                fps=env.fps if hasattr(env, "fps") else 30,
                reward=float(reward),
                distance_to_target=info.get("distance_to_target"),
                collision=info.get("collision", False),
                out_of_bounds=info.get("out_of_bounds", False),
                possession_owner=info.get("possession_owner"),
                event_type=info.get("event_type"),
                title=title,
            )
            frames.append(_fig_to_rgb(fig))

        obs = next_obs
        step += 1

    plt.close(fig)

    # Write video
    try:
        if HAS_IMAGEIO:
            writer = imageio.get_writer(str(output_path), fps=fps, codec="h264")
            for frame in frames:
                writer.append_data(frame)
            writer.close()
            return str(output_path)
        else:
            fallback_dir = output_path.parent / f"{output_path.stem}_frames"
            ensure_dir(fallback_dir)
            import imageio as iio

            for i, frame in enumerate(frames):
                iio.imwrite(str(fallback_dir / f"frame_{i:04d}.png"), frame)
            return str(fallback_dir)
    except Exception as exc:
        print(f"[WARNING] Video recording failed: {exc}")
        return None
