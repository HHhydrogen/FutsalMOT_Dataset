"""Comparison video generation for FutsalMOT-RL.

Creates side-by-side or overlay videos comparing Rule / BC / PPO trajectories.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from futsalmot_rl.core.rl_io import ensure_dir
from futsalmot_rl.core.rl_paths import (
    COURT_X_MAX,
    COURT_X_MIN,
    COURT_Y_MAX,
    COURT_Y_MIN,
    FPS,
)


def make_comparison_video(
    rule_positions: np.ndarray,
    bc_positions: np.ndarray,
    ppo_positions: np.ndarray,
    target_positions: np.ndarray,
    ball_positions: np.ndarray,
    output_path: str | Path,
    seq_id: str = "comparison",
    fps: int = 15,
) -> str | None:
    """Create a comparison video showing all three policies on one pitch.

    Args:
        rule_positions: (T, 2) array of rule Player_05 positions.
        bc_positions: (T, 2) array of BC Player_05 positions.
        ppo_positions: (T, 2) array of PPO Player_05 positions.
        target_positions: (T, 2) array of Player_01 positions.
        ball_positions: (T, 2) array of Ball positions.
        output_path: Output video path (.mp4).
        seq_id: Sequence ID for labeling.
        fps: Video frame rate.

    Returns:
        Path to video, or None on failure.
    """
    import imageio
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    n_frames = min(
        len(rule_positions), len(bc_positions), len(ppo_positions), len(target_positions)
    )
    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    fig.patch.set_facecolor("#0D0D0D")

    writer = imageio.get_writer(str(output_path), fps=fps, codec="h264")

    for t in range(n_frames):
        ax.clear()

        # ── Pitch background ────────────────────────────────────
        ax.set_facecolor("#1B5E20")
        ax.set_xlim(COURT_X_MIN, COURT_X_MAX)
        ax.set_ylim(COURT_Y_MIN, COURT_Y_MAX)
        ax.set_aspect("equal")
        ax.set_title(f"Policy Comparison — {seq_id}", color="white", fontsize=14)

        # Court boundary
        rect = plt.Rectangle(
            (COURT_X_MIN, COURT_Y_MIN),
            COURT_X_MAX - COURT_X_MIN,
            COURT_Y_MAX - COURT_Y_MIN,
            linewidth=2,
            edgecolor="#BDBDBD",
            facecolor="none",
        )
        ax.add_patch(rect)
        ax.axvline(x=0, color="#BDBDBD", linewidth=1, linestyle="--", alpha=0.5)

        # ── Ball ────────────────────────────────────────────────
        if t < len(ball_positions):
            ball = Circle(
                (float(ball_positions[t, 0]), float(ball_positions[t, 1])),
                12,
                facecolor="#FF6F00",
                edgecolor="white",
                linewidth=1,
            )
            ax.add_patch(ball)

        # ── Target (Player_01) ──────────────────────────────────
        target = Circle(
            (float(target_positions[t, 0]), float(target_positions[t, 1])),
            40,
            facecolor="#2979FF",
            edgecolor="white",
            linewidth=2,
        )
        ax.add_patch(target)
        ax.text(
            float(target_positions[t, 0]),
            float(target_positions[t, 1]),
            "T",
            color="white",
            fontsize=8,
            ha="center",
            va="center",
            fontweight="bold",
        )

        # ── Trajectory trails (last 30 frames) ──────────────────
        trail_len = 30
        t_start = max(0, t - trail_len)

        for positions, color, label in [
            (rule_positions, "#888888", "Rule"),
            (bc_positions, "#FF6F00", "BC"),
            (ppo_positions, "#4CAF50", "PPO"),
        ]:
            if t < len(positions):
                trail_x = [float(positions[i, 0]) for i in range(t_start, t + 1)]
                trail_y = [float(positions[i, 1]) for i in range(t_start, t + 1)]
                ax.plot(trail_x, trail_y, color=color, linewidth=1.5, alpha=0.6, label=label)

        # ── Player positions ────────────────────────────────────
        for positions, color, _label, edge_w in [
            (rule_positions, "#888888", "Rule", 2),
            (bc_positions, "#FF6F00", "BC", 2),
            (ppo_positions, "#4CAF50", "PPO", 3),
        ]:
            if t < len(positions):
                pos = Circle(
                    (float(positions[t, 0]), float(positions[t, 1])),
                    35,
                    facecolor=color,
                    edgecolor="white",
                    linewidth=edge_w,
                )
                ax.add_patch(pos)

        # ── Distance to target lines ────────────────────────────
        for positions, color in [
            (rule_positions, "#888888"),
            (bc_positions, "#FF6F00"),
            (ppo_positions, "#4CAF50"),
        ]:
            if t < len(positions):
                ax.plot(
                    [float(positions[t, 0]), float(target_positions[t, 0])],
                    [float(positions[t, 1]), float(target_positions[t, 1])],
                    color=color,
                    linewidth=1,
                    alpha=0.3,
                    linestyle=":",
                )

        # ── Legend ──────────────────────────────────────────────
        ax.legend(
            loc="upper left", fontsize=9, facecolor="black", edgecolor="white", labelcolor="white"
        )

        # ── Info panel ──────────────────────────────────────────
        time_s = t / FPS
        info_lines = [
            f"Frame: {t}/{n_frames - 1} ({time_s:.1f}s)",
            "",
            "Distance to target:",
        ]
        for name, positions in [
            ("Rule", rule_positions),
            ("BC", bc_positions),
            ("PPO", ppo_positions),
        ]:
            if t < len(positions):
                dist = np.hypot(
                    positions[t, 0] - target_positions[t, 0],
                    positions[t, 1] - target_positions[t, 1],
                )
                info_lines.append(f"  {name}: {dist:.0f} cm")

        info_lines.append("")
        # Out of bounds check
        for name, positions in [
            ("Rule", rule_positions),
            ("BC", bc_positions),
            ("PPO", ppo_positions),
        ]:
            if t < len(positions):
                x, y = positions[t]
                oob = not (
                    COURT_X_MIN + 10 <= x <= COURT_X_MAX - 10
                    and COURT_Y_MIN + 10 <= y <= COURT_Y_MAX - 10
                )
                if oob:
                    info_lines.append(f"  {name}: OUT OF BOUNDS")

        ax.text(
            0.02,
            0.98,
            "\n".join(info_lines),
            transform=ax.transAxes,
            color="white",
            fontsize=8,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="black", alpha=0.7),
        )

        # ── Render ──────────────────────────────────────────────
        fig.canvas.draw()
        try:
            buf = fig.canvas.buffer_rgba()
            writer.append_data(np.asarray(buf)[:, :, :3])
        except AttributeError:
            buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            writer.append_data(buf.reshape((*fig.canvas.get_width_height()[::-1], 3)))

    writer.close()
    plt.close(fig)
    return str(output_path)
