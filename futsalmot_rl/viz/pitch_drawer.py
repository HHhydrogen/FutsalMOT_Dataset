"""2D futsal pitch drawer for visualization and video recording."""

from __future__ import annotations

import math

import matplotlib
import matplotlib.pyplot as plt

from futsalmot_rl.core.rl_paths import COURT_X_MAX, COURT_X_MIN, COURT_Y_MAX, COURT_Y_MIN

matplotlib.use("Agg")  # Non-interactive backend for video generation

# Colors
COLOR_PITCH_BG = "#1B5E20"
COLOR_PITCH_LINE = "#BDBDBD"
COLOR_TEAM_A = "#2196F3"
COLOR_TEAM_B = "#F44336"
COLOR_PLAYER_05 = "#FF1744"
COLOR_PLAYER_01 = "#2979FF"
COLOR_BALL = "#FF6F00"
COLOR_GHOST = "#888888"
COLOR_TEXT = "#FFFFFF"
COLOR_INFO_BG = "rgba(0, 0, 0, 0.75)"

# Plot dimensions (in cm, scaled to fit)
COURT_W = COURT_X_MAX - COURT_X_MIN  # 3900
COURT_H = COURT_Y_MAX - COURT_Y_MIN  # 1900


def _setup_pitch(ax: plt.Axes, title: str = "") -> None:
    """Set up the futsal pitch background."""
    ax.set_facecolor(COLOR_PITCH_BG)
    ax.set_xlim(COURT_X_MIN, COURT_X_MAX)
    ax.set_ylim(COURT_Y_MIN, COURT_Y_MAX)
    ax.set_aspect("equal")
    ax.set_title(title, color=COLOR_TEXT, fontsize=14, pad=10)

    # Court boundary
    rect = plt.Rectangle(
        (COURT_X_MIN, COURT_Y_MIN),
        COURT_W,
        COURT_H,
        linewidth=2,
        edgecolor=COLOR_PITCH_LINE,
        facecolor="none",
    )
    ax.add_patch(rect)

    # Halfway line
    ax.axvline(x=0, color=COLOR_PITCH_LINE, linewidth=1, linestyle="--", alpha=0.5)

    # Center circle
    center_circle = plt.Circle(
        (0, 0),
        300,
        linewidth=1.5,
        edgecolor=COLOR_PITCH_LINE,
        facecolor="none",
        alpha=0.5,
    )
    ax.add_patch(center_circle)

    # Center dot
    center_dot = plt.Circle((0, 0), 15, facecolor=COLOR_PITCH_LINE, alpha=0.5)
    ax.add_patch(center_dot)

    # Goals
    goal_width = 300
    for goal_x, goal_label in [(COURT_X_MIN, "B"), (COURT_X_MAX, "A")]:
        ax.plot(
            [goal_x, goal_x],
            [-goal_width / 2, goal_width / 2],
            color=COLOR_PITCH_LINE,
            linewidth=3,
        )
        ax.text(
            goal_x + (60 if goal_x < 0 else -60),
            0,
            goal_label,
            color=COLOR_PITCH_LINE,
            fontsize=12,
            ha="center",
            va="center",
            fontweight="bold",
        )

    # Attack direction arrow
    ax.annotate(
        "",
        xy=(COURT_X_MAX - 200, 0),
        xytext=(COURT_X_MAX - 600, 0),
        arrowprops=dict(arrowstyle="->", color=COLOR_PITCH_LINE, lw=2),
    )

    ax.tick_params(colors=COLOR_PITCH_LINE, labelsize=8)
    ax.set_xlabel("X (cm)", color=COLOR_PITCH_LINE, fontsize=10)
    ax.set_ylabel("Y (cm)", color=COLOR_PITCH_LINE, fontsize=10)


def _draw_player(
    ax: plt.Axes,
    pos: tuple[float, float],
    player_id: str,
    color: str,
    *,
    size: int = 60,
    alpha: float = 1.0,
    edge_width: float = 2.0,
    show_id: bool = True,
) -> None:
    """Draw a player as a filled circle with ID label."""
    circle = plt.Circle(
        pos,
        size,
        facecolor=color,
        alpha=alpha,
        edgecolor="white",
        linewidth=edge_width,
    )
    ax.add_patch(circle)
    if show_id:
        # Short ID
        short_id = player_id.replace("Player_", "P")
        ax.text(
            pos[0],
            pos[1],
            short_id,
            color="white",
            fontsize=7,
            ha="center",
            va="center",
            fontweight="bold",
        )


def _draw_ball(ax: plt.Axes, pos: tuple[float, float], size: int = 15) -> None:
    """Draw the ball as a small filled circle."""
    circle = plt.Circle(pos, size, facecolor=COLOR_BALL, edgecolor="white", linewidth=1)
    ax.add_patch(circle)


def draw_pitch_frame(
    ax: plt.Axes,
    *,
    all_positions: dict[str, tuple[float, float]],
    ball_pos: tuple[float, float] | None = None,
    agent_id: str = "Player_05",
    target_id: str = "Player_01",
    agent_velocity: tuple[float, float] | None = None,
    agent_trail: list[tuple[float, float]] | None = None,
    ghost_positions: dict[str, tuple[float, float]] | None = None,
    frame: int = 0,
    fps: int = 30,
    reward: float | None = None,
    distance_to_target: float | None = None,
    collision: bool = False,
    out_of_bounds: bool = False,
    possession_owner: str | None = None,
    event_type: str | None = None,
    title: str = "",
) -> None:
    """Draw a single frame of the futsal pitch with all game elements.

    Args:
        ax: Matplotlib axes to draw on.
        all_positions: {player_id: (x, y)} for all 8 players.
        ball_pos: Ball (x, y) position.
        agent_id: The RL-controlled agent (highlighted).
        target_id: The target being marked (highlighted).
        agent_velocity: (vx, vy) for velocity arrow.
        agent_trail: List of recent positions for trail.
        ghost_positions: {player_id: (x, y)} for rule trajectory ghost overlay.
        frame: Current frame number.
        fps: Frames per second.
        reward: Current step reward (show if not None).
        distance_to_target: Distance to target in cm.
        collision: Whether collision is happening.
        out_of_bounds: Whether agent is out of bounds.
        possession_owner: Current ball possession owner ID.
        event_type: Current event type.
        title: Frame title overlay text.
    """
    ax.clear()
    _setup_pitch(ax, title=title)

    # Agent trail
    if agent_trail and len(agent_trail) > 1:
        trail_x = [p[0] for p in agent_trail]
        trail_y = [p[1] for p in agent_trail]
        ax.plot(trail_x, trail_y, color=COLOR_PLAYER_05, linewidth=1.5, alpha=0.5, linestyle=":")

    # Ghost positions (rule trajectory)
    if ghost_positions:
        for pid, pos in ghost_positions.items():
            if pid != agent_id:
                _draw_player(
                    ax,
                    pos,
                    pid,
                    COLOR_GHOST,
                    size=50,
                    alpha=0.3,
                    edge_width=1,
                    show_id=False,
                )
            else:
                # Ghost for the agent
                _draw_player(
                    ax,
                    pos,
                    pid,
                    COLOR_GHOST,
                    size=55,
                    alpha=0.3,
                    edge_width=1,
                    show_id=False,
                )

    # Draw all players
    for pid, pos in all_positions.items():
        if pid == agent_id:
            _draw_player(ax, pos, pid, COLOR_PLAYER_05, size=70, edge_width=3)
        elif pid == target_id:
            _draw_player(ax, pos, pid, COLOR_PLAYER_01, size=65, edge_width=3)
        elif pid in ("Player_01", "Player_02", "Player_03", "Player_04"):
            _draw_player(ax, pos, pid, COLOR_TEAM_A)
        else:
            _draw_player(ax, pos, pid, COLOR_TEAM_B)

    # Draw ball
    if ball_pos is not None:
        _draw_ball(ax, ball_pos)

    # Connection line: agent -> target
    if target_id in all_positions and agent_id in all_positions:
        apos = all_positions[agent_id]
        tpos = all_positions[target_id]
        ax.plot(
            [apos[0], tpos[0]],
            [apos[1], tpos[1]],
            color=COLOR_PLAYER_05,
            linewidth=1,
            alpha=0.6,
            linestyle="--",
        )

    # Velocity arrow
    if agent_velocity is not None and agent_id in all_positions:
        pos = all_positions[agent_id]
        vx, vy = agent_velocity
        speed = math.hypot(vx, vy)
        if speed > 5.0:  # Only draw if moving
            scale = 0.5  # visual scaling
            ax.arrow(
                pos[0],
                pos[1],
                vx * scale,
                vy * scale,
                head_width=30,
                head_length=20,
                fc=COLOR_PLAYER_05,
                ec=COLOR_PLAYER_05,
                alpha=0.8,
            )

    # Info panel (upper left)
    info_lines = [
        f"Frame: {frame}/{299} ({frame / fps:.1f}s)",
    ]
    if reward is not None:
        info_lines.append(f"Reward: {reward:.3f}")
    if distance_to_target is not None:
        info_lines.append(f"Dist to target: {distance_to_target:.0f} cm")
    if possession_owner:
        info_lines.append(f"Possession: {possession_owner}")
    if event_type:
        info_lines.append(f"Event: {event_type}")
    if collision:
        info_lines.append("⚠ COLLISION")
    if out_of_bounds:
        info_lines.append("⚠ OUT OF BOUNDS")

    info_text = "\n".join(info_lines)
    ax.text(
        0.02,
        0.98,
        info_text,
        transform=ax.transAxes,
        color=COLOR_TEXT,
        fontsize=9,
        verticalalignment="top",
        horizontalalignment="left",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="black", alpha=0.7),
    )

    # Frame count in upper right
    time_s = frame / fps
    ax.text(
        0.98,
        0.98,
        f"t={time_s:.1f}s",
        transform=ax.transAxes,
        color=COLOR_TEXT,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="right",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.5),
    )


def create_pitch_figure(figsize: tuple[int, int] = (12, 7)) -> tuple[plt.Figure, plt.Axes]:
    """Create a new figure for pitch drawing."""
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    fig.patch.set_facecolor("#0D0D0D")
    return fig, ax
