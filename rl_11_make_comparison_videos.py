#!/usr/bin/env python3
"""
rl_11_make_comparison_videos.py — Create Rule / BC / PPO comparison videos.

Generates overlay comparison videos showing all three policies on the same pitch.

Usage:
    D:/Anaconda/envs/yolov11/python.exe rl_11_make_comparison_videos.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_CODE_ROOT = Path(__file__).resolve().parent
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

from futsalmot_rl.core.rl_io import read_json, write_json_atomic
from futsalmot_rl.core.rl_paths import (
    DEMOS_DIR,
    MODELS_DIR,
    RUNS_DIR,
    VIDEOS_DIR,
    ensure_dirs,
)
from futsalmot_rl.core.rl_paths import COURT_X_MAX, COURT_X_MIN, COURT_Y_MAX, COURT_Y_MIN
from futsalmot_rl.data.a33_reader import get_player_positions_2d, load_a33_config
from futsalmot_rl.envs.defender_follow_env import FutsalDefenderFollowEnv
from futsalmot_rl.features.action_builder import extract_action_from_trajectory
from futsalmot_rl.models.policy_io import load_policy


def main() -> int:
    ensure_dirs()
    # Ensure comparison_videos directory exists
    comp_dir = VIDEOS_DIR / "comparison"
    comp_dir.mkdir(parents=True, exist_ok=True)

    # Find source episode
    source_path = RUNS_DIR / "production_run" / "episode_random_0001_t1_a33.json"
    if not source_path.is_file():
        print("[ERROR] Source A3.3 not found: {}".format(source_path))
        return 1

    seq_id = "episode_random_0001_t1"
    print("=" * 60)
    print("FutsalMOT-RL Comparison Videos")
    print("Source: {}".format(source_path))
    print("=" * 60)

    # ── Load rule trajectory ────────────────────────────────────
    cfg = load_a33_config(source_path)
    all_pos = get_player_positions_2d(cfg)
    rule_positions = np.array(
        [(float(x), float(y)) for x, y in all_pos.get("Player_05", [])],
        dtype=np.float32,
    )
    target_positions = np.array(
        [(float(x), float(y)) for x, y in all_pos.get("Player_01", [])],
        dtype=np.float32,
    )
    ball_positions = np.array(
        [(float(x), float(y)) for x, y in __import__("futsalmot_rl.data.a33_reader",
                                                      fromlist=["get_ball_positions_2d"])
         .get_ball_positions_2d(cfg)],
        dtype=np.float32,
    )

    n_frames = len(rule_positions)
    print("Rule trajectory: {} frames".format(n_frames))

    # ── Load BC policy and rollout ──────────────────────────────
    bc_model = MODELS_DIR / "defender_follow_bc_v1_best.pt"
    bc_positions = None
    if bc_model.is_file():
        print("Loading BC policy...")
        bc_policy, _, _ = load_policy(str(bc_model))
        env = FutsalDefenderFollowEnv(source_episode_path=str(source_path))
        bc_pos_list: list[tuple[float, float]] = []
        obs, _ = env.reset()
        done = False
        while not done:
            action = bc_policy.get_action(obs, deterministic=True)
            obs_next, _, term, trunc, info = env.step(action)
            pos = info.get("all_positions", {}).get("Player_05", (0, 0))
            bc_pos_list.append((float(pos[0]), float(pos[1])))
            done = term or trunc
            obs = obs_next
        env.close()
        bc_positions = np.array(bc_pos_list, dtype=np.float32)
        print("BC rollout: {} frames".format(len(bc_positions)))
    else:
        print("BC model not found, skipping BC comparison")

    # ── Load PPO policy and rollout ─────────────────────────────
    ppo_model = MODELS_DIR / "defender_follow_ppo_v1_best.pt"
    ppo_positions = None
    if ppo_model.is_file():
        print("Loading PPO policy...")
        ppo_policy, _, _ = load_policy(str(ppo_model))
        env = FutsalDefenderFollowEnv(source_episode_path=str(source_path))
        ppo_pos_list: list[tuple[float, float]] = []
        obs, _ = env.reset()
        done = False
        while not done:
            action = ppo_policy.get_action(obs, deterministic=True)
            obs_next, _, term, trunc, info = env.step(action)
            pos = info.get("all_positions", {}).get("Player_05", (0, 0))
            ppo_pos_list.append((float(pos[0]), float(pos[1])))
            done = term or trunc
            obs = obs_next
        env.close()
        ppo_positions = np.array(ppo_pos_list, dtype=np.float32)
        print("PPO rollout: {} frames".format(len(ppo_positions)))
    else:
        print("PPO model not found, skipping PPO comparison")

    # ── Generate videos ─────────────────────────────────────────
    from futsalmot_rl.viz.compare_video import make_comparison_video
    from futsalmot_rl.viz.video_recorder import record_episode_video

    results = []

    # Individual policy videos
    policy_data = [
        ("rule", rule_positions, bc_model.is_file() and bc_positions is not None),
        ("bc", bc_positions if bc_positions is not None else rule_positions, bc_positions is not None),
        ("ppo_v2", ppo_positions if ppo_positions is not None else rule_positions, ppo_positions is not None),
    ]

    for name, positions, available in policy_data:
        if not available:
            continue
        video_path = comp_dir / "{}_{}.mp4".format(name, seq_id)

        # Simple individual video: show the policy with target and ball
        import imageio
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle

        fig, ax = plt.subplots(1, 1, figsize=(12, 7))
        fig.patch.set_facecolor("#0D0D0D")
        writer = imageio.get_writer(str(video_path), fps=15, codec="h264")

        colors = {"rule": "#888888", "bc": "#FF6F00", "ppo_v2": "#4CAF50"}
        titles = {"rule": "Rule Baseline", "bc": "BC Policy", "ppo_v2": "PPO v2 Policy"}

        for t in range(min(n_frames, len(positions))):
            ax.clear()
            ax.set_facecolor("#1B5E20")
            ax.set_xlim(COURT_X_MIN, COURT_X_MAX)
            ax.set_ylim(COURT_Y_MIN, COURT_Y_MAX)
            ax.set_aspect("equal")
            ax.set_title("{} — {}".format(titles[name], seq_id),
                         color="white", fontsize=14)

            rect = plt.Rectangle(
                (COURT_X_MIN, COURT_Y_MIN),
                COURT_X_MAX - COURT_X_MIN, COURT_Y_MAX - COURT_Y_MIN,
                linewidth=2, edgecolor="#BDBDBD", facecolor="none",
            )
            ax.add_patch(rect)
            ax.axvline(x=0, color="#BDBDBD", linewidth=1, linestyle="--", alpha=0.5)

            # Ball
            if t < len(ball_positions):
                ax.add_patch(Circle(
                    (float(ball_positions[t, 0]), float(ball_positions[t, 1])),
                    12, facecolor="#FF6F00", edgecolor="white", linewidth=1,
                ))

            # Target
            if t < len(target_positions):
                ax.add_patch(Circle(
                    (float(target_positions[t, 0]), float(target_positions[t, 1])),
                    40, facecolor="#2979FF", edgecolor="white", linewidth=2,
                ))
                ax.text(
                    float(target_positions[t, 0]), float(target_positions[t, 1]),
                    "T", color="white", fontsize=8, ha="center", va="center",
                    fontweight="bold",
                )

            # Player
            color = colors[name]
            ax.add_patch(Circle(
                (float(positions[t, 0]), float(positions[t, 1])),
                40, facecolor=color, edgecolor="white", linewidth=3,
            ))

            # Trail
            trail_start = max(0, t - 30)
            ax.plot(
                [float(positions[i, 0]) for i in range(trail_start, t + 1)],
                [float(positions[i, 1]) for i in range(trail_start, t + 1)],
                color=color, linewidth=2, alpha=0.5,
            )

            # Distance line
            if t < len(target_positions):
                ax.plot(
                    [float(positions[t, 0]), float(target_positions[t, 0])],
                    [float(positions[t, 1]), float(target_positions[t, 1])],
                    color=color, linewidth=1, alpha=0.4, linestyle=":",
                )

            # Info
            dist = np.hypot(
                positions[t, 0] - target_positions[t, 0],
                positions[t, 1] - target_positions[t, 1],
            ) if t < len(target_positions) else 0
            ax.text(
                0.02, 0.98,
                "Frame: {}/{}\nDist to target: {:.0f} cm\nTime: {:.1f}s".format(
                    t, min(n_frames, len(positions)) - 1, dist, t / 30
                ),
                transform=ax.transAxes, color="white", fontsize=9,
                verticalalignment="top", fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="black", alpha=0.7),
            )

            fig.canvas.draw()
            try:
                buf = fig.canvas.buffer_rgba()
                writer.append_data(np.asarray(buf)[:, :, :3])
            except AttributeError:
                buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                writer.append_data(buf.reshape(fig.canvas.get_width_height()[::-1] + (3,)))

        writer.close()
        plt.close(fig)
        results.append({"policy": name, "video": str(video_path)})
        print("  [OK] {} video: {}".format(name, video_path))

    # ── Overlay comparison video ────────────────────────────────
    if bc_positions is not None and ppo_positions is not None:
        overlay_path = comp_dir / "compare_rule_bc_ppo_{}.mp4".format(seq_id)
        result = make_comparison_video(
            rule_positions=rule_positions,
            bc_positions=bc_positions,
            ppo_positions=ppo_positions,
            target_positions=target_positions,
            ball_positions=ball_positions,
            output_path=overlay_path,
            seq_id=seq_id,
        )
        if result:
            results.append({"policy": "overlay_rule_bc_ppo", "video": result})
            print("  [OK] Overlay comparison video: {}".format(result))

    # ── Report ──────────────────────────────────────────────────
    report = {
        "schema_version": "COMPARISON_VIDEO_V1",
        "source": str(source_path),
        "seq_id": seq_id,
        "videos": results,
    }
    report_path = comp_dir / "compare_video_report.json"
    write_json_atomic(report_path, report)
    print("\nReport: {}".format(report_path))
    print("[DONE] Comparison videos generated.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
