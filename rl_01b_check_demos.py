#!/usr/bin/env python3
"""
rl_01b_check_demos.py — Validate and visualize exported demonstration data.

Checks:
1. demo_index.json exists and is well-formed
2. Each .npz is readable
3. No NaN/Inf in obs or actions
4. Action values mostly in [-1, 1]
5. Episode lengths are reasonable
6. Generate sample trajectory plot and demo video

Usage:
    D:/Anaconda/envs/yolov11/python.exe rl_01b_check_demos.py
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

_CODE_ROOT = Path(__file__).resolve().parent
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

from futsalmot_rl.core.rl_io import read_json, write_json_atomic
from futsalmot_rl.core.rl_paths import (
    COURT_X_MAX,
    COURT_X_MIN,
    COURT_Y_MAX,
    COURT_Y_MIN,
    DEMOS_DIR,
    EVAL_DIR,
    REPORTS_DIR,
    VIDEOS_DIR,
    ensure_dirs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check exported demo data integrity.")
    parser.add_argument(
        "--demo-dir",
        type=str,
        default=None,
        help="Override demo directory (default: Saved/FutsalMOT_RL/demos/)",
    )
    parser.add_argument(
        "--sample-seq-id",
        type=str,
        default=None,
        help="Specific seq_id to sample for visualization (default: first demo)",
    )
    return parser.parse_args()


def check_npz(path: Path) -> dict:
    """Check a single .npz file for integrity."""
    result = {
        "path": str(path),
        "exists": False,
        "readable": False,
        "obs_shape": None,
        "actions_shape": None,
        "has_nan_obs": False,
        "has_inf_obs": False,
        "has_nan_actions": False,
        "has_inf_actions": False,
        "actions_out_of_range_pct": 0.0,
        "n_transitions": 0,
        "episode_length": 0,
        "errors": [],
    }

    if not path.is_file():
        result["errors"].append("File does not exist")
        return result
    result["exists"] = True

    try:
        data = np.load(path, allow_pickle=True)
        result["readable"] = True
    except Exception as exc:
        result["errors"].append("Cannot load: {}".format(exc))
        return result

    # Check obs
    if "obs" in data:
        obs = data["obs"]
        result["obs_shape"] = list(obs.shape)
        result["n_transitions"] = obs.shape[0]
        result["has_nan_obs"] = bool(np.any(np.isnan(obs)))
        result["has_inf_obs"] = bool(np.any(np.isinf(obs)))

    # Check actions
    if "actions" in data:
        actions = data["actions"]
        result["actions_shape"] = list(actions.shape)
        result["has_nan_actions"] = bool(np.any(np.isnan(actions)))
        result["has_inf_actions"] = bool(np.any(np.isinf(actions)))
        if actions.size > 0:
            out_of_range = np.sum(np.abs(actions) > 1.0 + 1e-6)
            result["actions_out_of_range_pct"] = float(
                out_of_range / actions.size * 100
            )

    # Check episode length
    if "dones" in data:
        dones = data["dones"]
        result["episode_length"] = len(dones)
    elif "obs" in data:
        result["episode_length"] = data["obs"].shape[0]

    # Check positions for NaN
    for key in ("positions_rule", "target_positions", "ball_positions"):
        if key in data:
            arr = data[key]
            if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
                result["errors"].append("{} has NaN/Inf".format(key))

    # Check seq_id
    for key in ("seq_id", "agent_id"):
        if key in data and data[key].ndim == 0:
            result[key] = str(data[key])

    data.close()
    return result


def generate_sample_plot(
    npz_path: Path,
    output_png: Path,
) -> None:
    """Generate a sample trajectory plot from a demo npz file."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        data = np.load(npz_path, allow_pickle=True)
        positions = data["positions_rule"]
        target_pos = data["target_positions"]
        ball_pos = data["ball_positions"]

        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        fig.patch.set_facecolor("#0D0D0D")
        ax.set_facecolor("#1B5E20")

        # Draw court
        ax.set_xlim(COURT_X_MIN, COURT_X_MAX)
        ax.set_ylim(COURT_Y_MIN, COURT_Y_MAX)
        ax.set_aspect("equal")

        # Player_05 trajectory
        ax.plot(positions[:, 0], positions[:, 1], color="#FF1744", linewidth=2, label="Player_05 (agent)")
        # Player_01 trajectory
        ax.plot(target_pos[:, 0], target_pos[:, 1], color="#2979FF", linewidth=2, label="Player_01 (target)")
        # Ball trajectory
        ax.plot(ball_pos[:, 0], ball_pos[:, 1], color="#FF6F00", linewidth=1.5, alpha=0.7, label="Ball")

        # Start/end markers
        ax.scatter([positions[0, 0]], [positions[0, 1]], color="#FF1744", s=100, marker="o", edgecolor="white", zorder=5)
        ax.scatter([positions[-1, 0]], [positions[-1, 1]], color="#FF1744", s=100, marker="s", edgecolor="white", zorder=5)

        ax.legend(loc="upper right", fontsize=10)
        ax.set_title("Demo Trajectory: {}".format(npz_path.stem), color="white", fontsize=14)
        ax.tick_params(colors="white")
        ax.set_xlabel("X (cm)", color="white")
        ax.set_ylabel("Y (cm)", color="white")

        output_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(output_png), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("  Sample plot saved to {}".format(output_png))

        data.close()

    except Exception as exc:
        print("  [WARNING] Sample plot generation failed: {}".format(exc))


def generate_demo_video(npz_path: Path, output_path: Path) -> None:
    """Generate a demo video from a npz file using pitch_drawer."""
    try:
        from futsalmot_rl.viz.pitch_drawer import create_pitch_figure, draw_pitch_frame

        data = np.load(npz_path, allow_pickle=True)
        positions = data["positions_rule"]  # (T, 2)
        target_positions = data["target_positions"]
        ball_positions = data["ball_positions"]
        seq_id = str(data["seq_id"]) if data["seq_id"].ndim == 0 else "unknown"

        n_frames = len(positions)
        fig, ax = create_pitch_figure()

        import imageio
        import matplotlib.pyplot as _plt

        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(str(output_path), fps=15, codec="h264")

        for t in range(n_frames):
            ax.clear()
            all_positions = {
                "Player_05": (float(positions[t, 0]), float(positions[t, 1])),
                "Player_01": (float(target_positions[t, 0]), float(target_positions[t, 1])),
            }
            ball_pos = (
                float(ball_positions[t, 0]),
                float(ball_positions[t, 1]),
            ) if t < len(ball_positions) else None

            draw_pitch_frame(
                ax,
                all_positions=all_positions,
                ball_pos=ball_pos,
                agent_id="Player_05",
                target_id="Player_01",
                frame=t,
                title="Demo: {}".format(seq_id),
            )
            fig.canvas.draw()
            try:
                buf = fig.canvas.buffer_rgba()
                writer.append_data(np.asarray(buf)[:, :, :3])
            except AttributeError:
                buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                writer.append_data(buf)

        writer.close()
        _plt.close(fig)
        print("  Demo video saved to {}".format(output_path))
        data.close()

    except Exception as exc:
        print("  [WARNING] Demo video generation failed: {}".format(exc))


def main() -> int:
    args = parse_args()
    ensure_dirs()

    demo_dir = Path(args.demo_dir) if args.demo_dir else DEMOS_DIR
    demo_index_path = demo_dir / "demo_index.json"

    if not demo_index_path.is_file():
        print("[ERROR] demo_index.json not found at {}".format(demo_index_path))
        print("  Run rl_01_export_demos.py first.")
        return 1

    index = read_json(demo_index_path)
    demos = index.get("demos", [])
    print("=" * 60)
    print("FutsalMOT-RL Demo Check")
    print("Index: {}".format(demo_index_path))
    print("Total demos: {}".format(len(demos)))
    print("=" * 60)

    results: list[dict] = []
    errors = 0

    for demo in demos:
        demo_path = Path(demo["path"])
        if not demo_path.is_absolute():
            demo_path = demo_dir / demo_path.name

        seq_id = demo.get("seq_id", "unknown")
        print("\nChecking {} ({})...".format(seq_id, demo_path.name))
        result = check_npz(demo_path)
        results.append(result)

        if result["errors"]:
            errors += 1
            for err in result["errors"]:
                print("  ERROR: {}".format(err))
        else:
            print("  OK: obs={} actions={} len={}".format(
                result["obs_shape"], result["actions_shape"], result["episode_length"]
            ))
            if result["has_nan_obs"]:
                print("  WARNING: obs contains NaN")
                errors += 1
            if result["has_inf_obs"]:
                print("  WARNING: obs contains Inf")
                errors += 1
            if result["actions_out_of_range_pct"] > 5:
                print("  WARNING: {:.1f}% actions outside [-1, 1]".format(
                    result["actions_out_of_range_pct"]
                ))

    # Generate sample visualization
    sample_seq_id = args.sample_seq_id
    sample_result = None
    for r in results:
        if sample_seq_id is None or r.get("seq_id") == sample_seq_id:
            sample_result = r
            break

    if sample_result and not sample_result["errors"]:
        print("\n--- Generating sample visualization ---")
        demo_path = Path(sample_result["path"])
        if not demo_path.is_absolute():
            demo_path = Path(sample_result["path"])

        seq_id = sample_result.get("seq_id", "sample")
        plot_path = EVAL_DIR / "demo_sample_{}.png".format(seq_id)
        video_path = VIDEOS_DIR / "demos" / "demo_sample_{}.mp4".format(seq_id)

        generate_sample_plot(demo_path, plot_path)
        generate_demo_video(demo_path, video_path)

    # Summary
    report = {
        "schema_version": "RL_DEMO_CHECK_V1",
        "demo_index_path": str(demo_index_path.resolve()),
        "total_demos": len(demos),
        "total_errors": errors,
        "results": results,
    }
    report_path = REPORTS_DIR / "demo_check_report.json"
    write_json_atomic(report_path, report)
    print("\nReport: {}".format(report_path))

    if errors:
        print("[WARNING] {} issue(s) found".format(errors))
        return 1

    print("[DONE] All demos passed integrity check.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
