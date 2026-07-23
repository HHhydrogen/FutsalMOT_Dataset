#!/usr/bin/env python3
"""
rl_06_export_rl_a33.py — Export RL-controlled trajectory as A3.3-compatible JSON.

Loads a trained PPO policy, rolls it out in the environment, merges the
RL-controlled Player_05 trajectory into a copy of the source A3.3 config,
and saves the result to Saved/FutsalMOT_RL/exported_a33/.

Usage:
    D:/Anaconda/envs/yolov11/python.exe rl_06_export_rl_a33.py
    D:/Anaconda/envs/yolov11/python.exe rl_06_export_rl_a33.py --model path/to/ppo_best.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

_CODE_ROOT = Path(__file__).resolve().parent
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

from futsalmot_rl.core.rl_paths import MODELS_DIR, REPORTS_DIR, RUNS_DIR, ensure_dirs
from futsalmot_rl.models.policy_io import load_policy
from futsalmot_rl.rollout.export_to_a33 import export_rl_a33


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export RL-controlled trajectory as A3.3-compatible JSON."
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Path to PPO model (default: defender_follow_ppo_v1_best.pt)",
    )
    parser.add_argument(
        "--source", type=str, default=None,
        help="Source A3.3 config path (default: production_run episode)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: Saved/FutsalMOT_RL/exported_a33/)",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Device (auto/cpu/cuda)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_dirs()

    # Find model
    model_path = (
        Path(args.model)
        if args.model
        else MODELS_DIR / "defender_follow_ppo_v1_best.pt"
    )
    if not model_path.is_file():
        print("[ERROR] PPO model not found: {}".format(model_path))
        print("  Try: --model path/to/your/model.pt")
        print("  Or run rl_04_train_ppo.py first.")
        return 1

    # Find source
    source_path = args.source
    if source_path is None:
        candidates = [
            RUNS_DIR / "production_run" / "episode_random_0001_t1_a33.json",
        ]
        for c in candidates:
            if c.is_file():
                source_path = str(c)
                break

    if source_path is None or not Path(source_path).is_file():
        print("[ERROR] No source A3.3 file found.")
        print("  Try: --source path/to/episode_a33.json")
        return 1

    device_str = args.device
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    print("=" * 60)
    print("FutsalMOT-RL A3.3 Export")
    print("Model:  {}".format(model_path))
    print("Source: {}".format(source_path))
    print("Device: {}".format(device))
    print("=" * 60)

    # Load model
    policy, cfg, _ = load_policy(model_path, device=device)
    policy.eval()

    # Define policy callable
    def policy_fn(obs, deterministic=True):
        return policy.get_action(obs, deterministic=deterministic)

    # Export
    print("\nRolling out policy and exporting...")
    report = export_rl_a33(
        source_a33_path=source_path,
        policy=policy_fn,
        output_dir=args.output_dir,
        agent_id="Player_05",
    )

    print("\n--- Export Report ---")
    print("  Seq ID:     {}".format(report.get("output_seq_id", "unknown")))
    print("  Output:     {}".format(report.get("output_path", "unknown")))
    print("  Frames:     {}".format(report.get("n_frames", 0)))
    print("  Total dist: {:.1f} cm".format(report.get("total_distance_cm", 0.0)))
    print("  Max speed:  {:.1f} cm/s".format(report.get("max_speed_cm_s", 0.0)))
    print("  Reward:     {:.3f}".format(report.get("total_reward", 0.0)))

    if "validation_error" in report:
        print("  [WARNING] Validation error: {}".format(report["validation_error"]))
    else:
        print("  Validation: OK (see report for details)")

    print("\n[DONE] RL A3.3 export complete.")
    print("  Do NOT update pipeline_current.json with this file.")
    print("  To render in UE: FUTSALMOT_CONFIG_PATH={}".format(
        report.get("output_path", "?")
    ))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
