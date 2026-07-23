#!/usr/bin/env python3
"""
rl_03_eval_bc.py — Evaluate a trained BC policy on test data.

Usage:
    D:/Anaconda/envs/yolov11/python.exe rl_03_eval_bc.py
    D:/Anaconda/envs/yolov11/python.exe rl_03_eval_bc.py --model path/to/model.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

_CODE_ROOT = Path(__file__).resolve().parent
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

from futsalmot_rl.core.rl_io import write_json_atomic
from futsalmot_rl.core.rl_paths import (
    DEMOS_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    VIDEOS_DIR,
    ensure_dirs,
)
from futsalmot_rl.evaluation.evaluate_policy import evaluate_bc_policy
from futsalmot_rl.models.policy_io import load_policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate BC policy.")
    parser.add_argument(
        "--model", type=str, default=None,
        help="Path to BC model (default: defender_follow_bc_v1_best.pt)",
    )
    parser.add_argument(
        "--demo-index", type=str, default=None,
        help="Path to demo_index.json (default: Saved/FutsalMOT_RL/demos/demo_index.json)",
    )
    parser.add_argument(
        "--max-episodes", type=int, default=5,
        help="Max test episodes to evaluate",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Device (auto/cpu/cuda)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_dirs()

    # Default model path: best BC model
    model_path = (
        Path(args.model)
        if args.model
        else MODELS_DIR / "defender_follow_bc_v1_best.pt"
    )
    demo_index = (
        Path(args.demo_index)
        if args.demo_index
        else DEMOS_DIR / "demo_index.json"
    )

    if not model_path.is_file():
        print("[ERROR] Model not found: {}".format(model_path))
        return 1
    if not demo_index.is_file():
        print("[ERROR] Demo index not found: {}".format(demo_index))
        return 1

    device_str = args.device
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    print("=" * 60)
    print("FutsalMOT-RL BC Evaluation")
    print("Model:      {}".format(model_path))
    print("Demo index: {}".format(demo_index))
    print("Device:     {}".format(device))
    print("=" * 60)

    # Load model
    policy, config, _ = load_policy(model_path, device=device)
    print("Model loaded: obs_dim={} act_dim={}".format(
        getattr(policy, "_obs_dim", getattr(policy, "obs_dim", "?")),
        getattr(policy, "_act_dim", getattr(policy, "act_dim", "?")),
    ))

    # Evaluate
    metrics = evaluate_bc_policy(
        policy,
        demo_index,
        device=device,
        max_episodes=args.max_episodes,
    )

    print("\n--- BC Evaluation Results ---")
    print("  Test action MSE:      {:.6f}".format(metrics["test_action_mse"]))
    print("  Mean position error:  {:.2f} cm".format(metrics["mean_position_error_cm"]))
    print("  Samples evaluated:    {}".format(metrics["n_samples"]))
    print("  Episodes evaluated:   {}".format(metrics["n_episodes"]))

    # Save report
    report_path = REPORTS_DIR / "bc_eval_report.json"
    write_json_atomic(report_path, metrics)
    print("\nReport: {}".format(report_path))

    # Generate final BC video
    try:
        from futsalmot_rl.evaluation.evaluate_policy import make_bc_video_callback

        callback = make_bc_video_callback(demo_index.parent)
        callback(policy, 0, VIDEOS_DIR / "bc")
        # Rename the epoch_0000 to final
        import shutil

        for f in (VIDEOS_DIR / "bc").glob("bc_epoch_0000_*.mp4"):
            final_name = str(f).replace("bc_epoch_0000_", "bc_final_")
            shutil.move(str(f), final_name)
            print("  Final BC video: {}".format(final_name))
    except Exception as exc:
        print("  [WARNING] Final video generation failed: {}".format(exc))

    # Check acceptance criteria
    if metrics["test_action_mse"] < 0.1:
        print("\n[PASS] BC evaluation: MSE={:.4f} < 0.1".format(metrics["test_action_mse"]))
    else:
        print("\n[INFO] BC evaluation: MSE={:.4f} (consider more training)".format(
            metrics["test_action_mse"]
        ))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
