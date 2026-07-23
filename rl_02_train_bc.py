#!/usr/bin/env python3
"""
rl_02_train_bc.py — Train Behavior Cloning policy from demonstration data.

Loads demos from Saved/FutsalMOT_RL/demos/demo_index.json and trains an MLP
policy using MSE loss against expert actions.

Usage:
    D:/Anaconda/envs/yolov11/python.exe rl_02_train_bc.py
    D:/Anaconda/envs/yolov11/python.exe rl_02_train_bc.py --epochs 50 --batch-size 256
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

_CODE_ROOT = Path(__file__).resolve().parent
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

from futsalmot_rl.core.rl_paths import DEMOS_DIR, MODELS_DIR, VIDEOS_DIR, ensure_dirs
from futsalmot_rl.training.train_bc import train_bc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Behavior Cloning policy.")
    parser.add_argument("--demo-index", type=str, default=None, help="Path to demo_index.json")
    parser.add_argument("--model-out", type=str, default=None, help="Output model path")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=512, help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=0.0003, help="Learning rate")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cpu/cuda)")
    parser.add_argument("--no-video", action="store_true", help="Disable video during training")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_dirs()

    demo_index = Path(args.demo_index) if args.demo_index else DEMOS_DIR / "demo_index.json"
    model_out = Path(args.model_out) if args.model_out else MODELS_DIR / "defender_follow_bc_v1.pt"

    if not demo_index.is_file():
        print("[ERROR] Demo index not found: {}".format(demo_index))
        print("  Run rl_01_export_demos.py first.")
        return 1

    # Video callback for training
    video_callback = None
    if not args.no_video:
        from futsalmot_rl.evaluation.evaluate_policy import make_bc_video_callback

        video_callback = make_bc_video_callback(demo_index.parent)

    config: dict[str, Any] = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
    }

    print("=" * 60)
    print("FutsalMOT-RL Behavior Cloning Training")
    print("Demo index: {}".format(demo_index))
    print("Model out:  {}".format(model_out))
    print("Epochs: {}  Batch size: {}  LR: {}".format(
        args.epochs, args.batch_size, args.learning_rate
    ))
    print("=" * 60)

    summary = train_bc(
        demo_index_path=demo_index,
        model_out=model_out,
        config=config,
        device=args.device,
        video_callback=video_callback,
    )

    if summary["best_val_loss"] is not None:
        print("\n[DONE] BC training complete. Best val loss: {:.6f}".format(
            summary["best_val_loss"]
        ))
        return 0
    else:
        print("\n[ERROR] BC training failed.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
