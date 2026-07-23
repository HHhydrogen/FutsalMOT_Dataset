#!/usr/bin/env python3
"""Train BC — thin wrapper.

Usage:
    python scripts/train_bc.py --demo-index <path> --model-out <path> --log-dir <path>
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
from futsalmot_rl.training.train_bc import train_bc

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo-index", required=True)
    parser.add_argument("--model-out", required=True)
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--video-dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    summary = train_bc(
        demo_index_path=args.demo_index,
        model_out=args.model_out,
        log_dir=args.log_dir,
        video_dir=args.video_dir,
        config={"epochs": args.epochs, "batch_size": args.batch_size},
        device=args.device,
    )
    print(f"BC training complete. Best loss: {summary.get('best_val_loss', 'N/A')}")
