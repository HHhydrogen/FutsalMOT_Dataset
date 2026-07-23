#!/usr/bin/env python3
"""
rl_09_ablation_ppo_from_scratch.py — Train PPO from scratch (no BC init).

Compares training from scratch vs BC-initialized PPO.
Uses the same reward v2 config and training settings.

Usage:
    D:/Anaconda/envs/yolov11/python.exe rl_09_ablation_ppo_from_scratch.py
    D:/Anaconda/envs/yolov11/python.exe rl_09_ablation_ppo_from_scratch.py --total-timesteps 100000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_CODE_ROOT = Path(__file__).resolve().parent
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

from futsalmot_rl.benchmark.ablation_runner import run_ablation_ppo_scratch
from futsalmot_rl.core.rl_paths import ABLATIONS_DIR, RUNS_DIR, ensure_dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PPO from scratch ablation experiment.")
    parser.add_argument(
        "--source", type=str, default=None,
        help="Source A3.3 file (default: production_run episode)",
    )
    parser.add_argument(
        "--total-timesteps", type=int, default=500000,
        help="Total training steps (default: 500000)",
    )
    parser.add_argument(
        "--eval-interval", type=int, default=25000,
        help="Eval video interval (default: 25000)",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Device (auto/cpu/cuda)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_dirs()

    # Find source
    source = args.source
    if source is None:
        candidates = [RUNS_DIR / "production_run" / "episode_random_0001_t1_a33.json"]
        for c in candidates:
            if c.is_file():
                source = str(c)
                break

    if source is None or not Path(source).is_file():
        print("[ERROR] Source A3.3 not found.")
        return 1

    print("=" * 60)
    print("FutsalMOT-RL Ablation: PPO from Scratch")
    print("=" * 60)
    print("This will train PPO for {} timesteps from random init.".format(args.total_timesteps))
    print("Estimated time: ~{} minutes".format(args.total_timesteps // 500000 * 11))
    print()

    summary = run_ablation_ppo_scratch(
        source_path=source,
        output_dir=ABLATIONS_DIR,
        total_timesteps=args.total_timesteps,
        eval_interval=args.eval_interval,
        device=args.device,
    )

    best_reward = summary.get("best_mean_reward")
    if best_reward is not None:
        print("\n[DONE] Ablation complete. Best mean reward: {:.3f}".format(best_reward))
        return 0
    else:
        print("\n[DONE] Ablation complete.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
