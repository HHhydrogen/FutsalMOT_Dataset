#!/usr/bin/env python3
"""
rl_04_train_ppo.py — Train PPO policy with BC initialization.

Loads a BC-pretrained model and fine-tunes with PPO.

Usage:
    D:/Anaconda/envs/yolov11/python.exe rl_04_train_ppo.py
    D:/Anaconda/envs/yolov11/python.exe rl_04_train_ppo.py --total-timesteps 100000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_CODE_ROOT = Path(__file__).resolve().parent
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

from futsalmot_rl.core.rl_paths import (
    MODELS_DIR,
    RUNS_DIR,
    VIDEOS_DIR,
    ensure_dirs,
    FPS,
)
from futsalmot_rl.envs.defender_follow_env import FutsalDefenderFollowEnv
from futsalmot_rl.training.callbacks import RLVideoEvalCallback
from futsalmot_rl.training.train_ppo import PPOTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO policy for defender follow.")
    parser.add_argument("--source", type=str, default=None, help="Source A3.3 config path")
    parser.add_argument(
        "--bc-model", type=str, default=None,
        help="BC model to initialize from (default: best BC model)",
    )
    parser.add_argument("--total-timesteps", type=int, default=500000, help="Total training steps")
    parser.add_argument("--learning-rate", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--n-steps", type=int, default=2048, help="Steps per rollout")
    parser.add_argument("--batch-size", type=int, default=64, help="Minibatch size")
    parser.add_argument("--eval-interval", type=int, default=25000, help="Eval video interval")
    parser.add_argument("--no-video", action="store_true", help="Disable video during training")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cpu/cuda)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_dirs()

    # Find source episode
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
        return 1

    # Find BC model
    bc_model = args.bc_model
    if bc_model is None:
        bc_candidates = [
            MODELS_DIR / "defender_follow_bc_v1_best.pt",
            MODELS_DIR / "defender_follow_bc_v1.pt",
        ]
        for c in bc_candidates:
            if c.is_file():
                bc_model = str(c)
                break

    print("=" * 60)
    print("FutsalMOT-RL PPO Training")
    print("Source:       {}".format(source_path))
    print("BC model:     {}".format(bc_model or "(none — training from scratch)"))
    print("Total steps:  {}".format(args.total_timesteps))
    print("Device:       {}".format(args.device))
    print("=" * 60)

    # Optimized reward config (stronger boundary/collision penalties)
    reward_cfg = {
        "out_of_bounds_penalty": -10.0,
        "collision_penalty": -5.0,
        "boundary_proximity_weight": -0.02,
        "boundary_proximity_margin_cm": 300.0,
        "goal_side_bonus": 0.5,
        "goal_side_penalty": -0.5,
        "acceleration_penalty": -0.002,
    }

    # Create environments
    train_env = FutsalDefenderFollowEnv(
        source_episode_path=source_path,
        reward_config=reward_cfg,
    )
    eval_env = FutsalDefenderFollowEnv(
        source_episode_path=source_path,
        reward_config=reward_cfg,
    )

    # Create trainer
    config = {
        "total_timesteps": args.total_timesteps,
        "learning_rate": args.learning_rate,
        "n_steps": args.n_steps,
        "batch_size": args.batch_size,
        "reward_config": reward_cfg,
    }

    trainer = PPOTrainer(
        env=train_env,
        eval_env=eval_env,
        config=config,
        device=args.device,
    )

    # Load BC pretrained model
    if bc_model:
        trainer.load_pretrained(bc_model)

    # Setup video callback
    if not args.no_video:
        video_callback = RLVideoEvalCallback(
            eval_env=eval_env,
            video_dir=VIDEOS_DIR / "rl_train",
            eval_freq=args.eval_interval,
            n_eval_episodes=1,
        )
        trainer.set_video_callback(video_callback)

    # Train
    summary = trainer.train(
        total_timesteps=args.total_timesteps,
        eval_interval=args.eval_interval,
    )

    # Cleanup
    train_env.close()
    eval_env.close()

    best_reward = summary.get("best_mean_reward")
    if best_reward is not None:
        print("\n[DONE] PPO training complete. Best mean reward: {:.3f}".format(best_reward))
        return 0
    else:
        print("\n[WARNING] PPO training completed but no best reward recorded.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
