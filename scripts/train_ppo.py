#!/usr/bin/env python3
"""Train PPO — thin wrapper.

Usage:
    python scripts/train_ppo.py --source <episode.json> --model-dir <dir> --log-dir <dir> --total-timesteps 2048
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
from futsalmot_rl.envs.defender_follow_env import FutsalDefenderFollowEnv
from futsalmot_rl.training.train_ppo import PPOTrainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--total-timesteps", type=int, default=2048)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--run-name", type=str, default="ppo")
    parser.add_argument("--bc-model", type=str, default=None)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    log_dir = Path(args.log_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    env = FutsalDefenderFollowEnv(source_episode_path=args.source)
    trainer = PPOTrainer(env, config={
        "total_timesteps": args.total_timesteps,
        "n_steps": args.n_steps,
        "batch_size": args.batch_size,
        "n_epochs": args.n_epochs,
        "learning_rate": args.lr,
    }, device=args.device)

    if args.bc_model:
        trainer.load_pretrained(args.bc_model)

    summary = trainer.train(
        total_timesteps=args.total_timesteps,
        log_dir=log_dir,
        model_dir=model_dir,
        run_name=args.run_name,
    )

    print(f"PPO training complete. Best reward: {summary.get('best_mean_reward', 'N/A')}")
