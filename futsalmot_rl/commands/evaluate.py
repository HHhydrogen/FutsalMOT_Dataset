"""`futsalmot-rl evaluate` — unified BC and PPO evaluation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from futsalmot_rl.evaluation.evaluator import evaluate_policy, save_evaluation


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("n-episodes must be at least 1")
    return parsed


def register_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("evaluate", help="Evaluate BC or RL policy")
    subs = p.add_subparsers(dest="eval_cmd", required=True)

    for algo in ("bc", "ppo"):
        sub = subs.add_parser(algo, help=f"Evaluate {algo.upper()} policy")
        sub.add_argument("--source", required=True, help="Episode JSON path")
        sub.add_argument("--model", required=True, help="Model checkpoint path")
        sub.add_argument("--output-dir", required=True, help="Output directory")
        sub.add_argument("--n-episodes", type=positive_int, default=5, help="Number of episodes (>=1)")
        sub.add_argument("--device", type=str, default="cpu", help='Device (cpu/cuda)')
        sub.add_argument("--seed", type=int, default=42, help="Base RNG seed")


def run(args: argparse.Namespace, project_root: str) -> int:
    from futsalmot_rl.models.policy_io import load_policy
    from futsalmot_rl.envs.defender_follow_env import FutsalDefenderFollowEnv

    source = Path(args.source)
    model = Path(args.model)
    output_dir = Path(args.output_dir)

    if not source.is_file():
        print(f"Source not found: {source}", file=sys.stderr)
        return 1
    if not model.is_file():
        print(f"Model not found: {model}", file=sys.stderr)
        return 1

    # Device resolution
    try:
        device = torch.device(args.device)
    except (TypeError, RuntimeError) as exc:
        print(f"Invalid device '{args.device}': {exc}", file=sys.stderr)
        return 2
    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA was requested but is not available.", file=sys.stderr)
        return 2

    try:
        policy, _, _ = load_policy(str(model), device=device)
    except Exception as exc:
        print(f"Failed to load model {model}: {exc}", file=sys.stderr)
        return 1

    env = FutsalDefenderFollowEnv(source_episode_path=str(source))

    def action_fn(obs):
        return policy.get_action(obs, deterministic=True)

    try:
        result = evaluate_policy(
            env=env,
            action_fn=action_fn,
            n_episodes=args.n_episodes,
            seed=args.seed,
            algorithm=args.eval_cmd,
            source_path=str(source.resolve()),
            model_path=str(model.resolve()),
            device=str(device),
        )

        save_evaluation(result, output_dir)
        summary = result.to_summary()

        mean_reward = summary.get("mean_episode_reward")
        std_reward = summary.get("std_episode_reward")
        reward_text = f"{mean_reward:.3f}" if mean_reward is not None else "N/A"
        std_text = f"{std_reward:.3f}" if std_reward is not None else "N/A"
        completed = summary.get("completed_episode_count", 0)
        requested = summary.get("requested_episode_count", args.n_episodes)

        print(
            f"Evaluation complete ({args.eval_cmd}): "
            f"{reward_text} ± {std_text} ({completed}/{requested} completed)"
        )
        print(f"Summary: {output_dir / 'evaluation_summary.json'}")

        if result.errors:
            for error in result.errors:
                msg = error.get("message", str(error)) if isinstance(error, dict) else str(error)
                print(f"Evaluation error: {msg}", file=sys.stderr)

        return 0 if not result.errors else 1
    finally:
        env.close()
