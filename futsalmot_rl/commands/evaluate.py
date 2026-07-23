"""`futsalmot-rl evaluate` — unified BC and PPO evaluation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from futsalmot_rl.core.local_config import load_local_paths
from futsalmot_rl.evaluation.evaluator import evaluate_policy, save_evaluation


def register_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("evaluate", help="Evaluate BC or RL policy")
    subs = p.add_subparsers(dest="eval_cmd")

    for algo in ("bc", "ppo"):
        sub = subs.add_parser(algo, help=f"Evaluate {algo.upper()} policy")
        sub.add_argument("--source", required=True, help="Episode JSON path")
        sub.add_argument("--model", required=True, help="Model checkpoint path")
        sub.add_argument("--output-dir", required=True, help="Output directory")
        sub.add_argument("--n-episodes", type=int, default=5, help="Number of episodes")
        sub.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
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

    try:
        policy, _, _ = load_policy(str(model))
    except Exception as exc:
        print(f"Failed to load model {model}: {exc}", file=sys.stderr)
        return 1

    env = FutsalDefenderFollowEnv(source_episode_path=str(source))

    def action_fn(obs):
        return policy.get_action(obs, deterministic=True)

    result = evaluate_policy(
        env=env,
        action_fn=action_fn,
        n_episodes=args.n_episodes,
        seed=args.seed,
        algorithm=args.eval_cmd,
        source_path=str(source.resolve()),
        model_path=str(model.resolve()),
        device=args.device,
    )

    save_evaluation(result, output_dir)
    summary = result.to_summary()

    print(f"Evaluation complete ({args.eval_cmd}): "
          f"{summary.get('mean_episode_reward', 'N/A'):.3f} ± "
          f"{summary.get('std_episode_reward', 'N/A'):.3f} "
          f"({args.n_episodes} episodes)")
    print(f"Summary: {output_dir / 'evaluation_summary.json'}")

    env.close()
    return 0 if not result.errors else 1
