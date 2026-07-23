"""`futsalmot-rl train` — train BC or PPO policy."""

from __future__ import annotations

import argparse

from futsalmot_rl.core.paths import ProjectPaths


def register_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("train", help="Train BC or PPO policy")
    subs = p.add_subparsers(dest="train_cmd")

    bc = subs.add_parser("bc", help="Train Behavior Cloning policy")
    bc.add_argument("--epochs", type=int, default=100)
    bc.add_argument("--batch-size", type=int, default=512)
    bc.add_argument("--learning-rate", type=float, default=0.0003)
    bc.add_argument("--device", type=str, default="auto")
    bc.add_argument("--no-video", action="store_true")

    ppo = subs.add_parser("ppo", help="Train PPO policy")
    ppo.add_argument("--total-timesteps", type=int, default=500000)
    ppo.add_argument("--learning-rate", type=float, default=0.0001)
    ppo.add_argument("--n-steps", type=int, default=2048)
    ppo.add_argument("--batch-size", type=int, default=64)
    ppo.add_argument("--eval-interval", type=int, default=25000)
    ppo.add_argument("--device", type=str, default="auto")
    ppo.add_argument("--no-video", action="store_true")


def run(args: argparse.Namespace, paths: ProjectPaths) -> int:
    import sys

    from futsalmot_rl.core.rl_paths import MODELS_DIR as OLD_MODELS_DIR
    sys.path.insert(0, str(OLD_MODELS_DIR.parent))

    if args.train_cmd == "bc":
        from rl_02_train_bc import main as bc_main
        return bc_main(args) if hasattr(args, 'epochs') else bc_main()

    elif args.train_cmd == "ppo":
        from rl_04_train_ppo import main as ppo_main
        # HACK: PPO训练涉及大量环境交互，当前阶段保留对旧入口的兼容
        return ppo_main()

    print(f"Unknown train command: {args.train_cmd}")
    return 1
