"""`futsalmot-rl train` — stub delegating to tools/legacy/ scripts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from futsalmot_rl.core.paths import ProjectPaths


def register_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("train", help="Train BC or PPO (uses legacy scripts)")
    subs = p.add_subparsers(dest="train_cmd")
    bc = subs.add_parser("bc", help="Train Behavior Cloning policy")
    bc.add_argument("--epochs", type=int, default=100)
    ppo = subs.add_parser("ppo", help="Train PPO policy")
    ppo.add_argument("--total-timesteps", type=int, default=500000)


def run(args: argparse.Namespace, paths: ProjectPaths) -> int:
    legacy_dir = Path(__file__).resolve().parents[2] / "tools" / "legacy"
    sys.path.insert(0, str(legacy_dir))

    if args.train_cmd == "bc":
        import rl_02_train_bc  # type: ignore[import-untyped]

        return rl_02_train_bc.main()
    elif args.train_cmd == "ppo":
        import rl_04_train_ppo  # type: ignore[import-untyped]

        return rl_04_train_ppo.main()

    print(f"Unknown train command: {args.train_cmd}")
    return 1
