"""FutsalMOT-RL unified command-line interface.

Usage:
    futsalmot-rl --project-root /path demos export
    futsalmot-rl train bc --epochs 50
    futsalmot-rl train ppo --total-timesteps 500000
    futsalmot-rl evaluate bc
    futsalmot-rl evaluate sanitize-env
    futsalmot-rl benchmark build
    futsalmot-rl export a33
"""

from __future__ import annotations

import argparse
import sys

from futsalmot_rl.commands import demos, evaluate
from futsalmot_rl.core.exceptions import ConfigurationError
from futsalmot_rl.core.paths import create_paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="FutsalMOT-RL: Reinforcement learning for futsal player control.",
    )
    parser.add_argument(
        "--project-root", type=str, default=None,
        help="UE project root (default: auto-detect via .uproject or env var)",
    )
    sub = parser.add_subparsers(dest="command")

    demos.register_parser(sub)

    # train (stub — full migration deferred to after PPO fixes)
    from futsalmot_rl.commands.train import register_parser as reg_train
    reg_train(sub)

    evaluate.register_parser(sub)

    # benchmark (stub)
    p = sub.add_parser("benchmark", help="Build benchmark tables")
    p.add_argument("cmd", nargs="?", default="build", choices=["build"])

    # export (stub)
    p = sub.add_parser("export", help="Export utilities")
    p.add_argument("cmd", nargs="?", default="a33", choices=["a33", "results"])

    # ue (stub)
    p = sub.add_parser("ue", help="UE closed-loop operations")
    p.add_argument("cmd", nargs="?", default="verify", choices=["prepare", "verify"])

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Resolve project root before any command runs
    try:
        paths = create_paths(project_root=args.project_root)
        paths.ensure_all()
    except ConfigurationError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    dispatch = {
        "demos": lambda: demos.run(args, paths),
        "train": lambda: _stub_run("train", args),
        "evaluate": lambda: evaluate.run(args, paths),
        "benchmark": lambda: _stub_run("benchmark", args),
        "export": lambda: _stub_run("export", args),
        "ue": lambda: _stub_run("ue", args),
    }

    fn = dispatch.get(args.command)
    if fn is None:
        parser.print_help()
        return 1

    return fn()


def _stub_run(name: str, args: argparse.Namespace) -> int:
    """Temporary dispatcher for commands not yet fully migrated."""
    cmd_map = {
        "train": "futsalmot-rl train bc/ppo",
        "benchmark": "futsalmot-rl benchmark build",
        "export": "futsalmot-rl export a33",
        "ue": "futsalmot-rl ue verify",
    }
    print(f"[INFO] '{name}' command not fully migrated yet.")
    print("       Use legacy script directly: python rl_*.py")
    print("       Or run: {}".format(cmd_map.get(name, "")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
