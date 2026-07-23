"""FutsalMOT-RL unified CLI.

Only commands with real implementations are registered.
Stubs that cannot execute are omitted — use legacy scripts in tools/legacy/.
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
        "--project-root",
        type=str,
        default=None,
        help="UE project root (default: auto-detect via .uproject or env var)",
    )
    sub = parser.add_subparsers(dest="command")

    demos.register_parser(sub)
    evaluate.register_parser(sub)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    # Only resolve project paths for commands that need them
    try:
        paths = create_paths(project_root=args.project_root)
    except ConfigurationError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    dispatch = {
        "demos": lambda: demos.run(args, paths),
        "evaluate": lambda: evaluate.run(args, paths),
    }

    fn = dispatch.get(args.command)
    if fn is None:
        print(f"Unknown command: {args.command}", file=sys.stderr)
        return 2

    return fn()


if __name__ == "__main__":
    raise SystemExit(main())
