"""FutsalMOT-RL CLI — only commands with real implementations."""

from __future__ import annotations

import argparse
import sys

from futsalmot_rl.commands import demos, evaluate
from futsalmot_rl.core.local_config import load_local_paths


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="FutsalMOT-RL")
    p.add_argument("--project-root", type=str, default=None, help="Override UE project root")
    sub = p.add_subparsers(dest="command")
    demos.register_parser(sub)
    evaluate.register_parser(sub)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    cfg = load_local_paths()
    project_root = args.project_root or cfg.get("ue_project_root") or ""

    dispatch = {
        "demos": lambda: demos.run(args, project_root),
        "evaluate": lambda: evaluate.run(args, project_root),
    }

    fn = dispatch.get(args.command)
    if fn is None:
        print(f"Unknown command: {args.command}", file=sys.stderr)
        return 2

    return fn()


if __name__ == "__main__":
    raise SystemExit(main())
