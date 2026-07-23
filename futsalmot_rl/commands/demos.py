"""`futsalmot-rl demos` — export and check demonstration data."""

from __future__ import annotations

import argparse
from pathlib import Path

from futsalmot_rl.core.paths import ProjectPaths
from futsalmot_rl.core.rl_io import write_json_atomic
from futsalmot_rl.data.a33_reader import find_rule_runs
from futsalmot_rl.data.demo_exporter import build_demo_index, export_demo_from_a33


def register_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("demos", help="Demo data export and check")
    subs = p.add_subparsers(dest="demos_cmd")

    ex = subs.add_parser("export", help="Export demonstration data from rule A3.3 trajectories")
    ex.add_argument("--max-episodes", type=int, default=None, help="Limit number of episodes")
    ex.add_argument("--agent", type=str, default="Player_05")
    ex.add_argument("--target", type=str, default="Player_01")

    ch = subs.add_parser("check", help="Check exported demo integrity")
    ch.add_argument("--sample-seq-id", type=str, default=None)


def run(args: argparse.Namespace, paths: ProjectPaths) -> int:
    import numpy as np

    from futsalmot_rl.core.rl_io import read_json

    if args.demos_cmd == "export":
        a33_paths = find_rule_runs(paths.runs_dir)
        if args.max_episodes:
            a33_paths = a33_paths[: args.max_episodes]

        entries = []
        for p in a33_paths:
            entry = export_demo_from_a33(p, agent_id=args.agent, target_id=args.target, output_dir=paths.demos_dir)
            entries.append(entry)

        index = build_demo_index(entries, paths.demos_dir)
        report = {"total": len(entries), "transitions": index.get("total_transitions", 0)}
        write_json_atomic(paths.reports_dir / "demo_export_report.json", report)
        print("Exported {} demos, {} transitions".format(report["total"], report["transitions"]))
        return 0

    elif args.demos_cmd == "check":
        index_path = paths.demos_dir / "demo_index.json"
        if not index_path.is_file():
            print("No demo_index.json found. Run 'futsalmot-rl demos export' first.")
            return 1

        index = read_json(index_path)
        demos = index.get("demos", [])
        errors = 0
        for d in demos:
            dp = Path(d["path"]) if Path(d["path"]).is_absolute() else paths.demos_dir / Path(d["path"]).name
            if not dp.is_file():
                print(f"Missing: {dp}")
                errors += 1
                continue
            data = np.load(str(dp), allow_pickle=True)
            obs = data["obs"]
            if np.any(np.isnan(obs)) or np.any(np.isinf(obs)):
                print(f"NaN/Inf in obs: {dp}")
                errors += 1
        print(f"Checked {len(demos)} demos, {errors} errors")
        return 1 if errors else 0

    print(f"Unknown demos command: {args.demos_cmd}")
    return 1
