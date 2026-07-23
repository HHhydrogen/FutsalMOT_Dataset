"""`futsalmot-rl demos` — export and check demonstration data."""

from __future__ import annotations

import argparse
from pathlib import Path

from futsalmot_rl.core.local_config import get_repo_root
from futsalmot_rl.core.rl_io import write_json_atomic
from futsalmot_rl.data.demo_exporter import build_demo_index, export_demo_from_a33
from futsalmot_rl.data.a33_reader import find_rule_runs


def _resolve_output_base(project_root: str) -> Path:
    if project_root:
        return Path(project_root) / "Saved" / "FutsalMOT_RL"
    return get_repo_root().parent.parent.parent / "Saved" / "FutsalMOT_RL"


def register_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("demos", help="Demo data export and check")
    subs = p.add_subparsers(dest="demos_cmd")
    ex = subs.add_parser("export", help="Export demo data")
    ex.add_argument("--max-episodes", type=int, default=None)
    ex.add_argument("--agent", type=str, default="Player_05")
    ex.add_argument("--target", type=str, default="Player_01")
    ch = subs.add_parser("check", help="Check demo integrity")
    ch.add_argument("--sample-seq-id", type=str, default=None)


def run(args: argparse.Namespace, project_root: str) -> int:
    import numpy as np

    out_base = _resolve_output_base(project_root)
    demos_dir = out_base / "demos"
    reports_dir = out_base / "reports"

    if args.demos_cmd == "export":
        runs_dir = get_repo_root() / "configs" / "runs"
        a33_paths = find_rule_runs(runs_dir)
        if args.max_episodes:
            a33_paths = a33_paths[: args.max_episodes]
        entries = []
        for p in a33_paths:
            entry = export_demo_from_a33(p, agent_id=args.agent, target_id=args.target, output_dir=demos_dir)
            entries.append(entry)
        index = build_demo_index(entries, demos_dir)
        report = {"total": len(entries), "transitions": index.get("total_transitions", 0)}
        write_json_atomic(reports_dir / "demo_export_report.json", report)
        print(f"Exported {report['total']} demos, {report['transitions']} transitions")
        return 0

    if args.demos_cmd == "check":
        index_path = demos_dir / "demo_index.json"
        if not index_path.is_file():
            print("No demo_index.json found")
            return 1
        from futsalmot_rl.core.rl_io import read_json
        index = read_json(index_path)
        errors = 0
        for d in index.get("demos", []):
            dp = Path(d["path"]) if Path(d["path"]).is_absolute() else demos_dir / Path(d["path"]).name
            if not dp.is_file():
                print(f"Missing: {dp}")
                errors += 1
                continue
            data = np.load(str(dp), allow_pickle=True)
            if np.any(np.isnan(data["obs"])) or np.any(np.isinf(data["obs"])):
                print(f"NaN/Inf in obs: {dp}")
                errors += 1
        print(f"Checked {len(index.get('demos', []))} demos, {errors} errors")
        return 1 if errors else 0

    print(f"Unknown demos command: {args.demos_cmd}")
    return 1
