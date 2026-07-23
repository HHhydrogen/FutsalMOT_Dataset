#!/usr/bin/env python3
"""
rl_01_export_demos.py — Export demonstration data from rule A3.3 trajectories.

Scans configs/runs/ for A3.3 configs and exports Player_05 observation-action
pairs as .npz files to Saved/FutsalMOT_RL/demos/.

Usage:
    D:/Anaconda/envs/yolov11/python.exe rl_01_export_demos.py
    D:/Anaconda/envs/yolov11/python.exe rl_01_export_demos.py --source-dir path/to/runs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

# Ensure code/ is on sys.path
_CODE_ROOT = Path(__file__).resolve().parent
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

from futsalmot_rl.core.rl_io import write_json_atomic
from futsalmot_rl.core.rl_paths import (
    DEMOS_DIR,
    REPORTS_DIR,
    RUNS_DIR,
    ensure_dirs,
)
from futsalmot_rl.data.a33_reader import find_rule_runs
from futsalmot_rl.data.demo_exporter import build_demo_index, export_demo_from_a33


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export rule demonstration data for RL/IL training.")
    parser.add_argument(
        "--source-dir",
        type=str,
        default=None,
        help="Override source runs directory (default: configs/runs/)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory (default: Saved/FutsalMOT_RL/demos/)",
    )
    parser.add_argument(
        "--agent",
        type=str,
        default="Player_05",
        help="Agent to extract demonstrations for (default: Player_05)",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="Player_01",
        help="Target that the agent is marking (default: Player_01)",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Maximum number of episodes to export (default: all found)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_dirs()

    source_dir = Path(args.source_dir) if args.source_dir else RUNS_DIR
    output_dir = Path(args.output_dir) if args.output_dir else DEMOS_DIR

    print("=" * 60)
    print("FutsalMOT-RL Demo Export")
    print("Source: {}".format(source_dir))
    print("Output: {}".format(output_dir))
    print("Agent: {}  Target: {}".format(args.agent, args.target))
    print("=" * 60)

    # Find all available A3.3 configs
    a33_paths = find_rule_runs(source_dir)
    if not a33_paths:
        print("[ERROR] No A3.3 configs found in {}".format(source_dir))
        return 1

    print("Found {} A3.3 config(s)".format(len(a33_paths)))

    if args.max_episodes is not None:
        a33_paths = a33_paths[: args.max_episodes]

    entries: list[dict[str, Any]] = []
    errors: list[tuple[str, str]] = []

    for path in a33_paths:
        try:
            print("  Exporting: {} ...".format(path.name), end=" ")
            entry = export_demo_from_a33(
                path,
                agent_id=args.agent,
                target_id=args.target,
                output_dir=output_dir,
            )
            entries.append(entry)
            print("OK ({} transitions)".format(entry["transitions"]))
        except Exception as exc:
            msg = "{}: {}".format(type(exc).__name__, exc)
            print("FAILED: {}".format(msg))
            errors.append((str(path), msg))

    # Build demo index
    index = build_demo_index(entries, output_dir)
    print("\nDemo index: {}/demo_index.json".format(output_dir))
    print("Total episodes: {}  Total transitions: {}".format(
        index["total_episodes"], index["total_transitions"]
    ))

    # Write export report
    report = {
        "schema_version": "RL_DEMO_EXPORT_V1",
        "source_dir": str(source_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "agent_id": args.agent,
        "target_id": args.target,
        "total_found": len(a33_paths),
        "total_exported": len(entries),
        "total_errors": len(errors),
        "total_transitions": index["total_transitions"],
        "entries": entries,
        "errors": errors,
    }
    report_path = REPORTS_DIR / "demo_export_report.json"
    write_json_atomic(report_path, report)
    print("Export report: {}".format(report_path))

    if errors:
        print("\n[WARNING] {} export(s) failed".format(len(errors)))
        return 1

    print("\n[DONE] Demo export complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
