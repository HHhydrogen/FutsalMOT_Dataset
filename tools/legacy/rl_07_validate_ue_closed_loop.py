#!/usr/bin/env python3
"""
rl_07_validate_ue_closed_loop.py — Prepare and verify RL A3.3 → UE rendering.

This script does NOT render in UE (that requires Unreal Editor).
It prepares the files and instructions, and later checks the UE outputs.

Usage:
    # Phase A: Prepare for UE (run this on Windows)
    python rl_07_validate_ue_closed_loop.py --prepare

    # Then follow the instructions in Saved/FutsalMOT_RL/ue_closed_loop/

    # Phase B: Check UE outputs (run AFTER UE rendering)
    python rl_07_validate_ue_closed_loop.py --check
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_CODE_ROOT = Path(__file__).resolve().parent
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

from futsalmot_rl.core.rl_paths import EXPORTED_A33_DIR
from futsalmot_rl.ue_bridge.ue_config_export import prepare_ue_config, UE_CLOSED_LOOP_DIR
from futsalmot_rl.ue_bridge.ue_closed_loop_check import check_ue_outputs, generate_layout_check_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RL A3.3 → UE closed-loop validation.")
    parser.add_argument(
        "--prepare", action="store_true",
        help="Prepare RL A3.3 for UE rendering and print instructions.",
    )
    parser.add_argument(
        "--check", action="store_true",
        help="Check UE rendering outputs (run after UE render).",
    )
    parser.add_argument(
        "--rl-a33", type=str, default=None,
        help="Path to RL A3.3 file (default: latest in exported_a33/).",
    )
    parser.add_argument(
        "--seq-id", type=str, default=None,
        help="Seq ID for checking (default: auto-detect from --rl-a33).",
    )
    return parser.parse_args()


def find_latest_rl_a33() -> Path | None:
    """Find the most recent RL A3.3 file in the export directory."""
    if not EXPORTED_A33_DIR.is_dir():
        return None
    a33_files = sorted(EXPORTED_A33_DIR.glob("*a33.json"))
    return a33_files[-1] if a33_files else None


def main() -> int:
    args = parse_args()

    if not args.prepare and not args.check:
        print("[ERROR] Use --prepare or --check")
        return 1

    if args.prepare:
        # Phase A: Prepare for UE
        rl_a33_path = args.rl_a33
        if rl_a33_path is None:
            latest = find_latest_rl_a33()
            if latest is None:
                print("[ERROR] No RL A3.3 file found. Run rl_06_export_rl_a33.py first.")
                return 1
            rl_a33_path = str(latest)

        print("=" * 60)
        print("FutsalMOT-RL UE Closed-Loop — Preparation")
        print("=" * 60)
        print("Preparing: {}".format(rl_a33_path))

        report = prepare_ue_config(rl_a33_path)

        print("\n[OK] UE preparation complete.")
        print("  RL A3.3 for UE: {}".format(report["rl_a33_for_ue"]))
        print("  Command file:   {}".format(report["ue_command_file"]))
        print("  Notes:          {}".format(report["notes_file"]))

        print("\n" + "=" * 60)
        print("NEXT STEPS — YOU MUST DO THESE IN UNREAL EDITOR:")
        print("=" * 60)
        print()
        print("▶ Step 1: Open a command prompt and run:")
        print()
        print('   set FUTSALMOT_CONFIG_PATH={}'.format(report["rl_a33_for_ue"]))
        print()
        print("   Then launch Unreal Editor from that same prompt.")
        print()
        print("▶ Step 2: In UE Python console, run:")
        print()
        print('   py "D:/projects/FustalMOT_UEDataset/Content/FutsalMOT/code/02_run_unreal.py"')
        print()
        print("▶ Step 3: Render with Movie Render Queue")
        print()
        print("▶ Step 4: After rendering, run:")
        print()
        print("   cd /d D:\\projects\\FustalMOT_UEDataset\\Content\\FutsalMOT\\code")
        print("   python 03_check_labels.py")
        print()
        print("▶ Step 5: Come back and run:")
        print()
        print("   python rl_07_validate_ue_closed_loop.py --check --seq-id <seq_id>")
        print()
        print("=" * 60)

    if args.check:
        # Phase B: Check UE outputs
        if args.seq_id:
            seq_id = args.seq_id
        elif args.rl_a33:
            # Extract from A3.3
            import json
            with open(args.rl_a33, encoding="utf-8") as f:
                data = json.load(f)
            seq_id = data.get("seq_id", data.get("episode_id", "unknown"))
        else:
            # Try to detect from exported_a33
            latest = find_latest_rl_a33()
            if latest:
                import json
                with open(latest, encoding="utf-8") as f:
                    data = json.load(f)
                seq_id = data.get("seq_id", data.get("episode_id", "rl_episode_random_0001_t1_p05"))
            else:
                seq_id = "rl_episode_random_0001_t1_p05"

        print("=" * 60)
        print("FutsalMOT-RL UE Closed-Loop — Checking UE Outputs")
        print("Seq ID: {}".format(seq_id))
        print("=" * 60)

        report = check_ue_outputs(seq_id=seq_id)

        print("\n--- Check Results ---")
        for key, value in report.get("checks", {}).items():
            print("  {}: {}".format(key, value))

        if report.get("errors"):
            print("\n[ERRORS]")
            for err in report["errors"]:
                print("  ❌ {}".format(err))

        if report.get("warnings"):
            print("\n[WARNINGS]")
            for warn in report["warnings"]:
                print("  ⚠ {}".format(warn))

        if report.get("passed"):
            print("\n[PASS] UE closed-loop verification passed!")
        else:
            print("\n[FAIL] Some checks did not pass. Review errors above.")

        print("\nReport: {}".format(UE_CLOSED_LOOP_DIR / "ue_render_check_report.json"))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
