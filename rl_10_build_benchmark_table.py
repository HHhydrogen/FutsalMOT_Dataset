#!/usr/bin/env python3
"""
rl_10_build_benchmark_table.py — Build standard benchmark table (Rule / BC / PPO).

Evaluates all available policies on multiple episodes and outputs
CSV, JSON, and Markdown summary tables.

Usage:
    D:/Anaconda/envs/yolov11/python.exe rl_10_build_benchmark_table.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_CODE_ROOT = Path(__file__).resolve().parent
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

from futsalmot_rl.benchmark.benchmark_runner import benchmark_policy, benchmark_rule
from futsalmot_rl.benchmark.report_writer import (
    write_benchmark_csv,
    write_benchmark_json,
    write_benchmark_summary_md,
    write_paper_table_csv,
)
from futsalmot_rl.core.rl_paths import (
    BENCHMARK_DIR,
    MODELS_DIR,
    RUNS_DIR,
    ensure_dirs,
)
from futsalmot_rl.data.a33_reader import find_rule_runs
from futsalmot_rl.models.policy_io import load_policy


def main() -> int:
    ensure_dirs()

    print("=" * 60)
    print("FutsalMOT-RL Benchmark")
    print("=" * 60)

    # Find source episodes
    source_paths = find_rule_runs()
    if not source_paths:
        print("[ERROR] No source episodes found.")
        return 1

    source_paths = source_paths[:5]  # Use first 5
    print("Source episodes: {}".format(len(source_paths)))

    all_results = []

    # ── Rule baseline ───────────────────────────────────────────
    print("\n--- Rule Baseline ---")
    rule_results = benchmark_rule(source_paths, n_episodes=5)
    all_results.extend(rule_results)

    # ── BC policy ───────────────────────────────────────────────
    bc_model_path = MODELS_DIR / "defender_follow_bc_v1_best.pt"
    if bc_model_path.is_file():
        print("\n--- BC Policy ---")
        bc_policy, _, _ = load_policy(str(bc_model_path))
        bc_results = benchmark_policy(
            lambda obs, **kw: bc_policy.get_action(obs, deterministic=True),
            source_paths,
            "bc",
            n_episodes=5,
        )
        all_results.extend(bc_results)

    # ── PPO v2 policy ──────────────────────────────────────────
    ppo_model_path = MODELS_DIR / "defender_follow_ppo_v1_best.pt"
    if ppo_model_path.is_file():
        print("\n--- PPO v2 Policy ---")
        ppo_policy, _, _ = load_policy(str(ppo_model_path))
        ppo_results = benchmark_policy(
            lambda obs, **kw: ppo_policy.get_action(obs, deterministic=True),
            source_paths,
            "ppo_v2",
            n_episodes=5,
        )
        all_results.extend(ppo_results)

    # ── Write outputs ───────────────────────────────────────────
    benchmark_dir = Path(BENCHMARK_DIR) if BENCHMARK_DIR.exists() else Path(str(BENCHMARK_DIR))
    print("\n--- Writing Reports ---")

    csv_path = benchmark_dir / "benchmark_rule_bc_ppo.csv"
    write_benchmark_csv(all_results, csv_path)
    print("CSV: {}".format(csv_path))

    json_path = benchmark_dir / "benchmark_rule_bc_ppo.json"
    write_benchmark_json(all_results, json_path)
    print("JSON: {}".format(json_path))

    md_path = benchmark_dir / "benchmark_summary.md"
    write_benchmark_summary_md(all_results, md_path)
    print("Markdown: {}".format(md_path))

    paper_csv = benchmark_dir / "benchmark_table_for_paper.csv"
    write_paper_table_csv(all_results, paper_csv)
    print("Paper table: {}".format(paper_csv))

    print("\n[DONE] Benchmark complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
