"""Write benchmark results to CSV, JSON, and Markdown."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from futsalmot_rl.core.rl_io import write_json_atomic, write_text_atomic

METRIC_LABELS = {
    "mean_marking_distance_cm": "Mean Marking Distance (cm)",
    "std_marking_distance_cm": "Std Marking Distance (cm)",
    "goal_side_success_rate": "Goal-side Success Rate",
    "time_behind_attacker_ratio": "Time Behind Attacker Ratio",
    "out_of_bounds_count": "Out of Bounds",
    "collision_count": "Collisions",
    "max_speed_cm_s": "Max Speed (cm/s)",
    "mean_speed_cm_s": "Mean Speed (cm/s)",
    "speed_warning_count": "Speed Warnings",
    "turn_angle_warning_count": "Turn Angle Warnings",
    "total_distance_cm": "Total Distance (cm)",
    "min_player_distance_cm": "Min Player Distance (cm)",
}


def write_benchmark_csv(
    results: list[dict[str, Any]],
    output_path: str | Path,
) -> None:
    """Write benchmark results as CSV."""
    if not results:
        return
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(results[0].keys())
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)


def write_benchmark_json(
    results: list[dict[str, Any]],
    output_path: str | Path,
) -> None:
    """Write benchmark results as JSON."""
    write_json_atomic(output_path, results)


def write_benchmark_summary_md(
    results: list[dict[str, Any]],
    output_path: str | Path,
) -> None:
    """Write a Markdown summary table grouped by policy type."""
    if not results:
        return
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Group by policy_type
    by_policy: dict[str, list[dict[str, Any]]] = {}
    for r in results:
        ptype = r.get("policy_type", "unknown")
        by_policy.setdefault(ptype, []).append(r)

    lines: list[str] = []
    lines.append("# FutsalMOT-RL Benchmark Summary\n")
    lines.append("*Generated: {}*\n".format(__import__("datetime").datetime.now().isoformat()))
    lines.append("")

    # Summary per policy (average across episodes)
    lines.append("## Per-Policy Summary (mean ± std)\n")
    lines.append("| Metric | {} |".format(" | ".join(by_policy.keys())))
    lines.append("|{}|".format("|".join("---" for _ in by_policy)))

    key_metrics = [
        "mean_marking_distance_cm",
        "std_marking_distance_cm",
        "goal_side_success_rate",
        "time_behind_attacker_ratio",
        "out_of_bounds_count",
        "collision_count",
        "max_speed_cm_s",
        "mean_speed_cm_s",
        "total_distance_cm",
        "min_player_distance_cm",
    ]

    for key in key_metrics:
        label = METRIC_LABELS.get(key, key)
        values = []
        for ptype in by_policy:
            vals = [r.get(key, 0) or 0 for r in by_policy[ptype]]
            mean = sum(vals) / max(1, len(vals))
            if len(vals) > 1:
                std = (sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5
                values.append(f"{mean:.2f} ± {std:.2f}")
            else:
                values.append(f"{mean:.2f}")
        lines.append("| {} | {} |".format(label, " | ".join(values)))

    lines.append("")
    lines.append("## Per-Episode Details\n")
    lines.append(
        "| seq_id | template | seed | policy | marking_dist | oob | coll | goal_side | max_speed |"
    )
    lines.append(
        "|--------|----------|------|--------|-------------|-----|------|-----------|----------|"
    )

    for r in sorted(results, key=lambda x: (x.get("policy_type", ""), x.get("seq_id", ""))):
        lines.append(
            "| {} | {} | {} | {} | {:.1f} | {} | {} | {:.2f} | {:.1f} |".format(
                r.get("seq_id", "?"),
                r.get("template_id", "?"),
                r.get("seed", "?"),
                r.get("policy_type", "?"),
                r.get("mean_marking_distance_cm", 0),
                r.get("out_of_bounds_count", 0),
                r.get("collision_count", 0),
                r.get("goal_side_success_rate", 0),
                r.get("max_speed_cm_s", 0),
            )
        )

    lines.append("")
    lines.append("---")
    lines.append("*Metrics computed by FutsalMOT-RL benchmark module.*")

    write_text_atomic(output_path, "\n".join(lines))


def write_paper_table_csv(
    results: list[dict[str, Any]],
    output_path: str | Path,
) -> None:
    """Write a paper-ready CSV (one row per policy, averaged metrics)."""
    if not results:
        return
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    by_policy: dict[str, list[dict[str, Any]]] = {}
    for r in results:
        ptype = r.get("policy_type", "unknown")
        by_policy.setdefault(ptype, []).append(r)

    paper_keys = [
        "mean_marking_distance_cm",
        "std_marking_distance_cm",
        "goal_side_success_rate",
        "out_of_bounds_count",
        "collision_count",
        "max_speed_cm_s",
        "total_distance_cm",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["Policy"] + [METRIC_LABELS.get(k, k) for k in paper_keys]
        writer.writerow(header)
        for ptype in sorted(by_policy.keys()):
            vals = by_policy[ptype]
            row = [ptype]
            for key in paper_keys:
                nums = [v.get(key, 0) or 0 for v in vals]
                mean = sum(nums) / max(1, len(nums))
                row.append(f"{mean:.2f}")
            writer.writerow(row)
