"""从预先固定的 seeds 中筛选 Motion Quality 合格 trajectory cohort。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from grf_ue_bridge import motion_quality


def assess_motion_quality(metrics: Mapping[str, Any]) -> Dict[str, Any]:
    """按现有 Motion Quality Gate 返回严格 valid 结果。"""
    players = metrics.get("players", {})
    outfields = [data for data in players.values() if not data.get("is_gk", False)]
    goalkeepers = [data for data in players.values() if data.get("is_gk", False)]
    failures: List[str] = []
    if not outfields or len(goalkeepers) != 2:
        failures.append("player_roles")
    if any(data.get("active_ratio", 0.0) < motion_quality.OUTFIELD_GK_RATIO_REQ for data in outfields):
        failures.append("outfield_active_ratio")
    if any(data.get("longest_stationary_streak_s", float("inf")) > motion_quality.OUTFIELD_STREAK_MAX_S for data in outfields):
        failures.append("outfield_stationary_streak")
    if metrics.get("team_active_outfield_coverage", 0.0) < motion_quality.TEAM_ACTIVE_COVERAGE_REQ:
        failures.append("team_active_coverage")
    return {"valid": not failures, "failures": failures}


def select_cohort(results: Sequence[Mapping[str, Any]], target_size: int = 5) -> Dict[str, Any]:
    """按 seed 数字升序选择前 target_size 个严格 valid seed。"""
    if target_size < 1:
        raise ValueError("target_size must be positive")
    valid_seeds = sorted(int(result["seed"]) for result in results if bool(result.get("valid")))
    if len(valid_seeds) < target_size:
        return {
            "status": "INSUFFICIENT",
            "valid_seeds": valid_seeds,
            "selected_seeds": [],
            "target_size": target_size,
        }
    return {
        "status": "PASS",
        "valid_seeds": valid_seeds,
        "selected_seeds": valid_seeds[:target_size],
        "target_size": target_size,
    }


def frames_sha256(path: Path) -> str:
    """返回 frames.jsonl 的稳定字节 hash。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_frames(path: Path, fps: float = 10.0, gk_ids: Iterable[str] = ("L0", "R0")) -> Dict[str, Any]:
    """读取单个 frames.jsonl，调用现有 Motion Quality Audit 并附 strict decision。"""
    frames = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    metrics = motion_quality.analyze_frames(frames, dt_s=1.0 / fps, gk_ids=list(gk_ids))
    decision = assess_motion_quality(metrics)
    return {
        "frame_count": len(frames),
        "min_outfield_active_ratio": round(
            min(
                data["active_ratio"]
                for pid, data in metrics["players"].items()
                if pid not in set(gk_ids)
            ),
            3,
        ),
        "max_outfield_stationary_streak_s": round(
            max(
                data["longest_stationary_streak_s"]
                for pid, data in metrics["players"].items()
                if pid not in set(gk_ids)
            ),
            2,
        ),
        "max_gk_stationary_streak_s": round(
            max(metrics["players"][pid]["longest_stationary_streak_s"] for pid in gk_ids), 2
        ),
        "team_active_outfield_coverage": metrics["team_active_outfield_coverage"],
        "longest_global_low_motion_plateau_s": metrics["longest_global_low_motion_plateau_s"],
        "valid": decision["valid"],
        "gate_failures": decision["failures"],
        "frames_sha256": frames_sha256(path),
    }


def run_candidates(
    output_root: Path,
    seeds: Sequence[int] = tuple(range(42, 60)),
    frames: int = 300,
    fps: float = 10.0,
    target_size: int = 5,
) -> Dict[str, Any]:
    """逐个生成候选 trajectory 并筛选 cohort；不执行 retry。"""
    from grf_ue_bridge.exporter import export_episode
    from grf_ue_bridge.config import ExportConfig
    from grf_ue_bridge.grf_runner import run_episode

    results = []
    for seed in seeds:
        episode_dir = output_root / f"seed_{seed}"
        config = ExportConfig(
            scenario="5_vs_5",
            seed=int(seed),
            num_steps=frames,
            target_fps=0,
            playback_fps=int(fps),
            field_length_m=40.0,
            field_width_m=20.0,
            render=False,
        )
        result = run_episode("5_vs_5", seed=int(seed), num_steps=frames, render=False)
        export_episode(config, result, episode_dir)
        audit = audit_frames(episode_dir / "frames.jsonl", fps=fps)
        results.append({"seed": int(seed), **audit})
    selection = select_cohort(results, target_size=target_size)
    return {
        "candidate_seeds": [int(seed) for seed in seeds],
        "scenario": "5_vs_5",
        "frames": frames,
        "fps": fps,
        "results": results,
        "cohort": selection,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--target-size", type=int, default=5)
    args = parser.parse_args()
    report = run_candidates(args.out_root, target_size=args.target_size)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
