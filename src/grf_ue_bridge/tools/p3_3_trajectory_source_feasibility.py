"""描述 P3-2 trajectory 失败模式，不改变既有 Motion Quality 判定。"""

from __future__ import annotations

from collections import Counter
import argparse
import hashlib
import json
import math
import time
from numbers import Real
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from grf_ue_bridge import motion_quality


BOUNDED_SCENARIO = "5_vs_5"
BOUNDED_FRAMES = 300
BOUNDED_FPS = 10.0
BOUNDED_SEEDS = (42, 43, 44, 45)


def validate_bounded_settings(*, scenario: str, frames: int, fps: float, seeds: Sequence[int]) -> None:
    """拒绝所有超出 Task 5 固定实验边界的设置。"""
    if scenario != BOUNDED_SCENARIO:
        raise ValueError("Task 5 scenario must be 5_vs_5")
    if frames != BOUNDED_FRAMES:
        raise ValueError("Task 5 frames must be 300")
    if fps != BOUNDED_FPS:
        raise ValueError("Task 5 fps must be 10.0")
    seed_tuple = tuple(seeds)
    if seed_tuple != BOUNDED_SEEDS and not (
        len(seed_tuple) == 1 and seed_tuple[0] in BOUNDED_SEEDS
    ):
        raise ValueError("Task 5 seeds must be 42, 43, 44, 45 or one repeat seed")


def write_grf_native_infeasibility(out_root: Path) -> Path:
    """写出不可运行 GRF-native 替代源的具体环境证据。"""
    path = out_root / "grf_native_alternative.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "candidate": "grf_native_alternative",
        "disposition": "INFEASIBLE_IN_CURRENT_ENVIRONMENT",
        "evidence": (
            "no local checkpoint/trace; GRF_MARL runtime dependencies/policies unavailable"
        ),
        "experiment_run": False,
    }, indent=2) + "\n", encoding="utf-8")
    return path


def assess_ue_native_feasibility() -> Dict[str, Any]:
    """根据当前代码和资产证据评估 UE 原生轨迹来源的可行性。"""
    return {
        "playback_contract_supported": True,
        "complete_5v5_simulation_available": False,
        "disposition": "ARCHITECTURALLY_POSSIBLE_NOT_READY",
    }


def build_project_owned_prototype(
    *, seed: int, frames: int, fps: float
) -> List[Dict[str, Any]]:
    """生成用于可行性试验的确定性运动学 5v5 轨迹。"""
    import random

    if frames < 0:
        raise ValueError("frames must be non-negative")
    if fps <= 0.0:
        raise ValueError("fps must be positive")

    rng = random.Random(seed)
    phases = {
        f"{team}{index}": rng.uniform(0.0, 2.0 * math.pi)
        for team in ("L", "R")
        for index in range(5)
    }
    role_scales = {
        f"{team}{index}": 0.8 + rng.random() * 0.4
        for team in ("L", "R")
        for index in range(5)
    }
    ids = [f"{team}{index}" for team in ("L", "R") for index in range(5)]
    previous_positions: Dict[str, tuple] = {}
    owner = "L0"
    frames_out: List[Dict[str, Any]] = []

    def position(entity_id: str, time_s: float) -> List[float]:
        team, index_text = entity_id[0], entity_id[1:]
        index = int(index_text)
        side = -1.0 if team == "L" else 1.0
        lane_y = (index - 2) * 2.3
        phase = phases[entity_id]
        scale = role_scales[entity_id]
        x = side * (14.0 + 1.8 * scale * math.sin(0.65 * time_s + phase))
        y = lane_y + 1.4 * scale * math.cos(0.8 * time_s + phase)
        return [round(x, 6), round(y, 6), 0.0]

    for step in range(frames):
        time_s = step / fps
        positions = {entity_id: position(entity_id, time_s) for entity_id in ids}

        # 每 12 帧使用当前持球人与确定性接应人的中点判断空间接管；
        # 输出球始终跟随最终持球人，避免持球状态与球位置脱节。
        if step and step % 12 == 0:
            owner_team = owner[0]
            receiver_index = (int(owner[1:]) + (1 if owner_team == "L" else -1)) % 5
            receiver = f"{owner_team}{receiver_index}"
            ball_xy = [
                0.3 * positions[owner][axis] + 0.7 * positions[receiver][axis]
                for axis in range(2)
            ]
        else:
            ball_xy = [positions[owner][0], positions[owner][1]]

        nearest_id = min(
            ids,
            key=lambda entity_id: math.hypot(
                positions[entity_id][0] - ball_xy[0],
                positions[entity_id][1] - ball_xy[1],
            ),
        )
        nearest_distance = math.hypot(
            positions[nearest_id][0] - ball_xy[0],
            positions[nearest_id][1] - ball_xy[1],
        )
        if nearest_id != owner and nearest_distance <= 1.25:
            owner = nearest_id
        ball_xy = [positions[owner][0], positions[owner][1]]

        players = []
        for entity_id in ids:
            current = positions[entity_id]
            previous = previous_positions.get(entity_id, current)
            velocity = [
                round((current[axis] - previous[axis]) * fps, 6)
                for axis in range(2)
            ]
            players.append({
                "id": entity_id,
                "position_m": current,
                "velocity_mps": velocity,
                "active": True,
                "has_ball": entity_id == owner,
            })
            previous_positions[entity_id] = tuple(current)

        ball_position = [round(ball_xy[0], 6), round(ball_xy[1], 6), 0.11]
        if frames_out:
            previous_ball = frames_out[-1]["ball"]["position_m"]
            ball_velocity = [
                round((ball_position[axis] - previous_ball[axis]) * fps, 6)
                for axis in range(3)
            ]
        else:
            ball_velocity = [0.0, 0.0, 0.0]
        owner_team = 0 if owner[0] == "L" else 1
        frames_out.append({
            "step": step,
            "time_seconds": round(time_s, 6),
            "score": [0, 0],
            "ball": {
                "position_m": ball_position,
                "velocity_mps": ball_velocity,
            },
            "players": players,
            "ball_owned_team": owner_team,
            "ball_owned_player": int(owner[1:]),
            "game_mode": 0,
        })

    return frames_out


def _repository_local_ppo_checkpoint_is_usable() -> bool:
    """只依据仓库内现有文件判断 PPO checkpoint 是否可探测。"""
    repo_root = Path(__file__).resolve().parents[3]
    grf_root = repo_root / ".external" / "google-research-football"
    if not grf_root.is_dir():
        return False
    return any(
        path.is_file()
        for pattern in ("**/*.pkl", "**/*.ckpt", "**/*.index", "**/*.data-*")
        for path in grf_root.glob(pattern)
    )


def inventory_grf_sources() -> List[Dict[str, Any]]:
    """返回基于固定源码证据和本地资源状态的 GRF source 清单。"""
    checkpoint_is_usable = _repository_local_ppo_checkpoint_is_usable()
    return [
        {
            "name": "builtin_ai",
            "api_evidence": "action_set_v2_index_19",
            "runtime_available": True,
            "disposition": "BASELINE",
        },
        {
            "name": "bot",
            "api_evidence": "sample_bot_player",
            "runtime_available": True,
            "disposition": "NOT_A_COMPLETE_POLICY_SOURCE",
        },
        {
            "name": "replay",
            "api_evidence": "replay_player_requires_trace",
            "runtime_available": True,
            "disposition": "REPLAY_ONLY",
        },
        {
            "name": "ppo_checkpoint",
            "api_evidence": "ppo2_cnn_player_requires_checkpoint",
            "runtime_available": checkpoint_is_usable,
            "disposition": (
                "AVAILABLE_FOR_PROBE"
                if checkpoint_is_usable
                else "INFEASIBLE_IN_CURRENT_ENVIRONMENT"
            ),
        },
        {
            "name": "grf_marl_policy",
            "api_evidence": "ippo_mappo_happo_policy_rollout_framework",
            "runtime_available": False,
            "disposition": "INFEASIBLE_IN_CURRENT_ENVIRONMENT",
        },
    ]


def characterize_failures(results: Sequence[Mapping]) -> Dict[str, Any]:
    """统计既有 strict gate 失败，并按严重程度做描述性分类。"""
    criterion_names = (
        "outfield_active_ratio",
        "outfield_stationary_streak",
        "team_active_coverage",
    )
    criterion_set = set(criterion_names)
    failure_counts = Counter()
    failure_combinations = Counter()
    severe_seeds = []
    single_seeds = []
    classification_by_seed = {}
    gk_metrics_by_seed = {}

    for row in results:
        seed = int(row["seed"])
        raw_failures = list(row.get("gate_failures", []))
        failures = [name for name in raw_failures if name in criterion_set]
        unique_failures = list(dict.fromkeys(failures))
        for name in unique_failures:
            failure_counts[name] += 1
        if unique_failures:
            failure_combinations["+".join(sorted(unique_failures, key=criterion_names.index))] += 1

        plateau_s = float(row.get("longest_global_low_motion_plateau_s", 0.0))
        if plateau_s >= 5.0:
            severe_seeds.append(seed)
        if len(raw_failures) == 1:
            single_seeds.append(seed)

        if plateau_s >= 5.0:
            classification = "severe_global_low_motion"
        elif len(raw_failures) > 1:
            classification = "multi_criterion_failure"
        elif len(raw_failures) == 1:
            classification = "single_criterion_near_miss"
        else:
            classification = "other"
        classification_by_seed[str(seed)] = classification
        gk_metrics_by_seed[str(seed)] = {
            "max_gk_stationary_streak_s": row.get("max_gk_stationary_streak_s"),
        }

    return {
        "criterion_failure_counts": {
            name: failure_counts[name] for name in criterion_names
        },
        "failure_combinations": dict(failure_combinations),
        "severe_global_low_motion_seeds": sorted(severe_seeds),
        "single_criterion_failure_seeds": sorted(single_seeds),
        "classification_by_seed": classification_by_seed,
        "gk_stationary_streak_is_gate": False,
        "gk_metrics_by_seed": gk_metrics_by_seed,
    }


def sha256_file(path: Path) -> str:
    """返回文件的稳定 SHA-256 hash。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def contract_compatibility(frames: Sequence[Mapping]) -> Dict[str, Any]:
    """检查 frames 是否满足现有固定实体和球位置契约。"""
    expected_ids = {f"{team}{index}" for team in ("L", "R") for index in range(5)}
    errors: List[Dict[str, Any]] = []

    def finite_vector(value, length: int) -> bool:
        return (
            isinstance(value, Sequence)
            and not isinstance(value, (str, bytes))
            and len(value) == length
            and all(
                isinstance(item, Real)
                and not isinstance(item, bool)
                and math.isfinite(item)
                for item in value
            )
        )

    for index, frame in enumerate(frames):
        if not isinstance(frame, Mapping):
            errors.append({"code": "frame_not_mapping", "frame_index": index})
            continue
        players = frame.get("players")
        player_ids = []
        if isinstance(players, list):
            for player_index, player in enumerate(players):
                if not isinstance(player, Mapping):
                    errors.append({
                        "code": "player_not_mapping",
                        "frame_index": index,
                        "player_index": player_index,
                    })
                    continue
                player_ids.append(player.get("id"))
                if not finite_vector(player.get("position_m"), 3):
                    errors.append({"code": "player_position_invalid", "frame_index": index, "player_id": player.get("id")})
                if "velocity_mps" in player and not finite_vector(player["velocity_mps"], 2):
                    errors.append({"code": "player_velocity_invalid", "frame_index": index, "player_id": player.get("id")})
                if "active" in player and not isinstance(player["active"], bool):
                    errors.append({"code": "player_active_invalid", "frame_index": index, "player_id": player.get("id")})
                if "has_ball" in player and not isinstance(player["has_ball"], bool):
                    errors.append({"code": "player_has_ball_invalid", "frame_index": index, "player_id": player.get("id")})
        if not isinstance(players, list) or (
            all(isinstance(player, Mapping) for player in players)
            and (set(player_ids) != expected_ids or len(player_ids) != len(expected_ids))
        ):
            errors.append({"code": "player_ids_invalid", "frame_index": index})
        ball = frame.get("ball")
        position = ball.get("position_m") if isinstance(ball, Mapping) else None
        if (
            not finite_vector(position, 3)
        ):
            errors.append({"code": "ball_position_invalid", "frame_index": index})
        if isinstance(ball, Mapping) and "velocity_mps" in ball and not finite_vector(ball["velocity_mps"], 3):
            errors.append({"code": "ball_velocity_invalid", "frame_index": index})

        if "ball_owned_team" in frame or "ball_owned_player" in frame:
            owned_team = frame.get("ball_owned_team")
            owned_player = frame.get("ball_owned_player")
            ownership_valid = (
                isinstance(owned_team, int)
                and not isinstance(owned_team, bool)
                and owned_team in (-1, 0, 1)
                and isinstance(owned_player, int)
                and not isinstance(owned_player, bool)
                and owned_player in range(-1, 5)
                and ((owned_team == -1 and owned_player == -1) or owned_team in (0, 1) and owned_player >= 0)
            )
            if not ownership_valid:
                errors.append({"code": "ownership_invalid", "frame_index": index})
        if "score" in frame and not (
            isinstance(frame["score"], Sequence)
            and not isinstance(frame["score"], (str, bytes))
            and len(frame["score"]) == 2
            and all(isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in frame["score"])
        ):
            errors.append({"code": "score_invalid", "frame_index": index})
        if "game_mode" in frame and not (
            isinstance(frame["game_mode"], int) and not isinstance(frame["game_mode"], bool)
        ):
            errors.append({"code": "game_mode_invalid", "frame_index": index})
    return {"compatible": not errors, "errors": errors}


def evaluate_frames_candidate(name: str, frames_path: Path, fps: float = 10.0) -> Dict[str, Any]:
    """读取并用既有 Motion Quality 审计评估一个 frames.jsonl 候选。"""
    if fps <= 0.0:
        raise ValueError("fps must be positive")
    frames = [
        json.loads(line)
        for line in frames_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    contract = contract_compatibility(frames)
    metrics = None
    motion_quality_error = None
    try:
        metrics = motion_quality.analyze_frames(frames, dt_s=1.0 / fps, gk_ids=["L0", "R0"])
    except (AttributeError, KeyError, TypeError, ValueError, IndexError) as error:
        motion_quality_error = f"{type(error).__name__}: {error}"
    if metrics is not None:
        from grf_ue_bridge.tools.p3_2_valid_trajectory_cohort import assess_motion_quality

        decision = assess_motion_quality(metrics)
        valid = decision["valid"] and contract["compatible"]
        gate_failures = decision["failures"]
    else:
        valid = False
        gate_failures = []
    result = {
        "candidate": name,
        "source": name,
        "frame_count": len(frames),
        "frames_sha256": sha256_file(frames_path),
        "contract": contract,
        "motion_quality": metrics,
        "valid": valid,
        "gate_failures": gate_failures,
    }
    if name == "project_owned_prototype":
        result["prototype"] = True
        result["production"] = False
    if motion_quality_error is not None:
        result["motion_quality_error"] = motion_quality_error
    return result


def _write_frames(frames: Sequence[Mapping[str, Any]], path: Path) -> None:
    """以现有 JSONL 帧契约写出诊断轨迹。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(frame, ensure_ascii=False, separators=(",", ":")) + "\n" for frame in frames),
        encoding="utf-8",
    )


def _run_candidate(candidate: str, seed: int, frames: int, fps: float, out_root: Path) -> Dict[str, Any]:
    """运行一个候选并返回带生成耗时的结果。"""
    started = time.perf_counter()
    frames_path = out_root / f"seed_{seed}" / "frames.jsonl"
    if candidate == "grf_builtin_ai":
        from grf_ue_bridge.config import ExportConfig
        from grf_ue_bridge.exporter import export_episode
        from grf_ue_bridge.grf_runner import run_episode

        config = ExportConfig(
            scenario="5_vs_5",
            seed=seed,
            num_steps=frames,
            target_fps=0,
            playback_fps=int(fps),
            field_length_m=40.0,
            field_width_m=20.0,
            render=False,
        )
        episode_dir = frames_path.parent
        export_episode(config, run_episode("5_vs_5", seed=seed, num_steps=frames, render=False), episode_dir)
    elif candidate == "project_owned_prototype":
        _write_frames(build_project_owned_prototype(seed=seed, frames=frames, fps=fps), frames_path)
    else:
        raise ValueError(f"unsupported candidate: {candidate}")
    result = evaluate_frames_candidate(candidate, frames_path, fps=fps)
    result["seed"] = seed
    result["generation_seconds"] = round(time.perf_counter() - started, 6)
    return result


def _run_baseline(args: argparse.Namespace) -> int:
    validate_bounded_settings(
        scenario=args.scenario,
        frames=args.frames,
        fps=args.fps,
        seeds=args.seeds,
    )
    results = [
        _run_candidate(args.candidate, seed, args.frames, args.fps, args.out_root)
        for seed in args.seeds
    ]
    report = {
        "candidate": args.candidate,
        "disposition": "RUNNABLE",
        "scenario": args.scenario,
        "frames": args.frames,
        "fps": args.fps,
        "seeds": list(args.seeds),
        "results": results,
        "strict_valid_count": sum(bool(row["valid"]) for row in results),
    }
    if args.candidate == "project_owned_prototype":
        report["prototype"] = True
        report["production"] = False
    report_path = args.out_root / "candidate_results.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(report_path)
    return 0


def _run_repeat(args: argparse.Namespace) -> int:
    validate_bounded_settings(
        scenario=args.scenario,
        frames=args.frames,
        fps=args.fps,
        seeds=[args.seed if args.seed in BOUNDED_SEEDS else args.seed],
    )
    result = _run_candidate(args.candidate, args.seed, args.frames, args.fps, args.out_root)
    report_path = args.out_root / args.candidate / f"seed_{args.seed}_repeat.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(report_path)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    baseline = subparsers.add_parser("baseline")
    baseline.add_argument("--candidate", choices=("grf_builtin_ai", "project_owned_prototype"), required=True)
    baseline.add_argument("--out-root", type=Path, required=True)
    baseline.add_argument("--seeds", type=int, nargs="+", required=True)
    baseline.add_argument("--frames", type=int, required=True)
    baseline.add_argument("--fps", type=float, required=True)
    baseline.set_defaults(handler=_run_baseline)
    repeat = subparsers.add_parser("repeat")
    repeat.add_argument("--candidate", choices=("grf_builtin_ai", "project_owned_prototype"), required=True)
    repeat.add_argument("--seed", type=int, required=True)
    repeat.add_argument("--frames", type=int, required=True)
    repeat.add_argument("--fps", type=float, required=True)
    repeat.add_argument("--out-root", type=Path, required=True)
    baseline.add_argument("--scenario", default=BOUNDED_SCENARIO)
    repeat.add_argument("--scenario", default=BOUNDED_SCENARIO)
    repeat.set_defaults(handler=_run_repeat)
    args = parser.parse_args()
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
