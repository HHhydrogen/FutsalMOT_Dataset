"""分析 GRF builtin_ai 轨迹中的低运动触发条件。

该工具只消费 run_episode 返回的快照，不参与轨迹生成，也不改变任何 GRF
配置。GRF 不会把 builtin_ai 在引擎内部选择的动作写回 observation，因此报告
会明确区分外部发送的固定 action 和不可见的内部 AI action。
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


PLAYER_IDS = tuple(f"{team}{index}" for team in ("L", "R") for index in range(5))


def _observation(snapshot: Mapping[str, Any]) -> Mapping[str, Any]:
    return snapshot.get("observation", snapshot)


def _owner(observation: Mapping[str, Any]) -> List[int]:
    return [
        int(observation.get("ball_owned_team", -1)),
        int(observation.get("ball_owned_player", -1)),
    ]


def _player_position(observation: Mapping[str, Any], player_id: str) -> Sequence[float]:
    key = "left_team" if player_id[0] == "L" else "right_team"
    return observation[key][int(player_id[1])]


def _player_direction(observation: Mapping[str, Any], player_id: str) -> Sequence[float]:
    key = "left_team_direction" if player_id[0] == "L" else "right_team_direction"
    return observation[key][int(player_id[1])]


def _direction_speed_mps(direction: Sequence[float], field_scale_m: float = 20.0, dt_s: float = 0.1) -> float:
    """将 GRF 归一化方向转换为近似米/秒，使用现有 40m 场地 x scale。"""
    return math.hypot(float(direction[0]), float(direction[1])) * field_scale_m / dt_s


def _pos_delta_mps(
    current: Sequence[float], previous: Sequence[float], field_scale_m: float = 20.0, dt_s: float = 0.1
) -> float:
    return math.hypot(
        (float(current[0]) - float(previous[0])) * field_scale_m,
        (float(current[1]) - float(previous[1])) * field_scale_m,
    ) / dt_s


def _runs(values: Iterable[Any]) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    for index, value in enumerate(values):
        if not result or value != result[-1]["value"]:
            result.append({"start_frame": index, "end_frame": index, "value": value})
        else:
            result[-1]["end_frame"] = index
    return result


def _run_with_duration(run: Mapping[str, Any], fps: float) -> Dict[str, Any]:
    output = dict(run)
    output["duration_frames"] = int(run["end_frame"]) - int(run["start_frame"]) + 1
    output["duration_s"] = round(output["duration_frames"] / fps, 3)
    return output


def analyze_snapshots(
    snapshots: Sequence[Mapping[str, Any]],
    fps: float = 10.0,
    low_motion_mps: float = 0.2,
    external_action_name: Optional[str] = None,
    external_action_index: Optional[int] = None,
) -> Dict[str, Any]:
    """返回 possession、球员运动、外部 action 和 game state 诊断数据。"""
    if fps <= 0.0:
        raise ValueError("fps must be positive")
    if not snapshots:
        raise ValueError("snapshots must not be empty")
    dt_s = 1.0 / fps
    observations = [_observation(snapshot) for snapshot in snapshots]

    owner_runs = [
        _run_with_duration(run, fps)
        for run in _runs(tuple(_owner(observation)) for observation in observations)
    ]
    possession_changes = max(0, len(owner_runs) - 1)
    owned_runs = [run for run in owner_runs if tuple(run["value"]) != (-1, -1)]
    longest_run = max(owned_runs, key=lambda run: run["duration_frames"], default=None)
    possession = {
        "runs": owner_runs,
        "owned_run_count": len(owned_runs),
        "possession_change_count": possession_changes,
        "owned_frame_ratio": round(sum(run["duration_frames"] for run in owned_runs) / len(observations), 3),
        "longest_run": longest_run,
    }

    players: Dict[str, Any] = {}
    for player_id in PLAYER_IDS:
        direction_speeds = [
            _direction_speed_mps(_player_direction(observation, player_id))
            for observation in observations
        ]
        delta_speeds = [0.0]
        for previous, current in zip(observations, observations[1:]):
            delta_speeds.append(
                _pos_delta_mps(
                    _player_position(current, player_id),
                    _player_position(previous, player_id),
                )
            )
        low_runs = [
            _run_with_duration(run, fps)
            for run in _runs(speed < low_motion_mps for speed in direction_speeds)
            if run["value"]
        ]
        players[player_id] = {
            "first_low_motion_frame": low_runs[0]["start_frame"] if low_runs else None,
            "longest_low_motion_run": max(low_runs, key=lambda run: run["duration_frames"], default=None),
            "direction_speed_mps": {
                "min": round(min(direction_speeds), 6),
                "max": round(max(direction_speeds), 6),
                "mean": round(sum(direction_speeds) / len(direction_speeds), 6),
            },
            "position_delta_speed_mps": {
                "min": round(min(delta_speeds), 6),
                "max": round(max(delta_speeds), 6),
                "mean": round(sum(delta_speeds) / len(delta_speeds), 6),
            },
            "direction_speed_samples_mps": {
                str(index): round(direction_speeds[index], 6)
                for index in (0, 50, 100, 140, 146, 147, 150, 160, 180, 200, 250, 299)
                if index < len(direction_speeds)
            },
            "active_flags": {
                "true": sum(1 for observation in observations if observation.get("left_team_active" if player_id[0] == "L" else "right_team_active", [])[int(player_id[1])]),
                "false": sum(1 for observation in observations if not observation.get("left_team_active" if player_id[0] == "L" else "right_team_active", [True] * 5)[int(player_id[1])]),
            },
        }

    state_fields = ("score", "steps_left", "game_mode")
    game_state: Dict[str, Any] = {}
    for field in state_fields:
        values = [observation.get(field) for observation in observations]
        game_state[field] = {
            "start": values[0],
            "end": values[-1],
            "distinct_values": list(dict.fromkeys(json.dumps(value, sort_keys=True) for value in values)),
        }
    game_state["done_frames"] = [
        int(snapshot.get("step", index))
        for index, snapshot in enumerate(snapshots)
        if bool(snapshot.get("done", False))
    ]
    game_state["ball"] = {
        "position_start": observations[0].get("ball"),
        "position_end": observations[-1].get("ball"),
        "direction_start": observations[0].get("ball_direction"),
        "direction_end": observations[-1].get("ball_direction"),
    }

    actions: Dict[str, Any] = {
        "external_action_name": external_action_name,
        "external_action_index": external_action_index,
        "external_action_count_per_step": 1 if external_action_name is not None else None,
        "external_action_distribution": dict(Counter({external_action_name: len(observations)}))
        if external_action_name is not None
        else {},
        "internal_builtin_ai_distribution": "not_exposed_by_grf_observation",
    }
    return {
        "total_frames": len(observations),
        "fps": fps,
        "low_motion_mps": low_motion_mps,
        "possession": possession,
        "players": players,
        "actions": actions,
        "game_state": game_state,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--frames", type=int, default=300)
    args = parser.parse_args()

    from grf_ue_bridge.grf_runner import run_episode

    result = run_episode("5_vs_5", seed=args.seed, num_steps=args.frames, render=False)
    report = analyze_snapshots(
        [
            {"step": snapshot.step, "observation": snapshot.observation, "done": snapshot.done}
            for snapshot in result.snapshots
        ],
        fps=10.0,
        external_action_name="builtin_ai",
        external_action_index=19,
    )
    report["seed"] = args.seed
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
