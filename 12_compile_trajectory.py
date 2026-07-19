#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FutsalMOT rule-based episode trajectory compiler.

Version
-------
A3_2_EVENT_TO_DENSE_TRAJECTORY_8P_MOVEMENT_V3

Purpose
-------
Compile a validated episode-event JSON into a UE-compatible dense trajectory
configuration. The output contains one keyframe per object per frame and can be
read directly by the existing A2.2 Unreal Engine build/annotation script.

Implemented event types
-----------------------
- hold
- move
- dribble
- pass
- receive
- defend_follow
- shot

Compilation principles
----------------------
1. Explicit player movement uses smoothstep interpolation.
2. defend_follow uses predictive, acceleration-limited pursuit with local separation avoidance.
3. Owned/dribbled ball positions are derived from the owner's dense trajectory.
4. Passes and shots use deterministic ball flight curves.
5. The base render configuration is deep-copied so cameras, animation, bbox,
   image settings, IDs, and other stable UE settings are retained.
6. The generated object trajectories use linear UE interpolation because the
   motion is already sampled densely at one keyframe per frame.

Default files
-------------
Input event config:
    configs/events/episode_test_0001.json
Base render config, resolved from the event JSON:
    configs/seq_test_0005.json
Output trajectory config, resolved from the event JSON:
    configs/generated/episode_test_0001.json

Exit codes
----------
0: generated and all enabled validators succeeded
1: output generated, but a validator returned a non-zero status
2: fatal compiler/configuration failure
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


SCRIPT_VERSION = "A3_2_EVENT_TO_DENSE_TRAJECTORY_8P_MOVEMENT_V3"

DEFAULT_EVENT_CONFIG = (
    Path(__file__).resolve().parent
    / "configs"
    / "events"
    / "episode_test_0001.json"
)

PLAYER_MOTION_TYPES = {
    "hold",
    "move",
    "dribble",
    "pass",
    "defend_follow",
    "shot",
}

SUPPORTED_EVENT_TYPES = PLAYER_MOTION_TYPES | {"receive"}

DEFAULTS: Dict[str, Any] = {
    "dense_keyframe_interval_frames": 1,
    "player_max_speed_cm_s": 750.0,
    "ball_max_speed_cm_s": 3000.0,
    "dribble_ball_ahead_cm": 45.0,
    "pass_arc_height_cm": 20.0,
    "shot_arc_height_cm": 55.0,
    "defender_default_follow_distance_cm": 180.0,
    "defender_default_side_offset_cm": -80.0,
    "defender_follow_speed_cm_s": 500.0,
    "defender_response_alpha": 0.22,
    "defender_response_time_sec": 0.38,
    "defender_max_acceleration_cm_s2": 850.0,
    "defender_lookahead_frames": 5,
    "defender_avoidance_radius_cm": 120.0,
    "defender_avoidance_weight": 0.65,
    "court_player_margin_cm": 35.0,
}


class CompileError(RuntimeError):
    """Fatal episode compilation error."""


def is_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def require_float(value: Any, field: str) -> float:
    if not is_number(value):
        raise CompileError(
            "字段 '{}' 必须是有限数值，当前值={!r}".format(field, value)
        )
    return float(value)


def require_int(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise CompileError(
            "字段 '{}' 必须是整数，当前值={!r}".format(field, value)
        )
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return int(value)
    raise CompileError(
        "字段 '{}' 必须是整数，当前值={!r}".format(field, value)
    )


def require_str(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CompileError("字段 '{}' 必须是非空字符串".format(field))
    return value.strip()


def parse_vec3(value: Any, field: str) -> Tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise CompileError(
            "字段 '{}' 必须是长度为 3 的数组 [x, y, z]".format(field)
        )
    return (
        require_float(value[0], field + "[0]"),
        require_float(value[1], field + "[1]"),
        require_float(value[2], field + "[2]"),
    )


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise CompileError("找不到 JSON 文件：{}".format(path))
    try:
        with path.open("r", encoding="utf-8-sig") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise CompileError(
            "JSON 解析失败：{}，line={} column={}".format(
                exc.msg, exc.lineno, exc.colno
            )
        ) from exc
    except OSError as exc:
        raise CompileError("读取 JSON 失败：{}；{}".format(path, exc)) from exc

    if not isinstance(data, dict):
        raise CompileError("JSON 顶层必须是 object：{}".format(path))
    return data


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )


def resolve_relative_path(
    raw_path: str,
    reference_file: Path,
) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = reference_file.resolve().parent / path
    return path.resolve()


def preserve_backup(output_path: Path) -> Optional[Path]:
    if not output_path.exists():
        return None

    candidate = output_path.with_suffix(output_path.suffix + ".bak")
    index = 1
    while candidate.exists():
        candidate = output_path.with_suffix(
            output_path.suffix + ".bak{}".format(index)
        )
        index += 1

    output_path.replace(candidate)
    return candidate


def time_to_frame(time_sec: float, fps: float) -> int:
    return int(round(time_sec * fps))


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def smoothstep01(t: float) -> float:
    t = clamp(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def distance_xy(a: Sequence[float], b: Sequence[float]) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def normalize_xy(dx: float, dy: float) -> Tuple[float, float]:
    length = math.hypot(dx, dy)
    if length <= 1e-9:
        return (1.0, 0.0)
    return (dx / length, dy / length)


def limit_xy_length(dx: float, dy: float, maximum: float) -> Tuple[float, float]:
    """Limit a 2-D vector magnitude while preserving direction."""
    magnitude = math.hypot(dx, dy)
    if magnitude <= maximum or magnitude <= 1e-12:
        return (dx, dy)
    scale = maximum / magnitude
    return (dx * scale, dy * scale)


def defender_avoidance_offset(
    current: Sequence[float],
    frame_index: int,
    compiled_paths: Dict[str, List[Tuple[float, float, float]]],
    target_player: Optional[str],
    radius_cm: float,
    weight: float,
) -> Tuple[float, float]:
    """Return a deterministic local repulsion offset from already compiled players.

    Repulsion includes the followed target during transient movement.  The
    tactical follow offset controls the steady-state location, while this local
    term prevents a defender from cutting through the marked player's body on
    the way to that offset.  Later defenders also avoid earlier defenders,
    without introducing cyclic dependencies.
    """
    if radius_cm <= 0.0 or weight <= 0.0:
        return (0.0, 0.0)

    offset_x = 0.0
    offset_y = 0.0
    for other_id in sorted(compiled_paths):
        path = compiled_paths[other_id]
        if not path:
            continue
        other = path[min(frame_index, len(path) - 1)]
        dx = current[0] - other[0]
        dy = current[1] - other[1]
        distance = math.hypot(dx, dy)
        if distance >= radius_cm:
            continue
        if distance <= 1e-6:
            # Stable tie break; do not use random state in the compiler.
            sign = -1.0 if sum(ord(ch) for ch in other_id) % 2 else 1.0
            dx, dy, distance = 0.0, sign, 1.0
        strength = (radius_cm - distance) / radius_cm
        scale = weight * radius_cm * strength / distance
        offset_x += dx * scale
        offset_y += dy * scale

    return limit_xy_length(offset_x, offset_y, radius_cm * max(0.5, weight))


def acceleration_limited_velocity(
    current_velocity: Tuple[float, float],
    desired_velocity: Tuple[float, float],
    max_acceleration_cm_s2: float,
    fps: float,
    max_speed_cm_s: float,
) -> Tuple[float, float]:
    max_delta = max_acceleration_cm_s2 / fps
    delta_x = desired_velocity[0] - current_velocity[0]
    delta_y = desired_velocity[1] - current_velocity[1]
    delta_x, delta_y = limit_xy_length(delta_x, delta_y, max_delta)
    next_velocity = (
        current_velocity[0] + delta_x,
        current_velocity[1] + delta_y,
    )
    return limit_xy_length(next_velocity[0], next_velocity[1], max_speed_cm_s)


def get_event_actor(event: Dict[str, Any]) -> Optional[str]:
    event_type = event["type"]
    if event_type == "pass":
        value = event.get("from")
    elif event_type in {
        "hold",
        "move",
        "dribble",
        "receive",
        "defend_follow",
        "shot",
    }:
        value = event.get("actor")
    else:
        value = None
    return value if isinstance(value, str) else None


def parse_timeline(event_config: Dict[str, Any]) -> Dict[str, Any]:
    raw = event_config.get("timeline")
    if not isinstance(raw, dict):
        raise CompileError("timeline 必须是 JSON object")

    result = {
        "fps": require_float(raw.get("fps"), "timeline.fps"),
        "duration_sec": require_float(
            raw.get("duration_sec"), "timeline.duration_sec"
        ),
        "frame_start": require_int(
            raw.get("frame_start"), "timeline.frame_start"
        ),
        "frame_end": require_int(
            raw.get("frame_end"), "timeline.frame_end"
        ),
    }

    if result["fps"] <= 0.0:
        raise CompileError("timeline.fps 必须大于 0")
    if result["duration_sec"] <= 0.0:
        raise CompileError("timeline.duration_sec 必须大于 0")
    if result["frame_end"] < result["frame_start"]:
        raise CompileError("timeline.frame_end 不能小于 frame_start")

    expected_end = (
        result["frame_start"]
        + int(round(result["duration_sec"] * result["fps"]))
        - 1
    )
    if result["frame_end"] != expected_end:
        raise CompileError(
            "timeline 不一致：duration_sec × fps 对应 frame_end={}，配置为 {}".format(
                expected_end, result["frame_end"]
            )
        )

    result["frame_count"] = result["frame_end"] - result["frame_start"] + 1
    return result


def parse_court(event_config: Dict[str, Any]) -> Dict[str, float]:
    raw = event_config.get("court")
    if not isinstance(raw, dict):
        raise CompileError("court 必须是 JSON object")

    court = {
        "x_min_cm": require_float(raw.get("x_min_cm"), "court.x_min_cm"),
        "x_max_cm": require_float(raw.get("x_max_cm"), "court.x_max_cm"),
        "y_min_cm": require_float(raw.get("y_min_cm"), "court.y_min_cm"),
        "y_max_cm": require_float(raw.get("y_max_cm"), "court.y_max_cm"),
        "player_z_cm": require_float(
            raw.get("player_z_cm"), "court.player_z_cm"
        ),
        "ball_z_cm": require_float(raw.get("ball_z_cm"), "court.ball_z_cm"),
    }

    if court["x_min_cm"] >= court["x_max_cm"]:
        raise CompileError("court.x_min_cm 必须小于 x_max_cm")
    if court["y_min_cm"] >= court["y_max_cm"]:
        raise CompileError("court.y_min_cm 必须小于 y_max_cm")
    return court


def parse_defaults(event_config: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(DEFAULTS)
    raw = event_config.get("compiler_defaults", {})
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise CompileError("compiler_defaults 必须是 JSON object")

    for key, value in raw.items():
        result[key] = value

    numeric_keys = [
        "player_max_speed_cm_s",
        "ball_max_speed_cm_s",
        "dribble_ball_ahead_cm",
        "pass_arc_height_cm",
        "shot_arc_height_cm",
        "defender_default_follow_distance_cm",
        "defender_default_side_offset_cm",
        "defender_follow_speed_cm_s",
        "defender_response_alpha",
        "defender_response_time_sec",
        "defender_max_acceleration_cm_s2",
        "defender_avoidance_radius_cm",
        "defender_avoidance_weight",
        "court_player_margin_cm",
    ]
    for key in numeric_keys:
        result[key] = require_float(result[key], "compiler_defaults." + key)

    result["dense_keyframe_interval_frames"] = require_int(
        result.get("dense_keyframe_interval_frames", 1),
        "compiler_defaults.dense_keyframe_interval_frames",
    )
    result["defender_lookahead_frames"] = require_int(
        result.get("defender_lookahead_frames", 5),
        "compiler_defaults.defender_lookahead_frames",
    )

    if result["dense_keyframe_interval_frames"] != 1:
        raise CompileError(
            "A3.2 当前要求 dense_keyframe_interval_frames=1"
        )
    if not 0.0 < result["defender_response_alpha"] <= 1.0:
        raise CompileError("defender_response_alpha 必须位于 (0, 1]")
    if result["defender_response_time_sec"] <= 0.0:
        raise CompileError("defender_response_time_sec 必须大于 0")
    if result["defender_max_acceleration_cm_s2"] <= 0.0:
        raise CompileError("defender_max_acceleration_cm_s2 必须大于 0")
    if result["defender_lookahead_frames"] < 0:
        raise CompileError("defender_lookahead_frames 不能小于 0")
    if result["defender_avoidance_radius_cm"] < 0.0:
        raise CompileError("defender_avoidance_radius_cm 不能小于 0")
    if not 0.0 <= result["defender_avoidance_weight"] <= 2.0:
        raise CompileError("defender_avoidance_weight 必须位于 [0, 2]")
    return result


def parse_players(
    event_config: Dict[str, Any],
    defaults: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    raw_players = event_config.get("players")
    if not isinstance(raw_players, dict) or not raw_players:
        raise CompileError("players 必须是非空 JSON object")

    players: Dict[str, Dict[str, Any]] = {}
    seen_track_ids: Set[int] = set()

    for player_id, raw in raw_players.items():
        if not isinstance(raw, dict):
            raise CompileError("players.{} 必须是 JSON object".format(player_id))

        player = {
            "team": require_str(raw.get("team"), "players.{}.team".format(player_id)),
            "role": require_str(raw.get("role"), "players.{}.role".format(player_id)),
            "track_id": require_int(
                raw.get("track_id"), "players.{}.track_id".format(player_id)
            ),
            "class_id": require_int(
                raw.get("class_id"), "players.{}.class_id".format(player_id)
            ),
            "start_loc": parse_vec3(
                raw.get("start_loc"), "players.{}.start_loc".format(player_id)
            ),
            "max_speed_cm_s": require_float(
                raw.get("max_speed_cm_s", defaults["player_max_speed_cm_s"]),
                "players.{}.max_speed_cm_s".format(player_id),
            ),
        }

        if player["track_id"] in seen_track_ids:
            raise CompileError(
                "重复 player track_id={}".format(player["track_id"])
            )
        seen_track_ids.add(player["track_id"])
        players[player_id] = player

    return players


def parse_ball(event_config: Dict[str, Any]) -> Dict[str, Any]:
    raw = event_config.get("ball")
    if not isinstance(raw, dict):
        raise CompileError("ball 必须是 JSON object")

    return {
        "object_id": require_str(raw.get("object_id"), "ball.object_id"),
        "track_id": require_int(raw.get("track_id"), "ball.track_id"),
        "class_id": require_int(raw.get("class_id"), "ball.class_id"),
        "initial_owner": require_str(
            raw.get("initial_owner"), "ball.initial_owner"
        ),
        "initial_loc": parse_vec3(raw.get("initial_loc"), "ball.initial_loc"),
        "height_cm": require_float(raw.get("height_cm"), "ball.height_cm"),
        "dribble_ahead_cm": require_float(
            raw.get("dribble_ahead_cm", DEFAULTS["dribble_ball_ahead_cm"]),
            "ball.dribble_ahead_cm",
        ),
    }


def parse_events(
    event_config: Dict[str, Any],
    timeline: Dict[str, Any],
    players: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    raw_events = event_config.get("events")
    if not isinstance(raw_events, list) or not raw_events:
        raise CompileError("events 必须是非空数组")

    result: List[Dict[str, Any]] = []
    seen_ids: Set[str] = set()

    for index, raw in enumerate(raw_events):
        if not isinstance(raw, dict):
            raise CompileError("events[{}] 必须是 JSON object".format(index))

        event = copy.deepcopy(raw)
        event_id = require_str(raw.get("event_id"), "events[{}].event_id".format(index))
        event_type = require_str(raw.get("type"), "events[{}].type".format(index))

        if event_id in seen_ids:
            raise CompileError("重复 event_id={}".format(event_id))
        seen_ids.add(event_id)

        if event_type not in SUPPORTED_EVENT_TYPES:
            raise CompileError(
                "{} 使用不支持的 event type={!r}".format(event_id, event_type)
            )

        event["event_id"] = event_id
        event["type"] = event_type

        if event_type == "receive":
            event["start_t"] = require_float(
                raw.get("time"), "{}.time".format(event_id)
            )
            event["end_t"] = event["start_t"]
            event["start_frame"] = time_to_frame(event["start_t"], timeline["fps"])
            event["end_frame_exclusive"] = event["start_frame"]
            event["instantaneous"] = True
        else:
            event["start_t"] = require_float(
                raw.get("start_t"), "{}.start_t".format(event_id)
            )
            event["end_t"] = require_float(
                raw.get("end_t"), "{}.end_t".format(event_id)
            )
            if event["end_t"] <= event["start_t"]:
                raise CompileError("{} 的 end_t 必须大于 start_t".format(event_id))
            event["start_frame"] = time_to_frame(event["start_t"], timeline["fps"])
            event["end_frame_exclusive"] = time_to_frame(
                event["end_t"], timeline["fps"]
            )
            event["instantaneous"] = False

        if event["start_t"] < 0.0 or event["end_t"] > timeline["duration_sec"] + 1e-9:
            raise CompileError(
                "{} 时间超出回合范围：{}..{}".format(
                    event_id, event["start_t"], event["end_t"]
                )
            )

        actor = get_event_actor(event)
        if actor is not None and actor not in players:
            raise CompileError(
                "{} 引用了不存在的球员 {}".format(event_id, actor)
            )

        if event_type in {"move", "dribble", "shot"}:
            event["target_loc"] = parse_vec3(
                raw.get("target_loc"), "{}.target_loc".format(event_id)
            )

        if event_type == "pass":
            from_player = require_str(raw.get("from"), "{}.from".format(event_id))
            to_player = require_str(raw.get("to"), "{}.to".format(event_id))
            if from_player not in players or to_player not in players:
                raise CompileError("{} 的传球球员不存在".format(event_id))
            event["from"] = from_player
            event["to"] = to_player

        if event_type == "receive":
            event["actor"] = require_str(raw.get("actor"), "{}.actor".format(event_id))
            event["source_event"] = require_str(
                raw.get("source_event"), "{}.source_event".format(event_id)
            )

        result.append(event)

    result.sort(
        key=lambda event: (
            event["start_frame"],
            0 if event["type"] == "receive" else 1,
            event["event_id"],
        )
    )
    return result


def events_by_actor(events: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    result: Dict[str, List[Dict[str, Any]]] = {}
    for event in events:
        if event["type"] not in PLAYER_MOTION_TYPES:
            continue
        actor = get_event_actor(event)
        if actor is not None:
            result.setdefault(actor, []).append(event)

    for actor_events in result.values():
        actor_events.sort(
            key=lambda event: (
                event["start_frame"],
                event["end_frame_exclusive"],
                event["event_id"],
            )
        )
    return result


def build_pass_receive_map(events: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    event_by_id = {event["event_id"]: event for event in events}
    result: Dict[str, Dict[str, Any]] = {}

    for event in events:
        if event["type"] != "receive":
            continue
        source_id = event["source_event"]
        source = event_by_id.get(source_id)
        if source is None or source["type"] != "pass":
            raise CompileError(
                "{} 的 source_event={} 不是有效 pass".format(
                    event["event_id"], source_id
                )
            )
        if source.get("to") != event.get("actor"):
            raise CompileError(
                "{} 的接球者与 {} 的传球目标不一致".format(
                    event["event_id"], source_id
                )
            )
        result[source_id] = event

    for event in events:
        if event["type"] == "pass" and event["event_id"] not in result:
            raise CompileError(
                "{} 没有匹配 receive".format(event["event_id"])
            )
    return result


def build_possession_frames(
    events: Sequence[Dict[str, Any]],
    pass_receive_map: Dict[str, Dict[str, Any]],
    initial_owner: str,
    timeline: Dict[str, Any],
) -> List[Dict[str, Any]]:
    transitions: Dict[int, List[Tuple[int, Dict[str, Any]]]] = {}

    transitions.setdefault(timeline["frame_start"], []).append(
        (0, {"state": "owned", "owner": initial_owner, "event_id": None})
    )

    for event in events:
        if event["type"] == "pass":
            receive = pass_receive_map[event["event_id"]]
            transitions.setdefault(event["start_frame"], []).append(
                (
                    20,
                    {
                        "state": "in_transit",
                        "owner": None,
                        "from": event["from"],
                        "to": event["to"],
                        "event_id": event["event_id"],
                        "flight_end_frame": receive["start_frame"],
                    },
                )
            )
        elif event["type"] == "receive":
            transitions.setdefault(event["start_frame"], []).append(
                (
                    10,
                    {
                        "state": "owned",
                        "owner": event["actor"],
                        "event_id": event["event_id"],
                    },
                )
            )
        elif event["type"] == "shot":
            transitions.setdefault(event["start_frame"], []).append(
                (
                    30,
                    {
                        "state": "shot",
                        "owner": None,
                        "from": event["actor"],
                        "to": None,
                        "event_id": event["event_id"],
                        "flight_end_frame": event["end_frame_exclusive"],
                    },
                )
            )

    current: Optional[Dict[str, Any]] = None
    result: List[Dict[str, Any]] = []

    for frame in range(timeline["frame_start"], timeline["frame_end"] + 1):
        for _, transition in sorted(transitions.get(frame, []), key=lambda item: item[0]):
            current = copy.deepcopy(transition)

        if current is None:
            raise CompileError("frame={} 无持球权状态".format(frame))

        frame_state = copy.deepcopy(current)
        frame_state["frame"] = frame
        result.append(frame_state)

    return result


def compress_possession_timeline(
    frame_states: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if not frame_states:
        return []

    ignored = {"frame", "flight_end_frame"}

    def signature(item: Dict[str, Any]) -> Tuple[Tuple[str, Any], ...]:
        return tuple(
            sorted((key, value) for key, value in item.items() if key not in ignored)
        )

    result: List[Dict[str, Any]] = []
    start = frame_states[0]["frame"]
    previous = frame_states[0]
    previous_signature = signature(previous)

    for item in frame_states[1:]:
        current_signature = signature(item)
        if current_signature != previous_signature:
            segment = {key: value for key, value in previous.items() if key != "frame"}
            segment["start_frame"] = start
            segment["end_frame"] = item["frame"] - 1
            segment.pop("flight_end_frame", None)
            result.append(segment)
            start = item["frame"]
            previous_signature = current_signature
        previous = item

    segment = {key: value for key, value in previous.items() if key != "frame"}
    segment["start_frame"] = start
    segment["end_frame"] = frame_states[-1]["frame"]
    segment.pop("flight_end_frame", None)
    result.append(segment)
    return result


def target_dependencies_for_actor(
    actor: str,
    actor_events: Sequence[Dict[str, Any]],
    events: Sequence[Dict[str, Any]],
    initial_owner: str,
) -> Set[str]:
    dependencies: Set[str] = set()
    has_possession_target = False

    for event in actor_events:
        if event["type"] != "defend_follow":
            continue
        target = event.get("target")
        if target == "possession_owner":
            has_possession_target = True
        elif isinstance(target, str):
            dependencies.add(target)

    if has_possession_target:
        dependencies.add(initial_owner)
        for event in events:
            if event["type"] == "pass":
                dependencies.add(event["from"])
                dependencies.add(event["to"])
            elif event["type"] in {"hold", "dribble", "receive", "shot"}:
                actor_value = get_event_actor(event)
                if actor_value is not None:
                    dependencies.add(actor_value)

    dependencies.discard(actor)
    return dependencies


def calculate_direction_series(
    positions: Sequence[Sequence[float]],
) -> List[Tuple[float, float]]:
    count = len(positions)
    raw: List[Optional[Tuple[float, float]]] = [None] * count

    for index in range(count):
        if count == 1:
            break
        if index == 0:
            dx = positions[1][0] - positions[0][0]
            dy = positions[1][1] - positions[0][1]
        elif index == count - 1:
            dx = positions[-1][0] - positions[-2][0]
            dy = positions[-1][1] - positions[-2][1]
        else:
            dx = positions[index + 1][0] - positions[index - 1][0]
            dy = positions[index + 1][1] - positions[index - 1][1]

        if math.hypot(dx, dy) > 1e-7:
            raw[index] = normalize_xy(dx, dy)

    last: Optional[Tuple[float, float]] = None
    for index in range(count):
        if raw[index] is not None:
            last = raw[index]
        elif last is not None:
            raw[index] = last

    next_value: Optional[Tuple[float, float]] = None
    for index in range(count - 1, -1, -1):
        if raw[index] is not None:
            next_value = raw[index]
        elif next_value is not None:
            raw[index] = next_value

    return [value if value is not None else (1.0, 0.0) for value in raw]


def clamp_player_location(
    loc: Sequence[float],
    court: Dict[str, float],
    margin_cm: float,
) -> Tuple[float, float, float]:
    return (
        clamp(loc[0], court["x_min_cm"] + margin_cm, court["x_max_cm"] - margin_cm),
        clamp(loc[1], court["y_min_cm"] + margin_cm, court["y_max_cm"] - margin_cm),
        loc[2],
    )


def compile_actor_path(
    actor: str,
    player: Dict[str, Any],
    actor_events: Sequence[Dict[str, Any]],
    compiled_paths: Dict[str, List[Tuple[float, float, float]]],
    possession_frames: Sequence[Dict[str, Any]],
    timeline: Dict[str, Any],
    court: Dict[str, float],
    defaults: Dict[str, Any],
) -> List[Tuple[float, float, float]]:
    frame_start = timeline["frame_start"]
    frame_end = timeline["frame_end"]
    count = timeline["frame_count"]
    fps = timeline["fps"]

    positions: List[Tuple[float, float, float]] = [
        tuple(player["start_loc"]) for _ in range(count)
    ]

    def local_index(frame: int) -> int:
        return frame - frame_start

    for event in actor_events:
        start_frame = max(frame_start, event["start_frame"])
        end_boundary = min(frame_end, event["end_frame_exclusive"])
        start_index = local_index(start_frame)
        end_index = local_index(end_boundary)
        start_loc = positions[start_index]
        event_type = event["type"]

        if event_type in {"move", "dribble"}:
            target = tuple(event["target_loc"])
            target = clamp_player_location(
                target, court, defaults["court_player_margin_cm"]
            )
            sample_span = max(1, end_index - start_index)

            for index in range(start_index, end_index + 1):
                u = (index - start_index) / sample_span
                s = smoothstep01(u)
                positions[index] = (
                    lerp(start_loc[0], target[0], s),
                    lerp(start_loc[1], target[1], s),
                    lerp(start_loc[2], target[2], s),
                )

            final_loc = positions[end_index]
            for index in range(end_index, count):
                positions[index] = final_loc

        elif event_type in {"hold", "pass", "shot"}:
            for index in range(start_index, end_index + 1):
                positions[index] = start_loc
            for index in range(end_index, count):
                positions[index] = start_loc

        elif event_type == "defend_follow":
            target_spec = event.get("target")
            follow_distance = require_float(
                event.get(
                    "follow_distance_cm",
                    defaults["defender_default_follow_distance_cm"],
                ),
                "{}.follow_distance_cm".format(event["event_id"]),
            )
            side_offset = require_float(
                event.get(
                    "side_offset_cm",
                    defaults["defender_default_side_offset_cm"],
                ),
                "{}.side_offset_cm".format(event["event_id"]),
            )
            positioning = str(event.get("positioning", "trailing")).strip().lower()
            if positioning not in {"trailing", "goal_side"}:
                raise CompileError(
                    "{}.positioning 必须是 trailing 或 goal_side".format(
                        event["event_id"]
                    )
                )
            longitudinal_sign = 1.0 if positioning == "goal_side" else -1.0
            follow_speed = require_float(
                event.get(
                    "max_speed_cm_s",
                    min(
                        player["max_speed_cm_s"],
                        defaults["defender_follow_speed_cm_s"],
                    ),
                ),
                "{}.max_speed_cm_s".format(event["event_id"]),
            )
            follow_speed = min(follow_speed, player["max_speed_cm_s"])
            if follow_speed <= 0.0:
                raise CompileError(
                    "{}.max_speed_cm_s 必须大于 0".format(event["event_id"])
                )

            response_time_raw = event.get("response_time_sec")
            if response_time_raw is None:
                # Backward compatibility with the V1 response_alpha schema.
                response_alpha = require_float(
                    event.get(
                        "response_alpha",
                        defaults["defender_response_alpha"],
                    ),
                    "{}.response_alpha".format(event["event_id"]),
                )
                if not 0.0 < response_alpha <= 1.0:
                    raise CompileError(
                        "{}.response_alpha 必须位于 (0, 1]".format(
                            event["event_id"]
                        )
                    )
                response_time = max(1.0 / fps, 1.0 / (response_alpha * fps))
            else:
                response_time = require_float(
                    response_time_raw,
                    "{}.response_time_sec".format(event["event_id"]),
                )
            if response_time <= 0.0:
                raise CompileError(
                    "{}.response_time_sec 必须大于 0".format(event["event_id"])
                )

            max_acceleration = require_float(
                event.get(
                    "max_acceleration_cm_s2",
                    defaults["defender_max_acceleration_cm_s2"],
                ),
                "{}.max_acceleration_cm_s2".format(event["event_id"]),
            )
            lookahead_frames = require_int(
                event.get(
                    "lookahead_frames",
                    defaults["defender_lookahead_frames"],
                ),
                "{}.lookahead_frames".format(event["event_id"]),
            )
            avoidance_radius = require_float(
                event.get(
                    "avoidance_radius_cm",
                    defaults["defender_avoidance_radius_cm"],
                ),
                "{}.avoidance_radius_cm".format(event["event_id"]),
            )
            avoidance_weight = require_float(
                event.get(
                    "avoidance_weight",
                    defaults["defender_avoidance_weight"],
                ),
                "{}.avoidance_weight".format(event["event_id"]),
            )
            if max_acceleration <= 0.0:
                raise CompileError(
                    "{}.max_acceleration_cm_s2 必须大于 0".format(
                        event["event_id"]
                    )
                )
            if lookahead_frames < 0:
                raise CompileError(
                    "{}.lookahead_frames 不能小于 0".format(event["event_id"])
                )
            if avoidance_radius < 0.0 or not 0.0 <= avoidance_weight <= 2.0:
                raise CompileError(
                    "{} avoidance 参数无效".format(event["event_id"])
                )

            direction_cache = {
                player_id: calculate_direction_series(path)
                for player_id, path in compiled_paths.items()
            }

            current = start_loc
            if start_index > 0:
                previous = positions[start_index - 1]
                velocity = (
                    (current[0] - previous[0]) * fps,
                    (current[1] - previous[1]) * fps,
                )
            else:
                velocity = (0.0, 0.0)
            velocity = limit_xy_length(
                velocity[0], velocity[1], follow_speed
            )
            positions[start_index] = current

            for index in range(start_index + 1, end_index + 1):
                frame = index + frame_start
                if target_spec == "possession_owner":
                    state = possession_frames[index]
                    if state["state"] == "owned":
                        target_player = state.get("owner")
                    elif state["state"] == "in_transit":
                        target_player = state.get("to")
                    elif state["state"] == "shot":
                        target_player = state.get("from")
                    else:
                        target_player = None
                else:
                    target_player = target_spec

                if target_player not in compiled_paths:
                    raise CompileError(
                        "{} frame={} 无法解析 defend_follow target={}".format(
                            event["event_id"], frame, target_player
                        )
                    )

                target_path = compiled_paths[target_player]
                target_index = min(
                    index + lookahead_frames, len(target_path) - 1
                )
                target_pos = target_path[target_index]
                forward = direction_cache[target_player][target_index]
                right = (-forward[1], forward[0])

                desired_x = (
                    target_pos[0]
                    + longitudinal_sign * forward[0] * follow_distance
                    + right[0] * side_offset
                )
                desired_y = (
                    target_pos[1]
                    + longitudinal_sign * forward[1] * follow_distance
                    + right[1] * side_offset
                )
                avoid_x, avoid_y = defender_avoidance_offset(
                    current,
                    index,
                    compiled_paths,
                    target_player,
                    avoidance_radius,
                    avoidance_weight,
                )
                desired = clamp_player_location(
                    (
                        desired_x + avoid_x,
                        desired_y + avoid_y,
                        player["start_loc"][2],
                    ),
                    court,
                    defaults["court_player_margin_cm"],
                )

                dx = desired[0] - current[0]
                dy = desired[1] - current[1]
                distance = math.hypot(dx, dy)
                if distance <= 1e-6:
                    desired_velocity = (0.0, 0.0)
                else:
                    braking_speed = math.sqrt(max(0.0, 2.0 * max_acceleration * distance))
                    desired_speed = min(
                        follow_speed,
                        distance / response_time,
                        braking_speed,
                    )
                    desired_velocity = (
                        dx / distance * desired_speed,
                        dy / distance * desired_speed,
                    )

                velocity = acceleration_limited_velocity(
                    velocity,
                    desired_velocity,
                    max_acceleration,
                    fps,
                    follow_speed,
                )
                step_x = velocity[0] / fps
                step_y = velocity[1] / fps
                step_distance = math.hypot(step_x, step_y)

                if distance <= step_distance + 1e-9:
                    next_loc = desired
                    velocity = (0.0, 0.0)
                else:
                    next_loc = clamp_player_location(
                        (
                            current[0] + step_x,
                            current[1] + step_y,
                            player["start_loc"][2],
                        ),
                        court,
                        defaults["court_player_margin_cm"],
                    )
                    velocity = (
                        (next_loc[0] - current[0]) * fps,
                        (next_loc[1] - current[1]) * fps,
                    )

                positions[index] = next_loc
                current = next_loc

            final_loc = positions[end_index]
            for index in range(end_index, count):
                positions[index] = final_loc

        else:
            raise CompileError(
                "不支持的球员事件类型：{}".format(event_type)
            )

    return positions


def compile_player_paths(
    players: Dict[str, Dict[str, Any]],
    events: Sequence[Dict[str, Any]],
    possession_frames: Sequence[Dict[str, Any]],
    timeline: Dict[str, Any],
    court: Dict[str, float],
    defaults: Dict[str, Any],
    initial_owner: str,
) -> Dict[str, List[Tuple[float, float, float]]]:
    actor_map = events_by_actor(events)
    compiled: Dict[str, List[Tuple[float, float, float]]] = {}

    pending: Set[str] = set(players.keys())

    while pending:
        progress = False

        for actor in sorted(list(pending)):
            actor_events = actor_map.get(actor, [])
            dependencies = target_dependencies_for_actor(
                actor, actor_events, events, initial_owner
            )

            if not dependencies.issubset(compiled.keys()):
                continue

            compiled[actor] = compile_actor_path(
                actor,
                players[actor],
                actor_events,
                compiled,
                possession_frames,
                timeline,
                court,
                defaults,
            )
            pending.remove(actor)
            progress = True

        if not progress:
            unresolved = {
                actor: sorted(
                    target_dependencies_for_actor(
                        actor,
                        actor_map.get(actor, []),
                        events,
                        initial_owner,
                    )
                    - compiled.keys()
                )
                for actor in sorted(pending)
            }
            raise CompileError(
                "defend_follow 依赖无法解析或形成循环：{}".format(unresolved)
            )

    return compiled


def active_dribble_ahead_by_frame(
    events: Sequence[Dict[str, Any]],
    timeline: Dict[str, Any],
    default_ahead: float,
) -> Dict[Tuple[str, int], float]:
    result: Dict[Tuple[str, int], float] = {}
    for event in events:
        if event["type"] != "dribble":
            continue
        ahead = require_float(
            event.get("ball_ahead_cm", default_ahead),
            "{}.ball_ahead_cm".format(event["event_id"]),
        )
        start = max(timeline["frame_start"], event["start_frame"])
        end = min(timeline["frame_end"], event["end_frame_exclusive"] - 1)
        for frame in range(start, end + 1):
            result[(event["actor"], frame)] = ahead
    return result


def foot_ball_position(
    player_id: str,
    frame: int,
    ahead_cm: float,
    player_paths: Dict[str, List[Tuple[float, float, float]]],
    direction_series: Dict[str, List[Tuple[float, float]]],
    timeline: Dict[str, Any],
    ball_height_cm: float,
) -> Tuple[float, float, float]:
    index = frame - timeline["frame_start"]
    player_loc = player_paths[player_id][index]
    direction = direction_series[player_id][index]
    return (
        player_loc[0] + direction[0] * ahead_cm,
        player_loc[1] + direction[1] * ahead_cm,
        ball_height_cm,
    )


def compile_ball_path(
    ball: Dict[str, Any],
    events: Sequence[Dict[str, Any]],
    possession_frames: Sequence[Dict[str, Any]],
    player_paths: Dict[str, List[Tuple[float, float, float]]],
    timeline: Dict[str, Any],
    defaults: Dict[str, Any],
) -> List[Tuple[float, float, float]]:
    event_by_id = {event["event_id"]: event for event in events}
    directions = {
        player_id: calculate_direction_series(path)
        for player_id, path in player_paths.items()
    }
    dribble_ahead = active_dribble_ahead_by_frame(
        events, timeline, ball["dribble_ahead_cm"]
    )

    count = timeline["frame_count"]
    positions: List[Tuple[float, float, float]] = [
        tuple(ball["initial_loc"]) for _ in range(count)
    ]

    pass_cache: Dict[str, Dict[str, Any]] = {}
    shot_cache: Dict[str, Dict[str, Any]] = {}

    for event in events:
        if event["type"] == "pass":
            start_frame = event["start_frame"]
            end_frame = event["end_frame_exclusive"]
            start_pos = foot_ball_position(
                event["from"],
                start_frame,
                ball["dribble_ahead_cm"],
                player_paths,
                directions,
                timeline,
                ball["height_cm"],
            )
            end_pos = foot_ball_position(
                event["to"],
                end_frame,
                ball["dribble_ahead_cm"],
                player_paths,
                directions,
                timeline,
                ball["height_cm"],
            )
            pass_cache[event["event_id"]] = {
                "start_frame": start_frame,
                "end_frame": end_frame,
                "start_pos": start_pos,
                "end_pos": end_pos,
                "arc_height_cm": require_float(
                    event.get("arc_height_cm", defaults["pass_arc_height_cm"]),
                    "{}.arc_height_cm".format(event["event_id"]),
                ),
            }

        elif event["type"] == "shot":
            start_frame = event["start_frame"]
            end_frame = event["end_frame_exclusive"]
            start_pos = foot_ball_position(
                event["actor"],
                start_frame,
                ball["dribble_ahead_cm"],
                player_paths,
                directions,
                timeline,
                ball["height_cm"],
            )
            shot_cache[event["event_id"]] = {
                "start_frame": start_frame,
                "end_frame": end_frame,
                "start_pos": start_pos,
                "end_pos": tuple(event["target_loc"]),
                "arc_height_cm": require_float(
                    event.get("arc_height_cm", defaults["shot_arc_height_cm"]),
                    "{}.arc_height_cm".format(event["event_id"]),
                ),
            }

    for index, state in enumerate(possession_frames):
        frame = state["frame"]

        if state["state"] == "owned":
            owner = state.get("owner")
            if owner not in player_paths:
                raise CompileError(
                    "frame={} 的 owner={} 不存在".format(frame, owner)
                )
            ahead = dribble_ahead.get(
                (owner, frame), ball["dribble_ahead_cm"]
            )
            positions[index] = foot_ball_position(
                owner,
                frame,
                ahead,
                player_paths,
                directions,
                timeline,
                ball["height_cm"],
            )

        elif state["state"] == "in_transit":
            flight = pass_cache.get(state.get("event_id"))
            if flight is None:
                raise CompileError(
                    "frame={} 找不到 pass flight {}".format(
                        frame, state.get("event_id")
                    )
                )
            span = max(1, flight["end_frame"] - flight["start_frame"])
            u = clamp((frame - flight["start_frame"]) / span, 0.0, 1.0)
            s = smoothstep01(u)
            base_z = lerp(flight["start_pos"][2], flight["end_pos"][2], s)
            positions[index] = (
                lerp(flight["start_pos"][0], flight["end_pos"][0], s),
                lerp(flight["start_pos"][1], flight["end_pos"][1], s),
                base_z + 4.0 * flight["arc_height_cm"] * u * (1.0 - u),
            )

        elif state["state"] == "shot":
            flight = shot_cache.get(state.get("event_id"))
            if flight is None:
                raise CompileError(
                    "frame={} 找不到 shot flight {}".format(
                        frame, state.get("event_id")
                    )
                )
            if frame >= flight["end_frame"]:
                positions[index] = flight["end_pos"]
            else:
                span = max(1, flight["end_frame"] - flight["start_frame"])
                u = clamp((frame - flight["start_frame"]) / span, 0.0, 1.0)
                s = smoothstep01(u)
                base_z = lerp(flight["start_pos"][2], flight["end_pos"][2], s)
                positions[index] = (
                    lerp(flight["start_pos"][0], flight["end_pos"][0], s),
                    lerp(flight["start_pos"][1], flight["end_pos"][1], s),
                    base_z + 4.0 * flight["arc_height_cm"] * u * (1.0 - u),
                )
        else:
            raise CompileError(
                "frame={} 使用未知 ball state={}".format(frame, state["state"])
            )

    # Preserve the event configuration's exact initial ball position.
    positions[0] = tuple(ball["initial_loc"])
    return positions


def calculate_path_stats(
    positions: Sequence[Sequence[float]],
    fps: float,
    frame_start: int,
) -> Dict[str, Any]:
    total_distance_xy = 0.0
    max_speed_xy = 0.0
    max_speed_3d = 0.0
    max_speed_frame: Optional[int] = None

    for index, (left, right) in enumerate(zip(positions, positions[1:]), start=1):
        dx = right[0] - left[0]
        dy = right[1] - left[1]
        dz = right[2] - left[2]
        dist_xy = math.hypot(dx, dy)
        dist_3d = math.sqrt(dx * dx + dy * dy + dz * dz)
        speed_xy = dist_xy * fps
        speed_3d = dist_3d * fps
        total_distance_xy += dist_xy

        if speed_xy > max_speed_xy:
            max_speed_xy = speed_xy
            max_speed_frame = frame_start + index
        max_speed_3d = max(max_speed_3d, speed_3d)

    return {
        "keyframe_count": len(positions),
        "total_distance_xy_cm": total_distance_xy,
        "max_speed_xy_cm_s": max_speed_xy,
        "max_speed_3d_cm_s": max_speed_3d,
        "max_speed_frame": max_speed_frame,
    }


def round_positions_to_keyframes(
    positions: Sequence[Sequence[float]],
    frame_start: int,
    decimals: int,
) -> List[Dict[str, Any]]:
    return [
        {
            "frame": frame_start + index,
            "loc": [round(float(value), decimals) for value in loc],
        }
        for index, loc in enumerate(positions)
    ]


def find_base_object_template(
    base_config: Dict[str, Any],
    object_id: str,
    category: str,
) -> Dict[str, Any]:
    objects = base_config.get("objects", {})
    if isinstance(objects, dict) and isinstance(objects.get(object_id), dict):
        template = copy.deepcopy(objects[object_id])
        template.pop("keyframes", None)
        template.pop("start", None)
        template.pop("end", None)
        return template

    if category == "player":
        for raw in objects.values() if isinstance(objects, dict) else []:
            if isinstance(raw, dict) and raw.get("category") == "player":
                template = copy.deepcopy(raw)
                template.pop("keyframes", None)
                template.pop("start", None)
                template.pop("end", None)
                return template
        return {
            "category": "player",
            "scale": [1.0, 1.0, 1.0],
            "use_animation": True,
        }

    if isinstance(objects, dict):
        for raw in objects.values():
            if isinstance(raw, dict) and raw.get("category") == "ball":
                template = copy.deepcopy(raw)
                template.pop("keyframes", None)
                template.pop("start", None)
                template.pop("end", None)
                return template

    return {
        "category": "ball",
        "scale": [0.5, 0.5, 0.5],
        "use_animation": False,
    }


def build_output_config(
    event_config: Dict[str, Any],
    event_config_path: Path,
    base_config: Dict[str, Any],
    base_config_path: Path,
    timeline: Dict[str, Any],
    players: Dict[str, Dict[str, Any]],
    ball: Dict[str, Any],
    events: Sequence[Dict[str, Any]],
    player_paths: Dict[str, List[Tuple[float, float, float]]],
    ball_path: List[Tuple[float, float, float]],
    possession_frames: Sequence[Dict[str, Any]],
    defaults: Dict[str, Any],
    decimals: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    output = copy.deepcopy(base_config)
    output_seq_id = require_str(
        event_config.get("output_seq_id"), "output_seq_id"
    )
    episode_id = require_str(event_config.get("episode_id"), "episode_id")

    output["schema_version"] = "3.0"
    output["seq_id"] = output_seq_id
    output["timeline"] = {
        "frame_start": timeline["frame_start"],
        "frame_end": timeline["frame_end"],
        "display_rate": timeline["fps"],
    }

    output_objects: Dict[str, Any] = {}
    track_id_map: Dict[str, int] = {}
    object_stats: Dict[str, Any] = {}

    for player_id, player in players.items():
        template = find_base_object_template(base_config, player_id, "player")
        template["category"] = "player"
        template["class_id"] = player["class_id"]
        template["track_id"] = player["track_id"]
        template["team"] = player["team"]
        template["role"] = player["role"]
        template["use_animation"] = bool(template.get("use_animation", True))
        template["interpolation"] = "linear"
        template["keyframes"] = round_positions_to_keyframes(
            player_paths[player_id], timeline["frame_start"], decimals
        )
        output_objects[player_id] = template
        track_id_map[player_id] = player["track_id"]
        object_stats[player_id] = calculate_path_stats(
            player_paths[player_id], timeline["fps"], timeline["frame_start"]
        )
        object_stats[player_id]["category"] = "player"

    ball_id = ball["object_id"]
    ball_template = find_base_object_template(base_config, ball_id, "ball")
    ball_template["category"] = "ball"
    ball_template["class_id"] = ball["class_id"]
    ball_template["track_id"] = ball["track_id"]
    ball_template["use_animation"] = False
    ball_template["interpolation"] = "linear"
    ball_template["keyframes"] = round_positions_to_keyframes(
        ball_path, timeline["frame_start"], decimals
    )
    output_objects[ball_id] = ball_template
    track_id_map[ball_id] = ball["track_id"]
    object_stats[ball_id] = calculate_path_stats(
        ball_path, timeline["fps"], timeline["frame_start"]
    )
    object_stats[ball_id]["category"] = "ball"

    output["objects"] = output_objects
    output["track_id_map"] = track_id_map
    if isinstance(event_config.get("roster"), dict):
        output["roster"] = copy.deepcopy(event_config["roster"])
    if isinstance(event_config.get("movement_optimization"), dict):
        output["movement_optimization"] = copy.deepcopy(
            event_config["movement_optimization"]
        )

    class_id_map = copy.deepcopy(output.get("class_id_map", {}))
    if not isinstance(class_id_map, dict):
        class_id_map = {}
    class_id_map["player"] = next(iter(players.values()))["class_id"]
    class_id_map["ball"] = ball["class_id"]
    output["class_id_map"] = class_id_map

    # Remove metadata that describes the source test trajectory rather than this episode.
    output.pop("trajectory_generation", None)

    event_timeline = []
    for event in events:
        item = {
            "event_id": event["event_id"],
            "type": event["type"],
            "actor": get_event_actor(event),
            "start_t": event["start_t"],
            "end_t": event["end_t"],
            "start_frame": event["start_frame"],
            "end_frame_exclusive": event["end_frame_exclusive"],
            "last_frame": (
                event["start_frame"]
                if event["instantaneous"]
                else min(timeline["frame_end"], event["end_frame_exclusive"] - 1)
            ),
        }
        for key in ("from", "to", "target", "source_event"):
            if key in event:
                item[key] = event[key]
        event_timeline.append(item)

    possession_timeline = compress_possession_timeline(possession_frames)
    output["possession_timeline"] = possession_timeline
    output["event_timeline"] = event_timeline
    output["episode_metadata"] = {
        "compiler_version": SCRIPT_VERSION,
        "compiled_at_utc": datetime.now(timezone.utc).isoformat(),
        "episode_id": episode_id,
        "source_event_config": str(event_config_path.resolve()).replace("\\", "/"),
        "base_render_config": str(base_config_path.resolve()).replace("\\", "/"),
        "event_count": len(events),
        "trajectory_type": "event_compiled_dense_v1",
        "dense_keyframe_interval_frames": 1,
        "coordinate_decimals": decimals,
        "player_motion_interpolation": "smoothstep",
        "defend_follow_method": "goal_side_predictive_acceleration_limited_pursuit_v2",
        "player_count": len(players),
        "ball_flight_method": "linear_xy_parabolic_z",
        "object_stats": object_stats,
        "compiler_defaults": defaults,
    }

    summary = {
        "episode_id": episode_id,
        "output_seq_id": output_seq_id,
        "frame_start": timeline["frame_start"],
        "frame_end": timeline["frame_end"],
        "fps": timeline["fps"],
        "frame_count": timeline["frame_count"],
        "event_count": len(events),
        "player_count": len(players),
        "object_count": len(players) + 1,
        "object_stats": object_stats,
        "possession_timeline": possession_timeline,
    }
    return output, summary


def run_validator(
    validator_path: Path,
    config_path: Path,
    strict_warnings: bool,
) -> int:
    command = [
        sys.executable,
        str(validator_path),
        "--config",
        str(config_path),
    ]
    if strict_warnings:
        command.append("--strict-warnings")

    print("[VALIDATOR] " + " ".join('"{}"'.format(part) for part in command))
    completed = subprocess.run(command, check=False)
    return int(completed.returncode)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compile a FutsalMOT episode-event JSON into a dense UE trajectory config."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_EVENT_CONFIG,
        help="Episode event configuration JSON.",
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        default=None,
        help="Override paths.base_render_config.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Override paths.output_trajectory_config.",
    )
    parser.add_argument(
        "--episode-validator",
        type=Path,
        default=None,
        help="Default: 10_validate_episode.py beside this script.",
    )
    parser.add_argument(
        "--trajectory-validator",
        type=Path,
        default=None,
        help="Default: 14_validate_trajectory.py beside this script.",
    )
    parser.add_argument(
        "--skip-episode-validation",
        action="store_true",
        help="Skip pre-compilation episode validation.",
    )
    parser.add_argument(
        "--skip-trajectory-validation",
        action="store_true",
        help="Skip post-compilation dense trajectory validation.",
    )
    parser.add_argument(
        "--strict-warnings",
        action="store_true",
        help="Pass --strict-warnings to both validators.",
    )
    parser.add_argument(
        "--decimals",
        type=int,
        default=6,
        help="Output coordinate decimal places, default 6.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Overwrite an existing output without moving it to .bak.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    event_config_path = args.config.expanduser().resolve()

    try:
        if args.decimals < 0 or args.decimals > 12:
            raise CompileError("--decimals 必须位于 0..12")

        script_dir = Path(__file__).resolve().parent
        episode_validator = (
            args.episode_validator.expanduser().resolve()
            if args.episode_validator is not None
            else script_dir / "10_validate_episode.py"
        )
        trajectory_validator = (
            args.trajectory_validator.expanduser().resolve()
            if args.trajectory_validator is not None
            else script_dir / "14_validate_trajectory.py"
        )

        if not args.skip_episode_validation:
            if not episode_validator.exists():
                raise CompileError(
                    "找不到事件验证器：{}".format(episode_validator)
                )
            return_code = run_validator(
                episode_validator,
                event_config_path,
                args.strict_warnings,
            )
            if return_code != 0:
                print(
                    "[FAILED] Episode validator return code={}".format(return_code)
                )
                return 1

        event_config = load_json(event_config_path)
        paths = event_config.get("paths")
        if not isinstance(paths, dict):
            raise CompileError("paths 必须是 JSON object")

        if args.base_config is not None:
            base_config_path = args.base_config.expanduser().resolve()
        else:
            base_config_path = resolve_relative_path(
                require_str(
                    paths.get("base_render_config"),
                    "paths.base_render_config",
                ),
                event_config_path,
            )

        if args.output is not None:
            output_path = args.output.expanduser().resolve()
        else:
            output_path = resolve_relative_path(
                require_str(
                    paths.get("output_trajectory_config"),
                    "paths.output_trajectory_config",
                ),
                event_config_path,
            )

        if output_path == event_config_path or output_path == base_config_path:
            raise CompileError("输出路径不能覆盖事件配置或基础配置")

        base_config = load_json(base_config_path)
        timeline = parse_timeline(event_config)
        court = parse_court(event_config)
        defaults = parse_defaults(event_config)
        players = parse_players(event_config, defaults)
        ball = parse_ball(event_config)

        if ball["initial_owner"] not in players:
            raise CompileError(
                "ball.initial_owner={} 不存在".format(ball["initial_owner"])
            )

        events = parse_events(event_config, timeline, players)
        pass_receive_map = build_pass_receive_map(events)
        possession_frames = build_possession_frames(
            events,
            pass_receive_map,
            ball["initial_owner"],
            timeline,
        )
        player_paths = compile_player_paths(
            players,
            events,
            possession_frames,
            timeline,
            court,
            defaults,
            ball["initial_owner"],
        )
        ball_path = compile_ball_path(
            ball,
            events,
            possession_frames,
            player_paths,
            timeline,
            defaults,
        )

        output_config, summary = build_output_config(
            event_config,
            event_config_path,
            base_config,
            base_config_path,
            timeline,
            players,
            ball,
            events,
            player_paths,
            ball_path,
            possession_frames,
            defaults,
            args.decimals,
        )

        backup_path = None
        if output_path.exists() and not args.no_backup:
            backup_path = preserve_backup(output_path)
        write_json(output_path, output_config)

        print("=" * 80)
        print("FutsalMOT episode trajectory compiler")
        print("VERSION =", SCRIPT_VERSION)
        print("EPISODE_ID =", summary["episode_id"])
        print("OUTPUT_SEQ_ID =", summary["output_seq_id"])
        print(
            "FRAME_RANGE = {}..{}".format(
                summary["frame_start"], summary["frame_end"]
            )
        )
        print("FPS =", summary["fps"])
        print("FRAME_COUNT =", summary["frame_count"])
        print("EVENT_COUNT =", summary["event_count"])
        print("-" * 80)

        for object_id, stats in summary["object_stats"].items():
            print(
                "[GENERATED] {:<12} category={:<6} keyframes={} "
                "max_speed_xy={:.3f} cm/s".format(
                    object_id,
                    stats["category"],
                    stats["keyframe_count"],
                    stats["max_speed_xy_cm_s"],
                )
            )

        print("-" * 80)
        for segment in summary["possession_timeline"]:
            print(
                "[POSSESSION] frames {}..{} state={} owner={}".format(
                    segment["start_frame"],
                    segment["end_frame"],
                    segment.get("state"),
                    segment.get("owner"),
                )
            )

        print("OUTPUT =", output_path)
        if backup_path is not None:
            print("BACKUP =", backup_path)
        print("=" * 80)

        if not args.skip_trajectory_validation:
            if not trajectory_validator.exists():
                raise CompileError(
                    "找不到轨迹验证器：{}".format(trajectory_validator)
                )
            return_code = run_validator(
                trajectory_validator,
                output_path,
                args.strict_warnings,
            )
            if return_code != 0:
                print(
                    "[FAILED] Output generated, but trajectory validator "
                    "return code={}".format(return_code)
                )
                return 1

        print("[DONE] Episode trajectory generated and validated.")
        return 0

    except CompileError as exc:
        print("=" * 80, file=sys.stderr)
        print("COMPILATION FAILED", file=sys.stderr)
        print("[ERROR]", exc, file=sys.stderr)
        print("EVENT_CONFIG =", event_config_path, file=sys.stderr)
        print("=" * 80, file=sys.stderr)
        return 2

    except Exception as exc:
        print("=" * 80, file=sys.stderr)
        print("COMPILATION FAILED", file=sys.stderr)
        print(
            "[UNEXPECTED ERROR] {}: {}".format(type(exc).__name__, exc),
            file=sys.stderr,
        )
        print("EVENT_CONFIG =", event_config_path, file=sys.stderr)
        print("=" * 80, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
