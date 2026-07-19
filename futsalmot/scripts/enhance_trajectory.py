#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FutsalMOT A3.3 action/facing trajectory enhancer.

Version
-------
A3_3_ACTION_YAW_BALL_SYNC_V1

Purpose
-------
Enhance the dense A3.2 event-compiled trajectory with:
1. Explicit per-frame player yaw values.
2. Complete per-player action timelines.
3. Ball state timeline and contact-frame metadata.
4. Event-to-frame mapping.
5. Optional visual dribble oscillation that returns to zero at event boundaries.

The script preserves the stable A3.2 compiler as a separate baseline. It reads
its generated dense JSON, or automatically invokes it when that JSON is absent.
The enhanced output remains compatible with the A2.2 UE builder because unknown
metadata fields are ignored and each keyframe still uses the standard frame/loc
format, with an additional supported yaw_deg field for player keyframes.

Default workflow
----------------
configs/events/episode_test_0001.json
    -> configs/generated/episode_test_0001.json       (A3.2 baseline)
    -> configs/generated/episode_test_0001_A3_3.json  (this script)

Exit codes
----------
0: enhanced output generated and validation succeeded
1: output generated, but a validator returned non-zero
2: fatal configuration or processing failure
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
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from futsalmot.core.paths import CODE_DIR, CONFIG_DIR


SCRIPT_VERSION = "A3_3_ACTION_YAW_BALL_SYNC_8P_MOVEMENT_V2"

DEFAULT_EVENT_CONFIG = (
    CONFIG_DIR
    / "events"
    / "episode_test_0001.json"
)

DEFAULT_ACTION_SETTINGS: Dict[str, Any] = {
    "yaw_smoothing_window_frames": 5,
    "max_yaw_speed_deg_s": 360.0,
    "minimum_motion_cm_per_frame": 2.0,
    "pass_prepare_frames": 5,
    "receive_prepare_frames": 5,
    "shot_prepare_frames": 5,
    "yaw_actor_offset_deg": 0.0,
    "player_yaw_offset_deg": {},
    "dribble_visual": {
        "enabled": True,
        # Forward back-and-forth was removed because it could move the ball
        # opposite to a slowly accelerating player and create artificial 180°
        # turns. Lateral/vertical touch motion is retained and speed-scaled.
        "oscillation_cm": 0.0,
        "lateral_cm": 1.5,
        "vertical_cm": 2.0,
        "cycle_sec": 0.45,
        "fade_frames": 4,
    },
}

ACTION_PRIORITY = {
    "idle": 0,
    "jog": 10,
    "defend": 20,
    "dribble": 30,
    "receive": 40,
    "pass": 50,
    "shot": 60,
}

YAW_PRIORITY = {
    "motion": 0,
    "defend": 30,
    "receive": 40,
    "pass": 50,
    "shot": 60,
}


class EnhanceError(RuntimeError):
    """Fatal A3.3 enhancement error."""


def is_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def require_float(value: Any, field: str) -> float:
    if not is_number(value):
        raise EnhanceError(
            "字段 '{}' 必须是有限数值，当前值={!r}".format(field, value)
        )
    return float(value)


def require_int(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise EnhanceError(
            "字段 '{}' 必须是整数，当前值={!r}".format(field, value)
        )
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return int(value)
    raise EnhanceError(
        "字段 '{}' 必须是整数，当前值={!r}".format(field, value)
    )


def require_str(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EnhanceError("字段 '{}' 必须是非空字符串".format(field))
    return value.strip()


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise EnhanceError("找不到 JSON 文件：{}".format(path))
    try:
        with path.open("r", encoding="utf-8-sig") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise EnhanceError(
            "JSON 解析失败：{}，line={} column={}".format(
                exc.msg, exc.lineno, exc.colno
            )
        ) from exc
    except OSError as exc:
        raise EnhanceError("读取 JSON 失败：{}；{}".format(path, exc)) from exc

    if not isinstance(data, dict):
        raise EnhanceError("JSON 顶层必须是 object：{}".format(path))
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


def resolve_relative_path(raw_path: str, reference_file: Path) -> Path:
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


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def smoothstep01(value: float) -> float:
    value = clamp(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def normalize_angle_deg(value: float) -> float:
    while value > 180.0:
        value -= 360.0
    while value <= -180.0:
        value += 360.0
    return value


def unwrap_angle_near(value: float, reference: float) -> float:
    value = float(value)
    while value - reference > 180.0:
        value -= 360.0
    while value - reference < -180.0:
        value += 360.0
    return value


def angle_to_target(
    origin: Sequence[float],
    target: Sequence[float],
) -> Optional[float]:
    dx = target[0] - origin[0]
    dy = target[1] - origin[1]
    if math.hypot(dx, dy) <= 1e-8:
        return None
    return math.degrees(math.atan2(dy, dx))


def parse_action_settings(event_config: Dict[str, Any]) -> Dict[str, Any]:
    settings = copy.deepcopy(DEFAULT_ACTION_SETTINGS)
    raw = event_config.get("action_generation", {})

    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise EnhanceError("action_generation 必须是 JSON object")

    for key, value in raw.items():
        if key == "dribble_visual":
            continue
        settings[key] = value

    raw_visual = raw.get("dribble_visual", {})
    if raw_visual is None:
        raw_visual = {}
    if not isinstance(raw_visual, dict):
        raise EnhanceError(
            "action_generation.dribble_visual 必须是 JSON object"
        )

    for key, value in raw_visual.items():
        settings["dribble_visual"][key] = value

    integer_keys = [
        "yaw_smoothing_window_frames",
        "pass_prepare_frames",
        "receive_prepare_frames",
        "shot_prepare_frames",
    ]
    for key in integer_keys:
        settings[key] = require_int(
            settings[key], "action_generation." + key
        )
        if settings[key] < 0:
            raise EnhanceError("action_generation.{} 不能为负数".format(key))

    window = settings["yaw_smoothing_window_frames"]
    if window == 0:
        window = 1
    if window % 2 == 0:
        window += 1
    settings["yaw_smoothing_window_frames"] = window

    float_keys = [
        "max_yaw_speed_deg_s",
        "minimum_motion_cm_per_frame",
        "yaw_actor_offset_deg",
    ]
    for key in float_keys:
        settings[key] = require_float(
            settings[key], "action_generation." + key
        )

    if settings["max_yaw_speed_deg_s"] <= 0.0:
        raise EnhanceError("max_yaw_speed_deg_s 必须大于 0")
    if settings["minimum_motion_cm_per_frame"] < 0.0:
        raise EnhanceError("minimum_motion_cm_per_frame 不能为负数")

    offsets = settings.get("player_yaw_offset_deg", {})
    if offsets is None:
        offsets = {}
    if not isinstance(offsets, dict):
        raise EnhanceError("player_yaw_offset_deg 必须是 JSON object")
    settings["player_yaw_offset_deg"] = {
        str(player_id): require_float(
            value,
            "action_generation.player_yaw_offset_deg.{}".format(player_id),
        )
        for player_id, value in offsets.items()
    }

    visual = settings["dribble_visual"]
    visual["enabled"] = bool(visual.get("enabled", True))
    for key in (
        "oscillation_cm",
        "lateral_cm",
        "vertical_cm",
        "cycle_sec",
    ):
        visual[key] = require_float(
            visual[key], "action_generation.dribble_visual." + key
        )
    visual["fade_frames"] = require_int(
        visual["fade_frames"],
        "action_generation.dribble_visual.fade_frames",
    )
    if visual["cycle_sec"] <= 0.0:
        raise EnhanceError("dribble_visual.cycle_sec 必须大于 0")
    if visual["fade_frames"] < 0:
        raise EnhanceError("dribble_visual.fade_frames 不能为负数")

    return settings


def parse_timeline(compiled_config: Dict[str, Any]) -> Dict[str, Any]:
    raw = compiled_config.get("timeline")
    if not isinstance(raw, dict):
        raise EnhanceError("compiled config 的 timeline 必须是 JSON object")

    frame_start = require_int(raw.get("frame_start"), "timeline.frame_start")
    frame_end = require_int(raw.get("frame_end"), "timeline.frame_end")
    fps = require_float(raw.get("display_rate"), "timeline.display_rate")

    if frame_end < frame_start:
        raise EnhanceError("timeline.frame_end 不能小于 frame_start")
    if fps <= 0.0:
        raise EnhanceError("timeline.display_rate 必须大于 0")

    return {
        "frame_start": frame_start,
        "frame_end": frame_end,
        "fps": fps,
        "frame_count": frame_end - frame_start + 1,
    }


def parse_event_frames(
    event_config: Dict[str, Any],
    timeline: Dict[str, Any],
) -> List[Dict[str, Any]]:
    raw_events = event_config.get("events")
    if not isinstance(raw_events, list) or not raw_events:
        raise EnhanceError("events 必须是非空数组")

    fps = timeline["fps"]
    frame_start = timeline["frame_start"]
    frame_end = timeline["frame_end"]
    result: List[Dict[str, Any]] = []
    seen = set()

    for index, raw in enumerate(raw_events):
        if not isinstance(raw, dict):
            raise EnhanceError("events[{}] 必须是 JSON object".format(index))

        event = copy.deepcopy(raw)
        event_id = require_str(
            event.get("event_id"), "events[{}].event_id".format(index)
        )
        event_type = require_str(
            event.get("type"), "events[{}].type".format(index)
        )

        if event_id in seen:
            raise EnhanceError("重复 event_id={}".format(event_id))
        seen.add(event_id)

        if event_type == "receive":
            time_sec = require_float(
                event.get("time"), "{}.time".format(event_id)
            )
            start_frame_value = int(round(time_sec * fps))
            end_exclusive = start_frame_value + 1
            instantaneous = True
            start_t = time_sec
            end_t = time_sec
        else:
            start_t = require_float(
                event.get("start_t"), "{}.start_t".format(event_id)
            )
            end_t = require_float(
                event.get("end_t"), "{}.end_t".format(event_id)
            )
            start_frame_value = int(round(start_t * fps))
            end_exclusive = int(round(end_t * fps))
            instantaneous = False

        start_frame_value = max(frame_start, min(frame_end, start_frame_value))
        end_exclusive = max(
            start_frame_value + 1,
            min(frame_end + 1, end_exclusive),
        )

        event.update(
            {
                "event_id": event_id,
                "type": event_type,
                "start_t": start_t,
                "end_t": end_t,
                "start_frame": start_frame_value,
                "end_frame_exclusive": end_exclusive,
                "last_frame": end_exclusive - 1,
                "instantaneous": instantaneous,
            }
        )
        result.append(event)

    return sorted(
        result,
        key=lambda event: (
            event["start_frame"],
            event["last_frame"],
            event["event_id"],
        ),
    )


def get_dense_keyframes(
    compiled_config: Dict[str, Any],
    object_id: str,
    timeline: Dict[str, Any],
) -> List[Dict[str, Any]]:
    objects = compiled_config.get("objects")
    if not isinstance(objects, dict):
        raise EnhanceError("compiled config 的 objects 必须是 JSON object")

    raw_object = objects.get(object_id)
    if not isinstance(raw_object, dict):
        raise EnhanceError("compiled config 缺少对象 {}".format(object_id))

    raw_keyframes = raw_object.get("keyframes")
    if not isinstance(raw_keyframes, list):
        raise EnhanceError("{}.keyframes 必须是数组".format(object_id))

    expected_count = timeline["frame_count"]
    by_frame: Dict[int, Dict[str, Any]] = {}

    for index, raw_keyframe in enumerate(raw_keyframes):
        if not isinstance(raw_keyframe, dict):
            raise EnhanceError(
                "{}.keyframes[{}] 必须是 JSON object".format(
                    object_id, index
                )
            )
        frame = require_int(
            raw_keyframe.get("frame"),
            "{}.keyframes[{}].frame".format(object_id, index),
        )
        loc = raw_keyframe.get("loc")
        if not isinstance(loc, list) or len(loc) != 3:
            raise EnhanceError(
                "{}.keyframes[{}].loc 必须为 [x,y,z]".format(
                    object_id, index
                )
            )
        loc_values = [
            require_float(
                value,
                "{}.keyframes[{}].loc[{}]".format(object_id, index, axis),
            )
            for axis, value in enumerate(loc)
        ]
        by_frame[frame] = {
            **copy.deepcopy(raw_keyframe),
            "frame": frame,
            "loc": loc_values,
        }

    result = []
    for frame in range(timeline["frame_start"], timeline["frame_end"] + 1):
        if frame not in by_frame:
            raise EnhanceError(
                "{} 缺少逐帧关键帧 frame={}".format(object_id, frame)
            )
        result.append(by_frame[frame])

    if len(result) != expected_count:
        raise EnhanceError(
            "{} keyframe 数量错误：{}，期望 {}".format(
                object_id, len(result), expected_count
            )
        )
    return result


def locations_from_keyframes(
    keyframes: Sequence[Dict[str, Any]],
) -> List[List[float]]:
    return [list(keyframe["loc"]) for keyframe in keyframes]


def build_movement_yaw(
    positions: Sequence[Sequence[float]],
    minimum_motion_cm_per_frame: float,
) -> List[Optional[float]]:
    count = len(positions)
    result: List[Optional[float]] = [None] * count

    for index in range(count):
        if count <= 1:
            break
        if index == 0:
            left_index = 0
            right_index = 1
        elif index == count - 1:
            left_index = count - 2
            right_index = count - 1
        else:
            left_index = index - 1
            right_index = index + 1

        span_frames = max(1, right_index - left_index)
        dx = positions[right_index][0] - positions[left_index][0]
        dy = positions[right_index][1] - positions[left_index][1]
        motion_per_frame = math.hypot(dx, dy) / span_frames

        if motion_per_frame >= minimum_motion_cm_per_frame:
            result[index] = math.degrees(math.atan2(dy, dx))

    return result


def fill_angle_gaps(values: Sequence[Optional[float]]) -> List[float]:
    result: List[Optional[float]] = list(values)
    last: Optional[float] = None

    for index, value in enumerate(result):
        if value is not None:
            last = float(value)
        elif last is not None:
            result[index] = last

    next_value: Optional[float] = None
    for index in range(len(result) - 1, -1, -1):
        value = result[index]
        if value is not None:
            next_value = float(value)
        elif next_value is not None:
            result[index] = next_value

    return [float(value) if value is not None else 0.0 for value in result]


def unwrap_angles(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    result = [normalize_angle_deg(float(values[0]))]
    for value in values[1:]:
        normalized = normalize_angle_deg(float(value))
        result.append(unwrap_angle_near(normalized, result[-1]))
    return result


def moving_average(values: Sequence[float], window: int) -> List[float]:
    if window <= 1 or len(values) <= 1:
        return list(values)
    radius = window // 2
    result: List[float] = []
    for index in range(len(values)):
        left = max(0, index - radius)
        right = min(len(values), index + radius + 1)
        result.append(sum(values[left:right]) / (right - left))
    return result


def rate_limit_angles(
    values: Sequence[float],
    max_delta_deg: float,
) -> List[float]:
    if not values:
        return []
    result = [float(values[0])]
    for value in values[1:]:
        target = unwrap_angle_near(float(value), result[-1])
        delta = clamp(target - result[-1], -max_delta_deg, max_delta_deg)
        result.append(result[-1] + delta)
    return result


def expand_possession_timeline(
    compiled_config: Dict[str, Any],
    timeline: Dict[str, Any],
) -> List[Dict[str, Any]]:
    raw_segments = compiled_config.get("possession_timeline")
    if not isinstance(raw_segments, list) or not raw_segments:
        raise EnhanceError("compiled config 缺少 possession_timeline")

    result: List[Optional[Dict[str, Any]]] = [None] * timeline["frame_count"]
    for raw in raw_segments:
        if not isinstance(raw, dict):
            raise EnhanceError("possession_timeline 项必须是 JSON object")
        start = require_int(raw.get("start_frame"), "possession.start_frame")
        end = require_int(raw.get("end_frame"), "possession.end_frame")
        start = max(timeline["frame_start"], start)
        end = min(timeline["frame_end"], end)
        for frame in range(start, end + 1):
            item = copy.deepcopy(raw)
            item["frame"] = frame
            result[frame - timeline["frame_start"]] = item

    missing = [
        timeline["frame_start"] + index
        for index, item in enumerate(result)
        if item is None
    ]
    if missing:
        raise EnhanceError(
            "possession_timeline 缺少帧：{}".format(missing[:10])
        )
    return [item for item in result if item is not None]


def initialize_actions(
    player_positions: Dict[str, List[List[float]]],
    timeline: Dict[str, Any],
    minimum_motion_cm_per_frame: float,
) -> Tuple[
    Dict[str, List[str]],
    Dict[str, List[int]],
    Dict[str, List[List[str]]],
]:
    actions: Dict[str, List[str]] = {}
    priorities: Dict[str, List[int]] = {}
    sources: Dict[str, List[List[str]]] = {}

    for player_id, positions in player_positions.items():
        player_actions: List[str] = []
        player_priorities: List[int] = []
        player_sources: List[List[str]] = []

        for index in range(timeline["frame_count"]):
            if index == 0:
                other = min(1, timeline["frame_count"] - 1)
                dist = math.hypot(
                    positions[other][0] - positions[index][0],
                    positions[other][1] - positions[index][1],
                )
            else:
                dist = math.hypot(
                    positions[index][0] - positions[index - 1][0],
                    positions[index][1] - positions[index - 1][1],
                )

            action = "jog" if dist >= minimum_motion_cm_per_frame else "idle"
            player_actions.append(action)
            player_priorities.append(ACTION_PRIORITY[action])
            player_sources.append([])

        actions[player_id] = player_actions
        priorities[player_id] = player_priorities
        sources[player_id] = player_sources

    return actions, priorities, sources


def apply_action_range(
    player_id: str,
    action: str,
    start_frame: int,
    end_frame: int,
    event_id: str,
    timeline: Dict[str, Any],
    actions: Dict[str, List[str]],
    priorities: Dict[str, List[int]],
    sources: Dict[str, List[List[str]]],
) -> None:
    if player_id not in actions:
        return

    start_frame = max(timeline["frame_start"], start_frame)
    end_frame = min(timeline["frame_end"], end_frame)
    priority = ACTION_PRIORITY[action]

    for frame in range(start_frame, end_frame + 1):
        index = frame - timeline["frame_start"]
        if priority >= priorities[player_id][index]:
            if priority > priorities[player_id][index]:
                sources[player_id][index] = []
            actions[player_id][index] = action
            priorities[player_id][index] = priority
        if event_id not in sources[player_id][index]:
            sources[player_id][index].append(event_id)


def build_action_frames(
    player_positions: Dict[str, List[List[float]]],
    events: Sequence[Dict[str, Any]],
    timeline: Dict[str, Any],
    settings: Dict[str, Any],
) -> Tuple[Dict[str, List[str]], Dict[str, List[List[str]]]]:
    actions, priorities, sources = initialize_actions(
        player_positions,
        timeline,
        settings["minimum_motion_cm_per_frame"],
    )

    for event in events:
        event_type = event["type"]
        event_id = event["event_id"]
        start = event["start_frame"]
        end = event["last_frame"]

        if event_type == "move":
            apply_action_range(
                event.get("actor"), "jog", start, end, event_id,
                timeline, actions, priorities, sources,
            )
        elif event_type == "hold":
            apply_action_range(
                event.get("actor"), "idle", start, end, event_id,
                timeline, actions, priorities, sources,
            )
        elif event_type == "dribble":
            apply_action_range(
                event.get("actor"), "dribble", start, end, event_id,
                timeline, actions, priorities, sources,
            )
        elif event_type == "defend_follow":
            apply_action_range(
                event.get("actor"), "defend", start, end, event_id,
                timeline, actions, priorities, sources,
            )
        elif event_type == "pass":
            prepare = settings["pass_prepare_frames"]
            apply_action_range(
                event.get("from"), "pass", start - prepare, end, event_id,
                timeline, actions, priorities, sources,
            )
        elif event_type == "receive":
            prepare = settings["receive_prepare_frames"]
            apply_action_range(
                event.get("actor"), "receive", start - prepare, start, event_id,
                timeline, actions, priorities, sources,
            )
        elif event_type == "shot":
            prepare = settings["shot_prepare_frames"]
            apply_action_range(
                event.get("actor"), "shot", start - prepare, end, event_id,
                timeline, actions, priorities, sources,
            )

    return actions, sources


def compress_action_timeline(
    actions: Sequence[str],
    sources: Sequence[Sequence[str]],
    frame_start: int,
) -> List[Dict[str, Any]]:
    if not actions:
        return []

    result: List[Dict[str, Any]] = []
    segment_start = frame_start
    current_action = actions[0]
    current_sources = tuple(sorted(sources[0]))

    for index in range(1, len(actions)):
        signature = (actions[index], tuple(sorted(sources[index])))
        current_signature = (current_action, current_sources)
        if signature != current_signature:
            item: Dict[str, Any] = {
                "start_frame": segment_start,
                "end_frame": frame_start + index - 1,
                "action": current_action,
            }
            if current_sources:
                item["source_events"] = list(current_sources)
            result.append(item)
            segment_start = frame_start + index
            current_action = actions[index]
            current_sources = tuple(sorted(sources[index]))

    item = {
        "start_frame": segment_start,
        "end_frame": frame_start + len(actions) - 1,
        "action": current_action,
    }
    if current_sources:
        item["source_events"] = list(current_sources)
    result.append(item)
    return result


def set_yaw_target(
    desired: List[Optional[float]],
    priorities: List[int],
    frame: int,
    target_angle: Optional[float],
    priority: int,
    timeline: Dict[str, Any],
) -> None:
    if target_angle is None:
        return
    if frame < timeline["frame_start"] or frame > timeline["frame_end"]:
        return
    index = frame - timeline["frame_start"]
    if priority >= priorities[index]:
        desired[index] = target_angle
        priorities[index] = priority


def build_player_yaws(
    player_positions: Dict[str, List[List[float]]],
    ball_positions: List[List[float]],
    events: Sequence[Dict[str, Any]],
    possession_frames: Sequence[Dict[str, Any]],
    timeline: Dict[str, Any],
    settings: Dict[str, Any],
) -> Tuple[Dict[str, List[float]], Dict[str, Dict[str, Any]]]:
    result: Dict[str, List[float]] = {}
    stats: Dict[str, Dict[str, Any]] = {}
    event_by_id = {event["event_id"]: event for event in events}

    for player_id, positions in player_positions.items():
        desired = build_movement_yaw(
            positions,
            settings["minimum_motion_cm_per_frame"],
        )
        priorities = [YAW_PRIORITY["motion"]] * len(desired)

        for event in events:
            event_type = event["type"]
            start = event["start_frame"]
            end = event["last_frame"]

            if event_type == "pass" and event.get("from") == player_id:
                target_id = event.get("to")
                if target_id not in player_positions:
                    continue
                for frame in range(
                    max(timeline["frame_start"], start - settings["pass_prepare_frames"]),
                    min(timeline["frame_end"], end) + 1,
                ):
                    index = frame - timeline["frame_start"]
                    target_angle = angle_to_target(
                        positions[index], player_positions[target_id][index]
                    )
                    set_yaw_target(
                        desired, priorities, frame, target_angle,
                        YAW_PRIORITY["pass"], timeline,
                    )

            elif event_type == "receive" and event.get("actor") == player_id:
                source = event_by_id.get(event.get("source_event"))
                for frame in range(
                    max(timeline["frame_start"], start - settings["receive_prepare_frames"]),
                    min(timeline["frame_end"], start) + 1,
                ):
                    index = frame - timeline["frame_start"]
                    ball_index = max(0, index - 1)
                    target_angle = angle_to_target(
                        positions[index], ball_positions[ball_index]
                    )
                    if target_angle is None and source is not None:
                        from_id = source.get("from")
                        if from_id in player_positions:
                            target_angle = angle_to_target(
                                positions[index], player_positions[from_id][index]
                            )
                    set_yaw_target(
                        desired, priorities, frame, target_angle,
                        YAW_PRIORITY["receive"], timeline,
                    )

            elif event_type == "shot" and event.get("actor") == player_id:
                target_loc = event.get("target_loc")
                if not isinstance(target_loc, list) or len(target_loc) != 3:
                    continue
                for frame in range(
                    max(timeline["frame_start"], start - settings["shot_prepare_frames"]),
                    min(timeline["frame_end"], end) + 1,
                ):
                    index = frame - timeline["frame_start"]
                    target_angle = angle_to_target(positions[index], target_loc)
                    set_yaw_target(
                        desired, priorities, frame, target_angle,
                        YAW_PRIORITY["shot"], timeline,
                    )

            elif event_type == "defend_follow" and event.get("actor") == player_id:
                for frame in range(
                    max(timeline["frame_start"], start),
                    min(timeline["frame_end"], end) + 1,
                ):
                    index = frame - timeline["frame_start"]
                    state = possession_frames[index]
                    target_position: Optional[Sequence[float]] = None

                    if state.get("state") == "owned":
                        owner = state.get("owner")
                        if owner in player_positions:
                            target_position = player_positions[owner][index]
                    elif state.get("state") in {"in_transit", "shot"}:
                        target_position = ball_positions[index]

                    target_spec = event.get("target")
                    if target_spec not in {None, "possession_owner"}:
                        if target_spec in player_positions:
                            target_position = player_positions[target_spec][index]

                    if target_position is not None:
                        target_angle = angle_to_target(
                            positions[index], target_position
                        )
                        set_yaw_target(
                            desired, priorities, frame, target_angle,
                            YAW_PRIORITY["defend"], timeline,
                        )

        filled = fill_angle_gaps(desired)
        unwrapped = unwrap_angles(filled)
        smoothed = moving_average(
            unwrapped,
            settings["yaw_smoothing_window_frames"],
        )
        max_delta = settings["max_yaw_speed_deg_s"] / timeline["fps"]
        limited = rate_limit_angles(smoothed, max_delta)

        player_offset = settings["player_yaw_offset_deg"].get(
            player_id,
            settings["yaw_actor_offset_deg"],
        )
        final_values = [
            normalize_angle_deg(value + player_offset)
            for value in limited
        ]
        result[player_id] = final_values

        actual_max_delta = 0.0
        for left, right in zip(limited, limited[1:]):
            actual_max_delta = max(actual_max_delta, abs(right - left))

        stats[player_id] = {
            "keyframe_count": len(final_values),
            "max_yaw_delta_deg_per_frame": actual_max_delta,
            "max_yaw_speed_deg_s": actual_max_delta * timeline["fps"],
            "configured_max_yaw_speed_deg_s": settings[
                "max_yaw_speed_deg_s"
            ],
            "yaw_actor_offset_deg": player_offset,
        }

    return result, stats


def dribble_envelope(
    frame: int,
    start_frame: int,
    last_frame: int,
    fade_frames: int,
) -> float:
    if last_frame <= start_frame:
        return 0.0
    if fade_frames <= 0:
        return 1.0

    fade_in = clamp((frame - start_frame) / fade_frames, 0.0, 1.0)
    fade_out = clamp((last_frame - frame) / fade_frames, 0.0, 1.0)
    return smoothstep01(min(fade_in, fade_out))


def apply_dribble_visual(
    ball_positions: List[List[float]],
    player_positions: Dict[str, List[List[float]]],
    player_yaws: Dict[str, List[float]],
    events: Sequence[Dict[str, Any]],
    timeline: Dict[str, Any],
    settings: Dict[str, Any],
) -> Dict[str, Any]:
    visual = settings["dribble_visual"]
    if not visual["enabled"]:
        return {
            "enabled": False,
            "modified_frame_count": 0,
            **copy.deepcopy(visual),
        }

    cycle_frames = max(1.0, visual["cycle_sec"] * timeline["fps"])
    modified_frames = set()

    for event in events:
        if event["type"] != "dribble":
            continue
        actor = event.get("actor")
        if actor not in player_positions or actor not in player_yaws:
            continue

        start = event["start_frame"]
        last = event["last_frame"]

        for frame in range(start, last + 1):
            index = frame - timeline["frame_start"]
            envelope = dribble_envelope(
                frame,
                start,
                last,
                visual["fade_frames"],
            )
            if envelope <= 0.0:
                continue

            phase = 2.0 * math.pi * (frame - start) / cycle_frames
            yaw_rad = math.radians(player_yaws[actor][index])
            forward = (math.cos(yaw_rad), math.sin(yaw_rad))
            right = (-forward[1], forward[0])

            previous_index = max(0, index - 1)
            player_step = math.hypot(
                player_positions[actor][index][0]
                - player_positions[actor][previous_index][0],
                player_positions[actor][index][1]
                - player_positions[actor][previous_index][1],
            )
            player_speed_cm_s = player_step * timeline["fps"]
            motion_scale = clamp(player_speed_cm_s / 120.0, 0.0, 1.0)

            # Do not oscillate the ball backwards along the running direction.
            # This preserves monotonic progress during acceleration/deceleration.
            forward_delta = 0.0
            lateral_delta = (
                visual["lateral_cm"]
                * math.sin(phase * 0.5)
                * envelope
                * motion_scale
            )
            vertical_delta = (
                visual["vertical_cm"]
                * abs(math.sin(phase))
                * envelope
                * motion_scale
            )

            ball_positions[index][0] += (
                forward[0] * forward_delta + right[0] * lateral_delta
            )
            ball_positions[index][1] += (
                forward[1] * forward_delta + right[1] * lateral_delta
            )
            ball_positions[index][2] += vertical_delta
            modified_frames.add(frame)

    return {
        "enabled": True,
        "modified_frame_count": len(modified_frames),
        "cycle_frames": cycle_frames,
        "motion_mode": "speed_scaled_lateral_vertical_no_backward_v2",
        **copy.deepcopy(visual),
    }


def calculate_speed_stats(
    positions: Sequence[Sequence[float]],
    fps: float,
    frame_start: int,
) -> Dict[str, Any]:
    max_speed_xy = 0.0
    max_speed_3d = 0.0
    max_speed_frame: Optional[int] = None

    for index, (left, right) in enumerate(
        zip(positions, positions[1:]), start=1
    ):
        dx = right[0] - left[0]
        dy = right[1] - left[1]
        dz = right[2] - left[2]
        speed_xy = math.hypot(dx, dy) * fps
        speed_3d = math.sqrt(dx * dx + dy * dy + dz * dz) * fps
        if speed_xy > max_speed_xy:
            max_speed_xy = speed_xy
            max_speed_frame = frame_start + index
        max_speed_3d = max(max_speed_3d, speed_3d)

    return {
        "max_speed_xy_cm_s": max_speed_xy,
        "max_speed_3d_cm_s": max_speed_3d,
        "max_speed_frame": max_speed_frame,
    }


def build_event_frame_map(
    events: Sequence[Dict[str, Any]],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    event_map: Dict[str, Any] = {}
    contact_frames: List[Dict[str, Any]] = []

    for event in events:
        item: Dict[str, Any] = {
            "type": event["type"],
            "start_frame": event["start_frame"],
            "end_frame_exclusive": event["end_frame_exclusive"],
            "last_frame": event["last_frame"],
        }

        for key in (
            "actor",
            "from",
            "to",
            "target",
            "source_event",
            "target_loc",
        ):
            if key in event:
                item[key] = copy.deepcopy(event[key])

        if event["type"] in {"pass", "shot"}:
            item["contact_frame"] = event["start_frame"]
            contact_frames.append(
                {
                    "event_id": event["event_id"],
                    "type": event["type"],
                    "frame": event["start_frame"],
                    "actor": event.get("from", event.get("actor")),
                }
            )
        elif event["type"] == "receive":
            item["contact_frame"] = event["start_frame"]
            contact_frames.append(
                {
                    "event_id": event["event_id"],
                    "type": "receive",
                    "frame": event["start_frame"],
                    "actor": event.get("actor"),
                    "source_event": event.get("source_event"),
                }
            )

        event_map[event["event_id"]] = item

    contact_frames.sort(key=lambda item: (item["frame"], item["event_id"]))
    return event_map, contact_frames


def build_ball_state_timeline(
    possession_timeline: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    for segment in possession_timeline:
        state = segment.get("state")
        if state == "owned":
            ball_state = "controlled"
        elif state == "in_transit":
            ball_state = "pass"
        elif state == "shot":
            ball_state = "shot"
        else:
            ball_state = str(state)

        item: Dict[str, Any] = {
            "start_frame": segment["start_frame"],
            "end_frame": segment["end_frame"],
            "state": ball_state,
            "owner": segment.get("owner"),
        }
        for key in ("from", "to", "event_id"):
            if key in segment:
                item[key] = segment[key]
        result.append(item)
    return result


def validate_enhanced_content(
    output: Dict[str, Any],
    player_ids: Sequence[str],
    ball_id: str,
    timeline: Dict[str, Any],
    settings: Dict[str, Any],
) -> None:
    objects = output.get("objects")
    if not isinstance(objects, dict):
        raise EnhanceError("输出 objects 无效")

    max_delta = settings["max_yaw_speed_deg_s"] / timeline["fps"]

    for player_id in player_ids:
        obj = objects.get(player_id)
        if not isinstance(obj, dict):
            raise EnhanceError("输出缺少 player {}".format(player_id))
        keyframes = obj.get("keyframes")
        if not isinstance(keyframes, list) or len(keyframes) != timeline["frame_count"]:
            raise EnhanceError("{} 的 keyframes 数量错误".format(player_id))
        for keyframe in keyframes:
            if not is_number(keyframe.get("yaw_deg")):
                raise EnhanceError(
                    "{} frame={} 缺少有效 yaw_deg".format(
                        player_id, keyframe.get("frame")
                    )
                )
        for left, right in zip(keyframes, keyframes[1:]):
            left_yaw = float(left["yaw_deg"])
            right_yaw = unwrap_angle_near(float(right["yaw_deg"]), left_yaw)
            if abs(right_yaw - left_yaw) > max_delta + 1e-4:
                raise EnhanceError(
                    "{} frame {}→{} yaw 变化 {:.4f}° 超过 {:.4f}°".format(
                        player_id,
                        left["frame"],
                        right["frame"],
                        abs(right_yaw - left_yaw),
                        max_delta,
                    )
                )
        action_timeline = obj.get("action_timeline")
        if not isinstance(action_timeline, list) or not action_timeline:
            raise EnhanceError("{} 缺少 action_timeline".format(player_id))
        if action_timeline[0]["start_frame"] != timeline["frame_start"]:
            raise EnhanceError("{} action_timeline 未从首帧开始".format(player_id))
        if action_timeline[-1]["end_frame"] != timeline["frame_end"]:
            raise EnhanceError("{} action_timeline 未覆盖末帧".format(player_id))

    ball_obj = objects.get(ball_id)
    if not isinstance(ball_obj, dict):
        raise EnhanceError("输出缺少 ball {}".format(ball_id))
    if not isinstance(ball_obj.get("state_timeline"), list):
        raise EnhanceError("ball 缺少 state_timeline")


def run_command(command: Sequence[str], label: str) -> int:
    print("[{}] {}".format(
        label,
        " ".join('"{}"'.format(part) for part in command),
    ))
    completed = subprocess.run(list(command), check=False)
    return int(completed.returncode)


def resolve_default_paths(
    event_config_path: Path,
    event_config: Dict[str, Any],
) -> Tuple[Path, Path]:
    paths = event_config.get("paths")
    if not isinstance(paths, dict):
        raise EnhanceError("事件配置中的 paths 必须是 JSON object")

    compiled_path = resolve_relative_path(
        require_str(
            paths.get("output_trajectory_config"),
            "paths.output_trajectory_config",
        ),
        event_config_path,
    )
    output_path = compiled_path.with_name(
        compiled_path.stem + "_A3_3" + compiled_path.suffix
    )
    return compiled_path, output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Enhance an A3.2 FutsalMOT dense episode with explicit yaw, "
            "action timelines, ball states, and contact frames."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_EVENT_CONFIG,
        help="Episode event JSON.",
    )
    parser.add_argument(
        "--compiled-config",
        type=Path,
        default=None,
        help="A3.2 dense config. Default: paths.output_trajectory_config.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="A3.3 output. Default: <compiled stem>_A3_3.json.",
    )
    parser.add_argument(
        "--seq-id",
        type=str,
        default=None,
        help="Output seq_id. Default: <compiled seq_id>_A3_3.",
    )
    parser.add_argument(
        "--base-compiler",
        type=Path,
        default=None,
        help="Default: 12_compile_trajectory.py beside this script.",
    )
    parser.add_argument(
        "--trajectory-validator",
        type=Path,
        default=None,
        help="Default: 14_validate_trajectory.py beside this script.",
    )
    parser.add_argument(
        "--recompile-base",
        action="store_true",
        help="Run the stable A3.2 compiler even when compiled config exists.",
    )
    parser.add_argument(
        "--skip-trajectory-validation",
        action="store_true",
        help="Skip final dense trajectory validator.",
    )
    parser.add_argument(
        "--strict-warnings",
        action="store_true",
        help="Treat trajectory validator warnings as failure.",
    )
    parser.add_argument(
        "--decimals",
        type=int,
        default=6,
        help="Coordinate/yaw decimals, default 6.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Overwrite an existing A3.3 output without .bak backup.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    event_config_path = args.config.expanduser().resolve()

    try:
        if args.decimals < 0 or args.decimals > 12:
            raise EnhanceError("--decimals 必须位于 0..12")

        script_dir = CODE_DIR
        event_config = load_json(event_config_path)
        default_compiled, default_output = resolve_default_paths(
            event_config_path, event_config
        )

        compiled_path = (
            args.compiled_config.expanduser().resolve()
            if args.compiled_config is not None
            else default_compiled
        )
        output_path = (
            args.output.expanduser().resolve()
            if args.output is not None
            else default_output
        )

        if output_path in {event_config_path, compiled_path}:
            raise EnhanceError("A3.3 输出不能覆盖事件配置或 A3.2 基线")

        base_compiler = (
            args.base_compiler.expanduser().resolve()
            if args.base_compiler is not None
            else script_dir / "futsalmot" / "scripts" / "compile_trajectory.py"
        )
        trajectory_validator = (
            args.trajectory_validator.expanduser().resolve()
            if args.trajectory_validator is not None
            else script_dir / "futsalmot" / "scripts" / "validate_trajectory.py"
        )

        if args.recompile_base or not compiled_path.exists():
            if not base_compiler.exists():
                raise EnhanceError("找不到 A3.2 编译器：{}".format(base_compiler))
            command = [
                sys.executable,
                str(base_compiler),
                "--config",
                str(event_config_path),
                "--output",
                str(compiled_path),
            ]
            if args.strict_warnings:
                command.append("--strict-warnings")
            return_code = run_command(command, "BASE COMPILER")
            if return_code != 0:
                print("[FAILED] A3.2 base compiler return code={}".format(return_code))
                return 1

        compiled = load_json(compiled_path)
        timeline = parse_timeline(compiled)
        settings = parse_action_settings(event_config)
        events = parse_event_frames(event_config, timeline)

        players_raw = event_config.get("players")
        if not isinstance(players_raw, dict) or not players_raw:
            raise EnhanceError("事件配置 players 必须是非空 JSON object")
        player_ids = list(players_raw.keys())

        ball_raw = event_config.get("ball")
        if not isinstance(ball_raw, dict):
            raise EnhanceError("事件配置 ball 必须是 JSON object")
        ball_id = require_str(ball_raw.get("object_id"), "ball.object_id")

        player_keyframes = {
            player_id: get_dense_keyframes(compiled, player_id, timeline)
            for player_id in player_ids
        }
        ball_keyframes = get_dense_keyframes(compiled, ball_id, timeline)
        player_positions = {
            player_id: locations_from_keyframes(keyframes)
            for player_id, keyframes in player_keyframes.items()
        }
        ball_positions = locations_from_keyframes(ball_keyframes)

        possession_frames = expand_possession_timeline(compiled, timeline)
        actions, action_sources = build_action_frames(
            player_positions, events, timeline, settings
        )
        player_yaws, yaw_stats = build_player_yaws(
            player_positions,
            ball_positions,
            events,
            possession_frames,
            timeline,
            settings,
        )

        dribble_stats = apply_dribble_visual(
            ball_positions,
            player_positions,
            player_yaws,
            events,
            timeline,
            settings,
        )

        output = copy.deepcopy(compiled)
        original_seq_id = require_str(compiled.get("seq_id"), "compiled.seq_id")
        output_seq_id = (
            args.seq_id.strip()
            if isinstance(args.seq_id, str) and args.seq_id.strip()
            else original_seq_id + "_A3_3"
        )
        output["schema_version"] = "3.1"
        output["seq_id"] = output_seq_id

        output_objects = output.get("objects")
        if not isinstance(output_objects, dict):
            raise EnhanceError("compiled objects 无效")

        for player_id in player_ids:
            obj = output_objects[player_id]
            enhanced_keyframes = copy.deepcopy(player_keyframes[player_id])
            for index, keyframe in enumerate(enhanced_keyframes):
                keyframe["yaw_deg"] = round(
                    player_yaws[player_id][index], args.decimals
                )
                keyframe["loc"] = [
                    round(float(value), args.decimals)
                    for value in keyframe["loc"]
                ]
            obj["keyframes"] = enhanced_keyframes
            obj["action_timeline"] = compress_action_timeline(
                actions[player_id],
                action_sources[player_id],
                timeline["frame_start"],
            )
            obj["yaw_source"] = "offline_event_aware_smoothed_world_actor_yaw"

        enhanced_ball_keyframes = copy.deepcopy(ball_keyframes)
        for index, keyframe in enumerate(enhanced_ball_keyframes):
            keyframe["loc"] = [
                round(float(value), args.decimals)
                for value in ball_positions[index]
            ]
        ball_state_timeline = build_ball_state_timeline(
            compiled["possession_timeline"]
        )
        output_objects[ball_id]["keyframes"] = enhanced_ball_keyframes
        output_objects[ball_id]["state_timeline"] = ball_state_timeline

        event_frame_map, contact_frames = build_event_frame_map(events)
        output["event_frame_map"] = event_frame_map
        output["contact_frames"] = contact_frames
        output["ball_state_timeline"] = ball_state_timeline

        ball_speed_stats = calculate_speed_stats(
            ball_positions, timeline["fps"], timeline["frame_start"]
        )
        action_counts = {
            player_id: {
                action: actions[player_id].count(action)
                for action in sorted(set(actions[player_id]))
            }
            for player_id in player_ids
        }

        metadata = output.get("episode_metadata")
        if not isinstance(metadata, dict):
            metadata = {}
            output["episode_metadata"] = metadata
        metadata.update(
            {
                "a3_3_enhancer_version": SCRIPT_VERSION,
                "a3_3_enhanced_at_utc": datetime.now(timezone.utc).isoformat(),
                "a3_2_source_config": str(compiled_path).replace("\\", "/"),
                "trajectory_type": "event_compiled_dense_action_yaw_v1",
                "explicit_player_yaw": True,
                "action_timeline_enabled": True,
                "ball_state_timeline_enabled": True,
                "contact_frame_metadata_enabled": True,
                "action_generation_settings": settings,
                "yaw_stats": yaw_stats,
                "action_frame_counts": action_counts,
                "dribble_visual_stats": dribble_stats,
                "enhanced_ball_speed_stats": ball_speed_stats,
            }
        )

        validate_enhanced_content(
            output,
            player_ids,
            ball_id,
            timeline,
            settings,
        )

        backup_path = None
        if output_path.exists() and not args.no_backup:
            backup_path = preserve_backup(output_path)
        write_json(output_path, output)

        print("=" * 84)
        print("FutsalMOT A3.3 action/yaw enhancer")
        print("VERSION =", SCRIPT_VERSION)
        print("SOURCE_SEQ_ID =", original_seq_id)
        print("OUTPUT_SEQ_ID =", output_seq_id)
        print(
            "FRAME_RANGE = {}..{}".format(
                timeline["frame_start"], timeline["frame_end"]
            )
        )
        print("FPS =", timeline["fps"])
        print("PLAYER_COUNT =", len(player_ids))
        print("EVENT_COUNT =", len(events))
        print("-" * 84)

        for player_id in player_ids:
            player_stat = yaw_stats[player_id]
            print(
                "[PLAYER] {:<12} keyframes={} action_segments={} "
                "max_yaw_speed={:.3f} deg/s".format(
                    player_id,
                    len(output_objects[player_id]["keyframes"]),
                    len(output_objects[player_id]["action_timeline"]),
                    player_stat["max_yaw_speed_deg_s"],
                )
            )

        print(
            "[BALL] keyframes={} max_speed_xy={:.3f} cm/s "
            "dribble_modified_frames={}".format(
                len(output_objects[ball_id]["keyframes"]),
                ball_speed_stats["max_speed_xy_cm_s"],
                dribble_stats["modified_frame_count"],
            )
        )
        print("[CONTACTS] count={}".format(len(contact_frames)))
        for contact in contact_frames:
            print(
                "  frame={} type={} event={} actor={}".format(
                    contact["frame"],
                    contact["type"],
                    contact["event_id"],
                    contact.get("actor"),
                )
            )
        print("OUTPUT =", output_path)
        if backup_path is not None:
            print("BACKUP =", backup_path)
        print("=" * 84)

        if not args.skip_trajectory_validation:
            if not trajectory_validator.exists():
                raise EnhanceError(
                    "找不到轨迹验证器：{}".format(trajectory_validator)
                )
            command = [
                sys.executable,
                str(trajectory_validator),
                "--config",
                str(output_path),
            ]
            if args.strict_warnings:
                command.append("--strict-warnings")
            return_code = run_command(command, "TRAJECTORY VALIDATOR")
            if return_code != 0:
                print(
                    "[FAILED] A3.3 output generated, but trajectory validator "
                    "return code={}".format(return_code)
                )
                return 1

        print("[DONE] A3.3 action/yaw trajectory generated and validated.")
        return 0

    except EnhanceError as exc:
        print("=" * 84, file=sys.stderr)
        print("A3.3 ENHANCEMENT FAILED", file=sys.stderr)
        print("[ERROR]", exc, file=sys.stderr)
        print("EVENT_CONFIG =", event_config_path, file=sys.stderr)
        print("=" * 84, file=sys.stderr)
        return 2

    except Exception as exc:
        print("=" * 84, file=sys.stderr)
        print("A3.3 ENHANCEMENT FAILED", file=sys.stderr)
        print(
            "[UNEXPECTED ERROR] {}: {}".format(type(exc).__name__, exc),
            file=sys.stderr,
        )
        print("EVENT_CONFIG =", event_config_path, file=sys.stderr)
        print("=" * 84, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
