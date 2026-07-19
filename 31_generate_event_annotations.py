#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FutsalMOT A3.3c event annotation generator.

Version:
    A3_3C_EVENT_ANNOTATIONS_8P_V3

Inputs:
- episode event config (A3.4/A3.1 schema)
- A3.3 enhanced dense trajectory config

Outputs:
- events_<seq_id>.json
- frame_states_<seq_id>.jsonl
- event_annotation_report_<seq_id>.json

All output intervals use [start_frame, end_frame_exclusive).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

SCRIPT_VERSION = "A3_3C_EVENT_ANNOTATIONS_8P_V3"
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_EVENT_CONFIG = SCRIPT_DIR / "configs" / "events" / "episode_test_0001.json"
DEFAULT_A3_CONFIG = SCRIPT_DIR / "configs" / "generated" / "episode_test_0001_A3_3.json"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "_agent_test_outputs" / "A3_3C_episode_test_0001"
SUPPORTED_EVENT_TYPES = {
    "hold",
    "move",
    "dribble",
    "pass",
    "receive",
    "defend_follow",
    "shot",
}
ALLOWED_RESULTS = {
    None,
    "completed",
    "failed",
    "intercepted",
    "goal",
    "missed",
    "blocked",
    "out_of_play",
}


class AnnotationError(RuntimeError):
    """Fatal input or consistency error."""


def is_finite_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def require_int(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise AnnotationError("{} 必须是整数".format(field))
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return int(value)
    raise AnnotationError("{} 必须是整数，当前={!r}".format(field, value))


def require_str(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AnnotationError("{} 必须是非空字符串".format(field))
    return value.strip()


def load_json(path: Path, label: str) -> Dict[str, Any]:
    if not path.is_file():
        raise AnnotationError("找不到{}：{}".format(label, path))
    try:
        with path.open("r", encoding="utf-8-sig") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise AnnotationError(
            "{} JSON 解析失败：{} line={} column={}".format(
                label, exc.msg, exc.lineno, exc.colno
            )
        ) from exc
    except OSError as exc:
        raise AnnotationError("无法读取{}：{}".format(label, exc)) from exc
    if not isinstance(data, dict):
        raise AnnotationError("{}顶层必须是 JSON object".format(label))
    return data


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.{}".format(os.getpid()))
    try:
        with tmp.open("w", encoding="utf-8", newline="\n") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(str(tmp), str(path))
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def atomic_write_json(path: Path, data: Any) -> None:
    raw = json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)
    atomic_write_text(path, raw + "\n")


def normalize_end_frame(raw: Dict[str, Any], field: str = "interval") -> Tuple[int, str]:
    """Normalize inclusive/exclusive source fields to end_frame_exclusive."""
    has_exc = "end_frame_exclusive" in raw and raw.get("end_frame_exclusive") is not None
    has_inc = "end_frame" in raw and raw.get("end_frame") is not None
    if has_exc and has_inc:
        exc = require_int(raw["end_frame_exclusive"], field + ".end_frame_exclusive")
        inc = require_int(raw["end_frame"], field + ".end_frame")
        if exc != inc + 1:
            raise AnnotationError(
                "{} 区间冲突：end_frame_exclusive={}，end_frame+1={}".format(
                    field, exc, inc + 1
                )
            )
        return exc, "both_consistent"
    if has_exc:
        return require_int(raw["end_frame_exclusive"], field + ".end_frame_exclusive"), "exclusive"
    if has_inc:
        return require_int(raw["end_frame"], field + ".end_frame") + 1, "inclusive"
    raise AnnotationError("{} 缺少 end_frame/end_frame_exclusive".format(field))


def normalize_segment(raw: Dict[str, Any], field: str) -> Tuple[int, int]:
    start = require_int(raw.get("start_frame"), field + ".start_frame")
    end_exc, _ = normalize_end_frame(raw, field)
    if end_exc <= start:
        raise AnnotationError("{} 区间无效：[{}, {})".format(field, start, end_exc))
    return start, end_exc


def timeline_info(a3_cfg: Dict[str, Any]) -> Dict[str, int]:
    raw = a3_cfg.get("timeline")
    if not isinstance(raw, dict):
        raise AnnotationError("A3.3 timeline 必须是 object")
    start = require_int(raw.get("frame_start"), "timeline.frame_start")
    end = require_int(raw.get("frame_end"), "timeline.frame_end")
    fps_value = raw.get("display_rate", raw.get("fps", 30))
    if not is_finite_number(fps_value) or float(fps_value) <= 0:
        raise AnnotationError("timeline.display_rate 必须为正数")
    fps = int(round(float(fps_value)))
    if end < start:
        raise AnnotationError("timeline.frame_end 小于 frame_start")
    return {
        "frame_start": start,
        "frame_end": end,
        "frame_end_exclusive": end + 1,
        "total_frames": end - start + 1,
        "fps": fps,
    }


def event_actor(raw: Dict[str, Any]) -> Optional[str]:
    value = raw.get("actor")
    if not value and raw.get("type") == "pass":
        value = raw.get("from")
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def event_target(raw: Dict[str, Any]) -> Optional[str]:
    value = raw.get("target")
    if not value and raw.get("type") == "pass":
        value = raw.get("to")
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def seconds_to_frame(value: Any, fps: int, rule: str) -> int:
    if not is_finite_number(value):
        raise AnnotationError("事件时间必须是有限数值，当前={!r}".format(value))
    scaled = float(value) * fps
    if rule == "floor":
        return int(math.floor(scaled + 1e-12))
    if rule == "ceil":
        return int(math.ceil(scaled - 1e-12))
    return int(round(scaled))


def source_event_frame_range(
    raw: Dict[str, Any],
    event_id: str,
    fps: int,
    time_rule: str,
) -> Tuple[int, int, str]:
    if "start_frame" in raw:
        start = require_int(raw.get("start_frame"), event_id + ".start_frame")
        end_exc, source = normalize_end_frame(raw, event_id)
        return start, end_exc, source
    if "start_t" in raw and "end_t" in raw:
        start = seconds_to_frame(raw["start_t"], fps, time_rule)
        end_exc = seconds_to_frame(raw["end_t"], fps, time_rule)
        return start, end_exc, "start_t/end_t"
    if "time" in raw:
        frame = seconds_to_frame(raw["time"], fps, time_rule)
        return frame, frame + 1, "instant_time"
    raise AnnotationError("事件 {} 无法确定帧范围".format(event_id))


def build_track_map(a3_cfg: Dict[str, Any]) -> Dict[str, int]:
    objects = a3_cfg.get("objects")
    if not isinstance(objects, dict) or not objects:
        raise AnnotationError("A3.3 objects 缺失或为空")
    result: Dict[str, int] = {}
    used: Dict[int, str] = {}
    for object_id, raw in objects.items():
        if not isinstance(raw, dict):
            raise AnnotationError("objects.{} 必须是 object".format(object_id))
        track_id = require_int(raw.get("track_id"), "objects.{}.track_id".format(object_id))
        if track_id in used:
            raise AnnotationError(
                "track_id={} 被 {} 和 {} 重复使用".format(track_id, used[track_id], object_id)
            )
        used[track_id] = object_id
        result[str(object_id)] = track_id
    return result


def build_team_map(event_cfg: Dict[str, Any]) -> Dict[str, str]:
    players = event_cfg.get("players", {})
    result: Dict[str, str] = {}
    if isinstance(players, dict):
        iterator = players.items()
    elif isinstance(players, list):
        iterator = []
        for raw in players:
            if isinstance(raw, dict):
                object_id = raw.get("object_id", raw.get("id"))
                iterator.append((object_id, raw))
    else:
        return result
    for object_id, raw in iterator:
        if not object_id or not isinstance(raw, dict):
            continue
        team = raw.get("team")
        if team is not None and str(team).strip():
            result[str(object_id)] = str(team).strip()
    return result


def build_events(
    event_cfg: Dict[str, Any],
    a3_cfg: Dict[str, Any],
    timeline: Dict[str, int],
    track_map: Dict[str, int],
) -> List[Dict[str, Any]]:
    raw_events = event_cfg.get("events")
    if not isinstance(raw_events, list):
        raise AnnotationError("episode events 必须是数组")

    time_rule = str(event_cfg.get("timeline", {}).get("time_to_frame_rule", "round")).lower()
    frame_map = a3_cfg.get("event_frame_map", {})
    if not isinstance(frame_map, dict):
        raise AnnotationError("A3.3 event_frame_map 必须是 object")

    contact_lookup: Dict[str, int] = {}
    raw_contact = a3_cfg.get("contact_frames", [])
    if not isinstance(raw_contact, list):
        raise AnnotationError("A3.3 contact_frames 必须是数组")
    for index, item in enumerate(raw_contact):
        if not isinstance(item, dict):
            raise AnnotationError("contact_frames[{}] 必须是 object".format(index))
        event_id = require_str(item.get("event_id"), "contact_frames[{}].event_id".format(index))
        frame = require_int(item.get("frame"), "contact_frames[{}].frame".format(index))
        if event_id in contact_lookup and contact_lookup[event_id] != frame:
            raise AnnotationError("事件 {} 存在多个不同 contact frame".format(event_id))
        contact_lookup[event_id] = frame

    team_map = build_team_map(event_cfg)
    seen_ids = set()
    events: List[Dict[str, Any]] = []

    for index, raw in enumerate(raw_events):
        if not isinstance(raw, dict):
            raise AnnotationError("events[{}] 必须是 object".format(index))
        event_id = require_str(raw.get("event_id"), "events[{}].event_id".format(index))
        if event_id in seen_ids:
            raise AnnotationError("event_id 重复：{}".format(event_id))
        seen_ids.add(event_id)

        event_type = require_str(raw.get("type"), event_id + ".type").lower()
        if event_type not in SUPPORTED_EVENT_TYPES:
            raise AnnotationError("{} 类型 {} 不受支持".format(event_id, event_type))
        actor = event_actor(raw)
        if actor is None:
            raise AnnotationError("{} 缺少 actor/from".format(event_id))
        if actor not in track_map:
            raise AnnotationError("{} actor={} 不在 A3.3 objects".format(event_id, actor))
        target = event_target(raw)
        target_selector = None
        if target == "possession_owner":
            target_selector = "possession_owner"
            target = None
        elif target is not None and target not in track_map:
            raise AnnotationError("{} target={} 不在 A3.3 objects".format(event_id, target))

        source_range = source_event_frame_range(raw, event_id, timeline["fps"], time_rule)
        start, end_exc, range_source = source_range
        mapped = frame_map.get(event_id)
        if mapped is not None:
            if not isinstance(mapped, dict):
                raise AnnotationError("event_frame_map.{} 必须是 object".format(event_id))
            mapped_start = require_int(mapped.get("start_frame"), "event_frame_map.{}.start_frame".format(event_id))
            mapped_end, _ = normalize_end_frame(mapped, "event_frame_map.{}".format(event_id))
            if (start, end_exc) != (mapped_start, mapped_end):
                raise AnnotationError(
                    "{} 源事件范围 [{}, {}) 与 event_frame_map [{}, {}) 不一致".format(
                        event_id, start, end_exc, mapped_start, mapped_end
                    )
                )
            start, end_exc = mapped_start, mapped_end
            range_source = "event_frame_map_verified"

        if not (
            timeline["frame_start"] <= start < end_exc <= timeline["frame_end_exclusive"]
        ):
            raise AnnotationError(
                "{} 范围 [{}, {}) 超出 [{}, {})".format(
                    event_id,
                    start,
                    end_exc,
                    timeline["frame_start"],
                    timeline["frame_end_exclusive"],
                )
            )

        contact_frame = contact_lookup.get(event_id)
        if contact_frame is not None and not (start <= contact_frame < end_exc):
            raise AnnotationError(
                "{} contact_frame={} 不在 [{}, {})".format(
                    event_id, contact_frame, start, end_exc
                )
            )

        result = raw.get("result")
        if result is not None:
            result = str(result).strip()
        if result not in ALLOWED_RESULTS:
            raise AnnotationError("{} result={} 不受支持".format(event_id, result))

        source_event_ids = [event_id]
        linked = raw.get("source_event")
        if linked is not None and str(linked).strip() not in source_event_ids:
            source_event_ids.append(str(linked).strip())

        events.append(
            {
                "event_id": event_id,
                "type": event_type,
                "actor_object_id": actor,
                "actor_track_id": track_map[actor],
                "target_object_id": target,
                "target_track_id": track_map[target] if target is not None else None,
                "target_selector": target_selector,
                "start_frame": start,
                "end_frame_exclusive": end_exc,
                "last_frame": end_exc - 1,
                "contact_frame": contact_frame,
                "team_id": team_map.get(actor),
                "team_id_source": (
                    "episode_config.players.{}.team".format(actor)
                    if actor in team_map
                    else None
                ),
                "result": result,
                "result_source": (
                    "episode_config.events.{}.result".format(event_id)
                    if result is not None
                    else None
                ),
                "source_event_ids": source_event_ids,
                "source_interval": range_source,
            }
        )

    missing_frame_map = sorted(set(seen_ids) - set(frame_map))
    extra_frame_map = sorted(set(frame_map) - set(seen_ids))
    if missing_frame_map:
        raise AnnotationError("event_frame_map 缺少事件：{}".format(missing_frame_map))
    if extra_frame_map:
        raise AnnotationError("event_frame_map 出现源配置不存在事件：{}".format(extra_frame_map))

    events.sort(key=lambda item: (item["start_frame"], item["event_id"]))
    return events


def sample_segment(
    timeline: Sequence[Dict[str, Any]],
    frame: int,
    field: str,
) -> Optional[Dict[str, Any]]:
    for index, raw in enumerate(timeline):
        if not isinstance(raw, dict):
            raise AnnotationError("{}[{}] 必须是 object".format(field, index))
        start, end_exc = normalize_segment(raw, "{}[{}]".format(field, index))
        if start <= frame < end_exc:
            return raw
    return None


def build_frame_states(
    a3_cfg: Dict[str, Any],
    events: Sequence[Dict[str, Any]],
    timeline: Dict[str, int],
    track_map: Dict[str, int],
) -> List[Dict[str, Any]]:
    objects = a3_cfg.get("objects", {})
    player_ids = sorted(
        [
            object_id
            for object_id, raw in objects.items()
            if isinstance(raw, dict) and raw.get("category") == "player"
        ],
        key=lambda object_id: track_map[object_id],
    )
    if not player_ids:
        raise AnnotationError("A3.3 中没有 player objects")

    action_timelines: Dict[str, Sequence[Dict[str, Any]]] = {}
    for player_id in player_ids:
        raw = objects[player_id].get("action_timeline")
        if not isinstance(raw, list) or not raw:
            raise AnnotationError("objects.{} 缺少 action_timeline".format(player_id))
        action_timelines[player_id] = raw

    ball_timeline = a3_cfg.get("ball_state_timeline")
    if not isinstance(ball_timeline, list):
        ball_raw = objects.get("Ball_01", {})
        ball_timeline = ball_raw.get("state_timeline", []) if isinstance(ball_raw, dict) else []
    if not isinstance(ball_timeline, list) or not ball_timeline:
        raise AnnotationError("缺少 ball_state_timeline")

    possession_timeline = a3_cfg.get("possession_timeline")
    if not isinstance(possession_timeline, list) or not possession_timeline:
        raise AnnotationError("缺少 possession_timeline")

    active_by_frame: Dict[int, List[str]] = {}
    contacts_by_frame: Dict[int, List[Dict[str, Any]]] = {}
    for event in events:
        for frame in range(event["start_frame"], event["end_frame_exclusive"]):
            active_by_frame.setdefault(frame, []).append(event["event_id"])
        contact_frame = event.get("contact_frame")
        if contact_frame is not None:
            contacts_by_frame.setdefault(contact_frame, []).append(
                {
                    "event_id": event["event_id"],
                    "type": event["type"],
                    "actor_object_id": event["actor_object_id"],
                    "actor_track_id": event["actor_track_id"],
                }
            )

    states: List[Dict[str, Any]] = []
    for frame in range(timeline["frame_start"], timeline["frame_end_exclusive"]):
        player_actions: List[Dict[str, Any]] = []
        for player_id in player_ids:
            segment = sample_segment(
                action_timelines[player_id],
                frame,
                "objects.{}.action_timeline".format(player_id),
            )
            if segment is None:
                raise AnnotationError(
                    "objects.{} action_timeline 未覆盖 frame={}".format(player_id, frame)
                )
            action = require_str(segment.get("action"), "action_timeline.action").lower()
            source_events = segment.get("source_events", segment.get("source_event_ids", []))
            if source_events is None:
                source_events = []
            if isinstance(source_events, str):
                source_events = [source_events]
            if not isinstance(source_events, list):
                raise AnnotationError("action_timeline.source_events 必须是数组或字符串")
            player_actions.append(
                {
                    "object_id": player_id,
                    "track_id": track_map[player_id],
                    "action": action,
                    "source_event_ids": sorted(
                        {str(value).strip() for value in source_events if str(value).strip()}
                    ),
                }
            )

        ball_segment = sample_segment(ball_timeline, frame, "ball_state_timeline")
        if ball_segment is None:
            raise AnnotationError("ball_state_timeline 未覆盖 frame={}".format(frame))
        ball_state = require_str(ball_segment.get("state"), "ball_state_timeline.state")

        possession_segment = sample_segment(
            possession_timeline, frame, "possession_timeline"
        )
        if possession_segment is None:
            raise AnnotationError("possession_timeline 未覆盖 frame={}".format(frame))
        owner = possession_segment.get("owner")
        possession_owner = None
        if owner is not None and str(owner).strip():
            owner_id = str(owner).strip()
            if owner_id not in track_map:
                raise AnnotationError("frame={} owner={} 不在 track map".format(frame, owner_id))
            possession_owner = {
                "object_id": owner_id,
                "track_id": track_map[owner_id],
            }

        states.append(
            {
                "frame": frame,
                "active_events": sorted(active_by_frame.get(frame, [])),
                "player_actions": player_actions,
                "ball_state": ball_state,
                "possession_owner": possession_owner,
                "contact_events": sorted(
                    contacts_by_frame.get(frame, []), key=lambda item: item["event_id"]
                ),
            }
        )
    return states


def verify_consistency(
    events: Sequence[Dict[str, Any]],
    frame_states: Sequence[Dict[str, Any]],
    event_cfg: Dict[str, Any],
    a3_cfg: Dict[str, Any],
    timeline: Dict[str, int],
    track_map: Dict[str, int],
) -> Dict[str, Any]:
    errors: List[str] = []
    warnings: List[str] = []
    event_ids = [event["event_id"] for event in events]
    event_id_set = set(event_ids)

    if len(event_ids) != len(event_id_set):
        errors.append("event_id 不唯一")
    if len(events) != len(event_cfg.get("events", [])):
        errors.append("输出事件数与源事件数不一致")

    source_contact = {
        (str(item.get("event_id")), int(item.get("frame")))
        for item in a3_cfg.get("contact_frames", [])
        if isinstance(item, dict) and item.get("event_id") is not None and item.get("frame") is not None
    }
    output_contact = {
        (event["event_id"], event["contact_frame"])
        for event in events
        if event.get("contact_frame") is not None
    }
    if source_contact != output_contact:
        errors.append(
            "contact_frames 与 A3.3 源数据不一致：source={} output={}".format(
                sorted(source_contact), sorted(output_contact)
            )
        )

    expected_frames = list(range(timeline["frame_start"], timeline["frame_end_exclusive"]))
    actual_frames = [state.get("frame") for state in frame_states]
    if actual_frames != expected_frames:
        errors.append("frame_states 帧范围或连续性错误")

    expected_player_ids = sorted(
        [
            object_id
            for object_id, raw in a3_cfg.get("objects", {}).items()
            if isinstance(raw, dict) and raw.get("category") == "player"
        ],
        key=lambda object_id: track_map[object_id],
    )

    invalid_event_references = 0
    uncovered_player_action_frames = 0
    possession_conflicts = 0
    for state in frame_states:
        frame = state["frame"]
        for event_id in state.get("active_events", []):
            if event_id not in event_id_set:
                invalid_event_references += 1
        action_ids = [item.get("object_id") for item in state.get("player_actions", [])]
        if action_ids != expected_player_ids:
            uncovered_player_action_frames += 1
        owner = state.get("possession_owner")
        ball_state = state.get("ball_state")
        if ball_state == "controlled" and owner is None:
            possession_conflicts += 1
            errors.append("frame={} ball_state=controlled 但无 owner".format(frame))
        if ball_state in {"in_transit", "shot", "free"} and owner is not None:
            possession_conflicts += 1
            errors.append("frame={} ball_state={} 但仍有 owner".format(frame, ball_state))
        for contact in state.get("contact_events", []):
            if contact.get("event_id") not in event_id_set:
                invalid_event_references += 1

    if uncovered_player_action_frames:
        errors.append(
            "{} 帧的 player_actions 未覆盖所有球员或排序错误".format(
                uncovered_player_action_frames
            )
        )
    if invalid_event_references:
        errors.append("存在 {} 个无效事件引用".format(invalid_event_references))

    for event in events:
        if event["last_frame"] != event["end_frame_exclusive"] - 1:
            errors.append("{} last_frame 不一致".format(event["event_id"]))
        if event["actor_track_id"] != track_map.get(event["actor_object_id"]):
            errors.append("{} actor track 映射不一致".format(event["event_id"]))
        target = event.get("target_object_id")
        if target is not None and event["target_track_id"] != track_map.get(target):
            errors.append("{} target track 映射不一致".format(event["event_id"]))

    return {
        "schema_version": "1.0",
        "generator_version": SCRIPT_VERSION,
        "status": "ERROR" if errors else ("WARNING" if warnings else "PASS"),
        "errors": errors,
        "warnings": warnings,
        "events": len(events),
        "frames": len(frame_states),
        "contact_frames": len(output_contact),
        "contact_frame_values": sorted(frame for _, frame in output_contact),
        "possession_conflicts": possession_conflicts,
        "uncovered_player_action_frames": uncovered_player_action_frames,
        "invalid_event_references": invalid_event_references,
    }


def output_paths(output_dir: Path, seq_id: str) -> Tuple[Path, Path, Path]:
    return (
        output_dir / "events_{}.json".format(seq_id),
        output_dir / "frame_states_{}.jsonl".format(seq_id),
        output_dir / "event_annotation_report_{}.json".format(seq_id),
    )


def write_outputs(
    output_dir: Path,
    event_config_path: Path,
    a3_config_path: Path,
    events: Sequence[Dict[str, Any]],
    frame_states: Sequence[Dict[str, Any]],
    a3_cfg: Dict[str, Any],
    timeline: Dict[str, int],
    track_map: Dict[str, int],
    report: Dict[str, Any],
) -> Tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    seq_id = require_str(a3_cfg.get("seq_id"), "A3.3 seq_id")
    events_path, states_path, report_path = output_paths(output_dir, seq_id)

    source_files = [
        {
            "role": "episode_event_config",
            "path": event_config_path.resolve().as_posix(),
            "sha256": sha256_file(event_config_path),
        },
        {
            "role": "a3_enhanced_config",
            "path": a3_config_path.resolve().as_posix(),
            "sha256": sha256_file(a3_config_path),
        },
    ]
    events_output = {
        "schema_version": "1.0",
        "generator_version": SCRIPT_VERSION,
        "seq_id": seq_id,
        "fps": timeline["fps"],
        "frame_start": timeline["frame_start"],
        "frame_end_exclusive": timeline["frame_end_exclusive"],
        "last_frame": timeline["frame_end"],
        "total_frames": timeline["total_frames"],
        "interval_convention": "[start_frame,end_frame_exclusive)",
        "source_files": source_files,
        "object_track_map": dict(sorted(track_map.items(), key=lambda item: item[1])),
        "roster": a3_cfg.get("roster"),
        "movement_optimization": a3_cfg.get("movement_optimization"),
        "player_count": sum(
            1
            for raw in a3_cfg.get("objects", {}).values()
            if isinstance(raw, dict) and raw.get("category") == "player"
        ),
        "event_count": len(events),
        "events": list(events),
    }
    states_text = "".join(
        json.dumps(state, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
        for state in frame_states
    )
    atomic_write_json(events_path, events_output)
    atomic_write_text(states_path, states_text)
    atomic_write_json(report_path, report)
    return events_path, states_path, report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A3.3c 事件标注生成器")
    parser.add_argument("--episode-config", type=Path, default=DEFAULT_EVENT_CONFIG)
    parser.add_argument("--a3-config", type=Path, default=DEFAULT_A3_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--strict", action="store_true", help="将 WARNING 视为失败")
    parser.add_argument("--overwrite", action="store_true", help="覆盖本 seq_id 的三份输出")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    event_config_path = args.episode_config.expanduser().resolve()
    a3_config_path = args.a3_config.expanduser().resolve()
    output_dir = args.output_dir.expanduser()
    if not output_dir.is_absolute():
        output_dir = (Path.cwd() / output_dir).resolve()
    else:
        output_dir = output_dir.resolve()

    try:
        event_cfg = load_json(event_config_path, "事件配置")
        a3_cfg = load_json(a3_config_path, "A3.3 配置")
        timeline = timeline_info(a3_cfg)
        track_map = build_track_map(a3_cfg)
        seq_id = require_str(a3_cfg.get("seq_id"), "A3.3 seq_id")
        paths = output_paths(output_dir, seq_id)
        existing = [path for path in paths if path.exists()]
        if existing and not args.overwrite:
            raise AnnotationError(
                "输出已存在；使用 --overwrite：{}".format(
                    ", ".join(str(path) for path in existing)
                )
            )

        events = build_events(event_cfg, a3_cfg, timeline, track_map)
        frame_states = build_frame_states(a3_cfg, events, timeline, track_map)
        report = verify_consistency(
            events, frame_states, event_cfg, a3_cfg, timeline, track_map
        )
        if args.strict and report["warnings"]:
            report["errors"].append("strict mode: warnings treated as errors")
            report["status"] = "ERROR"

        print("[A3.3c] {}".format(SCRIPT_VERSION))
        print("  seq_id={}".format(seq_id))
        print("  events={}".format(len(events)))
        print("  frames={}".format(len(frame_states)))
        print("  contact_frames={}".format(report["contact_frame_values"]))
        print("  status={} errors={} warnings={}".format(
            report["status"], len(report["errors"]), len(report["warnings"])
        ))

        # Always keep a validation report; do not emit usable annotations on ERROR.
        output_dir.mkdir(parents=True, exist_ok=True)
        if report["status"] == "ERROR":
            atomic_write_json(paths[2], report)
            for error in report["errors"]:
                print("  ERROR: {}".format(error), file=sys.stderr)
            print("[FAILED] 一致性 ERROR；未写出 events/frame_states。", file=sys.stderr)
            return 1

        events_path, states_path, report_path = write_outputs(
            output_dir,
            event_config_path,
            a3_config_path,
            events,
            frame_states,
            a3_cfg,
            timeline,
            track_map,
            report,
        )
        print("  Events JSON: {}".format(events_path))
        print("  Frame states: {}".format(states_path))
        print("  Report: {}".format(report_path))

        sample_frames = set()
        for frame in report["contact_frame_values"]:
            for candidate in (frame - 1, frame, frame + 1):
                if timeline["frame_start"] <= candidate <= timeline["frame_end"]:
                    sample_frames.add(candidate)
        lookup = {state["frame"]: state for state in frame_states}
        if sample_frames:
            print("[INFO] contact boundary frames:")
            for frame in sorted(sample_frames):
                state = lookup[frame]
                print(
                    "  frame {} ball={} owner={} contact={}".format(
                        frame,
                        state["ball_state"],
                        state["possession_owner"]["object_id"]
                        if state["possession_owner"]
                        else None,
                        [item["event_id"] for item in state["contact_events"]],
                    )
                )
        print("[DONE] A3.3c event annotations generated")
        return 0

    except AnnotationError as exc:
        print("[ERROR] {}".format(exc), file=sys.stderr)
        return 2
    except Exception as exc:
        print(
            "[UNEXPECTED ERROR] {}: {}".format(type(exc).__name__, exc),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
