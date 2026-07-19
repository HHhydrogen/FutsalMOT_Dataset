#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FutsalMOT episode-event configuration validator.

Version:
    A3_1_EPISODE_VALIDATOR_8P_V2

Checks:
- JSON/schema integrity.
- Timeline consistency and time-to-frame conversion.
- Player, ball, team, track ID, and class ID integrity.
- Event type-specific required fields.
- Event time range and frame alignment.
- Same-player interval overlap.
- Target coordinate and court-boundary validity.
- Pass/receive pairing.
- Possession continuity for dribble, pass, receive, and shot.
- Basic implied player-speed checks for explicit move/dribble targets.

Outputs:
    Saved/FutsalMOT/episode_reports/
    ├─ episode_report_<episode_id>.json
    └─ episode_timeline_<episode_id>.csv

Exit codes:
    0 = no ERROR
    1 = ERROR exists, or WARNING under --strict-warnings
    2 = fatal configuration/program failure
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from futsalmot.core.paths import CONFIG_DIR


SCRIPT_VERSION = "A3_1_EPISODE_VALIDATOR_8P_V2"

DEFAULT_CONFIG_PATH = (
    CONFIG_DIR
    / "events"
    / "episode_test_0001.json"
)

PASS = "PASS"
WARNING = "WARNING"
ERROR = "ERROR"

LEVEL_RANK = {PASS: 0, WARNING: 1, ERROR: 2}

INTERVAL_TYPES = {
    "hold",
    "move",
    "dribble",
    "pass",
    "defend_follow",
    "shot",
}
INSTANT_TYPES = {"receive"}
DEFAULT_SUPPORTED_TYPES = INTERVAL_TYPES | INSTANT_TYPES

EXCLUSIVE_ACTOR_TYPES = {
    "hold",
    "move",
    "dribble",
    "pass",
    "defend_follow",
    "shot",
}

POSSESSION_PRIORITY = {
    "receive": 0,
    "dribble": 1,
    "hold": 1,
    "pass": 2,
    "shot": 3,
}


class ConfigError(RuntimeError):
    pass


def is_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def require_float(value: Any, field: str) -> float:
    if not is_number(value):
        raise ConfigError(
            "字段 '{}' 必须是有限数值，当前值={!r}".format(field, value)
        )
    return float(value)


def require_int(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise ConfigError(
            "字段 '{}' 必须是整数，当前值={!r}".format(field, value)
        )
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return int(value)
    raise ConfigError(
        "字段 '{}' 必须是整数，当前值={!r}".format(field, value)
    )


def require_str(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError("字段 '{}' 必须是非空字符串".format(field))
    return value.strip()


def parse_vec3(value: Any, field: str) -> Tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ConfigError(
            "字段 '{}' 必须是长度为 3 的数组 [x, y, z]".format(field)
        )
    return (
        require_float(value[0], field + "[0]"),
        require_float(value[1], field + "[1]"),
        require_float(value[2], field + "[2]"),
    )


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise ConfigError("找不到配置文件：{}".format(path))
    try:
        with path.open("r", encoding="utf-8-sig") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise ConfigError(
            "JSON 解析失败：{}，line={} column={}".format(
                exc.msg, exc.lineno, exc.colno
            )
        ) from exc
    except OSError as exc:
        raise ConfigError("读取配置失败：{}".format(exc)) from exc

    if not isinstance(data, dict):
        raise ConfigError("配置顶层必须是 JSON object")
    return data


def issue(
    level: str,
    code: str,
    message: str,
    *,
    event_id: Optional[str] = None,
    player_id: Optional[str] = None,
    time_sec: Optional[float] = None,
    details: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "level": level,
        "code": code,
        "message": message,
    }
    if event_id is not None:
        result["event_id"] = event_id
    if player_id is not None:
        result["player_id"] = player_id
    if time_sec is not None:
        result["time_sec"] = time_sec
    if details:
        result["details"] = details
    return result


def max_level(levels: Iterable[str]) -> str:
    result = PASS
    for level in levels:
        if LEVEL_RANK.get(level, -1) > LEVEL_RANK[result]:
            result = level
    return result


def time_to_frame(time_sec: float, fps: float) -> int:
    return int(round(time_sec * fps))


def point_in_court(
    loc: Sequence[float],
    court: Dict[str, float],
    *,
    allow_goal_plane: bool = False,
) -> bool:
    x_tolerance = 1.0 if not allow_goal_plane else 60.0
    return (
        court["x_min_cm"] - x_tolerance
        <= loc[0]
        <= court["x_max_cm"] + x_tolerance
        and court["y_min_cm"] - 1.0
        <= loc[1]
        <= court["y_max_cm"] + 1.0
    )


def distance_xy(a: Sequence[float], b: Sequence[float]) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def get_event_actor(event: Dict[str, Any]) -> Optional[str]:
    event_type = event.get("type")
    if event_type in {"hold", "move", "dribble", "receive", "defend_follow", "shot"}:
        actor = event.get("actor")
        return actor if isinstance(actor, str) else None
    if event_type == "pass":
        actor = event.get("from")
        return actor if isinstance(actor, str) else None
    return None


def parse_timeline(config: Dict[str, Any]) -> Dict[str, Any]:
    raw = config.get("timeline")
    if not isinstance(raw, dict):
        raise ConfigError("timeline 必须是 JSON object")

    fps = require_float(raw.get("fps"), "timeline.fps")
    duration = require_float(
        raw.get("duration_sec"), "timeline.duration_sec"
    )
    frame_start = require_int(
        raw.get("frame_start"), "timeline.frame_start"
    )
    frame_end = require_int(raw.get("frame_end"), "timeline.frame_end")

    if fps <= 0:
        raise ConfigError("timeline.fps 必须大于 0")
    if duration <= 0:
        raise ConfigError("timeline.duration_sec 必须大于 0")
    if frame_end < frame_start:
        raise ConfigError("timeline.frame_end 不能小于 frame_start")

    return {
        "fps": fps,
        "duration_sec": duration,
        "frame_start": frame_start,
        "frame_end": frame_end,
        "time_to_frame_rule": raw.get("time_to_frame_rule", "round"),
        "interval_semantics": raw.get(
            "interval_semantics", "[start_t, end_t)"
        ),
    }


def parse_court(config: Dict[str, Any]) -> Dict[str, float]:
    raw = config.get("court")
    if not isinstance(raw, dict):
        raise ConfigError("court 必须是 JSON object")

    result = {
        "x_min_cm": require_float(raw.get("x_min_cm"), "court.x_min_cm"),
        "x_max_cm": require_float(raw.get("x_max_cm"), "court.x_max_cm"),
        "y_min_cm": require_float(raw.get("y_min_cm"), "court.y_min_cm"),
        "y_max_cm": require_float(raw.get("y_max_cm"), "court.y_max_cm"),
        "player_z_cm": require_float(
            raw.get("player_z_cm"), "court.player_z_cm"
        ),
        "ball_z_cm": require_float(
            raw.get("ball_z_cm"), "court.ball_z_cm"
        ),
    }

    if result["x_min_cm"] >= result["x_max_cm"]:
        raise ConfigError("court.x_min_cm 必须小于 x_max_cm")
    if result["y_min_cm"] >= result["y_max_cm"]:
        raise ConfigError("court.y_min_cm 必须小于 y_max_cm")
    return result


def parse_players(
    config: Dict[str, Any],
    court: Dict[str, float],
    issues: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    raw_players = config.get("players")
    if not isinstance(raw_players, dict) or not raw_players:
        raise ConfigError("players 必须是非空 JSON object")

    players: Dict[str, Dict[str, Any]] = {}
    seen_track_ids: Dict[int, str] = {}

    for player_id, raw in raw_players.items():
        if not isinstance(raw, dict):
            issues.append(
                issue(
                    ERROR,
                    "INVALID_PLAYER_CONFIG",
                    "players.{} 必须是 JSON object".format(player_id),
                    player_id=player_id,
                )
            )
            continue

        try:
            team = require_str(raw.get("team"), "players.{}.team".format(player_id))
            role = require_str(raw.get("role"), "players.{}.role".format(player_id))
            track_id = require_int(
                raw.get("track_id"), "players.{}.track_id".format(player_id)
            )
            class_id = require_int(
                raw.get("class_id"), "players.{}.class_id".format(player_id)
            )
            start_loc = parse_vec3(
                raw.get("start_loc"), "players.{}.start_loc".format(player_id)
            )
            max_speed = require_float(
                raw.get("max_speed_cm_s", 750.0),
                "players.{}.max_speed_cm_s".format(player_id),
            )
        except ConfigError as exc:
            issues.append(
                issue(
                    ERROR,
                    "INVALID_PLAYER_FIELD",
                    str(exc),
                    player_id=player_id,
                )
            )
            continue

        if track_id in seen_track_ids:
            issues.append(
                issue(
                    ERROR,
                    "DUPLICATE_PLAYER_TRACK_ID",
                    "track_id={} 同时用于 {} 和 {}".format(
                        track_id, seen_track_ids[track_id], player_id
                    ),
                    player_id=player_id,
                )
            )
        else:
            seen_track_ids[track_id] = player_id

        if not point_in_court(start_loc, court):
            issues.append(
                issue(
                    ERROR,
                    "PLAYER_START_OUTSIDE_COURT",
                    "{} 起点超出球场安全范围：{}".format(
                        player_id, list(start_loc)
                    ),
                    player_id=player_id,
                )
            )

        players[player_id] = {
            "team": team,
            "role": role,
            "track_id": track_id,
            "class_id": class_id,
            "start_loc": start_loc,
            "max_speed_cm_s": max_speed,
        }

    return players


def parse_ball(
    config: Dict[str, Any],
    players: Dict[str, Dict[str, Any]],
    court: Dict[str, float],
    issues: List[Dict[str, Any]],
) -> Dict[str, Any]:
    raw = config.get("ball")
    if not isinstance(raw, dict):
        raise ConfigError("ball 必须是 JSON object")

    ball = {
        "object_id": require_str(raw.get("object_id"), "ball.object_id"),
        "track_id": require_int(raw.get("track_id"), "ball.track_id"),
        "class_id": require_int(raw.get("class_id"), "ball.class_id"),
        "initial_owner": require_str(
            raw.get("initial_owner"), "ball.initial_owner"
        ),
        "initial_loc": parse_vec3(
            raw.get("initial_loc"), "ball.initial_loc"
        ),
        "height_cm": require_float(
            raw.get("height_cm"), "ball.height_cm"
        ),
        "dribble_ahead_cm": require_float(
            raw.get("dribble_ahead_cm", 45.0),
            "ball.dribble_ahead_cm",
        ),
    }

    if ball["initial_owner"] not in players:
        issues.append(
            issue(
                ERROR,
                "UNKNOWN_INITIAL_OWNER",
                "ball.initial_owner={} 不存在".format(
                    ball["initial_owner"]
                ),
            )
        )

    if not point_in_court(ball["initial_loc"], court):
        issues.append(
            issue(
                ERROR,
                "BALL_START_OUTSIDE_COURT",
                "ball.initial_loc 超出球场范围：{}".format(
                    list(ball["initial_loc"])
                ),
            )
        )

    if ball["initial_owner"] in players:
        player_loc = players[ball["initial_owner"]]["start_loc"]
        initial_distance = distance_xy(player_loc, ball["initial_loc"])
        expected = ball["dribble_ahead_cm"]

        if abs(initial_distance - expected) > 30.0:
            issues.append(
                issue(
                    WARNING,
                    "BALL_OWNER_START_DISTANCE",
                    (
                        "初始球与持球队员距离 {:.2f} cm，配置期望约 {:.2f} cm"
                    ).format(initial_distance, expected),
                    player_id=ball["initial_owner"],
                )
            )

    return ball


def check_roster_and_initial_spacing(
    config: Dict[str, Any],
    players: Dict[str, Dict[str, Any]],
    ball: Dict[str, Any],
    rules: Dict[str, Any],
    issues: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Validate optional 4v4 roster metadata and deterministic start spacing."""
    summary: Dict[str, Any] = {
        "player_count": len(players),
        "team_counts": {},
        "minimum_initial_player_distance_cm": None,
        "minimum_initial_player_pair": None,
    }

    team_counts: Dict[str, int] = {}
    for player in players.values():
        team = player.get("team")
        team_counts[team] = team_counts.get(team, 0) + 1
    summary["team_counts"] = dict(sorted(team_counts.items()))

    expected_count = rules.get("expected_player_count")
    if expected_count is not None and len(players) != expected_count:
        issues.append(
            issue(
                ERROR,
                "UNEXPECTED_PLAYER_COUNT",
                "球员数={}，预期 {}".format(len(players), expected_count),
                details={"actual": len(players), "expected": expected_count},
            )
        )

    expected_team_size = rules.get("expected_team_size")
    if expected_team_size is not None:
        for team_id in ("A", "B"):
            actual = team_counts.get(team_id, 0)
            if actual != expected_team_size:
                issues.append(
                    issue(
                        ERROR,
                        "UNEXPECTED_TEAM_SIZE",
                        "Team {} 场上球员数={}，预期 {}".format(
                            team_id, actual, expected_team_size
                        ),
                        details={
                            "team": team_id,
                            "actual": actual,
                            "expected": expected_team_size,
                        },
                    )
                )

    roster = config.get("roster")
    if roster is not None:
        if not isinstance(roster, dict):
            issues.append(issue(ERROR, "INVALID_ROSTER", "roster 必须是 JSON object"))
        else:
            summary["format"] = roster.get("format")
            declared_count = roster.get("player_count")
            if declared_count is not None:
                try:
                    declared_count = require_int(declared_count, "roster.player_count")
                    if declared_count != len(players):
                        issues.append(
                            issue(
                                ERROR,
                                "ROSTER_PLAYER_COUNT_MISMATCH",
                                "roster.player_count={}，实际 players={}".format(
                                    declared_count, len(players)
                                ),
                            )
                        )
                except ConfigError as exc:
                    issues.append(issue(ERROR, "INVALID_ROSTER_FIELD", str(exc)))

            goalkeepers = roster.get("goalkeepers", [])
            if not isinstance(goalkeepers, list):
                issues.append(
                    issue(ERROR, "INVALID_GOALKEEPERS", "roster.goalkeepers 必须是数组")
                )
            elif goalkeepers:
                issues.append(
                    issue(
                        ERROR,
                        "GOALKEEPERS_NOT_ALLOWED",
                        "当前 4v4 outfield 配置不允许守门员：{}".format(goalkeepers),
                    )
                )

            declared_teams = roster.get("teams")
            if declared_teams is not None:
                if not isinstance(declared_teams, dict):
                    issues.append(
                        issue(ERROR, "INVALID_ROSTER_TEAMS", "roster.teams 必须是 object")
                    )
                else:
                    declared_players: List[str] = []
                    for team_id, members in declared_teams.items():
                        if not isinstance(members, list):
                            issues.append(
                                issue(
                                    ERROR,
                                    "INVALID_ROSTER_TEAM_MEMBERS",
                                    "roster.teams.{} 必须是数组".format(team_id),
                                )
                            )
                            continue
                        member_ids = [str(value) for value in members]
                        declared_players.extend(member_ids)
                        actual_members = sorted(
                            player_id
                            for player_id, player in players.items()
                            if player.get("team") == team_id
                        )
                        if sorted(member_ids) != actual_members:
                            issues.append(
                                issue(
                                    ERROR,
                                    "ROSTER_TEAM_MEMBERS_MISMATCH",
                                    "roster.teams.{}={}，实际 {}".format(
                                        team_id, sorted(member_ids), actual_members
                                    ),
                                )
                            )
                    if len(declared_players) != len(set(declared_players)):
                        issues.append(
                            issue(
                                ERROR,
                                "DUPLICATE_ROSTER_PLAYER",
                                "roster.teams 中存在重复球员",
                            )
                        )
                    if set(declared_players) != set(players):
                        issues.append(
                            issue(
                                ERROR,
                                "ROSTER_PLAYER_SET_MISMATCH",
                                "roster.teams 球员集合与 players 不一致",
                            )
                        )

    player_track_ids = {player["track_id"] for player in players.values()}
    if ball.get("track_id") in player_track_ids:
        issues.append(
            issue(
                ERROR,
                "BALL_TRACK_ID_CONFLICT",
                "ball.track_id={} 与球员 track_id 冲突".format(ball.get("track_id")),
            )
        )

    minimum = float("inf")
    minimum_pair: Optional[Tuple[str, str]] = None
    player_items = sorted(players.items())
    for index, (left_id, left) in enumerate(player_items):
        for right_id, right in player_items[index + 1 :]:
            distance = distance_xy(left["start_loc"], right["start_loc"])
            if distance < minimum:
                minimum = distance
                minimum_pair = (left_id, right_id)

    if not math.isinf(minimum):
        summary["minimum_initial_player_distance_cm"] = minimum
        summary["minimum_initial_player_pair"] = list(minimum_pair) if minimum_pair else None
        required = rules.get("minimum_initial_player_distance_cm", 0.0)
        if required > 0.0 and minimum < required:
            issues.append(
                issue(
                    ERROR,
                    "INITIAL_PLAYER_SPACING_TOO_SMALL",
                    "{} 与 {} 初始距离 {:.2f} cm，小于要求 {:.2f} cm".format(
                        minimum_pair[0], minimum_pair[1], minimum, required
                    ),
                    details={
                        "minimum_distance_cm": minimum,
                        "required_distance_cm": required,
                        "pair": list(minimum_pair),
                    },
                )
            )

    return summary


def parse_rules(config: Dict[str, Any]) -> Dict[str, Any]:
    raw = config.get("event_rules", {})
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ConfigError("event_rules 必须是 JSON object")

    supported = raw.get("supported_types", sorted(DEFAULT_SUPPORTED_TYPES))
    if not isinstance(supported, list) or not supported:
        raise ConfigError("event_rules.supported_types 必须是非空数组")

    supported_types = set()
    for index, value in enumerate(supported):
        supported_types.add(
            require_str(
                value, "event_rules.supported_types[{}]".format(index)
            )
        )

    unknown = supported_types - DEFAULT_SUPPORTED_TYPES
    if unknown:
        raise ConfigError(
            "event_rules 中包含验证器不支持的类型：{}".format(
                sorted(unknown)
            )
        )

    return {
        "supported_types": supported_types,
        "require_pass_receive_pair": bool(
            raw.get("require_pass_receive_pair", True)
        ),
        "pass_receive_time_tolerance_sec": require_float(
            raw.get("pass_receive_time_tolerance_sec", 0.0001),
            "event_rules.pass_receive_time_tolerance_sec",
        ),
        "allow_same_actor_boundary_touch": bool(
            raw.get("allow_same_actor_boundary_touch", True)
        ),
        "require_possession_for_dribble": bool(
            raw.get("require_possession_for_dribble", True)
        ),
        "require_possession_for_pass": bool(
            raw.get("require_possession_for_pass", True)
        ),
        "require_possession_for_shot": bool(
            raw.get("require_possession_for_shot", True)
        ),
        "expected_player_count": (
            require_int(
                raw.get("expected_player_count"),
                "event_rules.expected_player_count",
            )
            if raw.get("expected_player_count") is not None
            else None
        ),
        "expected_team_size": (
            require_int(
                raw.get("expected_team_size"),
                "event_rules.expected_team_size",
            )
            if raw.get("expected_team_size") is not None
            else None
        ),
        "minimum_initial_player_distance_cm": require_float(
            raw.get("minimum_initial_player_distance_cm", 0.0),
            "event_rules.minimum_initial_player_distance_cm",
        ),
    }


def event_times(
    raw: Dict[str, Any],
    event_type: str,
    event_id: str,
) -> Tuple[float, float, bool]:
    if event_type in INTERVAL_TYPES:
        start_t = require_float(
            raw.get("start_t"), "events.{}.start_t".format(event_id)
        )
        end_t = require_float(
            raw.get("end_t"), "events.{}.end_t".format(event_id)
        )
        return start_t, end_t, False

    time_value = require_float(
        raw.get("time"), "events.{}.time".format(event_id)
    )
    return time_value, time_value, True


def validate_target_loc(
    raw: Dict[str, Any],
    field: str,
    event_id: str,
    event_type: str,
    court: Dict[str, float],
    issues: List[Dict[str, Any]],
) -> Optional[Tuple[float, float, float]]:
    try:
        loc = parse_vec3(raw.get(field), "events.{}.{}".format(event_id, field))
    except ConfigError as exc:
        issues.append(
            issue(ERROR, "INVALID_TARGET_LOC", str(exc), event_id=event_id)
        )
        return None

    if not point_in_court(
        loc,
        court,
        allow_goal_plane=(event_type == "shot"),
    ):
        issues.append(
            issue(
                ERROR,
                "TARGET_OUTSIDE_COURT",
                "{} 的目标坐标超出允许范围：{}".format(
                    event_id, list(loc)
                ),
                event_id=event_id,
            )
        )
    return loc


def parse_events(
    config: Dict[str, Any],
    timeline: Dict[str, Any],
    court: Dict[str, float],
    players: Dict[str, Dict[str, Any]],
    rules: Dict[str, Any],
    issues: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    raw_events = config.get("events")
    if not isinstance(raw_events, list) or not raw_events:
        raise ConfigError("events 必须是非空数组")

    events: List[Dict[str, Any]] = []
    seen_ids = set()
    duration = timeline["duration_sec"]
    fps = timeline["fps"]

    for index, raw in enumerate(raw_events):
        if not isinstance(raw, dict):
            issues.append(
                issue(
                    ERROR,
                    "INVALID_EVENT_CONFIG",
                    "events[{}] 必须是 JSON object".format(index),
                )
            )
            continue

        try:
            event_id = require_str(
                raw.get("event_id"), "events[{}].event_id".format(index)
            )
            event_type = require_str(
                raw.get("type"), "events[{}].type".format(index)
            )
        except ConfigError as exc:
            issues.append(issue(ERROR, "INVALID_EVENT_HEADER", str(exc)))
            continue

        if event_id in seen_ids:
            issues.append(
                issue(
                    ERROR,
                    "DUPLICATE_EVENT_ID",
                    "重复 event_id={}".format(event_id),
                    event_id=event_id,
                )
            )
        seen_ids.add(event_id)

        if event_type not in rules["supported_types"]:
            issues.append(
                issue(
                    ERROR,
                    "UNSUPPORTED_EVENT_TYPE",
                    "不支持 event type={!r}".format(event_type),
                    event_id=event_id,
                )
            )
            continue

        try:
            start_t, end_t, instantaneous = event_times(
                raw, event_type, event_id
            )
        except ConfigError as exc:
            issues.append(
                issue(ERROR, "INVALID_EVENT_TIME", str(exc), event_id=event_id)
            )
            continue

        if start_t < 0.0 or end_t > duration + 1e-9:
            issues.append(
                issue(
                    ERROR,
                    "EVENT_OUTSIDE_TIMELINE",
                    "{} 时间范围 {}..{} 超出 0..{}".format(
                        event_id, start_t, end_t, duration
                    ),
                    event_id=event_id,
                    time_sec=start_t,
                )
            )

        if not instantaneous and end_t <= start_t:
            issues.append(
                issue(
                    ERROR,
                    "NON_POSITIVE_EVENT_DURATION",
                    "{} 的 end_t 必须大于 start_t".format(event_id),
                    event_id=event_id,
                    time_sec=start_t,
                )
            )

        start_frame = time_to_frame(start_t, fps)
        end_frame_exclusive = (
            start_frame if instantaneous else time_to_frame(end_t, fps)
        )

        for label, time_value in (
            ("start/time", start_t),
            ("end", end_t),
        ):
            frame_float = time_value * fps
            if abs(frame_float - round(frame_float)) > 1e-6:
                issues.append(
                    issue(
                        WARNING,
                        "TIME_NOT_FRAME_ALIGNED",
                        (
                            "{} {}={} s 不严格落在帧上，换算为 {:.6f} frame"
                        ).format(
                            event_id, label, time_value, frame_float
                        ),
                        event_id=event_id,
                        time_sec=time_value,
                    )
                )

        parsed = dict(raw)
        parsed.update(
            {
                "event_id": event_id,
                "type": event_type,
                "_start_t": start_t,
                "_end_t": end_t,
                "_instantaneous": instantaneous,
                "_start_frame": start_frame,
                "_end_frame_exclusive": end_frame_exclusive,
                "_last_frame": (
                    start_frame
                    if instantaneous
                    else max(start_frame, end_frame_exclusive - 1)
                ),
            }
        )

        actor = get_event_actor(parsed)
        if actor is not None and actor not in players:
            issues.append(
                issue(
                    ERROR,
                    "UNKNOWN_EVENT_ACTOR",
                    "{} 引用了不存在的 actor={}".format(
                        event_id, actor
                    ),
                    event_id=event_id,
                    player_id=actor,
                )
            )

        if event_type in {"hold", "move", "dribble", "receive", "defend_follow", "shot"}:
            if not isinstance(raw.get("actor"), str):
                issues.append(
                    issue(
                        ERROR,
                        "MISSING_ACTOR",
                        "{} 缺少 actor".format(event_id),
                        event_id=event_id,
                    )
                )

        if event_type in {"move", "dribble", "shot"}:
            parsed["_target_loc"] = validate_target_loc(
                raw, "target_loc", event_id, event_type, court, issues
            )

        if event_type == "pass":
            from_player = raw.get("from")
            to_player = raw.get("to")

            if not isinstance(from_player, str) or from_player not in players:
                issues.append(
                    issue(
                        ERROR,
                        "INVALID_PASS_FROM",
                        "{} 的 from 无效：{}".format(
                            event_id, from_player
                        ),
                        event_id=event_id,
                    )
                )

            if not isinstance(to_player, str) or to_player not in players:
                issues.append(
                    issue(
                        ERROR,
                        "INVALID_PASS_TO",
                        "{} 的 to 无效：{}".format(
                            event_id, to_player
                        ),
                        event_id=event_id,
                    )
                )

            if from_player == to_player:
                issues.append(
                    issue(
                        ERROR,
                        "PASS_TO_SELF",
                        "{} 不能传给自己".format(event_id),
                        event_id=event_id,
                    )
                )

        if event_type == "receive":
            source_event = raw.get("source_event")
            if not isinstance(source_event, str) or not source_event.strip():
                issues.append(
                    issue(
                        ERROR,
                        "MISSING_RECEIVE_SOURCE",
                        "{} 缺少 source_event".format(event_id),
                        event_id=event_id,
                    )
                )

        if event_type == "defend_follow":
            target = raw.get("target")
            if target != "possession_owner" and target not in players:
                issues.append(
                    issue(
                        ERROR,
                        "INVALID_DEFEND_TARGET",
                        "{} 的 target 必须是球员或 possession_owner".format(
                            event_id
                        ),
                        event_id=event_id,
                    )
                )

            for numeric_field in ("follow_distance_cm", "side_offset_cm"):
                if numeric_field in raw:
                    try:
                        require_float(
                            raw[numeric_field],
                            "events.{}.{}".format(event_id, numeric_field),
                        )
                    except ConfigError as exc:
                        issues.append(
                            issue(
                                ERROR,
                                "INVALID_DEFEND_PARAMETER",
                                str(exc),
                                event_id=event_id,
                            )
                        )

        events.append(parsed)

    return events


def check_timeline_consistency(
    timeline: Dict[str, Any],
    issues: List[Dict[str, Any]],
) -> None:
    expected_frame_end = (
        timeline["frame_start"]
        + int(round(timeline["duration_sec"] * timeline["fps"]))
        - 1
    )

    if timeline["frame_end"] != expected_frame_end:
        issues.append(
            issue(
                ERROR,
                "TIMELINE_FRAME_END_MISMATCH",
                (
                    "duration_sec × fps 对应 frame_end={}，配置为 {}"
                ).format(expected_frame_end, timeline["frame_end"]),
            )
        )


def check_actor_overlaps(
    events: List[Dict[str, Any]],
    rules: Dict[str, Any],
    issues: List[Dict[str, Any]],
) -> None:
    by_actor: Dict[str, List[Dict[str, Any]]] = {}

    for event in events:
        if event["type"] not in EXCLUSIVE_ACTOR_TYPES:
            continue
        actor = get_event_actor(event)
        if actor is None:
            continue
        by_actor.setdefault(actor, []).append(event)

    tolerance = 1e-9

    for actor, actor_events in by_actor.items():
        actor_events.sort(
            key=lambda e: (e["_start_t"], e["_end_t"], e["event_id"])
        )

        for left, right in zip(actor_events, actor_events[1:]):
            if rules["allow_same_actor_boundary_touch"]:
                overlaps = right["_start_t"] < left["_end_t"] - tolerance
            else:
                overlaps = right["_start_t"] <= left["_end_t"] + tolerance

            if overlaps:
                issues.append(
                    issue(
                        ERROR,
                        "ACTOR_EVENT_OVERLAP",
                        (
                            "{} 的事件 {}({}–{}) 与 {}({}–{}) 重叠"
                        ).format(
                            actor,
                            left["event_id"],
                            left["_start_t"],
                            left["_end_t"],
                            right["event_id"],
                            right["_start_t"],
                            right["_end_t"],
                        ),
                        player_id=actor,
                        time_sec=right["_start_t"],
                        details={
                            "left_event": left["event_id"],
                            "right_event": right["event_id"],
                        },
                    )
                )


def check_pass_receive_pairs(
    events: List[Dict[str, Any]],
    rules: Dict[str, Any],
    issues: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    by_id = {event["event_id"]: event for event in events}
    pass_to_receive: Dict[str, Dict[str, Any]] = {}
    tolerance = rules["pass_receive_time_tolerance_sec"]

    receives = [event for event in events if event["type"] == "receive"]

    for receive in receives:
        source_id = receive.get("source_event")
        source = by_id.get(source_id)

        if source is None:
            issues.append(
                issue(
                    ERROR,
                    "RECEIVE_SOURCE_NOT_FOUND",
                    "{} 的 source_event={} 不存在".format(
                        receive["event_id"], source_id
                    ),
                    event_id=receive["event_id"],
                )
            )
            continue

        if source["type"] != "pass":
            issues.append(
                issue(
                    ERROR,
                    "RECEIVE_SOURCE_NOT_PASS",
                    "{} 的 source_event={} 不是 pass".format(
                        receive["event_id"], source_id
                    ),
                    event_id=receive["event_id"],
                )
            )
            continue

        if receive.get("actor") != source.get("to"):
            issues.append(
                issue(
                    ERROR,
                    "PASS_RECEIVER_MISMATCH",
                    (
                        "{} actor={} 与 pass {} 的 to={} 不一致"
                    ).format(
                        receive["event_id"],
                        receive.get("actor"),
                        source_id,
                        source.get("to"),
                    ),
                    event_id=receive["event_id"],
                )
            )

        if abs(receive["_start_t"] - source["_end_t"]) > tolerance:
            issues.append(
                issue(
                    ERROR,
                    "PASS_RECEIVE_TIME_MISMATCH",
                    (
                        "{} time={} 与 pass {} end_t={} 不一致"
                    ).format(
                        receive["event_id"],
                        receive["_start_t"],
                        source_id,
                        source["_end_t"],
                    ),
                    event_id=receive["event_id"],
                )
            )

        if source_id in pass_to_receive:
            issues.append(
                issue(
                    ERROR,
                    "MULTIPLE_RECEIVES_FOR_PASS",
                    "pass {} 对应多个 receive".format(source_id),
                    event_id=receive["event_id"],
                )
            )
        else:
            pass_to_receive[source_id] = receive

    if rules["require_pass_receive_pair"]:
        for event in events:
            if event["type"] == "pass" and event["event_id"] not in pass_to_receive:
                issues.append(
                    issue(
                        ERROR,
                        "PASS_WITHOUT_RECEIVE",
                        "{} 没有匹配 receive".format(event["event_id"]),
                        event_id=event["event_id"],
                    )
                )

    return pass_to_receive


def check_possession(
    events: List[Dict[str, Any]],
    initial_owner: str,
    rules: Dict[str, Any],
    pass_to_receive: Dict[str, Dict[str, Any]],
    issues: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    possession_events = [
        event
        for event in events
        if event["type"] in {"hold", "dribble", "pass", "receive", "shot"}
    ]

    possession_events.sort(
        key=lambda event: (
            event["_start_t"],
            POSSESSION_PRIORITY.get(event["type"], 99),
            event["event_id"],
        )
    )

    owner: Optional[str] = initial_owner
    in_transit: Optional[Dict[str, Any]] = None
    history: List[Dict[str, Any]] = [
        {
            "time_sec": 0.0,
            "state": "owned",
            "owner": owner,
            "event_id": None,
        }
    ]

    for event in possession_events:
        event_type = event["type"]
        event_time = event["_start_t"]

        if event_type in {"hold", "dribble"}:
            actor = event.get("actor")
            required = (
                rules["require_possession_for_dribble"]
                if event_type == "dribble"
                else True
            )

            if required and (in_transit is not None or owner != actor):
                issues.append(
                    issue(
                        ERROR,
                        "POSSESSION_REQUIRED",
                        (
                            "{} 的 actor={} 在 t={} 不拥有球；当前 owner={}"
                        ).format(
                            event["event_id"], actor, event_time, owner
                        ),
                        event_id=event["event_id"],
                        player_id=actor,
                        time_sec=event_time,
                    )
                )

        elif event_type == "pass":
            from_player = event.get("from")
            to_player = event.get("to")

            if (
                rules["require_possession_for_pass"]
                and (in_transit is not None or owner != from_player)
            ):
                issues.append(
                    issue(
                        ERROR,
                        "PASSER_DOES_NOT_OWN_BALL",
                        (
                            "{} 的 from={} 在 t={} 不拥有球；当前 owner={}"
                        ).format(
                            event["event_id"],
                            from_player,
                            event_time,
                            owner,
                        ),
                        event_id=event["event_id"],
                        player_id=from_player,
                        time_sec=event_time,
                    )
                )

            in_transit = {
                "pass_event_id": event["event_id"],
                "from": from_player,
                "to": to_player,
                "start_t": event["_start_t"],
                "end_t": event["_end_t"],
            }
            owner = None
            history.append(
                {
                    "time_sec": event["_start_t"],
                    "state": "in_transit",
                    "owner": None,
                    "from": from_player,
                    "to": to_player,
                    "event_id": event["event_id"],
                }
            )

        elif event_type == "receive":
            actor = event.get("actor")
            source_id = event.get("source_event")

            if in_transit is None:
                issues.append(
                    issue(
                        ERROR,
                        "RECEIVE_WITHOUT_TRANSIT",
                        "{} 在没有传球中的情况下接球".format(
                            event["event_id"]
                        ),
                        event_id=event["event_id"],
                        player_id=actor,
                        time_sec=event_time,
                    )
                )
            else:
                if in_transit["pass_event_id"] != source_id:
                    issues.append(
                        issue(
                            ERROR,
                            "RECEIVE_WRONG_TRANSIT",
                            (
                                "{} source_event={}，当前传球为 {}"
                            ).format(
                                event["event_id"],
                                source_id,
                                in_transit["pass_event_id"],
                            ),
                            event_id=event["event_id"],
                        )
                    )

                if in_transit["to"] != actor:
                    issues.append(
                        issue(
                            ERROR,
                            "RECEIVE_WRONG_PLAYER",
                            "{} actor={}，当前传球目标为 {}".format(
                                event["event_id"],
                                actor,
                                in_transit["to"],
                            ),
                            event_id=event["event_id"],
                            player_id=actor,
                        )
                    )

            owner = actor
            in_transit = None
            history.append(
                {
                    "time_sec": event_time,
                    "state": "owned",
                    "owner": actor,
                    "event_id": event["event_id"],
                }
            )

        elif event_type == "shot":
            actor = event.get("actor")
            if (
                rules["require_possession_for_shot"]
                and (in_transit is not None or owner != actor)
            ):
                issues.append(
                    issue(
                        ERROR,
                        "SHOOTER_DOES_NOT_OWN_BALL",
                        (
                            "{} 的 actor={} 在 t={} 不拥有球；当前 owner={}"
                        ).format(
                            event["event_id"], actor, event_time, owner
                        ),
                        event_id=event["event_id"],
                        player_id=actor,
                        time_sec=event_time,
                    )
                )

            owner = None
            in_transit = {
                "shot_event_id": event["event_id"],
                "from": actor,
                "to": None,
                "start_t": event["_start_t"],
                "end_t": event["_end_t"],
            }
            history.append(
                {
                    "time_sec": event["_start_t"],
                    "state": "shot",
                    "owner": None,
                    "from": actor,
                    "event_id": event["event_id"],
                }
            )

    if in_transit is not None and "pass_event_id" in in_transit:
        source_id = in_transit["pass_event_id"]
        if source_id not in pass_to_receive:
            issues.append(
                issue(
                    ERROR,
                    "UNRESOLVED_PASS_TRANSIT",
                    "回合结束时 pass {} 仍未接球".format(source_id),
                    event_id=source_id,
                )
            )

    return history


def check_explicit_motion_speeds(
    events: List[Dict[str, Any]],
    players: Dict[str, Dict[str, Any]],
    issues: List[Dict[str, Any]],
) -> None:
    """
    Conservative estimate using the last explicit target for each player.
    Defend-follow is excluded because its target is dynamic.
    """
    by_actor: Dict[str, List[Dict[str, Any]]] = {}

    for event in events:
        if event["type"] not in {"move", "dribble"}:
            continue
        actor = event.get("actor")
        if actor in players and event.get("_target_loc") is not None:
            by_actor.setdefault(actor, []).append(event)

    for actor, actor_events in by_actor.items():
        actor_events.sort(key=lambda event: (event["_start_t"], event["event_id"]))
        current_loc = players[actor]["start_loc"]
        current_time = 0.0

        for event in actor_events:
            # Holding position until event start is permitted.
            duration = event["_end_t"] - event["_start_t"]
            target = event["_target_loc"]
            speed = distance_xy(current_loc, target) / duration
            max_speed = players[actor]["max_speed_cm_s"]

            if speed > max_speed:
                issues.append(
                    issue(
                        ERROR,
                        "IMPLIED_PLAYER_SPEED_EXCEEDS_MAX",
                        (
                            "{} 事件 {} 的估算速度 {:.2f} cm/s 超过 {:.2f}"
                        ).format(
                            actor, event["event_id"], speed, max_speed
                        ),
                        event_id=event["event_id"],
                        player_id=actor,
                        time_sec=event["_start_t"],
                        details={"estimated_speed_cm_s": speed},
                    )
                )
            elif speed > max_speed * 0.8:
                issues.append(
                    issue(
                        WARNING,
                        "IMPLIED_PLAYER_SPEED_HIGH",
                        (
                            "{} 事件 {} 的估算速度 {:.2f} cm/s 接近上限"
                        ).format(actor, event["event_id"], speed),
                        event_id=event["event_id"],
                        player_id=actor,
                        time_sec=event["_start_t"],
                    )
                )

            current_loc = target
            current_time = event["_end_t"]


def resolve_project_root(
    config: Dict[str, Any],
    config_path: Path,
) -> Path:
    raw = config.get("project_root")
    if isinstance(raw, str) and raw.strip():
        path = Path(raw)
        if not path.is_absolute():
            path = (config_path.parent / path).resolve()
        return path

    # Expected: Content/FutsalMOT/code/configs/events/file.json
    parents = config_path.resolve().parents
    if len(parents) >= 6:
        return parents[5]
    raise ConfigError("缺少 project_root，且无法从路径推断")


def build_report(
    config_path: Path,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    issues: List[Dict[str, Any]] = []

    episode_id = require_str(config.get("episode_id"), "episode_id")
    output_seq_id = require_str(config.get("output_seq_id"), "output_seq_id")
    timeline = parse_timeline(config)
    court = parse_court(config)
    rules = parse_rules(config)

    check_timeline_consistency(timeline, issues)

    players = parse_players(config, court, issues)
    ball = parse_ball(config, players, court, issues)
    roster_summary = check_roster_and_initial_spacing(
        config, players, ball, rules, issues
    )
    events = parse_events(
        config, timeline, court, players, rules, issues
    )

    check_actor_overlaps(events, rules, issues)
    pass_to_receive = check_pass_receive_pairs(events, rules, issues)
    possession_history = check_possession(
        events,
        ball["initial_owner"],
        rules,
        pass_to_receive,
        issues,
    )
    check_explicit_motion_speeds(events, players, issues)

    event_summaries = []
    for event in sorted(
        events,
        key=lambda item: (
            item["_start_t"],
            item["_end_t"],
            item["event_id"],
        ),
    ):
        event_related = [
            item
            for item in issues
            if item.get("event_id") == event["event_id"]
        ]
        event_status = max_level(
            item["level"] for item in event_related
        )

        event_summaries.append(
            {
                "event_id": event["event_id"],
                "type": event["type"],
                "actor": get_event_actor(event),
                "start_t": event["_start_t"],
                "end_t": event["_end_t"],
                "instantaneous": event["_instantaneous"],
                "start_frame": event["_start_frame"],
                "end_frame_exclusive": event["_end_frame_exclusive"],
                "last_frame": event["_last_frame"],
                "status": event_status,
                "issue_count": len(event_related),
            }
        )

    error_count = sum(1 for item in issues if item["level"] == ERROR)
    warning_count = sum(
        1 for item in issues if item["level"] == WARNING
    )
    status = (
        ERROR
        if error_count
        else WARNING
        if warning_count
        else PASS
    )

    return {
        "validator": {
            "name": "FutsalMOT episode-event configuration validator",
            "version": SCRIPT_VERSION,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        "config_path": str(config_path.resolve()).replace("\\", "/"),
        "episode_id": episode_id,
        "output_seq_id": output_seq_id,
        "summary": {
            "status": status,
            "player_count": len(players),
            "event_count": len(events),
            "warning_count": warning_count,
            "error_count": error_count,
        },
        "timeline": timeline,
        "court": court,
        "roster": roster_summary,
        "players": {
            player_id: {
                **player,
                "start_loc": list(player["start_loc"]),
            }
            for player_id, player in players.items()
        },
        "ball": {
            **ball,
            "initial_loc": list(ball["initial_loc"]),
        },
        "events": event_summaries,
        "pass_receive_pairs": {
            pass_id: receive["event_id"]
            for pass_id, receive in pass_to_receive.items()
        },
        "possession_history": possession_history,
        "issues": issues,
    }


def write_report(
    report: Dict[str, Any],
    output_dir: Path,
) -> Tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    episode_id = report["episode_id"]
    json_path = output_dir / "episode_report_{}.json".format(
        episode_id
    )
    csv_path = output_dir / "episode_timeline_{}.csv".format(
        episode_id
    )

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(
            report,
            f,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )

    columns = [
        "event_id",
        "type",
        "actor",
        "start_t",
        "end_t",
        "instantaneous",
        "start_frame",
        "end_frame_exclusive",
        "last_frame",
        "status",
        "issue_count",
    ]

    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for event in report["events"]:
            writer.writerow(event)

    return json_path, csv_path


def print_summary(
    report: Dict[str, Any],
    json_path: Path,
    csv_path: Path,
) -> None:
    print("=" * 76)
    print("FutsalMOT episode validator")
    print("VERSION =", SCRIPT_VERSION)
    print("EPISODE_ID =", report["episode_id"])
    print("OUTPUT_SEQ_ID =", report["output_seq_id"])
    print(
        "TIMELINE = {} s @ {} fps, frames {}..{}".format(
            report["timeline"]["duration_sec"],
            report["timeline"]["fps"],
            report["timeline"]["frame_start"],
            report["timeline"]["frame_end"],
        )
    )
    print(
        "PLAYERS={} EVENTS={}".format(
            report["summary"]["player_count"],
            report["summary"]["event_count"],
        )
    )
    print("-" * 76)

    for event in report["events"]:
        print(
            "[{}] {:<10} {:<14} t={:.3f}..{:.3f} frames={}..{}".format(
                event["status"],
                event["event_id"],
                event["type"],
                event["start_t"],
                event["end_t"],
                event["start_frame"],
                event["last_frame"],
            )
        )

    if report["issues"]:
        print("-" * 76)
        for item in report["issues"]:
            location = ""
            if item.get("event_id"):
                location += " event={}".format(item["event_id"])
            if item.get("player_id"):
                location += " player={}".format(item["player_id"])
            print(
                "[{}] {}{}: {}".format(
                    item["level"],
                    item["code"],
                    location,
                    item["message"],
                )
            )

    print("-" * 76)
    print(
        "STATUS={} warnings={} errors={}".format(
            report["summary"]["status"],
            report["summary"]["warning_count"],
            report["summary"]["error_count"],
        )
    )
    print("JSON report:", json_path)
    print("CSV timeline:", csv_path)
    print("=" * 76)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate FutsalMOT episode event configuration."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Episode event JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Default: <project_root>/Saved/FutsalMOT/episode_reports"
        ),
    )
    parser.add_argument(
        "--strict-warnings",
        action="store_true",
        help="Return exit code 1 when WARNING exists.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config.expanduser().resolve()

    try:
        config = load_json(config_path)
        report = build_report(config_path, config)
        project_root = resolve_project_root(config, config_path)

        output_dir = (
            args.output_dir.expanduser().resolve()
            if args.output_dir is not None
            else project_root
            / "Saved"
            / "FutsalMOT"
            / "episode_reports"
        )

        json_path, csv_path = write_report(report, output_dir)
        print_summary(report, json_path, csv_path)

        if report["summary"]["error_count"] > 0:
            return 1
        if (
            args.strict_warnings
            and report["summary"]["warning_count"] > 0
        ):
            return 1
        return 0

    except ConfigError as exc:
        print("=" * 76, file=sys.stderr)
        print("VALIDATION ABORTED", file=sys.stderr)
        print("[CONFIG ERROR]", exc, file=sys.stderr)
        print("CONFIG_PATH =", config_path, file=sys.stderr)
        print("=" * 76, file=sys.stderr)
        return 2

    except Exception as exc:
        print("=" * 76, file=sys.stderr)
        print("VALIDATION ABORTED", file=sys.stderr)
        print(
            "[UNEXPECTED ERROR] {}: {}".format(
                type(exc).__name__, exc
            ),
            file=sys.stderr,
        )
        print("CONFIG_PATH =", config_path, file=sys.stderr)
        print("=" * 76, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
