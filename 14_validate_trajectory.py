#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FutsalMOT trajectory configuration validator.

Purpose
-------
Validate a FutsalMOT sequence JSON before it is imported into Unreal Engine.

The script checks:
1. Timeline and object configuration integrity.
2. Multi-keyframe order, coverage, duplicate frames, and valid coordinates.
3. Segment distance, duration, speed, speed jumps, and turn angles.
4. Court-boundary violations.
5. Player-player minimum distance over the full sequence.
6. Duplicate track IDs and class/track-map inconsistencies.
7. Basic vertical-motion anomalies.

Outputs
-------
Saved/FutsalMOT/trajectory_reports/
├─ trajectory_report_<seq_id>.json
└─ trajectory_segments_<seq_id>.csv

Exit codes
----------
0: no ERROR issues
1: one or more ERROR issues, or warnings treated as errors
2: configuration/file failure

Typical usage
-------------
py 14_validate_trajectory.py

py 14_validate_trajectory.py ^
  --config "D:/projects/FustalMOT_UEDataset/Content/FutsalMOT/code/configs/seq_test_0004.json"

py 14_validate_trajectory.py ^
  --config ".../seq_test_0004.json" ^
  --strict-warnings
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


SCRIPT_VERSION = "A2_5A_TRAJECTORY_VALIDATOR_8P_MOVEMENT_V2"

DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parent
    / "configs"
    / "seq_test_0004.json"
)

LEVEL_PASS = "PASS"
LEVEL_WARNING = "WARNING"
LEVEL_ERROR = "ERROR"

LEVEL_RANK = {
    LEVEL_PASS: 0,
    LEVEL_WARNING: 1,
    LEVEL_ERROR: 2,
}

SUPPORTED_CATEGORIES = {"player", "ball"}
SUPPORTED_INTERPOLATIONS = {"linear"}

DEFAULT_VALIDATION: Dict[str, Any] = {
    # Court safety region. The physical court is approximately
    # X ±2000 cm and Y ±1000 cm; these defaults preserve a 50 cm margin.
    "court_x_min_cm": -1950.0,
    "court_x_max_cm": 1950.0,
    "court_y_min_cm": -950.0,
    "court_y_max_cm": 950.0,
    "boundary_tolerance_cm": 1.0,

    # Horizontal speed thresholds.
    "player_warning_speed_cm_s": 500.0,
    "player_max_speed_cm_s": 750.0,
    "ball_warning_speed_cm_s": 1800.0,
    "ball_max_speed_cm_s": 3000.0,

    # Consecutive-segment speed discontinuity thresholds.
    "max_player_speed_jump_cm_s": 250.0,
    "max_player_speed_jump_error_cm_s": 500.0,
    "max_ball_speed_jump_cm_s": 800.0,
    "max_ball_speed_jump_error_cm_s": 1600.0,

    # Direction-change thresholds at an interior keyframe.
    "max_turn_angle_deg": 100.0,
    "max_turn_angle_error_deg": 150.0,

    # Pairwise player proximity.
    "minimum_player_distance_cm": 50.0,
    "minimum_player_distance_error_cm": 25.0,

    # Vertical-motion checks.
    "player_vertical_warning_speed_cm_s": 20.0,
    "player_vertical_max_speed_cm_s": 80.0,
    "ball_vertical_warning_speed_cm_s": 800.0,
    "ball_vertical_max_speed_cm_s": 1800.0,

    # Full-sequence sampling. 1 means every frame.
    "sample_stride_frames": 1,

    # Require first/last keyframes to cover the full timeline.
    "require_full_timeline_coverage": True,
}


class ConfigError(RuntimeError):
    """Fatal configuration error that prevents useful validation."""


def is_number(value: Any) -> bool:
    """True for finite int/float values, excluding booleans."""
    if isinstance(value, bool):
        return False
    if not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value))


def safe_float(value: Any, field_name: str) -> float:
    if not is_number(value):
        raise ConfigError(
            "字段 '{}' 必须是有限数值，当前值={!r}".format(field_name, value)
        )
    return float(value)


def safe_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise ConfigError(
            "字段 '{}' 必须是整数，当前值={!r}".format(field_name, value)
        )

    if isinstance(value, int):
        return value

    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return int(value)

    raise ConfigError(
        "字段 '{}' 必须是整数，当前值={!r}".format(field_name, value)
    )


def parse_vec3(value: Any, field_name: str) -> Tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ConfigError(
            "字段 '{}' 必须是长度为 3 的数组 [x, y, z]".format(field_name)
        )

    return (
        safe_float(value[0], field_name + "[0]"),
        safe_float(value[1], field_name + "[1]"),
        safe_float(value[2], field_name + "[2]"),
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
                exc.msg,
                exc.lineno,
                exc.colno,
            )
        ) from exc
    except OSError as exc:
        raise ConfigError("无法读取配置文件：{}；{}".format(path, exc)) from exc

    if not isinstance(data, dict):
        raise ConfigError("配置文件顶层必须是 JSON object")

    return data


def merge_validation_settings(config: Dict[str, Any]) -> Dict[str, Any]:
    settings = dict(DEFAULT_VALIDATION)
    raw = config.get("trajectory_validation", {})

    if raw is None:
        raw = {}

    if not isinstance(raw, dict):
        raise ConfigError("trajectory_validation 必须是 JSON object")

    for key, value in raw.items():
        if key not in settings:
            # Preserve unknown settings in the report but do not use them.
            settings[key] = value
        else:
            settings[key] = value

    numeric_keys = [
        "court_x_min_cm",
        "court_x_max_cm",
        "court_y_min_cm",
        "court_y_max_cm",
        "boundary_tolerance_cm",
        "player_warning_speed_cm_s",
        "player_max_speed_cm_s",
        "ball_warning_speed_cm_s",
        "ball_max_speed_cm_s",
        "max_player_speed_jump_cm_s",
        "max_player_speed_jump_error_cm_s",
        "max_ball_speed_jump_cm_s",
        "max_ball_speed_jump_error_cm_s",
        "max_turn_angle_deg",
        "max_turn_angle_error_deg",
        "minimum_player_distance_cm",
        "minimum_player_distance_error_cm",
        "player_vertical_warning_speed_cm_s",
        "player_vertical_max_speed_cm_s",
        "ball_vertical_warning_speed_cm_s",
        "ball_vertical_max_speed_cm_s",
    ]

    for key in numeric_keys:
        settings[key] = safe_float(
            settings[key],
            "trajectory_validation." + key,
        )

    settings["sample_stride_frames"] = safe_int(
        settings["sample_stride_frames"],
        "trajectory_validation.sample_stride_frames",
    )

    if settings["sample_stride_frames"] <= 0:
        raise ConfigError(
            "trajectory_validation.sample_stride_frames 必须大于 0"
        )

    settings["require_full_timeline_coverage"] = bool(
        settings["require_full_timeline_coverage"]
    )

    if settings["court_x_min_cm"] >= settings["court_x_max_cm"]:
        raise ConfigError("court_x_min_cm 必须小于 court_x_max_cm")

    if settings["court_y_min_cm"] >= settings["court_y_max_cm"]:
        raise ConfigError("court_y_min_cm 必须小于 court_y_max_cm")

    if (
        settings["player_warning_speed_cm_s"]
        > settings["player_max_speed_cm_s"]
    ):
        raise ConfigError(
            "player_warning_speed_cm_s 不能大于 player_max_speed_cm_s"
        )

    if (
        settings["ball_warning_speed_cm_s"]
        > settings["ball_max_speed_cm_s"]
    ):
        raise ConfigError(
            "ball_warning_speed_cm_s 不能大于 ball_max_speed_cm_s"
        )

    if (
        settings["minimum_player_distance_error_cm"]
        > settings["minimum_player_distance_cm"]
    ):
        raise ConfigError(
            "minimum_player_distance_error_cm "
            "不能大于 minimum_player_distance_cm"
        )

    return settings


def make_issue(
    level: str,
    code: str,
    message: str,
    *,
    object_id: Optional[str] = None,
    category: Optional[str] = None,
    frame: Optional[int] = None,
    segment_index: Optional[int] = None,
    details: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    issue: Dict[str, Any] = {
        "level": level,
        "code": code,
        "message": message,
    }

    if object_id is not None:
        issue["object_id"] = object_id
    if category is not None:
        issue["category"] = category
    if frame is not None:
        issue["frame"] = int(frame)
    if segment_index is not None:
        issue["segment_index"] = int(segment_index)
    if details:
        issue["details"] = details

    return issue


def max_level(levels: Iterable[str]) -> str:
    result = LEVEL_PASS
    for level in levels:
        if LEVEL_RANK.get(level, -1) > LEVEL_RANK[result]:
            result = level
    return result


def normalize_angle_deg(angle: float) -> float:
    while angle > 180.0:
        angle -= 360.0
    while angle <= -180.0:
        angle += 360.0
    return angle


def angle_difference_deg(a: float, b: float) -> float:
    """Smallest absolute difference between two angles in degrees."""
    return abs(normalize_angle_deg(a - b))


def distance_xy(
    a: Sequence[float],
    b: Sequence[float],
) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def distance_3d(
    a: Sequence[float],
    b: Sequence[float],
) -> float:
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    dz = b[2] - a[2]
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def direction_deg(
    a: Sequence[float],
    b: Sequence[float],
) -> Optional[float]:
    dx = b[0] - a[0]
    dy = b[1] - a[1]

    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return None

    return normalize_angle_deg(math.degrees(math.atan2(dy, dx)))


def point_in_court(
    loc: Sequence[float],
    settings: Dict[str, Any],
) -> bool:
    tol = settings["boundary_tolerance_cm"]

    return (
        settings["court_x_min_cm"] - tol
        <= loc[0]
        <= settings["court_x_max_cm"] + tol
        and settings["court_y_min_cm"] - tol
        <= loc[1]
        <= settings["court_y_max_cm"] + tol
    )


def interpolate_keyframes(
    keyframes: Sequence[Dict[str, Any]],
    frame: int,
) -> Tuple[float, float, float]:
    if not keyframes:
        raise ValueError("keyframes 为空")

    if frame <= keyframes[0]["frame"]:
        return tuple(keyframes[0]["loc"])

    if frame >= keyframes[-1]["frame"]:
        return tuple(keyframes[-1]["loc"])

    for left, right in zip(keyframes, keyframes[1:]):
        f0 = left["frame"]
        f1 = right["frame"]

        if f0 <= frame <= f1:
            if f1 == f0:
                return tuple(left["loc"])

            alpha = float(frame - f0) / float(f1 - f0)
            p0 = left["loc"]
            p1 = right["loc"]

            return (
                p0[0] + (p1[0] - p0[0]) * alpha,
                p0[1] + (p1[1] - p0[1]) * alpha,
                p0[2] + (p1[2] - p0[2]) * alpha,
            )

    return tuple(keyframes[-1]["loc"])


def parse_keyframes(
    object_id: str,
    raw_object: Dict[str, Any],
    frame_start: int,
    frame_end: int,
    issues: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    raw_keyframes = raw_object.get("keyframes")

    if raw_keyframes is None:
        # Legacy A1 fallback.
        if "start" not in raw_object or "end" not in raw_object:
            issues.append(
                make_issue(
                    LEVEL_ERROR,
                    "MISSING_TRAJECTORY",
                    "对象既没有 keyframes，也没有完整的 start/end",
                    object_id=object_id,
                )
            )
            return []

        try:
            start_loc = parse_vec3(
                raw_object["start"],
                "objects.{}.start".format(object_id),
            )
            end_loc = parse_vec3(
                raw_object["end"],
                "objects.{}.end".format(object_id),
            )
        except ConfigError as exc:
            issues.append(
                make_issue(
                    LEVEL_ERROR,
                    "INVALID_LEGACY_TRAJECTORY",
                    str(exc),
                    object_id=object_id,
                )
            )
            return []

        issues.append(
            make_issue(
                LEVEL_WARNING,
                "LEGACY_START_END",
                "对象仍使用旧 start/end 格式；建议升级为 keyframes",
                object_id=object_id,
            )
        )

        return [
            {"frame": frame_start, "loc": start_loc},
            {"frame": frame_end, "loc": end_loc},
        ]

    if not isinstance(raw_keyframes, list):
        issues.append(
            make_issue(
                LEVEL_ERROR,
                "INVALID_KEYFRAMES_TYPE",
                "keyframes 必须是数组",
                object_id=object_id,
            )
        )
        return []

    parsed_in_original_order: List[Dict[str, Any]] = []

    for index, raw_keyframe in enumerate(raw_keyframes):
        field_prefix = "objects.{}.keyframes[{}]".format(
            object_id,
            index,
        )

        if not isinstance(raw_keyframe, dict):
            issues.append(
                make_issue(
                    LEVEL_ERROR,
                    "INVALID_KEYFRAME_TYPE",
                    "{} 必须是 JSON object".format(field_prefix),
                    object_id=object_id,
                )
            )
            continue

        try:
            frame = safe_int(
                raw_keyframe.get("frame"),
                field_prefix + ".frame",
            )
            loc = parse_vec3(
                raw_keyframe.get("loc"),
                field_prefix + ".loc",
            )
        except ConfigError as exc:
            issues.append(
                make_issue(
                    LEVEL_ERROR,
                    "INVALID_KEYFRAME",
                    str(exc),
                    object_id=object_id,
                )
            )
            continue

        parsed_in_original_order.append({
            "frame": frame,
            "loc": loc,
        })

    if len(parsed_in_original_order) < 2:
        issues.append(
            make_issue(
                LEVEL_ERROR,
                "TOO_FEW_KEYFRAMES",
                "对象至少需要 2 个有效关键帧",
                object_id=object_id,
                details={"valid_keyframes": len(parsed_in_original_order)},
            )
        )
        return parsed_in_original_order

    previous_frame: Optional[int] = None
    seen_frames = set()

    for keyframe in parsed_in_original_order:
        frame = keyframe["frame"]

        if frame in seen_frames:
            issues.append(
                make_issue(
                    LEVEL_ERROR,
                    "DUPLICATE_KEYFRAME_FRAME",
                    "存在重复关键帧 frame={}".format(frame),
                    object_id=object_id,
                    frame=frame,
                )
            )
        seen_frames.add(frame)

        if previous_frame is not None and frame <= previous_frame:
            issues.append(
                make_issue(
                    LEVEL_ERROR,
                    "NON_INCREASING_KEYFRAMES",
                    "关键帧必须按 frame 严格递增；{} 后出现 {}".format(
                        previous_frame,
                        frame,
                    ),
                    object_id=object_id,
                    frame=frame,
                )
            )

        previous_frame = frame

        if frame < frame_start or frame > frame_end:
            issues.append(
                make_issue(
                    LEVEL_ERROR,
                    "KEYFRAME_OUTSIDE_TIMELINE",
                    "关键帧 frame={} 超出序列范围 {}..{}".format(
                        frame,
                        frame_start,
                        frame_end,
                    ),
                    object_id=object_id,
                    frame=frame,
                )
            )

    # Continue analysis in sorted order, but preserve the errors above.
    sorted_keyframes = sorted(
        parsed_in_original_order,
        key=lambda item: item["frame"],
    )

    # Deduplicate for safe downstream calculations.
    unique_keyframes: List[Dict[str, Any]] = []
    used_frames = set()

    for keyframe in sorted_keyframes:
        if keyframe["frame"] in used_frames:
            continue
        used_frames.add(keyframe["frame"])
        unique_keyframes.append(keyframe)

    return unique_keyframes


def severity_for_speed(
    category: str,
    speed_cm_s: float,
    settings: Dict[str, Any],
) -> Tuple[str, Optional[str]]:
    if category == "player":
        warning_threshold = settings["player_warning_speed_cm_s"]
        error_threshold = settings["player_max_speed_cm_s"]
    else:
        warning_threshold = settings["ball_warning_speed_cm_s"]
        error_threshold = settings["ball_max_speed_cm_s"]

    if speed_cm_s > error_threshold:
        return LEVEL_ERROR, "SPEED_EXCEEDS_MAX"
    if speed_cm_s > warning_threshold:
        return LEVEL_WARNING, "SPEED_ABOVE_WARNING"
    return LEVEL_PASS, None


def severity_for_speed_jump(
    category: str,
    speed_jump_cm_s: float,
    settings: Dict[str, Any],
) -> Tuple[str, Optional[str]]:
    if category == "player":
        warning_threshold = settings["max_player_speed_jump_cm_s"]
        error_threshold = settings["max_player_speed_jump_error_cm_s"]
    else:
        warning_threshold = settings["max_ball_speed_jump_cm_s"]
        error_threshold = settings["max_ball_speed_jump_error_cm_s"]

    if speed_jump_cm_s > error_threshold:
        return LEVEL_ERROR, "SPEED_JUMP_EXCEEDS_MAX"
    if speed_jump_cm_s > warning_threshold:
        return LEVEL_WARNING, "SPEED_JUMP_ABOVE_WARNING"
    return LEVEL_PASS, None


def severity_for_vertical_speed(
    category: str,
    speed_cm_s: float,
    settings: Dict[str, Any],
) -> Tuple[str, Optional[str]]:
    absolute_speed = abs(speed_cm_s)

    if category == "player":
        warning_threshold = settings["player_vertical_warning_speed_cm_s"]
        error_threshold = settings["player_vertical_max_speed_cm_s"]
    else:
        warning_threshold = settings["ball_vertical_warning_speed_cm_s"]
        error_threshold = settings["ball_vertical_max_speed_cm_s"]

    if absolute_speed > error_threshold:
        return LEVEL_ERROR, "VERTICAL_SPEED_EXCEEDS_MAX"
    if absolute_speed > warning_threshold:
        return LEVEL_WARNING, "VERTICAL_SPEED_ABOVE_WARNING"
    return LEVEL_PASS, None


def analyze_object(
    *,
    object_id: str,
    raw_object: Dict[str, Any],
    keyframes: List[Dict[str, Any]],
    category: str,
    track_id: Optional[int],
    class_id: Optional[int],
    frame_start: int,
    frame_end: int,
    fps: float,
    settings: Dict[str, Any],
    issues: List[Dict[str, Any]],
    ball_contact_frames: Optional[set[int]] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    object_issues_start = len(issues)

    interpolation = str(raw_object.get("interpolation", "linear")).lower()

    if interpolation not in SUPPORTED_INTERPOLATIONS:
        issues.append(
            make_issue(
                LEVEL_WARNING,
                "UNSUPPORTED_INTERPOLATION_FOR_VALIDATION",
                (
                    "当前验证器按 linear 轨迹分析，但配置 interpolation={!r}"
                ).format(interpolation),
                object_id=object_id,
                category=category,
            )
        )

    if len(keyframes) >= 1 and settings["require_full_timeline_coverage"]:
        if keyframes[0]["frame"] != frame_start:
            issues.append(
                make_issue(
                    LEVEL_ERROR,
                    "MISSING_START_COVERAGE",
                    "首个关键帧必须位于 frame_start={}".format(
                        frame_start
                    ),
                    object_id=object_id,
                    category=category,
                    frame=keyframes[0]["frame"],
                )
            )

        if keyframes[-1]["frame"] != frame_end:
            issues.append(
                make_issue(
                    LEVEL_ERROR,
                    "MISSING_END_COVERAGE",
                    "末个关键帧必须位于 frame_end={}".format(
                        frame_end
                    ),
                    object_id=object_id,
                    category=category,
                    frame=keyframes[-1]["frame"],
                )
            )

    for keyframe in keyframes:
        if not point_in_court(keyframe["loc"], settings):
            issues.append(
                make_issue(
                    LEVEL_ERROR,
                    "KEYFRAME_OUTSIDE_COURT",
                    (
                        "关键帧超出安全球场边界：frame={} loc={}"
                    ).format(
                        keyframe["frame"],
                        [round(v, 3) for v in keyframe["loc"]],
                    ),
                    object_id=object_id,
                    category=category,
                    frame=keyframe["frame"],
                    details={
                        "loc_cm": list(keyframe["loc"]),
                    },
                )
            )

    segments: List[Dict[str, Any]] = []
    previous_speed: Optional[float] = None
    previous_direction: Optional[float] = None

    total_distance_xy = 0.0
    total_distance_3d = 0.0
    max_speed = 0.0
    max_vertical_speed = 0.0

    for segment_index, (left, right) in enumerate(
        zip(keyframes, keyframes[1:])
    ):
        start_frame = left["frame"]
        end_frame = right["frame"]
        duration_frames = end_frame - start_frame

        segment_issue_codes: List[str] = []
        segment_levels: List[str] = [LEVEL_PASS]

        if duration_frames <= 0:
            issues.append(
                make_issue(
                    LEVEL_ERROR,
                    "INVALID_SEGMENT_DURATION",
                    "轨迹段时长必须大于 0",
                    object_id=object_id,
                    category=category,
                    frame=start_frame,
                    segment_index=segment_index,
                    details={
                        "start_frame": start_frame,
                        "end_frame": end_frame,
                    },
                )
            )
            duration_seconds = 0.0
        else:
            duration_seconds = duration_frames / fps

        start_loc = left["loc"]
        end_loc = right["loc"]

        dist_xy = distance_xy(start_loc, end_loc)
        dist_3d = distance_3d(start_loc, end_loc)
        delta_z = end_loc[2] - start_loc[2]
        direction = direction_deg(start_loc, end_loc)

        if duration_seconds > 0.0:
            speed_xy = dist_xy / duration_seconds
            speed_3d = dist_3d / duration_seconds
            vertical_speed = delta_z / duration_seconds
        else:
            speed_xy = float("inf")
            speed_3d = float("inf")
            vertical_speed = float("inf")

        speed_level, speed_code = severity_for_speed(
            category,
            speed_xy,
            settings,
        )

        if speed_code is not None:
            segment_issue_codes.append(speed_code)
            segment_levels.append(speed_level)

            threshold = (
                settings["player_max_speed_cm_s"]
                if category == "player"
                else settings["ball_max_speed_cm_s"]
            )

            issues.append(
                make_issue(
                    speed_level,
                    speed_code,
                    (
                        "{} frame {}→{} 平均速度 {:.2f} cm/s"
                    ).format(
                        object_id,
                        start_frame,
                        end_frame,
                        speed_xy,
                    ),
                    object_id=object_id,
                    category=category,
                    frame=end_frame,
                    segment_index=segment_index,
                    details={
                        "speed_xy_cm_s": speed_xy,
                        "maximum_reference_cm_s": threshold,
                    },
                )
            )

        vertical_level, vertical_code = severity_for_vertical_speed(
            category,
            vertical_speed,
            settings,
        )

        if vertical_code is not None:
            segment_issue_codes.append(vertical_code)
            segment_levels.append(vertical_level)

            issues.append(
                make_issue(
                    vertical_level,
                    vertical_code,
                    (
                        "{} frame {}→{} 垂直速度 {:.2f} cm/s"
                    ).format(
                        object_id,
                        start_frame,
                        end_frame,
                        vertical_speed,
                    ),
                    object_id=object_id,
                    category=category,
                    frame=end_frame,
                    segment_index=segment_index,
                    details={
                        "vertical_speed_cm_s": vertical_speed,
                        "delta_z_cm": delta_z,
                    },
                )
            )

        speed_jump: Optional[float] = None

        if previous_speed is not None:
            speed_jump = abs(speed_xy - previous_speed)
            jump_level, jump_code = severity_for_speed_jump(
                category,
                speed_jump,
                settings,
            )

            if jump_code is not None:
                segment_issue_codes.append(jump_code)
                segment_levels.append(jump_level)

                issues.append(
                    make_issue(
                        jump_level,
                        jump_code,
                        (
                            "{} 在 frame={} 的相邻轨迹段速度跳变 "
                            "{:.2f} cm/s"
                        ).format(
                            object_id,
                            start_frame,
                            speed_jump,
                        ),
                        object_id=object_id,
                        category=category,
                        frame=start_frame,
                        segment_index=segment_index,
                        details={
                            "previous_speed_cm_s": previous_speed,
                            "current_speed_cm_s": speed_xy,
                            "speed_jump_cm_s": speed_jump,
                        },
                    )
                )

        turn_angle: Optional[float] = None

        if (
            previous_direction is not None
            and direction is not None
        ):
            turn_angle = angle_difference_deg(
                direction,
                previous_direction,
            )

            is_ball_contact_turn = (
                category == "ball"
                and ball_contact_frames is not None
                and start_frame in ball_contact_frames
            )
            if turn_angle > settings["max_turn_angle_error_deg"]:
                if is_ball_contact_turn:
                    # A pass/receive/shot contact can legitimately redirect the
                    # ball sharply. Preserve it as an auditable warning rather
                    # than rejecting an otherwise valid episode.
                    turn_level = LEVEL_WARNING
                    turn_code = "BALL_CONTACT_TURN_ABOVE_WARNING"
                else:
                    turn_level = LEVEL_ERROR
                    turn_code = "TURN_ANGLE_EXCEEDS_MAX"
            elif turn_angle > settings["max_turn_angle_deg"]:
                turn_level = LEVEL_WARNING
                turn_code = (
                    "BALL_CONTACT_TURN_ABOVE_WARNING"
                    if is_ball_contact_turn
                    else "TURN_ANGLE_ABOVE_WARNING"
                )
            else:
                turn_level = LEVEL_PASS
                turn_code = None

            if turn_code is not None:
                segment_issue_codes.append(turn_code)
                segment_levels.append(turn_level)

                issues.append(
                    make_issue(
                        turn_level,
                        turn_code,
                        (
                            "{} 在 frame={} 转向 {:.2f}°"
                        ).format(
                            object_id,
                            start_frame,
                            turn_angle,
                        ),
                        object_id=object_id,
                        category=category,
                        frame=start_frame,
                        segment_index=segment_index,
                        details={
                            "previous_direction_deg": previous_direction,
                            "current_direction_deg": direction,
                            "turn_angle_deg": turn_angle,
                            "contact_frame_exception": is_ball_contact_turn,
                        },
                    )
                )

        segment_in_court = (
            point_in_court(start_loc, settings)
            and point_in_court(end_loc, settings)
        )

        if not segment_in_court:
            segment_issue_codes.append("SEGMENT_OUTSIDE_COURT")
            segment_levels.append(LEVEL_ERROR)

        segment_level = max_level(segment_levels)

        segment = {
            "seq_id": None,  # populated before output
            "object_id": object_id,
            "category": category,
            "track_id": track_id,
            "class_id": class_id,
            "segment_index": segment_index,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "duration_frames": duration_frames,
            "duration_s": duration_seconds,
            "start_x_cm": start_loc[0],
            "start_y_cm": start_loc[1],
            "start_z_cm": start_loc[2],
            "end_x_cm": end_loc[0],
            "end_y_cm": end_loc[1],
            "end_z_cm": end_loc[2],
            "distance_xy_cm": dist_xy,
            "distance_3d_cm": dist_3d,
            "delta_z_cm": delta_z,
            "speed_xy_cm_s": speed_xy,
            "speed_3d_cm_s": speed_3d,
            "vertical_speed_cm_s": vertical_speed,
            "direction_deg": direction,
            "turn_from_previous_deg": turn_angle,
            "speed_jump_from_previous_cm_s": speed_jump,
            "within_court": segment_in_court,
            "level": segment_level,
            "issue_codes": segment_issue_codes,
        }

        segments.append(segment)

        total_distance_xy += dist_xy
        total_distance_3d += dist_3d
        max_speed = max(max_speed, speed_xy)
        max_vertical_speed = max(
            max_vertical_speed,
            abs(vertical_speed),
        )

        previous_speed = speed_xy
        if direction is not None:
            previous_direction = direction

    if keyframes:
        xs = [keyframe["loc"][0] for keyframe in keyframes]
        ys = [keyframe["loc"][1] for keyframe in keyframes]
        zs = [keyframe["loc"][2] for keyframe in keyframes]
    else:
        xs = []
        ys = []
        zs = []

    object_issue_slice = issues[object_issues_start:]
    object_status = max_level(
        issue["level"] for issue in object_issue_slice
    )

    object_summary = {
        "object_id": object_id,
        "category": category,
        "track_id": track_id,
        "class_id": class_id,
        "interpolation": interpolation,
        "keyframe_count": len(keyframes),
        "segment_count": len(segments),
        "timeline_covered": bool(
            keyframes
            and keyframes[0]["frame"] == frame_start
            and keyframes[-1]["frame"] == frame_end
        ),
        "total_distance_xy_cm": total_distance_xy,
        "total_distance_3d_cm": total_distance_3d,
        "max_speed_xy_cm_s": max_speed,
        "max_vertical_speed_cm_s": max_vertical_speed,
        "bounds_cm": {
            "x_min": min(xs) if xs else None,
            "x_max": max(xs) if xs else None,
            "y_min": min(ys) if ys else None,
            "y_max": max(ys) if ys else None,
            "z_min": min(zs) if zs else None,
            "z_max": max(zs) if zs else None,
        },
        "status": object_status,
        "issue_count": len(object_issue_slice),
        "keyframes": [
            {
                "frame": keyframe["frame"],
                "loc_cm": list(keyframe["loc"]),
            }
            for keyframe in keyframes
        ],
    }

    return object_summary, segments


def analyze_player_proximity(
    *,
    players: Dict[str, List[Dict[str, Any]]],
    frame_start: int,
    frame_end: int,
    settings: Dict[str, Any],
    issues: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    player_ids = sorted(players.keys())
    stride = settings["sample_stride_frames"]
    pair_stats: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for i, left_id in enumerate(player_ids):
        for right_id in player_ids[i + 1:]:
            pair_stats[(left_id, right_id)] = {
                "player_a": left_id,
                "player_b": right_id,
                "minimum_distance_cm": float("inf"),
                "minimum_distance_frame": None,
                "warning_sample_count": 0,
                "error_sample_count": 0,
                "sample_count": 0,
            }

    frames = list(range(frame_start, frame_end + 1, stride))
    if not frames or frames[-1] != frame_end:
        frames.append(frame_end)

    for frame in frames:
        positions = {
            player_id: interpolate_keyframes(keyframes, frame)
            for player_id, keyframes in players.items()
            if keyframes
        }

        for pair, stat in pair_stats.items():
            left_id, right_id = pair
            if left_id not in positions or right_id not in positions:
                continue

            dist = distance_xy(
                positions[left_id],
                positions[right_id],
            )

            stat["sample_count"] += 1

            if dist < stat["minimum_distance_cm"]:
                stat["minimum_distance_cm"] = dist
                stat["minimum_distance_frame"] = frame
                stat["player_a_loc_cm"] = list(positions[left_id])
                stat["player_b_loc_cm"] = list(positions[right_id])

            if dist < settings["minimum_player_distance_error_cm"]:
                stat["error_sample_count"] += 1
            elif dist < settings["minimum_player_distance_cm"]:
                stat["warning_sample_count"] += 1

    results: List[Dict[str, Any]] = []

    for stat in pair_stats.values():
        minimum_distance = stat["minimum_distance_cm"]

        if math.isinf(minimum_distance):
            stat["minimum_distance_cm"] = None
            stat["level"] = LEVEL_PASS
            results.append(stat)
            continue

        if minimum_distance < settings["minimum_player_distance_error_cm"]:
            level = LEVEL_ERROR
            code = "PLAYER_DISTANCE_BELOW_ERROR"
        elif minimum_distance < settings["minimum_player_distance_cm"]:
            level = LEVEL_WARNING
            code = "PLAYER_DISTANCE_BELOW_WARNING"
        else:
            level = LEVEL_PASS
            code = None

        stat["level"] = level

        if code is not None:
            issues.append(
                make_issue(
                    level,
                    code,
                    (
                        "{} 与 {} 的最小距离为 {:.2f} cm，frame={}"
                    ).format(
                        stat["player_a"],
                        stat["player_b"],
                        minimum_distance,
                        stat["minimum_distance_frame"],
                    ),
                    frame=stat["minimum_distance_frame"],
                    details={
                        "player_a": stat["player_a"],
                        "player_b": stat["player_b"],
                        "minimum_distance_cm": minimum_distance,
                        "minimum_warning_cm": settings[
                            "minimum_player_distance_cm"
                        ],
                        "minimum_error_cm": settings[
                            "minimum_player_distance_error_cm"
                        ],
                    },
                )
            )

        results.append(stat)

    return results


def validate_track_ids_and_maps(
    config: Dict[str, Any],
    objects: Dict[str, Any],
    issues: List[Dict[str, Any]],
) -> None:
    track_id_map = config.get("track_id_map", {})
    class_id_map = config.get("class_id_map", {})

    if track_id_map is None:
        track_id_map = {}
    if class_id_map is None:
        class_id_map = {}

    if not isinstance(track_id_map, dict):
        issues.append(
            make_issue(
                LEVEL_ERROR,
                "INVALID_TRACK_ID_MAP",
                "track_id_map 必须是 JSON object",
            )
        )
        track_id_map = {}

    if not isinstance(class_id_map, dict):
        issues.append(
            make_issue(
                LEVEL_ERROR,
                "INVALID_CLASS_ID_MAP",
                "class_id_map 必须是 JSON object",
            )
        )
        class_id_map = {}

    seen_track_ids: Dict[int, str] = {}

    for object_id, raw_object in objects.items():
        if not isinstance(raw_object, dict):
            continue

        raw_track_id = raw_object.get(
            "track_id",
            track_id_map.get(object_id),
        )

        if raw_track_id is None:
            issues.append(
                make_issue(
                    LEVEL_ERROR,
                    "MISSING_TRACK_ID",
                    "对象缺少 track_id",
                    object_id=object_id,
                )
            )
        else:
            try:
                track_id = safe_int(
                    raw_track_id,
                    "objects.{}.track_id".format(object_id),
                )
            except ConfigError as exc:
                issues.append(
                    make_issue(
                        LEVEL_ERROR,
                        "INVALID_TRACK_ID",
                        str(exc),
                        object_id=object_id,
                    )
                )
                continue

            if track_id in seen_track_ids:
                issues.append(
                    make_issue(
                        LEVEL_ERROR,
                        "DUPLICATE_TRACK_ID",
                        "track_id={} 同时用于 {} 和 {}".format(
                            track_id,
                            seen_track_ids[track_id],
                            object_id,
                        ),
                        object_id=object_id,
                        details={
                            "track_id": track_id,
                            "other_object_id": seen_track_ids[track_id],
                        },
                    )
                )
            else:
                seen_track_ids[track_id] = object_id

            if object_id in track_id_map:
                try:
                    mapped_track_id = safe_int(
                        track_id_map[object_id],
                        "track_id_map.{}".format(object_id),
                    )
                    if mapped_track_id != track_id:
                        issues.append(
                            make_issue(
                                LEVEL_ERROR,
                                "TRACK_ID_MAP_MISMATCH",
                                (
                                    "对象 track_id={} 与 track_id_map 中的 {} 不一致"
                                ).format(
                                    track_id,
                                    mapped_track_id,
                                ),
                                object_id=object_id,
                            )
                        )
                except ConfigError as exc:
                    issues.append(
                        make_issue(
                            LEVEL_ERROR,
                            "INVALID_TRACK_ID_MAP_VALUE",
                            str(exc),
                            object_id=object_id,
                        )
                    )

        category = raw_object.get("category")
        raw_class_id = raw_object.get(
            "class_id",
            class_id_map.get(category),
        )

        if raw_class_id is None:
            issues.append(
                make_issue(
                    LEVEL_ERROR,
                    "MISSING_CLASS_ID",
                    "对象缺少 class_id，且 class_id_map 无对应类别",
                    object_id=object_id,
                    category=str(category),
                )
            )
            continue

        try:
            class_id = safe_int(
                raw_class_id,
                "objects.{}.class_id".format(object_id),
            )
        except ConfigError as exc:
            issues.append(
                make_issue(
                    LEVEL_ERROR,
                    "INVALID_CLASS_ID",
                    str(exc),
                    object_id=object_id,
                    category=str(category),
                )
            )
            continue

        if category in class_id_map:
            try:
                mapped_class_id = safe_int(
                    class_id_map[category],
                    "class_id_map.{}".format(category),
                )
                if mapped_class_id != class_id:
                    issues.append(
                        make_issue(
                            LEVEL_ERROR,
                            "CLASS_ID_MAP_MISMATCH",
                            (
                                "对象 class_id={} 与 class_id_map 中的 {} 不一致"
                            ).format(
                                class_id,
                                mapped_class_id,
                            ),
                            object_id=object_id,
                            category=str(category),
                        )
                    )
            except ConfigError as exc:
                issues.append(
                    make_issue(
                        LEVEL_ERROR,
                        "INVALID_CLASS_ID_MAP_VALUE",
                        str(exc),
                        object_id=object_id,
                        category=str(category),
                    )
                )


def build_report(
    config_path: Path,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    issues: List[Dict[str, Any]] = []

    seq_id = str(config.get("seq_id", "")).strip()
    if not seq_id:
        raise ConfigError("缺少非空 seq_id")

    timeline = config.get("timeline")
    if not isinstance(timeline, dict):
        raise ConfigError("timeline 必须是 JSON object")

    frame_start = safe_int(
        timeline.get("frame_start"),
        "timeline.frame_start",
    )
    frame_end = safe_int(
        timeline.get("frame_end"),
        "timeline.frame_end",
    )
    fps = safe_float(
        timeline.get("display_rate"),
        "timeline.display_rate",
    )

    if frame_end < frame_start:
        raise ConfigError("timeline.frame_end 不能小于 frame_start")

    if fps <= 0.0:
        raise ConfigError("timeline.display_rate 必须大于 0")

    objects = config.get("objects")
    if not isinstance(objects, dict) or not objects:
        raise ConfigError("objects 必须是非空 JSON object")

    settings = merge_validation_settings(config)

    ball_contact_frames: set[int] = set()
    raw_contact_frames = config.get("contact_frames", [])
    if isinstance(raw_contact_frames, list):
        for item in raw_contact_frames:
            if not isinstance(item, dict) or "frame" not in item:
                continue
            try:
                ball_contact_frames.add(
                    safe_int(item.get("frame"), "contact_frames.frame")
                )
            except ConfigError as exc:
                issues.append(
                    make_issue(
                        LEVEL_ERROR,
                        "INVALID_CONTACT_FRAME",
                        str(exc),
                        category="ball",
                    )
                )

    validate_track_ids_and_maps(config, objects, issues)

    track_id_map = config.get("track_id_map", {})
    class_id_map = config.get("class_id_map", {})

    object_summaries: Dict[str, Any] = {}
    all_segments: List[Dict[str, Any]] = []
    parsed_keyframes_by_object: Dict[str, List[Dict[str, Any]]] = {}
    player_keyframes: Dict[str, List[Dict[str, Any]]] = {}

    for object_id, raw_object in objects.items():
        if not isinstance(raw_object, dict):
            issues.append(
                make_issue(
                    LEVEL_ERROR,
                    "INVALID_OBJECT_CONFIG",
                    "objects.{} 必须是 JSON object".format(object_id),
                    object_id=object_id,
                )
            )
            continue

        category = str(raw_object.get("category", "")).strip().lower()

        if category not in SUPPORTED_CATEGORIES:
            issues.append(
                make_issue(
                    LEVEL_ERROR,
                    "UNSUPPORTED_CATEGORY",
                    "不支持或缺失 category={!r}".format(category),
                    object_id=object_id,
                    category=category or None,
                )
            )

        raw_track_id = raw_object.get(
            "track_id",
            track_id_map.get(object_id)
            if isinstance(track_id_map, dict)
            else None,
        )
        raw_class_id = raw_object.get(
            "class_id",
            class_id_map.get(category)
            if isinstance(class_id_map, dict)
            else None,
        )

        try:
            track_id = (
                safe_int(
                    raw_track_id,
                    "objects.{}.track_id".format(object_id),
                )
                if raw_track_id is not None
                else None
            )
        except ConfigError:
            track_id = None

        try:
            class_id = (
                safe_int(
                    raw_class_id,
                    "objects.{}.class_id".format(object_id),
                )
                if raw_class_id is not None
                else None
            )
        except ConfigError:
            class_id = None

        keyframes = parse_keyframes(
            object_id,
            raw_object,
            frame_start,
            frame_end,
            issues,
        )
        parsed_keyframes_by_object[object_id] = keyframes

        if category == "player" and keyframes:
            player_keyframes[object_id] = keyframes

        object_summary, segments = analyze_object(
            object_id=object_id,
            raw_object=raw_object,
            keyframes=keyframes,
            category=category,
            track_id=track_id,
            class_id=class_id,
            frame_start=frame_start,
            frame_end=frame_end,
            fps=fps,
            settings=settings,
            issues=issues,
            ball_contact_frames=ball_contact_frames,
        )

        object_summaries[object_id] = object_summary
        all_segments.extend(segments)

    proximity_results = analyze_player_proximity(
        players=player_keyframes,
        frame_start=frame_start,
        frame_end=frame_end,
        settings=settings,
        issues=issues,
    )

    for segment in all_segments:
        segment["seq_id"] = seq_id

    # Recompute object status after global/object issues are complete.
    for object_id, summary in object_summaries.items():
        related = [
            issue
            for issue in issues
            if issue.get("object_id") == object_id
        ]
        summary["status"] = max_level(
            issue["level"] for issue in related
        )
        summary["issue_count"] = len(related)

    error_count = sum(
        1 for issue in issues if issue["level"] == LEVEL_ERROR
    )
    warning_count = sum(
        1 for issue in issues if issue["level"] == LEVEL_WARNING
    )

    if error_count > 0:
        overall_status = LEVEL_ERROR
    elif warning_count > 0:
        overall_status = LEVEL_WARNING
    else:
        overall_status = LEVEL_PASS

    total_frames = frame_end - frame_start + 1
    duration_s = (
        (frame_end - frame_start) / fps
        if frame_end > frame_start
        else 0.0
    )

    return {
        "validator": {
            "name": "FutsalMOT trajectory configuration validator",
            "version": SCRIPT_VERSION,
            "generated_at_utc": datetime.now(
                timezone.utc
            ).isoformat(),
        },
        "config_path": str(config_path.resolve()).replace("\\", "/"),
        "seq_id": seq_id,
        "timeline": {
            "frame_start": frame_start,
            "frame_end": frame_end,
            "total_frames": total_frames,
            "display_rate_fps": fps,
            "duration_s_between_first_and_last_frame": duration_s,
        },
        "validation_settings": settings,
        "summary": {
            "status": overall_status,
            "object_count": len(objects),
            "player_count": sum(
                1
                for raw_object in objects.values()
                if isinstance(raw_object, dict)
                and raw_object.get("category") == "player"
            ),
            "ball_count": sum(
                1
                for raw_object in objects.values()
                if isinstance(raw_object, dict)
                and raw_object.get("category") == "ball"
            ),
            "segment_count": len(all_segments),
            "issue_count": len(issues),
            "warning_count": warning_count,
            "error_count": error_count,
        },
        "objects": object_summaries,
        "segments": all_segments,
        "player_proximity": proximity_results,
        "issues": issues,
    }


def write_json_report(
    report: Dict[str, Any],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(
            report,
            f,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )


CSV_COLUMNS = [
    "seq_id",
    "object_id",
    "category",
    "track_id",
    "class_id",
    "segment_index",
    "start_frame",
    "end_frame",
    "duration_frames",
    "duration_s",
    "start_x_cm",
    "start_y_cm",
    "start_z_cm",
    "end_x_cm",
    "end_y_cm",
    "end_z_cm",
    "distance_xy_cm",
    "distance_3d_cm",
    "delta_z_cm",
    "speed_xy_cm_s",
    "speed_3d_cm_s",
    "vertical_speed_cm_s",
    "direction_deg",
    "turn_from_previous_deg",
    "speed_jump_from_previous_cm_s",
    "within_court",
    "level",
    "issue_codes",
]


def write_segment_csv(
    report: Dict[str, Any],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=CSV_COLUMNS,
            extrasaction="ignore",
        )
        writer.writeheader()

        for segment in report["segments"]:
            row = dict(segment)
            row["issue_codes"] = ";".join(
                segment.get("issue_codes", [])
            )
            writer.writerow(row)


def print_report_summary(
    report: Dict[str, Any],
    json_path: Path,
    csv_path: Path,
) -> None:
    summary = report["summary"]
    timeline = report["timeline"]

    print("=" * 72)
    print("FutsalMOT trajectory validator")
    print("VERSION =", SCRIPT_VERSION)
    print("SEQ_ID =", report["seq_id"])
    print(
        "FRAME_RANGE = {}..{}".format(
            timeline["frame_start"],
            timeline["frame_end"],
        )
    )
    print("FPS =", timeline["display_rate_fps"])
    print("OBJECTS =", summary["object_count"])
    print("SEGMENTS =", summary["segment_count"])
    print("=" * 72)

    for object_id, object_summary in report["objects"].items():
        print(
            "[{}] {:<12} category={:<6} keyframes={} "
            "segments={} max_speed={:.2f} cm/s".format(
                object_summary["status"],
                object_id,
                object_summary["category"],
                object_summary["keyframe_count"],
                object_summary["segment_count"],
                object_summary["max_speed_xy_cm_s"],
            )
        )

    print("-" * 72)

    for issue in report["issues"]:
        location_parts = []

        if "object_id" in issue:
            location_parts.append(issue["object_id"])
        if "frame" in issue:
            location_parts.append("frame={}".format(issue["frame"]))
        if "segment_index" in issue:
            location_parts.append(
                "segment={}".format(issue["segment_index"])
            )

        location = " ".join(location_parts)
        if location:
            location = " [{}]".format(location)

        print(
            "[{}] {}{}: {}".format(
                issue["level"],
                issue["code"],
                location,
                issue["message"],
            )
        )

    print("-" * 72)
    print(
        "STATUS={}  warnings={}  errors={}".format(
            summary["status"],
            summary["warning_count"],
            summary["error_count"],
        )
    )
    print("JSON report:", json_path)
    print("CSV segments:", csv_path)
    print("=" * 72)


def resolve_project_root(
    config: Dict[str, Any],
    config_path: Path,
) -> Path:
    raw_project_root = config.get("project_root")

    if raw_project_root is None:
        # Expected layout:
        # Content/FutsalMOT/code/configs/<config>.json
        # Move upward to the project root.
        candidate = config_path.resolve()
        parents = candidate.parents

        if len(parents) >= 5:
            return parents[4]

        raise ConfigError(
            "配置缺少 project_root，且无法从配置路径推断项目根目录"
        )

    if not isinstance(raw_project_root, str) or not raw_project_root.strip():
        raise ConfigError("project_root 必须是非空字符串")

    project_root = Path(raw_project_root)

    if not project_root.is_absolute():
        project_root = (
            config_path.resolve().parent / project_root
        ).resolve()

    return project_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate FutsalMOT multi-keyframe trajectory JSON "
            "before Unreal Engine import."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=(
            "Trajectory config JSON. Default: "
            + str(DEFAULT_CONFIG_PATH)
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Override report directory. Default: "
            "<project_root>/Saved/FutsalMOT/trajectory_reports"
        ),
    )

    parser.add_argument(
        "--strict-warnings",
        action="store_true",
        help="Return exit code 1 when WARNING issues exist.",
    )

    parser.add_argument(
        "--sample-stride",
        type=int,
        default=None,
        help=(
            "Override trajectory_validation.sample_stride_frames "
            "for full-sequence proximity checks."
        ),
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config.expanduser().resolve()

    try:
        config = load_json(config_path)

        if args.sample_stride is not None:
            if args.sample_stride <= 0:
                raise ConfigError("--sample-stride 必须大于 0")

            raw_validation = config.setdefault(
                "trajectory_validation",
                {},
            )

            if not isinstance(raw_validation, dict):
                raise ConfigError(
                    "trajectory_validation 必须是 JSON object"
                )

            raw_validation["sample_stride_frames"] = (
                args.sample_stride
            )

        report = build_report(config_path, config)
        project_root = resolve_project_root(config, config_path)

        if args.output_dir is not None:
            output_dir = args.output_dir.expanduser().resolve()
        else:
            output_dir = (
                project_root
                / "Saved"
                / "FutsalMOT"
                / "trajectory_reports"
            )

        seq_id = report["seq_id"]
        json_path = (
            output_dir
            / "trajectory_report_{}.json".format(seq_id)
        )
        csv_path = (
            output_dir
            / "trajectory_segments_{}.csv".format(seq_id)
        )

        write_json_report(report, json_path)
        write_segment_csv(report, csv_path)
        print_report_summary(report, json_path, csv_path)

        error_count = report["summary"]["error_count"]
        warning_count = report["summary"]["warning_count"]

        if error_count > 0:
            return 1

        if args.strict_warnings and warning_count > 0:
            return 1

        return 0

    except ConfigError as exc:
        print("=" * 72, file=sys.stderr)
        print("VALIDATION ABORTED", file=sys.stderr)
        print("[CONFIG ERROR]", exc, file=sys.stderr)
        print("CONFIG_PATH =", config_path, file=sys.stderr)
        print("=" * 72, file=sys.stderr)
        return 2

    except Exception as exc:
        print("=" * 72, file=sys.stderr)
        print("VALIDATION ABORTED", file=sys.stderr)
        print(
            "[UNEXPECTED ERROR] {}: {}".format(
                type(exc).__name__,
                exc,
            ),
            file=sys.stderr,
        )
        print("CONFIG_PATH =", config_path, file=sys.stderr)
        print("=" * 72, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
