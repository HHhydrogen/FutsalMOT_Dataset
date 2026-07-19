#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FutsalMOT smooth dense trajectory generator.

Version
-------
A2_5B_PCHIP_DENSE_TRAJECTORY_V1

Purpose
-------
Convert sparse multi-keyframe trajectories into dense, one-keyframe-per-frame
trajectories before Unreal Engine import.

The generated configuration:
1. Preserves every original control point exactly.
2. Uses time-aware, shape-preserving cubic Hermite interpolation (PCHIP).
3. Avoids per-coordinate overshoot between adjacent control points.
4. Writes linear dense keyframes so Unreal Engine and annotation export consume
   exactly the same trajectory.
5. Optionally invokes 14_validate_trajectory.py after generation.

Typical usage
-------------
py 15_smooth_trajectory.py

py 15_smooth_trajectory.py ^
  --config "D:/projects/FustalMOT_UEDataset/Content/FutsalMOT/code/configs/seq_test_0004.json"

Default output
--------------
For seq_test_0004.json:
Content/FutsalMOT/code/configs/seq_test_0005.json

Exit codes
----------
0: generation and optional validation succeeded
1: output was generated, but validation reported ERROR / strict WARNING
2: generation failed
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from futsalmot.core.paths import CODE_DIR, CONFIG_DIR


SCRIPT_VERSION = "A2_5B_PCHIP_DENSE_TRAJECTORY_V1"

DEFAULT_CONFIG_PATH = (
    CONFIG_DIR
    / "seq_test_0004.json"
)

DEFAULT_VALIDATION_LIMITS: Dict[str, float] = {
    "court_x_min_cm": -1950.0,
    "court_x_max_cm": 1950.0,
    "court_y_min_cm": -950.0,
    "court_y_max_cm": 950.0,
    "boundary_tolerance_cm": 1.0,
    "player_max_speed_cm_s": 750.0,
    "ball_max_speed_cm_s": 3000.0,
    "player_vertical_max_speed_cm_s": 80.0,
    "ball_vertical_max_speed_cm_s": 1800.0,
}

SUPPORTED_CATEGORIES = {"player", "ball"}


class GenerationError(RuntimeError):
    """Fatal input or trajectory-generation error."""


def is_finite_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def require_number(value: Any, field_name: str) -> float:
    if not is_finite_number(value):
        raise GenerationError(
            "字段 '{}' 必须是有限数值，当前值={!r}".format(
                field_name,
                value,
            )
        )
    return float(value)


def require_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise GenerationError(
            "字段 '{}' 必须是整数，当前值={!r}".format(
                field_name,
                value,
            )
        )

    if isinstance(value, int):
        return value

    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return int(value)

    raise GenerationError(
        "字段 '{}' 必须是整数，当前值={!r}".format(
            field_name,
            value,
        )
    )


def require_vec3(value: Any, field_name: str) -> Tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise GenerationError(
            "字段 '{}' 必须是长度为 3 的数组 [x, y, z]".format(
                field_name
            )
        )

    return (
        require_number(value[0], field_name + "[0]"),
        require_number(value[1], field_name + "[1]"),
        require_number(value[2], field_name + "[2]"),
    )


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise GenerationError("找不到配置文件：{}".format(path))

    try:
        with path.open("r", encoding="utf-8-sig") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise GenerationError(
            "JSON 解析失败：{}，line={} column={}".format(
                exc.msg,
                exc.lineno,
                exc.colno,
            )
        ) from exc
    except OSError as exc:
        raise GenerationError(
            "无法读取配置文件：{}；{}".format(path, exc)
        ) from exc

    if not isinstance(data, dict):
        raise GenerationError("配置文件顶层必须是 JSON object")

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


def auto_increment_seq_id(source_seq_id: str) -> str:
    """
    seq_test_0004 -> seq_test_0005
    episode_0099  -> episode_0100
    name          -> name_smoothed
    """
    match = re.match(r"^(.*?)(\d+)$", source_seq_id)

    if not match:
        return source_seq_id + "_smoothed"

    prefix, digits = match.groups()
    next_number = int(digits) + 1

    return "{}{:0{}d}".format(
        prefix,
        next_number,
        len(digits),
    )


def parse_source_keyframes(
    object_id: str,
    raw_object: Dict[str, Any],
    frame_start: int,
    frame_end: int,
) -> List[Dict[str, Any]]:
    raw_keyframes = raw_object.get("keyframes")

    if raw_keyframes is None:
        if "start" not in raw_object or "end" not in raw_object:
            raise GenerationError(
                "objects.{} 既没有 keyframes，也没有完整 start/end".format(
                    object_id
                )
            )

        raw_keyframes = [
            {
                "frame": frame_start,
                "loc": raw_object["start"],
            },
            {
                "frame": frame_end,
                "loc": raw_object["end"],
            },
        ]

    if not isinstance(raw_keyframes, list) or len(raw_keyframes) < 2:
        raise GenerationError(
            "objects.{}.keyframes 至少需要 2 个关键帧".format(
                object_id
            )
        )

    result: List[Dict[str, Any]] = []
    seen_frames = set()
    previous_frame: Optional[int] = None

    for index, raw_keyframe in enumerate(raw_keyframes):
        prefix = "objects.{}.keyframes[{}]".format(
            object_id,
            index,
        )

        if not isinstance(raw_keyframe, dict):
            raise GenerationError(
                "{} 必须是 JSON object".format(prefix)
            )

        frame = require_int(
            raw_keyframe.get("frame"),
            prefix + ".frame",
        )
        loc = require_vec3(
            raw_keyframe.get("loc"),
            prefix + ".loc",
        )

        if frame in seen_frames:
            raise GenerationError(
                "objects.{} 存在重复关键帧 frame={}".format(
                    object_id,
                    frame,
                )
            )

        if previous_frame is not None and frame <= previous_frame:
            raise GenerationError(
                "objects.{} 的关键帧必须按 frame 严格递增".format(
                    object_id
                )
            )

        if frame < frame_start or frame > frame_end:
            raise GenerationError(
                "objects.{} 的 frame={} 超出时间轴 {}..{}".format(
                    object_id,
                    frame,
                    frame_start,
                    frame_end,
                )
            )

        parsed = {
            "frame": frame,
            "loc": loc,
        }

        if "yaw_deg" in raw_keyframe and raw_keyframe["yaw_deg"] is not None:
            parsed["yaw_deg"] = require_number(
                raw_keyframe["yaw_deg"],
                prefix + ".yaw_deg",
            )

        result.append(parsed)
        seen_frames.add(frame)
        previous_frame = frame

    if result[0]["frame"] != frame_start:
        raise GenerationError(
            "objects.{} 的首个关键帧必须等于 frame_start={}".format(
                object_id,
                frame_start,
            )
        )

    if result[-1]["frame"] != frame_end:
        raise GenerationError(
            "objects.{} 的末个关键帧必须等于 frame_end={}".format(
                object_id,
                frame_end,
            )
        )

    return result


def sign(value: float) -> int:
    if value > 0.0:
        return 1
    if value < 0.0:
        return -1
    return 0


def pchip_slopes(
    x: Sequence[float],
    y: Sequence[float],
) -> List[float]:
    """
    Time-aware Fritsch-Carlson / PCHIP first derivatives.

    This produces a C1-continuous, shape-preserving cubic Hermite curve.
    For monotonic data within a segment, it avoids overshooting the endpoint
    coordinate range.
    """
    n = len(x)

    if n != len(y):
        raise GenerationError("PCHIP x/y 长度不一致")

    if n < 2:
        raise GenerationError("PCHIP 至少需要两个点")

    h = [x[i + 1] - x[i] for i in range(n - 1)]

    if any(step <= 0.0 for step in h):
        raise GenerationError("PCHIP frame 必须严格递增")

    delta = [
        (y[i + 1] - y[i]) / h[i]
        for i in range(n - 1)
    ]

    if n == 2:
        return [delta[0], delta[0]]

    slopes = [0.0] * n

    for i in range(1, n - 1):
        left_delta = delta[i - 1]
        right_delta = delta[i]

        if (
            left_delta == 0.0
            or right_delta == 0.0
            or sign(left_delta) != sign(right_delta)
        ):
            slopes[i] = 0.0
            continue

        w1 = 2.0 * h[i] + h[i - 1]
        w2 = h[i] + 2.0 * h[i - 1]

        slopes[i] = (w1 + w2) / (
            w1 / left_delta + w2 / right_delta
        )

    # Left endpoint derivative with shape-preserving limiting.
    left_slope = (
        (2.0 * h[0] + h[1]) * delta[0]
        - h[0] * delta[1]
    ) / (h[0] + h[1])

    if sign(left_slope) != sign(delta[0]):
        left_slope = 0.0
    elif (
        sign(delta[0]) != sign(delta[1])
        and abs(left_slope) > abs(3.0 * delta[0])
    ):
        left_slope = 3.0 * delta[0]

    slopes[0] = left_slope

    # Right endpoint derivative with shape-preserving limiting.
    right_slope = (
        (2.0 * h[-1] + h[-2]) * delta[-1]
        - h[-1] * delta[-2]
    ) / (h[-1] + h[-2])

    if sign(right_slope) != sign(delta[-1]):
        right_slope = 0.0
    elif (
        sign(delta[-1]) != sign(delta[-2])
        and abs(right_slope) > abs(3.0 * delta[-1])
    ):
        right_slope = 3.0 * delta[-1]

    slopes[-1] = right_slope

    return slopes


def hermite_value(
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    slope0: float,
    slope1: float,
    x: float,
) -> float:
    h = x1 - x0

    if h <= 0.0:
        raise GenerationError("Hermite 区段长度必须大于 0")

    t = (x - x0) / h
    t2 = t * t
    t3 = t2 * t

    h00 = 2.0 * t3 - 3.0 * t2 + 1.0
    h10 = t3 - 2.0 * t2 + t
    h01 = -2.0 * t3 + 3.0 * t2
    h11 = t3 - t2

    return (
        h00 * y0
        + h10 * h * slope0
        + h01 * y1
        + h11 * h * slope1
    )


def locate_segment(
    frames: Sequence[int],
    frame: int,
) -> int:
    if frame <= frames[0]:
        return 0

    if frame >= frames[-1]:
        return len(frames) - 2

    low = 0
    high = len(frames) - 2

    while low <= high:
        middle = (low + high) // 2

        if frames[middle] <= frame <= frames[middle + 1]:
            return middle

        if frame < frames[middle]:
            high = middle - 1
        else:
            low = middle + 1

    return max(0, min(len(frames) - 2, low))


def unwrap_angles_deg(values: Sequence[float]) -> List[float]:
    if not values:
        return []

    result = [float(values[0])]

    for value in values[1:]:
        candidate = float(value)
        previous = result[-1]

        while candidate - previous > 180.0:
            candidate -= 360.0

        while candidate - previous < -180.0:
            candidate += 360.0

        result.append(candidate)

    return result


def normalize_angle_deg(value: float) -> float:
    while value > 180.0:
        value -= 360.0
    while value <= -180.0:
        value += 360.0
    return value


def dense_pchip_keyframes(
    source_keyframes: Sequence[Dict[str, Any]],
    frame_start: int,
    frame_end: int,
    frame_interval: int,
    tangent_scale: float,
    decimals: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    frames = [keyframe["frame"] for keyframe in source_keyframes]
    xs = [keyframe["loc"][0] for keyframe in source_keyframes]
    ys = [keyframe["loc"][1] for keyframe in source_keyframes]
    zs = [keyframe["loc"][2] for keyframe in source_keyframes]

    frame_values = [float(frame) for frame in frames]

    x_slopes = [
        slope * tangent_scale
        for slope in pchip_slopes(frame_values, xs)
    ]
    y_slopes = [
        slope * tangent_scale
        for slope in pchip_slopes(frame_values, ys)
    ]
    z_slopes = [
        slope * tangent_scale
        for slope in pchip_slopes(frame_values, zs)
    ]

    has_all_explicit_yaw = all(
        "yaw_deg" in keyframe
        for keyframe in source_keyframes
    )
    has_any_explicit_yaw = any(
        "yaw_deg" in keyframe
        for keyframe in source_keyframes
    )

    yaw_values: Optional[List[float]] = None
    yaw_slopes: Optional[List[float]] = None

    if has_all_explicit_yaw:
        yaw_values = unwrap_angles_deg(
            [keyframe["yaw_deg"] for keyframe in source_keyframes]
        )
        yaw_slopes = [
            slope * tangent_scale
            for slope in pchip_slopes(frame_values, yaw_values)
        ]

    source_by_frame = {
        keyframe["frame"]: keyframe
        for keyframe in source_keyframes
    }

    sample_frames = list(
        range(frame_start, frame_end + 1, frame_interval)
    )

    if not sample_frames or sample_frames[-1] != frame_end:
        sample_frames.append(frame_end)

    # Always preserve every original control frame, even when interval > 1.
    sample_frames = sorted(set(sample_frames).union(frames))

    dense: List[Dict[str, Any]] = []

    for frame in sample_frames:
        if frame in source_by_frame:
            source = source_by_frame[frame]
            output_keyframe: Dict[str, Any] = {
                "frame": frame,
                "loc": [
                    round(source["loc"][0], decimals),
                    round(source["loc"][1], decimals),
                    round(source["loc"][2], decimals),
                ],
            }

            if "yaw_deg" in source:
                output_keyframe["yaw_deg"] = round(
                    normalize_angle_deg(source["yaw_deg"]),
                    decimals,
                )

            dense.append(output_keyframe)
            continue

        segment = locate_segment(frames, frame)

        f0 = float(frames[segment])
        f1 = float(frames[segment + 1])

        x_value = hermite_value(
            f0,
            f1,
            xs[segment],
            xs[segment + 1],
            x_slopes[segment],
            x_slopes[segment + 1],
            float(frame),
        )
        y_value = hermite_value(
            f0,
            f1,
            ys[segment],
            ys[segment + 1],
            y_slopes[segment],
            y_slopes[segment + 1],
            float(frame),
        )
        z_value = hermite_value(
            f0,
            f1,
            zs[segment],
            zs[segment + 1],
            z_slopes[segment],
            z_slopes[segment + 1],
            float(frame),
        )

        output_keyframe = {
            "frame": frame,
            "loc": [
                round(x_value, decimals),
                round(y_value, decimals),
                round(z_value, decimals),
            ],
        }

        if yaw_values is not None and yaw_slopes is not None:
            yaw_value = hermite_value(
                f0,
                f1,
                yaw_values[segment],
                yaw_values[segment + 1],
                yaw_slopes[segment],
                yaw_slopes[segment + 1],
                float(frame),
            )
            output_keyframe["yaw_deg"] = round(
                normalize_angle_deg(yaw_value),
                decimals,
            )

        dense.append(output_keyframe)

    metadata = {
        "source_keyframe_count": len(source_keyframes),
        "dense_keyframe_count": len(dense),
        "source_control_frames": frames,
        "explicit_yaw_mode": (
            "all_control_yaws_smoothed"
            if has_all_explicit_yaw
            else "partial_yaws_removed"
            if has_any_explicit_yaw
            else "automatic_motion_direction"
        ),
    }

    return dense, metadata


def distance_xy(
    a: Sequence[float],
    b: Sequence[float],
) -> float:
    return math.hypot(
        b[0] - a[0],
        b[1] - a[1],
    )


def calculate_dense_stats(
    keyframes: Sequence[Dict[str, Any]],
    fps: float,
) -> Dict[str, Any]:
    max_speed_xy = 0.0
    max_vertical_speed = 0.0
    total_distance_xy = 0.0

    max_speed_frame: Optional[int] = None
    max_vertical_speed_frame: Optional[int] = None

    for left, right in zip(keyframes, keyframes[1:]):
        frame_delta = right["frame"] - left["frame"]

        if frame_delta <= 0:
            raise GenerationError(
                "生成后的 dense keyframes frame 非严格递增"
            )

        duration = frame_delta / fps
        loc0 = left["loc"]
        loc1 = right["loc"]

        dist_xy = distance_xy(loc0, loc1)
        speed_xy = dist_xy / duration
        vertical_speed = abs(loc1[2] - loc0[2]) / duration

        total_distance_xy += dist_xy

        if speed_xy > max_speed_xy:
            max_speed_xy = speed_xy
            max_speed_frame = right["frame"]

        if vertical_speed > max_vertical_speed:
            max_vertical_speed = vertical_speed
            max_vertical_speed_frame = right["frame"]

    return {
        "total_distance_xy_cm": total_distance_xy,
        "max_speed_xy_cm_s": max_speed_xy,
        "max_speed_frame": max_speed_frame,
        "max_vertical_speed_cm_s": max_vertical_speed,
        "max_vertical_speed_frame": max_vertical_speed_frame,
    }


def merged_limits(config: Dict[str, Any]) -> Dict[str, float]:
    result = dict(DEFAULT_VALIDATION_LIMITS)
    raw = config.get("trajectory_validation", {})

    if raw is None:
        raw = {}

    if not isinstance(raw, dict):
        raise GenerationError(
            "trajectory_validation 必须是 JSON object"
        )

    for key in result:
        if key in raw:
            result[key] = require_number(
                raw[key],
                "trajectory_validation." + key,
            )

    return result


def check_dense_object(
    object_id: str,
    category: str,
    dense_keyframes: Sequence[Dict[str, Any]],
    stats: Dict[str, Any],
    limits: Dict[str, float],
) -> List[str]:
    errors: List[str] = []
    tolerance = limits["boundary_tolerance_cm"]

    for keyframe in dense_keyframes:
        x, y, _ = keyframe["loc"]

        if not (
            limits["court_x_min_cm"] - tolerance
            <= x
            <= limits["court_x_max_cm"] + tolerance
        ):
            errors.append(
                "{} frame={} X={} 超出安全边界".format(
                    object_id,
                    keyframe["frame"],
                    x,
                )
            )
            break

        if not (
            limits["court_y_min_cm"] - tolerance
            <= y
            <= limits["court_y_max_cm"] + tolerance
        ):
            errors.append(
                "{} frame={} Y={} 超出安全边界".format(
                    object_id,
                    keyframe["frame"],
                    y,
                )
            )
            break

    if category == "player":
        max_speed = limits["player_max_speed_cm_s"]
        max_vertical_speed = limits[
            "player_vertical_max_speed_cm_s"
        ]
    else:
        max_speed = limits["ball_max_speed_cm_s"]
        max_vertical_speed = limits[
            "ball_vertical_max_speed_cm_s"
        ]

    if stats["max_speed_xy_cm_s"] > max_speed + 1e-6:
        errors.append(
            (
                "{} 平滑后最大速度 {:.3f} cm/s 超过限制 {:.3f} cm/s，"
                "frame={}"
            ).format(
                object_id,
                stats["max_speed_xy_cm_s"],
                max_speed,
                stats["max_speed_frame"],
            )
        )

    if (
        stats["max_vertical_speed_cm_s"]
        > max_vertical_speed + 1e-6
    ):
        errors.append(
            (
                "{} 平滑后最大垂直速度 {:.3f} cm/s 超过限制 "
                "{:.3f} cm/s，frame={}"
            ).format(
                object_id,
                stats["max_vertical_speed_cm_s"],
                max_vertical_speed,
                stats["max_vertical_speed_frame"],
            )
        )

    return errors


def preserve_backup(output_path: Path) -> Optional[Path]:
    if not output_path.exists():
        return None

    backup_path = output_path.with_suffix(
        output_path.suffix + ".bak"
    )

    counter = 1
    while backup_path.exists():
        backup_path = output_path.with_suffix(
            output_path.suffix + ".bak{}".format(counter)
        )
        counter += 1

    output_path.replace(backup_path)
    return backup_path


def generate_output_config(
    source_config: Dict[str, Any],
    source_config_path: Path,
    output_seq_id: str,
    frame_interval: int,
    tangent_scale: float,
    decimals: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    config = copy.deepcopy(source_config)

    source_seq_id = str(config.get("seq_id", "")).strip()

    if not source_seq_id:
        raise GenerationError("缺少非空 seq_id")

    timeline = config.get("timeline")

    if not isinstance(timeline, dict):
        raise GenerationError("timeline 必须是 JSON object")

    frame_start = require_int(
        timeline.get("frame_start"),
        "timeline.frame_start",
    )
    frame_end = require_int(
        timeline.get("frame_end"),
        "timeline.frame_end",
    )
    fps = require_number(
        timeline.get("display_rate"),
        "timeline.display_rate",
    )

    if frame_end <= frame_start:
        raise GenerationError(
            "timeline.frame_end 必须大于 frame_start"
        )

    if fps <= 0.0:
        raise GenerationError(
            "timeline.display_rate 必须大于 0"
        )

    objects = config.get("objects")

    if not isinstance(objects, dict) or not objects:
        raise GenerationError(
            "objects 必须是非空 JSON object"
        )

    limits = merged_limits(config)
    object_generation_stats: Dict[str, Any] = {}
    generation_errors: List[str] = []

    for object_id, raw_object in objects.items():
        if not isinstance(raw_object, dict):
            raise GenerationError(
                "objects.{} 必须是 JSON object".format(
                    object_id
                )
            )

        category = str(
            raw_object.get("category", "")
        ).strip().lower()

        if category not in SUPPORTED_CATEGORIES:
            raise GenerationError(
                "objects.{} category={!r} 不受支持".format(
                    object_id,
                    category,
                )
            )

        source_keyframes = parse_source_keyframes(
            object_id,
            raw_object,
            frame_start,
            frame_end,
        )

        dense_keyframes, dense_metadata = dense_pchip_keyframes(
            source_keyframes,
            frame_start,
            frame_end,
            frame_interval,
            tangent_scale,
            decimals,
        )

        stats = calculate_dense_stats(
            dense_keyframes,
            fps,
        )
        object_errors = check_dense_object(
            object_id,
            category,
            dense_keyframes,
            stats,
            limits,
        )

        generation_errors.extend(object_errors)

        # Unreal Engine consumes the already-smoothed dense points linearly.
        raw_object["interpolation"] = "linear"
        raw_object["keyframes"] = dense_keyframes
        raw_object.pop("start", None)
        raw_object.pop("end", None)

        object_generation_stats[object_id] = {
            "category": category,
            **dense_metadata,
            **stats,
            "status": "ERROR" if object_errors else "PASS",
            "errors": object_errors,
        }

    if generation_errors:
        raise GenerationError(
            "平滑轨迹未通过生成阶段安全检查：\n- "
            + "\n- ".join(generation_errors)
        )

    config["schema_version"] = "2.1"
    config["seq_id"] = output_seq_id

    config["trajectory_generation"] = {
        "generator_version": SCRIPT_VERSION,
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "source_config": str(
            source_config_path.resolve()
        ).replace("\\", "/"),
        "source_seq_id": source_seq_id,
        "output_seq_id": output_seq_id,
        "method": "shape_preserving_cubic_hermite_pchip",
        "dense_keyframes": True,
        "keyframe_interval_frames": frame_interval,
        "tangent_scale": tangent_scale,
        "coordinate_decimals": decimals,
        "preserve_original_control_points": True,
        "ue_interpolation": "linear",
        "object_stats": object_generation_stats,
    }

    summary = {
        "source_seq_id": source_seq_id,
        "output_seq_id": output_seq_id,
        "frame_start": frame_start,
        "frame_end": frame_end,
        "display_rate": fps,
        "frame_interval": frame_interval,
        "tangent_scale": tangent_scale,
        "object_count": len(objects),
        "object_stats": object_generation_stats,
    }

    return config, summary


def run_validator(
    validator_path: Path,
    output_path: Path,
    strict_warnings: bool,
) -> int:
    command = [
        sys.executable,
        str(validator_path),
        "--config",
        str(output_path),
    ]

    if strict_warnings:
        command.append("--strict-warnings")

    print("[STEP] Running validator:")
    print("       " + " ".join('"{}"'.format(part) for part in command))

    completed = subprocess.run(
        command,
        check=False,
    )

    return int(completed.returncode)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate dense shape-preserving PCHIP trajectories "
            "for FutsalMOT."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Source sparse trajectory config JSON.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output JSON path. Default: source config directory / "
            "<auto-incremented-seq-id>.json"
        ),
    )

    parser.add_argument(
        "--seq-id",
        type=str,
        default=None,
        help=(
            "Output seq_id. Default: auto increment trailing number, "
            "for example seq_test_0004 -> seq_test_0005."
        ),
    )

    parser.add_argument(
        "--frame-interval",
        type=int,
        default=1,
        help=(
            "Dense keyframe interval in frames. Default: 1. "
            "Original control frames are always retained."
        ),
    )

    parser.add_argument(
        "--tangent-scale",
        type=float,
        default=1.0,
        help=(
            "Scale PCHIP tangents. Range 0..1. "
            "Default 1.0 preserves the standard PCHIP curve."
        ),
    )

    parser.add_argument(
        "--decimals",
        type=int,
        default=6,
        help="Coordinate decimal places. Default: 6.",
    )

    parser.add_argument(
        "--validator",
        type=Path,
        default=None,
        help=(
            "Validator script path. Default: "
            "14_validate_trajectory.py beside this script."
        ),
    )

    parser.add_argument(
        "--skip-validator",
        action="store_true",
        help="Do not invoke the trajectory validator after generation.",
    )

    parser.add_argument(
        "--strict-warnings",
        action="store_true",
        help="Treat validator WARNING results as failure.",
    )

    parser.add_argument(
        "--no-backup",
        action="store_true",
        help=(
            "Do not move an existing output to .bak before overwriting."
        ),
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config.expanduser().resolve()

    try:
        if args.frame_interval <= 0:
            raise GenerationError(
                "--frame-interval 必须大于 0"
            )

        if not math.isfinite(args.tangent_scale):
            raise GenerationError(
                "--tangent-scale 必须是有限数值"
            )

        if not (0.0 <= args.tangent_scale <= 1.0):
            raise GenerationError(
                "--tangent-scale 必须位于 0..1"
            )

        if args.decimals < 0 or args.decimals > 12:
            raise GenerationError(
                "--decimals 必须位于 0..12"
            )

        source_config = load_json(config_path)
        source_seq_id = str(
            source_config.get("seq_id", "")
        ).strip()

        if not source_seq_id:
            raise GenerationError("缺少非空 seq_id")

        output_seq_id = (
            args.seq_id.strip()
            if isinstance(args.seq_id, str) and args.seq_id.strip()
            else auto_increment_seq_id(source_seq_id)
        )

        if args.output is None:
            output_path = (
                config_path.parent
                / "{}.json".format(output_seq_id)
            ).resolve()
        else:
            output_path = args.output.expanduser().resolve()

        if output_path == config_path:
            raise GenerationError(
                "输出路径不能与源配置相同；请保留稀疏控制点基线"
            )

        output_config, summary = generate_output_config(
            source_config,
            config_path,
            output_seq_id,
            args.frame_interval,
            args.tangent_scale,
            args.decimals,
        )

        backup_path = None

        if output_path.exists() and not args.no_backup:
            backup_path = preserve_backup(output_path)

        write_json(output_path, output_config)

        print("=" * 76)
        print("FutsalMOT smooth trajectory generator")
        print("VERSION =", SCRIPT_VERSION)
        print("SOURCE_SEQ_ID =", summary["source_seq_id"])
        print("OUTPUT_SEQ_ID =", summary["output_seq_id"])
        print(
            "FRAME_RANGE = {}..{}".format(
                summary["frame_start"],
                summary["frame_end"],
            )
        )
        print("FPS =", summary["display_rate"])
        print("METHOD = shape_preserving_cubic_hermite_pchip")
        print("TANGENT_SCALE =", summary["tangent_scale"])
        print("FRAME_INTERVAL =", summary["frame_interval"])
        print("-" * 76)

        for object_id, stats in summary["object_stats"].items():
            print(
                "[{}] {:<12} controls={} dense={} "
                "max_speed={:.3f} cm/s".format(
                    stats["status"],
                    object_id,
                    stats["source_keyframe_count"],
                    stats["dense_keyframe_count"],
                    stats["max_speed_xy_cm_s"],
                )
            )

        print("-" * 76)
        print("OUTPUT =", output_path)

        if backup_path is not None:
            print("BACKUP =", backup_path)

        print("=" * 76)

        if args.skip_validator:
            print("[DONE] Generated without external validation.")
            return 0

        validator_path = (
            args.validator.expanduser().resolve()
            if args.validator is not None
            else (
                CODE_DIR
                / "futsalmot"
                / "scripts"
                / "validate_trajectory.py"
            )
        )

        if not validator_path.exists():
            raise GenerationError(
                "找不到验证器：{}；可使用 --skip-validator 跳过".format(
                    validator_path
                )
            )

        validator_return_code = run_validator(
            validator_path,
            output_path,
            args.strict_warnings,
        )

        if validator_return_code != 0:
            print(
                "[FAILED] Output generated, but validator return code={}".format(
                    validator_return_code
                )
            )
            return 1

        print("[DONE] Smoothed config generated and validated.")
        return 0

    except GenerationError as exc:
        print("=" * 76, file=sys.stderr)
        print("GENERATION FAILED", file=sys.stderr)
        print("[ERROR]", exc, file=sys.stderr)
        print("CONFIG_PATH =", config_path, file=sys.stderr)
        print("=" * 76, file=sys.stderr)
        return 2

    except Exception as exc:
        print("=" * 76, file=sys.stderr)
        print("GENERATION FAILED", file=sys.stderr)
        print(
            "[UNEXPECTED ERROR] {}: {}".format(
                type(exc).__name__,
                exc,
            ),
            file=sys.stderr,
        )
        print("CONFIG_PATH =", config_path, file=sys.stderr)
        print("=" * 76, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
