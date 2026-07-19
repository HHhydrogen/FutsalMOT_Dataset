#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FutsalMOT A3.4 single-episode random generator for 4v4 outfield futsal.

Version:
    A3_4_RANDOM_EPISODE_8P_V2

The generated episode always contains eight outfield players (four per team) and
one ball.  There are no goalkeeper actors.  Team A attacks toward +X and Team B
defends that goal.

Movement design:
- role-based 4v4 starting shape with deterministic minimum spacing;
- one ball carrier, two width/support runners, and one anchor for Team A;
- primary/secondary markers plus a deeper cover defender for Team B;
- frame-aligned event times and deterministic retry seeds;
- defender events carry predictive/acceleration-limited pursuit parameters used
  by 12_compile_trajectory.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

SCRIPT_VERSION = "A3_4_RANDOM_EPISODE_8P_V2"
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = (SCRIPT_DIR / ".." / ".." / "..").resolve()

COURT_X_MIN, COURT_X_MAX = -1950.0, 1950.0
COURT_Y_MIN, COURT_Y_MAX = -950.0, 950.0
PLAYER_Z_CM = 90.0
BALL_Z_CM = 11.0
FPS = 30
DURATION_SEC = 10.0
TOTAL_FRAMES = int(DURATION_SEC * FPS)
PLAYER_MAX_SPEED_CM_S = 750.0
BALL_MAX_SPEED_CM_S = 3000.0
MIN_START_SPACING_CM = 175.0

TEAM_A = tuple("Player_{:02d}".format(i) for i in range(1, 5))
TEAM_B = tuple("Player_{:02d}".format(i) for i in range(5, 9))
PLAYER_IDS = TEAM_A + TEAM_B

TEMPLATE_NAMES = {
    1: "solo_dribble_shot_4v4",
    2: "dribble_pass_receive_4v4",
    3: "pass_receive_dribble_shot_4v4",
}


def sec(value: float) -> float:
    """Round a time to the nearest frame boundary."""
    frame = int(round(float(value) * FPS))
    return round(frame / float(FPS), 9)


def sec_ceil(value: float) -> float:
    """Round a positive time upward to the next frame boundary."""
    frame = int(math.ceil(float(value) * FPS - 1e-12))
    return round(frame / float(FPS), 9)


def distance_xy(a: Sequence[float], b: Sequence[float]) -> float:
    return math.hypot(float(b[0]) - float(a[0]), float(b[1]) - float(a[1]))


def safe_motion_duration(
    distance_cm: float,
    max_speed_cm_s: float,
    sampled_duration: float,
    *,
    safety: float = 1.58,
) -> float:
    """Return a frame-aligned duration safe for cubic smoothstep peak speed."""
    if max_speed_cm_s <= 0.0:
        raise ValueError("max_speed_cm_s must be positive")
    minimum = safety * float(distance_cm) / float(max_speed_cm_s) + 1.0 / FPS
    return sec_ceil(max(float(sampled_duration), minimum))


def atomic_write_json(path: Path, data: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.{}".format(os.getpid()))
    try:
        with tmp.open("w", encoding="utf-8", newline="\n") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(str(tmp), str(path))
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def deterministic_rng_seed(
    requested_seed: int,
    template_id: int,
    attempt_index: int,
) -> int:
    """Stable retry seed; never uses Python's randomized hash()."""
    if int(attempt_index) == 1:
        return int(requested_seed)
    payload = "{}|{}|{}|{}".format(
        SCRIPT_VERSION,
        int(requested_seed),
        int(template_id),
        int(attempt_index),
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def clip_x(value: float, margin: float = 50.0) -> float:
    return max(COURT_X_MIN + margin, min(COURT_X_MAX - margin, float(value)))


def clip_y(value: float, margin: float = 50.0) -> float:
    return max(COURT_Y_MIN + margin, min(COURT_Y_MAX - margin, float(value)))


def loc(
    x: float,
    y: float,
    z: float = PLAYER_Z_CM,
    margin: float = 50.0,
) -> List[float]:
    return [
        round(clip_x(x, margin), 1),
        round(clip_y(y, margin), 1),
        round(float(z), 1),
    ]


def minimum_pair_distance(points: Mapping[str, Sequence[float]]) -> float:
    values = list(points.items())
    minimum = float("inf")
    for index, (_, left) in enumerate(values):
        for _, right in values[index + 1 :]:
            minimum = min(minimum, distance_xy(left, right))
    return minimum


def formation_4v4(rng: random.Random) -> Dict[str, List[float]]:
    """Generate a spaced 4v4 outfield shape; Team A attacks toward +X.

    Defenders are initialized close to their intended goal-side marking lanes.
    This avoids the old transient where a defender started too deep, ran back
    through the attacker to reach the follow offset, and produced body overlap.
    """
    for _ in range(300):
        base_x = rng.uniform(-1050.0, -650.0)
        base_y = rng.uniform(-90.0, 90.0)

        p1 = loc(base_x, base_y)
        p2 = loc(
            base_x + rng.uniform(140.0, 300.0),
            base_y + rng.uniform(430.0, 620.0),
        )
        p3 = loc(
            base_x + rng.uniform(140.0, 300.0),
            base_y - rng.uniform(430.0, 620.0),
        )
        p4 = loc(
            base_x - rng.uniform(260.0, 430.0),
            base_y + rng.uniform(-150.0, 150.0),
        )

        starts = {
            # Team A: carrier, right width, left width, anchor.
            "Player_01": p1,
            "Player_02": p2,
            "Player_03": p3,
            "Player_04": p4,
            # Team B: goal-side press/markers and a deeper cover defender.
            "Player_05": loc(
                p1[0] + rng.uniform(190.0, 275.0),
                p1[1] + rng.uniform(-80.0, 80.0),
            ),
            "Player_06": loc(
                p2[0] + rng.uniform(205.0, 295.0),
                p2[1] + rng.uniform(85.0, 155.0),
            ),
            "Player_07": loc(
                p3[0] + rng.uniform(215.0, 310.0),
                p3[1] - rng.uniform(85.0, 155.0),
            ),
            "Player_08": loc(
                p1[0] + rng.uniform(455.0, 620.0),
                p1[1] + rng.uniform(170.0, 300.0),
            ),
        }
        if minimum_pair_distance(starts) >= MIN_START_SPACING_CM:
            return starts
    raise ValueError("unable to sample a collision-safe 4v4 starting formation")


def random_goal_target(rng: random.Random) -> List[float]:
    return loc(1900.0, rng.uniform(-320.0, 320.0), PLAYER_Z_CM)


def random_dribble_target(
    rng: random.Random,
    start: Sequence[float],
    *,
    min_advance: float = 800.0,
    max_advance: float = 1450.0,
) -> List[float]:
    min_x = max(350.0, float(start[0]) + min_advance)
    max_x = min(1550.0, float(start[0]) + max_advance)
    if min_x > max_x:
        min_x = max_x
    return loc(rng.uniform(min_x, max_x), rng.uniform(-260.0, 260.0))


def support_targets(
    rng: random.Random,
    reference: Sequence[float],
) -> Dict[str, List[float]]:
    ref_x = float(reference[0])
    return {
        "Player_02": loc(
            ref_x + rng.uniform(-100.0, 120.0),
            rng.uniform(500.0, 710.0),
        ),
        "Player_03": loc(
            ref_x + rng.uniform(-100.0, 120.0),
            -rng.uniform(500.0, 710.0),
        ),
        "Player_04": loc(
            ref_x - rng.uniform(300.0, 470.0),
            rng.uniform(-140.0, 140.0),
        ),
    }


def move_event(
    event_id: str,
    actor: str,
    start_t: float,
    end_t: float,
    target_loc: Sequence[float],
    *,
    tactical_role: str,
) -> Dict[str, Any]:
    return {
        "event_id": event_id,
        "type": "move",
        "actor": actor,
        "start_t": sec(start_t),
        "end_t": sec(end_t),
        "target_loc": list(target_loc),
        "tactical_role": tactical_role,
    }


def defend_event(
    event_id: str,
    actor: str,
    target: str,
    start_t: float,
    end_t: float,
    rng: random.Random,
    *,
    follow_distance: Tuple[float, float],
    side_offset: Tuple[float, float],
    speed: Tuple[float, float] = (440.0, 540.0),
    acceleration: Tuple[float, float] = (700.0, 950.0),
    lookahead_frames: Tuple[int, int] = (3, 6),
    tactical_role: str,
) -> Dict[str, Any]:
    return {
        "event_id": event_id,
        "type": "defend_follow",
        "actor": actor,
        "start_t": sec(start_t),
        "end_t": sec(end_t),
        "target": target,
        "positioning": "goal_side",
        "follow_distance_cm": round(rng.uniform(*follow_distance), 1),
        "side_offset_cm": round(rng.uniform(*side_offset), 1),
        "max_speed_cm_s": round(rng.uniform(*speed), 1),
        "max_acceleration_cm_s2": round(rng.uniform(*acceleration), 1),
        "lookahead_frames": rng.randint(*lookahead_frames),
        "response_time_sec": round(rng.uniform(0.30, 0.48), 3),
        "avoidance_radius_cm": round(rng.uniform(125.0, 155.0), 1),
        "avoidance_weight": round(rng.uniform(0.55, 0.82), 3),
        "tactical_role": tactical_role,
    }


def add_standard_defence(
    events: List[Dict[str, Any]],
    rng: random.Random,
    end_t: float,
    *,
    primary_target: str,
    secondary_target: str = "Player_02",
    wide_target: str = "Player_03",
    first_event_index: int,
) -> None:
    event_ids = ["event_{:03d}".format(first_event_index + i) for i in range(4)]
    events.extend(
        [
            defend_event(
                event_ids[0],
                "Player_05",
                primary_target,
                0.0,
                end_t,
                rng,
                follow_distance=(165.0, 225.0),
                side_offset=(-75.0, 75.0),
                tactical_role="primary_press",
            ),
            defend_event(
                event_ids[1],
                "Player_06",
                secondary_target,
                0.0,
                DURATION_SEC,
                rng,
                follow_distance=(180.0, 245.0),
                side_offset=(80.0, 135.0),
                tactical_role="receiver_mark",
            ),
            defend_event(
                event_ids[2],
                "Player_07",
                wide_target,
                0.0,
                DURATION_SEC,
                rng,
                follow_distance=(190.0, 260.0),
                side_offset=(-135.0, -80.0),
                tactical_role="wide_mark",
            ),
            defend_event(
                event_ids[3],
                "Player_08",
                "possession_owner",
                0.0,
                end_t,
                rng,
                follow_distance=(380.0, 480.0),
                side_offset=(145.0, 220.0),
                speed=(400.0, 480.0),
                acceleration=(600.0, 820.0),
                lookahead_frames=(4, 8),
                tactical_role="cover_defender",
            ),
        ]
    )


def gen_template_1(rng: random.Random) -> Tuple[Dict[str, List[float]], List[Dict[str, Any]]]:
    starts = formation_4v4(rng)
    dribble_target = random_dribble_target(rng, starts["Player_01"])
    shot_target = random_goal_target(rng)
    supports = support_targets(rng, dribble_target)

    dribble_duration = safe_motion_duration(
        distance_xy(starts["Player_01"], dribble_target),
        PLAYER_MAX_SPEED_CM_S,
        rng.uniform(2.6, 4.2),
    )
    shot_duration = safe_motion_duration(
        distance_xy(dribble_target, shot_target),
        BALL_MAX_SPEED_CM_S,
        rng.uniform(0.55, 0.82),
    )
    shot_end = sec_ceil(dribble_duration + shot_duration)
    if shot_end + 0.5 >= DURATION_SEC:
        raise ValueError("template 1 timing exceeds episode duration")

    events: List[Dict[str, Any]] = [
        {
            "event_id": "event_001",
            "type": "dribble",
            "actor": "Player_01",
            "start_t": 0.0,
            "end_t": sec(dribble_duration),
            "target_loc": dribble_target,
            "ball_ahead_cm": 45.0,
            "tactical_role": "central_progression",
        },
        {
            "event_id": "event_002",
            "type": "shot",
            "actor": "Player_01",
            "start_t": sec(dribble_duration),
            "end_t": sec(shot_end),
            "target_loc": shot_target,
            "arc_height_cm": 55.0,
        },
        move_event(
            "event_003",
            "Player_01",
            shot_end,
            DURATION_SEC,
            loc(1850.0, shot_target[1]),
            tactical_role="shot_follow_through",
        ),
        move_event(
            "event_004",
            "Player_02",
            0.0,
            DURATION_SEC,
            supports["Player_02"],
            tactical_role="right_width_support",
        ),
        move_event(
            "event_005",
            "Player_03",
            0.0,
            DURATION_SEC,
            supports["Player_03"],
            tactical_role="left_width_support",
        ),
        move_event(
            "event_006",
            "Player_04",
            0.0,
            DURATION_SEC,
            supports["Player_04"],
            tactical_role="rest_defence_anchor",
        ),
    ]
    add_standard_defence(
        events,
        rng,
        DURATION_SEC,
        primary_target="Player_01",
        first_event_index=7,
    )
    return starts, events


def gen_template_2(rng: random.Random) -> Tuple[Dict[str, List[float]], List[Dict[str, Any]]]:
    starts = formation_4v4(rng)
    receiver_target = loc(
        starts["Player_02"][0] + rng.uniform(100.0, 230.0),
        starts["Player_02"][1] + rng.uniform(-70.0, 70.0),
    )
    dribble_target = loc(
        receiver_target[0] - rng.uniform(120.0, 220.0),
        rng.uniform(-120.0, 180.0),
    )
    pass_target = loc(receiver_target[0], receiver_target[1], BALL_Z_CM)

    dribble_duration = safe_motion_duration(
        distance_xy(starts["Player_01"], dribble_target),
        PLAYER_MAX_SPEED_CM_S,
        rng.uniform(1.8, 3.2),
    )
    pass_duration = safe_motion_duration(
        distance_xy(dribble_target, receiver_target),
        BALL_MAX_SPEED_CM_S,
        rng.uniform(0.50, 0.82),
    )
    pass_end = sec_ceil(dribble_duration + pass_duration)
    if pass_end >= DURATION_SEC - 1.0:
        raise ValueError("template 2 timing exceeds episode duration")

    supports = support_targets(rng, receiver_target)
    continuation_target = loc(
        receiver_target[0] + rng.uniform(260.0, 430.0),
        receiver_target[1] + rng.uniform(-110.0, 110.0),
    )

    events: List[Dict[str, Any]] = [
        {
            "event_id": "event_001",
            "type": "dribble",
            "actor": "Player_01",
            "start_t": 0.0,
            "end_t": sec(dribble_duration),
            "target_loc": dribble_target,
            "ball_ahead_cm": 45.0,
        },
        {
            "event_id": "event_002",
            "type": "pass",
            "actor": "Player_01",
            "from": "Player_01",
            "to": "Player_02",
            "start_t": sec(dribble_duration),
            "end_t": sec(pass_end),
            "target_loc": pass_target,
            "arc_height_cm": 20.0,
        },
        move_event(
            "event_003",
            "Player_02",
            0.0,
            pass_end,
            receiver_target,
            tactical_role="timed_receive_run",
        ),
        {
            "event_id": "event_004",
            "type": "receive",
            "actor": "Player_02",
            "source_event": "event_002",
            "time": sec(pass_end),
        },
        move_event(
            "event_005",
            "Player_02",
            pass_end,
            DURATION_SEC,
            continuation_target,
            tactical_role="post_receive_progression",
        ),
        move_event(
            "event_006",
            "Player_03",
            0.0,
            DURATION_SEC,
            supports["Player_03"],
            tactical_role="far_side_width",
        ),
        move_event(
            "event_007",
            "Player_04",
            0.0,
            DURATION_SEC,
            supports["Player_04"],
            tactical_role="rest_defence_anchor",
        ),
    ]
    add_standard_defence(
        events,
        rng,
        DURATION_SEC,
        primary_target="Player_01",
        secondary_target="Player_02",
        first_event_index=8,
    )
    return starts, events


def gen_template_3(rng: random.Random) -> Tuple[Dict[str, List[float]], List[Dict[str, Any]]]:
    starts = formation_4v4(rng)
    p1_dribble_target = loc(
        starts["Player_01"][0] + rng.uniform(130.0, 260.0),
        starts["Player_01"][1] + rng.uniform(-70.0, 70.0),
    )
    receiver_target = loc(
        starts["Player_02"][0] + rng.uniform(80.0, 210.0),
        starts["Player_02"][1] + rng.uniform(-70.0, 70.0),
    )
    pass_target = loc(receiver_target[0], receiver_target[1], BALL_Z_CM)

    first_duration = safe_motion_duration(
        distance_xy(starts["Player_01"], p1_dribble_target),
        PLAYER_MAX_SPEED_CM_S,
        rng.uniform(0.8, 1.5),
    )
    pass_duration = safe_motion_duration(
        distance_xy(p1_dribble_target, receiver_target),
        BALL_MAX_SPEED_CM_S,
        rng.uniform(0.50, 0.82),
    )
    pass_end = sec_ceil(first_duration + pass_duration)

    dribble_target = random_dribble_target(
        rng,
        receiver_target,
        min_advance=620.0,
        max_advance=1150.0,
    )
    second_duration = safe_motion_duration(
        distance_xy(receiver_target, dribble_target),
        PLAYER_MAX_SPEED_CM_S,
        rng.uniform(2.2, 3.8),
    )
    dribble_end = sec_ceil(pass_end + second_duration)
    shot_target = random_goal_target(rng)
    shot_duration = safe_motion_duration(
        distance_xy(dribble_target, shot_target),
        BALL_MAX_SPEED_CM_S,
        rng.uniform(0.50, 0.82),
    )
    shot_end = sec_ceil(dribble_end + shot_duration)
    if shot_end + 0.5 >= DURATION_SEC:
        raise ValueError("template 3 timing exceeds episode duration")

    supports = support_targets(rng, dribble_target)
    events: List[Dict[str, Any]] = [
        {
            "event_id": "event_001",
            "type": "dribble",
            "actor": "Player_01",
            "start_t": 0.0,
            "end_t": sec(first_duration),
            "target_loc": p1_dribble_target,
            "ball_ahead_cm": 45.0,
        },
        {
            "event_id": "event_002",
            "type": "pass",
            "actor": "Player_01",
            "from": "Player_01",
            "to": "Player_02",
            "start_t": sec(first_duration),
            "end_t": sec(pass_end),
            "target_loc": pass_target,
            "arc_height_cm": 20.0,
        },
        move_event(
            "event_003",
            "Player_02",
            0.0,
            pass_end,
            receiver_target,
            tactical_role="timed_receive_run",
        ),
        {
            "event_id": "event_004",
            "type": "receive",
            "actor": "Player_02",
            "source_event": "event_002",
            "time": sec(pass_end),
        },
        {
            "event_id": "event_005",
            "type": "dribble",
            "actor": "Player_02",
            "start_t": sec(pass_end),
            "end_t": sec(dribble_end),
            "target_loc": dribble_target,
            "ball_ahead_cm": 45.0,
        },
        {
            "event_id": "event_006",
            "type": "shot",
            "actor": "Player_02",
            "start_t": sec(dribble_end),
            "end_t": sec(shot_end),
            "target_loc": shot_target,
            "arc_height_cm": 55.0,
        },
        move_event(
            "event_007",
            "Player_03",
            0.0,
            DURATION_SEC,
            supports["Player_03"],
            tactical_role="far_post_support",
        ),
        move_event(
            "event_008",
            "Player_04",
            0.0,
            DURATION_SEC,
            supports["Player_04"],
            tactical_role="rest_defence_anchor",
        ),
    ]
    add_standard_defence(
        events,
        rng,
        DURATION_SEC,
        primary_target="Player_01",
        secondary_target="Player_02",
        first_event_index=9,
    )
    return starts, events


TEMPLATES = {
    1: (TEMPLATE_NAMES[1], "4v4 单人带球射门", gen_template_1),
    2: (TEMPLATE_NAMES[2], "4v4 带球—传球—接球", gen_template_2),
    3: (TEMPLATE_NAMES[3], "4v4 传球—接球—带球—射门", gen_template_3),
}


def build_players(starts: Mapping[str, Sequence[float]]) -> Dict[str, Dict[str, Any]]:
    roles = {
        "Player_01": "ball_carrier",
        "Player_02": "receiver_support",
        "Player_03": "wide_support",
        "Player_04": "anchor",
        "Player_05": "primary_marker",
        "Player_06": "receiver_marker",
        "Player_07": "wide_marker",
        "Player_08": "cover_defender",
    }
    players: Dict[str, Dict[str, Any]] = {}
    for index, player_id in enumerate(PLAYER_IDS, start=1):
        players[player_id] = {
            "team": "A" if player_id in TEAM_A else "B",
            "role": roles[player_id],
            "track_id": index,
            "class_id": 0,
            "start_loc": list(starts[player_id]),
            "max_speed_cm_s": PLAYER_MAX_SPEED_CM_S,
        }
    return players


def build_episode_config(
    seed: int,
    template_id: int,
    rng: random.Random,
    attempt_index: int = 1,
    rng_seed: int | None = None,
) -> Dict[str, Any]:
    template_name, description, generator = TEMPLATES[template_id]
    starts, events = generator(rng)
    events.sort(
        key=lambda event: (
            float(event.get("start_t", event.get("time", 0.0))),
            str(event.get("event_id", "")),
        )
    )
    players = build_players(starts)

    initial_owner = "Player_01"
    initial_loc = starts[initial_owner]
    ball = {
        "object_id": "Ball_01",
        "track_id": 101,
        "class_id": 1,
        "initial_owner": initial_owner,
        "initial_loc": [
            round(initial_loc[0] + 45.0, 1),
            round(initial_loc[1], 1),
            BALL_Z_CM,
        ],
        "height_cm": BALL_Z_CM,
        "dribble_ahead_cm": 45.0,
    }

    episode_id = "episode_random_{:04d}_t{:d}".format(seed, template_id)
    return {
        "schema_version": "1.1",
        "project_root": PROJECT_ROOT.as_posix(),
        "episode_id": episode_id,
        "output_seq_id": episode_id,
        "description": "A3.4 4v4 outfield: {} (seed={}, template={})".format(
            description, seed, template_id
        ),
        "generator": {
            "version": SCRIPT_VERSION,
            "seed": int(seed),
            "requested_seed": int(seed),
            "template_id": int(template_id),
            "template_name": template_name,
            "generation_attempt": int(attempt_index),
            "rng_seed": int(rng_seed if rng_seed is not None else seed),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        "roster": {
            "format": "4v4_outfield_no_goalkeepers",
            "player_count": 8,
            "team_size_outfield": 4,
            "goalkeepers": [],
            "teams": {"A": list(TEAM_A), "B": list(TEAM_B)},
            "attack_direction": {"A": "+x", "B": "-x"},
        },
        "paths": {
            "base_render_config": "../../seq_test_0005.json",
            "output_trajectory_config": "./{}_a32.json".format(episode_id),
        },
        "timeline": {
            "fps": FPS,
            "duration_sec": DURATION_SEC,
            "frame_start": 0,
            "frame_end": TOTAL_FRAMES - 1,
            "time_to_frame_rule": "round",
            "interval_semantics": "[start_t, end_t)",
        },
        "court": {
            "x_min_cm": COURT_X_MIN,
            "x_max_cm": COURT_X_MAX,
            "y_min_cm": COURT_Y_MIN,
            "y_max_cm": COURT_Y_MAX,
            "player_z_cm": PLAYER_Z_CM,
            "ball_z_cm": BALL_Z_CM,
        },
        "compiler_defaults": {
            "player_interpolation": "smoothstep_dense",
            "dense_keyframe_interval_frames": 1,
            "player_max_speed_cm_s": PLAYER_MAX_SPEED_CM_S,
            "ball_max_speed_cm_s": BALL_MAX_SPEED_CM_S,
            "dribble_ball_ahead_cm": 45.0,
            "pass_arc_height_cm": 20.0,
            "shot_arc_height_cm": 55.0,
            "defender_default_follow_distance_cm": 180.0,
            "defender_default_side_offset_cm": 0.0,
            "defender_follow_speed_cm_s": 500.0,
            "defender_response_alpha": 0.22,
            "defender_response_time_sec": 0.38,
            "defender_max_acceleration_cm_s2": 850.0,
            "defender_lookahead_frames": 5,
            "defender_avoidance_radius_cm": 140.0,
            "defender_avoidance_weight": 0.7,
            "court_player_margin_cm": 35.0,
        },
        "movement_optimization": {
            "version": "ROLE_BASED_4V4_MOVEMENT_V2",
            "attack_direction": "+x",
            "minimum_start_spacing_cm": MIN_START_SPACING_CM,
            "defender_model": "goal_side_predictive_acceleration_limited_pursuit_v2",
            "off_ball_model": "lane_support_and_anchor_v1",
        },
        "players": players,
        "ball": ball,
        "event_rules": {
            "supported_types": [
                "hold",
                "move",
                "dribble",
                "pass",
                "receive",
                "defend_follow",
                "shot",
            ],
            "require_pass_receive_pair": True,
            "pass_receive_time_tolerance_sec": 0.0001,
            "allow_same_actor_boundary_touch": True,
            "require_possession_for_dribble": True,
            "require_possession_for_pass": True,
            "require_possession_for_shot": True,
            "expected_player_count": 8,
            "expected_team_size": 4,
            "minimum_initial_player_distance_cm": MIN_START_SPACING_CM,
        },
        "events": events,
    }


def validate_episode(config_path: Path, output_dir: Path | None = None) -> Dict[str, Any]:
    command: List[str] = [
        sys.executable,
        str(SCRIPT_DIR / "10_validate_episode.py"),
        "--config",
        str(config_path),
    ]
    if output_dir is not None:
        command.extend(["--output-dir", str(output_dir)])
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except OSError as exc:
        return {"rc": 2, "has_error": True, "out": "", "err": str(exc)}
    return {
        "rc": int(completed.returncode),
        "has_error": completed.returncode != 0,
        "out": completed.stdout or "",
        "err": completed.stderr or "",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A3.4 4v4 随机比赛回合生成器")
    parser.add_argument("--template", type=int, default=1, choices=sorted(TEMPLATES))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--smoke-test", action="store_true", help="运行 seed 1..5")
    parser.add_argument("--batch", type=int, default=None, help="批量生成 N 个")
    parser.add_argument("--skip-validator", action="store_true")
    parser.add_argument(
        "--validator-output-dir",
        type=str,
        default=None,
        help="候选事件验证报告目录；相对路径以 code/ 为基准",
    )
    parser.add_argument("--max-attempts", type=int, default=10)
    parser.add_argument(
        "--attempt-index",
        type=int,
        default=1,
        help="确定性重试起始序号，默认 1",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(
        args.output_dir or SCRIPT_DIR / "configs" / "events" / "generated"
    )
    if not output_dir.is_absolute():
        output_dir = (SCRIPT_DIR / output_dir).resolve()
    validator_output_dir = None
    if args.validator_output_dir:
        validator_output_dir = Path(args.validator_output_dir)
        if not validator_output_dir.is_absolute():
            validator_output_dir = (SCRIPT_DIR / validator_output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        "[A3.4] {} template={} players=8 format=4v4_no_goalkeepers".format(
            SCRIPT_VERSION, args.template
        )
    )

    if args.smoke_test:
        seeds_to_run = [1, 2, 3, 4, 5]
    elif args.batch is not None:
        seeds_to_run = [args.seed + index for index in range(args.batch)]
    else:
        seeds_to_run = [args.seed]

    if args.max_attempts <= 0 or args.attempt_index <= 0:
        print("[ERROR] max-attempts/attempt-index 必须大于 0", file=sys.stderr)
        return 2
    if any(seed < 0 for seed in seeds_to_run):
        print("[ERROR] seed 必须为非负整数", file=sys.stderr)
        return 2
    if args.batch is not None and args.batch <= 0:
        print("[ERROR] --batch 必须大于 0", file=sys.stderr)
        return 2

    accepted: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    total_attempts = 0

    for seed in seeds_to_run:
        final_path: Path | None = None
        last_error: str | None = None
        for local_attempt in range(args.max_attempts):
            attempt_index = args.attempt_index + local_attempt
            total_attempts += 1
            rng_seed = deterministic_rng_seed(seed, args.template, attempt_index)
            rng = random.Random(rng_seed)
            try:
                config = build_episode_config(
                    seed,
                    args.template,
                    rng,
                    attempt_index=attempt_index,
                    rng_seed=rng_seed,
                )
            except Exception as exc:
                last_error = "generation: {}: {}".format(type(exc).__name__, exc)
                continue

            final_path = output_dir / "{}.json".format(config["episode_id"])
            candidate_path = final_path.with_name(
                final_path.name + ".candidate.{}".format(os.getpid())
            )
            atomic_write_json(candidate_path, config)

            if not args.skip_validator:
                validation = validate_episode(candidate_path, validator_output_dir)
                if validation["has_error"]:
                    last_error = "validator rc={}: {}".format(
                        validation["rc"],
                        (
                            validation.get("err")
                            or validation.get("out")
                            or "validation failed"
                        ).strip()[-1200:],
                    )
                    try:
                        candidate_path.unlink()
                    except OSError:
                        pass
                    continue

            os.replace(str(candidate_path), str(final_path))
            accepted.append(
                {
                    "seed": seed,
                    "path": str(final_path),
                    "attempts": local_attempt + 1,
                    "generation_attempt": attempt_index,
                    "rng_seed": rng_seed,
                }
            )
            break
        else:
            if final_path is not None:
                for candidate in final_path.parent.glob(final_path.name + ".candidate.*"):
                    try:
                        candidate.unlink()
                    except OSError:
                        pass
            failures.append({"seed": seed, "error": last_error or "max attempts reached"})

    requested = len(seeds_to_run)
    accepted_count = len(accepted)
    rejected_attempts = max(0, total_attempts - accepted_count)
    print("  Accepted episodes: {}/{}".format(accepted_count, requested))
    print("  Total generation attempts: {}".format(total_attempts))
    print("  Rejected attempts: {}".format(rejected_attempts))
    if accepted_count:
        print(
            "  Episode completion rate: {:.1f}%".format(
                100.0 * accepted_count / requested
            )
        )
        print(
            "  Attempt acceptance rate: {:.1f}%".format(
                100.0 * accepted_count / total_attempts
            )
        )
        print(
            "  Avg attempts per accepted episode: {:.2f}".format(
                total_attempts / accepted_count
            )
        )

    for item in accepted:
        path = Path(item["path"])
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        print(
            "    Seed {} attempt={} rng_seed={} -> {} SHA256={}".format(
                item["seed"],
                item["generation_attempt"],
                item["rng_seed"],
                path,
                digest,
            )
        )

    for item in failures:
        print(
            "    [REJECTED] Seed {}: {}".format(item["seed"], item["error"]),
            file=sys.stderr,
        )

    return 0 if accepted_count == requested else 1


if __name__ == "__main__":
    raise SystemExit(main())
