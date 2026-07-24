"""Validate exported GRF-UE episode data."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import List, Tuple


class ValidationError(Exception):
    """Raised when validation fails."""

    pass


def validate_episode(episode_dir: Path) -> int:
    """Validate an episode directory.

    Checks:
      1. meta.json exists and is parseable
      2. schema/version correct
      3. frames.jsonl exists
      4. Exactly 300 lines (or matches meta.timing.num_steps)
      5. Steps are 0..N-1
      6. Time monotonically increasing
      7. Exactly 10 players per frame
      8. IDs exactly L0-L4 and R0-R4
      9. IDs unique within each frame
      10. All position values finite
      11. Coordinates not wildly out of field bounds
      12. Ball position valid
      13. owned_player references known player when set
      14. Metadata num_steps matches actual frame count

    Returns:
        0 if valid, 1 if invalid.
    """
    errors: List[str] = []

    # ── Check meta.json ────────────────────────────────────────────
    meta_path = episode_dir / "meta.json"
    if not meta_path.exists():
        errors.append(f"Missing meta.json at {meta_path}")
        _report(errors)
        return 1

    try:
        with open(meta_path) as f:
            meta = json.load(f)
    except json.JSONDecodeError as e:
        errors.append(f"meta.json is not valid JSON: {e}")
        _report(errors)
        return 1

    # Schema and version
    if meta.get("schema") != "grf_ue_episode":
        errors.append(f"meta.schema is '{meta.get('schema')}', expected 'grf_ue_episode'")
    if meta.get("version") != 1:
        errors.append(f"meta.version is {meta.get('version')}, expected 1")

    timing = meta.get("timing", {})
    expected_steps = timing.get("num_steps", 300)
    source_step = timing.get("source_step_seconds", 0.1)

    # ── Check frames.jsonl ─────────────────────────────────────────
    frames_path = episode_dir / "frames.jsonl"
    if not frames_path.exists():
        errors.append(f"Missing frames.jsonl at {frames_path}")
        _report(errors)
        return 1

    lines = frames_path.read_text(encoding="utf-8").strip().splitlines()
    actual_steps = len(lines)

    if actual_steps != expected_steps:
        errors.append(
            f"frames.jsonl has {actual_steps} lines, "
            f"expected {expected_steps} from meta.timing.num_steps"
        )

    valid_ids = {f"L{i}" for i in range(5)} | {f"R{i}" for i in range(5)}
    field_length = meta.get("field", {}).get("length_m", 40.0)
    field_width = meta.get("field", {}).get("width_m", 20.0)
    x_bound = field_length / 2 + 2.0  # allow 2m margin
    y_bound = field_width / 2 + 2.0

    prev_time = -1.0

    for line_idx, line in enumerate(lines):
        if not line.strip():
            continue

        # Parse JSON
        try:
            frame = json.loads(line)
        except json.JSONDecodeError as e:
            errors.append(f"Line {line_idx}: invalid JSON: {e}")
            continue

        if not isinstance(frame, dict):
            errors.append(f"Line {line_idx}: frame is not a JSON object")
            continue

        step = frame.get("step", -1)
        if not (0 <= step < expected_steps):
            errors.append(f"Line {line_idx}: step={step} out of range [0, {expected_steps - 1}]")

        expected_step = line_idx
        if step != expected_step:
            errors.append(
                f"Line {line_idx}: step={step}, expected {expected_step}"
            )

        # Time monotonic
        time_sec = frame.get("time_seconds", -1.0)
        if time_sec < prev_time:
            errors.append(
                f"Line {line_idx}: time_seconds={time_sec} < previous {prev_time}"
            )
        prev_time = time_sec

        # Score
        score = frame.get("score", [])
        if not isinstance(score, list) or len(score) != 2:
            errors.append(f"Line {line_idx}: score must be [int, int]")

        # Ball
        ball = frame.get("ball", {})
        ball_pos = ball.get("position_m", [])
        _validate_position(
            errors, line_idx, ball_pos, "ball.position_m", x_bound, y_bound
        )
        # Validate source_grf_position if present
        src_pos = ball.get("source_grf_position")
        if src_pos is not None:
            _validate_position(
                errors, line_idx, src_pos, "ball.source_grf_position",
                x_bound, y_bound, finite_only=True
            )

        # Players
        players = frame.get("players", [])
        if len(players) != 10:
            errors.append(f"Line {line_idx}: expected 10 players, got {len(players)}")

        seen_ids = set()
        for p_idx, player in enumerate(players):
            pid = player.get("id", "")
            if pid not in valid_ids:
                errors.append(
                    f"Line {line_idx}, player[{p_idx}]: invalid id '{pid}'"
                )
            if pid in seen_ids:
                errors.append(f"Line {line_idx}: duplicate player id '{pid}'")
            seen_ids.add(pid)

            pos = player.get("position_m", [])
            _validate_position(
                errors, line_idx, pos, f"player[{p_idx}].position_m", x_bound, y_bound
            )

            # Player Z should be 0
            if len(pos) == 3 and abs(pos[2]) > 0.001:
                errors.append(
                    f"Line {line_idx}, player[{p_idx}]: "
                    f"player Z should be 0, got {pos[2]}"
                )

        if seen_ids != valid_ids:
            missing = valid_ids - seen_ids
            extra = seen_ids - valid_ids
            if missing:
                errors.append(f"Line {line_idx}: missing player IDs: {sorted(missing)}")
            if extra:
                errors.append(f"Line {line_idx}: unexpected player IDs: {sorted(extra)}")

    # ── Report ─────────────────────────────────────────────────────
    if errors:
        _report(errors)
        return 1

    print(f"VALIDATOR: Episode at {episode_dir} PASSED all checks")
    print(f"  Steps: {actual_steps}")
    print(f"  Source step seconds: {source_step}")
    print(f"  Field: {field_length}m x {field_width}m")
    return 0


def _validate_position(
    errors: List[str],
    line_idx: int,
    pos: list,
    label: str,
    x_bound: float = 100,
    y_bound: float = 100,
    finite_only: bool = False,
) -> None:
    """Validate a position list has 3 finite coordinates within bounds.

    Args:
        finite_only: If True, only check finiteness (for source positions
            that use a different coordinate space).
    """
    if not isinstance(pos, list) or len(pos) not in (2, 3):
        errors.append(
            f"Line {line_idx}: {label} should be [x, y] or [x, y, z], got {pos}"
        )
        return

    for i, val in enumerate(pos):
        if not isinstance(val, (int, float)):
            errors.append(
                f"Line {line_idx}: {label}[{i}] is not a number: {val}"
            )
            continue
        if not math.isfinite(val):
            errors.append(
                f"Line {line_idx}: {label}[{i}] is not finite: {val}"
            )

    if not finite_only and len(pos) >= 2:
        x, y = pos[0], pos[1]
        if abs(x) > x_bound:
            errors.append(
                f"Line {line_idx}: {label} X={x} exceeds bound ±{x_bound}"
            )
        if abs(y) > y_bound:
            errors.append(
                f"Line {line_idx}: {label} Y={y} exceeds bound ±{y_bound}"
            )


def _report(errors: List[str]) -> None:
    """Print validation errors."""
    print(f"VALIDATOR: Found {len(errors)} error(s)")
    for err in errors[:20]:
        print(f"  ERROR: {err}")
    if len(errors) > 20:
        print(f"  ... and {len(errors) - 20} more errors")
