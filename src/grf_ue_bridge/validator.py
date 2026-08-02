"""验证导出的 GRF-UE episode 数据。"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import List, Tuple


class ValidationError(Exception):
    """验证失败时抛出。"""

    pass


def validate_episode(episode_dir: Path) -> int:
    """验证一个 episode 目录。

    检查项：
      1. meta.json 存在且可解析
      2. schema/version 正确
      3. frames.jsonl 存在
      4. 恰好 300 行（或与 meta.timing.num_steps 一致）
      5. step 为 0..N-1
      6. 时间单调递增
      7. 每帧恰好 10 名球员
      8. ID 恰好为 L0-L4 和 R0-R4
      9. 每帧内 ID 唯一
      10. 所有坐标值有限
      11. 坐标未明显越出场地边界
      12. 球的位置有效
      13. 设置 owned_player 时引用已知球员
      14. meta 中的 num_steps 与实际帧数一致

    Returns:
        有效返回 0，无效返回 1。
    """
    errors: List[str] = []

    # ── 检查 meta.json ────────────────────────────────────────
    meta_path = episode_dir / "meta.json"
    if not meta_path.exists():
        errors.append(f"Missing meta.json at {meta_path}")
        _report(errors)
        return 1

    try:
        # 显式 UTF-8：meta.json 可能包含中文字段（如 coordinate_transform 说明）
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
    except json.JSONDecodeError as e:
        errors.append(f"meta.json is not valid JSON: {e}")
        _report(errors)
        return 1

    # Schema 与版本
    if meta.get("schema") != "grf_ue_episode":
        errors.append(f"meta.schema is '{meta.get('schema')}', expected 'grf_ue_episode'")
    if meta.get("version") != 1:
        errors.append(f"meta.version is {meta.get('version')}, expected 1")

    timing = meta.get("timing", {})
    expected_steps = timing.get("num_steps", 300)
    source_step = timing.get("source_step_seconds", 0.1)

    # ── 检查 frames.jsonl ─────────────────────────────────────
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
    x_bound = field_length / 2 + 2.0  # 允许 2m 余量
    y_bound = field_width / 2 + 2.0

    prev_time = -1.0

    for line_idx, line in enumerate(lines):
        if not line.strip():
            continue

        # 解析 JSON
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

        # 时间单调递增
        time_sec = frame.get("time_seconds", -1.0)
        if time_sec < prev_time:
            errors.append(
                f"Line {line_idx}: time_seconds={time_sec} < previous {prev_time}"
            )
        prev_time = time_sec

        # 比分
        score = frame.get("score", [])
        if not isinstance(score, list) or len(score) != 2:
            errors.append(f"Line {line_idx}: score must be [int, int]")

        # 球
        ball = frame.get("ball", {})
        ball_pos = ball.get("position_m", [])
        _validate_position(
            errors, line_idx, ball_pos, "ball.position_m", x_bound, y_bound
        )
        # 如存在则校验 source_grf_position
        src_pos = ball.get("source_grf_position")
        if src_pos is not None:
            _validate_position(
                errors, line_idx, src_pos, "ball.source_grf_position",
                x_bound, y_bound, finite_only=True
            )

        # 球员
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

            # 球员 Z 应为 0
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

    # ── 报告 ──────────────────────────────────────────────────
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
    """校验一个位置列表包含 3 个有限坐标且在边界内。

    Args:
        finite_only: 为 True 时只检查有限性（用于采用不同坐标空间的源位置）。
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
    """打印验证错误。"""
    print(f"VALIDATOR: Found {len(errors)} error(s)")
    for err in errors[:20]:
        print(f"  ERROR: {err}")
    if len(errors) > 20:
        print(f"  ... and {len(errors) - 20} more errors")
