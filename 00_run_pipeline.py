#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FutsalMOT single-seed Windows pipeline for 4v4 outfield (8 players).

Version:
    RUN_SEED_8P_V4

For one requested seed/template, this script deterministically retries complete
candidate generation until one candidate passes the event and dense trajectory
validators, or ``--max-attempts`` is exhausted.

Pipeline per attempt:
    A3.4 random event config
    -> A3.1 event validation
    -> A3.2 dense trajectory compile
    -> A3.3 yaw/action/ball enhancement
    -> A2.5a dense trajectory validation

After one candidate is accepted:
    -> A3.3c event annotations
    -> atomic configs/pipeline_current.json pointer for Unreal scripts

No trajectory-validation ERROR is silently accepted by default.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

SCRIPT_VERSION = "RUN_SEED_8P_V4"
SCRIPT_DIR = Path(__file__).resolve().parent
CURRENT_RUN_POINTER = SCRIPT_DIR / "configs" / "pipeline_current.json"
TEMPLATE_NAMES = {
    1: "solo_dribble_shot_4v4",
    2: "dribble_pass_receive_4v4",
    3: "pass_receive_dribble_shot_4v4",
}


class PipelineError(RuntimeError):
    """Fatal pipeline/program error."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.{}".format(os.getpid()))
    try:
        with tmp.open("w", encoding="utf-8", newline="\n") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
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


def require_file(path: Path, label: str) -> None:
    if not path.is_file() or path.stat().st_size <= 0:
        raise PipelineError("{} 未生成或为空：{}".format(label, path))


def format_command(cmd: Sequence[object]) -> str:
    parts: List[str] = []
    for value in cmd:
        text = str(value)
        if any(ch.isspace() for ch in text) or '"' in text:
            text = '"{}"'.format(text.replace('"', '\\"'))
        parts.append(text)
    return " ".join(parts)


def tail_lines(text: str, count: int = 20) -> List[str]:
    return [line.rstrip() for line in text.splitlines() if line.strip()][-count:]


def run_step(
    label: str,
    cmd: Sequence[object],
    *,
    timeout: int,
    log_dir: Path,
) -> Tuple[int, str, str, float]:
    print("\n[RUN] {}".format(label))
    print("  {}".format(format_command(cmd)))
    start = time.time()
    try:
        completed = subprocess.run(
            [str(item) for item in cmd],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
        rc = int(completed.returncode)
        stdout = completed.stdout or ""
        stderr = completed.stderr or ""
    except subprocess.TimeoutExpired as exc:
        rc = 124
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
    except OSError as exc:
        rc = 127
        stdout = ""
        stderr = str(exc)

    elapsed = time.time() - start
    log_dir.mkdir(parents=True, exist_ok=True)
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in label)
    (log_dir / "{}_stdout.log".format(safe)).write_text(stdout, encoding="utf-8")
    (log_dir / "{}_stderr.log".format(safe)).write_text(stderr, encoding="utf-8")

    if rc == 0:
        print("  OK ({:.1f}s)".format(elapsed))
    else:
        print("  FAILED (rc={}, {:.1f}s)".format(rc, elapsed))
        if stdout.strip():
            print("  --- stdout tail ---")
            for line in tail_lines(stdout):
                print("    {}".format(line))
        if stderr.strip():
            print("  --- stderr tail ---", file=sys.stderr)
            for line in tail_lines(stderr):
                print("    {}".format(line), file=sys.stderr)
    return rc, stdout, stderr, elapsed


def resolve_output_dir(raw: Optional[str], episode_id: str) -> Path:
    if raw:
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = SCRIPT_DIR / path
        return path.resolve()
    return (SCRIPT_DIR / "_agent_test_outputs" / "pipeline_{}".format(episode_id)).resolve()


def snapshot_artifacts(debug_dir: Path, paths: Sequence[Path]) -> None:
    debug_dir.mkdir(parents=True, exist_ok=True)
    for path in paths:
        if path.is_file():
            shutil.copy2(str(path), str(debug_dir / path.name))


def remove_artifacts(paths: Sequence[Path]) -> None:
    for path in paths:
        try:
            if path.is_file():
                path.unlink()
        except OSError:
            pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FutsalMOT 单 Seed 数据集管线")
    parser.add_argument("--seed", type=int, required=True, help="非负随机种子")
    parser.add_argument(
        "--template",
        type=int,
        default=1,
        choices=sorted(TEMPLATE_NAMES),
        help="模板 ID",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="日志、验证报告和事件标注目录；相对路径以 code/ 为基准",
    )
    parser.add_argument(
        "--strict-warnings",
        action="store_true",
        help="将事件/轨迹验证 WARNING 视为候选失败",
    )
    parser.add_argument(
        "--skip-trajectory-validation",
        "--skip-validation",
        dest="skip_trajectory_validation",
        action="store_true",
        help="跳过轨迹验证；只生成诊断输出，不更新当前 UE 指针",
    )
    parser.add_argument(
        "--allow-trajectory-errors",
        action="store_true",
        help="允许轨迹 ERROR 继续生成诊断输出；不更新当前 UE 指针",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="每个 Windows 子步骤超时秒数",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=10,
        help="完整候选管线的确定性最大尝试次数",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.seed < 0 or args.timeout <= 0 or args.max_attempts <= 0:
        print("[ERROR] seed 必须非负，timeout/max-attempts 必须大于 0", file=sys.stderr)
        return 2

    seed = args.seed
    template = args.template
    template_name = TEMPLATE_NAMES[template]
    episode_id = "episode_random_{:04d}_t{:d}".format(seed, template)
    output_dir = resolve_output_dir(args.output_dir, episode_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    generated_dir = SCRIPT_DIR / "configs" / "events" / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)
    event_config = generated_dir / "{}.json".format(episode_id)
    a32_config = generated_dir / "{}_a32.json".format(episode_id)
    a33_config = generated_dir / "{}_a33.json".format(episode_id)
    artifacts = [event_config, a32_config, a33_config]

    print("=" * 76)
    print("FutsalMOT Pipeline {}".format(SCRIPT_VERSION))
    print("seed={} template={} ({}) players=8 format=4v4_no_goalkeepers".format(seed, template, template_name))
    print("episode_id={} max_attempts={}".format(episode_id, args.max_attempts))
    print("output_dir={}".format(output_dir))
    print("=" * 76)

    attempts: List[Dict[str, Any]] = []
    accepted_attempt: Optional[int] = None
    trajectory_validation_passed = False
    diagnostic_only = False

    try:
        for attempt_index in range(1, args.max_attempts + 1):
            print("\n" + "#" * 76)
            print("[CANDIDATE] attempt {}/{}".format(attempt_index, args.max_attempts))
            print("#" * 76)
            attempt_dir = output_dir / "attempt_{:02d}".format(attempt_index)
            logs_dir = attempt_dir / "logs"
            reports_dir = attempt_dir / "reports"
            attempt_steps: List[Dict[str, Any]] = []
            attempt_record: Dict[str, Any] = {
                "attempt_index": attempt_index,
                "status": "running",
                "steps": attempt_steps,
            }
            attempts.append(attempt_record)

            def step(label: str, cmd: Sequence[object]) -> int:
                rc, _, _, elapsed = run_step(
                    label, cmd, timeout=args.timeout, log_dir=logs_dir
                )
                attempt_steps.append(
                    {
                        "label": label,
                        "command": [str(item) for item in cmd],
                        "returncode": rc,
                        "elapsed_seconds": round(elapsed, 3),
                    }
                )
                return rc

            remove_artifacts(artifacts)

            rc = step(
                "01_A3_4_generate",
                [
                    sys.executable,
                    SCRIPT_DIR / "11_generate_random_episode.py",
                    "--seed",
                    seed,
                    "--template",
                    template,
                    "--attempt-index",
                    attempt_index,
                    "--max-attempts",
                    1,
                    "--skip-validator",
                ],
            )
            if rc != 0:
                attempt_record["status"] = "rejected_generation"
                if rc >= 2:
                    raise PipelineError("A3.4 生成器发生程序/参数错误，rc={}".format(rc))
                continue
            require_file(event_config, "A3.4 事件配置")

            event_cmd: List[object] = [
                sys.executable,
                SCRIPT_DIR / "10_validate_episode.py",
                "--config",
                event_config,
                "--output-dir",
                reports_dir / "episode",
            ]
            if args.strict_warnings:
                event_cmd.append("--strict-warnings")
            rc = step("02_A3_1_validate_episode", event_cmd)
            if rc != 0:
                snapshot_artifacts(attempt_dir / "rejected_artifacts", artifacts)
                remove_artifacts(artifacts)
                attempt_record["status"] = "rejected_episode_validation"
                if rc >= 2:
                    raise PipelineError("事件验证器发生致命错误，rc={}".format(rc))
                continue

            rc = step(
                "03_A3_2_compile",
                [
                    sys.executable,
                    SCRIPT_DIR / "12_compile_trajectory.py",
                    "--config",
                    event_config,
                    "--output",
                    a32_config,
                    "--skip-episode-validation",
                    "--skip-trajectory-validation",
                    "--no-backup",
                ],
            )
            if rc != 0:
                snapshot_artifacts(attempt_dir / "failed_artifacts", artifacts)
                raise PipelineError("A3.2 编译失败，rc={}".format(rc))
            require_file(a32_config, "A3.2 轨迹配置")

            rc = step(
                "04_A3_3_enhance",
                [
                    sys.executable,
                    SCRIPT_DIR / "13_enhance_trajectory.py",
                    "--config",
                    event_config,
                    "--compiled-config",
                    a32_config,
                    "--output",
                    a33_config,
                    "--skip-trajectory-validation",
                    "--seq-id",
                    episode_id,
                    "--no-backup",
                ],
            )
            if rc != 0:
                snapshot_artifacts(attempt_dir / "failed_artifacts", artifacts)
                raise PipelineError("A3.3 增强失败，rc={}".format(rc))
            require_file(a33_config, "A3.3 增强轨迹配置")

            if args.skip_trajectory_validation:
                diagnostic_only = True
                attempt_record["status"] = "accepted_diagnostic_validation_skipped"
                accepted_attempt = attempt_index
                break

            trajectory_cmd: List[object] = [
                sys.executable,
                SCRIPT_DIR / "14_validate_trajectory.py",
                "--config",
                a33_config,
                "--output-dir",
                reports_dir / "trajectory",
            ]
            if args.strict_warnings:
                trajectory_cmd.append("--strict-warnings")
            rc = step("05_A2_5_validate_trajectory", trajectory_cmd)
            if rc == 0:
                trajectory_validation_passed = True
                accepted_attempt = attempt_index
                attempt_record["status"] = "accepted"
                break
            if rc >= 2:
                snapshot_artifacts(attempt_dir / "failed_artifacts", artifacts)
                raise PipelineError("轨迹验证器发生致命错误，rc={}".format(rc))
            if args.allow_trajectory_errors:
                diagnostic_only = True
                accepted_attempt = attempt_index
                attempt_record["status"] = "accepted_diagnostic_with_trajectory_errors"
                break

            snapshot_artifacts(attempt_dir / "rejected_artifacts", artifacts)
            remove_artifacts(artifacts)
            attempt_record["status"] = "rejected_trajectory_validation"

        if accepted_attempt is None:
            raise PipelineError(
                "{} 次确定性候选均未通过完整验证；查看各 attempt 的报告。".format(
                    args.max_attempts
                )
            )

        require_file(event_config, "最终事件配置")
        require_file(a32_config, "最终 A3.2 配置")
        require_file(a33_config, "最终 A3.3 配置")

        event_annotations_dir = output_dir / "event_annotations"
        annotation_logs = output_dir / "final_logs"
        rc, _, _, elapsed = run_step(
            "06_A3_3C_event_annotations",
            [
                sys.executable,
                SCRIPT_DIR / "31_generate_event_annotations.py",
                "--episode-config",
                event_config,
                "--a3-config",
                a33_config,
                "--output-dir",
                event_annotations_dir,
                "--overwrite",
            ],
            timeout=args.timeout,
            log_dir=annotation_logs,
        )
        final_step = {
            "label": "06_A3_3C_event_annotations",
            "returncode": rc,
            "elapsed_seconds": round(elapsed, 3),
        }
        if rc != 0:
            raise PipelineError("A3.3c 事件标注失败，rc={}".format(rc))

        expected_events = event_annotations_dir / "events_{}.json".format(episode_id)
        expected_states = event_annotations_dir / "frame_states_{}.jsonl".format(episode_id)
        expected_report = event_annotations_dir / "event_annotation_report_{}.json".format(episode_id)
        for path, label in (
            (expected_events, "A3.3c events JSON"),
            (expected_states, "A3.3c frame states JSONL"),
            (expected_report, "A3.3c report"),
        ):
            require_file(path, label)

        with event_config.open("r", encoding="utf-8-sig") as f:
            event_data = json.load(f)
        generator_meta = event_data.get("generator", {})
        players_meta = event_data.get("players", {})
        team_counts: Dict[str, int] = {}
        if isinstance(players_meta, dict):
            for player_meta in players_meta.values():
                if isinstance(player_meta, dict):
                    team_id = str(player_meta.get("team", "unknown"))
                    team_counts[team_id] = team_counts.get(team_id, 0) + 1
        run_state: Dict[str, Any] = {
            "schema_version": "1.0",
            "pipeline_version": SCRIPT_VERSION,
            "generated_at_utc": utc_now(),
            "seed": seed,
            "template_id": template,
            "template_name": template_name,
            "episode_id": episode_id,
            "seq_id": episode_id,
            "player_count": len(players_meta) if isinstance(players_meta, dict) else None,
            "team_counts": dict(sorted(team_counts.items())),
            "roster": event_data.get("roster"),
            "generation_attempt": accepted_attempt,
            "rng_seed": generator_meta.get("rng_seed"),
            "status": "diagnostic_only" if diagnostic_only else "windows_complete",
            "trajectory_validation_passed": trajectory_validation_passed,
            "paths": {
                "event_config": event_config.resolve().as_posix(),
                "a3_2_config": a32_config.resolve().as_posix(),
                "a3_3_config": a33_config.resolve().as_posix(),
                "event_annotation_dir": event_annotations_dir.resolve().as_posix(),
                "events_json": expected_events.resolve().as_posix(),
                "frame_states_jsonl": expected_states.resolve().as_posix(),
                "event_annotation_report": expected_report.resolve().as_posix(),
                "pipeline_output_dir": output_dir.resolve().as_posix(),
            },
            "sha256": {
                "event_config": sha256_file(event_config),
                "a3_2_config": sha256_file(a32_config),
                "a3_3_config": sha256_file(a33_config),
                "events_json": sha256_file(expected_events),
                "frame_states_jsonl": sha256_file(expected_states),
                "event_annotation_report": sha256_file(expected_report),
            },
            "attempts": attempts,
            "final_step": final_step,
        }
        atomic_write_json(output_dir / "pipeline_run_report.json", run_state)
        if trajectory_validation_passed and not diagnostic_only:
            atomic_write_json(CURRENT_RUN_POINTER, run_state)
        else:
            print(
                "[WARNING] 诊断输出未更新 configs/pipeline_current.json。",
                file=sys.stderr,
            )

        project_root = SCRIPT_DIR.parents[2]
        annotation_json = (
            project_root
            / "Saved"
            / "FutsalMOT"
            / "annotations"
            / "objects_bbox_2d_clean_{}.json".format(episode_id)
        )

        print("\n" + "=" * 76)
        print("[OK] Windows pipeline complete")
        print("Episode: {}".format(episode_id))
        print("Accepted attempt: {}".format(accepted_attempt))
        print("Event config: {}".format(event_config))
        print("A3.3 config: {}".format(a33_config))
        print("Event annotations: {}".format(event_annotations_dir))
        if trajectory_validation_passed and not diagnostic_only:
            print("Current-run pointer: {}".format(CURRENT_RUN_POINTER))
            print("\nNext in Unreal Editor:")
            print('  py "{}"'.format((SCRIPT_DIR / "21_preflight.py").resolve().as_posix()))
            print('  py "{}"'.format((SCRIPT_DIR / "20_build_sequences.py").resolve().as_posix()))
            print("\nAfter MRQ render, run:")
            print(
                '  py 30_convert_and_check.py --annotation "{}"'.format(
                    annotation_json.as_posix()
                )
            )
        else:
            print("Current-run pointer: NOT UPDATED")
            print("Do not treat this diagnostic output as a validated UE dataset.")
        print("=" * 76)
        return 0

    except PipelineError as exc:
        failure = {
            "schema_version": "1.0",
            "pipeline_version": SCRIPT_VERSION,
            "generated_at_utc": utc_now(),
            "seed": seed,
            "template_id": template,
            "template_name": template_name,
            "episode_id": episode_id,
            "status": "failed",
            "error": str(exc),
            "attempts": attempts,
        }
        atomic_write_json(output_dir / "pipeline_run_report.json", failure)
        print("\n" + "=" * 76, file=sys.stderr)
        print("PIPELINE FAILED", file=sys.stderr)
        print("[ERROR] {}".format(exc), file=sys.stderr)
        print("Report: {}".format(output_dir / "pipeline_run_report.json"), file=sys.stderr)
        print("=" * 76, file=sys.stderr)
        return 1
    except Exception as exc:
        print("[UNEXPECTED ERROR] {}: {}".format(type(exc).__name__, exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
