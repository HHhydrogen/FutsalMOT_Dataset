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
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from futsalmot.core.hashing import sha256_file
from futsalmot.core.io import read_json, write_json_atomic
from futsalmot.core.paths import (
    CODE_DIR,
    CURRENT_RUN_POINTER,
    PIPELINE_CONFIG_PATH,
    PROJECT_ROOT,
    RUNS_DIR,
)
from futsalmot.core.process import run_logged_step
from futsalmot.pipeline.constants import TEMPLATE_NAMES, WINDOWS_PIPELINE_SCRIPTS

SCRIPT_VERSION = "RUN_SEED_8P_V4"
SCRIPT_DIR = CODE_DIR


class PipelineError(RuntimeError):
    """Fatal pipeline/program error."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def require_file(path: Path, label: str) -> None:
    if not path.is_file() or path.stat().st_size <= 0:
        raise PipelineError("{} 未生成或为空：{}".format(label, path))


def resolve_output_dir(raw: Optional[str], run_dir: Path) -> Path:
    if raw:
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = SCRIPT_DIR / path
        return path.resolve()
    return run_dir.resolve()


def load_pipeline_config(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise PipelineError("找不到总配置文件：{}".format(path))
    data = read_json(path)
    if not isinstance(data, dict):
        raise PipelineError("总配置文件顶层必须是 JSON object：{}".format(path))
    return data


def config_int(config: Dict[str, Any], key: str, default: int) -> int:
    value = config.get(key, default)
    if isinstance(value, bool):
        raise PipelineError("总配置字段 {} 必须是整数".format(key))
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise PipelineError("总配置字段 {} 必须是整数，当前={!r}".format(key, value)) from exc


def config_bool(config: Dict[str, Any], key: str, default: bool) -> bool:
    value = config.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    raise PipelineError("总配置字段 {} 必须是布尔值，当前={!r}".format(key, value))


def make_run_id(prefix: str, seed: int, template: int) -> str:
    safe_prefix = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in prefix).strip("_")
    if not safe_prefix:
        safe_prefix = "run"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return "{}_{}_seed{:04d}_t{}".format(safe_prefix, stamp, seed, template)


def unique_run_dir(run_id: str) -> Path:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    candidate = RUNS_DIR / run_id
    if not candidate.exists():
        return candidate
    for index in range(2, 1000):
        suffixed = RUNS_DIR / "{}_{:02d}".format(run_id, index)
        if not suffixed.exists():
            return suffixed
    raise PipelineError("无法创建唯一 run 目录：{}".format(run_id))


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
    parser.add_argument(
        "--config",
        type=Path,
        default=PIPELINE_CONFIG_PATH,
        help="总配置 JSON；默认 configs/pipeline_config.json",
    )
    parser.add_argument("--seed", type=int, default=None, help="覆盖总配置中的随机种子")
    parser.add_argument(
        "--template",
        type=int,
        default=None,
        choices=sorted(TEMPLATE_NAMES),
        help="覆盖总配置中的模板 ID",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="覆盖自动生成的唯一 run 目录名",
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
        default=None,
        help="覆盖总配置中的每个 Windows 子步骤超时秒数",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=None,
        help="覆盖总配置中的确定性最大尝试次数",
    )
    parser.add_argument(
        "--no-update-current-pointer",
        action="store_true",
        help="即使验证通过，也不更新 configs/pipeline_current.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config_path = args.config.expanduser()
        if not config_path.is_absolute():
            config_path = SCRIPT_DIR / config_path
        pipeline_config = load_pipeline_config(config_path.resolve())
        seed = args.seed if args.seed is not None else config_int(pipeline_config, "seed", 1)
        template = (
            args.template
            if args.template is not None
            else config_int(pipeline_config, "template_id", 1)
        )
        timeout = (
            args.timeout
            if args.timeout is not None
            else config_int(pipeline_config, "timeout_sec", 300)
        )
        max_attempts = (
            args.max_attempts
            if args.max_attempts is not None
            else config_int(pipeline_config, "max_attempts", 10)
        )
        strict_warnings = args.strict_warnings or config_bool(
            pipeline_config, "strict_warnings", False
        )
        skip_trajectory_validation = args.skip_trajectory_validation or config_bool(
            pipeline_config, "skip_trajectory_validation", False
        )
        allow_trajectory_errors = args.allow_trajectory_errors or config_bool(
            pipeline_config, "allow_trajectory_errors", False
        )
        update_current_pointer = (
            config_bool(pipeline_config, "update_current_pointer", True)
            and not args.no_update_current_pointer
        )
        run_id_prefix = str(pipeline_config.get("run_id_prefix", "run"))
    except PipelineError as exc:
        print("[ERROR] {}".format(exc), file=sys.stderr)
        return 2

    if seed < 0 or timeout <= 0 or max_attempts <= 0:
        print("[ERROR] seed 必须非负，timeout/max-attempts 必须大于 0", file=sys.stderr)
        return 2
    if template not in TEMPLATE_NAMES:
        print("[ERROR] template 必须是 {} 中之一".format(sorted(TEMPLATE_NAMES)), file=sys.stderr)
        return 2

    template_name = TEMPLATE_NAMES[template]
    episode_id = "episode_random_{:04d}_t{:d}".format(seed, template)
    run_id = args.run_id or make_run_id(run_id_prefix, seed, template)
    run_dir = unique_run_dir(run_id)
    output_dir = resolve_output_dir(args.output_dir, run_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generated_dir = run_dir
    generated_dir.mkdir(parents=True, exist_ok=True)
    event_config = generated_dir / "{}.json".format(episode_id)
    a32_config = generated_dir / "{}_a32.json".format(episode_id)
    a33_config = generated_dir / "{}_a33.json".format(episode_id)
    artifacts = [event_config, a32_config, a33_config]

    print("=" * 76)
    print("FutsalMOT Pipeline {}".format(SCRIPT_VERSION))
    print("config={}".format(config_path.resolve()))
    print("run_id={}".format(run_dir.name))
    print("run_dir={}".format(run_dir))
    print("seed={} template={} ({}) players=8 format=4v4_no_goalkeepers".format(seed, template, template_name))
    print("episode_id={} max_attempts={}".format(episode_id, max_attempts))
    print("output_dir={}".format(output_dir))
    print("=" * 76)

    attempts: List[Dict[str, Any]] = []
    accepted_attempt: Optional[int] = None
    trajectory_validation_passed = False
    diagnostic_only = False

    try:
        for attempt_index in range(1, max_attempts + 1):
            print("\n" + "#" * 76)
            print("[CANDIDATE] attempt {}/{}".format(attempt_index, max_attempts))
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
                result = run_logged_step(
                    label, cmd, timeout=timeout, log_dir=logs_dir
                )
                attempt_steps.append(
                    {
                        "label": label,
                        "command": [str(item) for item in cmd],
                        "returncode": result.returncode,
                        "elapsed_seconds": round(result.elapsed_seconds, 3),
                    }
                )
                return result.returncode

            remove_artifacts(artifacts)

            rc = step(
                "01_A3_4_generate",
                [
                    sys.executable,
                    SCRIPT_DIR / WINDOWS_PIPELINE_SCRIPTS["generate_episode"],
                    "--seed",
                    seed,
                    "--template",
                    template,
                    "--output-dir",
                    generated_dir,
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
                SCRIPT_DIR / WINDOWS_PIPELINE_SCRIPTS["validate_episode"],
                "--config",
                event_config,
                "--output-dir",
                reports_dir / "episode",
            ]
            if strict_warnings:
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
                    SCRIPT_DIR / WINDOWS_PIPELINE_SCRIPTS["compile_trajectory"],
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
                    SCRIPT_DIR / WINDOWS_PIPELINE_SCRIPTS["enhance_trajectory"],
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

            if skip_trajectory_validation:
                diagnostic_only = True
                attempt_record["status"] = "accepted_diagnostic_validation_skipped"
                accepted_attempt = attempt_index
                break

            trajectory_cmd: List[object] = [
                sys.executable,
                SCRIPT_DIR / WINDOWS_PIPELINE_SCRIPTS["validate_trajectory"],
                "--config",
                a33_config,
                "--output-dir",
                reports_dir / "trajectory",
            ]
            if strict_warnings:
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
            if allow_trajectory_errors:
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
                    max_attempts
                )
            )

        require_file(event_config, "最终事件配置")
        require_file(a32_config, "最终 A3.2 配置")
        require_file(a33_config, "最终 A3.3 配置")

        event_annotations_dir = output_dir / "event_annotations"
        annotation_logs = output_dir / "final_logs"
        annotation_result = run_logged_step(
            "06_A3_3C_event_annotations",
            [
                sys.executable,
                SCRIPT_DIR / WINDOWS_PIPELINE_SCRIPTS["event_annotations"],
                "--episode-config",
                event_config,
                "--a3-config",
                a33_config,
                "--output-dir",
                event_annotations_dir,
                "--overwrite",
            ],
            timeout=timeout,
            log_dir=annotation_logs,
        )
        final_step = {
            "label": "06_A3_3C_event_annotations",
            "returncode": annotation_result.returncode,
            "elapsed_seconds": round(annotation_result.elapsed_seconds, 3),
        }
        if annotation_result.returncode != 0:
            raise PipelineError("A3.3c 事件标注失败，rc={}".format(annotation_result.returncode))

        expected_events = event_annotations_dir / "events_{}.json".format(episode_id)
        expected_states = event_annotations_dir / "frame_states_{}.jsonl".format(episode_id)
        expected_report = event_annotations_dir / "event_annotation_report_{}.json".format(episode_id)
        for path, label in (
            (expected_events, "A3.3c events JSON"),
            (expected_states, "A3.3c frame states JSONL"),
            (expected_report, "A3.3c report"),
        ):
            require_file(path, label)

        event_data = read_json(event_config)
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
            "pipeline_config_path": config_path.resolve().as_posix(),
            "pipeline_config": pipeline_config,
            "run_id": run_dir.name,
            "run_dir": run_dir.resolve().as_posix(),
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
        write_json_atomic(output_dir / "pipeline_run_report.json", run_state)
        if trajectory_validation_passed and not diagnostic_only and update_current_pointer:
            write_json_atomic(CURRENT_RUN_POINTER, run_state)
        else:
            print(
                "[WARNING] 本次运行未更新 configs/pipeline_current.json。",
                file=sys.stderr,
            )

        annotation_json = (
            PROJECT_ROOT
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
        print("事件标注: {}".format(event_annotations_dir))
        print("")
        print("=" * 76)
        print("【第 2 步】在 Unreal Editor 中运行")
        print("=" * 76)
        print("在 UE Python 控制台执行：")
        print('  py "{}"'.format((SCRIPT_DIR / "02_run_unreal.py").resolve().as_posix()))
        print("")
        print("然后打开 Movie Render Queue，设置：")
        print("  Output Directory:")
        print("    {}".format(
            (PROJECT_ROOT / "Saved" / "FutsalMOT" / "images_clean" / episode_id).resolve().as_posix()
        ))
        print("  File Name Format: {frame_number}")
        print("  Image Format: PNG")
        print("  Resolution: 1920 x 1080")
        print("")
        print("渲染完成后进入第 3 步。")
        print("")
        print("=" * 76)
        print("【第 3 步】Windows 布局检查")
        print("=" * 76)
        check_cmd = '"{}" --annotation "{}" --step 5 --draw-keypoints'.format(
            (SCRIPT_DIR / "03_check_labels.py").resolve().as_posix(),
            annotation_json.as_posix()
        )
        print("在任意终端执行：")
        print('  py {}'.format(check_cmd))
        print("")
        if trajectory_validation_passed and not diagnostic_only and update_current_pointer:
            print("当前运行指针: {}".format(CURRENT_RUN_POINTER))
        print("=" * 76)
        return 0

    except PipelineError as exc:
        failure = {
            "schema_version": "1.0",
            "pipeline_version": SCRIPT_VERSION,
            "generated_at_utc": utc_now(),
            "pipeline_config_path": config_path.resolve().as_posix(),
            "run_id": run_dir.name,
            "run_dir": run_dir.resolve().as_posix(),
            "seed": seed,
            "template_id": template,
            "template_name": template_name,
            "episode_id": episode_id,
            "status": "failed",
            "error": str(exc),
            "attempts": attempts,
        }
        write_json_atomic(output_dir / "pipeline_run_report.json", failure)
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
