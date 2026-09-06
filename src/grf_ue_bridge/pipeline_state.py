"""固定 task workflow 的执行状态与原子持久化。"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List


class StepStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


FIXED_STEPS = ("export", "ue_sequence", "render", "postprocess", "audit", "cleanup")
_VALID_STATUSES = {status.value for status in StepStatus}
CURRENT_VERSION = 1


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _overall(steps: Dict[str, Dict[str, Any]]) -> str:
    statuses = {item.get("status") for item in steps.values()}
    if "FAILED" in statuses:
        return "FAILED"
    if "RUNNING" in statuses:
        return "RUNNING"
    if "PENDING" in statuses:
        return "PENDING"
    return "COMPLETED"


@dataclass
class PipelineState:
    schema: str = "futsalmot_pipeline_state"
    version: int = 1
    task_id: str = ""
    state: str = "PENDING"
    steps: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    updated_at: str = field(default_factory=_now)
    history: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": self.schema,
            "version": self.version,
            "task_id": self.task_id,
            "state": self.state,
            "steps": self.steps,
            "updated_at": self.updated_at,
            "history": self.history,
        }


def create_state(task_id: str) -> PipelineState:
    return PipelineState(
        task_id=task_id,
        steps={step: {"status": StepStatus.PENDING.value} for step in FIXED_STEPS},
    )


def state_path(repo_root: Path, task_id: str) -> Path:
    return Path(repo_root) / ".futsalmot" / "runtime" / task_id / "pipeline_state.json"


def _validate(data: Any) -> PipelineState:
    if not isinstance(data, dict) or data.get("schema") != "futsalmot_pipeline_state":
        raise ValueError("invalid pipeline state schema")
    version = data.get("version")
    if not isinstance(version, int) or isinstance(version, bool):
        raise ValueError("invalid pipeline state version")
    if version > CURRENT_VERSION:
        raise ValueError(f"Unsupported pipeline state version: {version}")
    if version != CURRENT_VERSION or not isinstance(data.get("task_id"), str) or not data["task_id"]:
        raise ValueError("invalid pipeline state version or task_id")
    if data.get("state") not in ("PENDING", "RUNNING", "FAILED", "COMPLETED"):
        raise ValueError("invalid pipeline state overall status")
    steps = data.get("steps")
    if not isinstance(steps, dict) or set(steps) != set(FIXED_STEPS):
        raise ValueError("pipeline state must contain all fixed steps")
    for step in FIXED_STEPS:
        if not isinstance(steps[step], dict) or steps[step].get("status") not in _VALID_STATUSES:
            raise ValueError(f"invalid pipeline state status: {step}")
    if not isinstance(data.get("updated_at"), str) or not isinstance(data.get("history", []), list):
        raise ValueError("invalid pipeline state history")
    expected_state = _overall(steps)
    if data["state"] != expected_state:
        raise ValueError(
            f"inconsistent pipeline state overall status: {data['state']} != {expected_state}"
        )
    return PipelineState(
        schema=data["schema"], version=data["version"], task_id=data["task_id"],
        state=expected_state, steps=steps, updated_at=data["updated_at"],
        history=list(data.get("history", [])),
    )


def load_state(path: Path) -> PipelineState:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"unable to read pipeline state: {path}: {exc}") from exc
    return _validate(data)


def save_state(state: PipelineState, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=".pipeline_state.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(state.to_dict(), stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


def update_step(state: PipelineState, step: str, status: Any, **detail: Any) -> None:
    if step not in FIXED_STEPS:
        raise ValueError(f"unknown pipeline step: {step}")
    status_value = status.value if isinstance(status, StepStatus) else status
    if status_value not in _VALID_STATUSES:
        raise ValueError(f"invalid pipeline step status: {status}")
    previous = state.steps[step].get("status")
    if previous == StepStatus.FAILED.value and status_value == StepStatus.COMPLETED.value:
        raise ValueError("failed pipeline step must be retried before completion")
    state.steps[step] = {**state.steps[step], "status": status_value, **detail}
    state.updated_at = _now()
    state.history.append({"step": step, "from": previous, "to": status_value, "at": state.updated_at})
    state.state = _overall(state.steps)


def mark_failed(state: PipelineState, step: str, exc: BaseException) -> None:
    update_step(
        state,
        step,
        StepStatus.FAILED,
        error_type=type(exc).__name__,
        error_message=str(exc),
    )


def resume_plan(state: PipelineState) -> Dict[str, str]:
    actions = {
        StepStatus.COMPLETED.value: "skip",
        StepStatus.FAILED.value: "retry",
        StepStatus.PENDING.value: "execute",
        StepStatus.SKIPPED.value: "keep",
        StepStatus.RUNNING.value: "unknown",
    }
    return {step: actions[state.steps[step]["status"]] for step in FIXED_STEPS}


def ensure_start_allowed(state: PipelineState, force: bool = False) -> None:
    """阻止并发启动；force 只由显式 CLI 选项启用。"""
    if state.state == "RUNNING" and not force:
        raise ValueError("Existing running pipeline state detected.")


def recover_running_step(state: PipelineState, step: str, artifact_state: str) -> str:
    """按步骤和外部证据处理 RUNNING；没有证据时返回 unknown。"""
    if step not in FIXED_STEPS:
        raise ValueError(f"unknown pipeline step: {step}")
    if state.steps[step].get("status") != StepStatus.RUNNING.value:
        return resume_plan(state)[step]
    if artifact_state == "complete":
        return "complete"
    if artifact_state in ("retry", "pending"):
        return "retry"
    return "unknown"
