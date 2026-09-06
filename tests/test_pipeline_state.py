"""Pipeline State 的纯 Python 契约测试。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from grf_ue_bridge.pipeline_state import (
    FIXED_STEPS,
    PipelineState,
    StepStatus,
    create_state,
    load_state,
    mark_failed,
    resume_plan,
    recover_running_step,
    ensure_start_allowed,
    save_state,
    state_path,
    update_step,
)


def test_create_state_has_fixed_pending_steps():
    state = create_state("task-1")

    assert state.schema == "futsalmot_pipeline_state"
    assert state.version == 1
    assert state.task_id == "task-1"
    assert state.state == "PENDING"
    assert list(state.steps) == list(FIXED_STEPS)
    assert {step["status"] for step in state.steps.values()} == {"PENDING"}


def test_step_lifecycle_and_history():
    state = create_state("task-1")

    update_step(state, "export", StepStatus.RUNNING)
    update_step(state, "export", StepStatus.COMPLETED, note="exported")

    assert state.steps["export"]["status"] == "COMPLETED"
    assert state.state == "PENDING"
    assert len(state.history) == 2
    assert state.steps["export"]["note"] == "exported"


def test_failure_records_exception_details():
    state = create_state("task-1")
    update_step(state, "export", StepStatus.RUNNING)

    mark_failed(state, "export", RuntimeError("boom"))

    assert state.steps["export"]["status"] == "FAILED"
    assert state.steps["export"]["error_type"] == "RuntimeError"
    assert state.steps["export"]["error_message"] == "boom"
    assert state.state == "FAILED"


def test_atomic_save_and_load(tmp_path):
    state = create_state("task-1")
    path = tmp_path / "runtime" / "pipeline_state.json"

    save_state(state, path)
    loaded = load_state(path)

    assert loaded.task_id == "task-1"
    assert json.loads(path.read_text(encoding="utf-8"))["schema"] == "futsalmot_pipeline_state"
    assert not list(path.parent.glob(".pipeline_state.*.tmp"))


def test_load_rejects_corrupt_state(tmp_path):
    path = tmp_path / "pipeline_state.json"
    path.write_text("{not-json", encoding="utf-8")

    with pytest.raises(ValueError, match="pipeline state"):
        load_state(path)


def test_resume_plan_distinguishes_running_as_unknown():
    state = create_state("task-1")
    update_step(state, "export", StepStatus.COMPLETED)
    update_step(state, "render", StepStatus.RUNNING)
    update_step(state, "postprocess", StepStatus.FAILED, error_message="bad")
    update_step(state, "audit", StepStatus.SKIPPED)

    actions = resume_plan(state)

    assert actions["export"] == "skip"
    assert actions["render"] == "unknown"
    assert actions["postprocess"] == "retry"
    assert actions["audit"] == "keep"
    assert actions["cleanup"] == "execute"


def test_state_path_is_stable(tmp_path):
    assert state_path(tmp_path, "task-1") == (
        tmp_path / ".futsalmot" / "runtime" / "task-1" / "pipeline_state.json"
    )


def test_unknown_step_is_rejected():
    state = create_state("task-1")

    with pytest.raises(ValueError, match="unknown pipeline step"):
        update_step(state, "unknown", StepStatus.RUNNING)


def test_failed_step_must_be_retried_before_completion():
    state = create_state("task-1")
    update_step(state, "export", StepStatus.FAILED, error_message="boom")

    with pytest.raises(ValueError, match="retried"):
        update_step(state, "export", StepStatus.COMPLETED)


def test_pipeline_completed_does_not_imply_validation_passed():
    state = create_state("task-1")
    for step in FIXED_STEPS:
        update_step(state, step, StepStatus.COMPLETED)

    assert state.state == "COMPLETED"
    # Pipeline State 只记录执行完成，不能替代 Audit/ValidationResult。
    assert state.state != "FAILED"


def test_resume_plan_keeps_running_unknown():
    state = create_state("task-1")
    update_step(state, "render", StepStatus.RUNNING)

    assert resume_plan(state)["render"] == "unknown"


def test_running_step_recovery_requires_step_specific_evidence():
    state = create_state("task-1")
    update_step(state, "render", StepStatus.RUNNING)

    assert recover_running_step(state, "render", "complete") == "complete"
    assert recover_running_step(state, "render", "pending") == "retry"
    assert recover_running_step(state, "render", "unknown") == "unknown"


def test_state_serializes_completion_metadata():
    state = create_state("task-1")
    update_step(
        state,
        "ue_sequence",
        StepStatus.COMPLETED,
        completion_type="command_generation",
        note="UE/MRQ 尚未完成",
    )

    assert state.steps["ue_sequence"]["completion_type"] == "command_generation"
    assert "UE/MRQ" in state.steps["ue_sequence"]["note"]


def test_resume_plan_has_all_fixed_steps():
    state = create_state("task-1")

    assert list(resume_plan(state)) == list(FIXED_STEPS)
    assert set(resume_plan(state).values()) == {"execute"}


def test_load_rejects_unsupported_future_version(tmp_path):
    path = tmp_path / "pipeline_state.json"
    state = create_state("task-1").to_dict()
    state["version"] = 999
    path.write_text(json.dumps(state), encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported pipeline state version: 999"):
        load_state(path)


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda state: state.pop("steps"), "steps"),
        (lambda state: state["steps"].__setitem__("render", {"status": "BROKEN"}), "status"),
    ],
)
def test_load_rejects_malformed_state(tmp_path, mutation, message):
    path = tmp_path / "pipeline_state.json"
    state = create_state("task-1").to_dict()
    mutation(state)
    path.write_text(json.dumps(state), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_state(path)


def test_running_state_blocks_new_execution_unless_forced():
    state = create_state("task-1")
    update_step(state, "export", StepStatus.RUNNING)

    with pytest.raises(ValueError, match="Existing running pipeline state detected"):
        ensure_start_allowed(state)
    ensure_start_allowed(state, force=True)


def test_status_fields_are_validated(tmp_path):
    path = tmp_path / "pipeline_state.json"
    # 仅验证 malformed overall state 在 load 时不被静默修正。
    data = create_state("task-1").to_dict()
    data["state"] = "BROKEN"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="overall"):
        load_state(path)


def test_load_rejects_inconsistent_overall_state(tmp_path):
    path = tmp_path / "pipeline_state.json"
    data = create_state("task-1").to_dict()
    data["state"] = "COMPLETED"
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="inconsistent"):
        load_state(path)
