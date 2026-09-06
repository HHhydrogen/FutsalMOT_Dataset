# Pipeline State + Resume Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** 为固定的 FutsalMOT task workflow 增加原子 Pipeline State、status 展示和基于现有 workflow 的 resume。

**Architecture:** 新增纯 Python `pipeline_state.py`，只管理执行进度和历史；各现有 CLI workflow 在边界更新固定六步的状态。`task resume` 根据状态和 step-specific artifact policy 调用已有 workflow，不复制 workflow 逻辑。`ValidationResult` 继续独立负责正确性。

**Tech Stack:** Python 3.9、标准库 `dataclasses`/`datetime`/`json`/`os`/`pathlib`/`tempfile`、Typer、pytest。

**Spec:** `docs/superpowers/specs/2026-09-06-pipeline-state-resume-design.md`

## Global Constraints

- 固定 steps 为 `export`、`ue_sequence`、`render`、`postprocess`、`audit`、`cleanup`，不设计动态 DAG。
- 状态文件固定为 `.futsalmot/runtime/<task_id>/pipeline_state.json`，不进入 Git。
- `ue_sequence=COMPLETED` 只表示 command generation 完成，不表示 UE/MRQ/render 完成。
- Pipeline State `COMPLETED` 只表示所有执行步骤结束，不表示数据正确。
- `ValidationResult.passed` 仍是最终数据正确性标准。
- `RUNNING` 永远视为 unknown，必须经过 step-specific recovery policy。
- Resume 必须调用现有 workflow，不复制或重写 workflow。
- 不修改 TaskRequirements、ValidationResult、ResolvedTask、Audit/Cleanup contract、UE Asset、Blueprint、MCP 或 artifact cleanup 策略。
- 保持 Python 3.9 兼容，不增加依赖。

---

### Task 1: Pipeline State Model and Atomic Persistence

**Files:**
- Create: `src/grf_ue_bridge/pipeline_state.py`
- Test: `tests/test_pipeline_state.py`

**Interfaces:**
- `StepStatus` enum with `PENDING`, `RUNNING`, `COMPLETED`, `FAILED`, `SKIPPED`.
- `PipelineState` dataclass with `schema`, `version`, `task_id`, `state`, `steps`, `updated_at`, `history`.
- `create_state(task_id)`, `load_state(path)`, `save_state(state, path)`, `update_step(state, step, status, **detail)`, `mark_failed(state, step, exc)`, `state_path(repo_root, task_id)`, `resume_plan(state)`.

- [ ] **Step 1: Write failing state tests**

  Assert `create_state("t1")` creates exactly the six fixed steps with `PENDING`, state `PENDING`, schema `futsalmot_pipeline_state`, and version `1`. Assert lifecycle updates preserve timestamps/history and failed updates contain `error_type` and `error_message`.

- [ ] **Step 2: Run state tests and confirm red**

  Run: `uv run python -m pytest tests/test_pipeline_state.py -q`

  Expected: import failure because `pipeline_state.py` does not exist.

- [ ] **Step 3: Implement the dataclasses and transition helpers**

  Use fixed step constants. Reject unknown step names and invalid statuses. Recompute overall state from step statuses, with `FAILED` taking precedence over `RUNNING`, then `PENDING`, then `COMPLETED`/`SKIPPED`. Add history entries for every transition. Never convert a failed step directly to completed.

- [ ] **Step 4: Implement atomic save/load**

  Serialize only JSON-safe fields. Write a temporary file in the destination directory, flush and fsync it, then replace the destination with `os.replace()`. Load must reject wrong schema/version, missing fixed steps, invalid status, malformed JSON, and ambiguous state. Test a failed replacement leaves the previous valid file intact.

- [ ] **Step 5: Implement resume planning**

  Return step actions: completed=`skip`, failed=`retry`, pending=`execute`, skipped=`keep`, running=`unknown`. Do not mark RUNNING as complete in the generic planner.

- [ ] **Step 6: Run state tests green**

  Run: `uv run python -m pytest tests/test_pipeline_state.py -q`

### Task 2: Workflow Lifecycle Integration

**Files:**
- Modify: `src/grf_ue_bridge/cli.py`
- Modify: `src/grf_ue_bridge/workflows/task_export.py` only if a callable wrapper is needed
- Modify: `src/grf_ue_bridge/workflows/task_postprocess.py` only if existing callable behavior must expose failure details
- Modify: `src/grf_ue_bridge/workflows/task_audit.py` only if existing callable behavior must expose report path
- Modify: `src/grf_ue_bridge/workflows/artifact_cleanup.py` only if existing callable behavior must expose cleanup result
- Test: `tests/test_pipeline_state.py`, `tests/test_task_cli.py`

**Interfaces:**
- Add CLI-local helpers to load/create state from `resolved.repo_root` and update one step around an existing callable.
- Existing workflow functions and integer return codes remain compatible.
- `task ue-command` writes `ue_sequence=COMPLETED` with `completion_type="command_generation"`; render remains `PENDING`.

- [ ] **Step 1: Write failing lifecycle integration tests**

  Use Typer `CliRunner` or monkeypatched existing workflow callables to assert `task export`, `task postprocess`, `task audit`, and `task cleanup` write RUNNING then COMPLETED on zero return and FAILED on exceptions/nonzero returns. Assert `task ue-command` does not mark render completed and records the command-generation note.

- [ ] **Step 2: Run integration tests and confirm red**

  Run: `uv run python -m pytest tests/test_pipeline_state.py tests/test_task_cli.py -q`

  Expected: state file missing or lifecycle assertions fail.

- [ ] **Step 3: Add state lifecycle wrapper at CLI boundaries**

  For each command, resolve the task using the existing `_resolve_runtime()` path. Create state if absent, update the step to RUNNING before invoking the existing workflow, then update to COMPLETED/FAILED according to return code or exception. Record error type/message for exceptions and a nonzero return-code error for ordinary failures.

- [ ] **Step 4: Integrate asynchronous `ue-command` semantics**

  After the existing command and resolved-task file are generated, mark only `ue_sequence` completed with `completion_type="command_generation"` and a note that UE/MRQ/render are not complete. Leave `render` pending.

- [ ] **Step 5: Keep correctness separate from state**

  When audit writes a report, keep the audit step execution status separate from `report["passed"]`. Status output must show both pipeline state and ValidationResult state.

- [ ] **Step 6: Run lifecycle tests green**

  Run: `uv run python -m pytest tests/test_pipeline_state.py tests/test_task_cli.py -q`

### Task 3: Status CLI and Recovery Policies

**Files:**
- Modify: `src/grf_ue_bridge/cli.py`
- Modify: `src/grf_ue_bridge/workflows/task_status.py`
- Modify: `src/grf_ue_bridge/pipeline_state.py` if recovery helpers belong there
- Test: `tests/test_pipeline_state.py`, `tests/test_task_cli.py`

**Interfaces:**
- `task status TASK` displays state file plus existing artifact summary.
- Add `task resume TASK`.
- Recovery helper returns one of `skip`, `retry`, `execute`, `keep`, `complete`, `unknown` and never treats RUNNING as automatically complete.

- [ ] **Step 1: Write failing status and recovery tests**

  Assert status output includes task, overall state, all six fixed steps, failed step/error/action. Assert no state file displays all six steps as PENDING. Assert RUNNING render with successful `render_summary.json` can be completed, while RUNNING export without complete trajectory is retry.

- [ ] **Step 2: Run tests and confirm red**

  Run: `uv run python -m pytest tests/test_pipeline_state.py tests/test_task_cli.py -q`

  Expected: status has no pipeline state section and resume command is absent.

- [ ] **Step 3: Implement status rendering**

  Preserve existing artifact counts. Add human-readable markers for completed, running, failed, pending and skipped. Read an existing canonical audit report using `validation_result_from_report()` and display its `passed` status without changing pipeline state.

- [ ] **Step 4: Implement step-specific recovery checks**

  Use only existing artifacts and validation helpers:

  - export: complete only when required trajectory files are present and readable; otherwise retry.
  - ue_sequence: preserve command-generation completion; do not infer UE execution.
  - render: complete only for readable successful `render_summary.json`; otherwise retry/keep pending.
  - postprocess: complete only when existing annotation/pose validation passes; otherwise retry.
  - audit: complete when report is readable and canonical/legacy parsing succeeds; report failure remains a validation failure.
  - cleanup: complete only when manifest explicitly says cleanup applied; otherwise retry through existing gate.

- [ ] **Step 5: Implement `task resume` using existing workflow callables**

  Resolve task, load/create state, process fixed steps in order, skip completed, keep skipped, retry failed, execute pending, and apply recovery policy to RUNNING. Invoke existing task workflow functions/CLI helpers; do not duplicate export, postprocess, audit or cleanup implementation. Stop and record FAILED on the first nonzero/exception result.

- [ ] **Step 6: Run status/resume tests green**

  Run: `uv run python -m pytest tests/test_pipeline_state.py tests/test_task_cli.py -q`

### Task 4: Documentation and Final Verification

**Files:**
- Modify: `README.md`
- Modify: `docs/DATA_CONTRACT.md`
- Modify: `docs/VALIDATION_AND_LIMITATIONS.md`
- Test: all existing tests and `tests/test_pipeline_state.py`

- [ ] **Step 1: Document state location and responsibilities**

  Explain the fixed schema, atomic runtime storage, lifecycle, asynchronous `ue-command` semantics, status output, resume policy and the distinction between Pipeline State and ValidationResult.

- [ ] **Step 2: Run focused tests**

  Run: `uv run python -m pytest tests/test_pipeline_state.py tests/test_task_cli.py tests/test_task_resolver.py -q`

- [ ] **Step 3: Run the complete default suite**

  Run: `uv run python -m pytest`

  Do not claim GRF integration or Unreal Editor coverage unless separately executed.

- [ ] **Step 4: Run static hygiene checks**

  Run: `git diff --check` and `git status --short --branch`.

  Confirm only source/tests/docs are staged; `.futsalmot/local.json`, `.futsalmot/runtime/`, logs and generated files remain ignored.

- [ ] **Step 5: Inspect and deliver**

  Review the complete diff, commit only intended inner-repository files with a Simplified Chinese message, push `origin/main`, then update the outer UE submodule pointer in a separate commit and push `origin/master`.
