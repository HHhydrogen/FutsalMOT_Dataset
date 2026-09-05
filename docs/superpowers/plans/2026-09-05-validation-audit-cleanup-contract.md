# Validation、Audit、Cleanup 统一契约实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 task/resolved-task 的 requirements 成为 Validation、Audit、Cleanup 的唯一功能开关来源，并用共享 `ValidationResult` 统一 PASS/FAIL/SKIPPED 语义。

**Architecture:** 新增两个无 UE 依赖的纯 Python 边界模块：`task_requirements.py` 负责从 task/resolved task 解析 required 功能，`validation_result.py` 负责 check 状态和最终成功结论。annotation validator、task audit、task postprocess 和 artifact cleanup 复用这两个模块；现有 audit detail 字段和 standalone CLI 返回码保留。

**Tech Stack:** Python 3.9、标准库 `dataclasses`/`json`/`pathlib`、现有 pytest/typer/Pydantic；不增加依赖，不修改 Unreal 资产或 UE Python。

**Spec:** `docs/superpowers/specs/2026-09-05-validation-audit-cleanup-contract-design.md`

## Global Constraints

- Python 版本固定为 `>=3.9,<3.10`，新增代码必须兼容 Python 3.9。
- Validation、Audit、Cleanup 必须共享同一套成功/失败语义。
- 未启用的功能必须是 `skipped`，不能成为失败条件。
- errors 非空或 required check failed 必须使 `passed=False`；只有 warnings 时必须通过。
- `validate-annotations` standalone 默认行为保持兼容；task workflow 必须显式传入 task requirements。
- Cleanup 默认 dry-run；`--apply` gate 失败时不得删除任何文件。
- 不删除 canonical artifacts，不扩大现有 transient 删除集合。
- 代码注释、文档字符串和技术文档使用简体中文；不自动 commit 或 push。

## 文件职责

- Create: `src/grf_ue_bridge/validation_result.py`，canonical result、check 状态、最终结论和 audit report 读取。
- Create: `src/grf_ue_bridge/task_requirements.py`，resolved task/config 到 requirements 的确定性解析。
- Modify: `src/grf_ue_bridge/annotation_validator.py`，增加 task-aware MOT requirement 和 structured result 入口。
- Modify: `src/grf_ue_bridge/workflows/task_postprocess.py`，把 shared requirements 传给 annotation validator，并输出 skipped/diagnostic 信息。
- Modify: `src/grf_ue_bridge/workflows/task_audit.py`，按 requirements 运行 checks，使用 canonical result 生成报告。
- Modify: `src/grf_ue_bridge/workflows/artifact_cleanup.py`，按 requirements 做 gate，优先读取 canonical audit，legacy fallback fail safe。
- Modify: `src/grf_ue_bridge/cli.py`，保持 CLI 入口并把 resolved task 传入 audit/validator 所需接口。
- Test: `tests/test_validation_contract.py`，集中覆盖 requirements、result、annotation、audit、cleanup gate 和删除安全性。
- Modify: `docs/VALIDATION_AND_LIMITATIONS.md`，删除已经修复的三项限制并记录真实行为。

### Task 1: 建立 canonical result 和 task requirements

**Files:**
- Create: `src/grf_ue_bridge/validation_result.py`
- Create: `src/grf_ue_bridge/task_requirements.py`
- Test: `tests/test_validation_contract.py`

**Interfaces:**
- Produces `ValidationResult`, `CheckStatus`, `add_check()`, `finalize()`, `to_dict()` 和 `validation_result_from_report()`。
- Produces `TaskRequirements` 和 `resolve_task_requirements(source)`；对象至少提供 `requires_render`、`requires_instance_mask`、`requires_mot`、`requires_yolo_det`、`requires_yolo_seg`、`requires_pose`。

- [ ] **Step 1: Write failing result tests**

```python
def test_result_errors_always_fail():
    result = ValidationResult()
    result.errors.append("broken")
    result.finalize()
    assert result.passed is False
    assert result.exit_code == 1


def test_result_warning_only_passes():
    result = ValidationResult()
    result.warnings.append("non-blocking")
    result.add_check("optional", status="skipped", required=False,
                     message="not enabled")
    result.finalize()
    assert result.passed is True
    assert result.exit_code == 0


def test_required_failed_check_fails():
    result = ValidationResult()
    result.add_check("render", status="failed", required=True,
                     message="render failed")
    result.finalize()
    assert result.passed is False
    assert result.exit_code == 1
```

- [ ] **Step 2: Run result tests and verify they fail**

Run: `uv run pytest tests/test_validation_contract.py -q`

Expected: collection or import failure because `validation_result.py` and its public interfaces do not exist.

- [ ] **Step 3: Write failing requirements tests**

```python
def test_json_only_task_does_not_require_mot():
    source = {
        "ue_profile": {"annotation_export": {
            "render_rgb": {"enabled": True},
            "instance_mask": {"enabled": True},
            "export_mot": False,
        }},
        "postprocess": {"formats": ["json"]},
    }
    req = resolve_task_requirements(source)
    assert req.requires_mot is False


def test_pose_requirement_is_config_not_file_driven():
    source = {"postprocess": {"yolo_pose": {"enabled": False}}}
    assert resolve_task_requirements(source).requires_pose is False


def test_explicit_mot_format_requires_mot():
    source = {"postprocess": {"formats": ["json", "mot"]}}
    assert resolve_task_requirements(source).requires_mot is True
```

- [ ] **Step 4: Run requirements tests and verify they fail**

Run: `uv run pytest tests/test_validation_contract.py -q`

Expected: import or attribute failure for `TaskRequirements`/`resolve_task_requirements`.

- [ ] **Step 5: Implement the result model**

Implement a Python 3.9-compatible dataclass. `add_check(name, status, required=False, message=None, **detail)` validates status in `passed/failed/skipped`, stores a dictionary under `checks[name]`, and appends a failed required message to `errors` only when the caller has not already supplied it. `finalize()` derives `passed` from `errors` and required failed checks, then sets `exit_code` to `0` or `1`; it must not treat warnings or skipped checks as failures. `to_dict()` emits only the canonical fields and keeps list/dict values JSON serializable. `validation_result_from_report()` must distinguish canonical reports from legacy reports and return a fail-safe result for malformed/ambiguous input.

- [ ] **Step 6: Implement deterministic requirement resolution**

Accept an object with `model_dump()`, a plain resolved-task dictionary, or a task-shaped dictionary. Normalize `ue_profile.annotation_export` and `ue.annotation_export` sources, plus `postprocess`. Set `requires_pose` only from `postprocess.yolo_pose.enabled`; set `requires_instance_mask` only from `annotation_export.instance_mask.enabled`; set `requires_render` from `annotation_export.render_rgb.enabled`, defaulting to `False` because the UE renderer also defaults the absent block to disabled; set YOLO requirements from formats. Set `requires_mot` when formats contains `mot`, or when `export_mot` is true and the task has no explicit formats; explicit `export_mot=false` must not be overridden by a JSON-only formats list. Do not inspect episode files.

- [ ] **Step 7: Run Task 1 tests**

Run: `uv run pytest tests/test_validation_contract.py -q`

Expected: result and requirement tests PASS.

### Task 2: Make annotation validation task-aware

**Files:**
- Modify: `src/grf_ue_bridge/annotation_validator.py:51-236,575-621`
- Modify: `src/grf_ue_bridge/cli.py:235-262`
- Modify: `src/grf_ue_bridge/workflows/task_postprocess.py:110-119`
- Test: `tests/test_validation_contract.py`

**Interfaces:**
- `validate_annotation_dir(annotation_dir, workers=0, validation_level="full", require_mot=True)` continues returning `int`.
- Add `validate_annotation_result(annotation_dir, workers=0, validation_level="full", require_mot=True) -> ValidationResult` or an equivalent structured API used by task workflows; the integer function delegates to it.

- [ ] **Step 1: Add MOT behavior tests**

Build the existing minimal camera fixture with `mot=False`, then assert:

```python
def test_annotation_json_only_missing_mot_passes(tmp_path):
    cam = _write_camera_without_mot(tmp_path)
    result = validate_annotation_result(tmp_path, workers=1,
                                        validation_level="quick",
                                        require_mot=False)
    assert result.passed is True
    assert result.checks["mot_export"]["status"] == "skipped"


def test_annotation_required_mot_missing_fails(tmp_path):
    cam = _write_camera_without_mot(tmp_path)
    result = validate_annotation_result(tmp_path, workers=1,
                                        validation_level="quick",
                                        require_mot=True)
    assert result.passed is False
    assert any("gt/gt.txt" in error for error in result.errors)
```

- [ ] **Step 2: Run the MOT tests and verify they fail**

Run: `uv run pytest tests/test_validation_contract.py -q`

Expected: JSON-only case fails because the current camera validator always appends the missing MOT error, and the structured API is absent.

- [ ] **Step 3: Thread `require_mot` through camera validation**

Add the parameter to `_validate_camera`, `_validate_camera_task`, and the process-pool task tuple. Keep all existing structural and content checks. At the missing `gt/gt.txt` branch, append an error only when `require_mot` is true; otherwise record the skipped state in the structured result. If `gt.txt` exists, continue validating it even when MOT is not required, but do not make its absence a failure.

- [ ] **Step 4: Implement structured annotation result without breaking the integer API**

Collect camera errors exactly as before, create a `ValidationResult`, add `mot_export` as passed/failed/skipped based on `require_mot` and missing/invalid MOT findings, add a camera validation check, copy errors, and finalize. Keep `_report()` and the existing CLI-facing integer return behavior. Do not make standalone callers pass a task config.

- [ ] **Step 5: Pass task requirements from postprocess**

In `run_postprocess()`, resolve requirements from `resolved`, pass `require_mot=requirements.requires_mot` to the structured/integer validator, and print a diagnostic line for skipped MOT. The generated formats remain controlled by `postprocess.formats`; do not alter `annotate_masks_dir()` output behavior.

- [ ] **Step 6: Run annotation and existing targeted tests**

Run: `uv run pytest tests/test_validation_contract.py tests/test_annotation_validator.py tests/test_task_cli.py -q`

Expected: new MOT matrix and existing standalone validator/CLI tests PASS. Existing tests that intentionally omit MOT must keep passing under their default `require_mot=True` behavior unless they explicitly model JSON-only task behavior.

### Task 3: Integrate canonical checks into Task Audit

**Files:**
- Modify: `src/grf_ue_bridge/workflows/task_audit.py:133-263,408-484,617-731`
- Modify: `src/grf_ue_bridge/cli.py:774-817`
- Test: `tests/test_validation_contract.py`

**Interfaces:**
- `task_audit.main(argv)` keeps its existing CLI arguments and integer return code.
- Audit JSON retains all current detail sections and adds top-level `checks` with canonical check entries.

- [ ] **Step 1: Add audit result tests for skipped and required features**

```python
def test_audit_report_has_canonical_skipped_pose_and_mot(tmp_path):
    # fixture contains the required canonical RGB/annotation files but no pose session or gt.txt
    report = run_audit_for_fixture(tmp_path, pose_enabled=False,
                                   formats=["json"])
    assert report["passed"] is True
    assert report["checks"]["runtime_pose"]["status"] == "skipped"
    assert report["checks"]["mot_export"]["status"] == "skipped"


def test_audit_required_pose_missing_session_fails(tmp_path):
    report = run_audit_for_fixture(tmp_path, pose_enabled=True,
                                   formats=["json"])
    assert report["passed"] is False
    assert "pose_session.json" in " ".join(report["errors"])
```

- [ ] **Step 2: Run audit tests and verify they fail**

Run: `uv run pytest tests/test_validation_contract.py -q`

Expected: reports do not contain `checks`, and missing Pose is currently skipped solely by file presence rather than task requirement.

- [ ] **Step 3: Extend audit entry point with task requirements**

Add an optional `requirements` argument to `main`-internal execution or an equivalent explicit parameter while retaining CLI flags. The task CLI computes requirements from `resolved` and passes the resolved requirement data through a deterministic command argument or direct callable helper. Standalone audit keeps current conservative defaults: mask checks enabled, render required, and Pose optional unless explicitly requested by a new internal/task parameter.

- [ ] **Step 4: Convert audit checks to canonical statuses**

Keep existing local detail dictionaries with their `ok` keys. Add canonical entries for camera structure, render summary, mask, MOT, YOLO, Pose, synchronization, mapping, calibration, cross-camera identity, and nested annotation validation. Required failed checks contribute errors; optional missing checks become skipped. `check_pose_coco17()` must accept `required` and report missing session as failed only when required; when disabled it must return `status=skipped` even if stale pose files exist. Use `require_mot` when invoking annotation validation.

- [ ] **Step 5: Finalize and write the report through `ValidationResult`**

Create a result from accumulated errors/warnings and canonical checks, call `finalize()`, then assign `report["errors"]`, `report["warnings"]`, `report["passed"]`, `report["exit_code"]`, and `report["checks"]`. Preserve current Markdown generation and all detail sections. Warnings must not be appended to errors.

- [ ] **Step 6: Run audit and CLI tests**

Run: `uv run pytest tests/test_validation_contract.py tests/test_task_cli.py tests/test_audit_fixes.py -q`

Expected: canonical audit matrix PASS, existing audit CLI fixtures PASS, and warnings-only reports remain successful.

### Task 4: Make Cleanup use the canonical task-aware gate

**Files:**
- Modify: `src/grf_ue_bridge/workflows/artifact_cleanup.py:82-179`
- Modify: `src/grf_ue_bridge/cli.py:857-887`
- Test: `tests/test_validation_contract.py`

**Interfaces:**
- `_validation_gate(dataset_episode_dir, resolved=None) -> list[str]` remains callable by existing code, with `resolved` now used when available.
- `plan_cleanup(..., resolved=None)` and `apply_cleanup(..., resolved=None)` keep old positional call compatibility.
- `apply_cleanup()` continues returning `ok`, `deleted_files`, and `deleted_bytes`, and adds diagnostic `gate_checks` where useful.

- [ ] **Step 1: Add cleanup gate tests before implementation**

```python
def test_cleanup_skips_pose_for_disabled_task(tmp_path):
    resolved = make_resolved_task(tmp_path, pose_enabled=False)
    write_successful_render_summary(tmp_path)
    assert _validation_gate(tmp_path, resolved=resolved) == []


def test_cleanup_blocks_canonical_failed_audit(tmp_path):
    resolved = make_resolved_task(tmp_path, pose_enabled=False)
    write_successful_render_summary(tmp_path)
    write_audit(tmp_path, {"passed": False, "exit_code": 1,
                          "errors": ["bad"], "warnings": [], "checks": {}})
    result = apply_cleanup(tmp_path, ["Cam_01"], resolved=resolved)
    assert result["ok"] is False
    assert result["deleted_files"] == 0


def test_cleanup_allows_warnings_only_audit(tmp_path):
    resolved = make_resolved_task(tmp_path, pose_enabled=False)
    write_successful_render_summary(tmp_path)
    write_audit(tmp_path, {"passed": True, "exit_code": 0,
                          "errors": [], "warnings": ["warn"], "checks": {}})
    assert _validation_gate(tmp_path, resolved=resolved) == []
```

- [ ] **Step 2: Run cleanup tests and verify they fail**

Run: `uv run pytest tests/test_validation_contract.py -q`

Expected: disabled Pose is rejected by the unconditional missing-session check, and canonical `passed=false` audit is not recognized.

- [ ] **Step 3: Replace unconditional file gates with requirements-driven checks**

Resolve requirements from `resolved`. Require `render_summary.json` and `status=success` only when `requires_render` is true. Require `pose_session.json`, valid JSON, and `capture_complete=true` only when `requires_pose` is true. For disabled Pose, add a skipped diagnostic internally but return no blocking problem. Do not infer requirements from artifact existence.

- [ ] **Step 4: Read audit reports fail-safe**

Use `validation_result_from_report()` for reports with canonical fields. If canonical `passed` is false, exit code is nonzero, errors are nonempty, or a required check failed, add a diagnostic problem. If canonical fields are absent, accept legacy `ok/failed_checks` only when unambiguous; `ok=false`, nonempty `failed_checks`, malformed JSON, or incomplete ambiguous reports must block. Warnings-only canonical reports do not block. Do not use a blanket exception that silently ignores parse failures.

- [ ] **Step 5: Preserve dry-run/apply safety and CLI wiring**

Pass `resolved` from `task_cleanup` to both `plan_cleanup()` and `apply_cleanup()`. Keep dry-run as a read-only report path. Keep `collect_transient()` unchanged except for any strictly necessary type-safe code; do not add canonical paths to deletion. On gate failure, return before any unlink and report reasons containing the failing check.

- [ ] **Step 6: Run cleanup matrix tests**

Run: `uv run pytest tests/test_validation_contract.py tests/test_task_cli.py -q`

Expected: Pose B1-B4, Audit C1-C3, and dry-run/apply E1-E3 PASS.

### Task 5: Update documentation and complete verification

**Files:**
- Modify: `docs/VALIDATION_AND_LIMITATIONS.md:42-60,74-94,117-137,139-147`
- Test: `tests/test_validation_contract.py` and all existing tests

- [ ] **Step 1: Update the documented behavior**

Rewrite the annotation validation section to state that `require_mot` is task-aware, standalone defaults remain compatible, and JSON-only tasks skip MOT. Rewrite Audit to document canonical `checks` plus retained detail sections. Rewrite Cleanup to document task-aware render/Pose requirements, canonical audit precedence, legacy fallback, and warnings-only non-blocking behavior. Remove the three fixed limitations without changing unrelated configuration, UE, or visual-validation limitations.

- [ ] **Step 2: Run focused regression tests**

Run: `uv run pytest tests/test_validation_contract.py tests/test_annotation_validator.py tests/test_task_cli.py tests/test_audit_fixes.py -q`

Expected: all targeted tests PASS.

- [ ] **Step 3: Run the complete default suite**

Run: `uv run pytest`

Expected: all non-`grf_integration` and non-UE MCP tests PASS. Do not claim GRF integration coverage unless separately run and successful.

- [ ] **Step 4: Run static hygiene checks**

Run: `git diff --check`

Expected: no whitespace errors and no generated files added.

- [ ] **Step 5: Inspect final changes and report status**

Run: `git status --short --branch` and `git diff --stat`

Confirm only Dataset submodule files are changed, no UE assets are modified, and no commit/push is performed. Final report must include root cause, implementation, changed files, behavior matrix, exact test commands/results, compatibility, out-of-scope findings, and an itemized Definition of Done answer.
