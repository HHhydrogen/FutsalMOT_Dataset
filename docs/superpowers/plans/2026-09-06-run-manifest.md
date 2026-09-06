# Unified Run Manifest Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** 为每次 task pipeline run 在 runtime 目录生成轻量统一 `run_manifest.json`，记录来源、运行时间、pipeline state、Audit validation 和 artifact summary。

**Architecture:** 新增纯 Python `run_manifest.py`，只负责结果/来源摘要和原子 JSON 持久化，不替代 Pipeline State、ValidationResult 或现有 artifact manager。现有 CLI workflow 在状态更新边界同步 manifest；cleanup 在删除前统计 artifact，避免删除后丢失摘要。

**Tech Stack:** Python 3.9、标准库 `dataclasses`/`datetime`/`hashlib`/`json`/`pathlib`/`subprocess`/`tempfile`、Typer、pytest。

**Spec:** 本对话中批准的 P1-1 Unified Run Manifest 设计。

## Global Constraints

- manifest 路径固定为 `.futsalmot/runtime/<task_id>/run_manifest.json`。
- `task_hash` 只哈希 Task Spec JSON 原始内容，不包含 resolved path、local config 或 machine path。
- Python code commit 使用 `git rev-parse HEAD`；UE commit 获取失败写 `null`，不阻断 workflow。
- artifact summary 是 best-effort 记录；统计失败写 error 或跳过字段，不影响 workflow 状态。
- Audit validation 直接使用 `ValidationResult.passed`，不重新判断。
- cleanup 在 artifact 删除前更新 manifest，不能依赖删除后的目录统计。
- 不修改 TaskRequirements、ValidationResult、ResolvedTask、Pipeline State schema、Audit/Cleanup contract 或 artifact format。
- 不新增 Dataset Manager、Artifact Manager、Experiment Tracker 或 Quality Report system。
- runtime manifest 不进入 Git；保持 `.futsalmot/` runtime ignore 规则。

---

### Task 1: Manifest Model, Provenance and Atomic Storage

**Files:**
- Create: `src/grf_ue_bridge/run_manifest.py`
- Test: `tests/test_run_manifest.py`

**Interfaces:**
- `run_manifest_path(repo_root: Path, task_id: str) -> Path`
- `task_spec_hash(task_file: Path) -> str`
- `git_commit(repo_root: Path) -> Optional[str]`
- `create_run_manifest(task_id: str, task_file: Path, repo_root: Path, started_at: Optional[str] = None) -> dict`
- `load_run_manifest(path: Path) -> dict`
- `save_run_manifest(manifest: dict, path: Path) -> None`
- `update_validation(manifest: dict, audit_report_path: Path) -> None`
- `update_artifacts(manifest: dict, dataset_episode_dir: Path) -> None`

- [ ] **Step 1: Write failing manifest tests**

  Test manifest creation with the exact schema/version, deterministic task hash from JSON bytes, code commit capture, unavailable UE commit as `None`, validation update from a canonical Audit report, artifact summary update, and malformed manifest rejection.

- [ ] **Step 2: Run tests and verify red**

  Run: `uv run python -m pytest tests/test_run_manifest.py -q`

  Expected: import failure because `run_manifest.py` does not exist.

- [ ] **Step 3: Implement source and runtime helpers**

  Compute SHA-256 directly from the task file bytes. Run Git commands with `subprocess.run(..., check=False, capture_output=True, text=True)` and return `None` for missing repository, command failure or invalid output. Keep UE commit lookup optional and non-blocking; use the outer repository path only when explicitly discoverable from `repo_root`.

- [ ] **Step 4: Implement atomic manifest persistence and schema validation**

  Require `schema=futsalmot_run_manifest`, `version=1`, string task id, source/runtime/pipeline/validation/artifacts objects. Save through a same-directory temporary file, flush/fsync, and `os.replace()`. Load errors must be explicit and must not silently reset an existing manifest.

- [ ] **Step 5: Implement best-effort artifact summary**

  Record only lightweight counts such as `images.count`, `annotations.count`, `mot.count`, `mask.count`, `pose.count`, and `cameras.count`. Catch filesystem/parse errors per field and record `artifacts.errors` without raising to callers.

- [ ] **Step 6: Run manifest unit tests green**

  Run: `uv run python -m pytest tests/test_run_manifest.py -q`

### Task 2: Integrate Manifest with Existing Task CLI Workflows

**Files:**
- Modify: `src/grf_ue_bridge/cli.py`
- Modify: `tests/test_run_manifest.py`
- Modify: `tests/test_task_cli.py`

**Interfaces:**
- Add CLI-local helpers that load/create the manifest from `resolved.repo_root` and save it without changing existing Pipeline State helpers.
- Existing command return codes, Pipeline State transitions, Audit reports and Cleanup Gate behavior remain unchanged.

- [ ] **Step 1: Write failing CLI integration tests**

  Assert `task ue-command` creates the runtime manifest with source fields, `task audit` writes `validation.passed` from the generated Audit report, `task status` displays manifest information, and cleanup records artifact counts before invoking deletion.

- [ ] **Step 2: Run integration tests and confirm red**

  Run: `uv run python -m pytest tests/test_run_manifest.py tests/test_task_cli.py -q`

  Expected: manifest file is absent or status/validation assertions fail.

- [ ] **Step 3: Add a CLI manifest helper**

  Load an existing manifest or create one using the original task file and resolved repository root. Set `runtime.started_at` only on first creation; update `runtime.finished_at` after a workflow completes or fails. Manifest save failures must be reported but must not replace the existing workflow return code.

- [ ] **Step 4: Integrate export, ue-command and postprocess**

  Create/update the manifest at command start. Keep Pipeline State updates independent. After the existing workflow call, set runtime finish time and best-effort artifact summary; do not alter ValidationResult or artifact generation.

- [ ] **Step 5: Integrate audit validation**

  After the existing audit command writes `audit/soak_audit_report.json`, load it through `validation_result_from_report()` and assign exactly `validation.passed = result.passed`, plus the report path. If the report cannot be read, record an error in the manifest only; preserve the audit command return code.

- [ ] **Step 6: Integrate cleanup before deletion**

  Before calling existing `plan_cleanup()` or `apply_cleanup()`, update artifact summary and save the manifest. This preserves counts of transient artifacts that may subsequently be deleted. Cleanup gate and deletion set remain unchanged.

- [ ] **Step 7: Extend status output without merging semantics**

  Display manifest path/source commit/UE commit and validation summary alongside Pipeline State. Never use manifest presence or pipeline completion as a substitute for `ValidationResult.passed`.

- [ ] **Step 8: Run CLI integration tests green**

  Run: `uv run python -m pytest tests/test_run_manifest.py tests/test_task_cli.py -q`

### Task 3: Documentation and Final Verification

**Files:**
- Modify: `README.md`
- Modify: `docs/DATA_CONTRACT.md`
- Modify: `docs/VALIDATION_AND_LIMITATIONS.md`
- Test: all existing tests and `tests/test_run_manifest.py`

- [ ] **Step 1: Document responsibilities and runtime location**

  Describe Run Manifest as source/result summary, Pipeline State as execution progress, ValidationResult as correctness, the manifest schema, provenance fallback to null, best-effort artifact counts, and cleanup-before-delete timing.

- [ ] **Step 2: Run focused tests**

  Run: `uv run python -m pytest tests/test_run_manifest.py tests/test_task_cli.py tests/test_pipeline_state.py -q`

- [ ] **Step 3: Run the complete default suite**

  Run: `uv run python -m pytest`

  Do not claim GRF integration or Unreal Editor coverage unless separately executed.

- [ ] **Step 4: Run static hygiene checks**

  Run: `git diff --check` and `git status --short --branch`.

  Confirm no `.futsalmot/local.json`, runtime JSON, logs or generated files are staged; the new manifest is runtime-only.

- [ ] **Step 5: Inspect, commit and push both repositories**

  Stage only Run Manifest source/tests/docs. Commit and push the inner `main` branch first. Confirm its remote commit exists, then update only the outer submodule gitlink, commit and push outer `master`.
