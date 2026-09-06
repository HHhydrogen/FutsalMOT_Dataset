# Portable Config Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** 让 task 配置脱离机器绝对路径，同时保持现有 task schema、resolved-task 字段和 UE/P1/P2 调用方式稳定。

**Architecture:** 保留 `DatasetTaskConfig.dataset_root` 和 `ue_project_root` 字段，新增轻量 machine-local runtime config 读取层。resolver 统一按 CLI override > `.futsalmot/local.json` > environment expansion > task placeholder > legacy task path 解析路径；只生成绝对路径，不创建目录或执行强环境检查。`task validate` 负责检查 UE root 和 `.uproject`，export 负责创建 dataset root。

**Tech Stack:** Python 3.9、标准库 `json`/`os`/`pathlib`/`re`、现有 Pydantic、Typer、pytest。

**Spec:** 本对话中批准的 P0-2 Portable Config 设计。

## Global Constraints

- 保留 `dataset_root` / `ue_project_root` schema 字段，不重写 task schema。
- runtime config 文件为 `.futsalmot/local.json`，真实文件不提交，模板为 `.futsalmot/local.example.json`。
- 路径优先级严格为 CLI override > local runtime config > environment expansion > task placeholder > legacy absolute task path。
- `ResolvedTask` 输出字段和 schema 保持不变，并继续提供绝对路径。
- resolver 不创建目录，不执行强环境检查。
- `task validate` 要求 `ue_project_root` 存在且包含 `.uproject`；`dataset_root` 可不存在。
- export 阶段负责创建 dataset root。
- 不修改 Validation/Audit/Cleanup contract、UE 资产、Blueprint、MCP、Pose pipeline 或 cleanup 策略。
- 新增代码、注释、文档字符串和技术文档使用简体中文；不自动混入无关文件。

---

### Task 1: Runtime Path Resolution Boundary

**Files:**
- Create: `src/grf_ue_bridge/config/runtime.py`
- Modify: `src/grf_ue_bridge/config/resolver.py`
- Modify: `src/grf_ue_bridge/config/models.py` only if optional task path parsing is required by existing Pydantic validation
- Test: `tests/test_portable_config.py`

**Interfaces:**
- `load_local_runtime_config(repo_root: Path, path: Optional[Path] = None) -> dict`
- `resolve_runtime_paths(task_file: Path, task, repo_root: Path, dataset_root: Optional[str] = None, ue_project_root: Optional[str] = None, local_config: Optional[Path] = None) -> tuple[Path, Path, List[str]]`
- `expand_runtime_path(value: str, env: Mapping[str, str]) -> str`
- `resolve_task(task_file, dataset_root=None, ue_project_root=None, local_config=None)` continues returning the existing `ResolvedTask` shape and stores path warnings on a non-contract diagnostic channel without changing serialized fields.

- [ ] **Step 1: Write failing tests for variable expansion and priority**

  Add tests that construct a task fixture with `${FUTSALMOT_DATASET_ROOT}` and `${FUTSALMOT_UE_ROOT}`, set environment values, create local config values, and pass explicit resolver overrides. Assert the resulting absolute paths follow CLI > local > env > task placeholder > legacy task path. Assert unresolved `${UNKNOWN_ROOT}` raises a clear `ValueError` containing `missing runtime path variable`.

- [ ] **Step 2: Run the new tests and verify they fail**

  Run: `uv run pytest tests/test_portable_config.py -q`

  Expected: import/signature failure because the runtime resolver boundary does not exist.

- [ ] **Step 3: Implement local config loading and safe variable expansion**

  Read `.futsalmot/local.json` relative to the repository root unless an explicit local config path is supplied. Require a JSON object and optional `paths` object. Expand only `${NAME}` variables using `os.environ`; unresolved variables raise `ValueError`. Do not create or mutate any directories while resolving.

- [ ] **Step 4: Implement deterministic path precedence**

  Resolve each path independently in this order:

  ```text
  explicit resolver/CLI override
  local.json paths.dataset_root / paths.ue_project_root
  environment value FUTSALMOT_DATASET_ROOT / FUTSALMOT_UE_ROOT
  task value after ${VAR} expansion
  legacy task absolute/relative value
  ```

  Treat a task value that is exactly a placeholder as a task placeholder source; retain non-placeholder legacy values and emit a warning containing `Using deprecated absolute path from task config`. Relative task paths remain resolved as paths relative to the task file or repository according to the existing path convention, without changing resolved output fields.

- [ ] **Step 5: Thread optional resolver inputs through `resolve_task()`**

  Keep the existing one-argument call valid. Add keyword-only optional overrides and use the new helper before building `ResolvedTask`; leave `ResolvedTask` model and `sanitize_resolved_task()` unchanged.

- [ ] **Step 6: Run the runtime resolution tests**

  Run: `uv run pytest tests/test_portable_config.py -q`

  Expected: environment, local config, CLI priority, legacy warning, missing variable, and resolved-task field tests pass.

### Task 2: CLI Resolution and Strict Validation Boundary

**Files:**
- Modify: `src/grf_ue_bridge/cli.py`
- Modify: `src/grf_ue_bridge/config/resolver.py`
- Modify: `src/grf_ue_bridge/workflows/task_export.py`
- Test: `tests/test_portable_config.py`, `tests/test_task_cli.py`, `tests/test_task_resolver.py`

**Interfaces:**
- `_resolve_runtime(task, dataset_root=None, ue_project_root=None, local_config=None)` uses the single resolver boundary.
- Task commands expose optional `--dataset-root`, `--ue-project-root`, and `--local-config` options without changing existing positional arguments.
- `validate_task()` performs strict UE root validation and does not require dataset root to exist.
- `run_export()` creates `resolved.dataset_root` immediately before writing export output.

- [ ] **Step 1: Write failing CLI and strict validation tests**

  Add tests asserting `task resolve` accepts a task with a placeholder and local config, `task validate` rejects a UE root without `.uproject`, accepts a missing dataset root, and explicit CLI path overrides win over local config. Add a test that `run_export()` creates a missing dataset root using a stubbed export operation.

- [ ] **Step 2: Run the tests and verify the expected failures**

  Run: `uv run pytest tests/test_portable_config.py tests/test_task_cli.py tests/test_task_resolver.py -q`

  Expected: CLI options are unknown, strict UE validation is absent, or export does not create the dataset root.

- [ ] **Step 3: Add shared CLI options at the task command boundary**

  Add the three options to each task command that resolves a task: validate, resolve, export, ue-command, postprocess, audit, cleanup, and manifest if it currently resolves the task. Pass the values only through `_resolve_runtime()`; do not duplicate path resolution in individual handlers.

- [ ] **Step 4: Add strict `task validate` path checks**

  Extend `validate_task()` with resolved runtime path inputs. Require `ue_project_root` to be an existing directory containing at least one `.uproject` file. Permit a non-existent `dataset_root`. Return explicit failures for missing variables, malformed local config, missing runtime path values, and invalid UE project roots. Preserve existing schema/camera/frame/duration checks.

- [ ] **Step 5: Create dataset root only in export**

  In `run_export()`, call `Path(resolved.dataset_root).mkdir(parents=True, exist_ok=True)` at the export boundary before writing trajectory/provenance output. Do not add directory creation to resolver or validation.

- [ ] **Step 6: Run CLI integration tests**

  Run: `uv run pytest tests/test_portable_config.py tests/test_task_cli.py tests/test_task_resolver.py -q`

  Expected: all portable path and existing task resolver/CLI tests pass.

### Task 3: Templates, Documentation, and Compatibility Tests

**Files:**
- Create: `.futsalmot/local.example.json`
- Modify: `.gitignore` only if a narrower explicit ignore is needed; preserve `.futsalmot/` ignore
- Modify: `README.md`
- Modify: `docs/DATA_CONTRACT.md`
- Modify: `docs/VALIDATION_AND_LIMITATIONS.md` only for path-resolution behavior if needed
- Test: `tests/test_portable_config.py`

**Interfaces:**
- Template documents `paths.dataset_root` and `paths.ue_project_root` with portable placeholder values.
- Existing absolute-path tasks still resolve and emit a deprecation warning.
- Existing resolved-task JSON remains loadable by `load_resolved_task()` with unchanged schema/version.

- [ ] **Step 1: Add failing template/documentation and compatibility assertions**

  Test that the example template exists, contains both required path keys, and is not treated as the real local config. Test round-trip save/load of a resolved task created from a portable task. Test old absolute task values still resolve and warning text is exposed by the resolver/CLI path.

- [ ] **Step 2: Implement the template and documentation**

  Add `.futsalmot/local.example.json` with `${FUTSALMOT_DATASET_ROOT}` and `${FUTSALMOT_UE_ROOT}` examples. Document the three layers, commit/ignore rules, precedence, CLI examples, environment variables, legacy fallback warning, and the distinction between resolve-time and validate/export-time checks.

- [ ] **Step 3: Run documentation and compatibility tests**

  Run: `uv run pytest tests/test_portable_config.py tests/test_task_resolver.py -q`

  Expected: all pass.

### Task 4: Full Verification and Delivery

**Files:**
- Test: all existing tests and `tests/test_portable_config.py`

- [ ] **Step 1: Run focused regression tests**

  Run: `uv run pytest tests/test_portable_config.py tests/test_task_cli.py tests/test_task_resolver.py tests/test_task_config.py -q`

- [ ] **Step 2: Run the complete default suite**

  Run: `uv run pytest`

  Do not claim GRF integration or Unreal Editor coverage unless separately executed.

- [ ] **Step 3: Run static hygiene checks**

  Run: `git diff --check`

  Confirm no generated `.futsalmot/local.json`, `__pycache__`, `.pyc`, runtime JSON, UE assets, or MCP files are staged.

- [ ] **Step 4: Inspect and commit only intended inner-repository files**

  Run: `git status --short --branch`, `git diff --stat`, and `git diff --cached --check`. Stage only the Portable Config implementation, tests, template, docs, and this plan. Commit with a Simplified Chinese message such as `实现可移植任务路径配置`.

- [ ] **Step 5: Push inner repository and verify remote HEAD**

  Run: `git push origin main`, then `git rev-parse HEAD` and `git ls-remote origin refs/heads/main`. Do not update or commit the outer UE submodule pointer unless explicitly requested in a separate step.
