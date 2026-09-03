# Config v3 Public Contract Closure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复 Config v3 的 Camera ID 贯穿、resolved-task 对照审计、可选 annotation cleanup gate 和动态 manifest/status 声明，同时清理文档残留。

**Architecture:** 保留现有 Config v3 schema 和 frozen public GT schema。resolver 输出显式 `camera_id -> camera_actor -> public_sequence_name` 关系；public writer/manifest/status 消费 resolved 摘要；audit 读取 resolved task 做契约比对；cleanup 按 requested annotations/classes 条件化 gate。

**Tech Stack:** Python 3.9、Pydantic、Typer、pytest、JSON、现有 Unreal 5.8 pipeline。

**Spec:** 用户本轮请求；不得修改 v3 主结构、public GT 格式、batch、Camera Manager、动画或 MRQ 新功能。

## Global Constraints

- `C03 -> FrontCamera` 必须生成 `FutsalMOT_<episode>_C03`；不得由 actor 名称或顺序推断 public ID。
- 同一 UE Camera Actor 不得被多个 public Camera ID 使用。
- public audit 必须对照 resolved task 检查 expected camera IDs、public sequence names、frames、annotations、classes、FPS、resolution。
- 只有 requested annotations 包含 `pose` 才要求 `pose_session.json` 和 Pose gate。
- manifest/status 按 resolved task 的 annotations/classes 生成，不硬编码全模态/全类别。
- public GT 文件结构保持不变。
- 文档必须以当前 Config v3 schema 为准。

---

### Task 1: Preserve Explicit Camera Identity

**Files:** `config/models.py`, `config/resolver.py`, `public_episode.py`, relevant camera tests.

- [ ] Add failing tests for C03/C07 mapping, duplicate actor rejection, and public sequence names.
- [ ] Run focused tests and confirm failure.
- [ ] Propagate `camera_id`, `camera_actor`, and `public_sequence_name` separately through resolved `ue_profile`, `config_v3`, and writer input.
- [ ] Reject duplicate actor values and never derive IDs from actor names/order when explicit mapping exists.
- [ ] Run focused camera tests and full related resolver/public tests.

### Task 2: Add Resolved-task Public Audit Contract

**Files:** `workflows/task_audit.py`, `public_validator.py`, `cli.py`, audit tests.

- [ ] Add failing tests for missing camera, short frame set, annotation/class/FPS/resolution mismatch.
- [ ] Load resolved task/config-v3 summary for public audit and compare exact expected values.
- [ ] Return audit FAIL for every missing/mismatched contract item with actionable errors.
- [ ] Preserve legacy audit when no public manifest/resolved v3 contract exists.
- [ ] Run focused audit tests.

### Task 3: Make Cleanup Gates Conditional

**Files:** `workflows/artifact_cleanup.py`, cleanup tests.

- [ ] Add failing mot-only and mot+mots cleanup tests without Pose files.
- [ ] Derive Pose gate from requested annotations; derive mask/annotation gates from requested modalities/classes where applicable.
- [ ] Keep canonical files protected and existing failed gates blocking.
- [ ] Run cleanup-focused tests.

### Task 4: Remove Hardcoded Capability Declarations

**Files:** `workflows/artifact_cleanup.py`, `dataset_manifest.py`, `task_status.py`, resolver/manifest tests.

- [ ] Add failing tests for dynamic annotations/classes in manifest and status.
- [ ] Use resolved `config_v3` values for actual public modalities/classes and sequence metadata.
- [ ] Keep legacy v2 fallback behavior unchanged.
- [ ] Run focused manifest/status tests.

### Task 5: Clean Config v3 Documentation and Verify

**Files:** `README.md`, `configs/README.md`, `CLAUDE.md`, `docs/REPRODUCIBILITY_AND_MANIFEST.md`, documentation tests if present.

- [ ] Remove v3-labeled legacy postprocess/ball/repeated-field examples and ensure v3 examples validate.
- [ ] Document explicit camera mapping and resolved audit/cleanup behavior.
- [ ] Run `uv run pytest` and `git diff --check`.
- [ ] Inspect status to ensure no real local config/runtime/secrets are tracked.
