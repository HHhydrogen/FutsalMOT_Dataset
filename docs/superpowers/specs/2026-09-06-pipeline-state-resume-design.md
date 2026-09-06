# Pipeline State + Resume Design

## Goal

为现有 FutsalMOT task workflow 增加执行进度记录、状态查看和失败恢复能力，同时保持 ValidationResult、TaskRequirements、ResolvedTask、Audit 和 Cleanup contract 不变。

## Scope

本设计只记录 pipeline 执行进度，不重写现有 workflow，不修改 artifact 格式，不实现 Pipeline State 之外的调度、分布式执行、Resume DAG 或 UE 侧状态系统。

不修改：

- Task schema 大结构
- ResolvedTask schema
- TaskRequirements
- ValidationResult
- UE Blueprint、UE Asset、MCP
- 现有 cleanup artifact 删除集合

## Responsibilities

```text
Task / local runtime config
        |
        v
ResolvedTask       负责运行时绝对路径
        |
        +--> PipelineState       负责执行到哪里
        |
        +--> ValidationResult    负责结果是否正确
```

Pipeline State 的 `COMPLETED` 只表示所有执行步骤结束，不表示数据集正确。最终数据正确性仍由 Audit 产生的 `ValidationResult.passed` 决定。

## State Storage

状态文件固定为：

```text
.futsalmot/runtime/<task_id>/pipeline_state.json
```

该路径属于 runtime layer，不进入 Git。`.futsalmot/local.example.json` 是唯一允许提交的模板；`.futsalmot/local.json`、`runtime/`、日志和其它生成物继续忽略。

## State Schema

```json
{
  "schema": "futsalmot_pipeline_state",
  "version": 1,
  "task_id": "episode001",
  "state": "RUNNING",
  "steps": {
    "export": {
      "status": "COMPLETED",
      "completion_type": "workflow_returned_zero",
      "note": "轨迹导出完成",
      "started_at": "2026-09-06T00:00:00Z",
      "completed_at": "2026-09-06T00:01:00Z",
      "error": null
    },
    "ue_sequence": {
      "status": "COMPLETED",
      "completion_type": "command_generation",
      "note": "仅表示 UE command 已生成，不表示 UE/MRQ/render 完成"
    },
    "render": {"status": "PENDING"},
    "postprocess": {"status": "PENDING"},
    "audit": {"status": "PENDING"},
    "cleanup": {"status": "PENDING"}
  },
  "updated_at": "2026-09-06T00:01:00Z",
  "history": []
}
```

固定步骤和顺序：

```text
export
ue_sequence
render
postprocess
audit
cleanup
```

步骤状态固定为：`PENDING`、`RUNNING`、`COMPLETED`、`FAILED`、`SKIPPED`。不设计动态 DAG。

## State API

新增 `src/grf_ue_bridge/pipeline_state.py`，提供无 UE 依赖的纯 Python API：

```python
create_state(task_id: str) -> PipelineState
load_state(path: Path) -> PipelineState
save_state(state: PipelineState, path: Path) -> None
update_step(state, step: str, status: StepStatus, **detail) -> None
mark_failed(state, step: str, exc: BaseException) -> None
state_path(repo_root: Path, task_id: str) -> Path
resume_plan(state: PipelineState) -> dict[str, str]
```

`save_state()` 使用同目录临时文件、flush/fsync（如平台支持）和 `os.replace()` 原子替换。写入失败不得破坏旧的有效 JSON。

状态更新规则：

```text
PENDING -> RUNNING -> COMPLETED
PENDING -> RUNNING -> FAILED
PENDING -> SKIPPED
FAILED  -> RUNNING -> COMPLETED/FAILED
RUNNING -> 根据 recovery policy 变为 retry/keep/complete
```

已失败的 step 不得在同一次执行中被标记为 completed，除非下一次明确 retry 后 workflow 返回成功。

## Workflow Integration

接入位置是现有 CLI workflow 边界，保留现有 workflow 函数和返回码：

- `task export`：包裹 `run_export()`，成功 `COMPLETED`，异常或非零返回 `FAILED`。
- `task ue-command`：command generation 成功时 `ue_sequence=COMPLETED`，`completion_type=command_generation`；`render` 不因命令生成而完成，保持 `PENDING`。
- `task postprocess`：包裹 `run_postprocess()`，成功/失败写入 `postprocess`。
- `task audit`：包裹 audit command，命令执行完成写入 `audit=COMPLETED/FAILED`；report 内的 `ValidationResult.passed` 单独展示，不映射为 pipeline execution success。
- `task cleanup`：包裹 cleanup command，保留现有 Cleanup Gate 和返回码；成功/失败写入 `cleanup`。

对于异常，step detail 至少记录 `error_type` 和 `error_message`。对于非零返回但没有异常，记录统一的 workflow return-code error。

## Resume Policy

新增 `task resume TASK`，读取同一 `pipeline_state.json`，调用现有 workflow，不复制 workflow 实现。

基础规则：

| 状态 | 行为 |
| --- | --- |
| `COMPLETED` | skip |
| `FAILED` | retry |
| `PENDING` | execute |
| `SKIPPED` | keep |
| `RUNNING` | 视为 unknown，按 step-specific recovery policy 处理 |

`RUNNING` 恢复策略：

- `export`：关键轨迹输出不完整时 retry；若输出已存在但没有可验证的完整结果，不把它标记 completed，避免误判。
- `ue_sequence`：不自动假定 UE 已执行；保留 completed 只表示 command generation 已完成。
- `render`：只有 `render_summary.json` 明确成功且可读取时才 complete，否则 retry/keep 为 pending，不把 RUNNING 当成功。
- `postprocess`：通过现有 annotation/pose validation 可证明结果完整时 complete，否则 retry。
- `audit`：canonical audit report 可读取时 complete，报告是否 passed 单独展示；报告缺失或 malformed 时 retry。
- `cleanup`：只有 cleanup manifest/status 明确表示已应用时 complete，否则重新调用现有 cleanup gate。

Resume 按固定顺序处理步骤，跳过已完成和保留 skipped；retry/execute 的 step 通过已有 CLI workflow callable 执行。Resume 不绕过 Audit 或 Cleanup Gate，不把 pipeline state 当作 validation result。

## Status CLI

`task status TASK` 优先读取 pipeline state，并保留现有 artifact summary。输出至少包含：

- task id / episode name
- pipeline overall state
- 每个固定 step 的状态
- failed step、error type/message 和建议 action
- audit/ValidationResult 的 passed 状态（如果已有 audit report）

没有 state 文件时显示固定六步为 `PENDING`，并标注 state 尚未创建；不能通过 artifact 数量臆造 completed。

## Overall State

整体 pipeline state 的计算只基于步骤执行状态：

- 任一步骤 `FAILED`：`FAILED`
- 任一步骤 `RUNNING`：`RUNNING`
- 仍有 `PENDING`：`PENDING` 或 `RUNNING`，按当前执行上下文决定
- 所有步骤为 `COMPLETED` 或 `SKIPPED`：`COMPLETED`

若所有执行步骤完成但 Audit 的 `ValidationResult.passed=false`，status 必须同时显示：

```text
Pipeline State: COMPLETED
Validation: FAIL
Task success: NO
```

## Tests

新增 `tests/test_pipeline_state.py`，覆盖：

- 新建 state 的固定六步全部 `PENDING`。
- PENDING → RUNNING → COMPLETED 生命周期。
- FAILED 记录异常类型和消息。
- 原子保存不会留下损坏 JSON，旧文件在替换失败时保持有效。
- resume plan 跳过 completed、重试 failed、执行 pending、保留 skipped。
- RUNNING 按 step policy 视为 unknown，不盲目 complete。
- status 输出状态、失败信息和建议 action。
- Pipeline State completed 与 Audit ValidationResult failed 可以同时存在且不报告任务成功。
- `task export`、`ue-command`、`postprocess`、`audit`、`cleanup` 的状态更新和失败回写。

## Compatibility

现有 task schema、resolved-task JSON、artifact 格式和 Validation/Audit/Cleanup contract 不变。UE 侧不需要知道 Pipeline State 存在；`ue-command` 只输出原有命令并额外在 P1 runtime 写状态。
