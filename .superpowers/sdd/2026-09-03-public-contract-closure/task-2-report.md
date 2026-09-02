# Task 2 Report: Resolved-task Public Audit Contract

## 完成内容

- `validate_public_episode` 支持接收 resolved task 路径、字典或模型对象。
- 从 resolved task 的 `config_v3` 摘要读取公开契约，不新增用户层 schema。
- 公开 manifest 审计现在逐项比较：
  - exact camera IDs
  - public sequence names
  - expected frames per camera
  - requested annotations/modalities
  - requested classes
  - FPS
  - resolution
- `grf-ue task audit` 自动保存并传入当前 resolved task，使 Config v3 任务使用同一份解析契约。
- 没有 manifest 或没有 Config v3 contract 时保留原有 legacy audit 行为。

## 测试

先添加并运行了会失败的契约测试，确认原实现因 `validate_public_episode` 不接受 resolved task 而失败：7 个测试失败。

随后完成最小实现并运行：

- `uv run pytest tests/test_public_validator.py -q`
  - `29 passed`
- `uv run pytest tests/test_task_cli.py::TestTaskStatusAudit::test_audit_passes_minimal tests/test_task_cli.py::TestTaskStatusAudit::test_status_readonly -q`
  - `2 passed`
- `uv run pytest -q`
  - `730 passed, 7 deselected`
- `git diff --check`
  - 通过

新增回归覆盖匹配 contract，以及以下六类 mismatch：missing camera、short frame set、annotation mismatch、class mismatch、FPS mismatch、resolution mismatch。

## 提交

- Commit message: `补强resolved task公开审计契约`
- 未 push。
