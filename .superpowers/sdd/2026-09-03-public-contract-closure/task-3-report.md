# Task 3：条件化 Cleanup Gate 报告

## 完成内容

- `artifact_cleanup` 增加可选 resolved task 输入，从 `config_v3.annotations` 和 `config_v3.classes` 读取公开请求。
- Config v3 任务只有在请求 `pose` 时才要求 `pose_session.json` 且校验 `capture_complete`。
- Config v3 任务仍要求 `render_summary.json` 成功；现有 audit 失败仍阻止 cleanup。
- CLI cleanup 将 resolved task 传入 dry-run 和 apply 两条路径，并将同一契约传入 public validator。
- 未提供 Config v3 annotations 的旧调用继续使用原有严格 render/Pose gate 行为。

## 测试

先新增并运行了两个失败测试，确认旧实现因 `plan_cleanup()` 不接受 resolved task 而失败。

随后实现最小修改并通过：

- `uv run pytest tests/test_jpeg_contract.py -q`：10 passed
- `uv run pytest -q`：732 passed，7 deselected，62 个既有弃用警告
- `git diff --check`：通过

新增回归覆盖：

- `mot` 任务缺少 Pose 文件时可 dry-run/apply cleanup。
- `mot+mots` 任务缺少 Pose 文件时可 dry-run/apply cleanup。

## 提交

提交信息：`条件化清理门禁`
