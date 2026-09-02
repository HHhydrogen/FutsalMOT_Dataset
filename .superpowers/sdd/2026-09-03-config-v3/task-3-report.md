# Task 3 报告

## 实现

- `task validate` 和 `task resolve` 新增 `--local-config`，Config v3 按 CLI 参数、`FUTSALMOT_LOCAL_CONFIG` 环境变量顺序选择本机配置，不自动搜索邻近文件。
- Config v3 解析失败时在写入 `resolved-task.json` 前退出，不创建或覆盖运行时文件。
- Config v2 保持无本机配置可用，并通过 `DeprecationWarning` 与 CLI warning 提示迁移；Config v3 不触发该 warning。
- `task resolve` 使用 resolver 摘要输出 episode、seed/steps、FPS、分辨率、相机 actor 映射、期望帧数、annotations、classes 和公开 sequence 名称。
- 增加 CLI 显式路径、环境回退、优先级、缺失配置、摘要、v2 warning 和失败不写运行时文件测试。

## 测试

命令：

```text
uv run pytest tests/test_task_cli.py -q
```

输出摘要：

```text
..................                                                       [100%]
18 passed, 9 warnings in 0.46s
```

命令：

```text
uv run pytest
```

输出摘要：

```text
============================= 687 passed, 7 deselected, 59 warnings in 11.81s =============================
```

warning 均为预期的 Config v2 弃用提示。

## 提交

提交：`待提交`
