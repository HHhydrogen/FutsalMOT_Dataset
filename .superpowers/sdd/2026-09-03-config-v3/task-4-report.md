# Task 4 报告

## 变更

- `tests/test_task_config.py`：增加 v2 加载必须发出匹配 `Config v2` 的 `DeprecationWarning` 回归测试。
- `tests/test_task_resolver.py`：增加 v2 无 local config 解析测试；增加 v3 resolved task 旧字段形状和 `validate_resolved_task()` 测试。
- `tests/test_ue_resolved_task.py`：增加 v2 resolved JSON 向 UE 和下游流程保留字段的测试。
- `src/grf_ue_bridge/config/loader.py`：无需修改，现有加载兼容逻辑已满足要求。

## 测试

命令：

```text
uv run pytest tests/test_task_config.py tests/test_task_resolver.py tests/test_task_cli.py tests/test_ue_resolved_task.py -q
```

输出：

```text
115 passed, 33 warnings in 0.83s
```

命令：

```text
uv run pytest
```

输出：

```text
=============== 691 passed, 7 deselected, 61 warnings in 11.81s ===============
```

警告均为既有 v2 `DeprecationWarning` 覆盖产生的预期警告；测试输出中的中文警告文本在当前终端显示为编码乱码，但 `pytest.warns(..., match="Config v2")` 已通过。
