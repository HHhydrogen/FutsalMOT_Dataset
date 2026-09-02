# Task 2 实施报告

## 状态

已完成 Task 2。实现了 local config 加载与优先级解析、Config v3 到旧版运行时字段的推导、v3 摘要，以及 local config 路径门禁。v2 配置路径保持兼容。

## 主要变更

- `load_local_machine_config(path)` 使用严格的 `LocalMachineConfig` 模型加载 JSON，拒绝任务字段。
- `resolve_local_config(cli_path, env)` 按显式 CLI 路径、`FUTSALMOT_LOCAL_CONFIG`、报错的顺序选择配置，不自动搜索邻近文件。
- v3 解析校验 `dataset_root`、`ue_project_root` 目录，并要求 UE 根目录恰好包含一个 `.uproject` 文件。
- v3 映射到现有 `export_profile`、`ue_profile`、`postprocess`、`audit`、`actor_mapping` 和 `artifact_policy` 字段。
- FPS、分辨率、相机映射、公开 Sequence 名称、标注类别、球类别和期望帧数均由 v3 单一来源推导。
- `ResolvedTask` 增加 `config_v3` 摘要字段；旧 public GT 输出契约未修改。
- 新增 resolver 测试，覆盖 v3 推导、优先级、禁止邻近搜索、路径失败、不覆盖已有 resolved 文件和 summary。

## TDD 证据

先加入测试并运行 focused suite，得到预期失败：

```text
7 failed, 13 passed in 0.33s
```

失败原因为 v3 resolver 参数、local config 选择和 summary 尚未实现。随后实现最小行为并完成回归验证。

## 测试

命令：

```text
uv run pytest tests/test_task_resolver.py -q
```

输出：

```text
....................                                                     [100%]
20 passed in 0.22s
```

命令：

```text
uv run pytest
```

输出：

```text
============================= test session starts =============================
platform win32 -- Python 3.9.25, pytest-8.4.2, pluggy-1.6.0
collected 673 items / 7 deselected / 666 selected
===================== 666 passed, 7 deselected in 11.98s ======================
```

## 提交

提交信息：`实现Config v3本地配置解析`

未推送。

## 关注事项

- Task 3 仍需接入 CLI 的 `--local-config` 参数和 v3 summary 输出；本任务仅提供 resolver/loader 接口。
- v3 没有用户层 actor mapping 字段，因此当前保留旧默认路径 `ue/actor_mapping.example.json`，由后续运行时流程继续解析。
