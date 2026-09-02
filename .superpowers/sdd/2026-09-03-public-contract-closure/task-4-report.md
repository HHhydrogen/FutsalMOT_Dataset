# Task 4 报告

## 实现

- 新增统一的公开能力解析：Config v3 从 `config_v3.annotations`、`classes`、相机映射和公开序列名生成声明。
- `artifact_cleanup.build_manifest` 改为使用解析后的 annotations/modalities/classes/camera mapping/sequence metadata，不再硬编码全模态和 player+ball。
- `task_status.collect_status` 返回相同的公开能力声明，并读取已有 `dataset_manifest.json` 的 `cleanup_status`。
- 无 `config_v3` 的 v2/旧 resolved task 保留 `mot`、`pose`、`mots` 与 `player`、`ball` 的历史默认回退。
- 保留公开 GT 文件 schema；Pose 未请求时 manifest 的 `pose_schema` 为 `null`。

## TDD

先加入 mot-only/player-only 与 mot+mots/player+ball 的 manifest/status 测试，以及 v2 回退测试。

首次运行：

`uv run pytest tests/test_jpeg_contract.py::test_manifest_and_status_reflect_resolved_public_capabilities tests/test_jpeg_contract.py::test_manifest_and_status_keep_legacy_capability_defaults -q`

结果：`3 failed`，失败原因为 manifest 缺少动态 `annotations` 字段，确认测试捕获了未实现行为。

实现后同一 focused tests：`3 passed in 0.24s`。

## 验证

`uv run pytest tests/test_jpeg_contract.py tests/test_task_resolver.py tests/test_dataset_manifest.py tests/test_task_cli.py -q`

结果：`107 passed, 18 warnings in 3.96s`。

`uv run pytest`

结果：`741 passed, 7 deselected, 62 warnings in 12.31s`。

`git diff --check`

结果：通过，无 whitespace errors。

## 变更文件

- `src/grf_ue_bridge/workflows/artifact_cleanup.py`
- `src/grf_ue_bridge/workflows/task_status.py`
- `tests/test_jpeg_contract.py`
- `.superpowers/sdd/2026-09-03-public-contract-closure/task-4-report.md`
