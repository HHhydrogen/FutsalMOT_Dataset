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

## Task 4 Review 修复

- `dataset_manifest.profile_file_paths`、`collect_episode` 和 `build_manifest` 现在接受 resolved task，并按 `config_v3.annotations` 选择公开 GT/checksum 内容；没有 v3 摘要时保留 v2 fallback。
- dataset manifest episode 条目新增 annotations、modalities、classes、camera mapping 和 public sequence names，来源支持部分 resolved 对象及 `ue_profile` 序列 fallback。
- player-only 配置不再声明 football 的 MOT/global/class policy；保留 player policy。包含 ball 时才声明 ball policy。
- 新增 dataset manifest 真实 API、部分 resolved fallback、sequence metadata、cleanup status 和 player-only policy 测试。

验证：

`uv run pytest tests/test_dataset_manifest.py tests/test_jpeg_contract.py tests/test_task_resolver.py -q`

结果：`82 passed, 8 warnings in 3.83s`。

`uv run pytest`

结果：`746 passed, 7 deselected, 62 warnings in 12.58s`。

`git diff --check`：通过。
