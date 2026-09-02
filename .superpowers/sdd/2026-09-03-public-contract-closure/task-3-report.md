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

## Review 修复

- 空或缺失 `config_v3` 的 v2/legacy resolved task 继续要求 `render_summary.json` 和 `pose_session.json`。
- 新增 v2 apply cleanup 阻断回归测试。
- 新增 v3 `mot`、`mot+mots` 无 Pose 文件通过测试，以及 v3 `pose` 缺少 Pose 文件阻断测试。
- 修正 public validator monkeypatch 签名，并断言 `resolved_task` 参数确实传入。

## Review 修复后的精确测试输出

命令：`uv run pytest tests/test_jpeg_contract.py -q`

```text
............                                                             [100%]
12 passed in 0.30s
```

命令：`uv run pytest -q`

```text
........................................................................ [  9%]
........................................................................ [ 19%]
........................................................................ [ 29%]
........................................................................ [ 39%]
........................................................................ [ 49%]
........................................................................ [ 58%]
........................................................................ [ 68%]
........................................................................ [ 78%]
........................................................................ [ 88%]
........................................................................ [ 98%]
..............                                                           [100%]
============================== warnings summary ===============================
tests/test_repository_hygiene.py: 28 warnings
  C:\Users\20113\AppData\Local\Temp\opencode\futsalmot-single-episode-public-output\tests\test_repository_hygiene.py:77: DeprecationWarning: Config v2 deprecated����Ǩ�Ƶ� Config v3����ͨ�� --local-config �� FUTSALMOT_LOCAL_CONFIG �ṩ�������á�
    loader.load_task_config(tf)  # �ɽ�������

tests/test_task_cli.py::TestTaskValidateCLI::test_validate_pass
tests/test_task_cli.py::TestTaskValidateCLI::test_validate_fail_missing_dataset_root
tests/test_task_config.py::TestTaskSchema::test_missing_machine_path_fails
tests/test_task_resolver.py::TestResolvePaths::test_wrong_camera_expectation
tests/test_task_resolver.py::TestResolvePaths::test_missing_dataset_root_fails
  C:\Users\20113\AppData\Local\Temp\opencode\futsalmot-single-episode-public-output\src\grf_ue_bridge\config\resolver.py:63: DeprecationWarning: Config v2 deprecated����Ǩ�Ƶ� Config v3����ͨ�� --local-config �FUTSALMOT_LOCAL_CONFIG �ṩ�������á�
    task = loader.load_task_config(task_file)

tests/test_task_cli.py: 8 warnings
tests/test_task_resolver.py: 6 warnings
tests/test_ue_resolved_task.py: 6 warnings
  C:\Users\20113\AppData\Local\Temp\opencode\futsalmot-single-episode-public-output\src\grf_ue_bridge\config\resolver.py:119: DeprecationWarning: Config v2 deprecated����Ǩ�Ƶ� Config v3����ͨ�� --local-config �ṩ�������á�
    task = loader.load_task_config(task_file)

tests/test_task_config.py::TestTaskSchema::test_valid_task
  C:\Users\20113\AppData\Local\Temp\opencode\futsalmot-single-episode-public-output\tests\test_task_config.py:63: DeprecationWarning: Config v2 deprecated����Ǩ�Ƶ� Config v3����ͨ�� --local-config �ṩ�������á�
    t = loader.load_task_config(tf)

tests/test_task_config.py::TestTaskSchema::test_bad_task_id_rejected
  C:\Users\20113\AppData\Local\Temp\opencode\futsalmot-single-episode-public-output\tests\test_task_config.py:80: DeprecationWarning: Config v2 deprecated����Ǩ�Ƶ� Config v3����ͨ�� --local-config �ṩ�������á�
    loader.load_task_config(tf)

tests/test_task_config.py::TestTaskSchema::test_bad_episode_name_rejected
  C:\Users\20113\AppData\Local\Temp\opencode\futsalmot-single-episode-public-output\tests\test_task_config.py:85: DeprecationWarning: Config v2 deprecated����Ǩ�Ƶ� Config v3����ͨ�� --local-config �FUTSALMOT_LOCAL_CONFIG �ṩ�������á�
    loader.load_task_config(tf)

tests/test_task_config.py::TestTaskSchema::test_bad_workers_rejected
  C:\Users\20113\AppData\Local\Temp\opencode\futsalmot-single-episode-public-output\tests\test_task_config.py:90: DeprecationWarning: Config v2 deprecated����Ǩ�Ƶ� Config v3����ͨ�� --local-config �ṩ�������á�
    loader.load_task_config(tf)

tests/test_task_config.py::TestTaskSchema::test_unsupported_format_rejected
  C:\Users\20113\AppData\Local\Temp\opencode\futsalmot-single-episode-public-output\tests\test_task_config.py:95: DeprecationWarning: Config v2 deprecated����Ǩ�Ƶ� Config v3����ͨ�� --local-config �ṩ�������á�
    loader.load_task_config(tf)  # load ʱ������ validate_formats

tests/test_task_config.py::TestTaskSchema::test_missing_export_block_fails
  C:\Users\20113\AppData\Local\Temp\opencode\futsalmot-single-episode-public-output\tests\test_task_config.py:110: DeprecationWarning: Config v2 deprecated����Ǩ�Ƶ� Config v3����ͨ�� --local-config �ṩ�������á�
    loader.load_task_config(tf)

tests/test_task_config.py::TestTaskSchema::test_export_scenario_overrides
  C:\Users\20113\AppData\Local\Temp\opencode\futsalmot-single-episode-public-output\tests\test_task_config.py:121: DeprecationWarning: Config v2 deprecated����Ǩ�Ƶ� Config v3����ͨ�� --local-config �提供�������á�
    t = loader.load_task_config(tf)

tests/test_task_config.py::TestTaskSchema::test_export_defaults_none
  C:\Users\20113\AppData\Local\Temp\opencode\futsalmot-single-episode-public-output\tests\test_task_config.py:129: DeprecationWarning: Config v2 deprecated����Ǩ�Ƶ� Config v3����ͨ�� --local-config �ṩ�������á�
    t = loader.load_task_config(tf)

tests/test_task_config.py::TestTaskSchema::test_bad_difficulty_rejected
  C:\Users\20113\AppData\Local\Temp\opencode\futsalmot-single-episode-public-output\tests\test_task_config.py:140: DeprecationWarning: Config v2 deprecated����Ǩ�Ƶ� Config v3����ͨ�� --local-config �ṩ�������á�
    loader.load_task_config(tf)

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
734 passed, 7 deselected, 62 warnings in 12.25s
```

## 无警告精确输出

命令：`uv run pytest tests/test_jpeg_contract.py -q --disable-warnings`

```text
............                                                             [100%]
12 passed in 0.30s
```

命令：`uv run pytest -q --disable-warnings`

```text
........................................................................ [  9%]
........................................................................ [ 19%]
........................................................................ [ 29%]
........................................................................ [ 39%]
........................................................................ [ 49%]
........................................................................ [ 58%]
........................................................................ [ 68%]
........................................................................ [ 78%]
........................................................................ [ 88%]
........................................................................ [ 98%]
..............                                                           [100%]
734 passed, 7 deselected, 62 warnings in 12.29s
```

## 最终精确输出

命令：`uv run pytest tests/test_jpeg_contract.py -q --disable-warnings`

```text
.............                                                            [100%]
13 passed in 0.32s
```

命令：`uv run pytest -q --disable-warnings`

```text
........................................................................ [  9%]
........................................................................ [ 19%]
........................................................................ [ 29%]
........................................................................ [ 39%]
........................................................................ [ 48%]
........................................................................ [ 58%]
........................................................................ [ 68%]
........................................................................ [ 78%]
........................................................................ [ 88%]
........................................................................ [ 97%]
...............                                                          [100%]
735 passed, 7 deselected, 62 warnings in 12.38s
```

## 剩余测试缺口修复后的精确输出

命令：`uv run pytest tests/test_jpeg_contract.py -k 'real_v3_public' -q`

```text
..                                                                       [100%]
2 passed, 14 deselected in 0.23s
```

命令：`uv run pytest tests/test_jpeg_contract.py -q --disable-warnings`

```text
................                                                         [100%]
16 passed in 0.33s
```

命令：`uv run pytest -q --disable-warnings`

```text
........................................................................ [  9%]
........................................................................ [ 19%]
........................................................................ [ 29%]
........................................................................ [ 39%]
........................................................................ [ 48%]
........................................................................ [ 58%]
........................................................................ [ 68%]
........................................................................ [ 78%]
........................................................................ [ 87%]
........................................................................ [ 97%]
..................                                                       [100%]
738 passed, 7 deselected, 62 warnings in 12.26s
```
