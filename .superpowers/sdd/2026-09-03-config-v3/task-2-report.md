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

## Task 2 Review 修复

- 为 `mot`、`mots`、`pose` 分别测试并启用 Object-ID/instance-mask 依赖；pose 同时启用 `yolo_pose`，不新增 pose public format。
- 使用现有 `ExportConfig` 和 `UeProfile` 默认值构造 v3 legacy profile，保留 `trajectory_time_scale`、场地尺寸、渲染开关、球滚动等字段。
- v3 FPS 在解析前要求为正的 10 的倍数。
- local config 的空白路径在 `Path.resolve()` 前拒绝。
- 增加环境变量 fallback、缺少 `.uproject`、advanced simulation 和已有 legacy 字段保留测试。
- 删除未使用的 loader import。

## Review 修复测试

命令：

```text
uv run pytest tests/test_task_resolver.py -q
```

精确输出：

```text
................................                                         [100%]
32 passed in 0.33s
```

命令：

```text
uv run pytest
```

精确输出：

```text
============================= test session starts =============================
platform win32 -- Python 3.9.25, pytest-8.4.2, pluggy-1.6.0
rootdir: C:\Users\20113\AppData\Local\Temp\opencode\futsalmot-single-episode-public-output
configfile: pyproject.toml
collected 685 items / 7 deselected / 678 selected
tests\test_annotation_utils.py ................                          [  2%]
tests\test_annotation_validator.py ........                              [  3%]
tests\test_audit_fixes.py ....................                           [  6%]
tests\test_camera_projection.py .......................                  [  9%]
tests\test_coordinate_transform.py .................                     [ 12%]
tests\test_cryptomatte.py ....                                           [ 12%]
tests\test_dataset_export.py ..........                                  [ 14%]
tests\test_dataset_manifest.py ....................                      [ 17%]
tests\test_dataset_regression.py .........                               [ 18%]
tests\test_debug.py .........                                            [ 20%]
tests\test_entity_roles.py .........                                     [ 21%]
tests\test_golden_fixture.py ......                                      [ 22%]
tests\test_import_grf_episode.py .....                                   [ 23%]
tests\test_instance_mask.py ............................................ [ 29%]
..................                                                       [ 32%]
tests\test_interpolate.py .....................                          [ 35%]
tests\test_jpeg_contract.py ........                                     [ 36%]
tests\test_mask_annotator.py ...........................                 [ 40%]
tests\test_motion_quality.py ....                                        [ 41%]
tests\test_perf_optimizations.py ...................................     [ 46%]
tests\test_player_motion.py ............................................ [ 52%]
.............................                                            [ 56%]
tests\test_pose_annotator.py .....................                       [ 60%]
tests\test_pose_bones.py ...............                                 [ 62%]
tests\test_pose_validator.py ...........                                 [ 63%]
tests\test_public_episode.py ...................                         [ 66%]
tests\test_public_validator.py ..............                            [ 68%]
tests\test_render_export.py ...........................                  [ 72%]
tests\test_render_preset.py ..................................           [ 77%]
tests\test_repository_hygiene.py .........                               [ 79%]
tests\test_schema.py ...........                                         [ 80%]
tests\test_seeds.py ................                                     [ 83%]
tests\test_task_cli.py ...........                                       [ 84%]
tests\test_task_config.py .............................................. [ 91%]
........                                                                 [ 92%]
tests\test_task_resolver.py ................................             [ 97%]
tests\test_ue_resolved_task.py .....                                     [ 98%]
tests\test_validator.py .............                                    [100%]

===================== 678 passed, 7 deselected in 12.07s =====================
```

## Review 修复提交

提交信息：`修复Config v3解析审查问题`

未推送。

## Task 2 剩余问题修复

- `V3SimulationConfig` 现在允许并保留 `trajectory_time_scale`、左右队 agent 控制数和 `ball_rolling`。
- v3 使用现有 `ExportConfig`、`UeProfile` 默认值，并仅覆盖 v3 明确派生或指定的字段。
- 新增共享 `validate_supported_fps()`，由 `validate_task()` 和 `resolve_task()` 同时调用；v2 校验行为保持不变。
- 新增 advanced simulation 到 resolved task 的传播测试，以及 validate 阶段非法 FPS 测试。

## 剩余问题修复测试

命令：

```text
uv run pytest tests/test_task_resolver.py -q
```

精确输出：

```text
..................................                                       [100%]
34 passed in 0.29s
```

命令：

```text
uv run pytest
```

精确输出：

```text
============================= test session starts =============================
platform win32 -- Python 3.9.25, pytest-8.4.2, pluggy-1.6.0
rootdir: C:\Users\20113\AppData\Local\Temp\opencode\futsalmot-single-episode-public-output
configfile: pyproject.toml
collected 687 items / 7 deselected / 680 selected
tests\test_annotation_utils.py ................                          [  2%]
tests\test_annotation_validator.py ........                              [  3%]
tests\test_audit_fixes.py ....................                           [  6%]
tests\test_camera_projection.py .......................                  [  9%]
tests\test_coordinate_transform.py .................                     [ 12%]
tests\test_cryptomatte.py ....                                           [ 12%]
tests\test_dataset_export.py ..........                                  [ 14%]
tests\test_dataset_manifest.py ....................                      [ 17%]
tests\test_dataset_regression.py .........                               [ 18%]
tests\test_debug.py .........                                            [ 20%]
tests\test_entity_roles.py .........                                     [ 21%]
tests\test_golden_fixture.py ......                                      [ 22%]
tests\test_import_grf_episode.py .....                                   [ 22%]
tests\test_instance_mask.py ............................................ [ 29%]
..................                                                       [ 32%]
tests\test_interpolate.py .....................                          [ 35%]
tests\test_jpeg_contract.py ........                                     [ 36%]
tests\test_mask_annotator.py ...........................                 [ 40%]
tests\test_motion_quality.py ....                                        [ 40%]
tests\test_perf_optimizations.py ...................................     [ 46%]
tests\test_player_motion.py ............................................ [ 52%]
.............................                                            [ 56%]
tests\test_pose_annotator.py .....................                       [ 59%]
tests\test_pose_bones.py ...............                                 [ 62%]
tests\test_pose_validator.py ...........                                 [ 63%]
tests\test_public_episode.py ...................                         [ 66%]
tests\test_public_validator.py ..............                            [ 68%]
tests\test_render_export.py ...........................                  [ 72%]
tests\test_render_preset.py ..................................           [ 77%]
tests\test_repository_hygiene.py .........                               [ 78%]
tests\test_schema.py ...........                                         [ 80%]
tests\test_seeds.py ................                                     [ 82%]
tests\test_task_cli.py ...........                                       [ 84%]
tests\test_task_config.py .............................................. [ 91%]
........                                                                 [ 92%]
tests\test_task_resolver.py ..................................           [ 97%]
tests\test_ue_resolved_task.py .....                                     [ 98%]
tests\test_validator.py .............                                    [100%]

===================== 680 passed, 7 deselected in 11.74s =====================
```

## 剩余问题修复提交

提交信息：`补充Config v3高级参数校验`

未推送。
