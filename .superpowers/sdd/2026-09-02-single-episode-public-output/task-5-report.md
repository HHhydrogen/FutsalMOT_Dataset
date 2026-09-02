# Task 5 报告

## 状态

完成（文档与 P1 验证）；UE 端到端 smoke 受环境限制未完成。当前会话没有可用的
Unreal MCP server、toolset 或 status 接口，无法通过 `FutsalMOTTools` 在 Unreal Editor
Python 环境执行 UE workflow。

## 文档变更

- `README.md`：补充单 episode 公开输出树、JPG RGB quality=95、MOT 9 列、MOTS 6 字段与 COCO 压缩 RLE、Pose/足球约定、稳定 ID、默认排除项及 public/temporary/debug/internal 生命周期；准确记录 cleanup 的条件门禁、删除 allowlist，以及按路径匹配删除而不区分文件 provenance。
- `docs/REPRODUCIBILITY_AND_MANIFEST.md`：明确干净公开 postprocess 不生成重复派生物且不会删除 pre-existing transient 文件；cleanup 通过条件门禁后按显式路径 allowlist 删除匹配文件，不论其 provenance；保留单 episode 范围，不宣称 batch/split/assembler。
- `CLAUDE.md`：同步 task cleanup 默认命令和 validation-gated 输出契约。

## 验证

- `uv run grf-ue task validate configs/pose_smoke_3frames_1cam.json`：PASS。
- `uv run pytest`：613 passed, 7 deselected。
- `uv run grf-ue task export configs/pose_smoke_3frames_1cam.json`：成功，生成 `G:\FutsalMOT_Dataset\episode_pose_smoke` 的 `meta.json`、3 行 `frames.jsonl` 和 `provenance/`。
- `uv run grf-ue task ue-command configs/pose_smoke_3frames_1cam.json`：成功生成 `ue/run_task.py --resolved-task ...` 命令。
- `uv run grf-ue task audit configs/pose_smoke_3frames_1cam.json`：FAIL，因 UE 未运行，输出目录尚无 camera 且缺少 `render_summary.json`；报告位于 `G:\FutsalMOT_Dataset\episode_pose_smoke\audit\`。
- `uv run grf-ue task status configs/pose_smoke_3frames_1cam.json`：轨迹存在，camera=0，render_summary=None。
- cleanup 的默认 dry-run 与 validation gate：由 `tests/test_jpeg_contract.py` 覆盖，包括 canonical 输出保留、transient 删除和 public/audit gate 失败时拒绝 apply；未对未完成 UE smoke 执行真实 cleanup。
- 最终测试命令：`uv run pytest tests/test_jpeg_contract.py -q`；结果：`8 passed in 0.31s`（退出码 0）。

## 实际输出路径

轨迹输出：`G:\FutsalMOT_Dataset\episode_pose_smoke`。

公开 JPG/MOT/MOTS/Pose 输出未生成，因为 UE smoke 未执行。

## Commit

未提交。按仓库策略保留工作区变更供控制器处理。

## concerns

- 当前环境缺少 Unreal MCP 工具暴露；因此未执行 UE smoke、公开后处理、cleanup 和真实输出结构检查。这是具体的工具限制，不要求用户手动操作。
- 导出日志含现有 Gym 弃用警告，但命令返回成功，未影响产物生成。

## Final review fix wave

- 公开序列改为 `FutsalMOT_<episode_id>_C<two-digit-camera-id>`，manifest 使用数值
  `schema_version=1`、固定公开类别列表和 `track_id_policy`。
- writer 在临时 episode 树中完成所有相机的 canonical 文件、图片规范化和 manifest，
  最后统一替换；后续相机失败时保留原 episode。
- Player Pose 缺少投影输入时固定输出 17 个 `[0.0, 0.0, 0]`；JPG/JPEG 原字节复制，
  PNG 仅转为 quality=95 JPEG。

## Final test output

Command: `uv run pytest`

```text
============================= test session starts =============================
platform win32 -- Python 3.9.25, pytest-8.4.2, pluggy-1.6.0
rootdir: C:\Users\20113\AppData\Local\Temp\opencode\futsalmot-single-episode-public-output
configfile: pyproject.toml
collected 623 items / 7 deselected / 616 selected
tests\test_annotation_utils.py ................                          [  2%]
tests\test_annotation_validator.py ........                              [  3%]
tests\test_audit_fixes.py ....................                           [  7%]
tests\test_camera_projection.py .......................                  [ 10%]
tests\test_coordinate_transform.py .................                     [ 13%]
tests\test_cryptomatte.py ....                                           [ 14%]
tests\test_dataset_export.py ..........                                  [ 15%]
tests\test_dataset_manifest.py ....................                      [ 19%]
tests\test_dataset_regression.py .........                               [ 20%]
tests\test_debug.py .........                                            [ 22%]
tests\test_entity_roles.py .........                                     [ 23%]
tests\test_golden_fixture.py ......                                      [ 24%]
tests\test_import_grf_episode.py .....                                   [ 25%]
tests\test_instance_mask.py ............................................ [ 32%]
..................                                                       [ 35%]
tests\test_interpolate.py .....................                          [ 38%]
tests\test_jpeg_contract.py ........                                     [ 40%]
tests\test_mask_annotator.py ...........................                 [ 44%]
tests\test_motion_quality.py ....                                        [ 45%]
tests\test_perf_optimizations.py ...................................     [ 50%]
tests\test_player_motion.py ............................................ [ 57%]
.............................                                            [ 62%]
tests\test_pose_annotator.py .....................                       [ 66%]
tests\test_pose_bones.py ...............                                 [ 68%]
tests\test_pose_validator.py ...........                                 [ 70%]
tests\test_public_episode.py ...................                         [ 73%]
tests\test_public_validator.py .............                             [ 75%]
tests\test_render_export.py ...........................                  [ 79%]
tests\test_render_preset.py ..................................           [ 85%]
tests\test_repository_hygiene.py .........                               [ 86%]
tests\test_schema.py ...........                                         [ 88%]
tests\test_seeds.py ................                                     [ 91%]
tests\test_task_cli.py ...........                                       [ 93%]
tests\test_task_config.py ............                                   [ 94%]
tests\test_task_resolver.py .............                                [ 97%]
tests\test_ue_resolved_task.py .....                                     [ 97%]
tests\test_validator.py .............                                    [100%]
===================== 616 passed, 7 deselected in 14.72s ======================
```

Command: `uv run pytest` (final run)

```text
============================= test session starts =============================
platform win32 -- Python 3.9.25, pytest-8.4.2, pluggy-1.6.0
rootdir: C:\Users\20113\AppData\Local\Temp\opencode\futsalmot-single-episode-public-output
configfile: pyproject.toml
collected 624 items / 7 deselected / 617 selected

tests\test_annotation_utils.py ................                          [  2%]
tests\test_annotation_validator.py ........                              [  3%]
tests\test_audit_fixes.py ....................                           [  7%]
tests\test_camera_projection.py .......................                  [ 10%]
tests\test_coordinate_transform.py .................                     [ 13%]
tests\test_cryptomatte.py ....                                           [ 14%]
tests\test_dataset_export.py ..........                                  [ 15%]
tests\test_dataset_manifest.py ....................                      [ 19%]
tests\test_dataset_regression.py .........                               [ 20%]
tests\test_debug.py .........                                            [ 22%]
tests\test_entity_roles.py .........                                     [ 23%]
tests\test_golden_fixture.py ......                                      [ 24%]
tests\test_import_grf_episode.py .....                                   [ 25%]
tests\test_instance_mask.py ............................................ [ 32%]
..................                                                       [ 35%]
tests\test_interpolate.py .....................                          [ 38%]
tests\test_jpeg_contract.py ........                                     [ 40%]
tests\test_mask_annotator.py ...........................                 [ 44%]
tests\test_motion_quality.py ....                                        [ 45%]
tests\test_perf_optimizations.py ...................................     [ 50%]
tests\test_player_motion.py ............................................ [ 57%]
.............................                                            [ 62%]
tests\test_pose_annotator.py .....................                       [ 65%]
tests\test_pose_bones.py ...............                                 [ 68%]
tests\test_pose_validator.py ...........                                 [ 70%]
tests\test_public_episode.py ...................                         [ 73%]
tests\test_public_validator.py ..............                            [ 75%]
tests\test_render_export.py ...........................                  [ 79%]
tests\test_render_preset.py ..................................           [ 85%]
tests\test_repository_hygiene.py .........                               [ 86%]
tests\test_schema.py ...........                                         [ 88%]
tests\test_seeds.py ................                                     [ 91%]
tests\test_task_cli.py ...........                                       [ 93%]
tests\test_task_config.py ............                                   [ 94%]
tests\test_task_resolver.py .............                                [ 97%]
tests\test_ue_resolved_task.py .....                                     [ 97%]
tests\test_validator.py .............                                    [100%]

===================== 617 passed, 7 deselected in 14.41s ======================
```

Command: `git diff --check`

```text
warning: in the working copy of 'CLAUDE.md', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'README.md', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'docs/REPRODUCIBILITY_AND_MANIFEST.md', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'src/grf_ue_bridge/public_episode.py', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'src/grf_ue_bridge/public_validator.py', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'tests/test_jpeg_contract.py', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'tests/test_public_episode.py', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'tests/test_public_validator.py', LF will be replaced by CRLF the next time Git touches it
```

## Final Validator Fix Verification

Command: `uv run pytest`

Exact result:

```text
=============== 719 passed, 7 deselected, 62 warnings in 12.05s ===============
```

Command: `git diff --check`

Exact result:

```text
warning: in the working copy of 'src/grf_ue_bridge/config/models.py', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'src/grf_ue_bridge/public_validator.py', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'tests/test_public_validator.py', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'tests/test_task_config.py', LF will be replaced by CRLF the next time Git touches it
```
