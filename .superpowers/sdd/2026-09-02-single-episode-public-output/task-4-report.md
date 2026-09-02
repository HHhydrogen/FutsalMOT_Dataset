# Task 4 报告

## 实现

- `task postprocess` 默认调用 `public_episode.write_public_episode`，从 `render_mask/*.exr` 直接生成 JPG、MOT、MOTS、Pose、`seqinfo.ini` 和 `episode_manifest.json`，随后调用公开 validator。
- 保留 `postprocess.public_output=false` 的 cryptomatte、mask annotate、YOLO Pose 和 debug legacy 链路；全 skip 的旧 CLI 行为保持不变。
- public manifest 存在时 cleanup 使用公开 validator 作为门禁，失败时不删除；dry-run 仍为默认。canonical JPG、`gt/gt.txt`、`gt/gt_pose.json`、`gt/gt_mots.txt`、`seqinfo.ini` 和 manifest 不加入 transient 删除集合。
- public dataset manifest 的 final profile 只纳入 canonical JPG/GT/episode manifest，并统计公开 Pose/MOTS 记录；legacy episode 保留原 mask/YOLO checksum 兼容行为。

## 测试

`uv run pytest tests/test_task_cli.py tests/test_dataset_manifest.py tests/test_audit_fixes.py -q`

结果：`49 passed in 7.64s`

`uv run pytest`

结果：`607 passed, 7 deselected in 14.65s`

## 注意事项

- 公开 writer 依赖 UE 阶段已经产生 `annotations.jsonl`、`pose_keypoints.jsonl`、`camera.json` 和 Cryptomatte EXR；本任务未改变 UE 渲染流程。
- 未实现多 episode assembly，符合任务范围。

## Task 4 Review 修复

- public cleanup 现在同时检查公开 validator、render summary、pose session 和 audit 报告；public manifest 存在时 audit 失败不会被短路。
- JPEG 规范化使用临时文件和 `os.replace`，成功后删除源 PNG/JPEG，并统一为六位数字 `.jpg`；重复执行保持幂等。
- 增加真实公开 episode 后处理、真实 validator/cleanup fixture、audit 失败门禁、legacy `public_output=false`、JPEG 重跑和 public final profile 排除 transient/debug 的测试。

`uv run pytest tests/test_public_episode.py tests/test_task_cli.py::TestTaskStatusAudit::test_postprocess_defaults_to_public_writer tests/test_task_cli.py::TestTaskStatusAudit::test_postprocess_public_output_false_keeps_legacy_skip_behavior tests/test_jpeg_contract.py::test_cleanup_preserves_public_outputs_and_removes_render_after_public_validation tests/test_jpeg_contract.py::test_cleanup_blocks_real_public_fixture_when_audit_report_fails tests/test_dataset_manifest.py::TestChecksumProfiles::test_public_final_profile_excludes_transient_and_debug -q`

结果：`19 passed in 0.52s`

`uv run pytest`

结果：`611 passed, 7 deselected in 14.85s`
