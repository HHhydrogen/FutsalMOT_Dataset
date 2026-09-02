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
