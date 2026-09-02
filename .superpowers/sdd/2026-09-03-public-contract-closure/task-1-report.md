# Task 1 报告：保留显式相机身份

## 完成内容

- Config v3 拒绝重复 UE camera actor。
- resolver 在 `ue_profile.sequences` 中分别保留 `camera_id`、`camera_actor` 和 `public_sequence_name`。
- resolver 在 annotation export 中保留兼容的 actor 列表，并增加显式 `camera_mapping`。
- public postprocess 将 resolved sequence 的三类相机身份透传给 public writer。
- public writer 优先使用显式 `camera_id` 与 `public_sequence_name`，不再从 actor 名或配置顺序推断 v3 public ID。
- legacy writer 输入仍保留原有按目录/序号生成公开名称的行为。
- 新增 C03/C07 映射、重复 actor 和公开 sequence 名称测试。

## 测试

- RED：新增测试首次运行结果为 4 个失败，分别覆盖重复 actor、resolver 显式 camera_id、writer C03/C07 身份和重复 actor 门禁。
- focused：`uv run pytest tests/test_task_config.py tests/test_task_resolver.py tests/test_public_episode.py tests/test_task_cli.py -q`
  - 151 passed，28 warnings（既有 Config v2 弃用警告）。
- full：`uv run pytest`
  - 722 passed，7 deselected，62 warnings（既有 Config v2 弃用警告）。
- `git diff --check`：通过。

## 备注

- 未修改 v3 顶层 schema 形状或 frozen public GT 字段。
- 未 dispatch agent、未 push。
