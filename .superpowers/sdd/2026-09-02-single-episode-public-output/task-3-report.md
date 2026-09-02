# Task 3 报告

## 实现

- 新增 `src/grf_ue_bridge/public_validator.py`，提供 `validate_public_episode(Path) -> ValidationResult`。
- 校验公开 manifest、序列目录与 `seqinfo.ini`，以及 JPG 连续性、可读性、尺寸和零字节文件。
- 校验 MOT、MOTS COCO RLE、Pose 17 点记录、球的 `keypoints: null`，并执行跨模态 `(frame_id, track_id)` 集合一致性和公开 ID/class 约束。
- `task_audit` 检测到 `episode_manifest.json` 时切换为公开校验，仅将公开规范文件作为必检；无公开 manifest 时保留原有审计路径。
- 新增有效球员+球 fixture、跨模态不一致、缺 JPG、MOT/MOTS/RLE 损坏、球关键点错误和 manifest 序列不匹配测试。

## 测试

指定命令：

`uv run pytest tests/test_public_validator.py tests/test_audit_fixes.py tests/test_annotation_validator.py -q`

结果：`33 passed in 3.63s`

全量命令：

`uv run pytest -q`

结果：`595 passed, 7 deselected in 13.84s`

## 注意事项

- 公共校验器依赖现有 Pillow；不读取 EXR、render/、mask/、YOLO 或内部 JSONL，它们按要求作为可选诊断产物。
- 未执行提交后的再次测试；提交前的全量测试已通过。
