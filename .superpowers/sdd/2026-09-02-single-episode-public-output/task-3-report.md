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

## Task 3 Review 修复

- 所有 manifest、文本标注、seqinfo 和 RLE 边界均转换为错误结果，不向调用方泄漏 OSError、UnicodeDecodeError、ValueError 或类型异常。
- 强制 root/sequence frame_count、canonical 尺寸、MOTS 尺寸和公开帧号约束；MOT 使用严格整数 token；Pose 整数字段和 visibility 拒绝布尔值及非法类型。
- 强制 `img1/` 下六位数字 `.jpg` 文件名，并报告 stale/non-JPG 文件；公开审计报告使用实际 manifest sequence/camera 名称。
- 新增不可读 UTF-8、manifest 类型/count、分数/越界帧号、非规范图片、MOTS 尺寸和公开 audit 集成测试。

修复后指定命令：

`uv run pytest tests/test_public_validator.py tests/test_audit_fixes.py tests/test_annotation_validator.py -q`

结果：`38 passed in 4.57s`

修复后全量命令：

`uv run pytest`

结果：`600 passed, 7 deselected in 15.69s`

## Task 3 Review 第二轮修复

- 在使用前校验 root/sequence `frame_count` 与 root `dimensions` 的类型和正值；无有效 sequence 时也报告尺寸错误。
- 将 seqinfo 读取、插值和字段展开统一包在异常处理内，保证损坏 manifest、seqinfo、MOT/MOTS 文本始终返回 `ValidationResult`。
- 公共 audit 现在把每个 sequence 的 JPG、MOT、MOTS、Pose 等实际统计传入报告，Markdown 显示非零 canonical 数量。
- 新增 malformed count/dimensions、seqinfo 插值异常和 public audit 实际统计测试。

修复后 focused 命令：

`uv run pytest tests/test_public_validator.py tests/test_audit_fixes.py tests/test_annotation_validator.py -q`

结果：

```text
.........................................                                [100%]
41 passed in 4.16s
```

修复后 full 命令：

`uv run pytest`

结果：

```text
============================= 603 passed, 7 deselected in 15.26s =============================
```
