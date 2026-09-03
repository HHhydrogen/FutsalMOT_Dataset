# Task 1 实施报告

## 实施内容

- 新增 `src/grf_ue_bridge/public_episode.py`。
- 实现 COCO 压缩 RLE 的 Fortran/列优先编码、解码，以及尺寸和 run 总数校验。
- 实现公开 track ID：`L0..L4 -> 1..5`、`R0..R4 -> 6..10`、`BALL -> 100`；mask ID 继续使用既有 `BALL -> 11` 规则。
- 从 `camera.json`、`annotations.jsonl`、`pose_keypoints.jsonl` 和 Cryptomatte EXR 生成每个序列的 `gt/gt.txt`、`gt/gt_pose.json`、`gt/gt_mots.txt`、`seqinfo.ini`，并将 RGB PNG 转为 JPEG。
- 仅输出 mask 中有可见像素的对象；MOT、MOTS、Pose 使用相同的 `(frame_id, track_id)` 集合，并按稳定 ID 排序。
- Pose 输出支持 COCO 17 点 `[x,y,v] * 17`，无效投影写 `v=0`，有效点写 `v=2`，遮挡点写 `v=1`；球输出 `class: "ball"` 且 `keypoints: null`。
- 新增 `tests/test_public_episode.py`，覆盖 RLE、稳定 ID、非方形尺寸、不可见帧、无效关键点、manifest trajectory ID 和跨模态身份集合。

## 验证

- `uv run pytest tests/test_public_episode.py -q`: **6 passed**。
- `uv run pytest -q`: **567 passed, 7 deselected**。
- `git diff --check`: 通过。

## 限制

- 当前环境没有安装 `ruff`，因此无法运行 `uv run ruff check ...`。
- EXR 读取依赖现有 `OpenEXR` 运行环境；单元测试通过替换 mask loader 隔离该依赖。

## 评审修复

- 为 `ue/` 纯模块增加现有仓库约定的路径 bootstrap，使 `grf_ue_bridge.public_episode` 可作为包导入。
- 通过 `source_step`、`source_step_seconds` 和 `playback_fps` 使用现有渲染帧映射；目标 EXR 缺失时明确抛出异常，不复用其他帧。
- 屏外投影现在写入 `v=0`；映射在写文件前严格校验为十一个公开实体。
- sequence 按名称和路径确定性排序；JPEG 使用临时文件和 `os.replace`，异常时清理临时文件。
- 增加评审要求的 RLE、帧映射、屏外投影、映射拒绝、排序、bbox、JPEG 原子写入回归测试。

## 修复后验证

命令：`uv run pytest tests/test_public_episode.py -q`

输出：

```text
.............                                                            [100%]
13 passed in 0.22s
```

命令：`uv run pytest -q`

输出：

```text
........................................................................ [ 12%]
........................................................................ [ 25%]
........................................................................ [ 37%]
........................................................................ [ 50%]
........................................................................ [ 62%]
........................................................................ [ 75%]
........................................................................ [ 87%]
......................................................................   [100%]
574 passed, 7 deselected in 14.35s
```
