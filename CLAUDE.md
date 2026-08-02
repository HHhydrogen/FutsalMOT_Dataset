# CLAUDE.md

本文件为 Claude Code（claude.ai/code）在本仓库中工作时提供指引。

## 项目概览

**GRF-UE 桥接**：使用 Google Research Football (GRF) 引擎生成足球比赛轨迹，导出为 JSONL 格式，再导入 Unreal Engine 生成 Level Sequence 回放。它是 FutsalMOT 跟踪数据集的合成数据来源。

两个运行时环境之间**仅共享**磁盘上的 JSONL 格式——其余毫无关联：

| 阶段 | 运行环境 | 入口 | 输出 |
|------|----------|------|------|
| P1 导出 | Python `.venv`（需 gfootball，Python 3.9） | `uv run grf-ue export` | `meta.json` + `frames.jsonl` |
| P2 导入 | Unreal Editor Python | `import_grf_episode.py` | Level Sequence 资产 |

## 工作流程约定（必须遵守）

- **绝不自动提交 git**。每次需要提交时，先向用户提出并等待明确确认，确认后再执行提交。
- **commit message 使用简体中文**。
- **代码注释、文档字符串、pydantic Field description 一律使用简体中文**。
- 项目文档（README 等）均为简体中文。

## 常用命令

所有 P1 命令都在 `code/` 下通过 `uv` 运行（`.venv`，Python 3.9 由 `.python-version` 固定）：

```powershell
uv sync                                   # 安装依赖（含 dev 组的 pytest）
uv run grf-ue export --config configs/mvp_builtin_5v5.json --output outputs/episode_0001
uv run grf-ue validate outputs/episode_0001
uv run pytest                             # 运行全部测试
uv run pytest tests/test_validator.py -v  # 运行单个测试文件
uv run grf-ue validate-annotations outputs/dataset   # 验证 CV 标注目录
uv sync --extra overlay && uv run grf-ue annotate-overlay outputs/dataset/episode_0001/Camera_01  # debug 可视化
```

P2 脚本**在 Unreal Editor 内**（Python Console）运行，绝不在 .venv 中运行：

```python
py "D:/projects/FustalMOT_UEDataset/Content/FutsalMOT/code/ue/import_grf_episode.py"
```

脚本会自动加载仓库根目录（`ue/` 同级）的 `ue_import_config.json`，无需命令行参数；可用 `--episode`、`--mapping`、`--mode`、`--replace-existing` 覆盖。

## 架构

### P1：导出（`src/grf_ue_bridge/`）

- `cli.py` — typer 应用；提供 `grf-ue export` / `grf-ue validate` 命令。
- `grf_runner.py` — 运行 GRF 环境。强制 `number_of_left_players_agent_controls=1`，每一步都发送 `action_builtin_ai`（索引 19，`action_set='v2'`），使所有球员都表现为内置 AI，同时把完整观测记录进 `EpisodeResult`/`StepSnapshot`。
- `config.py` — 从配置文件 JSON（见 `configs/mvp_builtin_5v5.json`）解析出的 `ExportConfig` pydantic 模型。
- `coordinate_transform.py` — `CoordinateTransform`：GRF 归一化 x/y `[-1, 1]` → 米 `[-half_field, +half_field]`；球的 z 原样透传（引擎 `Z_FIELD_SCALE=1`，地面约 0.11 m）；球员固定 z=0。
- `exporter.py` — 写入 `meta.json` + `frames.jsonl`（`dump_full_raw_observation` 时额外写 `raw_observations.jsonl`）。从 `external_sources.lock.json` 读取固定的提交号写入 `meta.source`。
- `schema.py` — pydantic 模型。**实体 ID 必须正好是 `L0`–`L4`、`R0`–`R4`、`BALL`；每帧恰好 10 名球员。** 注意 `Meta.schema_` 的别名为 `schema`，序列化时 `by_alias=True`。
- `validator.py` — `grf-ue validate`：检查 JSON 解析/schema 版本、帧数与 `meta.timing.num_steps` 一致、时间单调递增、球员 ID 及唯一性、坐标边界、球员 z==0。

### 数据契约（与 P2 共享）

`outputs/episode_0001/` 包含 `meta.json`（schema 版本、时序、场地尺寸、实体列表）与 `frames.jsonl`（每帧一个 JSON 对象：`step`、`time_seconds`、`score`、`ball.position_m`（及 `source_grf_position` 参照）、`players[].{id, position_m}`）。坐标为米，`[x, y, z]`。

### P2：Unreal 导入（`ue/`）

- `import_grf_episode.py` — 纯 UE Python + 标准库，**不导入 gfootball/.venv 任何模块**。加载 episode + actor 映射后按 `mode` 执行：
  - `preview` — 直接在关卡中设置 actor 变换。
  - `sequence` — 创建/覆盖 Level Sequence 资产，为球员和球写入关键帧的 Location/Rotation 轨道；`both`（默认）两者都执行。
- `actor_mapping.example.json` — 实体 ID → UE actor 标签映射（必须与关卡中的 actor 标签一致）。
- 导入约定：米→厘米（×100），球员 Z 固定为 `PLAYER_Z_CM = 90`，球 Z `+ BALL_Z_OFFSET_CM = 2`，Yaw 由位置增量计算并带低速滞回（`SPEED_THRESHOLD_CM = 5.0`）。球的滚动旋转按帧通过四元数累加实现（在 `ue_import_config.json` 的 `ball_rolling` 段配置）。

### CV 标注导出（annotation exporter）

在 Level Sequence 之上，`import_grf_episode.py --mode annotations` 导出 CV Ground-Truth 标注（详见 README「CV Dataset Annotation Export」）。模块职责：

- `ue/camera_projection.py`（纯，pytest 可测）— 相机内参换算（FOV/焦距）、外参、world→camera→pixel 投影、3D box 投影与近平面裁剪。
- `ue/annotation_utils.py`（纯）— bbox 转换/裁剪/in_frame/truncated、track_id 映射（`L0..L4→1..5`、`R0..R4→6..10`、`BALL→100`）。
- `ue/dataset_export.py`（纯）— JSONL/MOT/seqinfo/camera.json 序列化与原子写入；`load_episode`/`load_mapping`。
- `ue/scene_apply.py`（UE 侧）— preview 与 annotation 共享的 actor 变换/查找辅助（与 Level Sequence bake 一致）。
- `ue/annotation_exporter.py`（UE 侧）— 读 CineCamera 标定与 Actor 世界 AABB，逐帧生成标注。
- `src/grf_ue_bridge/annotation_validator.py` — `grf-ue validate-annotations`。
- `grf-ue annotate-overlay` — debug 可视化（需 pillow，可选依赖）。

关键点：bbox 来自 Actor 世界 AABB 的 8 角点投影（非固定尺寸）；`visibility` 第一版恒为 `null`。帧同步：GRF step → `time=step×0.1` → Sequence 帧 `round(time×playback_fps)` → 标注 `frame_index=step+1` → 图片 `img1/000001.png`。

### Sequencer API 注意事项（已内建在脚本中——请保留）

- UE 5.8+ 会给通道名追加 `_NNN` 数字后缀；`_build_channel_map` 通过 `_canonical_channel_name` 去除后缀后匹配。
- 关键帧通过显式 `unreal.FrameNumber` 包装添加（`add_double_channel_key`）；通道来自 `section.get_all_channels()`。
- 写真实关键帧前，会先创建一个临时 `_TEMP_SMOKE` Sequence 验证 `add_key`/`remove_key` 可用。
- CameraCut 轨道**无法**通过 UE 5.8 Python API 创建——脚本只绑定摄像机 actor，并记录需要在 Sequencer 中手动设置。
- 关键帧通过每个实体的 binding（`add_possessable` → `MovieScene3DTransformTrack` → section）写入，写完后会断言关键帧数量。

## 约定与注意事项

- **环境隔离必须严格**：UE 脚本绝不能导入 P1 代码（UE Python 没有 gfootball）。JSONL 格式是两者之间唯一的接口。
- 外部仓库（`google-research-football`、`GRF_MARL`）vendor 在 `.external/` 下（git 忽略），提交号在 `external_sources.lock.json` 中固定。`gfootball`/`gym<0.26` 要求 Python 3.9——不要升级 Python 版本。
- 生成的 episode 放在 `outputs/`（git 忽略）；仅本地的配置匹配 `configs/local*.json` / `*.local.json`（git 忽略）。
- 当前只有 `main` 分支（本地与远程同步）。旧的 `grf-reboot`、`archive/legacy-*` 分支已删除，历史完整保留在 `main` 中。
