# FutsalMOT Config v3 设计规格

## 范围

本次改造只重构单 episode 的配置入口和 resolver 边界，不重构现有 P1/P2 pipeline。

不处理 batch、train/val/test split、dataset assembler、Camera Manager、动画、motion preprocessing、新 MRQ 功能或 public GT 格式。

目标是让用户配置只描述期望的数据内容；机器路径由不入库的 local machine config 提供；resolver 将 Config v3、local config 和代码默认值展开为现有 pipeline 可直接消费的完整 `resolved_task.json`。

## 配置分层

### 用户层 Config v3

用户层文件可提交 Git，并且不包含绝对机器路径：

```json
{
  "schema": "futsalmot_task",
  "version": 3,
  "episode_id": "0001",
  "simulation": {
    "scenario": "5_vs_5",
    "seed": 42,
    "steps": 300
  },
  "cameras": {
    "C01": "CineCam_01",
    "C02": "CineCam_02",
    "C03": "CineCam_03",
    "C04": "CineCam_04"
  },
  "output": {
    "fps": 30,
    "resolution": [1920, 1080],
    "annotations": ["mot", "pose", "mots"],
    "classes": ["player", "ball"]
  },
  "debug": false
}
```

允许在 `simulation` 中保留确实属于任务设计、且无法由基础字段推导的高级参数，例如 `trajectory_time_scale`、`game_duration`、AI 难度和球滚动参数；这些字段有代码默认值，不重复镜像到其他层。

用户层不允许出现以下机器路径：`dataset_root`、`ue_project_root`、绝对 `actor_mapping` 路径或其他机器绑定路径。

### Local machine config

local config 只保存机器相关路径/环境信息，不允许混入 episode、camera、fps、resolution、annotations、classes、debug 或任何任务参数：

```json
{
  "dataset_root": "G:/FutsalMOT_Dataset",
  "ue_project_root": "G:/FutsalMOT_UE"
}
```

仓库提供 `configs/local.machine.example.json`，只包含占位路径或示例路径；真实 `configs/local.machine.json` 加入 `.gitignore`，不得提交。

local config 选择优先级固定为：

1. CLI 显式 `--local-config PATH`；
2. `FUTSALMOT_LOCAL_CONFIG` 环境变量；
3. 没有配置时立即报错。

不自动搜索 task 同目录或其他目录的 local config，避免 worktree、CI 或多机器环境误读。

## Resolver 输出

resolver 读取 Config v3、local config 和代码 defaults，生成完整旧版兼容结构：

```text
Config v3 + local machine config + code defaults
    ↓
resolver
    ↓
resolved_task.json
    ↓
existing P1/P2 pipeline
```

`resolved_task.json` 保留现有 `futsalmot_resolved_task` schema 和旧版内部字段，包括：

- `task_id`、`episode_name`、`source_task_file`；
- 绝对 `repo_root`、`dataset_root`、`ue_project_root`；
- `trajectory_output`、`dataset_episode_dir`；
- 完整 `export_profile`、`ue_profile`、`postprocess`、`audit`、`artifact_policy`；
- 绝对 `actor_mapping`。

同时增加一个稳定的 `config_v3` 摘要块，记录 `episode_id`、源 FPS、resolution、camera mapping、annotations 和 classes，供 `task resolve` summary 和后续诊断使用。旧 pipeline 继续读取原有字段，不要求 P1/P2 立即理解 v3 原始结构。

## 推导规则

以下值只从 Config v3 的单一来源推导，不允许用户重复配置：

- `output.fps` → `export.target_fps`、`export.playback_fps`、UE annotation playback FPS、MRQ frame rate；
- `output.resolution` → UE annotation image width/height、render resolution、camera calibration 期望尺寸；
- `cameras` mapping → UE sequence entries、annotation camera actors、公开 sequence names、camera count；
- `simulation.steps` 与 FPS → expected frames per camera；
- `output.annotations` → canonical MOT/Pose/MOTS 开关及其内部 render/pose/Object-ID 依赖；
- `output.classes` → `include_ball` 和公开 class enablement；
- camera mapping key `C01` → 公开 sequence `FutsalMOT_<episode_id>_C01`；
- camera 数量和 expected frames → audit `expected_cameras`、`expected_frames_per_camera`。

推导后的旧字段仍可存在于 resolved task，但不再允许在 v3 用户层重复填写。

`output.annotations` 为空、包含未知 modality、`output.classes` 为空或包含未知 class 时立即失败。当前支持 modality 为 `mot`、`pose`、`mots`，class 为 `player`、`ball`。

## Local config 失败门禁

Config v3 的 `validate` 和 `resolve` 都必须先解析并校验 local config，然后校验路径：

- 缺少 `--local-config` 且没有 `FUTSALMOT_LOCAL_CONFIG`：失败；
- local config 文件不存在、不是合法 JSON、字段缺失或含任务字段：失败；
- `dataset_root` 不存在或不是目录：失败；
- `ue_project_root` 不存在、不是目录或缺少 `.uproject`：失败；
- 任一失败都不写 `resolved_task.json`，不创建半完整 runtime 文件。

`resolve` 使用临时文件原子替换；失败时清理临时文件并保留已有的上一份 resolved task 不变。`validate` 只读，不创建 runtime 或输出目录。

## v2 兼容

现有 `futsalmot_dataset_task` version 2 文件继续通过当前模型读取。加载 v2 时输出明确的 deprecated warning，说明应迁移到 Config v3 + local config；v2 仍按旧字段运行，不强制要求 local config，以避免现有 pipeline 和历史配置立即失效。

v3 不接受 v2 的重复字段组合。用户层 v3 只允许简洁字段和明确的 `simulation` 高级可选字段；旧版完整 `export`、`ue`、`postprocess`、`audit`、机器路径字段不得混入 v3。

## CLI

```powershell
uv run grf-ue task validate configs/episode_0001.json --local-config configs/local.machine.json
uv run grf-ue task resolve configs/episode_0001.json --local-config configs/local.machine.json
```

若未提供 CLI 参数，resolver 使用 `FUTSALMOT_LOCAL_CONFIG`；两者都没有时立即报错。不提供自动同目录搜索。

`task resolve` 输出简洁 summary，至少包含：

- episode；
- seed / steps；
- FPS；
- resolution；
- cameras（公开 key 和 UE actor）；
- expected frames；
- annotations；
- classes；
- public sequence names。

## 测试要求

测试覆盖：

- 合法 Config v3 加 local config 能生成完整 resolved task；
- FPS、resolution、camera mapping、annotations、classes 各只配置一次并正确传播；
- derived target/playback/MRQ FPS、尺寸、sequence、camera count、audit expectation、include_ball、modality enablement 正确；
- CLI 参数优先于环境变量，环境变量优先于无配置错误；不自动搜索同目录 local config；
- 缺 local config、路径不存在、UE project 缺 `.uproject` 时 validate/resolve 失败且不生成/覆盖 resolved task；
- local config 含任务字段、v3 含旧重复字段、未知 annotation/class 时失败；
- v2 继续读取并产生 deprecated warning；
- `task resolve` summary 包含所有要求字段；
- public output canonical 文件结构和字段不发生任何变化。
