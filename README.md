# FutsalMOT Dataset — GRF-UE Bridge

使用 Google Research Football (GRF) 引擎生成足球比赛轨迹，导出为结构化 JSONL 格式，再导入 Unreal Engine 生成 Level Sequence 回放。

---

## 项目结构

```
code/
├── src/grf_ue_bridge/          # Python CLI 包 — 在 .venv 中运行
│   ├── cli.py                  #   grf-ue export / validate 命令入口
│   ├── config.py               #   ExportConfig 模型
│   ├── grf_runner.py           #   运行 GRF 环境，采集原始观测数据
│   ├── exporter.py             #   将 EpisodeResult 导出为 meta.json + frames.jsonl
│   ├── coordinate_transform.py #   GRF 归一化坐标 → UE 米坐标
│   ├── validator.py            #   验证导出的 episode 数据完整性
│   └── schema.py               #   数据模型定义
├── ue/                         # UE Python 脚本 — 在 Unreal Editor 内运行
│   └── import_grf_episode.py   #   读取 JSONL 并生成 Level Sequence
├── configs/
│   └── mvp_builtin_5v5.json    # 示例导出配置 (5v5, 300 步, built-in AI)
├── outputs/                    # 导出数据存放目录
├── tests/                      # pytest 测试
├── external_sources.lock.json  # 外部仓库 commit 锁定
├── ue_import_config.json       # UE 导入配置（自动加载）
├── pyproject.toml              # Python 包定义
└── README.md                   # 本文件
```

---

## 管线概览

```
┌──────────────────┐     ┌──────────────────┐     ┌─────────────────────┐
│  GRF Engine      │────▶│  grf-ue export   │────▶│  UE Python Script   │
│  (built-in AI)   │     │  → JSONL         │     │  → Level Sequence   │
└──────────────────┘     └──────────────────┘     └─────────────────────┘
  在 .venv 中运行          输出 meta.json +         在 Unreal Editor
  uv run grf-ue export     frames.jsonl             py import_grf_episode.py
```

### 两步流程

| 步骤              | 运行环境      | 工具                      | 输出                                          |
| ----------------- | ------------- | ------------------------- | --------------------------------------------- |
| **P1 导出** | Python .venv  | `uv run grf-ue export`  | `episode_0001/meta.json` + `frames.jsonl` |
| **P2 导入** | Unreal Editor | `import_grf_episode.py` | Level Sequence 资产                           |

---

## P1：导出 GRF 比赛数据

在 `code/` 目录下执行（需要 .venv）：

```powershell
# 安装依赖
uv sync

# 导出默认 5v5 比赛（built-in AI，300 步，约 10 秒）
uv run grf-ue export --config configs/mvp_builtin_5v5.json --output outputs/episode_0001
```

### 配置项（config JSON）

| 字段                                       | 默认值       | 说明                                              |
| ------------------------------------------ | ------------ | ------------------------------------------------- |
| `scenario`                               | `"5_vs_5"` | GRF 场景名                                        |
| `seed`                                   | `42`       | 随机种子                                          |
| `num_steps`                              | `300`      | 运行步数（GRF 仿真 10 FPS）                       |
| `playback_fps`                           | `30`       | UE 回放帧率                                       |
| `field_length_m`                         | `40.0`     | 场地长度（米）                                    |
| `field_width_m`                          | `20.0`     | 场地宽度（米）                                    |
| `render`                                 | `false`    | 是否显示渲染窗口                                  |
| `write_video`                            | `false`    | 是否录视频                                        |
| `number_of_left_players_agent_controls`  | `0`        | 左队由 agent 控制的玩家数（0 = 全部 built-in AI） |
| `number_of_right_players_agent_controls` | `0`        | 右队由 agent 控制的玩家数                         |

### 导出产物

```
outputs/episode_0001/
├── meta.json        # 元数据：schema 版本、timing、场地、实体列表、坐标变换说明
└── frames.jsonl     # 每帧一行 JSON：step、time、score、ball、10 名玩家位置
```

### 验证导出数据

```powershell
uv run grf-ue validate outputs/episode_0001
```

验证项目：JSON 完整性、schema 版本、步数范围、玩家数量/ID、坐标有界、时间单调递增、玩家 Z 必须为 0 等。

---

## P2：导入 Unreal Engine

### 前置准备

1. **Unreal Engine 5.x 项目**，关卡中放置好 11 个 Actor：
   | 实体 ID       | UE Actor 标签               | 说明      |
   | ------------- | --------------------------- | --------- |
   | `L0`~`L4` | `Player_L0`~`Player_L4` | 左队 5 人 |
   | `R0`~`R4` | `Player_R0`~`Player_R4` | 右队 5 人 |
   | `BALL`      | `Ball_01`                 | 足球      |
2. Actor 命名与 `ue/actor_mapping.example.json` 一致即可。

### 最简单方式：一键导入

在 **Unreal Editor > Python Console** 中执行：

```python
py "D:/projects/FustalMOT_UEDataset/Content/FutsalMOT/code/ue/import_grf_episode.py"
```

脚本会自动读取同目录下的 `ue_import_config.json`，无需任何命令行参数。

### 通过命令行参数覆盖

```python
# 仅覆盖 episode 目录
py "D:/projects/FustalMOT_UEDataset/Content/FutsalMOT/code/ue/import_grf_episode.py" --episode "D:/path/to/other_episode" --replace-existing

# 仅预览模式：直接在关卡中设置 Actor 变换（不创建 Sequence 资产）
py "D:/projects/FustalMOT_UEDataset/Content/FutsalMOT/code/ue/import_grf_episode.py" --mode preview
```

### ue_import_config.json 配置说明

```json
{
  "episode": "D:/.../outputs/episode_0001",   // episode 数据目录（必须）
  "mapping": "D:/.../actor_mapping.example.json", // Actor 映射文件（必须）
  "sequence_package_path": "/Game/FutsalMOT/Sequences", // Sequence 保存路径
  "sequences": [                              // 要创建的 Sequence 列表
    { "name": "LS_Cam_01", "camera_actor": "CineCam_01" },
    { "name": "LS_Cam_02", "camera_actor": "CineCam_02" }
  ],
  "mode": "both",                             // preview / sequence / both
  "replace_existing": true                    // 是否覆盖已有 Sequence
}
```

### 模式说明

| 模式             | 行为                                                          |
| ---------------- | ------------------------------------------------------------- |
| `preview`      | 直接在关卡中逐帧设置 Actor 位置/旋转（不创建资产）            |
| `sequence`     | 创建/覆盖 Level Sequence 资产，所有数据烘焙进 Transform Track |
| `both`（默认） | 先 preview 后 sequence                                        |

### 转换规则

- **坐标**：GRF 归一化坐标 `[-1, 1]` → 米坐标 `[-half_field, +half_field]`，默认 40m×20m
- **球员 Z**：固定 90cm（角色中枢在地面以上）
- **球 Z**：GRF Z 直接传递 + 2cm 偏移
- **朝向**：基于位置增量计算 Yaw，低速度时保持先前朝向（避免抖动）

---

## 球滚动旋转

脚本会自动为足球添加滚动旋转，基于每帧位移量计算四元数旋转，贴合地面滚动效果。

在 `ue_import_config.json` 的 `ball_rolling` 段中配置：

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `enabled` | `true` | 是否启用球滚动 |
| `radius_m` | `0.11` | 球半径（米） |
| `minimum_move_distance_m` | `0.0001` | 最小移动距离，忽略微小抖动 |
| `roll_sign` | `1.0` | 滚动方向，球倒着滚时设为 `-1.0` |

若要完全禁用球滚动，将 `ball_rolling` 设为 `null` 或将 `enabled` 设为 `false`。

---

## 常见问题

| 现象                  | 解决                                                        |
| --------------------- | ----------------------------------------------------------- |
| Actor 找不到          | 检查`actor_mapping.example.json` 中的标签是否和关卡中一致 |
| 球陷进地面            | 脚本中`BALL_Z_OFFSET_CM`（默认 2cm）可调大                |
| 球倒着滚              | `ball_rolling` 中 `roll_sign` 设为 `-1.0`               |
| Level Sequence 已存在 | 加`--replace-existing` 参数覆盖                           |

## 开发

```powershell
# 运行测试
uv run pytest

# 仅测试某个模块
uv run pytest tests/test_validator.py -v
```

## 当前阶段

GRF → JSONL → Unreal Engine 最小回放验证已跑通。以下功能**暂不包含**：

- MOT 标注生成
- 批量 episode 生成
- GRF_MARL 预训练策略接入
- 事件系统
- 旧版行为克隆 / PPO
