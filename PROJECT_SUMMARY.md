
# FutsalMOT 项目总结

> **项目描述**：基于 Unreal Engine 的 4v4 五人制足球（Futsal）合成数据集生成管线，用于多目标跟踪（MOT）研究。通过程序化生成比赛回合的轨迹配置，在 UE 中渲染成 RGB 图像并导出同步标注（bounding box、动作时间轴、事件标注、球权信息、骨骼 2D 关键点）。

> **项目路径**：`Content/FutsalMOT/code/`
>
> **技术栈**：Python 3.10+ (Windows) + Unreal Engine 5 (Python API)
>
> **版本标识**：`RUN_SEED_8P_V4`（主管线）/ `A3_4_RANDOM_EPISODE_8P_V2`（生成器）

---

## 目录

1. [项目范围与数据规模](#1-项目范围与数据规模)
2. [整体架构与三步工作流](#2-整体架构与三步工作流)
3. [目录结构](#3-目录结构)
4. [管线步骤详解](#4-管线步骤详解)
   - [第 1 步：Windows 轨迹生成](#步骤-1-windows-轨迹生成)
   - [第 2 步：Unreal Engine 渲染与标注](#步骤-2-unreal-engine-渲染与标注)
   - [第 3 步：Windows 布局检查与后处理](#步骤-3-windows-布局检查与后处理)
5. [核心模块说明](#5-核心模块说明)
   - [futsalmot/core — 基础设施](#futsalmotcore-基础设施)
   - [futsalmot/pipeline — 管线常量](#futsalmotpipeline-管线常量)
   - [futsalmot/scripts — 管线各步骤实现](#futsalmotscripts-管线各步骤实现)
6. [事件系统与运动模型](#6-事件系统与运动模型)
   - [支持的模板](#61-支持的模板)
   - [事件类型](#62-事件类型)
   - [运动模型](#63-运动模型)
   - [防守方 AI：预测性加速度受限追踪](#64-防守方-ai)
7. [数据格式与配置结构](#7-数据格式与配置结构)
   - [种子/回合配置 (A3.4)](#71-种子回合配置-a34)
   - [密集轨迹配置 (A3.2)](#72-密集轨迹配置-a32)
   - [增强轨迹配置 (A3.3)](#73-增强轨迹配置-a33)
   - [渲染基础配置](#74-渲染基础配置)
   - [主管线配置](#75-主管线配置)
8. [验证体系](#8-验证体系)
   - [事件验证器 (A3.1)](#81-事件验证器-a31)
   - [轨迹验证器 (A2.5a)](#82-轨迹验证器-a25a)
9. [动作时间轴与动画系统](#9-动作时间轴与动画系统)
10. [标注输出](#10-标注输出)
11. [测试状态](#11-测试状态)
12. [技术要点与设计决策](#12-技术要点与设计决策)

---

## 1. 项目范围与数据规模

| 参数 | 值 |
|---|---|
| 比赛格式 | 4v4 无守门员（8 名外场球员） |
| A 队球员 | `Player_01` ~ `Player_04` |
| B 队球员 | `Player_05` ~ `Player_08` |
| 足球 | `Ball_01` (track_id=101) |
| 每帧对象数 | 9（8 球员 + 1 球） |
| 相机数 | 4 个固定 CineCamera |
| 回合时长 | 10 秒 |
| 帧率 | 30 FPS |
| 总帧数 | 300 帧 |
| 球场尺寸 | X: -1950~1950 cm, Y: -950~950 cm |
| 球员最大速度 | 750 cm/s |
| 球最大速度 | 3000 cm/s |
| 球员 Z 高度 | 90 cm（地面） |
| 球 Z 高度 | 11 cm |
| 最小初始间距 | 175 cm |
| 预期标注记录 | 4 相机 × 300 帧 = 1200 条 |
| 模板数 | 3 |

---

## 2. 整体架构与三步工作流

管线分为三大步骤，在 **两个环境** 之间交替执行：

```
┌──────────────────────────────────────────────────────────────────┐
│              第 1 步：Windows 轨迹生成（纯 Python）               │
│  01_generate_trajectories.py                                     │
│                                                                  │
│  A3.4 随机事件配置生成 → A3.1 事件验证 → A3.2 密集轨迹编译       │
│  → A3.3 Yaw/动作/运球增强 → A2.5a 轨迹验证 → A3.3c 事件标注     │
│                                                                  │
│  输出：pipeline_current.json 指针 + 各种 JSON 配置               │
└───────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│          第 2 步：Unreal Engine 渲染与标注（UE Python）           │
│  02_run_unreal.py (在 UE Python 控制台执行)                      │
│                                                                  │
│  UE Preflight (只读检查) → UE Build Sequences (生成关卡序列、    │
│  设置关键帧、配置动画、渲染、导出标注)                           │
│                                                                  │
│  输出：PNG 图像 + objects_bbox_2d_clean_<seq_id>.json           │
└───────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│           第 3 步：Windows 布局检查与后处理（纯 Python）          │
│  03_check_labels.py                                              │
│                                                                  │
│  绘制布局检查图（bbox + 关键点 + 球场线）→ 可选导出 YOLO/MOT 标签 │
│                                                                  │
│  输出：layout_check/<seq_id>/ 图像                               │
└──────────────────────────────────────────────────────────────────┘
```

第 1 步在普通的 Windows Python 环境中运行；第 2 步必须在 Unreal Editor 的 Python 控制台中运行，因为需要 `unreal` 模块。第 3 步回到普通 Windows Python。

---

## 3. 目录结构

```
code/
├── 01_generate_trajectories.py      # 第 1 步入口
├── 02_run_unreal.py                 # 第 2 步入口（UE 控制台）
├── 03_check_labels.py               # 第 3 步入口
├── README.md                        # 使用说明
├── PROJECT_SUMMARY.md               # 本文件
├── pyproject.toml                   # 项目元数据
├── TEST_RESULTS_8PLAYERS.json       # 验证测试报告
│
├── configs/                         # 配置目录
│   ├── pipeline_config.json         # 主管线配置（seed/template/超时等）
│   ├── pipeline_current.json        # 当前运行指针（指向上次成功运行的输出）
│   ├── action_animation_map.json    # 动作 → UE 动画资产映射
│   ├── seq_test_0005.json           # 基础渲染配置（相机、动画、bbox 参数）
│   ├── seq_test_0004.json           # 旧版基础配置（用于回退验证）
│   ├── seq_test_0003.json           # 更旧版配置
│   ├── events/
│   │   ├── episode_test_0001.json   # 测试用回合事件配置
│   │   └── generated/               # A3.4 生成的回合事件配置
│   │       └── episode_random_*.json
│   └── runs/                        # 管线运行历史（按 run_id 组织）
│       ├── run_20260719_*/          # 每次运行的时间戳目录
│       └── smoke_config_*/          # 集成测试配置
│
├── futsalmot/                       # 主 Python 包
│   ├── __init__.py
│   ├── core/
│   │   ├── paths.py                 # 统一路径解析
│   │   ├── io.py                    # JSON 原子读写
│   │   ├── hashing.py               # SHA-256 文件哈希
│   │   └── process.py               # 子进程执行与日志记录
│   ├── pipeline/
│   │   └── constants.py             # 模板名与脚本路径常量
│   ├── scripts/                     # 所有管线步骤的实现
│   │   ├── run_pipeline.py          # 主管线编排器（A3.4→A3.3c）
│   │   ├── generate_random_episode.py  # A3.4 随机回合生成器
│   │   ├── validate_episode.py      # A3.1 事件验证器
│   │   ├── compile_trajectory.py    # A3.2 密集轨迹编译器
│   │   ├── enhance_trajectory.py    # A3.3 Yaw/动作/运球增强
│   │   ├── validate_trajectory.py   # A2.5a 轨迹验证器
│   │   ├── smooth_trajectory.py     # A2.5b PCHIP 平滑（备用）
│   │   ├── generate_event_annotations.py  # A3.3c 事件标注生成器
│   │   ├── convert_and_check.py     # A1.5 后处理/布局检查
│   │   ├── ue_preflight.py          # UE 预检（只读检查）
│   │   ├── ue_build_sequences.py    # UE 构建序列/渲染/导出
│   │   ├── ue_scan_animations.py    # UE 动画资源扫描
│   │   └── ue_setup_8_players.py    # UE 场景 8 人初始化
│   └── ue/                          # UE 相关工具（预留）
│       └── __init__.py
│
└── _backup_8player_pipeline_20260717_172803/  # 旧版管线备份
```

---

## 4. 管线步骤详解

### 步骤 1：Windows 轨迹生成

**入口**：`01_generate_trajectories.py`（仅调用 `futsalmot/scripts/run_pipeline.py` 的 `main()`）

**主编排器**：`run_pipeline.py`（版本 `RUN_SEED_8P_V4`）

对每个 `seed + template` 组合，确定性重试最多 `max_attempts` 次，直到候选通过所有验证：

```
┌─────────────────────────────────────────────────────────────────────┐
│ 单个 Attempt 的管线流程：                                           │
│                                                                     │
│  A3.4  生成随机事件配置                  generate_random_episode.py │
│    ↓                                                                 │
│  A3.1  事件验证                          validate_episode.py        │
│    ↓                                                                 │
│  A3.2  密集轨迹编译                      compile_trajectory.py      │
│    ↓                                                                 │
│  A3.3  Yaw/动作/运球增强                 enhance_trajectory.py      │
│    ↓                                                                 │
│  A2.5a 密集轨迹验证                      validate_trajectory.py     │
│    ↓  (验证通过后)                                                  │
│  A3.3c 事件标注生成                      generate_event_annotations.py
└─────────────────────────────────────────────────────────────────────┘
```

**关键特性**：
- **确定性重试**：每次 attempt 使用 SHA-256 派生的 RNG seed，而非随机化哈希
- **原子写入**：所有输出文件使用 `.tmp.<pid>` → `os.replace()` 原子操作
- **SHA-256 校验**：运行报告记录所有输出文件的哈希值
- **诊断模式**：可跳过轨迹验证仅生成诊断输出，不更新 UE 指针
- **运行报告**：每轮在 `output_dir/pipeline_run_report.json` 写入完整元数据

**输出产物**：
- `event_config`：`.json` — 原始回合事件配置
- `a3_2_config`：`_a32.json` — 密集轨迹配置（含逐帧位置关键帧）
- `a3_3_config`：`_a33.json` — 增强轨迹配置（含 yaw、动作时间轴）
- `event_annotations/` — 包含 `events_*.json`、`frame_states_*.jsonl`、`event_annotation_report_*.json`
- `pipeline_current.json` — 指向本次成功运行的指针，供 UE 脚本读取

**命令行参数**：

| 参数 | 说明 |
|---|---|
| `--config` | 总配置 JSON 路径 |
| `--seed` | 覆盖随机种子 |
| `--template` | 覆盖模板 ID (1/2/3) |
| `--run-id` | 覆盖自动生成的目录名 |
| `--strict-warnings` | 将 WARNING 视为候选失败 |
| `--skip-trajectory-validation` | 跳过轨迹验证 |
| `--allow-trajectory-errors` | 允许轨迹 ERROR |
| `--max-attempts` | 最大重试次数 |
| `--no-update-current-pointer` | 不更新 pipeline_current.json |

### 步骤 2：Unreal Engine 渲染与标注

**入口**：`02_run_unreal.py`（在 UE Python 控制台执行 `py "..."`）

**子步骤**：

1. **UE Preflight** (`ue_preflight.py`, 版本 `A3_3B_UE_PREFLIGHT_READ_ONLY_8P_V5`)
   - 只读检查：验证 skeletal mesh 存在、骨骼匹配、动画资产就绪、球体材质正确
   - 输出 preflight JSON 报告（不修改任何 UE 资产）

2. **UE Build Sequences** (`ue_build_sequences.py`, 版本 `A3_3B_ACTION_TIMELINE_ANIMATION_SECTIONS_8P_V3`)
   - 读取 A3.3 配置（通过 `pipeline_current.json` 或环境变量 `FUTSALMOT_CONFIG_PATH`）
   - 创建/更新关卡 Level Sequence
   - 为每个对象逐帧设置 `set_float_parameter` / `set_transformation`
   - 根据 `action_timeline` 为球员骨骼网格体设置动画切片（`AnimSequence` section）
   - 配置球体位置关键帧
   - 触发渲染和标注导出

**渲染配置**：
- 图像格式：PNG
- 分辨率：1920 × 1080
- 文件命名：`{frame_number}`
- 输出目录：`Saved/FutsalMOT/images_clean/<seq_id>/`

**标注导出**：
- 输出：`Saved/FutsalMOT/annotations/objects_bbox_2d_clean_<seq_id>.json`
- 内容：逐帧对象的 bounding box、类别、track_id

### 步骤 3：Windows 布局检查与后处理

**入口**：`03_check_labels.py`（调用 `convert_and_check.py` 的 `main()`）

**功能**：
- 读取 UE 导出的 `objects_bbox_2d_clean_<seq_id>.json`
- 按 `--step` 间隔绘制布局检查图
- 绘制内容：所有目标的 bounding box、球员骨骼关键点（黄色圆点）、场地 41 个关键点（红色圆点+名称）、场地边界线（蓝色线条）
- 可选导出 YOLO / MOT 格式标签
- 输出目录：`Saved/FutsalMOT/layout_check/<seq_id>/`

---

## 5. 核心模块说明

### futsalmot/core — 基础设施

| 文件 | 功能 |
|---|---|
| `paths.py` | 定义所有关键路径的 Path 对象（`CODE_DIR`、`PROJECT_ROOT`、`CONFIG_DIR`、`RUNS_DIR`、`CURRENT_RUN_POINTER` 等） |
| `io.py` | `read_json()`：带 BOM 的 UTF-8 JSON 读取；`write_json_atomic()`：原子 JSON 写入（临时文件→`os.replace`）；`write_text_atomic()` |
| `hashing.py` | `sha256_file()`：流式 SHA-256 计算（1MB chunks），适用于大文件 |
| `process.py` | `run_logged_step()`：子进程执行 + 超时控制 + stdout/stderr 日志分离（按 label 写入 `logs/` 目录）+ 失败时显示尾行 |

### futsalmot/pipeline — 管线常量

| 常量 | 值 |
|---|---|
| `TEMPLATE_NAMES` | `{1: "solo_dribble_shot_4v4", 2: "dribble_pass_receive_4v4", 3: "pass_receive_dribble_shot_4v4"}` |
| `WINDOWS_PIPELINE_SCRIPTS` | 6 个 Windows 子步骤脚本的路径 |
| `UE_PIPELINE_SCRIPTS` | 4 个 UE 相关脚本的路径 |

---

## 6. 事件系统与运动模型

### 6.1 支持的模板

| 模板 | 名称 | 战术场景 | 事件数量 |
|---|---|---|---|
| 1 | `solo_dribble_shot_4v4` | 单球员带球突破射门 | ~11 事件 |
| 2 | `dribble_pass_receive_4v4` | 带球→传球→接球→继续推进 | ~12 事件 |
| 3 | `pass_receive_dribble_shot_4v4` | 传球→接球→带球→射门 | ~13 事件 |

### 6.2 事件类型

| 类型 | 时段/瞬时 | 说明 | 关键字段 |
|---|---|---|---|
| `hold` | 区间 | 球员保持位置 | `actor`, `start_t`, `end_t` |
| `move` | 区间 | 球员线性移动到目标点 | `actor`, `target_loc`, `tactical_role` |
| `dribble` | 区间 | 带球移动（球在身前） | `actor`, `target_loc`, `ball_ahead_cm` |
| `pass` | 区间 | 传球飞行（弧线轨迹） | `from`, `to`, `target_loc`, `arc_height_cm` |
| `receive` | 瞬时 | 接球事件（在 pass end_t 发生） | `actor`, `source_event` (指向 pass) |
| `shot` | 区间 | 射门（球飞向球门） | `actor`, `target_loc`, `arc_height_cm` |
| `defend_follow` | 区间 | 防守方跟随标记目标球员 | `target`, `follow_distance_cm`, `side_offset_cm`, `max_speed_cm_s`, `max_acceleration_cm_s2`, `lookahead_frames`, `avoidance_radius_cm` |

### 6.3 运动模型

**非防守方（进攻方/无球球员）**：
- 使用 **cubic smoothstep** 插值：`s(t) = t²(3-2t)`
- 坐标在 `[start_t, end_t]` 范围内从起点平滑过渡到目标点
- 关键帧间隔为 1 帧（每帧一个 keyframe），UE 使用 linear interpolation

**足球运动**：
- 持球阶段：球位于持球队员前方 `dribble_ahead_cm` (45cm) 处，随球员方向移动
- 传球阶段：球在起脚点和接球点之间做 **线性 XY + 抛物线 Z** 飞行（smoothstep 插值 + 4× 抛物线高度）
- 射门阶段：同上，但结束点是球门目标
- 运球视觉效果：在 A3.3 增强中添加横向和垂直微振荡（speed-scaled）

**时间帧对齐**：所有时间值（`start_t`/`end_t`）在生成时被舍入到最近的帧边界（`sec()` 函数），确保每帧的时间一致性。

### 6.4 防守方 AI

防守方（B 队）使用 **预测性加速度受限追踪** 模型（`goal_side_predictive_acceleration_limited_pursuit_v2`）：

```
每帧算法：
1. 解析目标球员位置 + lookahead_frames 的预测位置
2. 计算期望偏移位置：
   desired_pos = target_pos + follow_distance × forward方向 
                  + side_offset × right方向
3. 添加局部回避项（避免穿过标记球员身体）
4. 计算期望速度（含制动距离估算）
5. 通过加速度限制更新实际速度
6. 限速到 max_speed_cm_s
```

**防守参数**：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `follow_distance_cm` | 165~260 | 与目标的纵向跟随距离 |
| `side_offset_cm` | -135~135 | 横向偏移 |
| `max_speed_cm_s` | 400~540 | 最大移动速度 |
| `max_acceleration_cm_s2` | 600~950 | 最大加速度 |
| `lookahead_frames` | 3~8 | 预测帧数 |
| `response_time_sec` | 0.30~0.48 | 响应延迟 |
| `avoidance_radius_cm` | 125~155 | 回避检测半径 |
| `avoidance_weight` | 0.55~0.82 | 回避强度 |

**球员角色分配**：

| 球员 | 团队 | 角色 |
|---|---|---|
| `Player_01` | A | `ball_carrier`（持球推进者） |
| `Player_02` | A | `receiver_support`（接球支援） |
| `Player_03` | A | `wide_support`（边路拉开） |
| `Player_04` | A | `anchor`（后场锚定） |
| `Player_05` | B | `primary_press`（贴身逼抢） |
| `Player_06` | B | `receiver_mark`（盯接球人） |
| `Player_07` | B | `wide_mark`（盯边路） |
| `Player_08` | B | `cover_defender`（拖后保护） |

**依赖解析**：编译轨迹时，防守球员的路径依赖其标记目标已编译完成。编译器使用 **拓扑排序**（`compile_player_paths()`），通过维护 `pending` 集合并检查依赖子集已编译来解析执行顺序。若存在循环依赖则报错。

---

## 7. 数据格式与配置结构

### 7.1 种子/回合配置 (A3.4)

版本标识：`A3_4_RANDOM_EPISODE_8P_V2` → schema_version `1.1`

```json
{
  "schema_version": "1.1",
  "project_root": "D:/projects/...",
  "episode_id": "episode_random_0001_t1",
  "output_seq_id": "episode_random_0001_t1",
  "generator": {
    "version": "A3_4_RANDOM_EPISODE_8P_V2",
    "seed": 1,
    "template_id": 1,
    "rng_seed": 42
  },
  "roster": {
    "format": "4v4_outfield_no_goalkeepers",
    "player_count": 8,
    "teams": {"A": ["Player_01", ...], "B": ["Player_05", ...]},
    "attack_direction": {"A": "+x", "B": "-x"}
  },
  "timeline": {"fps": 30, "duration_sec": 10.0, "frame_start": 0, "frame_end": 299},
  "court": {"x_min_cm": -1950, "x_max_cm": 1950, ...},
  "players": {"Player_01": {"team": "A", "role": "ball_carrier", "track_id": 1, "class_id": 0, "start_loc": [...], "max_speed_cm_s": 750}},
  "ball": {"object_id": "Ball_01", "track_id": 101, "class_id": 1, "initial_owner": "Player_01", ...},
  "compiler_defaults": {"player_interpolation": "smoothstep_dense", "defender_follow_method": "...", ...},
  "event_rules": { /* 验证规则 */ },
  "events": [ /* 事件数组 */ ]
}
```

### 7.2 密集轨迹配置 (A3.2)

版本标识：`A3_2_EVENT_TO_DENSE_TRAJECTORY_8P_MOVEMENT_V3` → schema_version `3.0`

特点：
- 每个对象每个帧有一个关键帧（`dense_keyframe_interval_frames=1`）
- UE interpolation: `linear`（因为运动已密集采样）
- 包含 `possession_timeline`（持球时段段）
- 包含 `event_timeline`（事件到帧映射）
- 包含 `object_stats`（运动统计）
- 基础渲染配置会被深拷贝以确保相机、动画、bbox 等稳定设置被保留

### 7.3 增强轨迹配置 (A3.3)

版本标识：`A3_3_ACTION_YAW_BALL_SYNC_8P_MOVEMENT_V2` → schema_version `3.1`

在 A3.2 基础上添加：
- 每个球员关键帧增加 `yaw_deg` 字段（世界坐标系偏航角）
- 每个球员对象增加 `action_timeline` 数组（动作段序列）
- 足球对象增加 `state_timeline`（球状态时段）
- 顶层增加 `event_frame_map`（事件 ID → 帧范围映射）
- 顶层增加 `contact_frames`（接触帧列表）
- 顶层增加 `ball_state_timeline`（球状态摘要）
- 运球视觉微振荡效果写入足球位置

**Yaw 生成算法**：
1. 基于移动方向计算原始 yaw
2. 事件感知覆盖：传球/接球/射门/防守时根据目标位置设置 yaw
3. 间隙填充（前后向传播）
4. 角度解缠绕（unwrap）
5. 移动平均平滑（可配窗口大小）
6. 角速度限幅（`max_yaw_speed_deg_s`）
7. 可配置的逐球员角度偏移

### 7.4 渲染基础配置

文件：`configs/seq_test_0005.json` (schema_version `2.1`)

包含：
- 图像尺寸（1920×1080）
- 时间轴帧范围
- 4 个相机序列名和 CineCamera 映射
- 动画配置（骨骼网格体动画资产路径）
- 球员配置（bbox 参数、地面 Z、中心偏移）
- 球体参数（半径）
- class_id_map / track_id_map
- 对象列表（球的基础 keyframe 位置）

### 7.5 主管线配置

文件：`configs/pipeline_config.json`

```json
{
  "seed": 1,
  "template_id": 1,
  "max_attempts": 10,
  "timeout_sec": 300,
  "strict_warnings": false,
  "skip_trajectory_validation": false,
  "allow_trajectory_errors": false,
  "update_current_pointer": true,
  "run_id_prefix": "run"
}
```

---

## 8. 验证体系

### 8.1 事件验证器 (A3.1)

版本：`A3_1_EPISODE_VALIDATOR_8P_V2`

**检查项**：
- JSON/schema 完整性
- 时间线一致性（`duration_sec × fps` 与 `frame_end` 匹配）
- 时间到帧转换正确性
- 球员/球队/球/track_id/class_id 完整性和唯一性
- 事件类型所需的字段完整性
- 事件时间范围和帧对齐（所有时间须落在帧边界）
- 同一球员的事件区间不重叠
- 目标坐标在球场范围内
- 传球/接球配对（pass_receive_pair）
- 持球连续性（dribble/pass/shot 前必须持球）
- 球员速度估算（显式目标间的保守速度检查）

**输出**：
- `episode_report_<seq_id>.json`
- `episode_timeline_<seq_id>.csv`

**退出码**：0=无错误, 1=有 ERROR/严格模式下 WARNING, 2=致命错误

### 8.2 轨迹验证器 (A2.5a)

版本：`A2_5A_TRAJECTORY_VALIDATOR_8P_MOVEMENT_V2`

**检查项**：
- 时间线和对象配置完整性
- 关键帧顺序、覆盖、重复帧、有效坐标
- 段距离/时长/速度/速度跳变/转向角
- 球场边界违规
- 球员间最小距离（全序列采样）
- 重复 track_id 和 class/track 映射不一致
- 垂直运动异常

**可配置阈值**（在 trajectory_validation 字段中覆盖）：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `court_x_min/max` | -1950/1950 cm | 球场 X 边界 |
| `court_y_min/max` | -950/950 cm | 球场 Y 边界 |
| `player_max_speed_cm_s` | 750 cm/s | 球员最大速度（ERROR） |
| `player_warning_speed_cm_s` | 500 cm/s | 球员警告速度 |
| `ball_max_speed_cm_s` | 3000 cm/s | 球最大速度 |
| `minimum_player_distance_cm` | 50 cm | 球员最小间距（WARNING） |
| `minimum_player_distance_error_cm` | 25 cm | 球员最小间距（ERROR） |
| `max_turn_angle_deg` | 100° | 最大转向角（WARNING） |
| `max_turn_angle_error_deg` | 150° | 最大转向角（ERROR） |
| `require_full_timeline_coverage` | true | 首末关键帧须覆盖全时间线 |

**输出**：
- `trajectory_report_<seq_id>.json`
- `trajectory_segments_<seq_id>.csv`

**退出码**：0=无错误, 1=有 ERROR/严格模式 WARNING, 2=致命错误

---

## 9. 动作时间轴与动画系统

A3.3 增强为每个球员生成 `action_timeline`：

```json
[
  {"start_frame": 0, "end_frame": 45, "action": "jog", "source_events": ["event_004"]},
  {"start_frame": 46, "end_frame": 98, "action": "dribble", "source_events": ["event_001"]},
  {"start_frame": 99, "end_frame": 99, "action": "pass", "source_events": ["event_002"]},
  ...
]
```

**动作类型与优先级**：

| 动作 | 优先级 | 触发事件 |
|---|---|---|
| `idle` | 0 | 默认（移动量 < threshold） |
| `jog` | 10 | `move` 事件或默认移动 |
| `defend` | 20 | `defend_follow` 事件 |
| `dribble` | 30 | `dribble` 事件 |
| `receive` | 40 | `receive` 事件（含 prepare_frames 提前量） |
| `pass` | 50 | `pass` 事件（含 prepare_frames 提前量） |
| `shot` | 60 | `shot` 事件（含 prepare_frames 提前量） |

UE 侧通过 `action_animation_map.json` 将动作名映射到 UE 骨骼网格体的 `AnimSequence` 资源。当前只配置了 `idle` 和 `jog` 的资产路径，其余动作（dribble/pass/receive/shot/defend）为 `null`（回退到 `fallback_action: "jog"`）。

**动画切片**：在 UE 中为每个球员的骨骼网格体创建 `AnimSequence` section，与动作时间轴对齐。`strict_action_assets=false` 时，缺失的动画资产静默回退到 jog。

---

## 10. 标注输出

A3.3c 事件标注生成器输出三份文件：

### events_<seq_id>.json

```json
{
  "schema_version": "1.0",
  "seq_id": "episode_random_0001_t1_A3_3",
  "fps": 30,
  "frame_start": 0,
  "frame_end_exclusive": 300,
  "object_track_map": {"Player_01": 1, ..., "Ball_01": 101},
  "source_files": [...],
  "events": [
    {
      "event_id": "event_001",
      "type": "dribble",
      "actor_object_id": "Player_01",
      "actor_track_id": 1,
      "target_object_id": null,
      "target_track_id": null,
      "target_selector": null,
      "start_frame": 0,
      "end_frame_exclusive": 96,
      "contact_frame": null,
      "team_id": "A",
      "result": "completed"
    },
    ...
  ]
}
```

### frame_states_<seq_id>.jsonl

每帧一行 JSON，包含：
- `frame`：帧号
- `active_events`：该帧活跃的事件 ID 列表
- `player_actions`：每个球员的当前动作和 source events
- `ball_state`：`"controlled"` / `"pass"` / `"shot"`
- `possession_owner`：当前持球球员
- `contact_events`：该帧的接触事件（传球、射门、接球）

### event_annotation_report_<seq_id>.json

一致性验证报告：
- 检查 event_id 唯一性
- 检查 contact_frames 与 A3.3 源一致
- 检查球权状态一致性（"controlled" 必须有 owner）
- 检查 `frame_states` 完整覆盖和排序

---

## 11. 测试状态

**代码静态检查**：
- Python compileall：**PASS**
- Windows CLI help：**PASS**
- UE 运行时：**NOT RUN**（需要 Unreal Editor）

**已测试的回合**（`TEST_RESULTS_8PLAYERS.json`）：

| Seed | Template | Episode ID | Attempts | Trajectory Status | Warnings | Contact Frames | Min Player Distance |
|---|---|---|---|---|---|---|---|
| 1 | 1 | `episode_random_0001_t1` | 1 | WARNING | 60 | [99] | 53.3 cm |
| 1 | 2 | `episode_random_0001_t2` | 1 | WARNING | 3 | [56, 78] | 100.3 cm |
| 1 | 3 | `episode_random_0001_t3` | 1 | WARNING | 107 | [25, 46, 127] | 49.6 cm |
| 2 | 3 | `episode_random_0002_t3` | 1 | WARNING | 69 | [25, 41, 126] | 73.8 cm |
| 4 | 2 | `episode_random_0004_t2` | 1 | WARNING | 1 | [68, 90] | 172.7 cm |

所有回合在首次 attempt 即通过。轨迹 WARNING 属于当前基线的正常范围（主要是速度接近警告阈值、转向角偏大等），不影响数据可用性。

事件标注状态在所有测试中均为 **PASS**。

---

## 12. 技术要点与设计决策

1. **确定性管线**：通过 `deterministic_rng_seed()` 使用 `hashlib.sha256` 派生重试种子，确保可重复生成。所有时间值在生成时对齐帧边界（`sec()`/`sec_ceil()` 函数）。

2. **防守方 AI 复杂度**：防守追踪实现了带加速度限制、预测 lookahead、制动距离估算和局部回避避免身体重叠的运动模型。编译时需拓扑排序解析球员依赖关系。

3. **原子写入**：所有 JSON 输出使用临时文件 + `os.replace()` 原子操作，避免中途崩溃导致文件损坏。`io.py` 中的 `write_json_atomic()` 确保写入完整性。

4. **双环境设计**：纯 Python 代码（第 1、3 步）与 UE Python（第 2 步）分离，通过 `pipeline_current.json` 指针和文件系统传递状态。UE 脚本使用 `exec()` 方式在 `02_run_unreal.py` 中执行，兼容 UE Python 控制台的限制（无法 `import` 外部 `.py` 文件）。

5. **分层验证**：先验证事件配置（A3.1），再验证最终轨迹（A2.5a），确保生成的任何中间产物都是质量可控的。两种验证器都支持 `--strict-warnings` 模式。

6. **动作绑定系统**：通过 `action_timeline` 将运动轨迹的事件语义转化为可驱动 UE 动画的切片。`action_animation_map.json` 作为外部映射（可在运行时通过 `FUTSALMOT_ACTION_MAP_PATH` 环境变量覆盖），解耦运动学与动画资产管理。

7. **防守预测方向**：防守方使用目标球员的移动方向（`calculate_direction_series`）来计算期望占据位置，而非直接扑向目标当前位置，这使得防守行为更自然。

8. **运球视觉效果**：A3.3 在球上添加 speed-scaled 的横向微振荡（1.5 cm）和上下挑动（2.0 cm），但移除了前后来回振荡以避免在低加速度时产生人工 180° 转向。

9. **版本标识系统**：每个脚本顶部有明确的版本标记（如 `A3_4_RANDOM_EPISODE_8P_V2`），输出配置中记录所有依赖脚本的版本和生成时间，便于溯源。

10. **运行管理**：每次管线运行在 `configs/runs/` 下创建唯一时间戳目录，包含所有 attempt 的日志、报告和产物。`pipeline_current.json` 作为最新的成功运行指针，供第 2 步 UE 脚本读取。任何步骤失败或不满足验证条件时，指针不会被更新。

---

*生成日期：2026-07-22*
