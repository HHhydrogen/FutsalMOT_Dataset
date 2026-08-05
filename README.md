# FutsalMOT Dataset — GRF-UE Bridge

使用 Google Research Football (GRF) 引擎生成足球比赛轨迹，导出为结构化 JSONL 格式，再导入 Unreal Engine 生成 Level Sequence 回放。

---

## 项目结构

```
code/
├── src/grf_ue_bridge/            # Python CLI 包 — 在 .venv 中运行
│   ├── cli.py                    #   grf-ue export/validate/validate-annotations/annotate-masks/annotate-overlay
│   ├── config.py                 #   ExportConfig 模型
│   ├── grf_runner.py             #   运行 GRF 环境，采集原始观测数据
│   ├── exporter.py               #   将 EpisodeResult 导出为 meta.json + frames.jsonl
│   ├── coordinate_transform.py   #   GRF 归一化坐标 → UE 米坐标
│   ├── validator.py              #   验证导出的 episode 数据完整性
│   ├── annotation_validator.py   #   验证导出的 CV 标注目录（语义/掩码校验）
│   ├── dataset_regression.py     #   端到端 dataset 回归校验（RGB/mask/annotation 全链路一致性）
│   ├── mask_annotator.py         #   由 Instance-ID Mask 生成 mask-primary bbox/分割标注（P1）
│   ├── cryptomatte.py            #   解析 Cryptomatte EXR → mask/*.png（mask_id 值，P1）
│   └── schema.py                 #   数据模型定义
├── ue/                           # UE Python 脚本 — 在 Unreal Editor 内运行
│   ├── import_grf_episode.py     #   读取 JSONL，生成 Level Sequence / 编排标注导出
│   ├── annotation_exporter.py    #   UE 内运行：读 Camera 标定与 Actor bounds 生成标注
│   ├── render_episode.py         #   MRQ 异步渲染 RGB + Instance-ID Mask
│   ├── render_preset.py          #   CV GT deterministic preset（纯配置，pytest 可测）
│   ├── instance_mask.py          #   纯 numpy：mask 解码 / bbox / 轮廓 / 多边形（pytest 可测）
│   ├── debug_object_id_exr.py    #   诊断：检查 Cryptomatte EXR 的 manifest/通道
│   ├── camera_projection.py      #   纯数学：相机投影（pytest 可测）
│   ├── annotation_utils.py       #   纯数学：bbox 裁剪 / track_id / mask_id 映射
│   ├── dataset_export.py         #   纯 Python：JSONL / MOT 序列化与原子写入
│   ├── scene_apply.py            #   UE 侧共享的 actor 变换辅助
│   └── actor_mapping.example.json
├── configs/
│   └── mvp_builtin_5v5.json      # 示例导出配置 (5v5, 300 步, built-in AI)
├── outputs/                      # 导出数据存放目录
├── tests/                        # pytest 测试
├── external_sources.lock.json    # 外部仓库 commit 锁定
├── ue_import_config.json         # UE 导入/标注配置（自动加载）
├── pyproject.toml                # Python 包定义
└── README.md                     # 本文件
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
| `target_fps`                             | `0`        | 目标导出帧率：`0`/`10`=原生 10fps（不插值）；`30`=把 10fps 位置线性插值到 30fps 导出（渲 900 标 900，1:1）。须为 10 的倍数 |
| `playback_fps`                           | `30`       | UE 回放帧率                                       |
| `field_length_m`                         | `40.0`     | 场地长度（米）                                    |
| `field_width_m`                          | `20.0`     | 场地宽度（米）                                    |
| `render`                                 | `false`    | 是否显示渲染窗口                                  |
| `write_video`                            | `false`    | 是否录视频                                        |
| `number_of_left_players_agent_controls`  | `0`        | 左队由 agent 控制的玩家数（0 = 全部 built-in AI） |
| `number_of_right_players_agent_controls` | `0`        | 右队由 agent 控制的玩家数                         |

### 30fps 标注模式（target_fps=30）

GRF 仿真固定 **10fps**（每步 0.1s），只给球 + 10 球员的位置。默认导出为 10fps episode（每 GRF 步一条标注），渲染 30fps 全量但只标注每 3 帧（渲 900 标 300，3:1）。

设置 `target_fps=30`（示例 `configs/mvp_5v5_30fps.json`）后，导出时把 10fps 位置**线性插值到 30fps**，得到自洽的 30fps episode：

- `meta.timing.num_steps` = 300×3 = 900，`source_step_seconds` = 1/30；`frames.jsonl` 900 行。
- 每 3 帧中整数倍下标帧 = **原 GRF 真值**原样保留；中间 2 帧为线性插值近似（非 GRF 真值）。
- 帧映射自动 1:1：`frame_index=step+1`（1..900）↔ `img1/000001..000900`，渲 900 标 900。
- 朝向（yaw）渲染与标注共用同一套 `build_yaw`（位置增量 + 低速滞回），在插值帧序列上自动一致。
- 插值只作用于位置；`frame_start/frame_end` 的单位变为导出帧（30fps 帧）。

> 注意：插值帧的位置是**近似**（GRF 只在每 3 帧有真值）；适合生成平滑的多视角标注视频，若需要纯 GRF 真值请用默认 10fps 模式。

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
| `annotations`  | 导出 CV Ground-Truth 标注（见下文「CV Dataset Annotation Export」） |

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

## CV Dataset Annotation Export

在 Level Sequence / Camera 回放之上，脚本可以导出 **CV Ground-Truth 标注**：每个可见球员/球获得 2D bounding box、稳定 track_id、team/role 等，逐 Camera 独立输出。

### 支持范围

- **2D bbox（primary GT 来自 Instance-ID Mask 像素）**：渲染时同步输出每帧的实例掩码，`annotate-masks` 从每个实体的可见 mask 像素直接取 min/max 得 pixel-tight bbox（严格贴合渲染轮廓，含遮挡后可见部分）。mask 中实体像素为 0（完全遮挡/离屏）→ `bbox_source="not_visible"`、可见 bbox 为 null，几何投影 bbox（Actor/Mesh 世界空间 AABB）只保留在 `geometry_bbox_*` 字段作 fallback/debug，**不回填**为可见 GT。
- **实例分割（modal/visible）**：每个实例由可见 mask 提取外轮廓 → RDP 简化多边形，导出 YOLO Segment 标签；原始 Instance-ID Mask 始终保留为最高精度 GT。
- 稳定 `track_id`（`L0..L4 → 1..5`，`R0..R4 → 6..10`，`BALL → 100`）与稳定 `mask_id`（`L0..L4 → 1..5`，`R0..R4 → 6..10`，`BALL → 11`）
- team / role / is_goalkeeper（取自 meta.json 的 entities）
- 足球 2D bbox + 实例 mask（`mask_id=11`）
- 每个 Camera 独立标注；frame index（1 基，与图片文件名一致）
- 是否进入画面（`in_frame`，**按最终渲染可见像素**：完全被遮挡/离屏 → `false`）
- `visible_pixel_count`（可见像素数）——遮挡的真实信号，不靠 bbox 重叠率伪造
- 原始 3D/world 信息（`world_position`，米）
- Camera metadata（内参 fx/fy/cx/cy + 外参 location/rotation + FOV/焦距/传感器）
- 导出：内部 JSONL（annotations.jsonl）+ MOTChallenge + YOLO Detect + YOLO Segment

暂不支持：pose keypoints / depth / **amodal** 分割 / occlusion visibility 比例（当前只给可见模态与可见像素数，见「限制」）。

### 运行方式

在 Unreal Editor Python Console 中：

```python
py "D:/.../code/ue/import_grf_episode.py" --mode annotations   # 几何 bbox 标注（fallback）
py "D:/.../code/ue/import_grf_episode.py" --mode full           # 或一键：建 Sequence + 几何标注 + 渲染 RGB + Instance-ID Mask
```

UE 侧导出几何标注并渲染出 RGB 与 Instance-ID Mask 后，在 P1（`.venv`）把 bbox/分割升级为 mask-primary：

```powershell
uv run grf-ue annotate-masks G:/FutsalMOT_Dataset [--include-ball]
```

`annotate-masks` 读取每帧 `mask/*.png`，对每个实体由其可见 mask 像素计算 pixel-tight bbox、可见像素数、分割多边形，覆盖写 `annotations.jsonl` 并生成 MOT / YOLO（可重复运行，幂等）。mask 中实体像素为 0 → `bbox_source="not_visible"`、可见 bbox 为 null、几何只保留在 `geometry_bbox_*`（不回填）；无 `mask/` 目录或缺某帧 mask → 保持几何 bbox（legacy fallback）。

脚本读取 `ue_import_config.json` 中的 `annotation_export` 配置段，常用字段：

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `enabled` | `true` | 是否启用标注导出 |
| `output_dir` | — | 标注输出根目录 |
| `image_width/height` | `1920/1080` | 标注对应的渲染分辨率 |
| `cameras` | — | 要导出的 Camera actor 名称列表 |
| `frame_start/frame_end` | `0/null` | 0 基 GRF step 范围（开区间） |
| `include_ball` | `false` | MOT 导出是否包含球 |
| `mot_visibility_mode` | `unoccluded` | MOT visibility 策略 |
| `ball_scale` | `null` | 球 actor 缩放覆盖（`null`=不覆盖，保持关卡实际 scale） |
| `ball_radius_m` | `null` | 球半径（米）；设置后球 bbox 直接用该半径生成，不依赖 mesh bounds（`0.11`=GRF 球半径） |
| `player_bbox.source` | `mesh` | 球员 bbox 数据源：`mesh`=用 `SkeletalMesh.get_bounds()` 模型边界；`capsule`=胶囊缩放 |
| `player_bbox.width_scale` | `0.7` | 仅 `capsule` 模式：球员 bbox 横向半宽 = 胶囊 radius × scale。胶囊 r=35→70cm 宽，比真人肩宽(~50cm)宽；0.7→≈49cm 贴合身体 |
| `player_bbox.height_scale` | `1.0` | 仅 `capsule` 模式：球员 bbox 纵向半高 = 胶囊 half_height × scale |
| `instance_mask.enabled` | `true` | 是否在渲染时同步输出 Instance-ID Mask（`mask/`） |
| `instance_mask.mask_source` | `object_id_pass` | `object_id_pass`（默认）=MoviePipelineObjectIdRenderPass + multilayer EXR（Cryptomatte，UE 5.8 实测可用），渲染后由 `grf-ue cryptomatte-to-mask` 转成 mask/；`post_process_material`=DeferredPass + stencil→颜色 post-process 材质（**实测本 5.8 不可用**，渲染全 0） |
| `instance_mask.mask_channel` | `r` | mask PNG 中携带实例 ID 的通道（r/g/b/a/gray） |
| `instance_mask.id_scale` / `id_offset` | `1.0` / `0.0` | 解码参数：像素值量化 = `round((v - id_offset) / id_scale)` |
| `instance_mask.polygon_tolerance_px` | `1.0` | YOLO segmentation 多边形 RDP 简化容差（像素） |
| `instance_mask.max_polygon_points` | `64` | 每个实例多边形最大点数（超出均匀抽样） |
| `instance_mask.post_process_material` | `null` | `mask_source="post_process_material"` 时 stencil→颜色 材质的资产路径 |

> 说明：渲染出 Instance-ID Mask 后，bbox 以 mask 像素为准（primary GT），`player_bbox` 几何投影仅作 fallback/debug。几何 bbox 数据源为 `player_bbox.source`——`mesh`=用 `SkeletalMesh.get_bounds()` 模型边界（严格贴合模型参考姿势，但可能手臂张开偏宽、随朝向旋转而变）；`capsule`=胶囊缩放（胶囊 r=35→70cm 宽，比真人肩宽~50cm 宽，`width_scale` 收窄消余量）。两种几何模式切换后需重新 `--mode annotations` 并 `annotate-overlay` 目视对比。

### 输出目录

```
<output_dir>/
└── episode_0001/
    └── Camera_01/
        ├── camera.json        # 相机标定（每 camera 一次）
        ├── annotations.jsonl  # 内部标注，每帧一行（mask-primary bbox）
        ├── mask_config.json   # annotate-masks 写入的解码参数（validator 复用）
        ├── img1/              # RGB 帧（MRQ 渲染，见「自动渲染 RGB(MRQ)」）
        ├── mask/              # Instance-ID Mask 帧（与 img1/ 同帧号，像素值 == mask_id）
        ├── render/            # MRQ 原始 RGB 输出（中转）
        ├── render_mask/       # MRQ 原始 mask 输出（中转）
        ├── gt/gt.txt          # MOTChallenge 标注
        ├── seqinfo.ini        # MOT 序列信息
        ├── labels/det/        # YOLO Detect：每帧一行 class cx cy w h（归一化）
        └── labels/seg/        # YOLO Segment：每帧一行 class x1 y1 x2 y2 ...（归一化）
```

### 内部 annotation schema（annotations.jsonl）

每行一个 frame：

```json
{
  "episode_id": "episode_0001",
  "camera_id": "Camera_01",
  "frame_index": 1,
  "source_step": 0,
  "time_seconds": 0.0,
  "objects": [
    {
      "entity_id": "L0",
      "track_id": 1,
      "mask_id": 1,
      "class": "player",
      "team": "left",
      "role": "goalkeeper",
      "is_goalkeeper": true,
      "world_position": [-1.23, 0.45, 0.9],
      "in_frame": true,
      "bbox_source": "instance_mask",
      "truncated": false,
      "visibility": null,
      "visible_pixel_count": 4321,
      "bbox_xywh": [100.0, 200.0, 120.0, 430.0],
      "bbox_xyxy": [100.0, 200.0, 220.0, 630.0],
      "segmentation": [0.05, 0.18, 0.11, 0.18, ...],
      "geometry_bbox_xywh": [99.0, 202.0, 125.0, 428.0],
      "geometry_bbox_xyxy": [99.0, 202.0, 224.0, 630.0],
      "raw_bbox_xywh": [98.0, 201.0, 127.0, 432.0],
      "raw_bbox_xyxy": [98.0, 201.0, 225.0, 633.0]
    }
  ]
}
```

字段说明：

- `frame_index`（1 基）= MOT 帧号 = 图片文件名 `000001.png`；`source_step`（0 基）= frames.jsonl 行号。
- `bbox_source`：三值，严格区分「可见像素 GT」与「几何投影 GT」：
  - `"instance_mask"`（primary）：bbox 由 mask 可见像素 min/max 计算；
  - `"not_visible"`：有有效 mask 帧但该实体 mask 像素为 0（完全遮挡/离屏）——可见 GT 全为 null，几何只留在 `geometry_bbox_*`；
  - `"geometry"` / 无该字段：legacy 几何标注，仅在**无 mask 数据**（无 `mask/` 目录或缺该帧 mask）时保留。
- `bbox_xywh` / `bbox_xyxy`：`instance_mask` 时是 mask 像素的 tight bbox（连续坐标，`xmax=max_x+1`，恰好覆盖全部可见像素）；`not_visible` 时为 `null`；legacy 几何时是裁剪到图像内的几何投影 bbox。
- `geometry_bbox_xyxy/xywh`：几何投影的裁剪 bbox（仅 geometry/debug/fallback 信息，**不代表** visible/modal GT；`not_visible` 时与可见 bbox 完全解耦，绝不回填）。`raw_bbox_*`：几何投影原始值（可能越出图像边界），`not_visible` 时为 `null`。
- `visible_pixel_count`：该实体可见 mask 像素数（遮挡的真实信号）；`instance_mask` 时 ≥1，`not_visible` 时为 0。
- `segmentation`：模态分割，YOLO 归一化 flat 点列表 `[x1,y1,x2,y2,...]`。单连通域为单个多边形；**多连通域（`segmentation_components>1`）时为派生近似**——YOLO 单多边形限制下用最近点桥接合并为单个 ring（弱简单，含零宽连接），raw Instance-ID Mask 始终为 canonical GT。`segmentation_fallback` 非空表示合并失败已回退为最大连通域。
- `segmentation_components`：可见连通域碎片数（1 = 单连通域；0 = 完全不可见）。
- `segmentation_merged`：是否经最近点桥接合并为单个 ring（多连通域且未回退时为 `true`）。
- `segmentation_fallback`：`null` 或 `"largest_component"`（合并的合法性/面积膨胀检查失败时回退只保留最大连通域）。
- `segmentation_fallback_reason`：回退原因（如 `extra_ratio=0.42>0.10`）；无回退时为 `null`。
- `mask_id`：实例稳定 ID（`L0..L4→1..5`、`R0..R4→6..10`、`BALL→11`），等于 mask 像素值。
- `in_frame`：**按最终渲染可见像素**——`bbox_source="instance_mask"` 时恒 `true`；`not_visible`（完全遮挡/离屏，mask 像素为 0）→ `false`。
- `truncated`：`instance_mask` 与 `not_visible` 时恒 `false`；legacy 几何时反映几何投影的边界截断。
- `visibility`：**不建模 amodal 遮挡**，恒为 `null`（要量化遮挡请用 `visible_pixel_count`）。
- bbox 单位为像素，坐标原点为图像左上角。

### Camera metadata（camera.json）

包含 `intrinsics`（width/height/fx/fy/cx/cy）、`extrinsics`（world_location_m、world_rotation_deg、forward/right/up）、`focal_length_mm`、`sensor_size_mm`、`horizontal/vertical_fov_deg` 及坐标系/单位说明。内参来自 CineCamera 的真实焦距与传感器尺寸（filmback 缺失时回退用 `current_fov`）。

坐标系约定：Unreal 世界为左手系 X 前 Y 右 Z 上；相机空间 X=前向、Y=右向、Z=上向；图像原点左上、x 右 y 下。投影 `u=cx+fx*(y_cam/x_cam)`、`v=cy-fy*(z_cam/x_cam)`。

### MOTChallenge 导出（gt/gt.txt + seqinfo.ini）

每行 `frame,id,x,y,w,h,conf,class,visibility`：

- `frame` 从 1 开始；`x/y/w/h` 为整数像素，满足 `x,y≥0`、`w,h≥1`、`x+w≤W`、`y+h≤H`。
- `conf` = 1。
- `class`：球员 = 1（MOT16/17 pedestrian），球 = 100（自定义，仅 `include_ball=true` 时出现）。
- `visibility`：由 `mot_visibility_mode` 控制——`unoccluded`（默认，写 1.0，即"未建模遮挡"的假设）或 `truncation`（裁剪面积/原始面积，仅反映边界截断，**不是**真实遮挡）。
- 不在画面中的对象不会写入 gt.txt。

### track ID 定义

确定性映射，一个 episode 内不变：`L0..L4 → 1..5`、`R0..R4 → 6..10`、`BALL → 100`。

### BALL 处理

内部标注始终包含球（class=`ball`，track_id=100）。标准 player MOT 导出默认不含球（`include_ball=false`），需要时可开启。

注意：球的 bbox 默认来自球 mesh 的真实世界 bounds。若球 mesh 资产的包围盒数据异常（例如渲染正常但 bounds 偏大），可设置 `annotation_export.ball_radius_m`（如 `0.11`=GRF 球半径），标注将直接用该半径生成球 bbox，不再依赖 mesh bounds。

### 帧同步规则

GRF step（0 基）→ `time_seconds = step × source_step_seconds`（0.1s）→ Sequence 帧 = `round(time_seconds × playback_fps)`（30 FPS）→ 标注 `frame_index = step + 1` → 图片 `img1/000001.png`。

**标注的 frame N 与 `img1/000N.png` 表示同一时刻**（同一 Level Sequence time 下的 actor/camera 变换）。`img1/` 由下一节「自动渲染 RGB(MRQ)」自动填充；也可手动把对应分辨率的 RGB 放入。

### 自动渲染 RGB(MRQ)

一键全流程（建 Sequence + 导标注 + 渲染 RGB）：

```python
py "D:/.../code/ue/import_grf_episode.py" --mode full
```

也可以只渲染已有 Sequence：

```python
py "D:/.../code/ue/import_grf_episode.py" --mode render
```

**异步执行**：渲染是**异步**的——脚本把所有 Sequence 加入同一个 MRQ queue 并提交后**立即返回**，不阻塞编辑器主线程（MRQ 的 PIE 渲染窗口需要编辑器主线程持续 tick 才能推进，任何 `time.sleep`/`Event.wait` 同步阻塞都会让渲染卡死）。渲染完成后自动把 RGB 帧复制到各 camera 的 `img1/`、Instance-ID Mask 复制到 `mask/`，并写完成标记 `<output_dir>/<episode_id>/render_summary.json`（记录 `status`：`success` / `partial` / `failed`、各 camera 的 `img1_frames`/`mask_frames` 与 annotation 帧数一致性）。渲染期间请保持编辑器运行；完成后查看控制台汇总与 `render_summary.json`。

**完成检测**：正常路径由 MRQ 的 `on_executor_finished_delegate` 回调驱动收尾（回调须用与委托一致的显式签名——UE 5.8 会拒绝 `*args` 变参并报 "incorrect number of arguments"）。脚本同时注册了一个 **slate post-tick watchdog** 兜底——每编辑帧非阻塞检查（不再渲染 且 目标帧齐全 / 文件数长期稳定 / 30 分钟硬超时），命中即完成同样的收尾。两条路径幂等，只收尾一次。

渲染使用 UE 的 **Movie Render Queue（MRQ）**，通过 `annotation_export.render_rgb` 配置：

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `enabled` | `false` | 是否渲染 RGB |
| `preset` | `cv_gt` | CV 渲染预设：`cv_gt`（默认，deterministic GT）/ `cinematic`（保留模式，未实现）/ `null`（不覆盖，保持关卡/相机现状） |
| `cv_gt` | 见下 | cv_gt 预设的详细开关（motion_blur / depth_of_field / chromatic_aberration / lens_distortion / anti_aliasing / temporal_sampling） |
| `output_resolution_x/y` | `1920/1080` | 渲染分辨率（默认复用 image_width/height） |
| `frame_rate` | `30` | 渲染帧率（须与 Sequence display rate 一致，默认 playback_fps） |
| `file_name_format` | `{frame_number}` | MRQ 输出文件名格式 |
| `zero_pad_frame_numbers` | `6` | MRQ 帧号补零位数 |

### CV GT deterministic preset

**正式 CV 数据集默认使用 `preset="cv_gt"`**（deterministic）：显式关闭会破坏「RGB 空间边界 == Instance-ID Mask 边界」的空间扩散效果，并把时间采样固定为单时刻，使 pixel-tight bbox / segmentation 与 RGB 严格对齐。

| cv_gt 字段 | 默认 | 说明 |
|-----------|------|------|
| `motion_blur` | `false` | 关闭运动模糊（后处理 `motion_blur_amount=0` + MRQ shutter 时间积分关闭）。运动模糊会让高速移动球员的 RGB 边缘沿运动方向拖尾，导致 RGB 目标超出/偏离同帧 mask 边界 |
| `depth_of_field` | `false` | 关闭景深（DOF 方法置 off + 大 f-stop 兜底）。景深模糊会让失焦目标的 RGB 边缘外扩，与 mask 硬边界不一致 |
| `chromatic_aberration` | `false` | 关闭色差（`chromatic_aberration_intensity=0`）。色差把 RGB 边缘拆成 R/G/B 色散，破坏与 mask 的像素级对应 |
| `lens_distortion` | `false` | 关闭镜头畸变。UE 无内置后处理属性，畸变仅来自失真后处理材质；管线不添加任何失真材质即默认关闭（文档保证） |
| `anti_aliasing` | `taa` | 保留合理 anti-aliasing（`taa` / `tsr` / `none`）。AA 只做亚像素边缘混合，不改变目标空间边界；`none` 时边缘完全硬边 |
| `temporal_sampling` | `false` | 单时刻渲染。`false` = 每输出帧单采样（`temporal_accumulation_method=NONE`、`temporal_sample_count=1`、`render_warm_up_count=0`），无 shutter 时间积分 |

**为什么关闭这些效果**：Instance-ID Mask（Object ID pass）表示的是**单一时刻**的几何覆盖像素；RGB 若启用运动模糊 / DOF / 时间累积，则表示**一段曝光时间**或空间上被模糊扩散后的结果——二者空间边界不再一致，pixel-tight bbox / segmentation 就会与 RGB 实际可见区域错位。cv_gt 模式强制 RGB 与 mask 都退化为同一时刻、无空间扩散的渲染，从而保证标注语义成立。保留的 lighting / shadow / material / texture 与合理 AA 不影响边界一致性。

**强制应用的位置**（不依赖关卡手工设置）：渲染前脚本把后处理覆盖写到每个 **CineCameraComponent 的 `post_process_settings`**（`post_process_blend_weight=1.0` 完全覆盖关卡 Post Process Volume）并保存关卡；MRQ RGB job 显式添加 **`MoviePipelineAntiAliasingSetting`**（`temporal_accumulation_method=NONE`、`temporal_sample_count=1`、`render_warm_up_count=0`，把每输出帧固定为单采样，消除时间域运动模糊）与 **`MoviePipelineConsoleVariableSetting`**（cvar `r.MotionBlur.Amount 0` / `r.DepthOfFieldQuality 0` / `r.SceneColorFringeQuality 0`，强制关闭后处理运动模糊 / 景深 / 色差，即使相机覆盖位在特定版本不生效也兜底）。RGB 与 mask 两个 job 都施加时间确定性。这些设置在每次渲染时由脚本显式应用，避免不同机器/关卡产生不同 GT 语义。

**切换方式**：把 `ue_import_config.json` 里 `annotation_export.render_rgb.preset` 改为 `null`（保持关卡现状）或 `"cinematic"`（未来模式，放开画质）。`cv_gt` 内部各开关也可按需覆盖（如 `anti_aliasing: "none"` 关 AA）。切换后重新 `--mode full` 或 `--mode render` 即生效。

**Camera Cut**：脚本在创建 Sequence 时会尝试自动设置 Camera Cut（`import_grf_episode.py` 的 `_add_camera_cut`）。若自动设置失败（不同 UE 版本 API 差异），需在 Sequencer 手动设置一次（右键摄像机轨道 → "Set as Camera Cut"）。没有 Camera Cut 时 MRQ 会用默认视角渲染，与标注投影不一致。

**对齐规则**：MRQ 以 `frame_rate`（30fps）渲染 Sequence 全范围，脚本按 `round((frame_index-1)*source_step*fps)` 选出与每个标注帧对应的渲染帧，复制为 `img1/{frame_index:06d}.png`。标注、MOT、RGB 三者同源同帧。

**渲染时长**：当前以 30fps 渲染全范围后每 3 帧取 1 帧，约 3 倍耗时；`instance_mask.enabled=true` 时 mask 为独立 MRQ job，总耗时再 ×2（见「Instance-ID Mask 像素级 GT」）。后续可优化为渲染 10fps 变体 Sequence / 同 job 多 pass。

**恢复（不重新渲染）**：若渲染已把 PNG 输出到各 camera 的 `render/`，但完成回调未触发导致 `img1/` 为空（极端情况），无需重新渲染，直接从现有 `render/` 恢复 `img1/` 并写 `render_summary.json`：

```powershell
uv run python ue/recover_render.py
```

（也可在 UE 控制台 `py "D:/.../code/ue/recover_render.py"`。该脚本纯 Python，读取 `ue_import_config.json`。）

### Instance-ID Mask 像素级 GT

> 状态：**RESOLVED**。完整排查过程与最终方案见仓库根 `MASK_RENDERING_STATUS.md`（CustomStencil 黑屏问题已作为历史记录归档，不再阻塞）。

`annotation_export.instance_mask.enabled=true` 时，MRQ 在渲染 RGB 的同时为每个 Sequence 额外渲染一份 **Instance-ID Mask**（独立 MRQ job，输出到 `render_mask/` 后复制为 `mask/`，与 `img1/` 同帧号同分辨率），每个实体的 mask 像素值 == 其稳定 `mask_id`（背景 0）：

| 实体 | mask_id | | 实体 | mask_id |
|------|---------|-|------|---------|
| `L0`..`L4` | `1`..`5` | | `R0`..`R4` | `6`..`10` |
| `BALL` | `11` | | 背景 | `0` |

**渲染机制**：mask job 用 **`MoviePipelineObjectIdRenderPass`（`id_type=ACTOR`）+ multilayer EXR**，输出 **Cryptomatte**（UE 5.8 实测可用）：manifest 在 EXR header（`{actor_label: "hex_id"}`），实体 ID 是 float32（存于 `RGBA` 层 R 通道，`hex_id` = 位模式大端十六进制）。渲染后 P1 用 `grf-ue cryptomatte-to-mask` 把 `render_mask/*.exr` 转成 `mask/*.png`（每实体像素 = `mask_id` 1~11，背景 0），再经 `annotate-masks` 从 mask 像素计算 bbox/分割。因此 bbox 严格贴合实际渲染轮廓（含遮挡后可见部分），不再依赖几何估算。

> 之前尝试的 `mask_source="post_process_material"`（stencil→颜色 材质）在本 5.8 **不可用**：PIE 里 stencil 值/开关/材质均正确，但渲染出的 mask 全 0；`MoviePipelinePostProcessPass` 是结构体非独立 pass。故采用 Object ID + Cryptomatte。

**mask job 是独立 MRQ job → 总渲染耗时约 2 倍**（RGB 一遍 + mask 一遍）。RGB job 与 mask job 独立构建：mask job 构建失败不影响 RGB 渲染（错误记入 `render_summary.json`）。

**校准 / 诊断**：若 mask 输出不正确，用 `ue/debug_object_id_exr.py` 检查 Cryptomatte EXR：

```powershell
uv run python ue/debug_object_id_exr.py G:/FutsalMOT_Dataset/episode_demo/CineCam_01/render_mask
```

它会打印 EXR 里的 manifest（实体名→ID）与各通道取值，确认 Cryptomatte 是否含全部实体。

**生成 mask-primary 标注（P1，渲染完成后）**：

```powershell
# 1. Cryptomatte EXR → mask/*.png（每实体像素 = mask_id 1~11）
uv run grf-ue cryptomatte-to-mask G:/FutsalMOT_Dataset/episode_demo [--workers 4] [--png-compress-level 1]

# 2. mask → mask-primary bbox / 分割 / MOT / YOLO
uv run grf-ue annotate-masks G:/FutsalMOT_Dataset/episode_demo --include-ball [--polygon-tolerance-px 1.0] [--max-polygon-points 64] [--workers 4]
```

`cryptomatte-to-mask`：读 `render_mask/*.exr` 的 manifest + `RGBA` 层 R 通道，对每实体按 float32 位模式（uint32 bit-exact）匹配生成 mask（缺省从 `ue_import_config.json` 读 mapping/episode，可用 `--mapping`/`--episode` 覆盖）。支持 `--workers`/`--chunk-size` 并行与 `--png-compress-level 0–9`。
`annotate-masks`：读 `mask/{frame}.png` + UE 导出的 `annotations.jsonl` → 每实体由可见 mask 像素算 tight bbox / `visible_pixel_count` / 模态分割多边形 → 覆盖写 `annotations.jsonl`（`bbox_source="instance_mask"`，几何 bbox 保留在 `geometry_bbox_*`；mask 像素为 0 → `bbox_source="not_visible"`、可见 bbox 置 null）→ 重写 MOT `gt/gt.txt` → 写 YOLO `labels/det/` 与 `labels/seg/` → 写 `mask_config.json`（解码参数）。幂等，可重复运行；原始 `mask/*.png` 永不修改。无 `mask/` 的 camera 或缺某帧 mask 保持几何 bbox（legacy fallback）。支持 `--workers`/`--chunk-size` 并行、`--formats`（all/mot/yolo-det/yolo-seg/json 逗号组合）与 `--no-segmentation`（跳过分割多边形，见下节 Performance）。

**MOT / YOLO 导出**：
- MOT `gt/gt.txt`：bbox 为 mask 可见 bbox，`visibility` 按 `mot_visibility_mode`（默认 `unoccluded` 写 1.0）；完全不可见实体（`bbox_source="not_visible"`）不写入。
- YOLO Detect `labels/det/`：每行 `class cx cy w h`（归一化，`0=player`、`1=ball`）。
- YOLO Segment `labels/seg/`：每行 `class x1 y1 x2 y2 ...`（归一化多边形，RDP 简化后 ≤ `max_polygon_points` 点，多连通域用最近点桥接合并为单行；raw mask 为最高精度 GT）。
- 球默认不导出到 MOT/YOLO，`--include-ball` 开启（`annotate-masks` 参数）。

### 调试可视化

```powershell
uv run grf-ue annotate-overlay G:/FutsalMOT_Dataset/episode_0001/Camera_01 --include-ball
```

把 bbox + entity_id + track_id 画到 `img1/` 的 RGB 帧上，输出到 `debug/000001_bbox.png`。没有 RGB 时跳过。需要 pillow（已是核心依赖）。

### 多视角标注视频（make-video）

把 `img1/` 帧编码成 **mp4 标注视频**（默认叠加 bbox；多视角 = 对每个 camera 目录各跑一次）：

```powershell
uv run grf-ue make-video G:/FutsalMOT_Dataset/episode_0001/CineCam_01 --include-ball
# → CineCam_01/video_30fps.mp4（默认读 seqinfo.ini frameRate 或 30fps）
```

参数：`--fps`（默认 30）、`--out`（输出路径）、`--plain`（不画 bbox，编码原图）、`--max-frames N`（smoke 只取前 N 帧）。帧序取 `annotations.jsonl` 的 `frame_index`。需要 `opencv-python`（已是核心依赖）。配合 30fps 标注（`target_fps=30`）可生成平滑的多视角标注视频。

### 验证

```powershell
uv run grf-ue validate-annotations G:/FutsalMOT_Dataset
```

这是**最终验收命令**，一条命令同时运行两类校验：

1. **逐 camera 语义/掩码校验**（`annotation_validator`）：bbox 合法性、frame_index 连续性、entity↔track 双向一致、MOT 行合法、resolution 一致、可见像素 GT 与几何 GT 语义分离（`bbox_source="instance_mask"` → `visible_pixel_count>0`；`visible_pixel_count==0` → 可见 bbox 为 null；`bbox_source="not_visible"` → mask 无像素；**不可见对象不进入 MOT/YOLO**，做交叉校验）等。若 camera 目录存在 `mask/`，额外检查：RGB(`img1/`) 与 mask 帧一一对应、mask 分辨率一致、mask 像素值合法（0 或 1..11）、`mask_id` 与实体确定性映射一致、`bbox_xyxy` 与 mask min/max 一致、YOLO 坐标 ∈ [0,1] 且 det/seg 行格式合法。
2. **端到端 dataset regression**（`dataset_regression`）：从 mask + annotations **重新派生** bbox / MOT / YOLO Det / YOLO Seg，与落盘产物逐项比对，验证整条 Instance-ID 标注链路的内部一致性（见下节）。

### Dataset Integrity Validation

`grf-ue validate-annotations` 末尾会自动运行统一的 **dataset regression validator**（`src/grf_ue_bridge/dataset_regression.py`），对每个 camera 检查：

- **帧数全链路对应**：`annotations.jsonl` 帧数 == `img1/` PNG 帧数 == `mask/` PNG 帧数；`frame_index` 与文件名 `{frame_index:06d}.png` 严格一一对应。
- **分辨率一致**：RGB / mask 每帧分辨率 == `camera.json` 的 `image_width × image_height`。
- **mask 不得全背景**：任何 mask 帧不得全为 0（疑似渲染失败/实体全部丢失）。
- **entity/class/track/mask 规则**：`BALL → class=ball, track_id=100, mask_id=11`；`L0..L4 → track/mask 1..5`；`R0..R4 → track/mask 6..10`，全帧稳定。
- **instance-mask bbox == mask 像素**：每个 `bbox_source="instance_mask"` 对象的 `bbox_xyxy` 必须等于 mask 可见像素 min/max（±0.5px），`visible_pixel_count` 必须等于 mask 非零像素数。
- **MOT / YOLO 重新派生比对**：用与生成完全相同的公式（`build_mot_gt` / `det_xyxy_to_yolo_norm` / segmentation flat）从 annotations 重新生成期望的 MOT gt.txt 与 YOLO det/seg 标签，逐行精确比对落盘文件——`bbox_source="not_visible"` 的不可见对象不会出现。
- **多连通域 merge 的 raster quality gate 复验**：对 `segmentation_merged=true`（多连通域成功桥接合并）的对象，用与 `annotate-masks` 相同的面积膨胀检查（extra/missing/iou 阈值）复验存储的 segmentation ring 仍通过。
- **BALL 规则**：球以 `class=ball / track_id=100 / mask_id=11` 进入 MOT/YOLO（仅 `--include-ball` 时）。

> 说明：对缺 `mask/` 的 legacy 纯几何目录，mask 相关检查自动放行；但若 `mask/` 存在（mask-primary 数据集），则要求 RGB / mask / annotation 全链路完整——任一帧缺 mask 或 RGB 都会判定不完整。

最小回归锚点：`tests/test_golden_fixture.py` 用确定性合成 64×64 fixture（2 帧、L0/L1/BALL 可见、R0 不可见），**手工核算 + 锁定** mask → bbox → MOT → YOLO Detect → YOLO Seg 的精确输出值；任何一环改动都会让 golden 断言失败。`tests/test_dataset_regression.py` 单独覆盖 regression validator 的负例（全背景 mask、分辨率不符、帧数缺失、MOT/YOLO 与 mask bbox 不一致）。

---

## Performance（后处理性能）

三个后处理命令（`cryptomatte-to-mask` / `annotate-masks` / `validate-annotations`）已针对 Windows 多核优化：

- **多进程并行**：`--workers N`（`0`=自动，`1`=串行，`>1`=多进程）。**多相机优先相机级并行**；**单相机（或相机数少于 worker 数）自动按连续帧区间分块并行**（绝不退回串行），`--chunk-size` 可显式给定块大小。worker 只做计算，主进程统一按 `frame_index` 升序合并、原子写盘——**相同配置下文本产物与 mask 逐字节确定**，与 worker 数无关。
- **NumPy 向量化单次扫描**：每帧只读/量化一次 mask，`np.bincount` + 稀疏坐标 min/max 一次拿到所有实例的 pixel_count / bbox / ROI。
- **OpenCV 原生连通域/轮廓**：`cv2.connectedComponentsWithStats` + `findContours`（反转对齐 Moore 点序，输出与旧实现逐点一致）+ 只在实例 ROI 内处理。无 cv2 时自动回退纯 Python（更慢）。
- **快速 Mask 读取**：单通道 `L` PNG 直接读二维 uint8（不做 RGBA 转换）；整数实例 ID 量化零复制透传。
- **Cryptomatte 位模式匹配**：float32 Actor ID 按位解释为 uint32 后精确整数比较（bit-exact，绝无浮点 ID 混淆）。
- **PNG 压缩等级**：`--png-compress-level 0–9`（默认 1，性能推荐；解码像素值不变）。

### 推荐的 worker 数

| 场景 | 建议 |
|------|------|
| 默认（自动） | 不传 `--workers`：按 `min(相机数, cpu//2)` 自动选择 |
| 4 相机、8+ 核 | `--workers 4`（相机级并行）或 `--workers 8`（含帧分块） |
| **单相机大 episode** | **`--workers <cpu//2>`（帧级并行）+ `--chunk-size 50` 可选**；不再退回串行 |
| 低内存设备 / 串行调试 | `--workers 1`（峰值内存最低） |

### 磁盘建议

- **NVMe**：`--workers` 可用满核（I/O 不成为瓶颈）。
- **SATA SSD**：`--workers <cpu//2`（避免过多 worker 抢 I/O 带宽），并行仍有效。
- **机械硬盘**：串行或 `--workers 2`；`cryptomatte-to-mask` 的 EXR 解码是磁盘+解压为主，并行收益小（建议保持默认）。
- **低内存设备**：`--workers 1`；并行时峰值内存约为「每 worker 一帧 mask」的倍数，不会无界增长（worker 逐帧处理，不整 episode 载入）。

### 完整标注 vs 快速 MOT 模式

`annotate-masks` 默认导出全量（bbox + 分割 + MOT + YOLO）。若只需要检测/跟踪，用 `--no-segmentation` 跳过轮廓/polygon/桥接/质量检查（`segmentation=null`，不生成 `labels/seg/`），再配合 `--formats` 只写需要的派生产物：

```powershell
# 完整（默认）：bbox + 分割 + MOT + YOLO det/seg
uv run grf-ue annotate-masks <episode_dir> --workers 4 --formats all

# 仅 MOT + 检测，跳过分割（最快；bbox/像素数/MOT/YOLO det 正常生成）
uv run grf-ue annotate-masks <episode_dir> `
  --workers 4 `
  --formats json,mot,yolo-det `
  --no-segmentation `
  --clean-stale
```

`--formats` 可选：`all` / `mot` / `yolo-det` / `yolo-seg` / `json`，逗号组合。`annotations.jsonl` 始终是核心输出。

### 陈旧派生产物清理（--clean-stale）

`annotate-masks` 是派生产物生成命令，运行后目录应准确反映本次选择的 `--formats`。默认 `--clean-stale`：

- 未选 `mot` → 删除 `gt/gt.txt`（`seqinfo.ini` 由 UE 侧生成且 validator 依赖，**不删**）；
- 未选 `yolo-det` → 删除 `labels/det/`；
- 未选 `yolo-seg` → 删除 `labels/seg/`；
- `annotations.jsonl`、`camera.json`、`mask/`、`mask_config.json` 永不删除。

只删除已知派生产物，不跟随符号链接，不递归删除未知用户文件。`--no-clean-stale` 保留旧文件（可能残留与当前命令声明不一致的目录）。该流程幂等：`formats=all` → `json,mot`（清理）→ `formats=all` 可恢复全部派生产物。

### PNG 压缩等级

`cryptomatte-to-mask --png-compress-level 1`（默认）在像素不变的前提下显著加快写入；`0` 无压缩最快但文件更大；`9` 最小但最慢。不同等级解码后的 mask 逐像素一致（有测试保证）。

### 兼容性声明（重要）

> 默认输出在**数据语义和解码像素层面**向后兼容（mask 解码像素、bbox、visible_pixel_count、in_frame、MOT、YOLO、JSON 语义一致）。当前优化版在**相同配置**下文本产物与 mask 保持确定性。**PNG / EXR 等压缩二进制文件不保证与旧版本逐字节相同**——压缩等级、Pillow / zlib / OpenEXR 版本都可能改变文件字节，尽管解码后的像素一致。

### full vs quick validation

```powershell
# 完整（默认）：逐帧重算 mask bbox/像素数，重新派生并比较 MOT/YOLO，检查全部帧
uv run grf-ue validate-annotations <episode_dir> --workers 4 --validation-level full

# quick：结构检查（文件/帧数/文件名对应/mask ID 合法/bbox 范围/track 映射/MOT·YOLO 语法）
# + 每相机抽样有限帧重算（快一个数量级，适合 CI / 快速门禁）
uv run grf-ue validate-annotations <episode_dir> --workers 4 --validation-level quick
```

`full` 是默认，保持完整语义；`quick` 牺牲逐帧全量重派生换取速度，二者对同一份正确数据都通过。

### 端到端基准（`scripts/benchmark_postprocess.py`）

```powershell
uv run python scripts/benchmark_postprocess.py `
  --input G:/FutsalMOT_Dataset/episode_demo `
  --repeat 3 `
  --workers 4 `
  --stage-img1

# 大数据集：cryptomatte/annotate 在 staging 子集测量，validate 在真实输入上只读运行
uv run python scripts/benchmark_postprocess.py `
  --input G:/FutsalMOT_Dataset/episode_0001 `
  --repeat 1 `
  --workers 8 `
  --validate-on-input
```

按命令报告 `cryptomatte-to-mask` / `annotate-masks` / `validate-annotations` 总耗时、
各阶段处理帧数与 FPS、camera-frame 总量（expected_total = Σ 每相机帧数）、
完整进程树峰值 RSS（root + 子进程，psutil 轮询）、以及结果元数据
（git commit / timestamp / python / platform / cpu / workers 等）。
每个阶段都校验执行状态与帧数：任一阶段不完整（cryptomatte 非 success、
退出码非 0、帧数不匹配）则 benchmark 以非零码失败，绝不输出伪成功耗时。

**进程树 RSS 限制**：进程 RSS 存在共享内存页重复计数问题，进程树 RSS 是各进程
RSS 的求和，可能重复计算共享页，主要用于比较相同环境下不同 worker 数的相对趋势。

样例数据（episode_demo，10 帧/相机，4 相机，单机 20 核）实测：

| 命令 | baseline（优化前） | 优化后串行 | 优化后并行（4 worker） |
|------|-------------------|-----------|------------------------|
| cryptomatte-to-mask | 1.03 s/相机 | 0.99 s/相机 | 0.64 s/相机 |
| annotate-masks | **29.48 s/相机** | **0.36 s/相机（82×）** | 0.22 s/相机（134×） |
| validate-annotations | 1.55 s/相机 | 1.09 s/相机 | 0.38 s/相机 |

> `annotate-masks` 的瓶颈从优化前的纯 Python 全图泛洪（每帧 ~3 s）变为 ROI 内 OpenCV 轮廓（每帧 ~16 ms，175×）。`cryptomatte-to-mask` 的 EXR 解码（PIZ 压缩，整帧解压）是固有 I/O+解压成本，主要靠并行摊薄。

### GPU 后端

未引入。CPU 优化后 `annotate-masks` 已从每相机 ~30 s 降到 <0.4 s，I/O（EXR 解码、PNG 读写、JSON 写入）成为主要剩余瓶颈——这些不适合 GPU。详见交付报告。

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

# 验证 CV 标注输出目录（含 Instance-ID Mask 校验）
uv run grf-ue validate-annotations G:/FutsalMOT_Dataset

# 由 Instance-ID Mask 生成 mask-primary bbox / 分割标注（渲染完成后）
uv run grf-ue annotate-masks G:/FutsalMOT_Dataset --include-ball

# 调试可视化（pillow 为核心依赖，无需额外安装）
uv run grf-ue annotate-overlay G:/FutsalMOT_Dataset/episode_0001/Camera_01 --include-ball
```

## 第二阶段浸泡测试（300 步 × 4 相机）

完整记录见 [`docs/SOAK_TEST_300STEP_4CAM.md`](docs/SOAK_TEST_300STEP_4CAM.md)：真实 UE 渲染（900 帧/相机 RGB + Object ID EXR）+ 1200 camera-frame 后处理 + full validation + 审计 + w1/2/4/8 基准 + 故障恢复。

新增辅助脚本（`scripts/`）：

```powershell
# soak 完整性审计：缺帧/重复/零字节/跨相机同步/track·mask 映射/标定/render_summary/validator
uv run python scripts/audit_soak_episode.py --input G:/FutsalMOT_Dataset/episode_0001 --expected-cameras 4 --expected-frames-per-camera 300 --episode outputs/episode_0001 --validation-level quick

# UE 渲染期间资源/目录增长采样（每 30s 追加 CSV）
uv run python scripts/monitor_soak_resources.py --input G:/FutsalMOT_Dataset/episode_0001 --interval 30

# 运行任意命令并报告墙钟时间 + 进程树峰值 RSS
uv run python scripts/measure_run.py uv run grf-ue cryptomatte-to-mask <dir> --workers 4 ...
```

> `render_summary.json` 对 Object ID EXR 的 mask 状态已修正：EXR 源统计对齐帧数并记 `mask_source="object_id_exr"`（mask/*.png 由 P1 生成），不再恒报 partial；`total_img1_frames`/`total_mask_frames` 分列统计。

## 当前阶段

GRF → JSONL → Unreal Engine 回放 → RGB + Instance-ID Mask → mask-primary bbox / 实例分割 / MOT / YOLO 标注已跑通。以下功能**暂不包含**：

- 批量 episode 生成（单个 episode 的 `--mode full` 全流程已支持）
- GRF_MARL 预训练策略接入
- 事件系统
- 旧版行为克隆 / PPO
- pose keypoints / semantic segmentation / depth
- **amodal** 分割与 occlusion visibility 比例（当前只给模态可见 mask、可见 bbox、`visible_pixel_count`；`visibility` 字段为 `null`）
