# FutsalMOT Dataset — GRF-UE Bridge

用 Google Research Football (GRF) 生成足球比赛轨迹 → Unreal Engine Level Sequence 回放 →
RGB + Instance-ID Mask 渲染 → 像素级 CV 标注（MOT / YOLO det / YOLO seg）。

本项目以**数据集任务（dataset task）**为统一入口：**一个自包含配置（单 config）**描述一次
完整的「导出 → UE 导入/渲染 → 后处理 → 审计」流程——导出参数、UE 相机/渲染参数与
机器路径（`dataset_root`/`ue_project_root`）都写在同一文件里，直接入库。

```
GRF 轨迹（P1, .venv）──→ JSONL ──→ UE Level Sequence + 渲染（Unreal Editor）──→ 标注数据集（G:）
     ↑ task export                ↑ ue/run_task.py                         ↑ task postprocess / audit
     └───────────────────────────── 同一 resolved task ───────────────────────────────┘
```

## 快速开始（task 工作流）

> **单 config（唯一用法）**：一个 task 文件包含全部参数与机器路径（真实路径入库），
> 所有产出（轨迹 + 相机数据）都落到 `<dataset_root>/<episode_name>/` 下**自包含**，
> 代码根 `outputs/` 不再产生新数据。

### 1. 使用或新建单 config

仓库内已提交的自包含单 config（`configs/*.json`）可直接运行（机器路径已入库）：
冒烟/demo 用 `configs/pose_smoke_3frames_1cam.json`（yolo_pose 已启用）。

新 episode：复制 `configs/example.json` 到新文件名，替换占位符并改参数。
**每个参数的说明与填写指南见 [`configs/README.md`](configs/README.md)**（含 `example.json`
完整模板、各字段默认值与一致性校验规则）。

### 2. 验证并解析

```powershell
uv run grf-ue task validate configs/my_dataset.json
uv run grf-ue task resolve configs/my_dataset.json
```

### 3. 导出轨迹（产出到 dataset_root）

```powershell
uv run grf-ue task export configs/my_dataset.json
```

产出：`<dataset_root>/<episode_name>/{meta.json, frames.jsonl, provenance/}`。

### 4. UE 运行

```powershell
uv run grf-ue task ue-command configs/my_dataset.json
```

把输出命令复制到 **Unreal Editor Python Console**（`py ".../ue/run_task.py" --resolved-task ...`）。
MRQ 渲染异步，完成后写 `render_summary.json`；相机数据写入同一 `<dataset_root>/<episode_name>/`。

### 5. 后处理 + 审计

```powershell
uv run grf-ue task postprocess configs/my_dataset.json
uv run grf-ue task audit configs/my_dataset.json
uv run grf-ue task status configs/my_dataset.json
```

### 输出布局（自包含，全在 dataset_root）

```text
<dataset_root>/<episode_name>/
├── meta.json / frames.jsonl / provenance/     # 轨迹（task export）
├── render_summary.json
├── CineCam_01/…{camera.json, img1/, mask/, render/, render_mask/, labels/, gt/}
└── …（每相机）
```

### 可选：active task

```powershell
uv run grf-ue task activate configs/my_dataset.json
uv run grf-ue task status            # 之后可省 task 参数
uv run grf-ue task deactivate
```

每次使用 active task 都会打印其来源，避免隐藏状态；显式 task 参数始终优先。

## 单 config 结构

```text
configs/     # 每个数据集任务一个自包含 JSON（导出 + UE + 机器路径，入库）
├── example.json                  # 完整参数模板（占位符路径）
├── README.md                     # 参数详解与填写指南
└── pose_smoke_3frames_1cam.json  # 冒烟/demo：3 步 × 1 相机 = 3 帧，yolo_pose 已启用
```

- 导出参数在 `export` 块（scenario / seed / num_steps / fps / 场地尺寸）。
- UE 参数在 `ue` 块（actor_mapping / sequences / ball_rolling / annotation_export）。
- 机器路径 `dataset_root` / `ue_project_root` 必填且直接入库；`repo_root` 自动探测。
- 所有产出统一落 `<dataset_root>/<episode_name>/`（轨迹 + 相机数据自包含）。
- **每个字段的含义、默认值与一致性规则见 [`configs/README.md`](configs/README.md)**。

## CLI 总览

```text
grf-ue
├── task            # 推荐入口
│   ├── validate / resolve / export / ue-command
│   ├── postprocess / audit / status
│   └── activate / deactivate
├── monitor <task>  # 渲染期间资源/目录增长监控
├── measure -- <cmd>  # 命令墙钟 + 进程树峰值 RSS
├── benchmark       # 后处理性能基准（透传参数）
├── export --config --output [--seed]   # Legacy（deprecated）
├── validate / validate-annotations / annotate-masks
├── annotate-pose / validate-pose / pose-overlay   # YOLO Pose（COCO 17 点）
├── cryptomatte-to-mask / annotate-overlay / make-video
└── build-manifest / verify-manifest    # 数据集 manifest
```

## 架构

### P1：导出（`src/grf_ue_bridge/`）

- `config/` — dataset task（单 config）与 resolved task 的模型、加载与解析。
- `grf_runner.py` — 运行 GRF；root seed 经 SHA-256 派生 `game_engine_random_seed`
  **真正传入引擎**（见 `docs/REPRODUCIBILITY_AND_MANIFEST.md`）。
- `exporter.py` — 写 `meta.json` + `frames.jsonl` + `provenance/` 配置快照。
- `workflows/` — `task_export` / `task_postprocess` / `task_audit` / `task_status`。
- `tools/` — `resource_monitor` / `process_measure` / `benchmark_postprocess`。
- `mask_annotator.py` / `cryptomatte.py` / `annotation_validator.py` — 标注链路。
- `pose_annotator.py` / `pose_validator.py` — YOLO Pose 标注与校验（见下）。
- `dataset_manifest.py` — 数据集索引 / 校验和 / fingerprint / 去重。

### P2：Unreal（`ue/`）

- `run_task.py` — **统一 UE 入口**：读同一 resolved task，调用既有导入/渲染逻辑；
  不再隐式读取根目录配置。
- `import_grf_episode.py` — Sequence 创建 / 标注导出 / 渲染（`--config` 为 legacy 模式）。
- `render_episode.py` — MRQ 异步渲染 RGB + Object ID EXR、watchdog、`render_summary`。
  渲染前**首帧 spawn 状态烘焙**（把 actor 设到第 0 帧并保存关卡，修复 MRQ 首帧因
  possessable 未被 Sequence 接管而渲成关卡默认位置的问题）。
- `pose_bones.py` / `pose_export.py` — COCO 17 点 ↔ UE 骨骼映射与关键点导出（见下）。
- `recover_render.py` — 从已有 `render/` 恢复 `img1/`（`--resolved-task`）。

### 数据契约（与 P2 共享）

`<dataset_root>/<episode_name>/` 含 `meta.json`（schema、时序、场地、实体、`randomness` 种子体系）与
`frames.jsonl`（每帧 `step/time_seconds/score/ball/players`），坐标为米 `[x,y,z]`。

## CV 标注链路（mask-primary）

1. UE 渲染：RGB → `img1/`，Object ID EXR → `render_mask/`（Cryptomatte）。
2. `grf-ue task postprocess` → cryptomatte → `mask/*.png`（mask_id 1~11）→
   mask-primary bbox / 分割 / MOT / YOLO。
3. bbox 的 primary GT 来自 Instance-ID Mask 可见像素；几何投影 bbox 保留在
   `geometry_bbox_*` 作 fallback。

详见 [`docs/architecture/INSTANCE_MASK_PIPELINE.md`](docs/architecture/INSTANCE_MASK_PIPELINE.md)。

## YOLO Pose（COCO 17 点人体关键点）

在 mask-primary 流程之上，为每个球员生成 Ultralytics YOLO Pose 标签（17 点，每行
`class xc yc w h x1 y1 v1 ... x17 y17 v17` 共 **56 字段**）。

### 开启

在 task 配置的 `postprocess.yolo_pose` 块开启（默认关闭，不影响原 pipeline）：

```json
"postprocess": {
  ...
  "yolo_pose": { "enabled": true }
}
```

启用后流程自动变为：`task export` → `task ue-command`（UE 运行，额外导出
`pose_keypoints.jsonl`）→ `task postprocess`（annotate-masks → **annotate-pose** →
validate-annotations → **validate-pose**）。

### COCO 17 点定义与顺序（严禁改动）

```text
0 nose  1 left_eye  2 right_eye  3 left_ear  4 right_ear
5 left_shoulder  6 right_shoulder  7 left_elbow  8 right_elbow
9 left_wrist  10 right_wrist  11 left_hip  12 right_hip
13 left_knee  14 right_knee  15 left_ankle  16 right_ankle
```

可见性：`v=0` 无效 / `v=1` 被遮挡（其他球员 / 球 / 自遮挡，基于 Instance-ID Mask
邻域 + UE 遮挡 trace）/ `v=2` 可见。bbox 复用 mask-primary bbox（与 YOLO det 完全一致）。

### 输出

```text
<episode_root>/<camera>/labels_pose/000001.txt   # YOLO Pose 标签
<episode_root>/yolo_pose/                        # 可训练暂存（images 硬链接 + labels）
<episode_root>/futsal_pose.yaml                  # dataset YAML（kpt_shape [17,3]）
```

### 常用命令

```powershell
uv run grf-ue task postprocess configs/my_dataset.json      # 含 annotate-pose + validate-pose
uv run grf-ue validate-pose <dataset_root>/<episode_name>   # 单独校验
uv run grf-ue pose-overlay <dataset_root>/<episode_name>/<camera> --frames 1,2,3  # 可视化验证
uv run grf-ue annotate-pose <dataset_root>/<episode_name>   # 单独重跑 pose 标签
```

### 用 Ultralytics 训练

```bash
yolo pose train model=yolo11n-pose.pt data=<dataset_root>/<episode_name>/futsal_pose.yaml
```

> `futsal_pose.yaml` 的 `train`/`val` 指向 episode 内 `yolo_pose/` 的 `images/` 目录，
> 标签在 `labels/` 同级目录（Ultralytics 自动按 `images→labels` 发现），无需改格式。
> 骨骼映射与脸部偏移说明见 [`docs/design/2026-08-11-yolo-pose-export.md`](docs/design/2026-08-11-yolo-pose-export.md)。

## 可复现性与 Manifest

- Seed：root seed → 子 seed（`futsalmot_seed_v1`），`game_engine_random_seed` 真正传入 GRF；
  同 seed 独立进程 `frames.jsonl` 完全一致。
- Manifest：`build-manifest` / `verify-manifest`，checksum profile（metadata/final/all）、
  稳定 fingerprint、重复轨迹检测。

详见 [`docs/REPRODUCIBILITY_AND_MANIFEST.md`](docs/REPRODUCIBILITY_AND_MANIFEST.md)。

## 开发

```powershell
uv run pytest tests/          # 默认套件（不含 GRF 集成）
uv run pytest -m grf_integration -q   # 真实 GRF seed 复现集成测试
uv build
```

## 常见问题

| 现象 | 解决 |
|------|------|
| `task validate` 报「解析失败」 | 检查 `configs/*.json` 的 schema/version、`dataset_root`/`ue_project_root` 与 export/ue 块是否完整 |
| UE 找不到 actor | 检查 `ue` 块的 `actor_mapping` 指向的 JSON 与关卡标签一致 |
| 球陷进地面 / 倒着滚 | 调 `ue` 块 `ball_rolling` 的 `BALL_Z_OFFSET_CM` / `roll_sign` |
| 渲染未写 `img1/` | 检查 `render_summary.json` 状态；用 `ue/recover_render.py --resolved-task ...` 恢复 |
| **首帧渲成关卡默认位置**（后续帧正常） | MRQ/PIE 第 0 帧 possessable actor 尚未被 Sequence 接管——`render_episode.py` 已做**首帧 spawn 状态烘焙**（提交渲染前把 actor 设到第 0 帧并保存关卡）；确认 UE 控制台打印 `[MRQ] 首帧 spawn 状态已烘焙` |
| pose 控制台报 `unreal.ETraceTypeQuery` / `无法读取球员骨骼名` | 是旧版代码：`run_task.py` 已带强制 reload，**重跑同一命令即可**（新代码 probe 解析骨骼、遮挡 trace 全防御，不崩溃） |
| pose-overlay 里关键点错位 | 用 `--keypoint-names` 核对；脸部五点调 `postprocess.yolo_pose.head_offsets_cm`；若因动画姿势，把 mesh 动画置空或确认导出/渲染姿势一致 |
