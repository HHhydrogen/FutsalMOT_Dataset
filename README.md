# FutsalMOT Dataset — GRF-UE Bridge

用 Google Research Football (GRF) 生成足球比赛轨迹 → Unreal Engine Level Sequence 回放 →
RGB + Instance-ID Mask 渲染 → 像素级 CV 标注（MOT / YOLO det / YOLO seg）。

本项目以**数据集任务（dataset task）**为统一入口：一个 task 配置描述一次完整的
「导出 → UE 导入/渲染 → 后处理 → 审计」流程，路径与参数由解析器（resolver）自动合并，
无需在多个命令里反复输入 episode / dataset / mapping / config 路径。

```
GRF 轨迹（P1, .venv）──→ JSONL ──→ UE Level Sequence + 渲染（Unreal Editor）──→ 标注数据集（G:）
     ↑ task export                ↑ ue/run_task.py                         ↑ task postprocess / audit
     └───────────────────────────── 同一 resolved task ───────────────────────────────┘
```

## 快速开始（task 工作流）

> **单一 config 用法（推荐）**：一个 task 文件即可控制全部参数——机器路径
> （`dataset_root`/`ue_project_root`）与任务参数（`seed`/相机/后处理）都写在一个文件里，
> 所有产出（轨迹 + 相机数据）都落到 `<dataset_root>/<episode_name>/` 下**自包含**，
> 代码根 `outputs/` 不再产生新数据。

### 1. 创建单一 config（本地任务，`tasks/` 已 gitignore）

```powershell
Copy-Item configs/tasks/soak_300frames_4cam.example.json tasks/my_dataset.json
```

编辑 `tasks/my_dataset.json`，填上 `dataset_root`（及可选的 `seed`、`ue_project_root`）：

```json
{
  "schema": "futsalmot_dataset_task",
  "version": 1,
  "task_id": "my_dataset",
  "episode_name": "episode_my",
  "dataset_root": "G:/FutsalMOT_Dataset",
  "ue_project_root": "D:/projects/FustalMOT_UEDataset",
  "seed": 1001,
  "export_profile": "../../configs/export/standard_300steps_10fps.json",
  "ue_profile": "../../configs/ue/4cam_1080p_cvgt.json",
  "postprocess": { "include_ball": true, "workers": 4, "validation_level": "full" },
  "audit": { "expected_cameras": 4, "expected_frames_per_camera": 300 }
}
```

> 想用「共享机器路径」而非单一 config：把 `dataset_root` 等放进 `.futsalmot.local.json`
> （`Copy-Item .futsalmot.local.example.json .futsalmot.local.json`），task 里可省略。
> 优先级：**task 内字段 > 环境变量 `FUTSALMOT_*` > `.futsalmot.local.json` > 默认**。

### 2. 验证并解析

```powershell
uv run grf-ue task validate tasks/my_dataset.json
uv run grf-ue task resolve tasks/my_dataset.json
```

### 3. 导出轨迹（产出到 dataset_root）

```powershell
uv run grf-ue task export tasks/my_dataset.json
```

产出：`<dataset_root>/<episode_name>/{meta.json, frames.jsonl, provenance/}`。

### 4. UE 运行

```powershell
uv run grf-ue task ue-command tasks/my_dataset.json
```

把输出命令复制到 **Unreal Editor Python Console**（`py ".../ue/run_task.py" --resolved-task ...`）。
MRQ 渲染异步，完成后写 `render_summary.json`；相机数据写入同一 `<dataset_root>/<episode_name>/`。

### 5. 后处理 + 审计

```powershell
uv run grf-ue task postprocess tasks/my_dataset.json
uv run grf-ue task audit tasks/my_dataset.json
uv run grf-ue task status tasks/my_dataset.json
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
uv run grf-ue task activate tasks/my_dataset.json
uv run grf-ue task status            # 之后可省 task 参数
uv run grf-ue task deactivate
```

每次使用 active task 都会打印其来源，避免隐藏状态；显式 task 参数始终优先。

## task 结构

```text
configs/
├── export/          # 导出 profile（ExportConfig 字段）
│   ├── smoke_3steps_10fps.json
│   ├── short_90frames_30fps.json      # 30 步 ×3 插值 = 90 帧（30fps 1:1）
│   ├── standard_300steps_10fps.json
│   └── standard_300steps_30fps.json
├── ue/              # UE profile（episode 无关：相机/分辨率/渲染/Mask，无本机路径）
│   ├── 1cam_1080p_cvgt.json
│   └── 4cam_1080p_cvgt.json
└── tasks/           # 任务示例（只引用 profile，不复制内容）
    ├── smoke_3frames_1cam.example.json
    ├── smoke_90frames_1cam.example.json
    ├── smoke_90frames_4cam.example.json
    └── soak_300frames_4cam.example.json
```

task 文件安全约束：**默认禁止盘符/UNC 绝对路径与 `..` 逃逸**；开发兼容可用
`--allow-absolute-paths`。机器路径只在 `.futsalmot.local.json` / 环境变量。

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
├── cryptomatte-to-mask / annotate-overlay / make-video
└── build-manifest / verify-manifest    # 数据集 manifest
```

## 架构

### P1：导出（`src/grf_ue_bridge/`）

- `config/` — 本地配置 / task / profile / resolved task 的模型、加载与路径解析。
- `grf_runner.py` — 运行 GRF；root seed 经 SHA-256 派生 `game_engine_random_seed`
  **真正传入引擎**（见 `docs/REPRODUCIBILITY_AND_MANIFEST.md`）。
- `exporter.py` — 写 `meta.json` + `frames.jsonl` + `provenance/` 配置快照。
- `workflows/` — `task_export` / `task_postprocess` / `task_audit` / `task_status`。
- `tools/` — `resource_monitor` / `process_measure` / `benchmark_postprocess`。
- `mask_annotator.py` / `cryptomatte.py` / `annotation_validator.py` — 标注链路。
- `dataset_manifest.py` — 数据集索引 / 校验和 / fingerprint / 去重。

### P2：Unreal（`ue/`）

- `run_task.py` — **统一 UE 入口**：读同一 resolved task，调用既有导入/渲染逻辑；
  不再隐式读取根目录配置。
- `import_grf_episode.py` — Sequence 创建 / 标注导出 / 渲染（`--config` 为 legacy 模式）。
- `render_episode.py` — MRQ 异步渲染 RGB + Object ID EXR、watchdog、`render_summary`。
- `recover_render.py` — 从已有 `render/` 恢复 `img1/`（`--resolved-task`）。

### 数据契约（与 P2 共享）

`outputs/<episode>/` 含 `meta.json`（schema、时序、场地、实体、`randomness` 种子体系）与
`frames.jsonl`（每帧 `step/time_seconds/score/ball/players`），坐标为米 `[x,y,z]`。

## CV 标注链路（mask-primary）

1. UE 渲染：RGB → `img1/`，Object ID EXR → `render_mask/`（Cryptomatte）。
2. `grf-ue task postprocess` → cryptomatte → `mask/*.png`（mask_id 1~11）→
   mask-primary bbox / 分割 / MOT / YOLO。
3. bbox 的 primary GT 来自 Instance-ID Mask 可见像素；几何投影 bbox 保留在
   `geometry_bbox_*` 作 fallback。

详见 [`docs/architecture/INSTANCE_MASK_PIPELINE.md`](docs/architecture/INSTANCE_MASK_PIPELINE.md)。

## 可复现性与 Manifest

- Seed：root seed → 子 seed（`futsalmot_seed_v1`），`game_engine_random_seed` 真正传入 GRF；
  同 seed 独立进程 `frames.jsonl` 完全一致。
- Manifest：`build-manifest` / `verify-manifest`，checksum profile（metadata/final/all）、
  稳定 fingerprint、重复轨迹检测。

详见 [`docs/REPRODUCIBILITY_AND_MANIFEST.md`](docs/REPRODUCIBILITY_AND_MANIFEST.md)。

## 迁移与兼容

- 根目录 `ue_import_config*.json` 已移除；机器路径在 `.futsalmot.local.json`。
- 旧 CLI 保留但 deprecated（打印 warning，不再读根配置）；删除计划见
  [`docs/migration/TASK_CONFIG_MIGRATION.md`](docs/migration/TASK_CONFIG_MIGRATION.md)。

## Legacy commands（旧多路径命令，已弃用）

```powershell
uv run grf-ue export --config configs/export/standard_300steps_10fps.json --output outputs/episode_0001 --seed 42
uv run grf-ue validate outputs/episode_0001
uv run grf-ue cryptomatte-to-mask G:/FutsalMOT_Dataset/episode_0001 --episode outputs/episode_0001 --mapping ue/actor_mapping.example.json
uv run grf-ue annotate-masks G:/FutsalMOT_Dataset/episode_0001 --include-ball
uv run grf-ue validate-annotations G:/FutsalMOT_Dataset/episode_0001 --validation-level full
```

## 开发

```powershell
uv run pytest tests/          # 默认套件（不含 GRF 集成）
uv run pytest -m grf_integration -q   # 真实 GRF seed 复现集成测试
uv build
```

## 常见问题

| 现象 | 解决 |
|------|------|
| `task validate` 报「缺少本地路径」 | 建 `.futsalmot.local.json` 或设 `FUTSALMOT_*` 环境变量 |
| UE 找不到 actor | 检查 ue profile 的 `actor_mapping` 指向的 JSON 与关卡标签一致 |
| 球陷进地面 / 倒着滚 | 调 `ball_rolling` 的 `BALL_Z_OFFSET_CM` / `roll_sign` |
| 渲染未写 `img1/` | 检查 `render_summary.json` 状态；用 `ue/recover_render.py --resolved-task ...` 恢复 |
