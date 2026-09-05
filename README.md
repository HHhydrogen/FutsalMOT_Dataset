# FutsalMOT GRF-UE 数据集代码

本仓库是 FutsalMOT 项目的 Python 数据集代码仓库，包名为 `grf-ue-bridge`。它把 Google Research Football 的 5v5 轨迹导出为 JSONL，再由 Unreal Engine 5.8 导入、创建 Level Sequence、渲染 RGB 和 Instance-ID Mask，最后生成 MOT、YOLO 和 COCO17 姿态标注。

本目录位于外层 UE 仓库的 `Content/FutsalMOT/code/`，但拥有独立的 Git 历史和远程仓库。外层通过 submodule gitlink 引用本仓库 commit；修改本仓库时必须在本目录执行 Git 命令，内层 commit 完成后再回到外层更新 gitlink。

## 两个运行环境

| 阶段 | 环境 | 代码入口 | 允许依赖 | 主要输出 |
| --- | --- | --- | --- | --- |
| P1 轨迹和后处理 | Python 3.9、`uv` 虚拟环境 | `src/grf_ue_bridge/cli.py`，命令为 `grf-ue` | GRF、NumPy、OpenEXR、Pillow、OpenCV 等 | episode 轨迹、Mask、MOT、YOLO、审计和 manifest |
| P2 导入和渲染 | Unreal Editor 内置 Python | `ue/run_task.py` | `unreal` 和 UE 侧纯 Python 模块 | Level Sequence、RGB、Cryptomatte EXR、Runtime Pose |

JSON/JSONL 文件是两个环境之间的磁盘接口。UE 脚本不能在 P1 `.venv` 中运行；P1 的 OpenEXR/NumPy 后处理也不能在 UE Python 中运行。`ue/` 中的 `camera_projection.py`、`annotation_utils.py`、`dataset_export.py`、`player_motion.py` 和 `pose_bones.py` 是纯 Python 模块，可被对应的两侧脚本复用；`instance_mask.py` 和 `cryptomatte.py` 依赖 NumPy/Pillow 或 OpenEXR，只在 P1 使用。

## 环境准备

源码和锁文件将 Python 限制为 3.9：`.python-version` 为 `3.9`，`pyproject.toml` 要求 `>=3.9,<3.10`，`uv.lock` 也固定为 3.9 系列。安装和测试：

```powershell
cd D:\projects\FustalMOT_UEDataset\Content\FutsalMOT\code
uv sync
uv run pytest
```

主要运行依赖包括 `gfootball>=2.10.2`、`gym<0.26`、`numpy`、`opencv-python`、`openexr`、`pillow`、`pydantic`、`six>=1.17.0` 和 `typer`；开发依赖为 `pytest`。`.external/` 下的外部仓库不由 Git 跟踪，版本锁定记录在 `external_sources.lock.json`：

- `google-research-football`：`3d9e754720a95621bba6475c4d3b0d56fe919014`
- `GRF_MARL`：`6cf67a509dc204f5f413adaa57619652580c80f1`

现有 `configs/*.json` 是单文件任务配置，包含 `dataset_root` 和 `ue_project_root` 的机器绝对路径。换机器时必须检查这些路径；不要假设配置中的盘符在其他环境存在。

## 推荐流程

以下流程假定目标 task 的路径和 UE 资产前置条件已经在当前机器上成立。完整字段定义见 `docs/DATA_CONTRACT.md`，当前限制和验收要求见 `docs/VALIDATION_AND_LIMITATIONS.md`。

### 1. 校验和导出轨迹

```powershell
uv run grf-ue task validate configs/pose_smoke_3frames_1cam.json
uv run grf-ue task resolve configs/pose_smoke_3frames_1cam.json
uv run grf-ue task export configs/pose_smoke_3frames_1cam.json
```

`task validate` 只读检查 task schema、格式、相机数、按 `num_steps * target_fps/10` 计算的预期帧数和部分时长条件。`task resolve` 将单文件 task 解析为运行时契约，并写入被忽略的 `.futsalmot/runtime/<task_id>/resolved-task.json`。`task export` 调用 GRF 并在 `<dataset_root>/<episode_name>/` 写入 `meta.json`、`frames.jsonl` 和 `provenance/`。

`task export` 内部会在需要时为时间缩放采集更多 GRF 源步，再按 task 的 `target_fps` 产生数据集帧。GRF 运行器强制使用 `representation="raw"`、`action_set="v2"`、左队名义控制 1 名球员并发送 `action_builtin_ai` 动作，因此实际轨迹是双方内置 AI 对局，而不是用户控制球员的交互轨迹。

### 2. 在 UE 中创建 Sequence、导出标注并提交 MRQ

```powershell
uv run grf-ue task ue-command configs/pose_smoke_3frames_1cam.json
```

该命令保存 resolved task 并输出类似下面的 UE Python Console 命令：

```python
py "D:/projects/FustalMOT_UEDataset/Content/FutsalMOT/code/ue/run_task.py" --resolved-task "D:/projects/FustalMOT_UEDataset/Content/FutsalMOT/code/.futsalmot/runtime/pose_smoke_3frames_1cam/resolved-task.json" --mode full
```

在 UE Editor Python 环境中，`run_task.py` 支持以下模式：

| 模式 | 实际行为 |
| --- | --- |
| `sequence` | 读取 episode 和 Actor mapping，创建或覆盖配置的 Level Sequence |
| `annotations` | 导出相机标定和几何 AABB 标注；若显式启用 Pose，使用旧的逐帧 `pose_export.py` |
| `full` | 创建 Sequence、导出几何标注；若启用 YOLO Pose 则准备 Runtime Pose Recorder，然后异步提交 RGB/Mask MRQ |
| `render` | 不创建 Sequence，只使用已有 Sequence 导出并提交渲染 |
| `pose-finalize` | 渲染结束后读取 Runtime Pose SaveGame，生成 Pose 和 COCO17 文件 |

`full` 不等待 MRQ 完成。MRQ 完成回调或非阻塞 watchdog 会把 RGB 对齐复制到 `img1/`，统计或保留 Object-ID EXR，并写 `render_summary.json`。执行期间必须保持 Editor 继续 tick；不要在 UE 主线程中用 `sleep` 或阻塞等待。

如果 task 的 `postprocess.yolo_pose.enabled` 为 `true`，MRQ 完成后还必须运行：

```python
py "D:/projects/FustalMOT_UEDataset/Content/FutsalMOT/code/ue/run_task.py" --resolved-task "D:/.../resolved-task.json" --mode pose-finalize
```

该步骤只从第一个配置相机的 5 个 Recorder slot 导出世界骨骼数据，因为 Runtime Pose 与相机无关。它会先删除旧的 `pose_capture.jsonl` 和 `pose_session.json`，只有捕获帧数完整且结构合法时才写入正式 Pose 文件。

`FutsalMOTMCP` 插件的 `run_python_file` 可在真实 UE Python 环境执行项目内脚本，`run_python_code` 适合短诊断。工具接口当前不接收 `run_task.py` 的命令行参数，因此需要参数时以 `task ue-command` 生成的命令为准，或由调用方为 `C5_RESOLVED_TASK`、`C5_RUN_MODE` 等环境变量准备好运行上下文。两种方式都必须确认脚本实际在 UE Editor 内执行。

### 3. P1 后处理和验收

```powershell
uv run grf-ue task postprocess configs/pose_smoke_3frames_1cam.json
uv run grf-ue task audit configs/pose_smoke_3frames_1cam.json --validation-level quick
uv run grf-ue task manifest configs/pose_smoke_3frames_1cam.json
uv run grf-ue task cleanup configs/pose_smoke_3frames_1cam.json
```

`task postprocess` 按顺序运行 Cryptomatte EXR -> `mask/`、`annotate-masks`、标注验证，以及可选的 YOLO Pose 和 debug 可视化。`task audit` 写入 episode 下的 `audit/soak_audit_report.json` 和 `.md`，检查相机、帧数、RGB、Mask、同步、ID 映射、标定、渲染完成标记和可选 Pose。`task manifest` 写入 episode 级 `dataset_manifest.json` 和 `checksums/` 中的校验文件。`task cleanup` 默认只做 dry-run；只有显式 `--apply` 才会删除已列入清理集合的临时产物。当前清理门禁和配置限制见 `docs/VALIDATION_AND_LIMITATIONS.md`。

## CLI 总览

推荐以 `task` 子命令为主：

| 命令 | 作用 |
| --- | --- |
| `task validate` | 只读校验单文件 task |
| `task resolve` | 写入 resolved task |
| `task export` | 运行 GRF 并导出 episode |
| `task ue-command` | 输出 UE `run_task.py` 命令 |
| `task postprocess` | Cryptomatte、Mask、MOT/YOLO、Pose 和 debug 后处理 |
| `task audit` | episode 完整性和跨相机一致性审计 |
| `task motion-quality` | 对 `frames.jsonl` 做运动质量分析 |
| `task manifest` | 写 episode 级 manifest |
| `task status` | 只读显示当前产物数量 |
| `task cleanup` | dry-run 或应用临时产物清理 |
| `task activate` / `deactivate` | 设置或清除默认 active task |

直接命令仍存在，用于局部操作或兼容旧流程：

```text
grf-ue export
grf-ue validate
grf-ue build-manifest
grf-ue verify-manifest
grf-ue validate-annotations
grf-ue annotate-masks
grf-ue cryptomatte-to-mask
grf-ue annotate-pose
grf-ue validate-pose
grf-ue annotate-overlay
grf-ue pose-overlay
grf-ue make-video
grf-ue debug
grf-ue monitor
grf-ue measure
grf-ue benchmark
```

直接 `export` 读取的是只含 `ExportConfig` 字段的旧式 JSON；包含 `schema`、`dataset_root`、`ue`、`postprocess` 和 `audit` 的单文件配置应使用 `task` 工作流。

## 代码分层

```text
src/grf_ue_bridge/
  cli.py                         Typer CLI
  config/models.py               task、resolved task 和导出配置模型
  config/loader.py               单文件 JSON 加载
  config/resolver.py             路径解析、active task、resolved task
  grf_runner.py                  GRF 环境和快照采集
  exporter.py                    meta.json、frames.jsonl、provenance
  interpolate.py                 目标帧率插值和时间缩放重采样
  cryptomatte.py                 EXR -> Instance-ID Mask
  mask_annotator.py              Mask -> bbox、MOT、YOLO
  pose_annotator.py              3D Pose -> YOLO Pose
  annotation_validator.py        CV 标注验证
  pose_validator.py              YOLO Pose 验证
  dataset_manifest.py            SHA-256、fingerprint、manifest
  workflows/                     task export/postprocess/audit/cleanup/status
  tools/                         监控、计时和性能基准

ue/
  run_task.py                    UE 正式统一入口
  scene_apply.py                 Actor 变换和查找
  import_grf_episode.py          旧式导入和 Sequence 创建实现
  annotation_exporter.py         UE 相机标定和几何标注
  render_episode.py              异步 MRQ RGB/Mask 渲染
  pose_render.py                 C4 Recorder 准备和专用 Pose 渲染
  pose_capture_export.py         SaveGame -> pose_capture.jsonl
  build_coco17.py                Runtime Pose -> COCO17 3D/2D
  pose_bones.py                  COCO17 与 UE 骨骼映射
  instance_mask.py               P1 Mask 纯函数，不在 UE 中导入
  archive_c4_diag/               历史诊断脚本，非正式入口
```

## 当前事实与限制

- `ue/run_task.py --mode full` 是当前正式 P2 总入口；它使用 Runtime Pose Recorder，不调用旧的逐帧 `pose_export.py`。
- `ue/run_task.py --mode annotations` 在启用 Pose 时仍会调用 Legacy `pose_export.py`，不能把它与 Runtime Pose 结果混为同一来源。
- Actor mapping 默认文件是 `ue/actor_mapping.example.json`，映射为 `L0..L4 -> Player_L0..Player_L4`、`R0..R4 -> Player_R0..Player_R4`、`BALL -> Ball_01`。代码的 Actor 查找在精确大小写不敏感匹配失败后还会使用包含匹配，因此错误标签可能匹配到非预期 Actor。
- 当前资产已经位于 `/Game/FutsalMOT/Blueprints/Pose/...`，但部分旧的构建、Smoke 和诊断脚本仍写着 `/Game/FutsalMOT/Blueprints/` 根路径。除 `run_task.py` 正式路径外，不能默认这些脚本可直接重建当前资产。
- `configs/*.json` 是实际入库的机器相关配置，不是所有文件都能在任意机器直接通过 `task validate`；必须先核对路径、预期相机数和预期帧数。
- 二进制 UE 资产、当前 Editor 中的 Actor 状态、Blueprint 编译结果和 MRQ 实际输出，必须在 UE Editor 中单独验收。
