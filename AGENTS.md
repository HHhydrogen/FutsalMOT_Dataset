# AGENTS.md

## 仓库边界

本目录是 `FutsalMOT_Dataset` 独立 Git 仓库，远程为 `git@github.com:HHhydrogen/FutsalMOT_Dataset.git`，当前分支为 `main`。外层 UE 仓库通过 submodule gitlink 引用本仓库；外层只记录本仓库 commit，不跟踪这里的 Python 文件内容。

修改本目录的代码、配置或文档时，在本目录执行 Git 命令。完成内层 commit 后，回到外层仓库单独更新 `Content/FutsalMOT/code` 的 submodule 指针。不要混淆两个仓库的工作树、分支或 commit；未经用户明确确认，不得自动 commit 或 push。

### 本仓库提交与推送

本仓库是子仓库本身，提交只影响 Python 仓库，不会自动更新外层 UE 仓库的 gitlink。提交前执行：

```powershell
git status --short --branch
git diff --check
uv run pytest
git diff -- <本次目标文件>
git add <仅限本次目标文件>
git diff --cached --stat
git commit -m "<简体中文提交说明>"
git push origin main
git rev-parse HEAD
```

内层 commit 必须先成功推送到 `origin/main`，外层才可以记录它。完成内层 push 后，外层仓库的更新顺序是：

```powershell
git -C D:\projects\FustalMOT_UEDataset status --short --branch
git -C D:\projects\FustalMOT_UEDataset add Content/FutsalMOT/code
git -C D:\projects\FustalMOT_UEDataset diff --cached --submodule=short -- Content/FutsalMOT/code
git -C D:\projects\FustalMOT_UEDataset commit -m "<简体中文提交说明>"
git -C D:\projects\FustalMOT_UEDataset push origin master
```

注意事项：

- 不要在内层仓库等待或创建“外层指针 commit”；外层指针只能由外层仓库提交。
- 不要在外层执行 `git add -f`、复制内层文件或提交内层 `.git` 目录；外层只应看到 mode `160000` 的 gitlink。
- 不要无选择地运行 `git add -A`，避免把用户或其他任务的改动带入本次 commit。推送前分别检查内层和外层的 `git status --short --branch`、`git log --oneline -10`。
- 如果当前 checkout 是 detached HEAD，先确认目标分支后执行 `git switch main`；外层 submodule 默认以记录的 commit checkout，日常修改前应切到 `main`。
- 如果 `origin/main` 有新提交，先执行 `git fetch origin`，确认历史后使用 `git pull --ff-only origin main`；不要 force-push 覆盖远程历史。
- 外层 clone 后需要在外层执行 `git submodule update --init --recursive`，不能只依赖内层目录是否物理存在。
- 内层 push 失败时不得继续让外层引用该新 commit；外层 push 失败时保留两边工作树，先 fetch/检查历史，不得强推。

## 权威技术文档

进入本仓库工作前先阅读：

1. `README.md`：架构、环境、推荐流程和 CLI 入口。
2. `docs/DATA_CONTRACT.md`：轨迹、相机、Mask、MOT、YOLO、Pose 和 COCO17 数据格式。
3. `docs/VALIDATION_AND_LIMITATIONS.md`：测试、验证、审计、清理和已知限制。

不要引用已删除的 `CLAUDE.md`、`configs/README.md`、`ue/README.md`、`docs/architecture/` 或 `docs/design/` 文档作为当前实现依据。

## 运行环境隔离

- P1 在 Python 3.9 的 `uv` 环境中运行，版本由 `.python-version` 和 `pyproject.toml` 固定；不要升级到 Python 3.10+。
- P1 入口是 `uv run grf-ue task ...`，输出轨迹、Mask 后处理、MOT/YOLO、Pose、audit 和 manifest。
- P2 脚本在外层 UE 5.8 Editor 的真实 Python 环境中运行，入口是 `ue/run_task.py`；绝不能在 P1 `.venv` 中运行依赖 `unreal` 的脚本。
- UE Python 不应导入依赖 NumPy、OpenEXR 或 OpenCV 的 P1 模块。`instance_mask.py`、`cryptomatte.py` 等 P1 模块只在 Python 环境中运行。
- P1/P2 之间的磁盘接口是 JSON/JSONL 和渲染文件；不要让 UE Python 依赖 P1 虚拟环境。

## 推荐工作流

对单个 task，优先使用以下顺序：

```powershell
uv run grf-ue task validate configs/<task>.json
uv run grf-ue task resolve configs/<task>.json
uv run grf-ue task export configs/<task>.json
uv run grf-ue task ue-command configs/<task>.json
# 在外层 UE Editor 中执行生成的 run_task.py --mode full
# MRQ 异步结束后，在 UE Editor 中执行 run_task.py --mode pose-finalize（启用 Pose 时）
uv run grf-ue task postprocess configs/<task>.json
uv run grf-ue task audit configs/<task>.json --validation-level quick
uv run grf-ue task manifest configs/<task>.json
uv run grf-ue task cleanup configs/<task>.json
```

`task ue-command` 只生成 UE 命令，不执行 UE。MRQ 是异步流程，不能在 Editor 主线程中使用阻塞等待。`task cleanup` 默认 dry-run，只有用户明确要求且门禁通过时才使用 `--apply`。

当前 `configs/*.json` 含机器绝对路径，换机器必须检查 `dataset_root`、`ue_project_root`、UE 地图、Actor mapping、相机和 Sequence。不要把所有配置都视为当前有效；先运行 `task validate`。

## Unreal MCP 执行规则

UE 相关任务优先使用外层项目已配置的 Unreal MCP：

1. 完整 UE Python 文件使用 `FutsalMOTTools.run_python_file`，例如 `ue/run_task.py`、渲染、导入和导出脚本。
2. 小型查询使用 `FutsalMOTTools.run_python_code`，例如 `import unreal`、查询 Actor、Transform、组件或输出一条日志。
3. 只查询 Editor 状态时优先使用 Unreal MCP 原生工具：Level、Actor、Transform、Components、Asset、Logs、Sequencer 和 AutomationTest。
4. 禁止使用受限的 `ProgrammaticToolset.execute_tool_script` 执行 `import unreal` 或本仓库的 UE 脚本。
5. 脚本失败时先读取工具返回的 `success/result/log`，再读取 UE `LogPython`、`LogBlueprint`、`LogMovieRenderPipeline` 或 `LogModelContextProtocol`，自行定位并重试。

只有在 MCP/API 无法覆盖、需要视觉人工验收、必须点击不可自动化的 UI、需要完整重启 UE，或操作存在破坏风险时，才请求用户手动操作。

## 正式入口与 Legacy 代码

- 正式 P2 总入口：`ue/run_task.py --resolved-task <path> --mode full`。
- `--mode pose-finalize` 在 MRQ 完成后导出 Runtime Pose、COCO17 和 `pose_keypoints.jsonl`。
- `--mode annotations` 仍可调用旧的逐帧 `pose_export.py`，属于 Legacy，不要与 Runtime Pose 结果混写或混称。
- `ue/import_grf_episode.py` 及 `ue/archive_c4_diag/` 中的旧导入、构建和诊断脚本不是默认正式入口。
- 当前正式 UE 资产路径位于 `/Game/FutsalMOT/Blueprints/Pose/`；若旧脚本仍引用 `/Game/FutsalMOT/Blueprints/` 根路径，先核验资产是否已迁移，不能直接假设脚本可用。

## 测试与文档

默认测试会跳过 GRF 集成测试：

```powershell
uv run pytest
uv run pytest -m grf_integration -q
```

测试通过不等于 UE Blueprint、地图、MRQ、相机标定或 RGB/Mask/Pose 像素对齐已验收。涉及 UE 的工作必须按 MCP 闭环执行，并把静态代码证据和实际 Editor 结果区分记录。

代码注释、文档字符串、Pydantic 字段说明和技术文档使用简体中文。新增或修改文档时只使用根 `README.md` 或平铺的 `docs/*.md`，不要重新创建文档子目录。

不要提交 `__pycache__/`、`*.pyc`、`.venv/`、`.external/`、`outputs/`、`.futsalmot/`、UE `Saved/`/`Intermediate/`/`DerivedDataCache/` 或本地日志和临时文件。
