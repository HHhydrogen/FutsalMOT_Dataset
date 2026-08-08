# 单 config 数据集任务 + 本地路径入库

日期：2026-08-08
状态：approved（用户已确认设计，随后实施）

## 背景与目标

当前一个数据集（单集）输出任务被拆成四份配置：

- `configs/tasks/<task>.example.json` — 任务文件，仅引用 profile（不复制内容）
- `configs/export/<profile>.json` — py/.venv 侧导出参数（ExportConfig）
- `configs/ue/<profile>.json` — UE 侧相机/Sequence/渲染参数（UeProfile）
- `.futsalmot.local.json`（git 忽略）+ `.futsalmot.local.example.json` + `FUTSALMOT_*` 环境变量 — 机器路径

另有整套「防止本地路径入库」机制：gitignore 规则、`--allow-absolute-paths`、路径安全
（拒绝盘符/绝对路径）、`LocalConfig` 模型、resolver 的三层优先级合并。

用户要求：

1. **接受本地路径入库**——取消全部防止本地路径入库的改动，机器路径直接写进提交的配置。
2. **单次数据集（单集）输出任务仅一个 config 文件**——导出参数与 UE 参数不再分开。

已确认的三项设计决策：

- **完全内联**：每个任务一个自包含 JSON（导出 + UE + 机器路径），删除 `configs/export/` 与 `configs/ue/`。
- **位置**：新建 `configs/<task_id>.json`，去掉 `.example` 后缀，真实路径直接入库；不再「复制到 tasks/」。
- **运行时**：保留 resolved task（`.futsalmot/runtime/`，git 忽略）作为 P1↔UE 契约；resolve 只是把单 config 归一化一次。

## 目标配置 schema（`futsalmot_dataset_task` v2）

```jsonc
// configs/production_300frames_4cam.json
{
  "schema": "futsalmot_dataset_task",
  "version": 2,
  "task_id": "production_300frames_4cam",
  "episode_name": "episode_production_300f_4cam",

  "dataset_root": "G:/FutsalMOT_Dataset",
  "ue_project_root": "D:/projects/FustalMOT_UEDataset",
  // repo_root 自动探测（按 pyproject.toml 向上定位），无需配置

  "export": { "scenario": "5_vs_5", "seed": 42, "num_steps": 300, "playback_fps": 30,
              "field_length_m": 40.0, "field_width_m": 20.0 },
  "ue": { "actor_mapping": "ue/actor_mapping.example.json",
          "sequence_package_path": "/Game/FutsalMOT/Sequences",
          "sequences": [ /* 相机列表 */ ],
          "replace_existing": true,
          "ball_rolling": { ... },
          "annotation_export": { "cameras": [...], "render_rgb": {...}, "instance_mask": {...} } },
  "postprocess": { "include_ball": true, "workers": 4, "chunk_size": 50, "png_compress_level": 1,
                   "formats": ["json","mot","yolo-det","yolo-seg"], "clean_stale": true,
                   "validation_level": "full" },
  "audit": { "expected_cameras": 4, "expected_frames_per_camera": 300 }
}
```

要点：

- `export` 块直接复用 `ExportConfig` 模型校验；`ue` 块复用 `UeProfile` 字段（去掉其独立 schema）。
- 删除 `paths` 覆盖块与顶层 `seed` 覆盖：轨迹与数据集统一落 `<dataset_root>/<episode_name>/`
  （与现默认一致），seed 直接写在 `export.seed`。
- schema/version 升到 2；resolved task schema 不变。

## 删除的机制（取消防本地路径入库）

| 项 | 处置 |
|----|------|
| `.futsalmot.local.json` / `.futsalmot.local.example.json` | 删除；机器路径并入单 config |
| `configs/export/`、`configs/ue/`、`configs/tasks/`（含 `.example`） | 删除；参数内联进 `configs/` |
| `FUTSALMOT_*` 环境变量（REPO/UE/DATASET/LOCAL_CONFIG） | 删除 |
| `LocalConfig` 模型、`load_local_config` / `resolve_local_paths` / `apply_task_path_overrides` / `resolve_paths_for_task` | 删除 |
| gitignore：`configs/local*.json`、`*.local.json`、`.futsalmot.local.json`、`tasks/local/` | 移除 |
| `--allow-absolute-paths` 标志（所有 task 子命令） | 移除 |
| `is_absolute_unsafe` / `resolve_with_allow_absolute` / 拒绝绝对路径 | 移除（task 现在直接含绝对路径） |
| `ExportProfile` 模型、`EXPORT_PROFILE_SCHEMA` / `UE_PROFILE_SCHEMA` / `LOCAL_CONFIG_SCHEMA` | 删除 |

**保留**：

- resolved task（`.futsalmot/runtime/` git 忽略）——P1↔UE 运行时契约不变，`ue/run_task.py` 不改。
- provenance 占位符（`${REPO_ROOT}` 等）——属数据集输出的可移植性（`sanitize_resolved_task`），
  与「仓库是否入库本地路径」无关，保留。
- `tasks/` 复制流程删除，但 `.futsalmot/`（运行时/active task）gitignore 保留。

## 解析器行为

- `repo_root`：`default_repo_root()` 自动探测（按 pyproject.toml），无需配置。
- `dataset_root` / `ue_project_root`：直接取 task 内字段（绝对路径，必填）。
- `trajectory_output` = `dataset_episode_dir` = `<dataset_root>/<episode_name>`。
- `actor_mapping`：相对仓库根解析（保留 `..` 逃逸检查；绝对路径放行）。
- `resolve` 产物（resolved task）字段与 v1 一致：`export_profile` / `ue_profile` dict、
  `task_id` / `episode_name` / `dataset_root` / `ue_project_root` / `trajectory_output` /
  `dataset_episode_dir` / `actor_mapping` / `postprocess` / `audit`。
- `validate_task` 校验：task 可解析（schema/version）、`dataset_root` / `ue_project_root` 存在、
  相机数 == `audit.expected_cameras`、导出帧数 == `audit.expected_frames_per_camera`。

## 影响面

### 代码

- `config/models.py`：删 `LocalConfig` / `ExportProfile` / `TaskPathOverrides`；`DatasetTaskConfig` 升 v2
  （`dataset_root`/`ue_project_root` 必填，`export`/`ue` 内联）；`UeProfile` 去 schema。
- `config/loader.py`：删本地配置函数与 profile 加载；`load_task_config` 直接解析内联单 config。
- `config/resolver.py`：`validate_task` / `resolve_task` 去掉 env/local/`allow_absolute_paths`，
  直接由 task 字段构造 resolved task。
- `config/paths.py`：删 env/local/绝对路径拒绝；保留 `find_repo_root` / `default_repo_root` /
  `sanitize_path` / 占位符 / `resolve_task_relative`（放行绝对路径、保留逃逸检查）。
- `config/__init__.py`：同步导出。
- `cli.py`：所有 task 子命令去掉 `--allow-absolute-paths`；`_resolve_runtime` 签名简化。
- legacy 命令（`export`/`validate`/`annotate-*`/`cryptomatte-to-mask`/`build-manifest`）不改。

### 配置与仓库

- 新增 `configs/smoke_3frames_1cam.json`、`configs/production_300frames_4cam.json`
  （完全内联，含真实路径）。
- 删除 `.futsalmot.local.*`、`configs/export/`、`configs/ue/`、`configs/tasks/`、`configs/.gitkeep`。
- `.gitignore` 移除本地配置相关规则。

### 测试

- `test_task_config.py`：删 `TestLocalConfig`；`_write_task` 改内联；`test_missing_profile_file` 改为
  「缺 dataset_root / export 块」类校验失败。
- `test_task_resolver.py`：`_make_task_dir` 内联、去 `_env`；删绝对路径拒绝/放行、逃逸、profile 引用测试；
  保留 sanitize / resolved-task 文件 / `resolve_task_relative` 测试。
- `test_task_cli.py`：`_make_task_dir` 内联、去 `_set_env`；`test_validate_fail_bad_profile` 改为内联配置失败用例。
- `test_ue_resolved_task.py`：`_make_resolved` 内联、去 env。
- `test_repository_hygiene.py`：删「配置不得含盘符」「gitignore 含 local」断言；新增
  `configs/*.json` 自包含（无 `export_profile`/`ue_profile` 引用、含 `dataset_root` 绝对路径）。

### 文档

- `README.md`：改为单 config 工作流；删除 `.futsalmot.local.*` / profile 树 / 复制流程说明。
- `CLAUDE.md`：常用命令与架构同步更新。
- `ue/README.md`：示例命令改 `configs/`。
- `docs/REPRODUCIBILITY_AND_MANIFEST.md`：如有本地配置引用则更新。

## 验证

- `uv run pytest tests/` 全绿。
- `uv run grf-ue task validate configs/smoke_3frames_1cam.json`
- `uv run grf-ue task resolve configs/smoke_3frames_1cam.json`
- `git status --short` / `git diff --stat` / `git diff --check`

## 不在范围

- resolved task schema 本身、`ue/run_task.py`、UE 导入/渲染/标注逻辑不变。
- legacy CLI 命令（`grf-ue export --config ...` 等）保留，仅不再有 `configs/export/` 示例。
- provenance 占位符机制保留。
