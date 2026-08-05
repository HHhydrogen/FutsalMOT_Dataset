# 配置与入口迁移表（Task Config 重构）

本页记录从「根目录多份 `ue_import_config*.json` + `scripts/`」到「task 配置 + 统一
`grf-ue` CLI」的迁移。重构基线 commit：`858a374`。

## 1. 配置迁移

| 旧文件 | 新位置 / 新用途 | 状态 |
|--------|----------------|------|
| `ue_import_config.json`（主，4 相机） | `configs/ue/4cam_1080p_cvgt.json`（episode 无关 profile） | replaced |
| `ue_import_config.smoke.json` | `configs/ue/1cam_1080p_cvgt.json` | replaced |
| `ue_import_config.smoke30.json` | `configs/ue/1cam_1080p_cvgt.json`（smoke30 与 smoke 同 profile） | replaced |
| `ue_import_config.soak.json` | `configs/ue/4cam_1080p_cvgt.json` + `configs/tasks/soak_300frames_4cam.example.json` | replaced |
| `ue_import_config.90fps.json` | `configs/ue/4cam_1080p_cvgt.json` + `configs/tasks/smoke_90frames_4cam.example.json` | renamed（纠正 90fps 命名） |
| `configs/mvp_builtin_5v5.json` | `configs/export/standard_300steps_10fps.json` | renamed |
| `configs/mvp_5v5_30fps.json` | `configs/export/standard_300steps_30fps.json` | renamed |
| `configs/mvp_5v5_90fps.json` | `configs/export/short_90frames_30fps.json` | renamed（90 帧不是 90 FPS） |
| `configs/smoke_5v5_3steps.json` | `configs/export/smoke_3steps_10fps.json` | renamed |
| `configs/demo_5v5_10steps.json` | 删除（一次性 demo） | deleted |

## 2. 入口迁移

| 旧入口 | 新入口 | 状态 |
|--------|--------|------|
| `uv run python scripts/audit_soak_episode.py` | `uv run grf-ue task audit <task>` | deprecated（薄包装保留） |
| `uv run python scripts/monitor_soak_resources.py` | `uv run grf-ue monitor <task>` | deprecated（薄包装保留） |
| `uv run python scripts/measure_run.py <cmd>` | `uv run grf-ue measure -- <cmd>` | deprecated（薄包装保留） |
| `uv run python scripts/benchmark_postprocess.py` | `uv run grf-ue benchmark ...` | deprecated（薄包装保留） |
| `py ".../ue/import_grf_episode.py" --config ...` | `uv run grf-ue task ue-command <task>` → `py ".../ue/run_task.py" --resolved-task ...` | deprecated |
| 根目录 `ue_import_config.json` 隐式读取 | 机器路径在 `.futsalmot.local.json`；任务内容在 task；两者由 resolver 合并 | removed |

## 3. 文档迁移

| 旧位置 | 新位置 | 状态 |
|--------|--------|------|
| `MASK_RENDERING_STATUS.md`（根） | `docs/architecture/INSTANCE_MASK_PIPELINE.md` | merged + deleted |
| `docs/SOAK_TEST_300STEP_4CAM.md` | `docs/validation/2026-08-05-soak-300frames-4cam.md`（加 YAML frontmatter） | moved |
| `docs/superpowers/specs/2026-08-03-multi-component-yolo-seg-design.md` | `docs/design/multi-component-yolo-seg.md` | moved |
| `docs/superpowers/specs/2026-08-03-visible-vs-geometry-gt-semantics.md` | `docs/design/visible-vs-geometry-gt-semantics.md` | moved |
| `docs/superpowers/plans/2026-08-03-multi-component-yolo-seg.md` | 删除（agent 执行计划/scratchpad） | deleted |

## 4. 删除计划

旧入口（`grf-ue export --config --output`、`grf-ue cryptomatte-to-mask <dir> --mapping ... --episode ...`、
`ue/import_grf_episode.py --config`、`scripts/*` 包装）在**下一版本**计划删除，当前保留但：
- 打印 deprecation warning；
- 不再读取根目录隐式配置；
- README 不再推荐。

## 5. 机器路径

- 用户真实路径只在 `.futsalmot.local.json`（被 gitignore）或 `FUTSALMOT_*` 环境变量；
- 仓库只提交 `.futsalmot.local.example.json` 脱敏示例；
- resolved task（含绝对路径）只写 `.futsalmot/runtime/`（被 gitignore）；
- 数据 provenance 用 `${REPO_ROOT}` / `${UE_PROJECT_ROOT}` / `${DATASET_ROOT}` 占位符，可移植。
