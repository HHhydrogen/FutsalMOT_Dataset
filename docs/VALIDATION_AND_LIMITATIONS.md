# 验证、清理和当前限制

本文档记录当前代码真正执行的验证方式、产物门禁和不能被静态文件证明的事项。它不是对历史设计目标的承诺；如果实现和目标不一致，以当前代码和测试结果为准。

## P1 测试

`pyproject.toml` 的 pytest 配置为：

```text
默认 addopts: -m not grf_integration --ignore=ue/mcp_smoke_test.py
```

因此普通测试命令不运行需要完整 GRF 引擎的集成测试：

```powershell
uv run pytest
uv run pytest -m grf_integration -q
```

测试覆盖的主要范围包括 task schema/resolver/CLI、episode schema 和 validator、seed 派生、坐标变换、插值、相机投影、MOT/YOLO 序列化、Cryptomatte 和 Mask、标注验证、Pose 生成和验证、MRQ 帧映射、manifest、Windows multiprocessing、资源监控和性能工具。这里的测试主要验证 Python 逻辑；它们不能替代 UE Editor 中的 Blueprint、地图、相机、MRQ 和视觉验收。

## 分阶段验证

### Task 验证

```powershell
uv run grf-ue task validate configs/<task>.json
```

代码会加载 `futsalmot_dataset_task` v2，检查 Pydantic 字段、后处理格式、Pose 覆盖字段键名、相机数量、按 `num_steps * max(1, target_fps/10)` 计算的预期帧数，以及在配置了 `game_duration` 时的源时长条件。它只检查 `dataset_root` 和 `ue_project_root` 非空，不会确认路径一定存在，也不会确认 UE 项目真的包含 `.uproject`、地图或 Actor。

`task resolve` 的输出 schema 为 `futsalmot_resolved_task` v1，并将 episode 输出路径限制在 `dataset_root` 下。它是 P1/P2 共享的运行时路径契约，不应手工编辑。

### Episode 验证

```powershell
uv run grf-ue validate <episode_dir>
```

该验证器检查 `meta.json`、`frames.jsonl`、schema/version、帧行数、step 连续性、时间单调性、10 名球员和固定 ID、有限坐标、场地边界、球员 Z 为 0、持球引用以及 `meta.timing.num_steps` 一致性。若有 `meta.randomness`，还会按 `futsalmot_seed_v1` 复核派生 seed；旧 episode 缺少该字段时只标记为 legacy seed metadata，不会伪装成已验证可复现。

### 相机标注验证

```powershell
uv run grf-ue validate-annotations <dataset_episode_dir>
uv run grf-ue validate-annotations <dataset_episode_dir> --validation-level quick
```

`full` 会逐帧重新读取 mask 并重新派生相关产物；`quick` 做结构、帧集合、分辨率、合法 ID、bbox 范围、track 映射、MOT/YOLO 语法和每相机有限抽样的 mask 重算。该验证器递归寻找包含 `camera.json` 的目录，因此支持 `<root>/<camera>/` 和 `<root>/<episode>/<camera>/` 两种层级。

如果 `mask/` 存在，验证器强制执行以下语义：

- `bbox_source="instance_mask"` 必须是 `in_frame=true` 且 `visible_pixel_count>0`。
- `bbox_source="not_visible"` 必须是 `in_frame=false`、可见 bbox 为 `null`、`visible_pixel_count=0`，几何 bbox 只能保留在 `geometry_bbox_*`。
- `bbox_source="instance_mask"` 的 bbox 必须与 mask min/max 一致。
- `img1/` 和 `mask/` 的帧号集合必须一致。
- mask 值只能是背景 `0` 或实体 `1..11`。
- 不可见对象不能泄漏到 MOT、YOLO Detect 或 YOLO Segment。

如果 `gt/gt.txt` 不存在，当前 `annotation_validator.py` 会报告错误，错误信息是“如需 MOT 导出”。这与 `annotation_export.export_mot=false`、`postprocess.formats=["json"]` 的配置能力存在实际冲突：关闭 MOT 的 smoke task 仍可能在默认 `task postprocess` 验证阶段失败。不要把 `export_mot=false` 描述为已经被完整 postprocess 验证器支持。

### Pose 验证

```powershell
uv run grf-ue annotate-pose <dataset_episode_dir>
uv run grf-ue validate-pose <dataset_episode_dir>
uv run grf-ue validate-pose <dataset_episode_dir> --validation-level quick
```

Pose 验证器检查 `labels_pose/` 和 `pose_keypoints.jsonl` 帧集合、每行 56 字段、bbox 和关键点坐标范围、visibility 值 `0/1/2`、图片对应关系、左右肩轴与左右髋轴是否同向。`full` 重新使用当前标注路径比对标签内容；`quick` 主要比对结构和每帧行数。

正式 `run_task.py --mode pose-finalize` 的完整性门禁在 UE 侧执行：5 个 SaveGame slot 都存在、root 帧数等于期望值、10 个 actor 和 13 个骨骼结构齐全后才写 `pose_capture.jsonl` 和 `pose_session.json`。然后 `build_coco17.py` 生成 COCO17 3D/2D；P1 再生成 YOLO Pose。

## Audit

```powershell
uv run grf-ue task audit configs/<task>.json --validation-level quick
uv run grf-ue task audit configs/<task>.json --validation-level full
```

`task audit` 检查：

- 相机目录数量和每相机标注帧数。
- `img1/`、`mask/`、`render/`、`render_mask/`、MOT/YOLO 文件的缺帧、重复帧和零字节文件。
- 使用 `source_step`、`source_step_seconds` 和 Sequence `playback_fps` 推导的渲染帧覆盖。
- 跨相机的 `frame_index`、时间、源步、episode ID、track ID 和 mask ID 同步。
- 相机内参、外参、分辨率和相机位置重复情况。
- `render_summary.json` 的 status 和每相机状态。
- 若存在且未跳过，Runtime Pose/COCO17 完整性。
- 可选的 `validate-annotations`。

报告字段由 `src/grf_ue_bridge/workflows/task_audit.py` 实际写出：`passed`、`exit_code`、`errors`、`warnings`，以及 `cameras`、`sync`、`mapping`、`calibration`、`render_summary`、`pose_coco17` 等检查结果。报告 Markdown 和 JSON 都是生成物，不是本仓库的技术文档。

当 episode 的 `dataset_manifest.json` 标记 `cleanup_status="applied"` 且 artifact profile 为 `research_minimal` 时，`task audit` 会跳过已经被清理的 mask/render/pose 原始产物，只审计 canonical 产物。这是清理后的特殊路径，不能用来证明原始渲染文件仍然存在。

## Manifest 和重复检测

```powershell
uv run grf-ue build-manifest <dataset_root>
uv run grf-ue verify-manifest <dataset_root>
uv run grf-ue task manifest configs/<task>.json
```

顶层 `build-manifest` 使用 `dataset_manifest.py`：

- 以 POSIX 相对路径记录文件，不把绝对路径放进 checksum 条目。
- `metadata` profile 包含 episode/camera 元数据、MOT 和 provenance；`final` 额外包括 `img1`、`mask`、YOLO 和 Pose 标签；`all` 再包括 `render/` 和 `render_mask/`。
- 对文件流式计算 SHA-256，写 `checksums/<episode_id>.jsonl`。
- 记录 `frames.jsonl` 原始 hash 和 canonical hash。
- 检测重复 `(root_seed, scenario, export_config_hash)`、重复轨迹和跨 seed 相同轨迹。
- fingerprint 不包含生成时间、绝对路径或机器信息。

`verify-manifest` 返回 `0` 表示匹配，`1` 表示文件缺失/大小或 hash 不匹配，`2` 表示 manifest、schema 或参数错误。默认不会因被忽略的 debug、audit、render、render_mask、视频、临时文件和 `yolo_pose/` staging 文件而失败；`--strict` 才会把额外文件计入失败。

task 级 `manifest` 和顶层 `build-manifest` 使用不同实现和字段，二者不能互换。

## Cleanup 的实际行为

```powershell
uv run grf-ue task cleanup configs/<task>.json
uv run grf-ue task cleanup configs/<task>.json --apply
```

默认是 dry-run。`--apply` 只有在 `_validation_gate` 通过时才会删除已知临时路径。当前代码的门禁要求：

- `render_summary.json` 存在且 `status == "success"`。
- `pose_session.json` 存在且 `capture_complete == true`。
- 如果存在 audit JSON，则检查其中的 `ok == false` 或 `failed_checks`。

这造成两个当前限制：

1. 不启用 Runtime Pose 的任务通常没有 `pose_session.json`，即使 RGB 和标注完整也会被 cleanup 拒绝。
2. `task_audit.py` 生成的是 `passed`、`exit_code`、`errors`、`warnings`，不是 cleanup 门禁主要读取的 `ok`、`failed_checks`。因此 audit 失败字段与 cleanup 的读取契约不完全一致，不能声称 audit 一定会阻止 cleanup。

清理集合当前包括相机下的 `render/`、`render_mask/`、`debug/`、`mask/*.png`，episode 根的 `pose_capture.jsonl`，以及 `yolo_pose/images/`、`yolo_det/images/`、`yolo_seg/images/` 下的 PNG。canonical 的 `img1/`、相机标定、annotations、MOT、COCO17、YOLO labels、provenance、audit 和 manifest 不在删除集合中。

虽然 task 模型提供 `artifact_policy.profile`，当前 `collect_transient()` 对不同 profile 没有实现不同删除分支；文档不能把 `full` profile 描述为已经保留全部 transient 原始文件。

## 已知实现限制

### 配置与时长

- 配置中的 `game_duration` 在 `ExportConfig`/`grf_runner.py` 中表示单个 GRF 回合的引擎帧数；`resolver.validate_task()` 的源时长检查却把它按秒比较。这是当前单位不一致，不能把 task validate 的时长结果当作完整的 GRF 回合覆盖证明。
- `target_fps=30` 时输出帧数是 `num_steps * 3`。若 `audit.expected_frames_per_camera` 仍按原始 `num_steps` 或其他旧值填写，`task validate` 会拒绝该 task。本次在当前仓库逐个执行 `task validate`（排除占位模板 `example.json`）时，被拒绝的文件为 `c5_smoke_1280.json`、`c5_smoke_1920.json`、`c6p0_visual_1p0.json`、`c6p0_visual_1p5.json`、`c6p0_visual_1p75.json` 和 `c6p0_visual_2p0.json`；其余当前配置通过该静态校验。不要把所有 `configs/*.json` 统称为当前机器上可直接执行的有效配置。
- `ue_project_root` 和 `dataset_root` 由 task 保存为机器绝对路径；resolver 对 UE 项目路径主要做字符串和 resolved task 结构处理，实际目录、`.uproject`、地图和资产仍需单独检查。
- `export_mot=false` 与当前 annotation validator 对 `gt/gt.txt` 的无条件要求存在冲突，JSON-only smoke 配置需要显式跳过验证或修复实现后才能作为完整流程使用。

### Pose 覆盖字段

`postprocess.yolo_pose` 模型接受 `bone_overrides` 和 `head_offsets_cm`，并且旧的 `ue/pose_export.py` 会读取这些覆盖。但正式 `pose-finalize -> build_coco17.py` 当前调用 `resolve_limb_bone_map(all_bones)` 和默认的 `apply_head_offsets()`，没有把 resolved task 中的两个覆盖字典传进去。因此当前文档只能说这些字段对部分旧/显式路径有效，不能声称它们一定改变正式 Runtime Pose 的 COCO17 结果。

### Legacy 脚本和资产路径

正式入口的当前资产路径在 `/Game/FutsalMOT/Blueprints/Pose/` 下，但部分旧脚本仍引用根目录旧路径，例如旧的 C4 BurnIn/Recorder 构建脚本、C5 smoke/MRQ 脚本和 `ue/archive_c4_diag/`。这些文件仍在代码仓库中，但不能默认它们能重建当前已迁移的资产。`ue/import_grf_episode.py` 仍保留旧式 `preview`、`sequence`、`both`、`annotations` 和 `full` 兼容入口；正式 task 流程应使用 `ue/run_task.py --resolved-task`。

### UE 运行时和视觉验收

- MRQ 是异步的；脚本提交后立即返回，Editor 必须继续运行和 tick。
- Camera Cut 的自动设置使用 UE API fallback，API 版本差异可能导致设置失败；没有有效 Camera Cut 时，MRQ 可能没有期望的相机视角。
- 代码会在 MRQ 前尝试把 Actor 烘焙到第 0 帧，以避免 possessable 接管前使用关卡默认位置，但这不是对当前关卡状态的静态证明。
- `camera.json` 的数值来自当前 CineCamera；代码不能仅凭配置证明焦距、filmback、位置、朝向或分辨率与实际 MRQ 完全一致。渲染阶段会拒绝已发现的 RGB 与标定分辨率不一致，但仍需要视觉 overlay 检查。
- `pose_bones.py` 中脸部点是 head 局部偏移估计，不是眼鼻耳真实骨骼；必须用 Pose overlay 或 UE 侧检查确认是否适合目标角色。
- 代码和测试不能证明地图中的 Actor 标签、Blueprint 编译、Sequence binding、MRQ Object-ID pass 输出、RGB/mask/pose 像素对齐或最终视觉质量。上述事项必须在 UE Editor 中验收并保留运行日志。

## 验收顺序

建议按以下顺序记录一次可复现运行：

1. 在 P1 运行 `task validate`，确认机器路径、相机数和帧数没有配置错误。
2. 运行 `task export` 和 `grf-ue validate <episode>`，确认轨迹契约。
3. 在 UE 中确认地图、Actor mapping、Pose tags、相机和 Sequence，然后运行 `run_task.py --mode full`。
4. 等待 `render_summary.json`，检查每相机 RGB 和 Object-ID EXR 是否齐全。
5. 若启用 Pose，运行 `run_task.py --mode pose-finalize`，确认 `pose_session.capture_complete=true`。
6. 在 P1 运行 `task postprocess`、`validate-annotations` 和 `validate-pose`。
7. 运行 `task audit`，再根据需要运行 `task manifest` 和 `verify-manifest`。
8. 只在确认门禁和保留策略符合需要后运行 `task cleanup --apply`。
