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

代码会加载 `futsalmot_dataset_task` v2，检查 Pydantic 字段、后处理格式、Pose 覆盖字段键名、相机数量、按 `num_steps * max(1, target_fps/10)` 计算的预期帧数，以及在配置了 `game_duration` 时的源时长条件。路径解析支持 `.futsalmot/local.json`、环境变量和 CLI override；`task validate` 会确认 `ue_project_root` 存在且包含 `.uproject`，但允许 `dataset_root` 在 export 前不存在。`task resolve` 不创建目录或执行强环境检查，旧 task 绝对路径继续兼容并发出 warning。

`task resolve` 的输出 schema 为 `futsalmot_resolved_task` v1，并将 episode 输出路径限制在 `dataset_root` 下。它是 P1/P2 共享的运行时路径契约，不应手工编辑。

`configs/player_appearance_smoke_10frames_1cam.json` 是 seeded appearance 的 10 帧、单相机 smoke task：使用完整 7/9/5 catalog、`SkinTexture`、现有 `Player_L0..R4` actor mapping、`LS_Cam_01` / `CineCam_01` Sequence-camera 配对，以及该 smoke 已有 RGB/Object-ID mask 设置。Pose 和 debug/video postprocess 均关闭；这个 smoke 不依赖 Runtime Pose Recorder 或 pose-finalize。task 路径继续通过 `${FUTSALMOT_DATASET_ROOT}`、`${FUTSALMOT_UE_ROOT}` 与被忽略的 `.futsalmot/local.json` 按现有优先级解析。配置校验和解析不等于 UE 资产或视觉验收。

### Episode Appearance P1/P2

启用 `ue.appearance` 时，P1 `task export` 在轨迹和 meta 成功写入后生成 `provenance/appearance.json`，并立即重验 manifest、catalog 与当前 `frames.jsonl`。重复输入产生确定性分配；seeded appearance 与非空 `ue.player_visuals` 同时配置会被拒绝。旧 task 没有 `ue.appearance` 时不要求 manifest/application report；Audit 仅以当前 resolved catalog 判断是否要求 appearance，不根据 episode 中残留的 appearance 文件推断启用状态。Cleanup 对所有 task 都会检查实际存在的恢复证据；legacy task 没有恢复文件时按原有流程处理，但若发现损坏或未终结证据仍 fail-closed。

P2 必须在 Unreal Editor 真实 Python 环境运行 `ue/run_task.py`。外观 apply/restore 之前的只读操作门禁是 `ue/diagnose_episode_appearance.py`：当前 Level 必须精确为 `/Game/FutsalMOT/Maps/L_FutsalCourt`；十个映射 entity 各自必须唯一对应 Actor，Actor path 和 External Actor package 不得重复或 dirty；关键 Mesh、Texture2D、Material、Skeleton、AnimBlueprint 等资产必须存在且 class 匹配，查询过程不能有错误。只有诊断报告 `task7_preflight_can_start=true` 才可继续 apply/restore、Task 7 mutation 或 MRQ。诊断只检查当前 Editor 状态，不负责切图，也不以文件路径存在替代 Unreal AssetData class 检查。`run_task.py` 及 appearance adapter 还会执行自己的 manifest、资产、材质 slot、地图、十人 roster 和 dirty-package preflight；任一失败必须在 Actor mutation 和 MRQ 提交前停止。

如需切换地图，只能调用 Unreal MCP 原生 `SceneTools.load_level`。调用返回后另发只读查询核验当前 Level 和十名 Actor，再运行 appearance 诊断。诊断或 apply/restore 过程中不得保留从 `unreal.load_asset()` / `unreal.load_object()` 获取的 `UWorld`，也不得用 `EditorLoadingAndSavingUtils.load_map()` 作为 Python 回退；原生 load-level 能力不可用时停止并报告 `MANUAL_REQUIRED`。

apply 在变更前捕获十名 Actor 的视觉组件/原网格、动画类、attachment/relative transform、完整 material override 数组、可见性与 package dirty 基线，并捕获参与事务的资产 package 基线。若任一相关 package 预先 dirty，必须在 mutation 前 fail-closed。恢复快照写到被忽略的本机 `.futsalmot/runtime/<task_id>/appearance-rollback.json`；它不是 episode artifact，绝不进入 portable provenance。appearance 只临时修改视觉组件，不保存 randomized External Actor 或组合材质；恢复时逐项回读状态，并按验证过的方式恢复 package clean 基线。不能证明恢复成功时保留快照并以失败结束。

Episode `provenance/appearance_application.json` 是便携报告，schema 为 `futsalmot_appearance_application` v1。它关联 `manifest_fingerprint`、`catalog_fingerprint`、`trajectory_sha256`、transaction ID、十名 actor 的映射/应用资产与 slot、错误/警告、每个 MRQ job 的 camera 和终态、`mrq_pending`、终态 `render_result` 及恢复状态；不含本机文件路径。状态包括 `not_started`、`preflight_failed`、`applied`、`rendering`、`restored`、`restore_failed`。Audit 对启用 appearance 的 episode 要求 manifest 与当前输入一致、报告指纹和轨迹摘要一致、所有十名 Actor `restore_status=restored`、报告整体 `state=restored`；要求 render 时还要求每个 camera/job 到达终态并与 `render_summary.json` 对应。

Sequence/annotations 等同步工作在完成或异常后立即恢复 appearance。full/render 提交的 MRQ 是异步的：事务在所有 camera job 成功、失败或取消等终态前保持已应用状态，UE Editor 必须继续 tick；MRQ submit 异常立即尝试恢复，异步完成回调拥有恢复责任，在全部 job settle 后恢复 Actor 并发布终态报告。MRQ 提交前先持久化本机 rollback 快照和 application report；进入 rendering 后同时更新本机报告与 episode portable report。不能在 Editor 主线程阻塞等待，也不能因一个 camera 先完成就提前恢复。若 Editor 进程中断，重新打开已保存地图应回到未随机化的基线；下次运行必须先审查既有恢复证据并核对 Actor 状态，禁止覆盖未解决的快照或 report。

Cleanup 对 appearance 恢复证据采用 fail-closed：本机 report 或 episode report 无法读取、状态未知/非终态、存在 `applied`/`rendering`/`restore_failed` 状态，或本机 rollback snapshot 仍存在时，均拒绝 `cleanup --apply` 并保留证据；即使 portable report 写着 `restored`，只要 rollback snapshot 留存，P1 也无法验证当前 Editor 中的 baseline，因此仍阻止清理。缺少 task ID/repo root 而无法检查本机证据同样阻止已启用 appearance 的清理。不得手工删除恢复证据来绕过门禁。

本节合同测试可以证明纯 Python 的 seed/hash/schema、输入校验、report 状态机、fail-closed 分支及 orchestration 逻辑；它们不证明当前 UE 资产真实可加载、Material 实际呈现指定皮肤、slot/retarget 外观正确、External Actor package clean/已恢复、MRQ callback/watchdog 在真实 Editor 中生效，或最终 RGB/Mask/Pose 视觉对齐。自动化测试不能代替短 episode 的 UE apply/restore 和实际 MRQ 渲染检查。

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

`validate-annotations` 的 standalone 默认仍要求 MOT，保持旧 CLI 行为。`task postprocess`、`task audit` 和 `task cleanup` 则从 resolved task 统一解析 requirement：当 `annotation_export.export_mot=false` 且 `postprocess.formats` 未选择 `mot` 时，缺失 `gt/gt.txt` 和 `seqinfo.ini` 是 `skipped`，不会阻断 JSON-only task；若 MOT 被要求，缺失或非法文件仍失败。即使 MOT 未要求，已有的 `gt/gt.txt` 仍会校验其内容。

### Pose 验证

```powershell
uv run grf-ue annotate-pose <dataset_episode_dir>
uv run grf-ue validate-pose <dataset_episode_dir>
uv run grf-ue validate-pose <dataset_episode_dir> --validation-level quick
```

Pose 验证器检查 `labels_pose/` 和 `pose_keypoints.jsonl` 帧集合、每行 41 字段、12 个关键点坐标范围、visibility 值 `0/1/2`、图片对应关系、左右肩轴与左右髋轴是否同向。`full` 重新使用当前标注路径比对标签内容；`quick` 主要比对结构和每帧行数。

正式 `run_task.py --mode pose-finalize` 的完整性门禁在 UE 侧执行：5 个 SaveGame slot 都存在、root 帧数等于期望值、10 个 actor 和每个 actor/frame 的 13 个 raw 骨骼结构齐全后才写 `pose_capture.jsonl` 和 `pose_session.json`。然后 `build_coco17.py` 生成 12 点肢体 3D/2D；P1 再生成 41 字段 YOLO Pose。

## Audit

```powershell
uv run grf-ue task audit configs/<task>.json --validation-level quick
uv run grf-ue task audit configs/<task>.json --validation-level full
```

`task audit` 由 resolved task 的 render、instance mask、MOT、YOLO 和 Runtime Pose 开关决定 required checks；未启用功能写为 `skipped`，不会因文件缺失失败。它检查：

- 相机目录数量和每相机标注帧数。
- `img1/`、`mask/`、`render/`、`render_mask/`、MOT/YOLO 文件的缺帧、重复帧和零字节文件。
- 使用 `source_step`、`source_step_seconds` 和 Sequence `playback_fps` 推导的渲染帧覆盖。
- 跨相机的 `frame_index`、时间、源步、episode ID、track ID 和 mask ID 同步。
- 相机内参、外参、分辨率和相机位置重复情况。
- 被要求时的 `render_summary.json` status 和每相机状态。
- 被要求时的 Runtime Pose/raw 13 骨与 12 点最终 Pose 完整性；Pose disabled 时不根据残留 `pose_session.json` 反推 requirement。
- 可选的 `validate-annotations`。

报告的 canonical 字段是 `passed`、`exit_code`、`errors`、`warnings` 和 `checks`。errors 或 required failed check 使报告失败；warnings-only 与 skipped checks 保持通过。原有 `cameras`、`sync`、`mapping`、`calibration`、`render_summary`、`pose_coco17` 等 detail sections 及其 local `ok` 字段继续保留。报告 Markdown 和 JSON 都是生成物，不是本仓库的技术文档。

当 episode 的 `dataset_manifest.json` 标记 `cleanup_status="applied"` 且 artifact profile 为 `research_minimal` 时，`task audit` 会跳过已经被清理的 mask/render/pose 原始产物，只审计 canonical 产物。这是清理后的特殊路径，不能用来证明原始渲染文件仍然存在。

Pipeline State 是独立的执行进度记录，位置为 `.futsalmot/runtime/<task_id>/pipeline_state.json`。`task status` 展示固定的 `export`、`ue_sequence`、`render`、`postprocess`、`audit`、`cleanup` 六步；`task resume` 对 completed 跳过、failed 重试、pending 执行、skipped 保持，`RUNNING` 一律按步骤证据处理为 unknown。Pipeline State 的 completed 不表示数据通过验证；数据正确性仍由 Audit report 的 `ValidationResult.passed` 决定。`ue_sequence` 的 completed 只表示 command generation 完成。

Run Manifest 位于 `.futsalmot/runtime/<task_id>/run_manifest.json`，只汇总一次运行的来源、时间、Pipeline State 和 Audit validation，以及尽力统计的 artifact 数量。`task_hash` 只基于原始 Task Spec JSON 内容；local config、resolved path 和 machine path 不参与 hash。Python/UE commit 无法获取时写 `null`，artifact 统计失败只记录 error，不阻断已有 workflow。Cleanup 在删除 transient 前写入摘要，避免删除后丢失计数。

Audit report 的可选 `metrics` 是 best-effort 规模统计，不是新的质量报告系统。它统计数据集帧/图片/相机数量、MOT track/frame 摘要和 mask 非零 instance ID 数量；读取或统计失败只写入 `metrics.errors`，不改变 Audit checks、`ValidationResult`、Pipeline State 或退出码。`task status` 只读取并展示已有 metrics，不重新计算。

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

默认是 dry-run，绝不删除文件。`--apply` 只有在 `_validation_gate` 通过时才会删除既有临时路径，并在 gate blocked 时以非零退出。gate 与 task requirement 一致：只在 task 要求 Render 时检查 `render_summary.status == "success"`，只在 task 要求 Runtime Pose 时检查 `pose_session.capture_complete == true`。Pose disabled task 缺少 `pose_session.json` 是 skipped，不会阻止 cleanup。

cleanup 必须先读取 canonical audit JSON：缺少 audit 报告直接 fail safe 阻止 `--apply`。canonical `passed/exit_code/errors/warnings/checks` 中，`passed=false`、errors、nonzero exit 或 required failed check 一定阻止 `--apply`；warnings-only 不阻止。仅有 `{"passed": true}` 的最小成功报告可兼容读取；其它 canonical 字段缺失、格式混合或无法解析时 fail safe 拒绝 cleanup。没有 canonical 字段时才只读兼容旧 `ok/failed_checks`，避免错误删除。

清理集合当前包括相机下的 `render/`、`render_mask/`、`debug/`、`mask/*.png`，episode 根的 `pose_capture.jsonl`，以及 `yolo_pose/images/`、`yolo_det/images/`、`yolo_seg/images/` 下的 PNG。canonical 的 `img1/`、相机标定、annotations、MOT、12 点 Pose、YOLO labels、provenance、audit 和 manifest 不在删除集合中。

虽然 task 模型提供 `artifact_policy.profile`，当前 `collect_transient()` 对不同 profile 没有实现不同删除分支；文档不能把 `full` profile 描述为已经保留全部 transient 原始文件。

## 已知实现限制

### Camera Coverage 语义

相机 profile 的 `coverage` 字段描述 camera distribution strategy 的覆盖目标，而不是单个相机 footprint 的独立能力：

- `full_field` 的判断对象是选定 camera set 的联合覆盖范围。当前 `anchor_only` 和 `anchor_plus_one_partial` 的 canonical Anchor set 均为 `C1..C5`。
- `C1..C5` 中每个相机可以只覆盖球场的一部分；只要选定 Anchor set 的 footprint 合并后覆盖完整场地，就符合 set-level `full_field` 语义。
- `partial_field` 表示局部补充相机。当前 `P01` 是 `partial_field`，区域意图为 `left_half`。
- 单个相机的二维理论 footprint 不能证明像素覆盖、遮挡、球员可见性或 MOT 质量；set-level 几何覆盖也不替代这些验证。

### 配置与时长

- 配置中的 `game_duration` 在 `ExportConfig`/`grf_runner.py` 中表示单个 GRF 回合的引擎帧数；`resolver.validate_task()` 的源时长检查却把它按秒比较。这是当前单位不一致，不能把 task validate 的时长结果当作完整的 GRF 回合覆盖证明。
- `target_fps=30` 时输出帧数是 `num_steps * 3`。若 `audit.expected_frames_per_camera` 仍按原始 `num_steps` 或其他旧值填写，`task validate` 会拒绝该 task。本次在当前仓库逐个执行 `task validate`（排除占位模板 `example.json`）时，被拒绝的文件为 `c5_smoke_1280.json`、`c5_smoke_1920.json`、`c6p0_visual_1p0.json`、`c6p0_visual_1p5.json`、`c6p0_visual_1p75.json` 和 `c6p0_visual_2p0.json`；其余当前配置通过该静态校验。不要把所有 `configs/*.json` 统称为当前机器上可直接执行的有效配置。
- `ue_project_root` 和 `dataset_root` 由 task 保存为机器绝对路径；resolver 对 UE 项目路径主要做字符串和 resolved task 结构处理，实际目录、`.uproject`、地图和资产仍需单独检查。

### Pose 覆盖字段

`postprocess.yolo_pose` 的 `bone_overrides` 只接受 12 个正式肢体点名称。正式 `pose-finalize -> build_coco17.py` 保留 13 骨 raw capture 的完整性门禁，但最终 3D/2D arrays、`pose_keypoints.jsonl`、YOLO 标签和 overlay 都只允许 12 个肢体点；不再支持 `head_offsets_cm` 或脸部估计点。

### Legacy 脚本和资产路径

正式入口的当前资产路径在 `/Game/FutsalMOT/Blueprints/Pose/` 下，但部分旧脚本仍引用根目录旧路径，例如旧的 C4 BurnIn/Recorder 构建脚本、C5 smoke/MRQ 脚本和 `ue/archive_c4_diag/`。这些文件仍在代码仓库中，但不能默认它们能重建当前已迁移的资产。`ue/import_grf_episode.py` 仍保留旧式 `preview`、`sequence`、`both`、`annotations` 和 `full` 兼容入口；正式 task 流程应使用 `ue/run_task.py --resolved-task`。

### UE 运行时和视觉验收

- MRQ 是异步的；脚本提交后立即返回，Editor 必须继续运行和 tick。Object-ID 正式路径将 RGB、Mask 和 Runtime Pose 放在同一个 MRQ job 中求值，降低跨 job 时刻差异；这不能单独证明每个 AnimGraph/Retarget 姿态几何正确。
- Camera Cut 的自动设置使用 UE API fallback，API 版本差异可能导致设置失败；没有有效 Camera Cut 时，MRQ 可能没有期望的相机视角。
- 代码会在 MRQ 前尝试把 Actor 烘焙到第 0 帧，以避免 possessable 接管前使用关卡默认位置，但这不是对当前关卡状态的静态证明。
- `camera.json` 的数值来自当前 CineCamera；代码不能仅凭配置证明焦距、filmback、位置、朝向或分辨率与实际 MRQ 完全一致。渲染阶段会拒绝已发现的 RGB 与标定分辨率不一致，但仍需要视觉 overlay 检查。
- `coco17_*.jsonl` 是历史兼容文件名，不能据文件名推断当前仍为 17 点；最终数组长度和 `keypoint_names` 才是有效契约。
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
