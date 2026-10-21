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


### Episode Environment v1.5 只读诊断

`ue/diagnose_episode_environment.py` 的报告 schema 为 `futsalmot_episode_environment_diagnostic` v1，包含 `current_level`、`environment_policy`、六类环境 Actor 的 `actors` 记录、`package_dirty`、`errors` 和 `preflight_can_start`。正式地图必须精确为 `/Game/FutsalMOT/Maps/L_FutsalCourt`，policy 必须为 `futsalmot_environment_v1_5`；`DirectionalLight`、`SkyLight`、`SkyAtmosphere`、`VolumetricCloud`、`ExponentialHeightFog` 和 `PostProcessVolume` 必须按 label/class 唯一匹配。每个 Actor 记录 exact path、class、package path、match count、discovered properties、baseline readback 和 evidence source。

UE property evidence 只接受当前 Editor 中真实 `get_editor_property`、`get_scalar_parameter_value` 或 `get_actor_rotation` 结果。每个目标字段至少包含 `readback` 和非占位 `evidence_source`；缺失、未知、dirty package、错误地图、policy 不匹配、重复或错误 class 均使 `preflight_can_start=false`。单个已确认属性最多读取两次；两次失败时记录 `MANUAL_REQUIRED` 并停止该 Actor 的 discovery。`VolumetricCloud.cloud_coverage` / `cloud_density` 读取组件的 `material` 后调用 `get_scalar_parameter_value("Cloud_GlobalCoverage"/"Cloud_GlobalDensity")`；`PostProcessVolume` 的 settings 字段同时记录对应 `override_*` 开关 baseline，供 restore 写回。未通过 discovery 的 UE 属性名称不得猜测，也不得据此开始 Task 4 apply/restore。

Task 3 诊断只读当前 Editor，不创建 Actor、不进入 PIE、不写入属性、不保存地图或资产，也不使用文件系统 `.uasset` 编辑。没有 resolved v1.5 environment profile 时报告必须显式 `skipped/inactive`；磁盘中的 environment manifest 不能单独激活诊断。

Task 4/5 的纯 Python contract tests 已覆盖 snapshot/report identity、manifest/trajectory drift、逐字段 readback、environment-before-appearance ordering、MRQ active gate、portable summary metadata、terminal callback ownership 和 restore isolation。环境 active 时，MRQ gate 必须在 `_mrq_subsystem()`、queue 和 job 创建之前通过；render summary 的 environment metadata 不复制完整参数 manifest。Task 6 的 audit/cleanup 只由 resolved environment policy 激活，残留文件不能改变 legacy task 的 requirement。

P2 environment transport（`UEEnvironmentTransport`）只用已验证 API：`set_actor_rotation`、`get/set_editor_property`、`MaterialLibrary.create_dynamic_material_instance` + MID `set_scalar_parameter_value`、`SkyLightComponent.recapture_sky`、`EditorAssetSubsystem.set_dirty_flag`。UE 的 `Color` 位置参数顺序是 `(b, g, r, a)`，必须用关键字参数写入；manifest 的 0–1 float 与 8bit FColor 回读按半档 `1/255` 容差比较。VolumetricCloud 只临时挂接 transient MID，restore 通过 MID 的 `parent` 或进程内记录恢复原材质。apply 失败或 restore 结束后清除本次事务触碰的 External Actor package dirty flag，使 package clean baseline 可验证。

早期 adapter 的 Editor evidence（2026-10-21，UE 5.8；不代表本文末尾的新 catalog 验收）：`preflight_can_start=true`；`package_dirty=[]`；`configs/player_environment_smoke_3frames_4cam.json` 经 P1 export 生成 `G:/FutsalMOT_Dataset/episode_player_environment_smoke_3f`。直接调用旧 adapter 的端到端结果为：环境 Actor apply/readback 通过；restore `ok=true`、`state=restored`；六类环境 Actor baseline 与 apply 前一致；cloud material class 回到 `MaterialInstanceConstant`；dirty map/content package 均为空。该证据不覆盖本次新增的十灯控制路径。

`ue/run_task.py --mode full` smoke（同一 episode）结果：environment apply 在 sequence/annotations/MRQ 之前完成；MRQ 4 个 camera 全部 `success`，RGB `12` 帧、mask `12` 帧，`render_summary.json` 含 environment policy/fingerprint/preset 且 `environment_restore_ready=true`；MRQ 终态后自动 restore，`environment_application.json` 为 `state=restored`、`restore_status=restored`、无错误；本机 rollback/report 在公共 report 写入后自动删除。P1 `task postprocess`、`task audit --validation-level quick` 通过，audit 的 `environment` gate 为 `ok=true/required=true/enabled=true`；`task cleanup` dry-run 为 `gate_ok=true`、`gate_problems=[]`、计划删除 28 个中间文件，未执行 `--apply`。

该 smoke 暴露并已修复一个真实缺陷：MRQ setup 中的 `_apply_cv_gt_camera_post_process` 和 `_bake_first_frame_for_render` 会在 environment 事务期间调用 `save_current_level`，把随机化环境写进 External Actor package（实测 DirectionalLight `.uasset` 出现 preset 值 `8.267379`）。现在这两个路径与 stencil 保存都用 `_should_save_level(appearance_active, environment_active)` 门禁，environment active 时只保留内存状态。修复前的磁盘污染已通过“恢复 baseline 后显式保存一次”修复，并验证 DirectionalLight package 重新只含基线 `3.0`、不再含 preset 值。`render_result` 写入 environment report 已实现并有单元测试；早期 report 由修复前代码生成，该字段为 `null`；本轮新 episode 使用修复后的代码，不能用早期 report 代替新证据。



### Weather preset 可视化与已修复的 preset 数据缺陷

做 preset 对比时发现两个 preset 数据缺陷，已修复并重新 export/resolve/render：

1. `directional_light.rotation_deg` 的 pitch 写成正值（如 `+32`），UE 中光向上照射，MRQ RGB 几乎全黑。现在所有 preset 使用 `pitch <= 0`，并有 `test_every_preset_directional_light_points_downward` 回归测试。修复后 4 机位 `img1/000001.png` 平均 RGB 约 `[47.9, 71.3, 27.0]`，亮像素占比 > 0.97。
2. `post_process.white_balance_tint` 使用 `2.0` / `-2.0`，超出 UE `WhiteTint` 的 `[-1, 1]` 范围，会产生极端色偏。现修正为 `0.15` / `-0.15`，并有 `test_preset_white_balance_stays_inside_ue_ranges` 回归测试。预览时每个 preset 应用前必须先把所有受管字段写回 baseline，否则会继承上一个 preset 的未指定字段。

已验证的限制：在当前 Editor 会话中，`AutomationLibrary.take_high_res_screenshot` 在 MRQ 之后会延迟数分钟才执行，甚至写到错误目录，不能作为可靠的颜色验收手段。需要看不同天气时应修改 `ue.environment.preset`，重新 export/resolve 后跑 `run_task.py --mode full`，以 `img1/` 的正式 MRQ 输出为准。

3. `exponential_height_fog.height_falloff` 与 `fog_height_falloff` 映射到同一个 UE 属性 `fog_height_falloff`。P1 之前允许两者取不同随机值（如 light_haze 的 `0.11` / `0.12`），P2 的 duplicate-target 检查会拒绝 apply。现在 `fog_height_falloff` 固定为 `height_falloff` 的别名，随机化后仍保持同值，并有 `test_fog_height_falloff_is_an_alias_of_height_falloff` 回归测试。

原始环境版本曾将六个 preset 的 4 机位首帧渲染到 `G:/FutsalMOT_Dataset/env_presets/<preset>/CineCam_0X_f000001.png`（24 张）；这些图片不包含 matte 草地材质和当前场馆灯光设置，不能作为新版本 evening preset 的验收结果。当前工作树将补光切换到用户手动放置的 10 个 Spot Light：初始六档采用白天 5 EV、evening 10 EV；当前八档已按厚云亮度重新校准，详见末尾表格；仅修改手动灯的强度和单位。这是新十灯加入时的阶段记录；当前 catalog 校准和实际 MRQ 证据见本文末尾，不能继续据此判断当前路径未验证。

本次已完成的真实 UE 资产修改是 `/Game/FutsalMOT/Materials/MI_Futsal_Field_Grass_Matte_v1`：它继承 `M_Futsal_Field_Grass`，把 `Grass_Specular/Param` 设为 `0.08`，并已保存到 `SM_FutsalCourt_Floor` 的 External Actor。最终 canonical `daylight_clear_v1` 已使用该 matte 材质重新渲染，4 个 camera 各有 3 个 RGB 和 3 个 mask 帧，postprocess、quick audit 和 environment restore 通过。

transient `RectLight` 的既有 UE 5.8 限制仍作为历史证据保留。新十灯路径的只读诊断要求十个预期 label 各自唯一、Actor class 和 component class 正确、EV 单位一致、当前强度一致；apply/restore 只写 intensity/unit，并跟踪其 External Actor package。这是新路径加入时的验收门槛；本轮仍按 fresh diagnostic、apply/readback/restore、实际 MRQ 和 full audit 分别记录证据。

### 当前八档天气验收（2026-10-21，catalog revision 2）

本节覆盖前文早期 adapter/十灯初始接入记录。最终 catalog fingerprint 为 `48a095a31efd17443a82e255cc2175330d7df3321c09b1880821c2f8516285c2`。

八档均在正式地图、matte 草地、seed 42、4 个 `CineCam_01..04`、每机位 3 帧、1920×1080、10 FPS 和 RGB/Object-ID 同 job 路径下完成。每档 P1 validate/export、UE 只读 environment/appearance 双诊断、环境 apply/readback、异步 MRQ、自动 restore、Cryptomatte 解码、annotation validator 和 full audit 均通过；共 96 RGB PNG、96 mask PNG。未启用 appearance/号码或 Pose，因此不作为这些功能的联合验收。

| preset | 十灯 EV | 四机位场地平均亮度范围 | 球员暗像素最大占比 |
| --- | ---: | ---: | ---: |
| `daylight_clear_v1` | 5 | 49.9–83.7 | 9.46% |
| `overcast_soft_v1` | 8 | 38.3–47.5 | 4.85% |
| `warm_evening_v1` | 10 | 64.9–65.2 | 0.90% |
| `cool_evening_v1` | 10 | 65.8–66.1 | 0.62% |
| `light_haze_v1` | 5 | 41.6–73.5 | 12.76% |
| `cloudy_variation_v1` | 8.5 | 51.6–55.6 | 3.61% |
| `dawn_cool_v1` | 10 | 67.3–67.6 | 0.46% |
| `night_floodlit_v1` | 10 | 66.4–66.8 | 0.83% |

亮度是每个 camera 的 annotation frame 1（source step 0）统计：用相机标定将 40×20m 场地投影为 ROI，RGB 使用 `0.2126R+0.7152G+0.0722B`（0–255）；球员区域由 mask ID 1..10 决定，暗像素为亮度 <8。八档共 32 张首帧的球员区域均未出现 RGB 三通道同时 ≥250 的截白像素。首帧亮度统计只用于校准，不等于整段轨迹的视觉质量证明；所有 3 帧的结构和 mask 数据由 full audit 检查。

最终只读诊断：正式 Level、PIE/MRQ inactive、dirty packages 为空；六类环境 Actor 的 baseline readback 与开始前一致；十灯强度恢复为 5 EV，单位、位置、旋转、scale、颜色、半径、内外锥角及阴影开关均与开始前一致；本轮所有 `weather_*` runtime rollback 均已按正式恢复流程退役。逐 Actor application report 的 restore 状态与整体 restored 状态一致。UE 诊断直接读取 P1 manifest 参数，不再依赖 P1 包的 import path 或旧 Editor 会话缓存。

天气独立提交版本的默认单元测试：`1463 passed, 7 deselected`；另有 106 条既有路径兼容和 API warning。包含未提交号码开发内容的原工作树曾通过 1689 项测试，该数字不代表本次提交范围。原六个 portable-config 失败来自单元测试读取开发机 local config，已通过临时仓库/环境隔离修复，生产路径解析优先级未改变。新增 fresh-UE 回归测试先复现缺少 P1 包的失败，再验证磁盘契约修复。

最终数据位于 `G:/FutsalMOT_Dataset/weather_validation_20261021_r3/`；可计入验收的八个 episode 由 `Saved/EnvironmentPreviews/weather_20261021_r3/index.json` 明确列出，其中 cool evening 使用 `_r3_retry2`。同目录的 `verification_summary.json`、`*_quality.json`、`final_baseline.json`、`weather_overview.png` 和 `weather_all_cameras.png` 是本机生成证据，不入库。测试 Sequence 保存在 `/Game/FutsalMOT/Tests/WeatherValidation/<preset>/LS_Cam_01..04`，未覆盖正式 Sequence。

连续大量 MRQ 的已知风险：本轮旧 Editor 会话在冷暮光重渲染时发生 D3D12 `Out of video memory`；另有一次 P1 内存分配失败和一张不可解码 EXR。失败尝试保留，未计入上述八档成功结果。Editor 重启后先用双诊断检查基线，再通过正式 environment transaction restore 处理中断快照，发布失败渲染/已恢复报告后才继续。后续验收改为 MRQ 结束再以 1 worker 串行后处理，重渲染通过。`render_summary.status=success` 不能替代 EXR 解码和 audit；本轮未证明连续大批量渲染的显存稳定性。不要在既有恢复证据未解决时重跑或删除快照绕过门禁。

可用入口为 `configs/weather_seeded_smoke_3frames_4cam.json`；auto seed 42 解析为 `cloudy_variation_v1`，seed 47 解析为 `night_floodlit_v1`。复现需要相同 catalog fingerprint、资产与地图基线及渲染设置；UE/GPU 的时序与浮点差异不保证 PNG 字节一致。新增 revision 后旧 manifest 会按 catalog drift 拒绝，不得手改指纹。900 帧生产批次仍需单独确认 seed/preset 清单，并先做多 seed 和长时显存稳定性验收，尚未启动。


2026-10-21 提交前另以隔离后的天气代码完成四机位 × 三帧真实 UE MRQ 回归，12 张 RGB、12 张 mask 全部解码，full audit 通过，环境与十名球员基线完整恢复。该测试临时给十名球员分别绑定 Set_06–15 的新服装，用于资产渲染验收；没有修改正式 appearance v1 的 Set_01–05 候选契约，也不代表号码功能已实现。

号码功能仍为后续待办：背部号码贴合与骨骼跟随、遮挡可见性、号码 mask 与 RGB 同步、实例唯一性及 apply/restore 生命周期尚未验收。本次不提交号码代码、资产或 v2 开关；正式默认配置继续使用 v1。
