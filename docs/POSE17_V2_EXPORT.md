# Pose17 v2 导出准备与接入边界

此实现是可测试的 CPU 数据准备层，**不是已验收的新正式批次**。现有 8 组 v1 未采集 Runtime Pose，不能仅从现有 RGB/mask 还原真实骨骼，更不能把估计器输出冒充合成真值。旧 `coco17_3d.jsonl` 和 `coco17_2d.jsonl` 实际为 12 点，保持原读写行为。所有新产物使用独立 v2 目录。

## 可执行接口

`src/grf_ue_bridge/pose17_export.py` 提供：

- `project_landmarks(world_points, camera)`：使用已有 camera.json 的内参和 UE 世界米坐标外参投影。返回 UV、前向深度及 unknown/outside/behind_camera/missing；不由投影推断遮挡。
- `validate_capture(frame)`：校验 17 点、实体、帧身份、坐标和可见性证据。
- `validate_capture_coverage(frames, expected_cameras, expected_frame_indices)`：根据独立冻结计划检查单 seed/clip 的精确机位/帧集合与逐帧十实体；不能只比较唯一帧计数或全局骨骼集合。CLI 是草稿导出，不默认声称通过此生产完整性门禁。
- `build_coco(frames, masks)`：帧及二维单通道整数实例 mask 一一对应；生成 COCO JSON 与保留完整世界点、UV、来源和不确定状态的 sidecar。
- `validate_coco(coco)`：检查 ID 引用、RLE 面积/bbox、51 个关键点值与 num_keypoints。

P1 环境命令：

```powershell
.venv/Scripts/python.exe -m pytest tests/test_pose17_export.py -q
.venv/Scripts/python.exe -m grf_ue_bridge.pose17_export <v2目录>/capture_bundle.json <v2目录>/coco_export_a01
```

CLI 输入 JSON 为 `{"frames": [...]}`。每帧字段见单元测试 `capture()` 的完整可运行夹具。mask 文件相对输入 bundle 定位；COCO file_name 相对媒体根定位，导出不复制媒体。输出目录必须不存在，禁止写入 `v1_no_numbers_4k`。路径是 POSIX 相对路径，不允许盘符或 `..`。这是独立入口，尚未加入生产 task resolver 或旧 annotate-pose。

每帧必填 schema=`futsalmot_pose17_capture_v2`、keypoint_names、seed、clip_id、camera_id、width/height、file_name、mask_file_name、camera_calibration_ref、world_coordinates、time_seconds、frame_index、source_frame_index、sequence_frame_index、capture_id、render_sample_id、objects。

- 世界坐标声明固定 `ue_left_handed_x_forward_y_right_z_up_m`，即 UE 左手 X 前、Y 右、Z 上，**单位米**。C++ 采集 cm 在接入时显式乘 0.01 一次。
- frame_index 从当前发布片段 0 起；source_frame_index 对应完整 GRF 时间线；sequence_frame_index 对应实际 Sequence 帧，不混用 MRQ shot frame。
- capture_id 必须与 render_sample_id 一致；这两个字符串相等只是静态门禁，真实生产须由同一次 MRQ 输出回调生成，不能事后靠文件名配对伪造同步。
- L0..L4 track_id=1..5；R0..R4=6..10。同 seed/clip 跨机位保持实体及 world_m 相同；数据集全局追踪身份是 `(seed, clip_id, track_id)`，不可只用局部数字合并不同比赛。
- mask_id 与 track_id 分开声明；背景0；球等非球员实例须显式列入 excluded_mask_ids，未映射像素 ID 报错。
- 每点有 world_m、uv_px、source_method、visibility_state、visibility_method。头部应额外保留 landmark_manifest_sha256、mesh_identity/source evidence 等字段，sidecar 不删除这些额外 provenance。

COCO17 固定顺序是鼻、左眼、右眼、左耳、右耳、左肩、右肩、左肘、右肘、左腕、右腕、左髋、右髋、左膝、右膝、左踝、右踝。左/右为角色解剖方向，不是图像方向。

## 可见性及 mask 语义

`visible/occluded` 分别输出 COCO v2/v1，必须同时有有效 world_m/uv_px、visibility_method=`surface_depth_test` 或 `manual_verified` 以及 visibility_evidence_ref。本模块只校验声明，**不自动证明证据真实**。骨骼中心位于身体内部，普通相机到骨中心 trace 的提前命中不能单独证明自遮挡；mask 同实例邻域不能单独证明点可见。

`unknown/outside/behind_camera/missing` 在 COCO 视图为 `[0,0,0]`，不假装 v1。原始 UV 和 world_m 仍保留 sidecar。unknown 有效世界点可以用于另外定义的多视角/3D任务；不能宣称这些点有精确可见性或完整 COCO17 可见标签。正式训练前须统计每关键点已知率、可见率、缺失率，不能把 num_keypoints=0 的准备数据当成合格 Pose 数据。

COCO bbox 是可见实例像素的半开 xywh（单像素宽高为1），area 是可见实例像素数，不是 bbox 面积或 amodal 人体面积。segmentation 使用 COCO 列优先未压缩 RLE。完全遮挡/无可见 mask 的实体保留 sidecar，但不写入 mask 主 COCO annotation。关节可位于此可见 bbox 之外。该面积定义会影响 OKS 归一化，论文必须披露并与目标真实数据语义对齐；不能将其直接称为与所有真实 COCO 数据完全一致。

## C++/MRQ 接入设计及未验证项

当前旧流程是 MRQ BurnIn OnOutputFrameStarted 调五个 Blueprint Recorder、每球员13骨，采集首相机世界姿态，然后跨机位重投影。现有 pose_session 主要检验唯一 root 数和全局 actor/bone 集合，不能充分证明每个 actor/frame 都恰有每个骨且无重复。旧编辑器 preview-frame 采样与 MRQ 动画姿态可能不同，不用于新真值。

新运行时逻辑按项目要求 C++ 优先：

1. 在 MRQ 输出帧、完成 SkeletalMesh 动画评估之后采样；保存 source/Sequence/output frame、subsample、时间、Actor、Mesh、骨变换版本、相机标定与实际渲染状态。smoke 首先关闭 motion blur/时间抗锯齿多样本，或明确标签为曝光中心样本，避免模糊 RGB 与单姿态真值含糊配对。
2. 躯干12点使用角色实际生产 VisualMesh 的最终骨变换；5头点使用每 Mesh 经语义校准且身份绑定的 deformed-surface barycentric anchors（独立 `ue/pose_landmarks_v2.py` 准备）。必须绑定实际导入拓扑与蒙皮位置，不复用任意模型的 head 骨固定偏移。
3. 逐帧逐实体验证17点、来源、网格身份、时间戳；逐相机采集一次，或先证明同 seed/camera 重放姿态相同，再共享世界点。当前 adapter 会拒绝同世界帧跨相机不一致，而不是静默接受首相机数据。
4. 对表面锚点以同帧深度/精确表面遮挡证据标注可见性；肢体骨中心点与对应可见表面之间的定义仍须明确，无法判断的点保留 unknown。不要把全身骨中心 depth test 直接当作 COCO 解剖可见性真值。
5. 新 v2 smoke 三个以上动作帧、两机位、正侧背与遮挡覆盖；同 RGB/mask/frame 配对 overlay、多视角重投影残差、像素分辨率、头部语义和已知率验收后，才接生产 resolver。

建议 v2 以 seed 为独立组，组级单份 `pose3d_world.jsonl` 与 landmark/外观/地板 manifest，camera 级 `pose2d.jsonl`、mask 与标定，COCO JSON 为派生导出。实际按是否已证明跨机位姿态一致决定3D是否能共享。按组划分训练/开发，不随机拆机位或邻帧。正式 v1 不改写、不增加重复媒体。

CPU 单元测试只证明数学、ID、mask编码与拒绝规则；没有证明真实 UE 资产骨骼、5个面部语义、视觉准确度、MRQ时序、自遮挡或新数据可提高模型。

## 最小 C++ 采样库

外层新增 `UFutsalPose17Library`（`Source/FutsalMOT/Public/FutsalPose17Library.h`），提供 Blueprint/Python 反射入口。代码已按本机 `E:/UE_5.8/Engine/Source` 核对 API。2026-10-23 root 已完成 `FutsalMOTEditor Win64 Development` 构建 exit0、新 Editor 反射加载及原生 `FutsalMOT.Pose17.SurfaceTriangle` 自动化测试（1 passed、0 errors）。真实 Player003 VisualMesh 参考读取返回17813顶点、30732三角，证据位于外层 `Saved/Pose17Development/actual_visual_reference.json`。这些结果证明最小库可加载及参考拓扑可读，尚未证明实际头部语义锚点、完整17点采样或 MRQ 配对。后续新增并行动画拒绝门禁和下述MRQ pass仍需重新构建验证。

- `DescribeVisualMesh`：输入显式实际渲染组件，返回网格资产路径、LOD0顶点数、参考几何和拓扑 SHA256。拒绝隐藏、未注册或 CharacterMesh0 驱动底模；不自动按 Actor 获取默认 Mesh。
- `ReadVisualReferenceGeometry`：返回 UE 导入后的参考形状（component cm）与有序三角；用于 Blender→UE 顶点映射及校准，不能当成动画 Pose。
- `SampleVisualPose17`：输入实际组件、已校准网格身份、COCO顺序的5个头部锚点和 manifest SHA256；输出17世界米坐标、组件路径、身份、采样方法、引擎frame和骨变换revision前后证据。骨骼12点从同一 VisualMesh 读取。头锚点必须是该UE LOD0实际有序三角，有语义证据且权重有效。

签名方案为 `ue_lod_le_u32_f32_v1`：拓扑SHA对整个LOD0 IndexBuffer有序uint32小端字节计算；几何SHA对LOD0参考position逐顶点XYZ float32小端字节计算，单位component cm。没有JSON空格/小数编码，也不含资产路径等字段；资产路径/LOD/顶点数单独逐项比较。它**不是** `pose_landmarks_v2.py` 中 Blender 候选 JSON 签名，必须经UE拓扑映射、语义和重投影验收后另建 UE runtime manifest，不能直接复制 Blender 顶点编号或 hash。

UE5.8源码 `SkinnedMeshComponent.cpp::GetCPUSkinnedVertices` 会临时切LOD、RefreshBoneTransforms、切CPU skin/recreate renderstate并阻塞渲染线程。库默认**不调用此接口**，仅读取已经启用且预测LOD0的CPU skin缓存，否则fail-closed。`bAllowSlowDiagnosticSnapshot=true` 才允许上述昂贵的诊断采样；结果明确记录可能重评姿态的方法。既有CPU缓存也可能滞后于本次RGB，local stable 只说明本地读过程稳定，**所有结果 bMRQSynchronizationVerified 恒为false**。CPU skin也不自动证明与GPU布料、WPO等最终渲染几何完全一致。

因此该库可用于下一步真实模型锚点诊断、减少对Blueprint Recorder改动；不能直接将诊断调用放到正式MRQ每帧后就声称精确同步。正式接线仍须由MRQ输出回调绑定已完成动画评估的帧、精确LOD及最终形变证据，并通过overlay与多视角检查；必要时改为渲染管线GPU读回，避免每帧重建蒙皮和刷新线程。

## 独立 MRQ 元数据诊断 pass（已构建并通过独立小样）

`UFutsalPose17MRQPass` 直接继承 `UMoviePipelineRenderPass`，重写 `RenderSample_GameThreadImpl`，不注册普通 OnEndFrame、不输出图像、不修改旧 Recorder 或 render_episode，不声明任何 OutputPassIdentifier，避免输出合并器等待不存在的图像。

已核对 UE5.8 实际调用链：

1. `MoviePipeline.cpp::OnEngineTickBeginFrame` 调 `TickProducingFrames` 计算时间步及缓存状态，不等于角色动画已完成。
2. 引擎tick后 `OnEngineTickEndFrame` 调 `RenderFrame`。
3. `MoviePipelineRendering.cpp::RenderFrame` 首先调用 `FlushAsyncEngineSystems`（174行），随后遍历tile/spatial/sample，逐个调用 `RenderPass->RenderSample_GameThread(SampleStateForCurrentResolution)`（510行）。新pass在此收到真实FMoviePipelineRenderPassMetrics，时序比普通Setting OnEndFrame早注册的hook明确。
4. `OnOutputFrameStarted` 不是UMoviePipeline全局delegate；它是BurnInWidget事件，由BurnIn渲染pass在其RenderSample内、所选最后spatial/temporal样本触发。因此不依赖此Blueprint UI事件进行新Pose采样。

**post-flush仍不等于完整动画同步**：`FlushAsyncEngineSystems` 的源码只显式处理流送、shader/asset编译、grass和虚拟贴图，不直接承诺等待 SkeletalMesh 并行动画任务。新底层库检查实际VisualMesh及其LeaderPose是否 `IsRunningParallelEvaluation()`，存在任务则拒绝，不在注释采样里强制重评或等待。即使任务已结束，CPU缓存仍可能不是本次最终GPU几何，所有记录仍保留 `bMRQSynchronizationVerified=false`。

编译并加载后，仅在独立、单shot、单相机smoke队列的PrimaryConfig使用 `find_or_add_setting_by_class(unreal.FutsalPose17MRQPass)`，设置Targets（EntityID、精确Actor UObject名、精确VisualMesh组件名、ExpectedIdentity、已验收5头点anchors、manifestSHA）及MaxRecords。禁止用默认CharacterMesh0代替VisualMesh。pass强制temporal=1、spatial=1、tile=1、motionblur=0，跳过discard/warmup，不放宽失败条件。

采样记录包含真实OutputFrame、SequenceSourceFrame、ShotIndex、CameraIndex/CameraName（不能自行冒填）、temporal/spatial index、HookEngineFrame和底层sample证据或错误。默认64条、最大4096条（按实体计数），达到上限设置bRecordLimitReached，不假装完整。当前明确限定单shot诊断；pipeline teardown期间持久化native JSON，禁止在PIE完成延迟回调读取已销毁setting/pipeline。

该pass默认不启用CPU skin；仅显式诊断opt-in允许受控PIE准备，不写资产、不自动渲染。只有实际同帧RGB/mask、17点头部语义、重投影与遮挡验收后，才考虑升级为正式输出。

### PIE-only 准备及同步（a02小样已执行）

上述默认行为保持；新增明确 opt-in 的 `bPrepareCPUForDiagnostic` 和 `bHideDriverForDiagnostic`，只允许PIE实例。底层 `PrepareVisualCPUSampling` 单次开启CPU skin并以1-based `SetForcedLOD(1)`强制LOD0，`RestoreVisualCPUSampling` 恢复原CPU/分别的rendered/streamed LOD并读回核验。`GetVisualSamplingState` 是只读入口，`r.SkeletalMeshForceLOD` 必须为-1，避免把全局覆盖误当原组件值。开启CPU skin可能unlink共享资产的streaming，恢复CPU标志并不保证重新link；独立报告 `bStreamingStateRestartRequired`，不伪造完全恢复，必要时安全重开Editor。

PIE内显式隐藏目标Owner的CharacterMesh0时，快照visible/hidden/tick option，并保持 `AlwaysTickPoseAndRefreshBones`，避免隐藏驱动网格后动画停掉；teardown恢复并读回。该行为不修改Editor原始Actor或地图。当前 `AFutsalPlayerBase` C++没有负责隐藏driver的BeginPlay逻辑，不能假定Blueprint或PIE必定自动修复双模型污染。

实际CPU缓存是 `FSkeletalMeshObjectCPUSkin::UpdateDynamicData_RenderThread` 更新，不是游戏线程GetCached就天然为当前帧。`SynchronizeVisualCPUCaches` 用全部目标共用一次 `MarkRenderDynamicDataDirty → World.SendAllEndOfFrameUpdates → FlushRenderingCommands`，不重新评估动画；UE5.8的Flush实现会stop recording render command pipes并等待RT/RHI fence。同步放在本pass `OnFrameStartImpl`：引擎先对所有pass调用OnFrameStart，然后才逐个RenderSample，避免RGB pass排在metadata前时先读旧缓存。采样时再校验同engineframe/pose revisions；仍不冒充已经证明RGB/mask一致。

`SetupForPipelineImpl` 在PIE中对显式相机对象做PostProcessSettings/BlendWeight快照，设置MotionBlurAmount override=0、BlendWeight=1；pipeline teardown恢复。MRQ的逻辑shutter来自实际Camera/Volume的MotionBlurAmount，单独关渲染motionblur cvar不足以保证TimeData.MotionBlurFraction=0。含动态PostProcess轨道的源序列须另行核验，不能任意覆盖后就假定相机一致。

### 独立 smoke runner

`ue/pose17_smoke_runner.py` 默认仅 `preflight`，该默认probe已由root真实MCP执行通过。可选显式计划模板 `configs/pose17_smoke_player003_diagnostic.json` 是独立runner格式，**不是** grf-ue task格式。模板使用当前Player003、显式两个源Sequence和精确相机Actor对象名；真实运行前仍须核对当前资产与时间范围。

UE内由root通过真实MCP设置环境变量后调用该文件：

```python
os.environ['FUTSAL_POSE17_SMOKE_PLAN'] = '<代码仓库>/configs/pose17_smoke_player003_diagnostic.json'
os.environ['FUTSAL_POSE17_SMOKE_ACTION'] = 'preflight'  # 默认，只读
# 门禁/appearance诊断通过后，独立调用prepare，再独立调用submit。
```

`prepare` 检查dirty/PIE/MRQ、实际VisualMesh、源Sequence绑定/CameraCut/动态轨道与帧范围；复制源Sequence至 `/Game/FutsalMOT/Development/Pose17/<attempt>/<camera>`，对比绑定/轨道，不改源序列。新Saved attempt拒绝覆盖，保存计划、读只门禁、landmark原始快照；新资产只通过UE API创建保存。CPU/隐藏driver均须计划显式true。候选landmark未production accepted时，必须显式 `allow_unaccepted_landmarks_for_diagnostic=true`，输出保持诊断性质。

`submit` 使用自己的 `MoviePipelineQueue` 与 `MoviePipelinePIEExecutor.execute(queue)`，不清空/占用原队列。异步回调为每job写Pose/恢复收据，保留真实MRQ样本索引、世界17点、unknown可见性、cache fence标记和restore/streaming状态；最后检查预期记录数量、唯一帧实体键和PNG/EXR计数。模块会话持有独立队列/回调，跨MCP调用不丢引用。Executor成功只能得到 `rendered_requires_RGB_mask_pose_alignment_acceptance`，没有actual Pose/RGB/mask配对验收就不升级生产成功。

新增CPU runner/导出针对测试31 passed；这些测试不调用UE，不能替代新C++构建、PIE角色动态性、实际回调、相机恢复或渲染同步验收。源Sequence含动态轨道仍不证明采到的三个动作不同，须比较实际多帧Pose与RGB。原始ObjectID EXR须后续在P1解码为实例mask，UE环境不导入NumPy/OpenEXR。

### a01生命周期故障与native收据修复

2026-10-23实际 a01 首机位输出3张4K FinalImage PNG及3份EXR后，PIE清理后的Python job完成回调触发访问异常，没有生成Pose收据；失败attempt及CrashContext保留，不补写成功或覆盖。引擎 `FMoviePipelineOutputData::Pipeline` 的文档明确仅用于身份，PIEExecutor先清理World并清ActiveMoviePipeline，再下一tick广播完成通知；非空Python wrapper不证明底层Pipeline或Job仍可解引用。先前从延迟data.pipeline取config的实现已删除。

新实现由C++ pass的 `TeardownForPipelineImpl` 在PIE清理前将全部记录与恢复摘要写入显式 `DiagnosticReceiptFile`，限制在会话Saved/smoke目录且不覆盖。记录增加相机世界基/位置、当前Camera ViewInfo FOV/aspect/constraint、输出尺寸、jitter等投影来源；它们仍需与实际render view验证，而不是未经验证声称获得了最终像素投影矩阵。

Python不再订阅携带PIE Pipeline/Job参数的job_finished/error delegate，仅保留强持有Executor的最终bool回调，仅读已落盘native JSON及媒体文件；Editor恢复只读检查改由结束后独立 `ACTION=postflight` MCP调用。`ue/pose17_native_receipt.py` 是不依赖unreal的标准化入口，处理UStruct字段命名、向量及unknown保留。默认诊断模板使用a02，a01保持失败历史。

### a02真实小样与CPU配对验收（2026-10-23）

新C++实际构建成功，`player003_mrq_a02` 两机位×3帧原生3840×2160已完成，native落盘、最终纯文件回调及独立Editor postflight均通过。原生几何自动测试先前实际1 passed；新增CPU导出/runner/native/配对测试共49 passed。当前源仅为短诊断Sequence，不是正式GRF连续轨迹，目标仅L0实际Player003；候选头锚点仍未升级production accepted。

CPU工具运行示例（在内层Python3.9环境）：

```powershell
.venv/Scripts/python.exe -m grf_ue_bridge.tools.inspect_pose17_smoke <Saved/smoke/attempt> <全新分析目录>
```

工具读取native收据、EXR实际渲染相机元数据与精确FinalImage RGB，排除WorldDepth/ObjectID调试PNG；核对媒体/native源帧、相机位置/坐标基/FOV、RGB/ObjectID FOV、4K尺寸、AA单sample、零shutter/jitter、CPU fence同引擎帧及pose revision稳定。仅写新目录的实例mask、17点overlay、裁剪图与report.json，不改媒体或正式数据。EXR相机对称透视采用现有像素中心`(width-1)/2`约定；最终renderer projection matrix未独立读回，非对称视锥/overscan/squeeze会拒绝。

实际报告 `Saved/Pose17Development/smoke/player003_mrq_a02_alignment_a03/report.json` 的结构配对通过、errors为空；两机位3个源帧的17点world坐标逐点相同，最大差0m。相邻帧去除髋中点平移后的最大变化为0.081m与0.143m，证明确有身体姿态变化。overlay已由Agent查看，Pose跟随实际角色；第二机位球门网遮挡明显。mask命中仅输出观测，所有可见性保留unknown；所有报告`MRQ_sync_verified=false`，不把此小样升级成生产精确遮挡/全表面像素同步证据。

采样UWorld时钟相对MRQ缓存时钟有约0.1秒恒定偏移。引擎 `MoviePipeline.cpp:297` 初始化MRQ缓存世界时间，`MoviePipelineTiming.cpp:711` 独立按FrameDeltaTime累加；它不是每次渲染直接读取UWorld。报告同时保留两时钟与差值，不能仅因绝对值不同断言Pose晚一帧，也不能删除差值伪造相等。实际配对依赖独立源帧/渲染相机/同引擎帧缓存证据，精确动态遮挡与最终表面几何仍需后续量化验证。

### 完成会话的显式release

连续独立小样无需重启Editor来清除runner自有强引用。`ACTION=release`只接受executor成功、所有native采样/恢复收据完整、独立postflight成功且实时Level正确、dirty/PIE为空、全局及本executor均未渲染的会话；重新读取native收据/state/editor_after，核对当前players仍与postflight一致。未知streaming恢复需求、失败/未完成会话均拒绝。

释放通过UE5.8公开 `remove_callable` / `contains_callable` 按本会话保存的callable解绑并读回，不清空delegate、不清空全局或其他队列、不删除Sequence/媒体。新写 `session_release.json`，原state仅增加release字段，保留诊断完成状态；清空本会话字典以解除旧globals/closure持有的queue/executor/settings，历史仅保留纯JSON。release不代表生产验收，后续prepare仍重新执行全部门禁并使用新attempt。新增release反例及所有Pose17针对CPU测试61 passed；真实MCP release需现场执行核验。
