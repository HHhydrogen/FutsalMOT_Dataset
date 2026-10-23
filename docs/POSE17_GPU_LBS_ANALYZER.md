# GPU渲染/CPU-LBS诊断分析门禁

2026-10-23新增独立opt-in路径。`inspect_pose17_smoke.py`保留旧CPU分支的共同帧/EXR相机/fence/revision检查，新增`pose_sampling_path=gpu_render_cpu_lbs`严格分支。初版先完成CPU契约测试；随后003 a07已有独立实际配对/头深度报告，但仍不是全表面等价或COCO可见性验收。004组合scope的CPU契约另行记录在下文，不能继承003的角色验收。

## 显式计划与真实收据

原无override路径要求计划`allow_gpu_render_cpu_lbs_diagnostic=true`、`allow_pie_cpu_preparation=false`、scope精确为`pose17_gpu_render_cpu_lbs`、warmup为8..120整数。这个scope仍不接受视觉mesh override；新004组合必须使用下文的独立scope。只存在计划不能通过；没有native收据、完整RGB/EXR帧集合或实际采样记录，分析器拒绝生成通过结果。

native顶层必须落盘相同path/scope及`gpu_lbs_diagnostic_requested=true`，有实际prepared engine frame、无角色override，teardown camera恢复成功、没有restore/pipeline错误或streaming重启标记。每个目标从计划中真实landmark manifest文件计算SHA，与native行绑定，并对比其UE资产路径、签名scheme、LOD0、顶点数、拓扑/参考几何SHA。不能用Blender裸索引或仅同名mesh替代。当前UE `LOD`属性真实JSON写作`lOD`，经既有normalizer是`l_od`；新分析器显式采用这一键，不修改共享normalizer。

## 每行GPU/LBS证据

采样方法精确为`gpu_render_component_pose_cpu_lbs_no_morph_cloth_deformer_wpo_diagnostic`。实际读取时CPU skinning=false、非static、predicted LOD=0；目标为计划绑定的真实PIE Actor/MPFBVisual，骨pose来源也须是同一个PIE Actor的组件。

原同帧门禁完整保留：cache fence成功、local read stable、engine before=after=hook、pose revision before=after、EXR源帧/相机及单temporal/spatial sample、零jitter、零motion blur。新分支另外要求：

- native单sample索引均为0、camera_name匹配计划；不接受缺失或伪装成数字的bool证据。
- `pose_fence_method=gpu_dynamic_data_endframe_tasks_finished_render_commands_flush`，end-frame任务已结束。
- 实际warmup计数恰好等于hook frame减prepared frame，且不少于计划阈值；不能把旧attempt或相同source frame视为等价warmup。
- 当前RefToLocal SHA使用`ue_ref_to_local_rowmajor_f32_le_v1`；组件LOD0 skin weights SHA使用`ue_component_lod0_fixed_skinweight_raw_v1`。不接受空值、未知scheme或非64位小写hex。
- 当前骨名唯一、包含实际12肢体骨，骨数与显式Transform数组一致。骨及actor/visual世界Transform采用`ue_transform_translation_cm_quat_xyzw_scale_xyz_v1`：translation_cm[3]、rotation_xyzw[4]、scale_xyz[3]；要求有限、四元数单位且scale不退化，不猜FTransform ExportText。

`gpu_lbs_eligibility`必须来自实际native扫描，checks_executed/eligible/PIE/可读weights双重显式确认；CPU/static/Nanite/并行动画/非有限morph/cloth数据/variable influences均明确false。活跃morph、manual morph、external morph sets、cloth模拟/section、deformer资产/实例、WPO和空材质slot计数逐项为整数0，缺字段也拒绝。WPO计数包含overlay材质。权重与reference顶点数都等于实际mesh identity vertex_count，最大固定影响数大于0。不能从方法字符串或缺省零值推定没有形变。

这些签名/骨pose/资格值绑定的是native报告的当前组件状态；没有导出所有矩阵与GPU最终vertex buffer并做逐顶点重算比较。分析通过仍不证明最终GPU表面完全等价。报告保留`MRQ_sync_verified=false`、`final_rendered_surface_equivalence_verified=false`，并携带完整eligibility与签名上下文。

## 与头部深度工具的关系

新报告保留原camera、世界点投影、keypoints、RGB/EXR路径与帧配对格式，额外写sampling evidence；`inspect_pose17_head_depth.py`仍消费真实EXR和这些点。没有修改该工具、5mm阈值、边界/coverage/深度跳变门禁或unknown可见性。低残差只属于原有表面一致候选，不会因切到GPU/LBS自动升级可见性。

## 本次CPU证据

`tests/test_inspect_pose17_smoke.py`的新GPU数据全部明确是toy schema假例，不是实际native捕获。测试覆盖逐个eligibility缺失/错误、真实CPU/static/LOD冲突、manifest/mesh/bone/transform/签名错配、warmup、共同fence/frame/revision/相机拒绝以及真实writer的`lOD`规范化。另有“仅计划无native输出必失败”的反例。

修改前旧CPU配对和头深度18项测试通过；初版新增GPU后两个模块119项测试通过。root随后补两种精确FinalImage命名和2项测试，并使head_depth传入sampling_context；这些补丁完整保留。003 a07实际报告见`Saved/Pose17Development/smoke/player003_closeup_a07_alignment_a02/report.json`及`player003_closeup_a07_head_depth_a02/report.json`。本次未读媒体或重算哈希，仅复核既有JSON：三相机各3帧、九行当前native门禁兼容，世界点与已记录配对报告最大差0；5mm、unknown和MRQ未正式验收标志均保留。

## Player004独立GPU/LBS＋PIE override组合

scope精确为`pose17_gpu_render_cpu_lbs_character_calibration`；原scope不能因CPU=false自动接受override。单角色组合已泛化为计划整数`calibration_player_id=4..9`、单L0/Player_L0/MPFBVisual，各自精确生产资产及新manifest身份，不能跨角色复用。资产使用literal白名单，003路径没有`Mesh/`子目录，004–009有；不按同名mesh或路径prefix放行。不新增重复native角色ID门槛。要求GPU-LBS opt-in、PIE override和隐藏driver均显式true，CPU渲染准备false。两套计划warmup须为相等8..120整数。

native override证据必须唯一，记录原003到新004，实际PIE组件与每帧样本相同；共享`SKEL_MPFB_Base`、`ABP_MPFB_RuntimeRetarget`和原PIE动画实例路径明确。以下bool都须显式true：identity_verified、apply_succeeded、animation_binding_preserved、new_asset_materials_verified、restore_attempted、restore_succeeded、streaming_state_matched、gpu_lbs_validation_attempted、gpu_lbs_validation_succeeded；apply/restore/GPU validation error均空。

`gpu_lbs_eligibility_after_override`必须有与每帧相同的完整GPU资格字段，权重/reference顶点数等于新004 mesh identity；不能只看CPU=false或eligible=true。它是swap后、GPU LOD准备前实际native扫描。prepared frame不得早于applied frame；每帧GPU elapsed精确为hook-prepared，override elapsed精确为hook-applied，后者至少等于前者且均达到计划阈值。共同frame/revision/fence和相机门禁仍执行。

材质证据保留实际applied asset material路径和原OverrideMaterials快照（原空指针按native格式为空串），并要求原生pointer逐项读回布尔与恢复布尔通过。这里没有独立材质hash、恢复后材质全值清单或重新读取UE资产的证据，不虚构这些保证。恢复/材质布尔缺失、失败或路径错配时严格拒绝。角色变更只发生在PIE，不把native报告视为Editor资产修改授权。

组合补丁后两个模块194项CPU测试通过，含旧003/CPU分支、root两个媒体命名反例、post-override逐字段资格、九项apply/恢复/材质状态、004身份及两套warmup反例。此后root实际004组合native事务、三相机4K配对和5mm头深度报告无错误，并查看六张图；这属于004窄范围诊断，仍保留unknown、不代表正式COCO可见性或独立定位精度通过。005–009不再逐角色重复近景渲染和像素审计，改为下述一次联合小样。

## 十实体联合快速诊断

独立scope为`pose17_gpu_render_cpu_lbs_joint_diagnostic`，显式`allow_joint_pose17_diagnostic=true`，保留GPU、PIE override、隐藏driver及禁CPU准备门禁。固定映射为L0=003、L1=004、L2=005、L3=006、L4=007、R0=008、R1=009、R2=003、R3=004、R4=005；每target计划整数`calibration_player_id`与其实际manifest身份、Actor/MPFBVisual和`Player_<entity>`绑定。已有003的两实体不override，其他八实体各有原生事务。重复004/005按实际组件匹配，不按applied资产路径合并。

联合scope允许原资产为003–009精确生产白名单；native validator另与Editor前置快照绑定。单角色scope仍只允许原003，不随联合分支放宽。八事务均保留身份、共享骨架、动画、材质、恢复和swap后GPU资格检查，prepared不早于任一apply，两种warmup分别核对真实引擎帧差。不新增native角色ID或额外材质hash。

四个原相机、各自显式`source_sequence`、`start_frame`和明确3或10个连续输出帧是短诊断边界。原十帧联合小样保持不变；三帧用于地板配对快速诊断。只允许`frame_count=3, review_frame_indices=[1]`或`frame_count=10, review_frame_indices=[5]`，其他帧数/抽样帧拒绝，不能由bool冒充整数。文件名使用相对输出帧0..N-1；EXR `sequenceFrameNumber`须等于native源帧且为`start_frame+output_frame`，`sequenceFrameNumberRelative`须等于文件/native输出帧。非零起点不能被当作帧0重新解释。十实体/七角色/四相机和全部身份/fence门禁不因缩短帧数而改变。

全40帧/400记录（十帧方案）或12帧/120记录（三帧方案）读取EXR公开`OpenEXR.File(..., header_only=True).header()`、Cryptomatte header manifest和RGB尺寸，逐条执行身份、fence/revision、帧与相机门禁；不为此加载所有帧channels。仅显式中间输出帧5或1的四帧完整解码mask，各保存一张mask和一张全员overlay，共八张PNG；不输出全量单人4K图或裁图。其他帧bbox、instance area和mask hit明确为`None`、`pixel_evaluation_status=not_evaluated`，不能当正式mask/bbox结果或全量像素审计。姿态位置仍投影，visibility保留unknown。两方案都核对完整帧集合，缺一帧也拒绝。

本机OpenEXR 3.4.13公开构造器文档确认`header_only=True`只读header、不读像素。新增微型CPU假媒体集成实际通过normalizer、40帧/400条配对、仅四次中间帧解码和四张全员总览，并检查其余帧空值；针对联合分支及少量旧CPU/003/单角色/媒体名回归共16项通过，190项历史参数矩阵未重复执行。所有新fixture均为自建toy，不是联合MRQ捕获证据。本补丁不读取真实媒体、不修改head_depth/5mm或训练流程，真实联合小样仍由root另执行。

地板三帧扩展先用CPU反例复现`frame_count==10`硬编码拒绝，再最小允许3/10两种计划。新增3帧合法、错误中间帧/bool/其他帧数拒绝、缺一RGB帧拒绝，并将实际normalizer→分析器tiny集成参数化为3帧/10帧；新鲜针对性回归19项通过、190项未重复执行。已有十帧实际报告没有重写；失败分析目录保留，后续实际地板配对使用新分析目录。
