# COCO17 头部表面锚点：独立实验契约

最新增量：003和004已完成GPU渲染/CPU-LBS三近景头点方法验收；七模型已共同完成十实体四机位真实GRF短动作。当前使用[逐字节校准快照](../calibration/diagnostic/pose17_head_surface_v1/README.md)，原静态候选字段不回写。联合抽样不是逐点表面精度或遮挡验收，所有visibility仍unknown，正式批产未启用。下文保留标定过程与历史时点限制。

当前状态（2026-10-23）：Player_003–009七个生产角色的35个静态候选已映射到各自真实UE LOD0三角；Player_003已通过两机位三帧动态采样及结构配对，近景五点语义、动态表面深度与可见性仍在验收，不得开启正式COCO17批量渲染。下文保留Player_002及后续生产角色的诊断历史，历史候选不自动获得新验收状态。现有`ue/pose_bones.py`十二肢体点不受修改。

## 已核实的候选来源

2026-10-23 本机 MPFB 插件自带 `data/rigs/standard/rig.openpose.json`，SHA256 为 `b841bbeb452532f59e119531c55a750ce05b7e1c3db247af7c137c7a9d4b7cc4`。

| COCO 点 | MPFB 定义 | 原始 basemesh 顶点均值 |
| --- | --- | --- |
| nose | Neck.tail | 297, 7063 |
| left_eye | LEye.tail | 14634, 14661 |
| right_eye | REye.tail | 14706, 14733 |
| left_ear | LEar.tail | 12106, 12195 |
| right_ear | REar.tail | 5507, 5598 |

上述语义索引必须作用于**当前角色 Shape Keys 求值后的几何**；不能读取 `Human.data.vertices[].co` 的 Basis坐标。最初误用Basis导致nose落在下巴、eyes落在嘴角、ears落在下颌两侧。后续实际重算已证明该失败不应归因于rig语义定义错误。骨名对应tail，不能直接取head；JSON的 `default_position` 不能代表当前角色形态。

只读 headless Blender 5.1.1 实际打开 `Player_002/MPFB_Player_002.blend`：METRIC/1.0，Human scale=1，原始顶点19158；经 Armature、Hide helpers 与服装 MASK 的 evaluated Human 顶点7992。只读诊断前后 `bpy.data.is_dirty=false`，未保存文件。

最初Basis语义均值到当前evaluated Human最近表面距离：鼻19.61mm、两眼各1.78mm、两耳各27.16mm。这是失败诊断距离，**不是语义定位误差或已验收精度**。CPU LBS 与Basis候选相差不足0.00024mm，仅排除了骨骼姿态空间差，不能排除Shape Keys。插件base.obj与角色原始19158顶点、18486面逐面索引完全相同。MPFB实际AI输出 `_openposeconstants.py` 使用另一套定义（鼻5134、耳12033/12226与5434/5629，眼取eye bone），两套均须在真实形态上检验，不能仅因定义不同判任一套错误。

实际Shape Keys后重算旧rig索引并在相同三视图叠图：鼻尖、双眼中心已经对齐；耳落在耳廓外侧，区别于手动方案选的耳道中心，仍需统一论文/COCO耳语义并复核。鼻z=1.5779146m、眼z=1.6182570m、耳z=1.5999528m；鼻到body表面0mm，耳0.407mm。眼到body3.258mm不代表错误，因为实际眼球是独立mesh。图片 `mpfb_player002_shape_head_*.png`，量化报告 `mpfb_player002_shape_head_space_diagnosis.json`。语义索引可恢复为自动候选生成途径，仍不自动获得精度/可见性/UE动态验收。

报告在外层 `Saved/ResearchDiagnostics/mpfb_player002_head_candidates.json`、`mpfb_player002_head_space_diagnosis.json`、`mpfb_openpose_topology_check.json`；旧候选前/侧图为 `mpfb_player002_head_*.png`。

## 已实际生成的像素射线表面候选

主agent在绑定1000×1000、ortho_scale=0.35m的诊断图上选鼻尖、双眼中心与耳道中心。脚本 `Saved/ResearchDiagnostics/calibrate_mpfb_head_pixel_rays.py` 对实际 evaluated mesh 求射线/三角相交并保存重心权重；鼻耳仅射真实 Human body，双眼仅射独立 `Human.low-poly` 眼球，避免射到眼眶后表面。

`mpfb_player002_pixel_ray_calibration.json` 记录所有像素、相机 matrix_world、投影与单位、ray、命中三角/重心坐标、3D坐标、逐表面拓扑/几何SHA（ordered_json_v1），状态为 candidate_unreviewed。前、左、右、斜视图为 `mpfb_player002_ray_*.png`。鼻z=1.5792m、眼z=1.6184m、耳z=1.6002m，已肉眼查看鼻/双眼与左耳落点合理；右耳初始参考为对称视图像素，仍需独立审核。

这些像素定义与落点自洽不等于独立语义精度验证，不能用同一像素构造/重投影的近零误差宣称标注精度。所有候选还需明确角色自身左右、独立多视角语义验收、UE合并后的实际LOD0映射及动态验证。Blender源文件未保存，UE资产未修改。

主agent已实际检查 front/left/right 三图并接受 Player_002 源mesh静态五点落点；范围仅为鼻尖、双眼中心、双耳中心的静态语义定位，不代表 UE/动画/遮挡/独立精度通过。`mpfb_player002_source_mapped_head_candidates.json` 保存该review_scope及四张图SHA。脚本 `map_mpfb_head_candidates_to_source.py` 在不保存的临时mesh copy上添加 POINT INT original_index，经过原有 modifier 求值后读取该属性，要求复制求值后的全部顶点/拓扑SHA与审核对象一致。

实际 Human 7992点与眼球96点均保留唯一有效原始索引且无启用的细分生成点，五点已全部映射到原始三角顶点+barycentric。原始 Human19158点有序面SHA为 `1f78e347f544c2377abee83216f297dc9499b73b7a149bd8b8d849b5553d95db`；原始眼球96点有序面SHA为 `61657f077ecffe9687ecaa9a7d61d2020ba67a5533f2357c8f27e15fd50d0d8b`。后续角色只有逐模型匹配各自原始拓扑并重新视觉检查，才可将这些源语义锚点作为迁移候选；不可直接复用7992点的求值后索引。启用SUBSURF时整数属性可能发生插值，脚本显式拒绝把此类生成点当原始索引。

## 新模块与约束

`ue/pose_landmarks_v2.py` 仅使用 Python 标准库，可在 P1/P2 两侧使用：

- `COCO17_NAMES` / `COCO17_FLIP_IDX`：标准17点顺序与左右翻转。
- `mpfb_openpose_head_candidates(rig)`：读取语义索引，强制返回 `candidate_unreviewed`；调用方必须提供Shape Keys后的几何，不是Basis坐标。
- `mesh_fingerprints(vertices, triangles)`：有序三角拓扑与坐标 SHA，禁止不同 mesh/LOD 共用索引。
- `evaluate_surface_anchor(deformed_vertices, indices, weights)`：对真实形变后三角顶点做重心插值；坐标系和单位保持不变。
- `validate_landmark_manifest(...)`：缺少某个头部点、逐点语义review、完整mesh身份、校准证据或重投影精度门槛时拒绝准入；不接受固定 head 偏移。

manifest schema 为 `futsalmot_head_surface_landmarks`、version=1，含 `mesh_identity`（asset_path、lod、vertex_count、geometry_sha256、topology_sha256）、`coordinate_frame`（blender_object_m 或 ue_component_cm）、`source_method=deformed_surface_barycentric`、`calibration`（accepted 状态、evidence_ref、实测 max_reprojection_error_px 与预先约定 acceptance_threshold_px）、`anchors` 五点。

当前validator针对单个已合并mesh；源Blender眼/身体分mesh候选尚不能直接通过它。实际UE合并映射后按真实组件建立该manifest。Python参考 SHA 是有序JSON编码；UE采样库若使用little-endian uint32索引/float32坐标，必须显式标记 `signature_scheme=ue_lod_le_u32_f32_v1`，不能混用或比较两个方案的SHA。

## 源参考几何至FBX/UE的严格对应

`ue/pose_geometry_correspondence_v2.py` 实现显式4×4仿射转换、容差网格搜索及source→多个target候选，兼容UV/法线拆点和顶点重排。三角锚点只允许在真实目标有序三角拓扑找到唯一对应面，重排顶点时同步重排barycentric。零面或多个面时拒绝，不采用“最近三角就算成功”的回退。9项新测试通过。

`export_pose17_source_reference_geometry.py` 已只读导出源blend两套坐标（原始Basis及禁用modifier但保留Shape Keys的真实参考）和实际交付FBX的Blender重新导入几何。真实FBX路径 `D:/projects/FutsalMOT_blender/characters/Player_002/export/MPFB_Player_002.fbx`，含body7992、eye96、hair2196、eyebrow124、clothes2136、shoe1594六个对象；Blender重新导入world坐标与源米坐标一致，object matrix约0.01、mesh data约厘米。**这仅是Blender FBX importer的实测结果，不推定UE Legacy importer的轴/缩放。**

严格容差10µm、显式identity变换：使用Basis时body0%对应，使用真实Shape Keys参考后FBX body7992点100%在source中找到、最大误差0.541µm；眼96点100%、最大0.360µm。五个人工表面锚点都找到唯一实际FBX三角。source Human全19158点包含已删除helper/衣服遮挡区域，所以source→FBX覆盖41.716%符合MASK裁剪；不能要求被删除区域也有对应。完整结果在 `mpfb_player002_source_fbx_correspondence.json`。

接UE阶段须读取同一角色实际asset_path、LOD0 positions_cm、ordered triangles、section/material与triangle归属（若有）、参考骨骼变换、ImportData实际FBX路径及ConvertScene/ConvertSceneUnit/ForceFrontXAxis和import scale；不能用Player002参考几何直接匹配Player003。SOP明确若干轴转换参数未验证，因此工具不猜矩阵。经实际参考几何确认米→厘米与轴映射后，预先固定容差；全部锚点三角须唯一对应，较大误差/匹配率不足时停下诊断，禁止调整容差直到“通过”。

### Player_003 实际UE映射进展

2026-10-23 root通过真实UE C++ API导出Player003 LOD0 17813顶点/30732三角及实际ImportData：源为 `characters/Player_003/export/MPFB_Player_003_SOP.fbx`，convert_scene=true、force_front_x_axis=false、convert_scene_unit=false、import_scale=1且translation/rotation=0。

只读核对Player002与Player003原始body和眼球有序面拓扑相同后，迁移语义三角，重新读取Player003自己的Shape Keys参考。实际SOP.fbx body11134点，不等同Player002的7992点；当前实际FBX含七mesh。枚举48个固定100比例signed-axis、禁止自由缩放/位移拟合，只有 `UEcm=(100*BlenderX,-100*BlenderY,100*BlenderZ)` 达到全FBX/UE参考几何双向100%顶点对应，最大位置误差1.9604e-5cm；镜像X候选仅90.6%，已拒绝。

鼻与双眼找到唯一真实UE三角，双耳由于同quad重三角化不能复用原始三个顶点组合；在同一个原始quad内投影到实际UE三角的距离0.00458724cm=0.0458724mm，超过事先固定的0.001cm几何对应容差。因此严格对应收据仍明确记双耳REJECTED，未改阈值。经root指示，另将该耳表面位置作为 `recalibrated_on_ue_retriangulated_polygon` 的**独立重标定候选**，不称源锚点同一点。此限制同原始polygon的重三角化投影逻辑有独立通过/超容差拒绝测试。

实际四视图为 `mpfb_player003_mapped_{front,left,right,threequarter}.png`。root已实际检查front/left/right并接受当前角色的静态鼻尖/双眼/双耳语义候选。`mpfb_player003_ue_surface_head_candidates.json` 含UE真实三角/重心、身份SHA、每点方法与源→UE改变量、相机、static_semantic_review_scope及三图SHA；状态仍为candidate，明确not_animated、not_MRQ_verified、precision_not_measured。严格失败收据为 `mpfb_player003_source_fbx_ue_correspondence.json`，不可覆盖成全五点严格匹配成功。

### 正式8组实际角色覆盖

只读将8份group.json的80个实体player_id与各自catalog_ref.models连接，实际只使用Player003–009七个mesh，未使用001/002。分配次数003=12、004=10、005=16、006=10、007=12、008=13、009=7。收据 `production_pose17_mesh_inventory.json` 保留逐组实体→catalog→mesh，不能由目录里有几个角色推断使用集合。

004–009源文件已分别只读打开、提取各自Shape Keys参考并核对原始body/眼球有序面SHA，七角色均与标定源原始拓扑一致；源候选在 `production_source_head_candidates.json`。当前各角色五个源头部语义坐标完全相同，这不证明整个身体几何相同，但说明目前这部分头部几何没有可宣称的多样性；不能把头发/肤色/衣服变化自动写成脸型变化。

每个新增角色仅生成一套front/left/right静态图与相机/图SHA，命名 `mpfb_playerNNN_source_{front,left,right}.png`，独立候选收据 `mpfb_playerNNN_source_head_candidates.json`。静态源语义审核与实际UE角色资产参考映射分别准入，不把共享源拓扑当成共享UE顶点编号。批量只读源提取和出图不保存源.blend，也不切换UE Actor或更改场景。

root已实际检查004–009全部18张源图，记录 `source_static_semantic_reviewed_only`。随后资产版只读UE API导出7个实际mesh的参考位置/有序三角/ImportData，无场景mutation。逐角色按照其实际ImportData的SOP.fbx重新导入，全部唯一signed-axis变换与003一致，全FBX/UE双向顶点覆盖100%、最大误差1.9604e-5cm；鼻与眼唯一三角匹配，耳保留相同严格匹配失败并独立重标定。不能因此宣称五点都是严格源点对应或已通过动画验收。

正式7个实际FBX body的Z>1.50m裸头区域各4243点，经逐角色独立提取、1µm固定容差双向比较，覆盖100%、最大误差0；结果在 `production_bare_head_geometry_equivalence.json`，范围不含独立眼球/头发。因此相同头部几何有实际全区域证据，不能写成当前数据已有多种脸型。

`production_pose17_head_candidate_registry.json` 给出7个角色独立候选路径/SHA/UE身份；各 `mpfb_playerNNN_ue_surface_head_candidates.json` 均标明未验声明、逐点方法及静态审核范围。`verify_production_head_candidate_registry.py` 已核验正式catalog覆盖、文件SHA、实际UE身份、35个真实有序三角/重心插值坐标；收据 `production_pose17_head_candidate_integrity.json` 明确仅诊断完整性，不是生产验收。当前CPU头部/几何/旧十二点测试38 passed。

每个点含 `triangle_vertices` 三个索引、`barycentric` 三个非负和为一权重及 `semantic_review`（reviewed、method、evidence_ref）。校准时对参考几何做身份校验；动画采集时对**同一拓扑的真实形变顶点**计算插值，不把每帧变形几何 SHA 当静态参考 SHA。

## 下一步实际标定闭环

1. 对每个实际生产角色先确认源文件/UE资产对应关系，不把所有角色视为 MPFB Player_002。非 MPFB mesh 单独建 manifest；无对应校准则停用其 COCO17，不能填猜测点。
2. 在固定参考姿态、固定空间读取 MPFB 语义候选，确认 helper/服装删除与当前动画没有造成空间混用。前、侧、后及斜视对比鼻尖、眼中心和耳语义，记录截图/相机和误差。
3. 候选需要时投影到对应头部/眼表面，并确认语义正确；最近面搜索仅是候选生成，不是自动验收。保存逐模型表面三角与权重、实际 mesh 指纹及标定证据。
4. Blender→FBX→UE 会拆点/焊接/重排。必须读取实际 UE LOD0 并建立可验证映射；禁止直接把 Blender 顶点号交给 UE。UE 运行时由 C++ 对实际 visual skeletal mesh 的最终蒙皮顶点取样，采样时序须与 MRQ 的动画/FootIK/retarget 结果相同。
5. 若短期采用逐模型标定的 head-local 锚点，只可标为近刚性无表情近似；绑定mesh/标定参考姿态与误差，不能标为 `deformed_surface_barycentric` 或称精确蒙皮真值。
6. 三维坐标经 `pose17_export.project_landmarks` 投影后保持 `unknown` 可见性，直到真实深度/遮挡证据齐备。Surface 点已准确不代表可见性已准确；self-occlusion、外部遮挡与超出画面需分别处理。

只有单角色、多姿态、多视角 RGB/点叠图与 mask/深度共同验证通过，才建立新的 Pose17 小规模输出；不修改当前正式8组数据或补写虚构 Pose。

## 已运行验证

`tests/test_pose_landmarks_v2.py`：18 passed。覆盖 COCO 顺序、flip 互逆、重心插值对仿射变换等变、非法/退化三角形、非有限权重、mesh重排、未审核语义/超阈值/缺点/固定偏移拒绝、MPFB head/tail 语义。CPU测试不等于 UE 蒙皮、跨引擎映射、视觉语义或真实可见性已通过。
