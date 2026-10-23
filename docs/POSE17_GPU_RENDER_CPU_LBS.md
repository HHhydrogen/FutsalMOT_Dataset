# 保留GPU渲染的CPU LBS诊断路径

状态：2026-10-23，原Player003新路径已实际完成a07三近景×3帧原生4K，native采样/恢复、Editor独立postflight及session release均通过，streaming_restart_required=false，配对报告errors为空。它仍是短诊断动作，不是正式GRF连续序列或完整可见性验收。新增单004角色组合路径的七个CPU测试文件当前共203 passed；该组合尚待新的C++构建和真实小样。

## 故障证据与目的

Player004切换后的EngineWarmUpCount=8小样发生RHI LocalVF/View uniform buffer不匹配；原Player003、不切mesh、相同8帧预热也复现。因此切换角色不是必要条件。源码显示CPU蒙皮会更新LocalVF，而Lumen卡片捕获使用缓存MeshDrawCommand；缓存绑定失效是有依据的假设，尚没有运行时指针证据证明根因。不能把关闭Lumen、去掉预热或绕过RHI校验当作正式修复。

本路径保持实际GPU蒙皮渲染，只按当前组件骨姿态计算注释坐标，避免为了采样改变渲染MeshObject。旧`cpu_skin_cache`默认行为保留，不能静默回退或把旧CPU结果冒充GPU路径结果。

## 显式计划

仅独立smoke计划允许：

```json
{
  "pose_sampling_path": "gpu_render_cpu_lbs",
  "allow_gpu_render_cpu_lbs_diagnostic": true,
  "diagnostic_scope": "pose17_gpu_render_cpu_lbs",
  "allow_pie_cpu_preparation": false,
  "allow_pie_driver_hide": true,
  "gpu_lbs_warmup_frames": 8
}
```

预热为整数8..120。此原始GPU scope仅原角色003，禁止混入PIE角色override。runner以原生pass的`use_gpu_render_cpu_lbs_for_diagnostic`及`minimum_gpu_lbs_warmup_frames`属性配置，明确EngineWarmUpCount、RenderWarmUpCount=0并读回；仍只输出独立新attempt，不提交正式batch。

## 原生实现边界

`PrepareGPUVisualLBS`只快照当前状态并用`SetForcedLOD(1)`锁LOD0，不调用`SetCPUSkinningEnabled`、`GetCPUSkinnedVertices`或重评动画。`SampleVisualPose17GPULBS`读取实际VisualMesh、其当前骨pose来源和RefToLocal矩阵、组件有效LOD0 skin-weight buffer，调用UE5.8公开`USkinnedMeshComponent::ComputeSkinnedPositions`计算参考顶点的线性蒙皮位置。五头点采用已绑定真实UE三角的重心插值，十二身体点读取实际骨位置，世界坐标仍为米。

实际检查并拒绝CPU render/static、非LOD0、Nanite、未完成并行动画、非零morph权重/曲线、external morph sets、cloth、mesh deformer、不可读或可变影响数skin weights，以及基础slot或Overlay的WPO。不通过关闭这些功能伪装支持。形变支持范围保存在`gpu_lbs_eligibility`实际布尔和计数中，缺字段不能默认通过。

有效deformer资产检查使用UE5.8公开`HasMeshDeformer()`：该实现直接查询内部有效集合，覆盖组件override、资产默认及project/UBI fallback。当前引擎实现明确只支持0或1个有效deformer，故`mesh_deformer_asset_count`据该公开结果记为0或1；不通过protected访问或以实例数量代替资产判断。`mesh_deformer_instance_count`仍独立读取公开实例集合，任何一项非零均拒绝。

每个输出sample前执行全部目标的动态数据标记，排空既有和本次World end-frame异步更新，再等待render commands完成。样本记录同引擎帧及pose revision稳定性、实际CPU=false/static=false/LOD0、准备到采样实际引擎帧数。骨名与组件空间变换、RefToLocal矩阵SHA和实际skin weights SHA一并记录，矩阵/权重各自含明确signature scheme。变换使用显式`translation_cm`、`rotation_xyzw`、`scale_xyz`数组编码；不依赖FTransform文本导出的未声明语法。

恢复仅作用于同一PIE组件/mesh，恢复公开rendered/streamed LOD接口并读回，要求CPU始终关闭。streaming差异单列restart_required，不声称全部资产内部状态已恢复。原始Editor地图和角色不发生mesh切换。

## 尚待真实验收

原003 a07已完成8帧预热、原生采样及配对，主agent确认Lumen skeletal cards保持启用。CPU配对工具仍须对每个新角色验证实际GPU路径收据、相机计划位置/方向、RGB/ObjectID/depth、当前骨pose和帧证据；头点深度使用预注册5mm残差门槛，边界和无法判定遮挡保持unknown。a07的第0→1源帧17点相同，第1→2有姿态变化；不能以3帧文件数量假定每帧身体姿态都变化。

a04旧CPU参考的EngineWarmUpCount=0/RenderWarmUpCount=32与新样预热不同，不能只凭相同Sequence source frame声称骨姿态相同。只有实际骨pose/Actor/Visual变换等证据匹配，才比较旧CPU和新GPU坐标；否则以本次GPU图像/深度配对独立验证，并明确不是旧参考的严格同pose A/B。衣裤外观质量、真实GRF连续序列和003–009逐角色校准各自仍需验收。

所有`mrq_synchronization_verified`及正式visibility仍保留false/unknown；单个角色诊断通过不自动开放正式Pose监督或批量渲染。

## 单004 GPU LBS与PIE角色切换组合

新增独立scope `pose17_gpu_render_cpu_lbs_character_calibration`，要求上述GPU opt-in及`allow_pie_visual_mesh_override=true`同时成立，CPU preparation必须false，driver hide必须true。`override_warmup_frames`与`gpu_lbs_warmup_frames`必须为相同整数8..120。当前仅允许原Player003→单个Player004，目标总数为1；不自动批量推进其他角色。实际新资产及landmark manifest路径继续使用[角色切换接口](POSE17_PIE_CHARACTER_OVERRIDE.md)已有target字段。

所有目标、新mesh真实LOD0身份/锚点、共享骨架、共享动画及材质slots先只读检查，之后只在PIE切mesh，使用新asset自身材质。切换后立刻执行完整GPU eligibility，记录`gpu_lbs_validation_attempted/succeeded`、`gpu_lbs_eligibility_after_override`和`gpu_lbs_validation_error`。若后检查失败，立即按snapshot恢复原mesh/材质/动画，保留真实失败收据，不继续渲染成正式成功。

成功后才在shot中准备GPU LOD并记录prepared engine frame；它不得早于applied engine frame。每行两种warmup分别为hook减对应frame，override elapsed不得小于GPU prepared elapsed。post-override eligibility的reference/weight顶点数须与新mesh sample身份一致。teardown顺序为GPU LOD→原mesh/材质/动画→driver，Editor原始003保持不变。完整事务验收器`ue/pose17_override_receipt.py`已支持组合scope，同时仍要求全部相机、实际lookat=false、精确003/004身份、材质/动画/恢复布尔及路径、Editor快照一致和诚实streaming结果；它不替代像素/外观验收。
