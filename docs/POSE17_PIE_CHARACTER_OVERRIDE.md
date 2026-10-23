# 逐角色PIE动态校准接口

最新结果：新GPU-LBS组合`player004_closeup_a02_gpu_lbs`已完成真实三近景及头点/恢复验收，随后`joint_seed1048_a01_gpu_lbs`一次覆盖003–009七模型、十实体、四机位十帧。按用户要求不再逐005–009重复近景；仍不代表正式Pose/visibility准入。下段“尚未通过”仅描述旧CPU失败时点，不是当前GPU-LBS状态；当前入口见[联合诊断](POSE17_JOINT_DIAGNOSTIC.md)。

状态：旧CPU角色override C++已实际构建，Player004首次a01在8-frame warmup中发生真实RHI LocalVF/View uniform buffer不匹配崩溃，没有媒体或native收据；随后003无override、相同8warmup也真实复现，角色切换不是必要条件。角色切换**尚未**真实验收通过，禁止去掉warmup/屏蔽校验/关闭Lumen假装修复。原003的[保留GPU渲染的独立CPU LBS路径](POSE17_GPU_RENDER_CPU_LBS.md) a07已实际完成；新增GPU-LBS单004组合scope已完成203项相关CPU回归，尚待主agent统一构建及真实小样。组合scope不会自动批量推广003–009。

## 计划字段

在独立smoke计划增加：

```json
{
  "allow_pie_visual_mesh_override": true,
  "diagnostic_scope": "pose17_character_calibration",
  "override_warmup_frames": 8,
  "allow_pie_cpu_preparation": true,
  "allow_pie_driver_hide": true,
  "targets": [{
    "entity_id": "L0",
    "actor_object_name": "<当前L0精确UObject名称>",
    "component_object_name": "MPFBVisual",
    "sequence_binding_name": "Player_L0",
    "diagnostic_visual_mesh_override": "/Game/FutsalMOT/Characters/MPFB/Players/Player_004/Mesh/MPFB_Player_004.MPFB_Player_004",
    "landmark_manifest": "<本地Player004实际UE候选manifest路径>"
  }]
}
```

warmup只允许整数8..120；一次仅一个override，003–009以外拒绝。原有相机、3帧以上范围、显式候选授权等smoke门禁仍须保留。runner读override资产真实LOD0几何/拓扑身份，逐项与该角色manifest比较；不拿当前Editor的003身份冒充004。C++在PIE setup独立再次验证新资产、真实有序三角、权重、五头点顺序及12骨。没有override字段时不切mesh。

## 事务与证据

所有相机及全部目标先只读通过，随后才开始PIE mutation。新旧mesh必须共享同一`SKEL_MPFB_Base`资产，组件已经运行`ABP_MPFB_RuntimeRetarget_C`且保持同一AnimInstance/Leader/attach/tick。物理、未完成并行动画、额外未知overlay或不兼容postprocess动画会拒绝。不会通过重设Blueprint、Editor Actor、动画类或新建替代Actor绕过门禁。

快照原mesh、OverrideMaterials、动画实例/类/模式、postprocess实例、Leader、attachment、tick、手工morph曲线和新旧streaming状态。使用`SetSkeletalMesh(new,false)`保留共享动画，再`EmptyOverrideMaterials`使用新角色自身材质slots，逐项读回，避免把003材质应用到004不同UV。runtime每帧继续检查该绑定。warmup通过MRQ明确EngineWarmUpCount设置并读回；记录实际apply到采样的引擎帧数，未达到预注册最小值不产生成功Pose。

恢复按CPU→原mesh/材质/morph→driver进行；所有读回与失败记录到native收据，setup部分失败也尝试恢复并保留partial失败事实。streaming单独记录，不擅自LinkStreaming或冒称完全恢复。PIE对象快照在teardown清空，最终收据只有路径/数值，不引用已销毁World。Editor原003从未切换，独立postflight要求mesh、材质、动画与原始快照一致。

native收据新增`visual_mesh_override_requested`、`diagnostic_scope`、`pipeline_setup_error`及`visual_mesh_overrides`；每个事务包含原/新asset、共享骨架、动画类/原实例路径、新旧材质路径、身份/应用/恢复/streaming结果。记录增加Actor和Visual世界FTransform（位置UE厘米，旋转/scale使用原生FTransform编码）、CineCamera实际lookat bool及是否成功观测标志。bool只读，不为采样修改地图。

## PNG与质量边界

runner改用项目已正式验证的`FutsalMoviePipelinePNGOutput`，只输出精确FinalImage RGB；类不可用即failclosed。EXR不变，保留RGB/ObjectID/depth。后续004实测仍需核验每相机3张4K RGB及3份EXR、原生收据、配对/overlay和Editor postflight。

头锚点深度残差通过不代表角色外观质量通过。a04衣裤露肤/破碎区域仍需独立skin CPU/GPU A/B调查，不用Pose或材质slots读回掩盖。当前全部visibility仍unknown，不启用正式COCO监督。

## 后续最小CPU/GPU A/B方案（本次未实现）

另加显式`pose17_skinning_ab_control` scope及`control_render_only`开关。GPU控制设置`prepare_cpu=false`，仍保留hide-driver、相机设置、native生命周期收据与postflight；不调用CPU fence或头部采样，记录`control_only=true,pose_sampled=false`，绝不伪造17点成功。控制记录只含实际CPU mode、Actor/Visual变换、组件空间骨名/transform供A/B同pose核对。

该模式需独立完成状态/收据schema或明确分支，完成应为“控制渲染完成待A/B验收”，不能被现有Pose采样成功门槛误报故障，也不能被Pose导出器当GT。CPU对照须使用同一新mesh、其自身材质、同Sequence/相机/AA/warmup，仅蒙皮模式不同；比较实际骨pose与变换后，再比较腰髋/衣裤RGB或mask。父agent统一执行，当前没有新增或运行GPU-control。
