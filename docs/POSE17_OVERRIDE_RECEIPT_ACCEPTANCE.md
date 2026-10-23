# Player003→004原生事务收据验收

`ue/pose17_override_receipt.py` 是独立纯CPU验收器，不导入unreal、不改runner/C++、不触发渲染或修复。已覆盖旧CPU scope及新增GPU-LBS组合/联合scope的反例。Player004首次a01在warmup阶段RHI崩溃，没有native收据，失败attempt保持原样；后续独立`player004_closeup_a02_gpu_lbs`已通过真实三近景切换/采样/恢复与头点配对，`joint_seed1048_a01_gpu_lbs`又完成七模型十实体四机位短诊断。后两者不回写旧a01，也不提升正式生产准入。

完整attempt完成且独立postflight成功后，在内层Python3.9环境执行：

```powershell
.venv/Scripts/python.exe ue/pose17_override_receipt.py <Saved/Pose17Development/smoke/完整attempt> <全新验收结果.json>
```

输入均从该attempt读取：plan/state/preflight/editor_after、冻结L0_landmark_snapshot及全部显式相机native收据。固定核对精确原Player003和新Player004对象路径，不通过角色名字片段、CPU模式或单一成功布尔代替完整身份。冻结plan SHA和manifest SHA、LOD0完整几何/拓扑签名、COCO17顺序、source/output帧集合、PIE组件与相机名称、每记录实际lookat已观测false、实际warmup≥8且与应用/采样engineframe差精确一致、cache fence/revision及17有限世界点均逐项检查。

原生事务必须记录精确共享SKEL_MPFB_Base和ABP、原动画实例路径、身份/材质/动画/应用/恢复全部通过、完整新mesh材质slot路径、原override材质与Editor快照可对应。Editor前后玩家数组及目标原003也必须完全一致。跨机位同源帧world Pose最大差不得超过预注册1e−5m。所有判定仍保持MRQ_sync_verified=false、visibility unknown；本工具不读取RGB/EXR，需继续真实像素/overlay/头语义及衣裤质量验收。

streaming独立处理：组件/动画/材质恢复成功但streaming不匹配，且native/state明确记录restart_required时，输出`component_transaction_passed=true`、`streaming_restart_required=true`、`acceptance_passed=false`，退出码2，表示需安全重启后再继续。streaming未恢复却声称无需重启则是收据矛盾，退出码1。全部原生与Editor门禁通过且无需streaming重启为退出码0，仅表示原生组件切换门槛通过。

当前收据的精度边界会写入evidence_gaps：没有独立序列化每个恢复后PIE字段值；没有显式actual CPU mode字段，因此不能作为GPU/CPU A/B证据；UObject class依赖已执行的原生强类型和runner检查，没有独立class字段。材质/动画布尔来自原生读回实现，CPU无法重新打开UE资产验证这些原生声明。不能把它们描述成所有图像/材质/全表面精度已经通过。

新增`pose17_gpu_render_cpu_lbs_character_calibration`仅单004双opt-in、CPU preparation=false、相等的GPU/override预热计划。它额外要求实际CPU=false、完整GPU eligibility、post-override eligibility成功且顶点数绑定新mesh、applied→prepared→hook时序、两种实际warmup差值及GPU endframe fence。该新scope已经显式输出actual CPU mode，旧收据缺失此字段的evidence gap不再套用到新scope；其余材质/恢复全值及像素限制保留。完整GPU/LBS资格由runner纯CPU门禁函数共同验收，不能把旧CPU cached成功记录直接换scope字符串冒充通过。
