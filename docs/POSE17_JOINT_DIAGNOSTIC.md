# 十角色联合GPU Pose17小样

用户要求复用已通过003/004小样的采样通路，加快全模型接入。当前不再逐005–009重复近景渲染或完整像素审计，改为一次十实体、四原相机、十连续帧的独立联合诊断。代码及必要CPU契约测试已完成；联合真实构建/UE小样由主agent执行，不能把已有003/004结果冒充全角色已验证。

2026-10-23真实执行完成：`joint_seed1048_a01_gpu_lbs`输出40张RGB与40份EXR、400实体记录，队列34.510秒；原生身份/切换/恢复、独立Editor postflight和配对报告均通过。主agent检查四张源105全员叠图，十实体均有去平移后的姿态变化。联合机位world最大差4.273760266320904e-6m，后续诊断导出保留每相机原始world，不把微小差异平均或强行共享。visibility保持unknown，正式生产及训练准入仍未通过。本机收据为外层`Saved/Pose17Development/joint_seed1048_a01_root_review.json`，不把Saved媒体纳入Git。

计划scope为`pose17_gpu_render_cpu_lbs_joint_diagnostic`，显式`allow_joint_pose17_diagnostic=true`、GPU LBS与PIE mesh override双opt-in、CPU preparation=false、driver hide=true，两个warmup字段为相同整数8..120。每target有自己的整数`calibration_player_id`、精确Actor/MPFBVisual、`Player_<entity>` Sequence绑定及真实UE manifest。固定映射如下：

| 实体 | 模型 |
| --- | --- |
| L0 / R2 | Player003 |
| L1 / R3 | Player004 |
| L2 / R4 | Player005 |
| L3 | Player006 |
| L4 | Player007 |
| R0 | Player008 |
| R1 | Player009 |

003两目标不切mesh；其余八目标只PIE切换。角色路径采用`ue/pose17_calibration_assets.py`完整白名单，不按名字、prefix或新增重复native角色ID推断身份。每条native记录已有实际mesh完整identity、manifest SHA、entity和frame，足以绑定计划。

所有十目标先只读通过真实mesh/manifest/锚点、共享骨架/AnimBP、GPU形变门禁，再逐个快照并apply八个新mesh。每次切换后继续执行完整GPU eligibility；任一partial失败逆序恢复全部已改目标并写失败收据。每sample十目标共用endframe/fence，采样与恢复顺序沿用已验证通路：全部GPU LOD恢复→全部原mesh/材质/动画恢复→driver恢复。重复004/005资产以各自component+entity区分事务。

真实GRF来源计划使用seed1048冻结source前110帧，在全新Development Sequence导入完整前缀以保留运动状态历史，仅渲染source100..109；该联合roster为显式诊断覆盖，不是1048原appearance的seed复现。原始四camera及源文件不覆盖。每job `MaxRecords=10*frame_count=100`，共400实体记录；不启动正式批量生产。

文件名使用`{frame_number_rel}`，输出为attempt相对0..9；native `OutputFrameNumber`同为0..9，`SequenceSourceFrame`及EXR `unreal/sequenceFrameNumber`为100..109。`review_frame_indices=[5]`指attempt相对帧5，即源105，不能当成GRF5。`unreal/sequenceFrameNumberRelative`保留相对索引作独立元数据核对。

联合收据验收复用同一入口：

```powershell
.venv/Scripts/python.exe ue/pose17_override_receipt.py <完整joint attempt> <全新结果.json>
```

它只读取十份冻结manifest、四份native收据、state及Editor前后快照，核对身份/帧/fence/恢复，不打开媒体。配对工具按计划只对相对中间帧5输出四张全员overlay及mask，其余帧只读header/metadata；不生成400张单人图或逐角色深度复验。所有正式visibility仍unknown，`MRQ_sync_verified=false`；该小样通过也不自动宣称正式GT、全动作范围或训练收益已验证。
