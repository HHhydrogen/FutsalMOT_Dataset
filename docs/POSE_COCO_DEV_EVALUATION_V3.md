# 独立v3 Pose-only开发评价

入口`experiments/sim2real/evaluate_coco_pose_location_v3.py`，身份适配模块`src/grf_ue_bridge/coco_pose_dev_eval_v3.py`。它不修改旧`coco_pose_dev_eval.py`、旧CLI、Trainer或协议。推理、17xy评分策略、LetterBox、NMS、官方COCOeval与辅助位置计算直接调用旧实现，并将旧数值文件SHA锁为`c5f2cf886cd6549385b82504e0a5cdfab2fa16242a1ba8315fb984daef84e7ee`。旧实现变化时拒绝执行，不能把协议版本变化变成评价计算变化。

## 身份链

v3使用独立schema：协议`futsalmot_pose_location_poseonly_protocol_v3`、准入`futsalmot_pose_location_poseonly_admission_v3`、checkpoint`futsalmot_pose_location_poseonly_checkpoint_v3`、运行收据`futsalmot_pose_location_poseonly_run_v3`。原v2文件不能直接作为v3输入。必须用新协议实际SHA，不伪装成旧SHA。

验证链为：新冻结协议及root准入 → `source_v2_protocol/source_v2_admission`实际文件SHA → 旧冻结协议 → 原候选与COCO准备收据 → 相同的200张公开val ID和GT。R/S JSONL及各seed的原共享采样表也核验实际SHA，源码清单与准入一致。初始化及R/RS曝光预算保留原值，v3新`paired_training`与policy另行验证（1000更新、640、SGD lr1e-4、cv4-only、BN固定、显式COCO sigma、FP32 EMA）。这不声称学习率、冻结及sigma是单一因素实验。

这里只读取所需JSON/JSONL、源码和已选checkpoint，不打开图片来验证身份。所有图片推理仍由显式checkpoint评价命令触发。封存test边界、整图crowd排除偏差、公开dev不能作最终测试等限制沿用旧入口。

## 最终模型门禁

只接受实际`last.pt`的EMA，完成更新、计划更新和EMA计数均为1000，并与完整运行收据的策略、arm/seed、实际曝光数、初始化状态和曝光链匹配。检查标准bindings与`v3_`别名都指新协议/准入/Trainer，`source_v2_bindings`及R/S/采样表仍精确指原来源。

v3 checkpoint的choice为`last_actual_FP32_EMA_no_best`，完成run的choice为`last1000_actual_FP32_EMA_no_best`，分别按真实writer验证。`actual_ema_fp32=true`之外，还在任何`.float()`转换之前检查实际EMA全部浮点state确为FP32；实际头需保留物理51通道并严格索引为17×2。不能把FP16先转FP32后声称原EMA就是FP32。不重建YAML、不挑best、不读取GT来选择预测。

可用`--checkpoint-role pretrained_reference`在同一v3开发绑定下评价原官方严格xy初始化，数值策略完全相同。历史v2预训练报告的GT/数值相同并不等于其收据自动变成v3；缓存收据必须有正确v3schema与新身份链。

## 命令与输出

```powershell
& D:/projects/FutsalMOT/.venv/Scripts/python.exe experiments/sim2real/evaluate_coco_pose_location_v3.py `
  --protocol <独立v3冻结协议.json> --protocol-sha256 <新协议SHA> `
  --admission <root新v3准入.json> --admission-sha256 <新准入SHA> `
  --checkpoint <本arm>/last.pt --checkpoint-sha256 <last实际SHA> `
  --run-receipt <本arm>/run_receipt.json --run-receipt-sha256 <完成收据SHA> `
  --device cpu --output <全新v3评价目录>
```

默认CPU，GPU仍需显式cuda:0且通过旧推理函数的UE串行门禁；不得在其他GPU作业并行时擅自启动。输出独立v3 predictions receipt/report及官方日志，原始关键点第三列仍统一广播person预测分数，不是物理visibility或独立关节confidence。主指标AP/AR与辅助bbox、覆盖率、位置误差完全复用旧函数，不改GT v/area或点顺序。

CPU夹具验证新旧协议拒绝、来源SHA篡改、预算/别名/实际FP16拒绝、同一夹具的官方评价数字逐值相同。没有真实v3 checkpoint/开发图像推理或GPU结果，本文件不授权训练。
