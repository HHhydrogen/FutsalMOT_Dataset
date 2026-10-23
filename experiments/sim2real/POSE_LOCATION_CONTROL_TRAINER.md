# 固定更新的位置监督研究入口

`train_pose_location_control.py`是独立研究循环，消费floor准备的R/S JSONL和成对采样manifest，不生成第二套选择规则、不修改既有toy Trainer，不读取SoccerNet final test。当前CPU验收仅证明训练与续跑工程；不能据此宣称合成数据提高AP或MOT。

## 准入与冻结

候选协议本身`training_authorized=false`，不能直接启动。每次运行先要求主Agent另行签署有限位置准入收据，并实际核对冻结协议、schedule manifest、R/S JSONL、正常退出收据及源候选证据的SHA和内容。root准入与源的`diagnostic_only/not_training/MRQ_sync_verified=false`分开保存，源标志不翻转。

准入模板（由root在真实验收后填写，工具不自动签署）：

```json
{
  "schema": "futsalmot_pose_location_training_admission_v1",
  "status": "approved_location_only",
  "approved_by": "root",
  "training_authorized": true,
  "location_only": true,
  "location_alignment_accepted": true,
  "visibility_supervision": false,
  "source_not_training_flags_preserved": true,
  "bound_candidate_protocol_sha256": "b2946b58b8e2b10510f185815018e698cee25b705433590494ae3913f7d15c33",
  "media_policy": "export_mask_decode_and_text_evidence_no_media_rehash",
  "protocol": {"path": "实际冻结协议", "sha256": "实际SHA"},
  "schedule_manifest": {"path": "实际采样manifest", "sha256": "实际SHA"},
  "real_jsonl": {"path": "实际COCO train JSONL", "sha256": "实际SHA"},
  "synthetic_jsonl": {"path": "实际1440图S train JSONL", "sha256": "实际SHA"},
  "shutdown_receipt": {"path": "实际退出收据", "sha256": "实际SHA"},
  "source_evidence": {"path": "实际四seed位置候选收据", "sha256": "实际SHA"},
  "UE_serial_gate": {"all_rendering_finished": true, "editor_safely_exited": true}
}
```

正常退出收据须`normal_exit_confirmed=true`；不要求凭空填写exitcode=0。源候选收据须包含四seed、480图/4800人物各组的真实native/投影文本检查通过以及保留原诊断标志。每次只读这些绑定文本和JSONL，不重做原native/EXR审计。程序启动时另做新鲜只读Editor进程门禁，不自行关闭UE。

R的图像SHA由既有Loader初始化时验证一次；后续曝光复用确定性缓存。S没有媒体SHA，只有上述root限定媒体策略才允许其Loader关闭图像SHA验证；不生成伪SHA，不关闭R检查，不重复扫S媒体。

## 预算与位置损失

本入口仅实现Ultralytics8.4.60、PyTorch2.6.0+cu124，单设备、FP32、SGD(lr=.001,momentum=.9,weight_decay=0)、batch=nbs=4、无累积、恒定学习率/无warmup、确定性640中心LetterBox。复用已发布LocationPoseDataset、exact_full_conv_then_select头和separated_location_loss；固定DFL投影不训练，梯度裁剪max_norm=10，使用同一EMA规则，两arm完全相同。EMA及所有优化设置均保存，不通过early stop或best挑模型。

每个训练seed的两个arm消费同一1000行采样表：三真实槽各完整覆盖1000真实ID一次；R第四槽再次覆盖同一ID集合，RS25第四槽为冻结1000个S各一次。因此R=4000R、RS25=3000R+1000S，实际更新均1000。相同不同真实图像集合与计算预算得到控制，真实曝光次数不同，不能宣称两者同时相等。

COCO v0点只屏蔽位置损失，人物框/类别仍保留；v1/v2画内点保留位置。S的unknown与独立17bool位置mask分开，不能伪造COCO v。物理51通道卷积仍保留，输出只取34个xy行；未输出的visibility行梯度屏蔽、逐步核对模型值不变，EMA也精确保持这些行。visibility BCE一旦执行立即报错。

曝光日志每更新记录实际R/S原始index、来源image key/路径、person框数、有效/全缺点人数、有效点数、实际前景anchor数、被分配有效点数、effective_pose_weight、loss及LR。`effective_pose_weight`明确定义为上游pose gain乘前景中至少一个有效点的比例；不是更改loss权重，保留全缺点person造成的实际归一化影响。没有前景时为0。

## 运行与续跑

以下命令只是入口示例；只有真实root准入文件存在且内容通过后才可执行：

```powershell
D:/projects/FutsalMOT/.venv/Scripts/python.exe experiments/sim2real/train_pose_location_control.py `
  --protocol D:/projects/FutsalMOT_Research/pose/coco2017_location_a01/frozen_protocol_a01.json `
  --schedule-manifest D:/projects/FutsalMOT_Research/pose/location_pilot_4seed_a01/schedule_manifest.json `
  --admission D:/projects/FutsalMOT_UEDataset/Saved/Pose17Development/location_training_admission_a01.json `
  --arm RS25 --seed 20261024 --device cuda:0 `
  --output D:/projects/FutsalMOT_Research/pose/runs/RS25_seed20261024_a01
```

R及其余两个训练seed分别运行，禁止自动选择GPU或DDP。默认总缓存3072MiB，R缓存1000张；RS将上限分给1000R和1000个选中S。缓存的是原图确定LetterBox后的uint8张量，返回clone防止修改缓存；记录预热时间、大小/淘汰数。不够内存时可显式`--cache-mib 0..4096`，不改变样本或增强。

新运行必须为新输出目录。每次完成更新以临时文件加os.replace保存完整FP32`last.pt`，包含model、EMA、SGD动量、RNG、实际完成更新索引、初始权重state SHA及协议/数据/代码绑定。记录checkpoint耗时以便后续判断I/O开销。

Windows第二seed曾在完整`last.pt.tmp`写入后发生`os.replace` WinError5。独立`run_pose_location_control_windows.py`只对指定输出目录的三个自有临时文件重试WinError5/32，间隔为.05/.1/.2/.4/.8秒；不删除目标、不绕过权限、不修改原Trainer数值代码或source SHA，重试仍失败就保留现场退出。调用方式是在原参数前使用该入口替代`train_pose_location_control.py`，其余参数及`--resume`不变。包装器SHA另记入驱动收据。

只有已完整保存的临时检查点、真实优化/EMA计数、全部有限状态、源绑定及最后曝光链逐项一致，主agent才可显式提升临时文件并补齐其原始曝光，然后从已完成索引继续；不能据一个文件名就重放未知更新。本次实际恢复第265步、优化重放0次，原last已备份。冻结v2 loss仍保持原样；17×2导致uniform sigma的差异及独立COCO sigma/pose-only候选见`docs/POSE_LOCATION_V3_CANDIDATE.md`，不把候选修正混进正在运行的v2对照。

Ctrl+C会延后到当前更新保存完成后安全暂停，或可用`--stop-after N`在累计N步暂停；二者不改变总1000预算。继续同命令加`--resume`，禁止重建优化器或从新epoch重置预算。代码、SHA、arm/seed、曝光日志链及EMA计数必须一致。

`update_transaction.json`在优化更新前写started，checkpoint和曝光日志持久化后写committed。若硬崩溃留下尚未提交的更新，无法知道优化是否部分执行，程序拒绝自动重放并伪称严格预算，现场交root处理；若checkpoint已提交而日志仅差最后一行，可从checkpoint最后原始曝光恢复这一行。只有completed_updates=1000时run_receipt.complete才为true。

评估使用最后1000步的EMA，不选best。`last.pt`schema为`futsalmot_pose_location_control_checkpoint_v1`，model/ema都是实际exact_xy PoseModel；先导入`grf_ue_bridge.pose_xy_head_transfer`再torch.load(weights_only=False)，不从YAML重建头。此入口不做AP评估，也不执行未冻结测试集。

## 定向CPU证据

`tests/test_pose_location_control_trainer.py`使用自绘toy图和随机初始化小Pose模型，只做两步CPU更新，比较连续运行与1步暂停后恢复：model、EMA和SGD动量逐值相同，visibility loss为0、未输出行保持、全部缺点人物框仍在、原始曝光链连续。另覆盖root准入/实际SHA/退出与源证据拒绝、固定采样曝光预算、缓存clone及未提交更新拒绝。它不启动正式R/RS训练，不宣称GPU路径或AP已验收。
