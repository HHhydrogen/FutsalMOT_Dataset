# 独立Pose-only v3位置对照

## 已完成的第一轮独立GPU对照（2026-10-23）

root已另签新协议和准入，并完成三seed各R/RS25的1000次实际SGD更新及相同200张COCO公开dev评价。新协议SHA为`a6bf2edc26d7f7a690dcb084871ba0ec93f4fb5b3d729cf1642ebecd22a0a19c`，准入SHA为`33378233429073f208836ae5ff523a0cc64a5fe831a8b7e3e706b44fb9561b1d`。完整结果位于`D:/projects/FutsalMOT_Research/pose/poseonly_v3/control_a01/paired_summary.json`；下文模板未授权指模板本身，不能覆盖实际新冻结协议的范围。

| seed | R AP | RS25 AP | RS−R AP点 | R AR | RS25 AR |
| --- | ---: | ---: | ---: | ---: | ---: |
| 20261024 | 47.4804 | 47.6911 | +0.2107 | 56.0311 | 55.8171 |
| 20261025 | 46.4790 | 47.1930 | +0.7140 | 55.0584 | 55.4669 |
| 20261026 | 46.9649 | 47.4084 | +0.4435 | 55.3502 | 55.6226 |
| 平均 | 46.9748 | 47.4309 | +0.4561 | 55.4799 | 55.6355 |

AP配对差样本标准差0.2519点，三个seed均正；AR平均差+0.1556点、一个seed下降。AP50平均−0.4430点、medium AP−0.4954点、large AP+1.3959点，保留全部方向。六个模型的bbox AP均为53.091408，和预训练参考一致。

原预训练参考AP为49.1896，R/RS平均仍分别低2.2148/1.7588点。这说明本配置下加入合成相对同更新R有小幅位置收益，不能写成超过预训练、真实足球关联提高、真实五人制泛化或跨比赛统计证明。sigma、训练范围、BN和学习率共同改变，不能将v2到v3差异归因于单一修复；R与RS真实曝光也不同。原v2六臂完整保留。

固定真实检测的六臂Pose附件/关联也已完成，复用同一raw缓存、lambda0.1和几何配置；RS−R平均AssA+0.5524、HOTA+0.2556、IDF1+0.3961点，但RS平均AssA仅比预训练高0.0236点，主要减轻退化。完整结果与边界见`docs/POSE_ASSOCIATION_TRACKEVAL.md`。最终test116/187保持封存。

`train_pose_location_poseonly_v3.py`采用独立协议，不修改原v2训练、权重、曝光或评价文件。v3同时改变sigma、可训练范围、BN策略和学习率，改善也不能宣称其中单一因素的因果作用。以下记录实现与门禁；只有另外冻结并获root限定准入的实际协议可以执行。

## 数学与数据边界

复用同一冻结1000真实COCO图、1440候选合成位置图、其中冻结1000个S和三seed成对采样表；R每步4真实，RS25每步3个共享真实加1个S。1000次实际`optimizer.step()`、batch4、640，无增强/AMP/累积/早停/选best。R为4000次真实曝光，RS为3000次真实加1000次S；图像比例不是有效关键点监督比例。S仍保留diagnostic/not_training、MRQ_sync_verified=false、unknown visibility；准入只允许独立bool位置监督，不提升为正式Pose GT。

使用新`pose_location_supervision_v3`显式COCO17 OKS sigma；继承完整人物框和v0屏蔽位置、v1/v2已标位置有效的规则。只训练Pose `cv4`参数。共享backbone/neck、检测分支固定；整体eval但保留autograd，所有BN running mean/var/count固定，cv4 BN affine可训练。物理51通道保留，输出17×2；visibility行屏蔽梯度，SGD weight_decay=0，visibility BCE禁止。SGD固定lr=.0001、momentum=.9，位置loss仍由现有分配/归一化计算，不添加未登记的权重。

EMA为真实FP32 `ModelEMA`。每步更新可训练Pose参数后，将冻结参数、BN buffer及未输出物理visibility行直接复制回EMA，避免仅因EMA插值舍入让固定检测器漂移。随机toy验证actual和EMA的检测前五通道与初始化逐值相同；这不是已完成真实COCO/足球推理验收。

无前景或零位置梯度不删框、不跳图，仍调用一次optimizer并记录`zero_gradient_update`。已有动量可能继续改变可训练参数，这是固定SGD的正常语义；所有曝光和更新仍计入预算。

## 独立签署

新协议schema=`futsalmot_pose_location_poseonly_protocol_v3`，状态必须`frozen`，并显式training_authorized=true；新准入schema=`futsalmot_pose_location_poseonly_admission_v3`、status=`approved_poseonly_location_v3`、approved_by=`root`。准入精确绑定新协议、同一schedule manifest、新代码/隔离/loss/依赖SHA、只读source_v2_protocol/source_v2_admission及新的UE安全退出串行门禁。

旧授权不能直接启动v3。只有新v3门禁通过后，才调用旧入口的纯输入验证器复核相同R/S选择和来源证据；不调用旧训练循环或修改旧模块。新协议的初始化和arms曝光预算须与原冻结协议完全一致。模板的所有授权默认为false，不能直接训练。

```powershell
# P1只做JSON模板；不加载Torch/图片或开始训练。输出须全新。
.venv/Scripts/python.exe experiments/sim2real/train_pose_location_poseonly_v3.py template --base-protocol D:/projects/FutsalMOT_Research/pose/coco2017_location_a01/frozen_protocol_a01.json --base-admission D:/projects/FutsalMOT_UEDataset/Saved/Pose17Development/location_training_admission_a01.json --schedule-manifest D:/projects/FutsalMOT_Research/pose/location_pilot_4seed_a01/schedule_manifest.json --output D:/projects/FutsalMOT_UEDataset/Saved/Pose17Development/poseonly_v3_candidate_a01
```

主agent之后另存冻结协议和新准入，再用研究Python3.10执行`train`子命令，显式传`--protocol --schedule-manifest --admission --output --arm R|RS25 --seed`。仅允许新`D:/projects/FutsalMOT_Research/pose/poseonly_v3/`下的独立run；默认cuda:0，启动前新鲜检查UE进程，不允许写入v2目录。本文不授权实际GPU运行。

## 事务、暂停和硬崩溃

每步先fsync `started_not_committed`事务，再计算/更新，随后fsync完整曝光及SHA链，最后写committed事务。完整model、actual FP32 EMA、optimizer动量、Python/NumPy/Torch/CUDA RNG、更新计数、曝光链、来源bindings保存在`last.pt`。初始0步、每50步、最终和安全暂停保存完整断点；每个写盘使用自有tmp及有界Windows WinError5/32重试，不monkeypatch全局或活跃v2。

SIGINT只请求在更新边界暂停，不中断正在发生的优化更新；边界到达后保存最新完整数学状态。`--stop-after N`同样是累计更新暂停，不改变1000步预算。显式`--resume`要求last、曝光行数/链和最新committed事务精确一致。

若硬崩溃导致日志或事务超过last、存在未提交intent、partial或曝光缺失，恢复必须拒绝。不会截断曝光、补写一行、丢弃已发生更新或从旧last自动重算。此尝试不能再宣称同样实际更新预算，须由主agent处理独立失败记录/新attempt。保留失败现场。完整last1000再次请求train将拒绝，避免覆盖最终断点。

## 评价和最小证据

checkpoint schema=`futsalmot_pose_location_poseonly_checkpoint_v3`，run schema=`futsalmot_pose_location_poseonly_run_v3`。标准bindings的protocol/admission/trainer SHA均指新v3，同时保留source_v2_bindings、R/S JSONL及schedule身份。公开评价只接受completed_updates=ema_updates=1000的last，actual_ema_fp32=true、physical_output_channels=51、keypoint_shape=[17,2]及完整run绑定，checkpoint_choice=`last1000_actual_FP32_EMA_no_best`。旧COCO评价器不改，由独立v3评价适配核对相同dev200。

最小验证为`tests/test_pose_location_poseonly_v3_trainer.py`：新准入/修改拒绝、间隔保存计数、64px随机toy两更新、信号请求式安全暂停/恢复、实际/EMA检测输出精确不变、model/EMA/动量恢复逐值一致、所有BN和visibility固定、硬崩溃超last不重放、partial拒绝。仅CPU，不读真实媒体、不启动新GPU或UE，不接触封存test。
