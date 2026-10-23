# 固定检测关联的独立TrackEval入口

`experiments/sim2real/evaluate_pose_association_dev.py`只读取固定开发追踪文本和已登记的派生player GT，不运行模型、不读取图片、不改旧评价或GT。计算薄层为`src/grf_ue_bridge/pose_association_trackeval_v1.py`，复用本机`D:/projects/FutsalMOT_Research/runtime_dependencies`中的TrackEval1.3.0。

本轮只允许lambda0基线和看结果前登记的唯一lambda0.1。必须先有相同cache、关联实现、协议及三个片段全部150帧的实际lambda0输出数组/轨迹状态逐值等价收据；评价器不重复运行Tracker。`run_pose_association_dev.py`输出目录包含`receipt.json`和`data/SNMOT-151.txt`等三文件，直接作为输入。

## 数据边界与无过滤

仅评估SNMOT151/152/153各前150帧，共450帧，三者均为game9开发片段。GT只允许source_manifest精确登记并绑定SHA的`round4_combined_eval/datasets/real_player_pilot_v1/mot_gt/<seq>/gt/gt.txt`，不重新从完整GT挑人或修框。派生GT需全部mark/class/visibility=1，seqinfo为1920×1080、25FPS、150帧。封存116/187在路径stat前拒绝。

实际`MotChallenge2DBox.DO_PREPROC=False`，不会根据GT匹配删除预测、过滤distractor或重做player筛选；MOT原行全部保留，不新增置信度阈值。语法门禁只拒绝非法/重复ID、越界帧、非有限值或无效框，不自动修复。每片段实际TrackEval的Dets、GT_Dets、Frames必须与原文件行数/150一致，避免隐式丢行。

既有追踪receipt没有生成时MOT内容SHA。入口首次评价记录实际三个文件SHA，并明确`original_tracking_receipt_bound_all_MOT_SHA=false`及限制，不补写原receipt、不把首次评价SHA伪称生成时已绑定。若未来生成收据提供各sequence的`mot_file_sha256`，入口会要求精确一致。

## 指标与单位

`report.json.raw_results`保存全部每片段和`COMBINED_SEQ`的TrackEval原始HOTA/CLEAR/Identity/Count结果，包括19个HOTA alpha阈值数组及各计数。官方详细CSV和summary文本也保留。`COMBINED_SEQ`由官方`combine_sequences`生成，不能用片段分数算术平均代替。

`summary.<seq或COMBINED_SEQ>`包含：

- `metrics_fraction`：HOTA、AssA、DetA按官方alpha数组取均值；IDF1、MOTA为原始比例，MOTA允许负值。
- `metrics_percent`：上述比例乘100，跨arm的差为百分点pp。
- `IDSW_count`：整数次数，不乘100，也不混入percent。

主终点为AssA，同时保留HOTA、DetA、IDF1、MOTA、IDSW以及每片段负向结果。此入口不根据结果挑lambda、删片段或修改有效性规则。不把同比赛三个片段当作独立比赛/最终测试，也不据此直接宣称五人制泛化。

## 命令

```powershell
& D:/projects/FutsalMOT/.venv/Scripts/python.exe experiments/sim2real/evaluate_pose_association_dev.py `
  --tracking-dir <bbox_lambda0或pose_lambda01目录> `
  --cache-manifest <完整raw-cache>/cache_manifest.json --cache-sha256 <固定cache SHA> `
  --protocol <本轮已登记protocol.json> --protocol-sha256 <协议SHA> `
  --lambda0-receipt <bbox_lambda0>/receipt.json --output <全新metrics目录>
```

lambda0自身评价也把自己的receipt作为`--lambda0-receipt`。追踪及lambda0 receipt的SHA默认在首次读取时计算并记录；可分别用`--tracking-receipt-sha256`、`--lambda0-receipt-sha256`显式钉住。输入cache和协议的SHA必须显式提供。所有输出使用新目录，失败保留failure及原TrackEval日志，不改变输入。

CPU测试：

```powershell
$env:PYTHONPATH = 'src;D:/projects/FutsalMOT_Research/runtime_dependencies'
& D:/projects/FutsalMOT/.venv/Scripts/python.exe -m unittest discover -s tests -p test_pose_association_trackeval_v1.py -v
```

实际TrackEval夹具使用三序列各3帧，包括低分0.001误检及ID切换；验证全部预测保留、raw数组、Count、COMBINED权重和单位。以下实际阶段结果另由主agent执行，没有用夹具分数替代研究结果。

## 2026-10-23实际阶段结果：预训练Pose对比bbox

主agent已完成共享450帧/6901个raw检测的lambda0与预训练Pose lambda0.1。缓存SHA为`3028fa8cf73737ae459afda4226bad33b55c6ef34cd817efb548ae417e6c462d`，看结果前登记的协议SHA为`52133eb43be96c96782bbc92abc0a7659d21442b817199c8e7c1ff1767e988db`。Pose模型是原COCO预训练严格xy适配权重，SHA为`c41fee957f91da4657f5b4f8822e22d53c6c527074c102e0a8c7490cf97736ec`，不是本项目真实/合成继续训练后的模型。

全部lambda0输出/状态等价门禁、各片段TrackEval预测与GT计数均已通过。实际结果保留于`D:/projects/FutsalMOT_Research/pose/association_dev_a01/metrics_bbox_lambda0/report.json`及`metrics_pretrained_pose_lambda01/report.json`；未修改原追踪receipt，生成时未绑定MOT SHA的限制仍明确保留。

下表除IDSW为次数外，均为百分数；COMBINED_SEQ是官方合并。

| 片段 | 方法 | AssA | HOTA | DetA | IDF1 | MOTA | IDSW |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SNMOT-151 | bbox lambda0 | 79.617196 | 74.316621 | 69.517408 | 89.222930 | 82.512821 | 4 |
| SNMOT-151 | 预训练Pose lambda0.1 | 79.617196 | 74.316621 | 69.517408 | 89.222930 | 82.512821 | 4 |
| SNMOT-152 | bbox lambda0 | 60.399922 | 55.870489 | 51.807915 | 72.398901 | 62.736336 | 16 |
| SNMOT-152 | 预训练Pose lambda0.1 | 62.313522 | 56.753643 | 51.801210 | 73.644933 | 62.839464 | 15 |
| SNMOT-153 | bbox lambda0 | 77.477620 | 70.598702 | 64.443744 | 87.079163 | 75.786463 | 8 |
| SNMOT-153 | 预训练Pose lambda0.1 | 77.329687 | 70.548628 | 64.477928 | 87.027765 | 76.072450 | 6 |
| COMBINED_SEQ | bbox lambda0 | 71.358459 | 65.372727 | 60.043854 | 81.162521 | 71.580907 | 28 |
| COMBINED_SEQ | 预训练Pose lambda0.1 | 72.121231 | 65.727164 | 60.048111 | 81.719279 | 71.682464 | 25 |

合并AssA增加0.762771pp、HOTA增加0.354437pp、IDF1增加0.556758pp，IDSW减少3次。收益主要来自152；151不变，153的AssA/HOTA/IDF1小幅下降，不能只报告正向片段。固定的是raw检测输入，关联后的轨迹输出及Kalman状态可以不同，因此TrackEval DetA微变不能写成原检测器AP提高。

**该结果只表明已有预训练Pose在这三个开发片段的固定检测关联上有帮助，不是合成数据有效性的证据。** 合成数据作用仍需在相同检测、初始化、关联参数和预算下比较真实R与真实加合成RS的Pose模型。当前三片段同为game9且用于开发，无独立比赛、最终封存测试或五人制泛化结论；也未独立验证几何mask的物理可见性或关节精度。

## v3六模型实际关联结果（2026-10-23）

独立v3三seed各R/RS25的1000步last实际FP32 EMA已完成官方COCO评价并逐一通过实际模型加载门禁，再在相同450帧raw缓存生成Pose附件、以固定lambda0.1关联和CPU TrackEval评价。未重新生成raw或预训练缓存，未更改关联及几何参数。完整结果：`D:/projects/FutsalMOT_Research/pose/association_dev_a01/v3_paired_association_summary_a01.json`。

| seed | arm | AssA | HOTA | IDF1 | MOTA | IDSW |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 20261024 | R | 71.327910 | 65.361654 | 81.151926 | 71.631686 | 26 |
| 20261024 | RS25 | 72.121231 | 65.727164 | 81.719279 | 71.682464 | 25 |
| 20261025 | R | 71.327910 | 65.361654 | 81.151926 | 71.631686 | 26 |
| 20261025 | RS25 | 72.121231 | 65.727164 | 81.719279 | 71.682464 | 25 |
| 20261026 | R | 72.121231 | 65.727164 | 81.719279 | 71.682464 | 25 |
| 20261026 | RS25 | 72.191915 | 65.762794 | 81.772784 | 71.682464 | 25 |

RS−R的AssA配对差为+0.793321/+0.793321/+0.070684点，均值+0.552442点、样本标准差0.417215；HOTA平均+0.255550、IDF1平均+0.396071点，IDSW差为−1/−1/0。看结果前登记的AssA均值至少+0.5点、至少2/3 seed正、HOTA均值不降及IDF1不下降超过0.5点均通过。

**这个有限开发进步主要表现为减轻纯真实继续训练的退化。** 前两seed的RS轨迹结果与预训练相同；三seed RS平均AssA仅比预训练高0.023561点。它不是稳定超过预训练、跨比赛或真实五人制效用的证据，也不能把三个训练seed看作三场比赛。R为4000真实曝光，RS为3000真实+1000合成，实际更新相同但真实曝光不同；因此仍需等真实曝光及真实梯度归一化控制，排除单纯少做真实继续训练的解释。所有每片段原始指标与负向变化保留，未调最终test。
