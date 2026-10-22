# 合成数据对真实足球模型的迁移实验

本实验检验：固定真实标注预算和优化步数时，加入合成五人制足球数据能否提高真实足球上的检测与多目标追踪表现。结果可能提升、下降或无明显变化，不能预先宣布有效。

## 数据与隔离

本轮正式合成数据为8个seed、八天气、32个连续4K视角序列，共20,116候选RGB；全部seed保持train，号码和Pose关闭。选择协议冻结在paper_pilot_protocol.json，不按模型结果选场景。每seed所有机位和邻帧保持同一集合。

真实来源是SoccerNet Tracking，仅保留player/goalkeeper并统一为player：
- train：SNMOT-060、SNMOT-097，共300帧，按比赛隔离。
- val：SNMOT-151，150帧。
- final test：SNMOT-116、SNMOT-187，保持封存，不用于本轮调参。

这不是官方SoccerNet全类别榜单。真实框为人体框，合成框为可见实例像素框；标注域差异需要报告。尚无真实五人制标注，公开足球结果不能直接证明真实五人制追踪收益。

## 对照与预算

| 组 | 每epoch600样本 | 独立真实标注 |
| --- | --- | ---: |
| A pretrained | 不微调COCO权重 | 0 |
| B real_only | 300真实图各重复2次 | 300 |
| C real_plus_synthetic | 同300真实图＋300合成图 | 300 |
| D synthetic_only | 600合成图，可选 | 0 |

B已完成，保留旧权重和receipt。C从同一官方yolo11n.pt开始，使用相同单类检测头、seed20261021、batch4、640输入、5epochs、SGD、750次实际优化更新；nbs=batch、warmup=0、AMP关闭，不以early stopping改变预算。评估last.pt，不按验证集波动改变训练长度。

C的真实曝光次数是B的一半，该差异明确报告。合成300张按seed/相机均衡分配，在完整连续序列上按时间分层中点抽样；不截短正式MOT序列。快照保存候选列表、选中帧号、来源和配置SHA；已经冻结的C不因数据集后续增长而重新选帧。

## 同设备评测

A和B用新名称在同一GPU重新评估，历史CPU/GPU混合结果保留为历史记录。C完成训练后使用相同real_val、尺寸、ByteTrack及阈值。

检测报告AP50/AP50–95、Precision、Recall。固定ByteTrack使用conf=0.1、NMS IoU=0.7，逐真实序列重置追踪器；TrackEval报告HOTA、DetA、AssA、IDF1、MOTA及IDSW。检测改进不等于身份关联改进。

当前环境保持Ultralytics8.4.60、PyTorch2.6.0+cu124，不为本轮比较升级依赖。ByteTrack配置显式指定，不能依赖版本的默认追踪器。框架用法参考[训练文档](https://docs.ultralytics.com/modes/train/)和[追踪文档](https://docs.ultralytics.com/modes/track/)。

## 数据准备和运行

唯一实现是本目录pipeline.py。研究根D:/projects/FutsalMOT_Research保存公开数据、派生标签、权重和结果；训练Python位于D:/projects/FutsalMOT/.venv，不修改P1的Python3.9环境。

规范输入G:/FutsalMOT_Dataset/v1_no_numbers_4k，支持seed_flat_v1和旧多片段布局。当前基准full、后续quick/basic准入如实保留；本轮明确使用structure检查复用发布证据，不重复读取媒体SHA或像素，不声称全部full审计。

YOLO和MOT使用--reference-images，通过Windows目录联接直接读取G盘正式RGB，不跨盘复制整套4K。训练清单保留D盘images词法路径，让YOLO找到D盘labels；不能resolve到G盘rgb。正式图像不可原地编辑，删除联接不能递归跟随其目标。

```powershell
./run_serial_pilot.ps1                  # 只打印计划
./run_serial_pilot.ps1 -Phase Readiness # 8组清单与结构、准入、标定核验
./run_serial_pilot.ps1 -Phase Prepare   # CPU标签派生与C/D来源、预算冻结
./run_serial_pilot.ps1 -Phase Gpu       # 同GPU A/B重评 -> C训练 -> C评测 -> 汇总
```

Gpu阶段拒绝仍存在的UnrealEditor。全部渲染结束并安全关闭UE后运行；任何同名输出已存在都拒绝覆盖，从经过receipt核验的未完成步骤继续。固定B输入和receipt不重写，final test和可选D不会由默认Gpu阶段自动运行。

输出：runs/<arm>_seed20261021_e5，evaluations、tracking、tracking_metrics和comparison。compare要求各组同设备、尺寸、真实来源和追踪器凭证一致；C未完成不能生成伪造对照结论。

## 论文证据边界

本轮是小样本pilot，不足以单独支撑稳健发表结论。后续扩大独立比赛、至少多个训练seed、真实数据预算和合成规模曲线，进行天气/外观消融并报告均值方差。真实五人制标注到位后再验证目标场景。号码的身份恢复收益须单独实验，不能由无号码检测训练代替。
