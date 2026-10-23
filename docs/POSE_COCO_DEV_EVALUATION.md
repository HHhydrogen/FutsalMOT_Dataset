# 固定COCO开发集的位置头评价

入口为`experiments/sim2real/evaluate_coco_pose_location.py`，计算代码为`src/grf_ue_bridge/coco_pose_dev_eval.py`。只接受当前协议SHA、准备收据链和精确200个`val2017` ID；读取原始关键点标注子集，不把训练用`location_valid`伪写成COCO v。它不签署训练准入、不冻结候选协议、不读取SNMOT116/187或任何test目录。

当前准备目录是`D:/projects/FutsalMOT_Research/pose/coco2017_location_a01`。原候选协议仍保留候选状态；预训练参考可以直接绑定该候选。正式last评价必须绑定单独的root冻结协议及实际SHA，沿`root_freeze.candidate_protocol.{path,sha256}`核验原候选，再核验COCO准备收据中原候选SHA；**不重写旧准备收据成冻结SHA**。冻结协议需`training_authorized=true`、`root_freeze.approved_by='root'`，且real_selection/data_root/initialization/paired_training_candidate/arms/development_evaluation_candidate与原候选一致。可新增S、采样表与准入收据绑定，但不能借冻结步骤更换dev IDs或对照预算。报告分别记录实际冻结协议SHA与原候选SHA，不把二者混用。

## 指标与置信度

主指标由官方`pycocotools.COCOeval(iouType='keypoints')`计算，输出AP、AP50、AP75、APM、APL及对应AR，标准OKS sigma与`maxDets=20`保持官方默认。官方`-1`保留，表示该面积组无适用目标，不改为0。没有pycocotools时异常退出并写失败记录，绝不仿算AP。研究Python3.10现有pycocotools2.0.11；CPU数值夹具已真实调用官方实现，完美预测AP=1、空预测AP=0只证明夹具与入口，不是实际模型结果。

模型只有17×2坐标输出。每个预测人的检测分数统一复制到17个点的结果第三列，作为固定的预测分数载体；**不是可见性v，也不是校准过的独立关节置信度**。不读取GT v来生成预测，不使用旧物理51通道中未训练的visibility行。官方评价只用GT v>0的位置计算OKS。真实`num_keypoints=0`人物继续保留，其关键点ignore行为由官方实现决定，检测框辅助评价仍计入这些人。输入COCO关键点area保持原值；bbox辅助评价若有准备阶段严格join的instances area，使用独立GT拷贝恢复该area，不修改关键点评价。

辅助指标包括官方bbox AP/AR、按检测分数排序且bbox IoU≥0.5贪心一对一配对后的像素L2、按原COCO关键点area平方根归一化误差、PCK，以及全部GT人物框/有标注点的匹配覆盖率、全缺点人数和尺度分布。位置误差是匹配条件下的统计，必须与漏检人数及覆盖率共同阅读；不能靠漏掉难例得到看似更低的误差。bbox评价使用同一最多20个预测，虽COCO bbox AR100参数保留，实际输入仍受20上限约束。

本开发子集只有200张且整图排除了crowd，不能当作完整COCO val或独立最终测试。初始化已经COCO预训练，结果属于固定子集上的继续训练。Pose AP包含检测质量变化；不能据此宣称AssA、HOTA或足球追踪提升。

## 固定推理与模型选择

默认CPU，只有显式`--device cuda:0`才请求GPU，并先只读检查Windows没有UnrealEditor进程。调用者负责在渲染完整结束和Editor安全退出后安排GPU。无自动设备选择、下载、AutoBackend、fuse、TTA、AMP或half。每张图只读一次并核对准备收据SHA/尺寸；只打开精确val图片，不读train图像。

输入为与训练一致的640中心LetterBox，OpenCV双线性缩放、整数padding、填充值114、RGB/255。NMS采用本机Ultralytics8.4.60，confidence=0.001、IoU=0.7、单类、每图最多20人。预测框回原图后截断；关键点保留有限的画外amodal坐标，不截成边界点，也不改变unknown状态。这一固定策略用于全部训练arm和预训练参考，不额外挑阈值。

正式对照只读`last.pt`中的实际FP32 EMA PoseModel。必须是完成的1000步，并核验`completed_updates`、`planned_updates`、`ema_updates`、协议与采样表SHA、arm/seed、曝光链及完成运行收据的完整bindings。要求实际保留51通道卷积再索引xy的严格适配，不从YAML重建34通道塔。`torch.load(weights_only=False)`只在实际文件SHA等于调用者明确钉住的可信本地checkpoint后执行；不能用SHA替代对文件来源的信任。预训练参考只接受协议中的官方或已绑定的严格xy权重SHA，并统一转换成同一xy输出策略。

## 命令

在内层目录使用现有研究环境；以下变量值由固定文件/收据填入，不自行选择best或更换dev ID。

```powershell
$researchPython = 'D:/projects/FutsalMOT/.venv/Scripts/python.exe'
$protocolPath = 'experiments/sim2real/pose_location_coco_candidate_protocol.json'
$protocolSha = 'b2946b58b8e2b10510f185815018e698cee25b705433590494ae3913f7d15c33'

# 预训练参考：同一17xy输出、相同分数策略；显式CPU。
& $researchPython experiments/sim2real/evaluate_coco_pose_location.py `
  --protocol $protocolPath --protocol-sha256 $protocolSha `
  --checkpoint D:/projects/FutsalMOT_Research/weights/yolo11n-pose_xy_location_exact_a01.pt `
  --checkpoint-sha256 c41fee957f91da4657f5b4f8822e22d53c6c527074c102e0a8c7490cf97736ec `
  --checkpoint-role pretrained_reference --device cpu --output <全新评价目录>

# R/RS25：使用Research下实际frozen协议和它的SHA；不得沿用候选SHA。
# run_receipt与last必须完成同一1000步，采样表SHA从冻结收据取得。
& $researchPython experiments/sim2real/evaluate_coco_pose_location.py `
  --protocol D:/projects/FutsalMOT_Research/pose/coco2017_location_a01/frozen_protocol_a01.json `
  --protocol-sha256 <冻结协议实际SHA> `
  --checkpoint <本arm目录>/last.pt --checkpoint-sha256 <last实际SHA> `
  --run-receipt <本arm目录>/run_receipt.json --run-receipt-sha256 <运行收据SHA> `
  --schedule-manifest-sha256 <冻结采样表清单SHA> --device cpu --output <全新评价目录>
```

输出`predictions.json`、`predictions_receipt.json`、`report.json`和官方原始统计日志。全部200张必须记录processed ID，包括零检测图。缓存重评使用`--predictions`、`--prediction-receipt`及`--prediction-receipt-sha256`，仍重验完整开发集绑定、预测文件SHA、固定后处理和点分数策略。该模式验证收据一致性，不能鉴别有人伪造了整个预测及收据；公开结果仍应保留首次推理的checkpoint和日志。

所有输出要求全新目录，失败保留`failure.json`。CPU测试：

```powershell
$env:PYTHONPATH = 'src'
& $researchPython -m unittest discover -s tests -p test_coco_pose_dev_eval.py -v
```

测试仅使用数值夹具和自绘小图；没有读取实际dev图像、真实test、运行真实模型GPU或宣称研究AP。它覆盖元数据篡改/ID泄漏、缺失零检测图、伪可见性、完成预算门禁、奇数LetterBox、真实CPU NMS布局和官方COCOeval的完美/空预测。
