# 保留COCO预训练Pose容量的仅xy适配

独立工具 `src/grf_ue_bridge/pose_xy_head_transfer.py`，仅验证Ultralytics8.4.60普通三尺度、非end2end、未融合Pose/PoseModel，不支持Pose26、DDP或任意版本。P1不安装Torch，模块按调用延迟导入研究依赖。

## 原问题与实际失败

本机 `Pose.__init__` 使用 `c4=max(ch[0]//4,nk)`。YOLO11n的COCO17三维头nk=51，Pose塔隐藏宽度51；直接按17×2 YAML重新建模，nk=34且隐藏宽度变为34，改变容量并使大量预训练权重shape不匹配。这会混淆合成数据收益。

第一种实现deepcopy实际模型，保留隐藏51，末端卷积仅裁取每点x/y行 `[0,1,3,4,...,48,49]` 到34，权重逐项等于源行。但随机CPU实际测试发现：改变卷积输出通道数后原始xy不再普遍bitwise相等，因浮点kernel路径变化；三种feature尺寸的最大raw差2.98e-8至4.47e-8、decoded差7.63e-6至1.53e-5px。该模式名为 `compact_xy_rows`，只保留作明确诊断，不宣称严格输出一致。

## 默认严格模式

`map_pose_xy(actual_model)` 默认 `exact_full_conv_then_select`：

1. deepcopy实际checkpoint模型，保持backbone、检测头、Pose隐藏层、BN及完整末端51通道卷积的全部权重与计算形状。
2. 每尺度末端变为 `Sequential(original_conv51, SelectPoseXYRows)`；索引层只做 `index_select`，暴露34个xy输出，没有额外可训练参数。
3. 设置实际头 `nk=34`、`kpt_shape=(17,2)`，完整模型yaml同步记录17×2与版本标记。禁止按二维YAML重新创建隐藏34的Pose塔。
4. 三维visibility行仍物理保存在末端卷积内且参与卷积运算，但不暴露于输出、不进入visibility loss。仅xy梯度测试显示这些行梯度为零；优化器weight decay仍可能作用于整块参数，但不会通过被丢弃行影响xy。不能描述为末端物理卷积已经缩成34。
5. 不改变输入dtype/device，不消耗随机状态；不传路径触发自动下载或重建。已映射模型只在模式一致且标记/容量一致时接受，否则拒绝。

```python
from grf_ue_bridge.pose_xy_head_transfer import map_pose_xy, pose_xy_trainer_class
adapted = map_pose_xy(actual_pose_model)
# 未来训练必须显式使用保持实际网络结构的trainer；此处不自动启动训练。
trainer_type = pose_xy_trainer_class()
```

`pose_xy_trainer_class().get_model` 只接受实际完整PoseModel权重，拒绝weights=None，避免标准PoseTrainer依据二维YAML重建。后续自绘toy夹具已通过单进程CPU完整Trainer、SGD两次分阶段更新及真实resume验证，范围和限制见[CPU兼容小样](POSE17_XY_TRAINER_CPU_SMOKE.md)；未验证真实数据训练、增强、GPU或DDP。自定义索引层有稳定模块名与延迟类解析，已验证新进程通过官方YOLO loader加载独立派生checkpoint。

## 随机模型与实际官方权重验证

随机模型：8个CPU unittest通过，覆盖末端行/隐藏权重保留、三种raw/decoded路径、检测输出相等、完整随机YOLO11n、内存checkpoint、随机状态不变、Trainer拒绝重建、visibility行零梯度。另在独立脚本上三种feature尺寸均严格raw/decoded bitwise相等，并通过新进程checkpoint验证。compact模式的非bitwise结果保留在 `Saved/ResearchDiagnostics/pose_xy_transfer_random_cpu_a01/receipt.json`，没有通过修改精确门槛掩盖。

在root后续授权下，从[Ultralytics官方v8.4.0资产release](https://github.com/ultralytics/assets/releases/tag/v8.4.0)下载了 `yolo11n-pose.pt`。官方[YOLO11说明](https://docs.ultralytics.com/models/yolo11/)列出该Pose模型，[官方模型卡](https://huggingface.co/Ultralytics/YOLO11/blob/main/README.md)说明Pose模型使用COCO预训练。来源API是 `https://api.github.com/repos/ultralytics/assets/releases/tags/v8.4.0`，资产ID340060946，大小6255593bytes，下载SHA与GitHub发布digest完全一致：

`869e83fcdffdc7371fa4e34cd8e51c838cc729571d1635e5141e3075e9319dc0`

源文件 `D:/projects/FutsalMOT_Research/weights/yolo11n-pose.pt` 未改动，原始及适配模型在随机输入128×128、128×160、192×256上：raw xy、decoded xy、bbox及person score均bitwise相等，最大差0。派生文件保存到 `yolo11n-pose_xy_location_exact_a01.pt`，SHA：

`c41fee957f91da4657f5b4f8822e22d53c6c527074c102e0a8c7490cf97736ec`

派生文件新进程经官方 `YOLO(path,task='pose')` loader加载后输出仍bitwise相等。收据 `Saved/ResearchDiagnostics/pose_xy_official_cpu_a01/receipt.json` 与研究weights下相邻provenance.json记录来源、源/派生SHA及验证结果。[许可证来源](https://www.ultralytics.com/license)与[官方LICENSE](https://github.com/ultralytics/ultralytics/blob/main/LICENSE)仅作来源记录，本轮未分发模型或媒体。

上述等价验证没有GPU、优化器更新、训练、真实图像读取或最终test评估；另行记录的toy Trainer兼容小样确实执行CPU优化器更新，不能算研究训练。原始COCO17语义来源已记录，但本文CPU等价只证明适配前后同模型xy函数一致，不证明我们输出的真实17点语义、MRQ同步或迁移实验收益。

## 公平对照及备用路径

真实-only与真实+合成必须使用同一源checkpoint SHA、同一严格适配代码版本/标记、相同初始化与数据增强种子、真实曝光与总更新预算对照。不要让一个arm使用原[17,3]隐藏51、另一个使用重建[17,2]隐藏34后比较；也不要把某个arm切换compact模式。既有真实最终测试集保持封存。

CPU bitwise验证只覆盖本机Torch2.6.0/Ultralytics8.4.60与上述非fused普通Pose路径，不承诺CUDA/AMP/TensorRT/ONNX/DDP或完整Trainer。完整训练接入后仍需先验证模型加载后实际Pose塔容量与初始化xy，而不是只检查数据yaml。

如果未来部署/导出不接受自定义索引层，可以保留原[17,3]模型完全不改结构，单独在loss侧提供 `location_valid`，对unknown visibility不训练visibility loss，评估只取xy。此备选保留预训练visibility输出但明确不作为新数据监督或可信可见性；必须用独立loss适配而不是把unknown伪造为v1/v2。该替代尚未实现/验证。
