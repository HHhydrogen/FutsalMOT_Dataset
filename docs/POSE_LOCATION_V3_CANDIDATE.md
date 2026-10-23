# 显式COCO sigma与Pose分支更新候选

本轮只新增候选模块及CPU测试，不修改正在运行或已冻结的v2 loss、Trainer、协议、数据或检查点，也不启动下一轮GPU训练。

## 已确认的loss语义差异

本机Ultralytics8.4.60的`v8PoseLoss.__init__`仅在`kpt_shape==[17,3]`时采用COCO17 `OKS_SIGMA`；位置头改成[17,2]后，原v2桥接落入`ones(17)/17`。实际只读CPU加载官方与derived初始化已确认：原鼻.026、眼.025、肩.079、髋.107等，v2全部约.05882353，均为float32。因此相对COCO预训练，脸部误差容差变宽、部分肢体变窄。这是确定差异，不能单凭它解释全部开发AP下降。

首个纯R开发结果相对预训练由keypoint AP .491896降至.288049，bbox AP .530914降至.500820；已有报告显示条件L2由11.04px至15.86px。纯R不含S，不能归因合成。既有曝光记录loss均有限、visibility始终0；actual/EMA更新1000。只读checkpoint检查确认EMA87个BN全eval且所有state有限；BN相对初始化有实际统计漂移，并非已发现BN忘记切eval。EMA/评价路径与真实数据几何另有独立核对，仍须把遗忘、小样本适配、学习率和BN变化当作待检验因素。

## 独立v3位置loss

`src/grf_ue_bridge/pose_location_supervision_v3.py`提供`coco_sigma_location_loss_class()`。它继续复用已验独立17bool `location_valid`打包、上游目标分配、bbox/cls/dfl与位置归一化，但显式将COCO17 sigma安装到KeypointLoss。没有visibility输出/BCE；真实v0仍只屏蔽位置，人物框保留；v1/v2位置均有效，不把S unknown变成visible。

CPU测试直接与原上游17×3 `v8PoseLoss.calculate_keypoints_loss`实算比较，剥离visibility BCE后，位置loss和xy梯度逐值相同；全缺点人物及无效点梯度为0。另用完整随机toy PoseModel的原始3输出头和exactXY头前向/分配/loss/反向对照，检测加位置loss及物理输出梯度在指定float容差内一致。测试没有复制一份期望公式作为唯一证据。

原v2仍保留uniform sigma且原结果完整保留，不能把新v3模块偷偷替换进旧run。

## Pose-only候选核心

`src/grf_ue_bridge/pose_only_calibration_v3.py`提供`configure_pose_only(model)`，原地配置实际exactXY模型并返回guard：

- 只允许Pose cv4分支参数训练，冻结共享backbone/neck和检测分支。
- 整体eval模式下仍保留autograd，固定所有BN running mean/var/count；Pose分支BN affine参数可训练。
- 物理51通道中的未输出visibility行梯度屏蔽并检查值不变；候选优化器须weight_decay=0，避免行级梯度屏蔽被权重衰减绕过。
- `guard.assert_unchanged()`核对冻结参数、BN统计及visibility物理行；`guard.enforce_eval_mode()`用于外层曾调用train后的显式恢复；`guard.close()`移除自有梯度hook。

调用方不能随后任意`model.train()`而声称BN固定，也不能包`no_grad()`切断Pose学习。没有Pose前景时，后续正式循环须明确处理零梯度更新及预算，不自动改成删框或丢样本。此文件不是新Trainer，也没有替root冻结lr、更新步数或选择结果。

单步CPU随机toy集成已验证：位置loss/梯度非零、Pose输出发生变化，同时原始box logits、class logits和解码检测前五通道逐值相同，所有冻结参数/BN及未输出物理行不变。该结果证明此独立CPU构建能隔离检测器，不替代未来真实图/GPU配对或MOT收益。

## 后续写盘候选

旧首R共350.29秒，其中checkpoint约228.33秒；缓存预热仅3.09秒且无淘汰，不能将本次主要开销归因图像缓存。下一独立Trainer可每50步、最终和明确安全暂停保存完整model/EMA/optimizer/RNG，逐步保留曝光和事务；正常暂停须保存当前完整数学状态。硬崩溃时若实际曝光更新超出last，必须将该尝试标失败，不能从旧last重放后宣称相同实际更新预算。该策略只是下一候选建议，本模块没有改活跃写盘频率。

## 最小CPU验证

研究Python运行`tests/test_pose_location_supervision_v3.py`与`tests/test_pose_only_calibration_v3.py`，本次4项通过。研究环境若没有pytest，可仅借用P1已安装的纯Python pytest，不安装/升级Torch或P1 Python；测试强制CUDA不可见，仅用64×64随机toy，不读取真实dev图片或封存test。
