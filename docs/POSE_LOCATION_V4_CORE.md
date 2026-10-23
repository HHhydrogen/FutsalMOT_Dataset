# v4候选：相同真实曝光与分域位置loss

`src/grf_ue_bridge/pose_location_supervision_v4.py`是独立数学核心，不管理采样、优化器、训练准入、checkpoint或GPU。它不修改v2/v3冻结文件，也不是新的算法创新或已证明收益。

## 数学契约

两臂按原采样表同顺序取`shared_real_indices[3]`；每臂1000次更新、3000张真实曝光。混合臂另取原表`synthetic_index`的1000S，不重新选比例或样本。

本机Ultralytics 8.4.60返回五项loss向量，并已乘batch size。核心只取带原pose gain的**位置项**，不对box、class或visibility求导：

```text
meanR = C_R3[location] / 3
meanS = C_S1[location] / 1
R3 baseline = meanR
R3S1 candidate = meanR + (1/3) * meanS
```

`1/3`固定为乘法系数，不提供搜索α接口。真实与合成分别前向、分别调用两个criterion；禁止先拼四图再共用TAL/有效点分母。R3内部保留上游前景归一化，不能把“除batch size”解释成每人或每点等权。S的有效点数与前景数量仍影响其自身项；消除的是S改变R分母的混杂，不是宣称原始监督点预算相同。

原v3是R4000对R3000+S1000，且对混合batch共用分母；本v4同时改变真实曝光契约和batch缩放。因此不能将其结果混称为v3单因素消融。混合臂多1000次S计算，不等FLOPs。

## 最小接入

调用方继续使用已验exact-xy模型、`LocationPoseDataset.collate_fn`及固定采样表。图像需已转换为模型同设备/类型的`[0,1]`浮点tensor，不能传uint8或重复除255。

```python
from grf_ue_bridge.pose_location_supervision_v4 import SeparateDomainLocationObjective

# model为已验17×2 exact_xy，保留末端物理51通道；这里不加载或下载权重。
objective = SeparateDomainLocationObjective(model)
try:
    # real_batch只含本次原shared3真实；synthetic_batch是单幅S、本域batch_idx从0开始。
    result = objective(real_batch)                  # R3基线
    # 或：result = objective(real_batch, synthetic_batch)  # 固定α=1/3
    optimizer.zero_grad(set_to_none=True)
    result.loss.backward()
    # 调用方在此记录/裁剪梯度、step、EMA和事务，使用独立新协议。
    optimizer.step()
    objective.guard.assert_unchanged()
    log_receipt(result.report)
finally:
    objective.close()
```

也可传`freeze_guard=configure_pose_only(model)`复用调用方已有guard；此时`close()`只移除本核心的观察hook，不关闭外部guard。优化器必须只含可训练cv4参数，避免weight decay/旧optimizer state更新被冻结物理visibility行；恢复后继续核对guard。模块拥有guard时会配置仅cv4、所有BN eval、保留物理visibility行梯度hook。

每个分域batch要求显式`sources`、原`visibility_state`、17×2位置和17bool。R必须`coco_real`且COCO unlabeled/v0点不能有效；S是本批`synthetic`且所有visibility为unknown。unknown不屏蔽合法几何位置，不进入BCE。全缺点人物框继续参与分配，其位置梯度为零。非法大小、混合域索引、v0提升或合成可见性提升会拒绝。

返回`SeparatedLocationResult`包含可求导的`loss/real_mean/synthetic_mean`及独立R/S计数：图片、人物框、原始有效点、已分配有效点、前景/非空前景、keypoint mean元素数和batch除数。报告固定`training_authorized=false`，不能拿它代替主agent签署的协议/准入。

## CPU验证与限制

`tests/test_pose_location_supervision_v4.py`实际创建随机YOLO11n Pose模型，先映射exact_xy，再强制CPU、单线程及eval-with-autograd；不下载权重，不读取真实媒体。2026-10-23新鲜执行**5 passed，0 skipped，4.14秒**：

1. R3基线与实际计算S但α=0的CPU探针，R loss及全部模型裁剪前梯度逐值相同；稀疏S1和稠密S10人均不改变R前景/有效性分母。
2. R项与直接调用原v3 COCO-sigma上游位置项再除3的损失及梯度逐值相同。
3. 固定α=1/3、unknown S产生位置梯度；一次CPU toy更新后检测输出、BN及物理visibility行逐值不变，位置输出变化。
4. 原无效关节及全缺点人物的已分配位置梯度为零；不调用visibility BCE。
5. 拒绝R4、合成visibility提升、跨域batch_idx、非bool mask及COCO v0提升。

`alpha_zero_cpu_probe()`只允许CPU且报告`probe_only=true`，不能作为正式研究臂。逐值梯度检查发生在裁剪前、同模型参数/同R输入下；混合梯度裁剪、Adam与EMA的非线性不能被解释为实际两臂训练全过程的R梯度相同。

测试中曾发现`(1/3)*S`与`S/3`的浮点末位差异；登记算子固定为前者，公式测试也使用前者。α=0损失/梯度的逐值标准没有放宽。

本模块未运行真实GPU、未做COCO或追踪评价。后续独立Trainer仍需冻结1000步采样、相同初始化、optimizer/clip/EMA与checkpoint预算、原始曝光、按源归一化证据及相对预训练的固定检测AssA门槛；当前UE渲染和GPU训练继续串行。尺度ROI候选未接入本核心，也未改当前R/S图像。
