# 头部5表面点深度残差诊断

独立工具 `src/grf_ue_bridge/tools/inspect_pose17_head_depth.py` 仅消费已经结构配对通过的小样；不改runner、C++、既有配对器、COCO标注或任何正式数据。所有点的visibility仍为unknown，MRQ_sync_verified仍为false。

```powershell
.venv/Scripts/python.exe -m grf_ue_bridge.tools.inspect_pose17_head_depth <smoke-attempt> <结构配对report.json> <depth_encoding_a02.json> <全新输出目录>
```

工具读取native五头点世界米坐标并投影到当前EXR实际相机；读取 `FinalImageWorldDepth.R` float32线性前向深度厘米并除100。容差预注册固定5mm，不提供调大阈值的命令行选项。先要求实际平面标定的最大误差不超过5mm；平面标定不代表头发、眼球、Masked/WPO等最终表面已一致。

Cryptomatte的三个rank层保留全部coverage。UE5.8 `MovieRenderOverlappedMask.cpp::FetchFinalPixelDataLinearColor` 以R/B存两rank ID、G/A存对应归一化coverage；本工具保留中心ID/coverage、目标总coverage和3×3支持。同一dominant实例不自动证明深度通道采到相同透明表面。混合ID、coverage不足、图像边界或无效深度不生成明确候选；3×3深度spread超过10mm时也保守保持unknown，这可能包括斜面而非实际几何断裂，因此单列原始spread而不称其必为遮挡。

每个image frame分别使用Pose frame−1/0/+1投影，并输出有符号残差 `anchor_forward_z - depth_first_hit`、绝对残差、目标实例命中、全部候选状态及支持数量。正残差仅为较近首命中候选，不能自动变成COCO遮挡标签。统计同时列目标纯实例内部支持及保守边界过滤后的支持；另列三个偏移均有效的同点共同支持，避免错配丢掉大量点却得到较小中位数的偏差。不会自动选择最佳相位或更改帧映射。

## a02真实诊断

报告：`Saved/Pose17Development/smoke/player003_mrq_a02_head_depth_a02/report.json`，errors为空，保留首命中dominant ID对应Actor标签。两机位×3帧，共30个同帧头点和40个相邻帧对照记录。a01分析为前一诊断版本，保留未覆盖。

| 机位 | Pose偏移 | 目标纯实例内部支持数（深度spread过滤前） | 绝对残差中位数 |
|---|---:|---:|---:|
| CineCam_01 | −1 | 10 | 161.07mm |
| CineCam_01 | 0 | 11 | 7.30mm |
| CineCam_01 | +1 | 4 | 37.39mm |
| CineCam_02 | −1 | 1 | 130.87mm |
| CineCam_02 | 0 | 1 | 146.15mm |
| CineCam_02 | +1 | 0 | 无有效支持 |

第一机位三偏移共同支持仅1点，其−1/0/+1残差为181.66/15.68/40.33mm。同帧更接近，但所有第一机位点均触发深度spread保守门禁，不能宣布5mm表面一致性通过。第二机位球门网和身体自遮挡显著，不能据其低覆盖数据判断相位。需要更近、无遮挡的独立诊断视角，保留相同5mm阈值后再看残差与支持数量。

当前新增工具8个CPU针对测试通过；连同配对、收据、runner及导出共69 passed。真实UE深度标定由主agent完成；该工具的CPU测试不替代真实头部语义/最终蒙皮表面验收。

## a04近景真实诊断

`player003_closeup_a04_alignment_a01/report.json` 的三机位×3帧结构配对通过。配对工具新增独立计划门禁：EXR实际位置必须等于计划closeup_location_cm/100，EXR实际forward必须等于从该位置指向closeup_look_at_cm的单位方向；仅native与EXR一致不够。a04全部位置误差0m、forward最大分量误差5.55e−8，证明近景视向实际执行；没有直接观测PIE中look-at bool，因此不宣称读取过运行时false。a03虽然渲染完成，但三机位均未执行计划方向，失败证据保留在 `player003_closeup_a03_alignment_rejected_a01/report.json`，不作为近景验收。

深度报告：`player003_closeup_a04_head_depth_a01/report.json`，errors为空；依旧固定5mm、unknown与未正式同步状态。

| 机位 | 同帧目标纯实例内部数 | 同帧绝对残差中位数 | 保守边界过滤后数 | 5mm一致候选 |
|---|---:|---:|---:|---:|
| CineCam_01 正面 | 15 | 0.201mm | 10 | 9 |
| CineCam_02 负Y侧 | 11 | 1.159mm | 11 | 6 |
| CineCam_03 正Y侧 | 9 | 0.428mm | 8 | 5 |

45个同帧点中20个表面一致候选、9个较近自身表面候选、16个边界未知；仍不写COCO visibility。正面鼻/双眼9点绝对残差最大0.327mm；负Y侧近耳3点最大0.047mm、近眼3点最大1.159mm；正Y侧近耳3点最大0.056mm。正Y侧近眼帧1/2残差0.428/0.172mm，帧0深度spread门禁拒绝。侧面远耳约132–139mm较近自身表面，与角色自遮挡解释一致，但只作候选。

相邻帧−1/+1对照在各机位保守有效支持上均无点通过5mm，绝对残差中位数约58–180mm；正面相同image/keypoint的三个偏移共同支持3点，−1/0/+1中位数219.45/0.327/57.76mm。此次证据显著支持当前相位优于前后相邻整帧，不能替代最终全表面/全部角色/正式GRF长序列验收。新增计划方向反例后，Pose17针对CPU测试共70 passed。
