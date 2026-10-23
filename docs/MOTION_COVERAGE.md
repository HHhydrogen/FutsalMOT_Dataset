# 根轨迹运动与近距离交互覆盖

该工具帮助选择后续 Pose 数据的运动窗口与缺失因素；不把根位置运动代理包装成骨骼动作真值。新功能由独立模块 `src/grf_ue_bridge/motion_coverage.py` 和 CLI `tools/motion_coverage_report.py` 实现，不依赖 UE、GPU、NumPy 或 GRF 运行环境。

## 入口与输入冻结

```powershell
.venv/Scripts/python.exe -m grf_ue_bridge.tools.motion_coverage_report `
  --dataset-root G:/FutsalMOT_Dataset/v1_no_numbers_4k `
  --production-plan D:/projects/FutsalMOT_UEDataset/Saved/Production/v1_no_numbers_4k/production_plan.json `
  --project-root D:/projects/FutsalMOT_UEDataset `
  --output D:/projects/FutsalMOT_UEDataset/Saved/Pose17Development/motion_coverage_a01
```

Python 3.9 P1 环境运行。按 `dataset.json.groups[].path` 与 `group.json.clips[].path` 解析，不硬编码旧 `clips/clip_000` 层级。读取冻结生产计划，要求 seed 集合、split、帧数及 source_frame_range 与正式组一致；按 clip 的 SHA 验证一次轨迹文本。报告保存 26 个索引/计划/group/clip/轨迹输入的绝对路径、SHA 与大小，另记录当前工作树运动代码路径/SHA/相关行及动画目录文件名。不会读取、扫描、重新哈希 RGB/mask，不修改正式 G 盘数据，不访问真实足球测试集。

已有 output 拒绝覆盖；下一次统计使用明确的新 attempt。报告写入项目前述 Saved 路径，禁止写入正式数据根。首次调试发现 `group.entities` 同时含 BALL，已修正为球员统计排除 BALL，增加回归测试；正式输出目录在输入校验失败时尚未创建，成功运行后才写 a01。

## 统计定义

- 每个连续采样间隔 `dt=t[i+1]-t[i]`，从世界米制 XY 位置差分计算根速度，不直接相信导出的 `speed_mps`。时间必须严格递增、step 连续、球员集合每帧唯一；出现缺帧拒绝跨缺帧求导。
- 时间跨度是 `(N-1)` 个已观察间隔的总和，而非 `N/FPS`；四机位共享运动，不将时长乘四。
- 速度带沿用运动层数值边界：idle ≤0.3，walk (0.3,1]，jog (1,2]，run (2,3.5]，sprint >3.5 m/s。它们是描述规则，不是运动医学或人工动作类别。步态时长按真实 dt 加权；速率分位数按观测间隔取样，当前正式帧间隔均为 0.1 s。
- 相邻区间速度的中心时间差计算标量速率变化，≥1.2 m/s² 记加速，≤−1.2 记制动。该导数的支持时间少于轨迹跨度一个边界间隔；保存在 `derivative_support_s`。未平滑，10Hz 差分可能放大抖动，不能解释为真实人体加速度标签。
- 用相邻根速度向量叉积/点积的 atan2 计算最短方向变化，处理 ±180° 环绕。两段速度均≥0.5 m/s 时才统计，绝对方向角速率≥90°/s 记 sharp_turn。它是**运动方向变化**，不是身体朝向、脚步 pivot 或骨骼旋转真值。
- 团队活跃要求同一区间至少 4 名非门将速度≥0.5 m/s；连续真区间才累加为同一次活动，短暂停顿不会被拼接成一段。该规则与生产准入的整体活跃覆盖相关，但不是新的逐帧质量淘汰规则。
- 每对球员以区间中点线性 XY 位置计算≤1 m、≤2 m 邻近，固定两个距离用于敏感性对照。报告 pair-seconds、跨队 pair-seconds、每对连续时长及≥0.5 s 的持续邻近段数量。多个球员对同时邻近时，pair-seconds 可以大于独立运动秒数；它不是额外的视频时长。

## 本次真实结果

`Saved/Pose17Development/motion_coverage_a01/motion_coverage.json` 和 `.md` 已成功生成。8 个 seed、8 条唯一轨迹、5029 帧、502.1 秒（约 8 分 22 秒）独立运动，十名球员共 5021 player-seconds。32 个相机序列不能被当作 32 场独立运动。

|seed|连续片段跨度 s|团队活跃比例|最长连续团队活跃 s|
|---|---:|---:|---:|
|1007|43.2|0.882|12.3|
|1021|90.1|0.957|37.8|
|1025|42.9|0.939|19.6|
|1042|46.2|0.916|16.4|
|1044|82.1|0.938|30.3|
|1048|95.1|0.948|24.9|
|1102|41.3|0.903|34.2|
|1131|61.2|0.951|14.9|

全部组按时长加权，团队活跃比例为 93.53%。idle/walk/jog/run/sprint 分别为 1448.6/943.3/1860.8/768.3/**0.0** player-seconds。当前数据没有超过 3.5 m/s 这一数值阈值的根轨迹冲刺，不能声称覆盖高速冲刺。

加速、制动、快速运动方向变化分别为 1258.3/846.2/430.2 player-seconds；≤1m、≤2m 跨队邻近分别为 253.1/656.6 pair-seconds。这些值是已定义代理统计，不是真实传射、对抗、遮挡标注。

## 静态动画能力与研究界限

当前工作树 `ue/player_motion.py` 定义 `KICK/PASS/SHOT/RECEIVE` 但注释为“只预留枚举，暂不消费”；该 Python 运动参数路径返回 `action_state=MotionAction.NONE`。这仅证明当前这条代码路径未消费足球语义动作，不证明所有实际 AnimBP 或其他路径都不存在 kick 动画。该文件存在 locomotion、pivot 和 GK 方向类的规则输出，规则输出本身不证明对应动画资源实际运行。

项目 canonical 动画目录可枚举 Idle、Walk、Jog、Jump/Fall/Land、BlendSpace、FootIK 与 AnimBP 等文件名；Unarmed Walk/Jog 的名称也不能被描述成经过真人足球动作验证的序列。C++ `FutsalPlayerAnimInstance.cpp` 的位置差分和运动输出可静态查证，但未在本任务中打开 UE 查询 AnimGraph、加载骨骼资产或逐帧检查动作。文件名搜索未发现传射类名称也不能证明整个资产库绝对没有相关资源。

因此建议后续论文使用“运动/邻近覆盖”表述，并将 **骨骼动作有效性** 单独验收：逐角色/动画姿态校验、跨相机投影一致性、遮挡/出画可见性，以及真实数据上的 Pose AP/PCK/追踪实验。当前正式 v1 没有 Pose，本报告不证明新 Pose 数据集已完成。若缺少传射动作资源和真实事件同步，就暂不将“足球动作理解”列为已支持任务。

## 验证

`tests/test_motion_coverage.py`：12 passed（0.10s）。覆盖不规则 dt 的真实时长、位置/速度单位、±180° 转向环绕、低速方向抑制、邻近阈值、持续段不跨间隙、BALL 排除、NaN/重复实体/缺帧/重复时间拒绝及 manifest 路径越界拒绝。单测证明统计实现，不证明运动或肢体动画真实。
