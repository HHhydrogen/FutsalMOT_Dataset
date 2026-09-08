# P2-8 Camera Distribution Stability Validation

## Dataset Setup

本次按计划生成了 4 个 seed 的 A/B 轨迹：

```text
Seeds: 42, 43, 44, 45
Frames per episode: 300
Scenario: 5_vs_5

A: anchor_only
   C1 C2 C3 C4 C5

B: anchor_plus_one_partial
   C1 C2 C3 C4 C5 P01
```

每个 seed 的 A/B 使用相同 root seed、相同导出规则和相同 trajectory generation 规则。A/B 的轨迹导出均成功，`frames.jsonl` 均为 300 行。

生成目录：

```text
.futsalmot/p2_8_validation/seed_42/
.futsalmot/p2_8_validation/seed_43/
.futsalmot/p2_8_validation/seed_44/
.futsalmot/p2_8_validation/seed_45/
```

## Controlled Variables

所有生成配置保持以下变量不变：

- 场景：`5_vs_5`。
- field dimensions：`40m x 20m`。
- C1-C5 profile、P01 profile、lens、resolution 和 camera policy 定义。
- playback/source FPS：`10`。
- 每个 seed 的 A/B 使用同一个 root seed。
- A/B 只在 camera set 上有差异：B 比 A 多 P01。

本阶段先执行 trajectory validity gate，再进行 UE annotation。原因是如果 trajectory 本身出现长时间静止，后续 visibility variation 无法作为稳定的 Camera Distribution 证据。本阶段未运行 MRQ。

## Trajectory Validity Gate

使用现有 Motion Quality Audit 指标检查：

- 外场球员 active ratio。
- 外场球员最长静止 streak。
- goalkeeper 最长静止 streak。
- team active outfield coverage。
- longest global low-motion plateau。

结果：

| Seed | Min outfield active ratio | Max outfield stationary streak | Max GK stationary streak | Team active coverage | Longest global low-motion plateau | Decision |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 42 | 0.390 | 13.5s | 22.0s | 0.390 | 15.3s | invalid |
| 43 | 0.587 | 2.6s | 13.4s | 0.760 | 1.3s | invalid |
| 44 | 0.763 | 1.1s | 12.2s | 0.913 | 0.7s | valid |
| 45 | 0.727 | 1.3s | 11.6s | 0.890 | 0.8s | conditional invalid |

采用的有效性解释：

- 外场球员不能出现明显长时间静止；
- team active coverage 应达到现有 Motion Quality 目标 `0.90`；
- seed 42 明显无效；
- seed 43 的外场 active ratio 和静止 streak 不达标；
- seed 45 的 team active coverage 为 `0.890`，低于 `0.90` 门槛，因此不纳入严格稳定性结论；
- 只有 seed 44 通过当前严格门槛。

A/B 在每个 seed 内的 trajectory metrics 完全相同，因为两组共享同一导出轨迹。

## Stability Results

本阶段没有对不合格轨迹继续执行 UE annotation，也没有把无效 seed 的 P01 visibility 结果用于稳定性结论。

seed 44 的有效 A/B annotation 已完成，结果为：

```text
C1-C5 baseline A/B: annotations、MOT gt、camera.json 全部一致
P01 observation slots: 1000
P01 visible observations: 994
P01 invisible observations: 6
P01 unique visibility events: 2
P01 event player: R0
P01 event durations: 3 frames + 3 frames
P01 event ranges: frames 1-3 and 91-93
```

这些结果只说明 seed 44 中观察到了两个 viewpoint-specific visibility interruption，不足以形成跨 seed 稳定性结论。

当前可确认的总体结果是：

```text
有效 seed 数量: 1 / 4
有效 seed: 44
严格跨 seed stability evidence: insufficient
```

seed 44 的 A/B 轨迹已经生成，但由于当前工作阶段要求先通过完整的多 seed trajectory validity gate，尚未将单个有效 seed 的 Camera visibility 结果升级为跨 seed 结论。

P2-7 的 seed=42、100-frame 结果仍然保留为历史单 episode 证据，但本次 300-frame 复核显示 seed=42 在更长窗口存在明显低运动平台，因此不能将它作为 P2-8 的有效稳定性样本。

## C1-C5 Baseline Consistency

本阶段仅对有效 seed 44 执行了两组 UE annotation，因此没有完成全部四个 seed 的 C1-C5 consistency 对比。

seed 44 的新增 consistency 结果为：

- C1-C5 annotation A/B identical；
- C1-C5 MOT GT A/B identical；
- C1-C5 camera calibration A/B identical。

现有 P2-7 seed=42 证据仍显示：

- C1-C5 annotation A/B identical；
- C1-C5 MOT GT A/B identical；
- C1-C5 camera calibration A/B identical。

这些结果不能替代本次四 seed、300-frame 的完整 baseline consistency 验证。

## P01 Complementarity Stability

本阶段不报告跨 seed P01 unique visibility event 稳定性，因为：

- seed 42 和 43 的 trajectory motion quality 不合格；
- seed 45 未达到 team active coverage 门槛；
- 只剩 seed 44 一个合格 seed，无法构成跨 seed stability evidence；
- 未对四个 seed 全部完成 UE annotation，避免将不合格轨迹的 visibility 结果强行解释为 Camera 价值。

因此当前没有足够证据判断：

- P01 unique visibility event 是否稳定重复；
- 事件涉及的 player 是否跨 seed 稳定；
- 事件持续时间是否具有统计稳定性。

## Redundancy Analysis

当前没有新增 P01 redundancy 结论。

P2-7 已有证据表明 P01 与 Anchor 在以下方面不完全重复：

- position
- rotation
- focal length / FOV
- theoretical footprint
- annotation visibility pattern

但 P2-8 的目标是验证这些互补现象是否跨 seed 稳定。本次 trajectory gate 结果不足以支持更强的稳定性结论。

本次没有引入：

- image embedding
- deep feature similarity
- Re-ID
- tracking algorithm
- ID switch evaluation

## Conclusion

### 1. P01 的互补价值是否跨 seed 稳定？

当前仍无法证明。

原因不是 P01 已被证明无效，而是四个候选 seed 中只有 seed 44 通过严格 trajectory validity gate，并且只有该 seed 完成了 annotation 闭环：

```text
42: invalid
43: invalid
44: valid
45: conditional invalid
```

有效样本数量不足以做稳定性结论。

### 2. `anchor_plus_one_partial` 是否值得保留？

基于 P2-7 单 episode 结果，P01 仍可作为可选 policy 保留；但 P2-8 尚未提供足够的跨 seed 稳定性证据来升级这个结论。

更准确的当前判断是：

```text
有限有效，但稳定性尚未确认。
```

### 3. 当前是否需要继续增加 P02/P03？

不需要。

在 trajectory validity gate 和 P01 多 seed 稳定性尚未建立前，不增加 P02/P03，不调整 P01 placement，也不新增 Camera Evaluation Framework、Coverage Manager、Tracking 或 Re-ID 系统。

## Limitations

- 本阶段已完成 42-45 四个 seed 的 A/B 轨迹导出，但只有 seed 44 通过严格运动有效性门槛。
- 因为轨迹有效性门槛未满足，不继续执行其余 6 组 UE annotation，避免产生难以解释的无效 visibility 结论。
- seed 42 的 300 帧轨迹出现 15.3 秒全局低运动平台，不能直接复用 P2-7 的短窗口结论作为稳定性证据。
- seed 43 和 seed 45 也未达到本次严格纳入标准。
- 本报告不证明 P01 跨 seed 稳定有效，也不证明 P01 无效。
- 本报告没有修改 Camera Profile、UE Actor、Sequence、Camera Cut、MRQ、runtime camera selection 或 placement logic。
- `.futsalmot/p2_8_validation/` 是本地验证产物，不应提交到 Git。
