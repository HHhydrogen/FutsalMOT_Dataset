# P2-7 Camera Distribution Dataset Generation Validation

## Dataset Setup

本验证复用已经完成的两个受控数据集，不修改 Camera Distribution 设计：

```text
A: anchor_only
   cameras: C1 C2 C3 C4 C5

B: anchor_plus_one_partial
   cameras: C1 C2 C3 C4 C5 P01
```

产物目录：

```text
.futsalmot/p2_6_validation/p2_6_anchor_only_100f/
.futsalmot/p2_6_validation/p2_6_anchor_plus_one_partial_100f/
```

配置文件：

```text
configs/p2_6_anchor_only_100f.json
configs/p2_6_anchor_plus_one_partial_100f.json
```

两组均完成了现有的轨迹、annotation、MOT 和 RGB 产物生成。A 的渲染摘要为 `success`，共 500 张 `img1`；B 的渲染摘要为 `success`，共 600 张 `img1`。

## Controlled Variables

两组使用相同的：

- 场景：`5_vs_5`。
- root seed：`42`。
- `futsalmot_seed_v1` 派生 seed。
- trajectory 和 100 帧时间窗口。
- playback/source FPS：`10`。
- field dimensions：`40m x 20m`。
- Actor mapping。
- C1-C5 Camera Actor、transform、lens、filmback、resolution 和 Sequence mapping。
- 输出分辨率：`1920x1080`。

公平性复核结果：

| Camera | annotations A/B | MOT gt A/B | camera.json A/B |
| --- | --- | --- | --- |
| C1 | identical | identical | identical |
| C2 | identical | identical | identical |
| C3 | identical | identical | identical |
| C4 | identical | identical | identical |
| C5 | identical | identical | identical |

annotation 比较忽略了每组不同的 `episode_id`。因此 B 相比 A 的有效变量是增加 P01，没有发现 C1-C5 baseline 被改变。

## Observation Coverage

### Anchor Cameras

所有 C1-C5 在两组中的结果均为：

```text
annotation frames: 100
player observation slots: 1000
visible player observations: 1000
average visible players/frame: 10.0
MOT gt rows: 1000
```

逐 Camera 结果：

| Camera | A visible | B visible | Average/frame | GT rows |
| --- | ---: | ---: | ---: | ---: |
| C1 | 1000/1000 | 1000/1000 | 10.0 | 1000 |
| C2 | 1000/1000 | 1000/1000 | 10.0 | 1000 |
| C3 | 1000/1000 | 1000/1000 | 10.0 | 1000 |
| C4 | 1000/1000 | 1000/1000 | 10.0 | 1000 |
| C5 | 1000/1000 | 1000/1000 | 10.0 | 1000 |

C1-C5 在 A/B 中完全一致，说明增加 P01 没有改变 Anchor baseline。

### P01

P01 结果为：

```text
annotation frames: 100
player observation slots: 1000
visible player observations: 996
average visible players/frame: 9.96
MOT gt rows: 996
```

P01 的 4 个缺失 observation 全部是：

```text
entity: R0
frames: 1, 2, 3, 4
source steps: 0, 1, 2, 3
```

同一时间窗口中，C1-C5 均能观察到 R0。因此 P01 没有增加新的 player identity，但增加了 viewpoint-specific 的不可见信息：同一轨迹和时间下，P01 对 R0 产生了 Anchor 中不存在的局部缺失。

## P01 Complementarity

### Visibility Difference

P01 与 C1-C5 的主要可见性差异是：

```text
C1-C5: 每帧 10 名球员
P01:    平均每帧 9.96 名球员
```

差异集中在 P01 的前 4 帧，只有 R0 不可见。其余 9 名球员在 P01 中均为 100/100 可见。

### New Observation

从 player identity 的“新增可见对象”角度：

```text
新增可见 player identity: 0
```

从视角条件和缺失模式角度：

```text
新增局部不可见 observation: 4
涉及 entity: R0
```

所以 P01 的当前价值不是扩大 Anchor 的 player identity 集合，而是提供不同 camera viewpoint 下的 visibility variation 和局部不可见样本。

### Spatial Difference

已有 Camera metadata 和 footprint 证据表明 P01 不是 Anchor 的重复：

- P01 使用独立 Camera Actor 和 Sequence。
- P01 使用 `12.0mm` focal length；C1-C4 使用 `15.0mm`，C5 使用 `10.0mm`。
- P01 使用 `partial_field + left_half` profile intent。
- P01 的 position、rotation 和理论 footprint 与 C1-C5 不同。
- P01 与最近 Anchor 的位置差异不是零距离重复。

该结论基于已有 camera metadata、理论 footprint 和 annotation visibility，不使用图像 embedding 或深度视觉特征。

## Redundancy Analysis

检查维度：

- camera position
- camera rotation
- focal length / FOV
- resolution
- theoretical footprint
- annotation visibility pattern

结果：

```text
P01 与 C1-C5 不存在 transform、lens、resolution 或 calibration 的完全重复。
P01 产生了 R0 前 4 帧的独立不可见模式。
```

因此没有证据表明 P01 只是 C5 的重复 Camera。

但应保留结论边界：当前没有计算 pixel-level overlap、图像 embedding similarity 或 Re-ID 相似度，所以不能给出更强的视觉冗余结论。

## Track Observation Analysis

本次只使用已有 geometrical annotation 和 MOT GT，不运行 tracking、Re-ID 或 ID-switch evaluation。

Anchor C1-C5：

```text
每个 Camera:
  player track segments: 10
  visible duration per player: 100 frames
  mid-window exits: 0
  mid-window re-entries: 0
```

P01：

```text
player track segments: 10
R0 visible duration: 96 frames
其他 9 名球员 visible duration: 100 frames
mid-window re-entry: 0
```

R0 的不可见发生在窗口开头，不构成中途 exit/re-entry。因此当前 100 帧结果只能证明 P01 提供了局部 visibility interruption 样本，不能证明其已经带来丰富的长期 tracking interruption。

## Conclusion

### 1. `anchor_plus_one_partial` 是否比 `anchor_only` 提供额外信息？

是，但属于有限、明确的额外信息：

- C1-C5 baseline 完全不变。
- P01 增加了 1000 个 player observation slots。
- P01 产生了 996 个可见 observations 和 996 行 MOT GT。
- P01 增加了 4 个 Anchor 中不存在的局部不可见 observation。
- 该差异涉及 R0 的 source steps 0-3。

需要准确表述为：

```text
P01 增加了视角条件下的 visibility variation，
但没有增加新的 player identity。
```

### 2. P01 是否值得作为可选 Camera Distribution policy？

值得保留为可选 policy。

理由：

- A/B 对照公平性成立。
- P01 的 transform、lens、footprint 和 visibility pattern 与 Anchor 不完全重复。
- P01 产生了可复核的局部不可见事件。
- 增加成本明确为一个 Camera 和 100 帧对应输出。
- policy 保持确定性、可复现、可审计。

不建议将 P01 强制替换 `anchor_only`，因为本次证据显示的是有限的补充价值，而不是全面的 tracking 质量提升。

### 3. 是否需要继续扩展 P02/P03？

当前不需要立即扩展。

建议先在更长时间窗口或更多独立 seed 上确认：

- 中途 exit/re-entry 是否重复出现；
- P01 是否在不同轨迹中提供稳定的区域互补；
- visibility variation 是否具有统计稳定性。

在这些证据出现前，不增加 P02/P03，也不新增 Camera Evaluation Framework、Coverage Manager 或 Tracking/Re-ID 系统。

## Limitations

- 本报告复用已完成的 100 帧 A/B 数据，没有重新运行 MRQ 或生成数据集。
- 本窗口较短，P01 的缺失集中在开头，没有观察到中途 re-entry。
- 当前 annotation 是几何/可见性真值，不等于运行真实 tracking 算法后的 ID switch 或 Re-ID 结果。
- 未计算 pixel-level Camera overlap、image embedding similarity、遮挡或深度质量。
- 当前结论证明的是有限的 viewpoint-specific visibility complementarity，不是 P01 对整体 MOT 指标的显著性提升。
- `.futsalmot/` 产物属于本地生成物，不应提交到 Git。
