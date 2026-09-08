# P2-6 Camera Distribution Validation Report

## Validation Setup

本验证比较两个 Camera Distribution policy：

```text
A: anchor_only
   C1 C2 C3 C4 C5

B: anchor_plus_one_partial
   C1 C2 C3 C4 C5 P01
```

两组使用相同的：

- GRF 场景：`5_vs_5`。
- root seed：`42`。
- 轨迹长度：`100` frames。
- playback/source FPS：`10`。
- 输出分辨率：`1920x1080`。
- actor mapping。
- actor 状态和 source frames。
- C1-C5 Camera Actor、Sequence、transform、focal length、filmback 和 resolution。

两组使用独立输出目录：

```text
.futsalmot/p2_6_validation/p2_6_anchor_only_100f/
.futsalmot/p2_6_validation/p2_6_anchor_plus_one_partial_100f/
```

配置文件：

```text
configs/p2_6_anchor_only_100f.json
configs/p2_6_anchor_plus_one_partial_100f.json
```

两组都生成了相同类型的现有产物：RGB、annotations、MOT `gt.txt`、`camera.json` 和 audit report。本次没有启用 Mask，也没有接入 `camera_state.jsonl`。

## Execution Result

两组任务均通过 task validation、annotation export、MRQ render 和 audit。

### Test A

```text
policy: anchor_only
cameras: 5
RGB: 5 x 100 = 500 frames
render_summary.status: success
audit: 5/5 cameras passed
```

### Test B

```text
policy: anchor_plus_one_partial
cameras: 6
RGB: 6 x 100 = 600 frames
render_summary.status: success
audit: 6/6 cameras passed
```

Test B 的 MRQ 日志明确列出：

```text
LS_Cam_P01 -> CineCam_P01
```

P01 不是 C1-C5 中任意 Camera 的复制路径。

## Fairness Check

C1-C5 在 A/B 两组的 `camera.json` 对比结果：

- location：一致。
- rotation：一致。
- focal length：一致。
- sensor/filmback：一致。
- resolution：一致。

每个 C1-C5 的 camera calibration 均为 `1920x1080`，且对应现有 Camera Actor 和 Sequence 不变。

去除每组独立的 `episode_id` 后，C1-C5 的 `annotations.jsonl` 内容逐 Camera 完全一致：

```text
CineCam_01: equal
CineCam_02: equal
CineCam_03: equal
CineCam_04: equal
CineCam_Main: equal
```

C1-C5 的 MOT `gt.txt` hash 也逐 Camera 一致。

因此，B 相比 A 的有效变化是增加 P01；没有发现 C1-C5 参数或 actor/source frame 差异污染比较。

说明：RGB 文件 hash 在两次独立 MRQ 执行间不完全一致，但这不是 Camera Distribution 差异证据；C1-C5 的 calibration、annotation 和 MOT 输出一致，渲染图像存在正常的独立 MRQ/时间性差异。价值判断主要使用几何 annotation/MOT 数据，而不是跨次渲染的二进制图像 hash。

## P01 Coverage Check

P01 `camera.json` 实际读回：

```text
camera: CineCam_P01
sequence: LS_Cam_P01
location: [-10.0, -25.0, 16.0] m
rotation: [-32.471191, 90.0, 0.0] deg
focal length: 12.0 mm
horizontal FOV: 89.424168 deg
sensor: 23.76 x 13.365 mm
resolution: 1920 x 1080
```

P01 resolved profile：

```text
coverage: partial_field
region: left_half
placement_template: left_half_sideline_high_v1
```

P2-4.1 的几何 Coverage validation 仍通过。

从实际 P01 annotation 看，P01 不是 full-field 的简单复制：

- 在前 4 个 source frames 中，`R0` 不在 P01 视野内。
- 其余时间 `R0` 重新进入视野。
- 这产生了真实的局部视野边界效果。

本次无法直接读取 RGB 图片附件进行人工视觉验收，因此“主要观察左半场”的结论基于 P01 的 resolved transform、FOV 几何检查、camera calibration 和 annotation 结果，不宣称完成了人工像素级视觉检查。

## MOT Value Comparison

### Player Visibility

现有 100 帧几何 annotation 的统计如下：

```text
A / C1-C5:
  visible player frames: 1000 / 1000 per camera
  average visible players per frame: 10.0

B / C1-C5:
  visible player frames: 1000 / 1000 per camera
  average visible players per frame: 10.0

B / P01:
  visible player frames: 996 / 1000
  average visible players per frame: 9.96
  missing player observations: 4
```

P01 的缺失观察全部对应：

```text
entity: R0
source frames: 0, 1, 2, 3
```

这说明 P01 至少带来了 Anchor 视角中没有出现的局部不可见事件：同一 source frame 中 C1-C5 都能观察到 R0，而 P01 在前 4 帧看不到 R0。

### Track Behavior

当前 annotation/MOT 是几何真值，不运行新的 tracking 或 Re-ID 算法。可直接计算的 track continuity 指标为：

```text
A / each C1-C5:
  player track segments: 10
  player entries: 10
  player exits: 0

B / each C1-C5:
  player track segments: 10
  player entries: 10
  player exits: 0

B / P01:
  player track segments: 10
  player entries: 10
  player exits: 0
  R0 visible frames: 96 / 100
```

由于 R0 的缺失集中在序列开头，当前 100 帧窗口没有形成完整的中途 exit/re-entry transition；因此不能夸大为已经证明了丰富的长期 track interruption。可以确认的是 P01 产生了 4 个真实的局部不可见样本，并降低了该 entity 的可见帧覆盖。

### Camera Complementarity

P01 的实际位置与各 Anchor 的欧氏距离：

```text
to CineCam_01: 52.079 m
to CineCam_02: 41.379 m
to CineCam_03: 21.500 m
to CineCam_04: 38.239 m
to CineCam_Main: 48.146 m
```

P01 与最近的 `CineCam_03` 仍有约 `21.5 m` 的位置差异，同时拥有不同的 transform、12 mm focal length 和 89.42° horizontal FOV。它提供了独立的侧向高位观察角度，不是现有 Anchor 的名称别名或参数复制。

### Camera Redundancy

没有发现 P01 与任意 C1-C5 的 transform、lens、resolution 或 calibration 完全重复。

从 annotation 结果看，P01 的投影 bbox 也不同于 C1-C5；例如同一组 actor 在 P01 中具有明显不同的图像位置和尺寸。当前没有计算图像 embedding 或 pixel-level similarity，因此不能给出更强的视觉相似度结论。

结论：P01 不属于高度冗余 Camera。它与 C1-C5 存在空间互补，且已经产生了额外不可见 observation。

## Quantitative Evidence

| Metric | A: anchor_only | B: C1-C5 | B: P01 |
| --- | ---: | ---: | ---: |
| Camera count | 5 | 5 | 1 |
| RGB frames | 500 | 500 | 100 |
| Annotation frames | 100/camera | 100/camera | 100 |
| Player observations | 1000/camera | 1000/camera | 996 |
| Average visible players/frame | 10.0 | 10.0 | 9.96 |
| Player track segments | 10/camera | 10/camera | 10 |
| Player entries | 10/camera | 10/camera | 10 |
| Player exits | 0/camera | 0/camera | 0 |
| MOT GT rows | 1000/camera | 1000/camera | 996 |
| Resolution | 1920x1080 | 1920x1080 | 1920x1080 |

当前未计算：

- Re-ID accuracy。
- ID switches。
- Image embedding similarity。
- Pixel-level camera overlap。
- 长时段轨迹 interruption/re-entry rate。

这些指标需要额外 tracking 或视觉评估能力，本次不为它们新增框架。

## Conclusion

### 1. P01 是否符合 left_half partial coverage？

在当前可验证证据范围内，符合：

- profile 明确声明 `partial_field + left_half`。
- placement template 正确。
- position/rotation/FOV 几何检查通过。
- 实际 `camera.json` 读回与 resolved profile 一致。
- P01 产生了 4 个 C1-C5 未出现的局部不可见 player observations。

人工 RGB 视觉检查本次未完成，因为当前模型无法直接读取图像附件；因此保留这一项视觉验收限制。

### 2. P01 是否与 C1-C5 明显不同？

是。P01 的 transform、lens、FOV 和投影结果均独立，且 MRQ 日志确认使用 `CineCam_P01 / LS_Cam_P01`。

### 3. P01 是否增加有价值的 MOT 观察？

是，但价值属于“有限、已观测到的补充价值”，不是已经证明的大规模 tracking 难度提升：

- P01 产生了 4 个 Anchor 中不存在的 `R0` 不可见 frame/entity observation。
- P01 平均可见球员数为 `9.96`，低于 Anchor 的 `10.0`。
- 当前窗口过短且缺失集中在开头，尚未产生中途 re-entry transition。

### 4. P01 是否存在明显冗余？

没有发现明显冗余。P01 距最近 Anchor 仍有 21.5 m 位置差异，且投影与 calibration 不同。

### 5. 是否值得保留 `anchor_plus_one_partial`？

值得保留，作为有效的可选 episode policy。理由是：

- 增加的渲染成本明确为 1 个 Camera。
- C1-C5 baseline 完全保持一致。
- P01 提供了可验证的局部不可见样本。
- policy 仍然是确定性的，容易复现和审计。

但它应保持“可选扩展”，不应替换 `anchor_only` 或强制所有旧任务启用。

## Recommended Next Step

建议：**保持 P01，并在更长的相同控制变量轨迹上继续观察其 coverage 和 track interruption 价值；暂不实现 P02/P03，也不引入 Dynamic Broadcast。**

优先级是先验证 P01 在更长时间窗口中是否产生更多中途出视野、重新出现和区域互补事件，再决定是否调整 P01 placement。当前结果不足以证明需要立即调整 placement，也不足以支持暂停 Partial 扩展。

本次没有实现任何下一步 Camera 功能。
