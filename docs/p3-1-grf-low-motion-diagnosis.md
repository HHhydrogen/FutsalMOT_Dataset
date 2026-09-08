# P3-1.2 GRF Low-Motion Trigger Diagnosis

## Problem

P2-8 在 `5_vs_5`、10 FPS、300 frames 下遇到 trajectory quality blocker。Seed 42 出现 frames `147-299` 的 `15.3s` global low-motion plateau；seed 43 和 seed 45 未通过或未完全达到现有 Motion Quality Gate；seed 44 作为有效对照样本，没有出现同等规模的全局低运动平台。

本任务只做 GRF 诊断，不修改 trajectory generator、`grf_runner.py`、action wiring、difficulty、Motion Quality Gate、seed contract、scenario、Camera、UE、MRQ 或 annotation pipeline。

## Previous Hypothesis

P3-1 首轮提出的假设是：

```text
单 agent + builtin_ai control wiring 可能只正确驱动一个 agent 控制入口，
导致较长窗口内的球员进入低运动状态。
```

该假设已经通过 seed 42 的 A/B 干预被拒绝：

- Experiment A：left controls `1`，action vector length `1`。
- Experiment B：left controls `4`，action vector length `4`，controlled players 为 `L1..L4`。
- 两者的 300-frame trajectory、position delta、direction、Motion Quality 指标、`steps_left` 和 `done` 行为完全一致。

因此本报告不把 control wiring 写成 confirmed root cause。

## Seed 42 Analysis

### Low-Motion 时间段

seed 42 的现有 Motion Quality 结果：

```text
min outfield active ratio: 0.390
max outfield stationary streak: 13.5s
max goalkeeper stationary streak: 22.0s
team active coverage: 0.390
longest global low-motion plateau: 15.3s
```

global low-motion plateau 为 frames `147-299`。逐球员首次进入连续低 direction-speed 区间的时间并不相同：

| Player | First low-motion frame | Longest low-motion interval |
| --- | ---: | --- |
| L0 | 0 | 80-299, 22.0s |
| R0 | 0 | 85-299, 21.5s |
| L4 | 44 | 164-299, 13.6s |
| L2 | 25 | 181-299, 11.9s |
| L3 | 0 | 184-299, 11.6s |
| R1 | 65 | 178-299, 12.2s |
| R3 | 0 | 204-299, 9.6s |
| R2 | 3 | 210-299, 9.0s |
| L1 | 89 | 231-263, 3.3s |
| R4 | 0 | 263-299, 3.7s |

这不是一个单一球员先停止后其它球员立即停止的简单顺序。门将很早就有低运动区间，外场球员随后逐步停滞，最终在 frame `147` 之后形成全局低运动平台。

### Ball Possession

seed 42 的 possession runs：

| Frames | Owner | Duration |
| --- | --- | ---: |
| 0-49 | none | 5.0s |
| 50-85 | left player 3 | 3.6s |
| 86-94 | none | 0.9s |
| 95-299 | left player 2 | 20.5s |

在 frames `95-299`，GRF observation 持续报告 owner `(left team, player 2)`，即 `L2`。该 possession run 覆盖了后半段的大部分时间，并且包含 low-motion plateau 的全部区间。

但 ball possession 本身不是充分解释：

- frame `95` 到 `147` 期间虽然已经是 `L2` 持球，仍有多名球员继续运动；
- frame `147` 之后 ball 的水平位置和方向仍有小幅变化，直到约 frame `200` 后才接近固定；
- `L2` 不是一开始就停止，它在 frames `147-180` 仍有明显 direction-derived motion，约 frame `181` 后才进入长低运动区间。

因此更准确的表述是：

```text
长时间单一球员持球与 seed 42 的低运动平台高度重叠，
但“持球锁定”尚未被证明是独立且充分的触发条件。
```

### Player Motion

诊断同时读取：

- GRF `left_team` / `right_team` positions；
- GRF `left_team_direction` / `right_team_direction`；
- position delta 转换出的近似速度；
- `left_team_active` / `right_team_active`。

seed 42 的 position delta 与 direction-derived speed 在低运动区间同时接近零，说明不是 exporter 只错误地写入了速度字段。`active` 标志在 300 frames 内对两队所有球员都始终为 true，因此它不能作为“球员实际持续移动”的替代证据。

低运动形成过程表现为全局逐步塌缩：

- GK 在更早阶段已经长时间低速；
- L4、L2、L3、R1、R3、R2 先后进入长低速区间；
- L1 和 R4 后续也降低运动；
- 最终至少 6 名外场球员无法达到 active threshold，形成 team-level plateau。

这更接近 GRF 内置 AI 的整体状态退化，而不是单个 player 的导出异常。

### Action / AI Behavior

当前 generator 外部每 step 发送的 action 是固定值：

```text
action_set: v2
external action count: 300
external action distribution: builtin_ai = 300
```

GRF `builtin_ai` 的内部逐球员决策不会写回 public observation。当前 observation 只提供球员位置、direction、active、球权和 game state，不提供引擎内部 AI 最终选择的 action。因此：

- 可以确认外部没有出现 action index 随时间变化或 no-op action 序列；
- 不能从现有 API 诚实地计算内部 builtin AI 的 action distribution；
- 不能把低 direction 直接命名为“内部 action 长期为 idle”，因为该 action 没有暴露。

seed 42 的外部 action 没有 collapse，实际塌缩发生在 GRF engine 输出的运动结果中。

### Game State

seed 42 低运动期间的 game state：

```text
score: [0, 0] throughout
steps_left: 3000 -> 2701
game_mode: 0 throughout
done frames: none
```

低运动期间不是 episode 结束、reset、score 后重启或显式非 normal game mode。球权仍为 `L2`，球 position/direction 在早期仍有小幅变化，后期才基本停在局部区域。

因此当前没有证据表明它是一个由 `steps_left` 耗尽、score、restart 或 game-mode transition 造成的 generator-level 状态错误。它可以是 GRF 合法 gameplay state，但该 gameplay state 的引擎触发机制尚未定位到更底层。

## Seed 43/45 Comparison

### Seed 43

```text
min outfield active ratio: 0.587
max outfield stationary streak: 2.6s
max goalkeeper stationary streak: 13.4s
team active coverage: 0.760
longest global low-motion plateau: 1.3s
```

seed 43 有 13 次 possession owner transition，最长单一 player possession 为 `3.7s`，没有 seed 42 的 `20.5s` 单一持球锁定。它的主要问题是多个 player/GK 的中等长度低速区间叠加，导致 team active coverage 不足，但没有形成持续 15 秒的全局平台。

### Seed 45

```text
min outfield active ratio: 0.727
max outfield stationary streak: 1.3s
max goalkeeper stationary streak: 11.6s
team active coverage: 0.890
longest global low-motion plateau: 0.8s
```

seed 45 的最长单一 possession 为 frames `140-234`，即 `9.5s`，owner 为 right player 4。尽管 possession 明显较长，外场球员仍持续运动，global plateau 只有 `0.8s`。这直接说明“长 possession”可能增加风险，但不是单独足以造成全局低运动的条件。

## Seed 44 Comparison

seed 44 是当前有效对照：

```text
min outfield active ratio: 0.763
max outfield stationary streak: 1.1s
max goalkeeper stationary streak: 3.4s
team active coverage: 0.913
longest global low-motion plateau: 0.7s
```

possession 共有 11 次 owner transition，最长单一持球为 frames `257-299`，即 `4.3s`。其它 possession run 的持续时间主要为 `0.4-3.6s`，球权在双方和无持球状态之间多次切换。

与 seed 42 的差异：

- 没有持续 20 秒以上的单一 player possession；
- 没有 ball owner 长时间保持不变并与全局平台完全重叠；
- 外场球员在 frame `147` 附近仍普遍保持非低速状态；
- seed 44 的 player low-motion runs 多为 `0.4-1.2s`，而非持续到 episode 末尾；
- game mode 虽包含少量 `3`，但低运动期间并未形成持续的非 normal mode，且 seed 42 全程为 mode `0`，所以 game mode 不是 seed 42 低运动的必要条件。

seed 44 支持以下可能触发组合：较频繁的 possession turnover、较短的持球 run、以及球员持续收到可导致追逐/回防的 gameplay state。它不能单独证明某一个 GRF 内部条件。

## Root Cause Assessment

### Confirmed

- seed 42 的低运动不是由 exporter 末帧重复、episode reset 或 `done` 引起。
- seed 42 的 position delta 和 GRF direction 同时显示低运动，现象存在于 GRF 输出状态本身。
- seed 42 的外部 action 始终是固定 `builtin_ai`；不存在由 generator 侧逐帧切换成 idle 的证据。
- 扩展为左队 4 个显式 `builtin_ai` action 没有改善 seed 42，因此 control-wiring hypothesis 被拒绝。
- `active=true` 不能代表实际移动；seed 42 的 active flags 全程为 true，但实际速度仍长时间接近零。
- seed 42 的长 possession 与低运动平台时间上高度重叠。
- seed 45 的 `9.5s` possession 没有产生同等级 global plateau，故长 possession 不是充分条件。

### Suspected

- GRF `builtin_ai` 在特定 deterministic engine state 下可能进入一种合法但低动态的 gameplay equilibrium。
- seed 42 的长时间 `L2` possession 可能是触发组合中的重要条件，可能造成进攻推进停止、队友/对手等待或局部状态锁定。
- 触发机制可能涉及 ball control、球员相对位置、AI 决策状态和物理接触的组合，而不是单独的 possession flag。
- seed 44 未触发严重平台，可能与更频繁的 possession turnover 和更连续的球员追逐状态有关。

### Unresolved

GRF public Python observation 没有暴露 builtin AI 内部逐球员 action。因此当前不能回答：

- 内部 AI 是否长期选择 idle；
- 哪个具体内部 action/state transition 让 `L2` 保持球权；
- 是否存在 engine-level ball trapping、碰撞或 AI decision deadlock；
- seed 42 与 seed 44 的内部 AI action sequence 在 frame 147 前后具体有何不同。

## Recommended Next Step

当前最小、信息增益最高的下一步不是修改 trajectory generator，而是继续只读诊断：

1. 使用 GRF 已有 debug/dump 或 engine trace 能力，确认是否能取得 builtin AI 内部 action、active controller decision 或 ball contact state。
2. 如果内部 action 不可得，增加不改变生产行为的诊断字段收集，重点对比 seed 42/44 在 frames `120-180` 的 ball contact、球员相对距离、ball velocity 和 collision/restart 相关状态。
3. 仅在取得明确可重复的触发条件后，提出单一最小 production fix；不要先修改 difficulty、action wiring、gate 或加入 retry。

本轮不运行 UE annotation/MRQ，不启动 P2-8 Camera Distribution Stability Validation，不增加 P02/P03。

## Final Decision

1. **低运动是否来自 control wiring、builtin_ai behavior、ball possession lock、game state 或其他？**

   当前证据排除了 control wiring 作为充分原因。低运动发生在 GRF `builtin_ai` 输出的运动状态中；seed 42 的长单一 possession 与平台高度重叠，说明 ball possession lock 是重要的 suspected trigger，但 seed 45 证明长 possession 单独不是充分条件。没有证据支持 `done`、`steps_left` 或 game-mode lock 是主要原因。更准确的分类是：**疑似 GRF builtin_ai/gameplay state 与长 possession 共同形成的低动态状态，具体触发机制未确认。**

2. **是否已经找到 confirmed root cause？**

   没有。已确认的是现象和若干排除项；尚未确认 GRF 内部 AI/action/physics 的具体触发机制。

3. **下一步是否值得修改 production trajectory generator？**

   不值得立即修改。当前证据不足，不进行生产修改。

明确结论：

```text
当前证据不足，不进行生产修改。
```
