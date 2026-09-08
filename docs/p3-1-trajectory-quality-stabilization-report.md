# P3-1 Trajectory Quality Stabilization

## Problem

P2-8 在 `5_vs_5`、10 FPS、300 frames 下观察到较长低运动区间。Seeds `42, 43, 45` 未通过现有 Motion Quality Gate，只有 seed `44` 通过。

本轮 P3-1 先验证一个控制连接假设，不改变生产默认行为：

```text
Initial Hypothesis:
“单 agent + builtin_ai”控制拓扑可能只正确驱动一个 agent 控制入口，
导致部分或全部球员在较长窗口内进入低运动状态。
```

本轮不修改 Camera、UE、MRQ、annotation、Motion Quality Gate、seed contract 或场景定义。

## Root-Cause Diagnosis

### Existing Control Path

当前 `grf_runner.py` 通过 GRF `create_environment()` 创建：

```text
number_of_left_players_agent_controls = 1
number_of_right_players_agent_controls = 0
action_set = v2
```

`5_vs_5.py` 将两队门将创建为 `controllable=False`，其余每队 4 名球员为 controllable。因此在当前配置下，单 agent 观察数量和 action vector 长度均为 `1`。

GRF API 的实际顺序来自 `FootballEnv._convert_observations()`：左队控制槽按场景 player order 输出。对 `5_vs_5` 来说，左队顺序是 `L1, L2, L3, L4`；门将 `L0` 不在 controllable player 列表中。

### Evidence From P2-8 Artifacts

- Seed 42 的最长 global low-motion plateau 为 frames `147-299`，即 `15.3s`。
- 同一段时间内多名外场球员的 position delta 和 GRF direction 同时接近或等于零。
- Seeds `42-45` 的 `steps_left` 都从 `3000` 持续递减到 `2701`。
- Seeds `42-45` 均没有 `done`，因此该现象不是 episode 结束后的 reset 或末帧重复。
- `left_team_active` 和 `right_team_active` 持续为真，但这不能替代实际速度证据。
- 修改 difficulty 参数的前置对照没有改变任何轨迹指标，故没有把 difficulty 调参作为 P3-1 修复方向。

这些证据支持“GRF 内置 AI 行为在部分 seed 下进入低运动状态”的现象，但在 A/B 干预前不能确认其具体控制 wiring 根因。

## Control-Wiring Experiment

本轮只对 seed `42` 执行 A/B 对照，因为它具有最明显的 `15.3s` global low-motion plateau。

### Experiment A

保持当前生产路径：

```text
left agent controls: 1
right agent controls: 0
environment observation length after reset: 1
action space: Discrete(20)
action vector sent each step: [builtin_ai]
action vector length: 1
```

### Experiment B

使用 GRF 已有 multi-agent action 接口，让左队所有当前 controllable player 显式接收 `builtin_ai`：

```text
left agent controls: 4
right agent controls: 0
environment observation length after reset: 4
action space: MultiDiscrete([20, 20, 20, 20])
action vector sent each step: [builtin_ai, builtin_ai, builtin_ai, builtin_ai]
action vector length: 4
controlled player order: L1, L2, L3, L4
```

右队控制语义没有改变，仍由 GRF 内置逻辑控制。两组都使用 `action_set=v2`、相同 root seed 和相同 GRF engine seed。

## Evidence Before/After

实验 A/B 均运行 300 frames，`dt=0.1s`，没有 `done`，且 `steps_left` 都是 `3000 -> 2701`。

| Metric | A: 1 control | B: 4 controls | Difference |
| --- | ---: | ---: | ---: |
| Min outfield active ratio | 0.390 | 0.390 | 0.000 |
| Max outfield stationary streak | 13.5s | 13.5s | 0.0s |
| Max GK stationary streak | 22.0s | 22.0s | 0.0s |
| Team active coverage | 0.390 | 0.390 | 0.000 |
| Longest global low-motion plateau | 15.3s | 15.3s | 0.0s |
| Action vector length | 1 | 4 | expected API difference |
| Frame count | 300 | 300 | 0 |
| Done steps | none | none | unchanged |

逐球员的 mean speed、active ratio、stationary streak、max speed、position delta 和 GRF direction-derived motion 也完全一致。seed 42 的低运动区间仍然位于约 frames `147-299`。

## Confirmed Root Cause 或 Hypothesis Rejected

```text
Control-wiring hypothesis: rejected by seed-42 intervention.
```

虽然 B 确实按 GRF API 使用了合法的 4-player action vector，但它没有改变 seed 42 的轨迹内容或 Motion Quality 指标。因此目前不能把“单 agent control wiring”写成 confirmed root cause。

当前保留的事实结论是：

- GRF `builtin_ai` 在部分 deterministic seed 下会产生长时间低运动状态；
- 单 agent 到四 controllable-player 的显式 action wiring 不是该现象的充分修复；
- 仍需继续定位 GRF engine / scenario behavior 的具体触发条件，但本轮按要求停止，不堆参数或引入人工运动逻辑。

## Minimal Production Change

本轮没有生产代码修改。

未修改：

- `src/grf_ue_bridge/grf_runner.py`
- `src/grf_ue_bridge/seeds.py`
- `src/grf_ue_bridge/motion_quality.py`
- scenario、field dimensions、action set v2
- Camera、UE、MRQ、annotation 和任何 pipeline contract

由于干预实验未改善结果，不执行 42-45 扩展对照，也不进入 production wiring change 阶段。

## Regression Tests

本轮没有生产行为修改，因此没有新增回归测试。实验本身验证了：

- GRF API 对应的 action vector 长度为 `1` 与 `4`；
- B 的 observation/action 顺序为左队 controllable player `L1..L4`；
- 两种 wiring 都保持 300 frames；
- 两种 wiring 都保持现有 seed 派生和 `steps_left` 行为；
- 没有引入 seed-specific branch。

现有 seed 派生测试和 Motion Quality Audit 测试未被修改。

## Multi-Seed Validation

按实验判定规则，seed 42 没有明显改善，因此本轮不运行 seeds `43-49` 的扩展验证，也不重新计算 `valid >= 6/8` 目标。

P2-8 基线仍为：

| Seed | Baseline decision |
| ---: | --- |
| 42 | invalid |
| 43 | invalid |
| 44 | valid |
| 45 | conditional invalid |

P3-1 的 42-49 目标状态：

```text
not evaluated
```

## Determinism

实验 A/B 使用同一个 root seed `42` 和相同派生 `grf_game_engine_seed`。A/B 的 300 帧轨迹内容和所有比较指标一致，说明本次对照没有引入非确定性差异。

本轮没有执行报告要求的“修改前失败、修改后通过 seed”重复检查，因为没有发生生产修改，也没有出现修改后通过的 seed。

## Remaining Limitations

- 当前仍不知道 GRF 内置 AI 进入低运动状态的更深层触发条件。
- difficulty 属性覆盖的现有实现路径仍需单独修复或验证，但它不是本轮允许的修复方向，也没有作为 P3-1 生产变更执行。
- 不能仅凭 `active=true` 判断球员在运动；必须继续使用 position delta、direction 和 Motion Quality Audit。
- 不能把 seed 44 的有效结果外推为 generator 在长窗口下普遍稳定。
- 本轮没有运行 UE annotation、MRQ 或 Camera Distribution 验证。

## Final Decision

1. **trajectory generation 的主要低运动根因是什么？**

   尚未确认。当前证据表明是 GRF `builtin_ai` 在部分 seed 下产生低运动状态；“单 agent control wiring” 假设经 A/B 干预后被拒绝，不能写成 confirmed root cause。

2. **本次具体修改了什么？**

   没有修改生产代码。只执行了合法的 GRF control-wiring A/B 临时对照，并更新本报告。

3. **是否保持 deterministic？**

   是。A/B 使用同一 root seed 和同一派生 engine seed，300 帧 trajectory 内容完全一致；本轮没有改变 seed contract。

4. **seeds 42-49 中多少通过现有 gate？**

   P3-1 未运行 42-49 扩展验证，因此不能报告新的 42-49 通过数量。已知 P2-8 基线中 `1/4` 通过，即只有 seed `44`。

5. **是否达到 `>=6/8` 的工程目标？**

   未评估，且当前没有证据表明已达到。

6. **是否已经足以重新启动 P2-8 Camera Distribution Stability Validation？**

   否。control-wiring intervention 没有改善 seed 42，trajectory quality blocker 仍然存在。按要求本轮到此停止，不继续 P2-8，不增加 P02/P03。
