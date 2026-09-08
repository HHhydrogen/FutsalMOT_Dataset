# Problem Statement

当前 FutsalMOT 使用 Google Research Football `5_vs_5` 的 `builtin_ai` 作为主要轨迹源。P3-1 已确认低运动现象出现在 GRF 输出中，且已测试的单智能体控制接线干预没有改善该现象。P3-2 在固定 seeds `42-59`、`300` 帧、`10` FPS 下只得到 `1/18` strict-valid trajectory。因此本阶段评估现有 GRF 默认源、当前环境中的 GRF-native 选项、UE-native 选项和轻量项目自有原型，形成一个源层架构决策。

本报告只评估来源可行性，不修改生产生成器、默认配置、seed contract、Motion Quality thresholds、UE 资产或下游管线，也不执行 UE annotation、MRQ、Camera Distribution 或 P01 验证。

# P3-2 Failure Characterization

P3-2 使用现有 strict criteria，未增加 GK gate：

- 外场 active ratio `>= 0.75`：`13/18` 失败，失败频率 `72.2%`。
- 外场最长 stationary streak `<= 2.0s`：`10/18` 失败，失败频率 `55.6%`。
- team active outfield coverage `>= 0.90`：`16/18` 失败，失败频率 `88.9%`。
- severe global low-motion 使用现有 `longest_global_low_motion_plateau_s >= 5.0s` 描述性分类：seeds `[42, 47, 50, 56, 58]`，共 `5/18`。

失败组合按现有三项 strict criteria 统计：

| 失败组合 | Seeds | 数量 |
| --- | --- | ---: |
| 三项同时失败 | `42, 43, 47, 50, 51, 55, 56, 58` | 8 |
| active ratio + team coverage | `45, 49, 52, 59` | 4 |
| 仅 active ratio | `46` | 1 |
| 仅 team coverage | `48, 54` | 2 |
| stationary streak + team coverage | `53, 57` | 2 |

单 criterion near miss 为 seeds `[46, 48, 54]`。这些分类仅描述失败形态，不改变既有 validity 语义。Seed `44` 是唯一 strict-valid seed；P3-2 因 `1/18 < 5` 未形成目标 cohort。

GK stationary streak 只作为描述性指标记录，从未作为新的 strict gate。P3-2 中唯一 strict-valid seed `44` 的 GK stationary streak 为 `12.2s`，这正说明不能把该指标追加为阻断条件。

# Candidate Trajectory Sources

## Current GRF builtin_ai

当前基线是 pinned `.external/google-research-football` 中的 `builtin_ai` 路径。GRF action evidence 为 `.external/google-research-football/gfootball/env/football_action_set.py` 中的 `action_builtin_ai`，并由 inventory 记录为 `action_set_v2_index_19`。该路径可运行、可生成现有 `frames.jsonl`，并保持当前 contract compatibility，但稳定性不足。

## GRF-native alternatives

当前仓库和环境中的 source inventory 如下：

| Source | 当前路径/API evidence | Runtime disposition | 结论 |
| --- | --- | --- | --- |
| `bot` | `.external/google-research-football/gfootball/env/players/bot.py`，`sample_bot_player` | runtime API 存在 | `NOT_A_COMPLETE_POLICY_SOURCE`，且只支持 sample bot player 路径 |
| `replay` | `.external/google-research-football/gfootball/env/players/replay.py`，`replay_player_requires_trace` | replay API 存在，但当前无 trace | `REPLAY_ONLY`，不能作为当前可运行新源 |
| `ppo_checkpoint` | `.external/google-research-football/gfootball/env/players/ppo2_cnn.py`，`ppo2_cnn_player_requires_checkpoint` | 没有可用本地 checkpoint/runtime 证据 | `INFEASIBLE_IN_CURRENT_ENVIRONMENT` |
| `grf_marl_policy` | `.external/GRF_MARL` 的 IPPO/MAPPO/HAPPO rollout framework | 所需 runtime dependencies/policies 不可用 | `INFEASIBLE_IN_CURRENT_ENVIRONMENT` |

权威不可行性 artifact 为 `.futsalmot/p3_3_feasibility/grf_native_alternative.json`，其中 `experiment_run=false`，证据为 `no local checkpoint/trace; GRF_MARL runtime dependencies/policies unavailable`。因此不能把文档中存在的 API、`desc.pkl` 或研究框架描述成已运行的更可靠 GRF source。

## UE-native source

UE 侧已有 source-neutral playback：`ue/import_grf_episode.py` 读取 `meta.json` 和 `frames.jsonl`，将 `players[*].position_m` 与 `ball.position_m` 映射为 Level Sequence transform keys；`ue/dataset_export.py` 也以相同 episode artifact 为输入。该事实支持 playback contract，但仓库没有完整、确定性的 5v5 simulation，尤其没有已验证的 possession、ball interaction 和 episode-level reproducibility facility。Task 4 的静态 assessment 为：

```text
playback_contract_supported: true
complete_5v5_simulation_available: false
```

## Project-owned prototype

项目自有原型位于诊断工具 `src/grf_ue_bridge/tools/p3_3_trajectory_source_feasibility.py`，输出 artifact 位于 `.futsalmot/p3_3_feasibility/project_owned_prototype/`。它是确定性的轻量运动学 prototype，包含固定 `L0..L4`、`R0..R4`、`BALL`、team-side separation、bounded movement 和基于空间规则的 possession handoff。其所有结果明确标注 `prototype=true`、`production=false`。

该原型是诊断证据，不是已批准的生产 simulation，也不是完整 tactical AI。

## External source status

本阶段没有新增第三方依赖、下载 checkpoint、训练策略或执行广泛迁移。除 pinned `.external/google-research-football` 和 `.external/GRF_MARL` 的现有源证据外，没有发现当前环境中可直接复用并已验证的外部完整 source。

# Evaluation Criteria

所有可运行候选按相同标准比较：

1. Motion Quality：在不改 gate 的情况下通过既有 strict criteria 的能力。
2. Determinism：同 seed 和设置生成完全相同的 trajectory artifact。
3. Controllability：玩家运动、活动水平、possession 和 episode 时长是否可控且不依赖 invalid-seed filtering。
4. Futsal plausibility：5v5 空间分离、球员与球交互、进攻/防守运动和非随机行为。
5. Integration cost：对 `frames.jsonl`、actor mapping、UE playback、annotation 和 Camera pipeline 的改动成本。
6. Contract compatibility：是否复用现有 artifact 与下游 contract。
7. Reproducibility：是否适合论文和数据集再生成。
8. Maintenance risk：是否依赖难以调试或不可维护的行为。
9. Dataset value：是否提供持续且有价值的 MOT observation，而不是只通过 gate。

# Feasibility Experiments

本阶段只使用已完成的 bounded artifacts，不运行新的 candidate experiment。权威设置为：

```text
scenario: 5_vs_5
frames: 300
FPS: 10
seeds: 42, 43, 44, 45
```

基线 artifact：`.futsalmot/p3_3_feasibility/grf_builtin_ai/candidate_results.json`。

原型 artifact：`.futsalmot/p3_3_feasibility/project_owned_prototype/candidate_results.json`。

GRF-native infeasibility artifact：`.futsalmot/p3_3_feasibility/grf_native_alternative.json`。

确定性 repeat artifact：

- GRF：`.futsalmot/p3_3_feasibility/repeats/grf_builtin_ai/seed_42_repeat.json` 和 `seed_44_repeat.json`。
- Prototype：`.futsalmot/p3_3_feasibility/prototype_repeats/project_owned_prototype/seed_42_repeat.json` 和 `seed_44_repeat.json`。

未运行 Camera、annotation、MRQ、P01 或 UE candidate validation。

# Results

## Bounded candidate comparison

| Candidate | Artifact | Strict-valid | Global mean speed | Team active coverage | Longest low-motion plateau | Contract | Generation seconds |
| --- | --- | ---: | --- | --- | --- | --- | --- |
| `grf_builtin_ai` | `.futsalmot/p3_3_feasibility/grf_builtin_ai/candidate_results.json` | `1/4` | seeds `42-45`: `0.597, 1.361, 1.544, 1.511` | `0.390, 0.760, 0.913, 0.890` | `15.3s, 1.3s, 0.7s, 0.8s` | `4/4 compatible` | `1.447136, 0.725911, 0.801428, 1.174897` |
| `project_owned_prototype` | `.futsalmot/p3_3_feasibility/project_owned_prototype/candidate_results.json` | `4/4` | seeds `42-45`: `1.043, 1.113, 1.106, 1.078` | `0.940, 0.937, 0.947, 0.953` | `0.6s, 0.7s, 0.7s, 0.4s` | `4/4 compatible` | `0.038442, 0.035122, 0.035131, 0.036393` |

两项候选均为 `300` 帧，duration 为 `29.9s`，均保持固定十个 player IDs 和 ball contract。Prototype 在该 bounded sample 中明显改善 Motion Quality 和生成时间，但该结果只证明 contract、determinism 和 gate behavior，不证明完整 futsal plausibility 或生产 readiness。

## Deterministic repeats

| Candidate | Seed | Initial SHA-256 | Repeat SHA-256 | Byte-identical | Repeat generation seconds |
| --- | ---: | --- | --- | --- | ---: |
| `grf_builtin_ai` | 42 | `9aef1323c3866beac9dcc4bde41dfe01345d2f96c0089db58feef547dd59afe3` | `9aef1323c3866beac9dcc4bde41dfe01345d2f96c0089db58feef547dd59afe3` | yes | `1.511128` |
| `grf_builtin_ai` | 44 | `62e0703cb78e2a572d47088d330911d7c31be92761c6f7f05774929226484f57` | `62e0703cb78e2a572d47088d330911d7c6f7f05774929226484f57` | yes | `1.568704` |
| `project_owned_prototype` | 42 | `32e6b0e6668b838defd737d8a5ed9c8bf2c7a2e9c3eb7101084686e843fd2e2f` | `32e6b0e6668b838defd737d8a5ed9c8bf2c7a2e9c3eb7101084686e843fd2e2f` | yes | `0.038431` |
| `project_owned_prototype` | 44 | `b69a3ff8459744fa263a123909a4ba7e2f5d70202fa3fc931a2d63698d525195` | `b69a3ff8459744fa263a123909a4ba7e2f5d70202fa3fc931a2d63698d525195` | yes | `0.038008` |

# GRF Assessment

`builtin_ai` 仍然具有研究和 comparison value：它是当前已验证、契约兼容、可复现的 GRF baseline，且两个 repeat seed 均字节级一致。但其默认稳定性不足：P3-2 为 `1/18` strict-valid，bounded P3-3 baseline 为 `1/4`，并出现 `42` 的 `15.3s` 全局低运动 plateau。该问题不是由缺少下游 contract 造成，而是源输出本身的运动质量不稳定。

当前环境没有一个已经验证、无需大规模依赖迁移的 GRF-native replacement。bot 不是完整策略源，replay 需要 trace，PPO 需要 checkpoint/runtime，GRF_MARL 需要不可用的 runtime dependencies/policies。因此不能以 GRF-native 选项存在于源码中为理由保留 builtin_ai 的默认地位，也不能声称已运行这些不可用选项。

# Replacement Assessment

项目自有 prototype 在 `42,43,44,45` 上达到 `4/4` strict-valid、`4/4` contract-compatible，并在 seeds `42` 和 `44` 上 byte-identical repeat。它还显示出较低且稳定的生成耗时，以及可控的团队分离、目标运动和空间 possession 规则。

但是，这个 prototype 明确是 `prototype=true`、`production=false`。当前 artifact 没有 UE playback、annotation、Camera 或 P01 结果，也没有足够证据证明完整 futsal plausibility、长期行为多样性、边界情况和生产维护成本。依据任务规则，`4/4` gate 结果单独不足以支持 `REPLACE_GRF`。

UE-native 方案也不能作为立即 replacement：现有 UE playback 现实可行，但完整确定性 5v5 simulation 尚不存在，状态、possession、ball interaction 和可复现 episode 仍需新建并验证。

# Contract Impact

在候选源只替换 source layer、继续输出现有 artifact 的前提下，以下 contract 可以保留：

- `meta.json` 的 timing、field 和 entity metadata。
- `frames.jsonl` episode/frame representation。
- 固定 entity IDs `L0..L4`、`R0..R4`、`BALL`。
- `futsalmot_seed_v1` 或明确兼容的 deterministic seed contract。
- actor mapping。
- UE actor-state playback 和 Level Sequence generation。
- Camera Distribution。
- annotation 和 MOT GT generation。
- Run Manifest、Pipeline State、ValidationResult、Audit/Cleanup contract。

本阶段没有证据要求修改这些下游 contract；也没有执行下游验证，所以“可保留”表示接口边界评估，不表示新 source 已完成端到端验收。

# Decision

## `GRF_OPTIONAL`

1. **当前 GRF builtin_ai 是否适合作为默认 trajectory source？**

   不适合。它仍是可复现的研究 baseline，但当前质量稳定性不足：P3-2 为 `1/18` strict-valid，bounded baseline 为 `1/4`，并有严重 global low-motion cases。

2. **当前环境是否存在更可靠的 GRF-native path？**

   没有已验证可运行的 path。bot 不是完整 policy source，replay 缺少 trace，PPO 缺少 checkpoint/runtime，GRF_MARL 缺少可用 runtime dependencies/policies。不可用选项没有被实验运行。

3. **UE-native trajectory generation 是否现实？**

   作为架构方向是可能的，因为 UE playback contract 已支持 source-neutral frame positions；作为当前可用的完整生成源不现实，仓库没有完整确定性 5v5 simulation。Task 4 状态为 `ARCHITECTURALLY_POSSIBLE_NOT_READY`。

4. **轻量 project-owned generator 是否现实？**

   作为下一阶段 prototype 方向现实。当前诊断 prototype 已证明确定性、contract compatibility 和 bounded Motion Quality 可行，但尚未证明 full futsal plausibility 或 production readiness。

5. **是否应正式替换 GRF？**

   现在不应正式替换。Prototype 的 `4/4` gate 结果单独不足以支持 `REPLACE_GRF`，因此当前只将 GRF 降为 optional research/comparison source，不执行迁移。

6. **哪些现有 contract 可以保留？**

   可以保留 `meta.json`、`frames.jsonl`、固定 IDs、seed contract、actor mapping、UE playback/Level Sequence、Camera Distribution、annotation/MOT GT，以及 Run Manifest、Pipeline State、ValidationResult、Audit/Cleanup。这里只保留 source boundary，不宣称新 source 已完成端到端验证。

7. **下一阶段应是 GRF repair 还是 Replacement Prototype？**

   应是 `Replacement Prototype`，因为当前 GRF-native alternatives 在环境中不可运行，而 project-owned prototype 已显示明确的 bounded feasibility signal。下一阶段仍需先验证 futsal plausibility、行为多样性、长序列稳定性和 production boundary，再决定是否迁移；本阶段在此 decision 后停止，不自动启动下一阶段。

# Recommended Next Phase

进入受控的 Replacement Prototype 阶段，而不是修改当前生产 GRF 默认路径。下一阶段应只在明确范围内扩展当前 diagnostic prototype，优先验证 possession/球员交互、攻防行为、长期多样性、边界状态、与现有 `frames.jsonl` 的端到端 source-layer 接口和必要的 UE playback probe；不得把当前 prototype 直接提升为 production，也不得以 retry、invalid-seed filtering 或 post-hoc repair 掩盖失败。

本阶段已在 decision 后停止，不自动开始下一阶段。
