# FutsalMOT-RL 下一阶段工作计划

> 用途：交给 Agent 继续执行下一阶段开发。  
> 当前状态：独立 RL 管线 v1 已完成，已实现规则示范导出、BC、PPO v2、2D 视频、RL rollout、A3.3 兼容轨迹导出。  
> 下一阶段目标：验证 RL 轨迹真正接入 UE 渲染与标注闭环，建立 Rule / BC / PPO 系统对比，补齐消融实验，并准备 FA-2 Goal-side Defense。

---

## 1. 当前状态

当前 FutsalMOT-RL 管线是独立于原 FutsalMOT 主管线的角色控制实验管线。

已完成内容：

```text
1. 独立代码包 futsalmot_rl/
2. 独立入口脚本 rl_01 到 rl_06
3. 从规则 A3.3 轨迹导出 Player_05 示范数据
4. 训练 BC 行为克隆模型
5. 构建 FutsalDefenderFollowEnv
6. 从 BC 初始化进行 PPO 微调
7. 训练中每 25k steps 保存 2D 视频
8. 保存最终 2D 结果视频
9. RL rollout 导出 A3.3 兼容轨迹
10. 所有输出写入 Saved/FutsalMOT_RL/
```

当前控制对象：

```text
Player_05
```

当前任务：

```text
FA-1 Defender Follow
Player_05 跟防 Player_01
```

当前结果摘要：

```text
BC:
  test MSE ≈ 0.0012
  平均位置误差 ≈ 0.63 cm

PPO v1:
  存在出界和碰撞问题

PPO v2:
  出界次数 = 0
  碰撞次数 = 0
  平均标记距离 ≈ 208 cm
  轨迹验证 ERROR = 0
```

---

## 2. 下一阶段总体目标

下一阶段命名建议：

```text
FutsalMOT-RL v1.1：UE Closed-loop Validation and Benchmarking
```

核心目标：

```text
目标 1：验证 RL A3.3 能否真正接入 UE 渲染与标注流程
目标 2：建立 Rule / BC / PPO 的标准化对比评估
目标 3：补齐 PPO from scratch 消融实验
目标 4：整理可写入论文的实验结果
目标 5：准备 FA-2 Goal-side Defense
```

本阶段的重点不是继续堆算法，而是证明：

```text
学习型控制器生成的轨迹可以服务于 FutsalMOT 合成数据集生成。
```

关键闭环：

```text
RL policy
  ↓
RL rollout
  ↓
A3.3 compatible JSON
  ↓
UE render
  ↓
bbox / layout_check
  ↓
benchmark and paper results
```

---

## 3. 工作原则

Agent 必须遵守：

```text
1. 不修改现有 01_generate_trajectories.py / 02_run_unreal.py / 03_check_labels.py。
2. 不覆盖 pipeline_current.json。
3. 不覆盖 configs/runs/ 中已有规则结果。
4. 不覆盖 Saved/FutsalMOT/ 中已有图像、标注、layout_check。
5. 所有新增结果继续写入 Saved/FutsalMOT_RL/。
6. 如需 UE 渲染，必须使用独立 seq_id 或显式环境变量指定 RL A3.3 文件。
7. 不直接批量渲染，先做 1 个 episode 的闭环验证。
8. 每个阶段必须输出 JSON / CSV / Markdown 报告。
9. 每个可视化对比任务必须输出 MP4；如果视频失败，至少保存 PNG 帧序列。
10. 新增代码继续放在 futsalmot_rl/ 或 rl_07 之后的独立入口脚本中。
```

---

## 4. 建议新增代码

新增入口脚本：

```text
Content/FutsalMOT/code/
├─ rl_07_validate_ue_closed_loop.py
├─ rl_08_compare_rule_bc_ppo.py
├─ rl_09_ablation_ppo_from_scratch.py
├─ rl_10_build_benchmark_table.py
├─ rl_11_make_comparison_videos.py
└─ rl_12_prepare_fa2_goal_side.py
```

新增模块：

```text
Content/FutsalMOT/code/futsalmot_rl/
├─ ue_bridge/
│  ├─ __init__.py
│  ├─ ue_config_export.py
│  └─ ue_closed_loop_check.py
├─ benchmark/
│  ├─ __init__.py
│  ├─ metrics.py
│  ├─ benchmark_runner.py
│  ├─ ablation_runner.py
│  └─ report_writer.py
├─ viz/
│  ├─ compare_video.py
│  └─ distance_curve_video.py
└─ academy/
   ├─ __init__.py
   └─ fa2_goal_side_defense.py
```

新增输出目录：

```text
Saved/FutsalMOT_RL/
├─ ue_closed_loop/
├─ benchmark/
├─ ablations/
├─ comparison_videos/
├─ paper_tables/
└─ academy_fa2/
```

---

## 5. Task 1：RL A3.3 → UE 闭环验证

### 5.1 目标

验证 PPO v2 导出的 A3.3 文件能否进入 UE 完成完整流程：

```text
RL A3.3
  ↓
UE Sequencer
  ↓
多视角渲染
  ↓
objects_bbox_2d_clean
  ↓
layout_check
  ↓
YOLO / MOT 可选导出
```

这是下一阶段最优先任务。

### 5.2 输入

选择一个已导出的 RL A3.3：

```text
Saved/FutsalMOT_RL/exported_a33/rl_<seq_id>_Player_05_a33.json
```

### 5.3 输出

```text
Saved/FutsalMOT_RL/ue_closed_loop/
├─ selected_rl_a33.json
├─ ue_env_command.txt
├─ ue_render_check_report.json
├─ layout_check_report.json
├─ screenshots/
└─ notes.md
```

### 5.4 执行方式

不得自动修改 `pipeline_current.json`。使用环境变量方式：

```bat
set FUTSALMOT_CONFIG_PATH=D:\projects\FustalMOT_UEDataset\Saved\FutsalMOT_RL\exported_a33\rl_<seq_id>_Player_05_a33.json
```

在 UE Python 控制台运行：

```python
py "D:/projects/FustalMOT_UEDataset/Content/FutsalMOT/code/02_run_unreal.py"
```

回到 Windows 后运行：

```bat
cd /d D:\projects\FustalMOT_UEDataset\Content\FutsalMOT\code
python 03_check_labels.py
```

### 5.5 检查项

```text
1. UE 是否成功读取 FUTSALMOT_CONFIG_PATH 指向的 RL A3.3。
2. Player_05 是否按 RL 轨迹移动。
3. Player_01~04、Player_06~08、Ball_01 是否保持规则 replay。
4. Player_05 是否出现跳帧、瞬移、方向异常。
5. bbox 是否正常。
6. layout_check 中 Player_05 框是否正确。
7. track_id 是否保持不变。
8. 是否覆盖了原始规则数据。
```

### 5.6 验收标准

```text
1. UE 渲染成功。
2. objects_bbox_2d_clean_<seq_id>.json 正常生成。
3. layout_check 正常生成。
4. Player_05 bbox 不明显错位。
5. 未覆盖任何原始规则数据。
6. 形成 ue_closed_loop_report.json。
```

---

## 6. Task 2：Rule / BC / PPO v2 三策略二维对比视频

### 6.1 目标

固定同一个 episode，对比三种策略：

```text
Rule baseline
BC policy
PPO v2 policy
```

### 6.2 新增入口

```text
rl_11_make_comparison_videos.py
```

### 6.3 输出

```text
Saved/FutsalMOT_RL/comparison_videos/
├─ rule_<seq_id>.mp4
├─ bc_<seq_id>.mp4
├─ ppo_v2_<seq_id>.mp4
├─ compare_rule_bc_ppo_<seq_id>.mp4
└─ compare_video_report.json
```

### 6.4 视频内容要求

视频中至少展示：

```text
1. 2D 五人制球场
2. 8 名球员和 Ball_01
3. Player_01 target
4. Player_05 当前策略位置
5. Rule Player_05 ghost 轨迹
6. BC Player_05 轨迹
7. PPO Player_05 轨迹
8. 跟防距离
9. out_of_bounds / collision 状态
10. 当前 frame / time
11. 当前 event type
12. 当前 possession owner
```

推荐先做单画面叠加：

```text
同一个球场中同时显示 Rule / BC / PPO 三条轨迹。
```

### 6.5 验收标准

```text
1. 能清晰分辨 Rule / BC / PPO 的 Player_05 轨迹。
2. 视频中能看到 Player_01 和 Ball_01。
3. 视频中显示跟防距离和碰撞/出界状态。
4. 至少生成 1 个完整 10 秒 episode 的对比视频。
```

---

## 7. Task 3：建立标准 Benchmark 表格

### 7.1 目标

建立统一指标，系统比较：

```text
Rule baseline
BC
PPO v1
PPO v2
```

### 7.2 新增入口

```text
rl_10_build_benchmark_table.py
```

### 7.3 输出

```text
Saved/FutsalMOT_RL/benchmark/
├─ benchmark_rule_bc_ppo.csv
├─ benchmark_rule_bc_ppo.json
├─ benchmark_summary.md
└─ benchmark_table_for_paper.csv
```

### 7.4 指标

每个策略、每个 episode 输出：

```text
seq_id
template_id
seed
policy_type
episode_reward
out_of_bounds_count
collision_count
mean_marking_distance_cm
std_marking_distance_cm
min_player_distance_cm
max_speed_cm_s
mean_speed_cm_s
speed_warning_count
turn_angle_warning_count
trajectory_error_count
trajectory_warning_count
goal_side_success_rate
time_behind_attacker_ratio
ue_render_success
layout_check_success
```

### 7.5 验收标准

```text
1. 能输出 CSV / JSON。
2. 至少包含 Rule / BC / PPO v2。
3. 如果 PPO v1 结果存在，也一起纳入。
4. 能生成 benchmark_summary.md。
```

---

## 8. Task 4：PPO from scratch 消融实验

### 8.1 目标

验证 BC 初始化是否有价值。

比较：

```text
PPO from scratch
BC + PPO
```

### 8.2 新增入口

```text
rl_09_ablation_ppo_from_scratch.py
```

### 8.3 实验设置

保持与 PPO v2 相同：

```text
相同 observation
相同 action
相同 reward v2
相同训练步数
相同 eval episodes
```

唯一差异：

```text
不加载 BC 初始模型，随机初始化 PPO policy。
```

### 8.4 输出

```text
Saved/FutsalMOT_RL/ablations/
├─ ppo_scratch_model.pt
├─ ppo_scratch_train_log.jsonl
├─ ppo_scratch_reward_curve.png
├─ ppo_scratch_eval_report.json
├─ ppo_scratch_final_<seq_id>.mp4
└─ ablation_bc_init_report.md
```

### 8.5 对比指标

```text
1. reward 收敛速度
2. 训练中出界次数
3. 训练中碰撞次数
4. 最终 mean_marking_distance_cm
5. 最终 collision_count
6. 最终 out_of_bounds_count
7. 是否通过轨迹验证
```

---

## 9. Task 5：形成论文实验结果说明

### 9.1 目标

将当前实验整理成可写入论文的结果材料。

### 9.2 输出

```text
Saved/FutsalMOT_RL/paper_tables/
├─ table_rule_bc_ppo_metrics.csv
├─ table_ablation_bc_init.csv
├─ figure_reward_curve.png
├─ figure_marking_distance_curve.png
├─ figure_compare_trajectories.png
└─ experiment_results_draft.md
```

### 9.3 `experiment_results_draft.md` 内容

应包含：

```text
1. 实验目的
2. 数据来源
3. 控制对象 Player_05
4. observation / action / reward 简述
5. Rule / BC / PPO 对比
6. PPO v1 到 PPO v2 的 reward 改进说明
7. BC 初始化消融
8. UE closed-loop 验证结果
9. 当前局限
10. 下一步 FA-2 扩展
```

---

## 10. Task 6：准备 FA-2 Goal-side Defense

### 10.1 目标

在 FA-1 稳定后，准备第二个 Academy 任务：

```text
FA-2：Goal-side Defense
```

任务目标：

```text
Player_05 不仅要跟防 Player_01，还要尽量站在 Player_01 与本方球门之间，形成防守阻挡。
```

### 10.2 新增入口

```text
rl_12_prepare_fa2_goal_side.py
```

### 10.3 新增模块

```text
futsalmot_rl/academy/fa2_goal_side_defense.py
```

### 10.4 FA-2 新增指标

```text
goal_side_success_rate
time_behind_attacker_ratio
mean_goal_line_offset_cm
shot_lane_block_score
```

### 10.5 Reward 初步设计

在 FA-1 reward 基础上提高 goal-side 权重：

```text
r_goal_side = +1.0 if good_goal_side_position else -1.0
r_behind_attacker = -1.0 if defender is behind attacker else 0
r_shot_lane_block = +0.5 * block_score
```

其余保持：

```text
r_marking_point
r_distance_band
r_smoothness
r_boundary
r_collision
```

### 10.6 本阶段只准备，不急着训练

FA-2 本阶段任务是：

```text
1. 定义指标
2. 定义 reward
3. 定义配置
4. 选择 eval episodes
5. 生成 FA-2 README
```

---

## 11. 建议执行顺序

```text
第 1 步：RL A3.3 → UE 闭环验证
第 2 步：Rule / BC / PPO v2 对比视频
第 3 步：Benchmark CSV / JSON
第 4 步：PPO from scratch 消融
第 5 步：论文实验结果整理
第 6 步：准备 FA-2 Goal-side Defense
```

不建议跳过第 1 步直接扩展 FA-2。  
如果 RL A3.3 不能稳定接入 UE，那么后续学习型控制模块对数据集生成的意义就不完整。

---

## 12. 本阶段最终交付物

完成后应交付：

```text
1. ue_closed_loop_report.json
2. UE layout_check 截图或视频证据
3. Rule / BC / PPO v2 对比视频
4. benchmark_rule_bc_ppo.csv
5. benchmark_summary.md
6. ppo_from_scratch 消融报告
7. table_rule_bc_ppo_metrics.csv
8. experiment_results_draft.md
9. FA-2 Goal-side Defense 任务定义文件
10. README_FutsalMOT_RL_v1_1.md
```

---

## 13. 本阶段成功标准

```text
1. 至少 1 个 RL A3.3 episode 成功接入 UE。
2. UE 渲染、bbox、layout_check 成功。
3. Rule / BC / PPO v2 三策略对比视频生成。
4. Benchmark 表格能系统展示三类策略差异。
5. PPO from scratch 消融完成。
6. 能说明 BC 初始化和 reward v2 的价值。
7. 形成可写入论文的实验结果草稿。
8. FA-2 任务定义完成，可以进入下一阶段训练。
```

---

## 14. 后续扩展路线

FA-2 完成后，建议按以下顺序继续：

```text
FA-3：Pass Lane Block
  Player_05 封堵 Player_01 → Player_02 的传球线路

FA-4：Two Defenders
  Player_05 + Player_06 两名防守者协同

FA-5：Four Defenders
  Player_05~08 全 B 队防守

FA-6：Off-ball Support
  Player_02~04 学习无球接应跑位

FA-7：Ball Carrier Decision
  Player_01 学习带球方向选择

FA-8：4v4 Multi-agent Decision
  多智能体联合决策 / self-play
```

---

## 15. 给 Agent 的最终提醒

本阶段核心不是继续证明“PPO 能训练”，而是证明：

```text
学习型控制器生成的轨迹可以服务于 FutsalMOT 合成数据集生成。
```

务必先完成：

```text
RL A3.3 → UE render → bbox → layout_check
```

再扩展新的 Academy 任务。
