# FutsalMOT-RL 独立强化学习管线实施方案

> 文件用途：交给 Agent 直接执行开发。  
> 项目目标：在不破坏现有 FutsalMOT 合成数据生成管线的前提下，新增一套独立的 RL/IL 角色控制实验管线。  
> 核心路线：规则示范数据 → 模仿学习（Behavior Cloning）→ 强化学习微调（PPO）→ 2D 可视化视频检查 → 导出 A3.3 兼容轨迹 → 可选接入 UE 渲染与标注。  
> 新增要求：
>
> 1. 训练中途必须定期保存二维可视化比赛视频，用于检查训练过程中的策略表现。
> 2. RL 管线必须与已有 FutsalMOT 主生成管线相互独立，不得修改现有 `01_generate_trajectories.py`、`02_run_unreal.py`、`03_check_labels.py` 和已有 `futsalmot/scripts/` 的核心管线逻辑。
> 3. RL 管线可以读取已有规则管线生成的 A3.3 轨迹作为示范数据，但所有训练产物、模型、日志、可视化视频、RL rollout、评估报告必须写入独立目录。
> 4. RL 输出如需进入 UE，只能通过“导出 A3.3 兼容 JSON”的方式手动或显式接入，不能自动覆盖当前 `pipeline_current.json`。

---

## 1. 当前项目背景

当前 FutsalMOT 项目已经实现了一个基于 Unreal Engine 的 4v4 五人制足球合成数据集生成管线。当前已有系统支持：

```text
4v4 无守门员
8 名外场球员
1 个足球
4 个固定 CineCamera
10 秒回合
30 FPS
300 帧
规则事件生成
密集轨迹编译
Yaw / 动作时间轴增强
球权信息
事件标注
UE 渲染
bbox 标注
骨骼 2D 关键点
layout_check
```

当前主流程分为三步：

```text
第 1 步：Windows 轨迹生成
  01_generate_trajectories.py

第 2 步：Unreal Engine 渲染与标注
  02_run_unreal.py

第 3 步：Windows 布局检查与后处理
  03_check_labels.py
```

现有规则管线大致过程：

```text
A3.4 随机事件配置生成
  ↓
A3.1 事件验证
  ↓
A3.2 密集轨迹编译
  ↓
A3.3 Yaw / 动作 / 运球增强
  ↓
A2.5a 轨迹验证
  ↓
A3.3c 事件标注
  ↓
UE 渲染与 bbox 导出
  ↓
Windows layout_check / YOLO / MOT
```

当前已有事件类型包括：

```text
hold
move
dribble
pass
receive
shot
defend_follow
```

其中防守方目前由规则控制器控制，已实现预测性加速度受限追踪，包括：

```text
lookahead 预测
跟防距离
横向偏移
制动距离估计
加速度限制
最大速度限制
局部回避
```

本方案的目标不是推翻这些内容，而是在旁边建立一套独立的 FutsalMOT-RL 实验管线，让角色控制逐步从规则控制过渡到学习型控制。

---

## 2. 总体方案概述

本方案采用四阶段路线：

```text
阶段 1：规则示范导出
阶段 2：模仿学习 / 行为克隆
阶段 3：强化学习微调
阶段 4：RL rollout 导出与可选 UE 接入
```

整体流程：

```text
已有规则 A3.3 轨迹
        ↓
导出 Player_05 的 observation-action 示范数据
        ↓
训练行为克隆模型 BC
        ↓
定期保存 2D 可视化视频
        ↓
用 BC 作为初始策略
        ↓
PPO 强化学习微调
        ↓
训练中定期保存 2D 可视化视频
        ↓
保存最终 2D 可视化视频
        ↓
导出 RL 控制的 A3.3 兼容轨迹
        ↓
可选接入 UE 渲染与标注
```

第一版只训练：

```text
Player_05
```

角色定位：

```text
B 队 primary_press / 主防守者
```

其余对象全部沿用规则轨迹：

```text
Player_01~Player_04：规则 replay
Player_06~Player_08：规则 replay
Ball_01：规则 replay
Player_05：BC / RL 控制
```

这是一种 hybrid rollout：

```text
规则进攻 + RL 主防守者 + 规则其他防守者
```

---

## 3. 独立性要求

### 3.1 不能修改的现有文件

Agent 不得修改以下现有主流程入口：

```text
Content/FutsalMOT/code/01_generate_trajectories.py
Content/FutsalMOT/code/02_run_unreal.py
Content/FutsalMOT/code/03_check_labels.py
```

不得直接修改现有管线核心脚本：

```text
Content/FutsalMOT/code/futsalmot/scripts/run_pipeline.py
Content/FutsalMOT/code/futsalmot/scripts/generate_random_episode.py
Content/FutsalMOT/code/futsalmot/scripts/validate_episode.py
Content/FutsalMOT/code/futsalmot/scripts/compile_trajectory.py
Content/FutsalMOT/code/futsalmot/scripts/enhance_trajectory.py
Content/FutsalMOT/code/futsalmot/scripts/validate_trajectory.py
Content/FutsalMOT/code/futsalmot/scripts/generate_event_annotations.py
Content/FutsalMOT/code/futsalmot/scripts/ue_build_sequences.py
Content/FutsalMOT/code/futsalmot/scripts/convert_and_check.py
```

### 3.2 不得覆盖的现有文件

Agent 不得自动覆盖：

```text
Content/FutsalMOT/code/configs/pipeline_current.json
Content/FutsalMOT/code/configs/pipeline_config.json
Content/FutsalMOT/code/configs/runs/
Saved/FutsalMOT/annotations/
Saved/FutsalMOT/images_clean/
Saved/FutsalMOT/layout_check/
```

### 3.3 RL 独立输出目录

所有 RL 产物统一放入：

```text
D:/projects/FustalMOT_UEDataset/Saved/FutsalMOT_RL/
```

目录结构：

```text
Saved/FutsalMOT_RL/
├─ demos/
├─ models/
├─ train_logs/
├─ eval/
├─ videos/
│  ├─ demos/
│  ├─ bc/
│  ├─ rl_train/
│  ├─ rl_eval/
│  └─ final/
├─ rollouts/
├─ exported_a33/
└─ reports/
```

### 3.4 RL 独立代码目录

建议新增独立代码包：

```text
Content/FutsalMOT/code/futsalmot_rl/
```

而不是把大量 RL 代码塞进现有 `futsalmot/` 管线包。

新增结构：

```text
Content/FutsalMOT/code/
├─ futsalmot_rl/
│  ├─ __init__.py
│  ├─ core/
│  │  ├─ rl_paths.py
│  │  ├─ rl_io.py
│  │  └─ rl_seed.py
│  ├─ data/
│  │  ├─ demo_exporter.py
│  │  ├─ demo_dataset.py
│  │  └─ a33_reader.py
│  ├─ envs/
│  │  └─ defender_follow_env.py
│  ├─ features/
│  │  ├─ obs_builder.py
│  │  ├─ action_builder.py
│  │  └─ normalization.py
│  ├─ rewards/
│  │  └─ defender_rewards.py
│  ├─ models/
│  │  ├─ mlp_policy.py
│  │  └─ policy_io.py
│  ├─ training/
│  │  ├─ train_bc.py
│  │  ├─ train_ppo.py
│  │  └─ callbacks.py
│  ├─ rollout/
│  │  ├─ policy_rollout.py
│  │  └─ export_to_a33.py
│  ├─ viz/
│  │  ├─ pitch_drawer.py
│  │  ├─ video_recorder.py
│  │  └─ plot_metrics.py
│  └─ evaluation/
│     ├─ evaluate_policy.py
│     └─ compare_rule_bc_rl.py
│
├─ rl_01_export_demos.py
├─ rl_01b_check_demos.py
├─ rl_02_train_bc.py
├─ rl_03_eval_bc.py
├─ rl_03_env_sanity_check.py
├─ rl_04_train_ppo.py
├─ rl_05_eval_rl.py
└─ rl_06_export_rl_a33.py
```

所有 RL 入口脚本使用 `rl_` 前缀，与现有 `01/02/03` 主流程区分。

---

## 4. 参考 Google Research Football 的方式

本方案参考 Google Research Football 的思想，但不直接使用 GRF 作为项目环境。

参考点：

```text
1. Academy 思路：不要一开始训练完整比赛，而是从简单任务开始。
2. Benchmark 思路：每个任务要有明确评价指标。
3. Reward shaping：不用进球这种稀疏奖励，而用密集奖励引导基础行为。
4. 结构化 observation：使用球员、足球、球权、事件、边界等状态。
5. 分阶段训练：先单智能体，再多智能体。
6. 算法基线：先 BC，再 PPO，后续可扩展 IPPO / MAPPO / self-play。
```

本项目自己的任务体系命名为：

```text
Futsal Academy
```

第一版任务：

```text
FA-1：Defender Follow
控制 Player_05 跟防 Player_01
```

后续任务：

```text
FA-2：Goal-side Defense
FA-3：Pass Lane Block
FA-4：Two Defenders
FA-5：Four Defenders
FA-6：Off-ball Support
FA-7：Ball Carrier Decision
FA-8：4v4 Multi-agent Decision
```

---

## 5. 阶段 1：规则示范数据导出

### 5.1 目标

从现有 A3.3 规则轨迹中提取 Player_05 的训练样本。

新增入口：

```text
Content/FutsalMOT/code/rl_01_export_demos.py
```

核心实现：

```text
futsalmot_rl/data/demo_exporter.py
```

### 5.2 输入

读取已有规则管线输出，但只读不改：

```text
Content/FutsalMOT/code/configs/runs/**/**_a33.json
Content/FutsalMOT/code/configs/runs/**/event_annotations/events_*.json
Content/FutsalMOT/code/configs/runs/**/event_annotations/frame_states_*.jsonl
```

### 5.3 输出

输出到独立目录：

```text
Saved/FutsalMOT_RL/demos/
├─ demo_index.json
├─ demo_Player_05_<seq_id>.npz
└─ demo_export_report.json
```

每个 `.npz` 包含：

```text
obs: float32 [T-1, obs_dim]
actions: float32 [T-1, 2]
next_obs: float32 [T-1, obs_dim]
dones: bool [T-1]
frames: int32 [T-1]
positions_rule: float32 [T, 2]
target_positions: float32 [T, 2]
ball_positions: float32 [T, 2]
seq_id: str
agent_id: str
```

### 5.4 Observation 设计

第一版 `obs` 使用结构化状态，不使用图像。

Agent：`Player_05`  
Target：`Player_01`

Observation 字段：

```text
self_x_norm
self_y_norm
self_vx_norm
self_vy_norm
self_yaw_sin
self_yaw_cos

target_x_norm
target_y_norm
target_vx_norm
target_vy_norm

ball_x_norm
ball_y_norm
ball_vx_norm
ball_vy_norm

own_goal_x_norm
own_goal_y_norm

distance_to_target_norm
distance_to_ball_norm
distance_to_own_goal_norm

angle_to_target_sin
angle_to_target_cos
angle_to_ball_sin
angle_to_ball_cos

boundary_left_norm
boundary_right_norm
boundary_top_norm
boundary_bottom_norm

possession_is_target
possession_is_teammate
possession_is_free

steps_left_norm
event_type_onehot
```

坐标归一化：

```text
x_norm = x / 1950
y_norm = y / 950
vx_norm = vx / 750
vy_norm = vy / 750
distance_norm = distance / 2200
steps_left_norm = steps_left / episode_total_steps
```

### 5.5 Action 设计

第一版动作使用连续速度控制：

```text
action = [desired_vx_norm, desired_vy_norm]
```

从规则轨迹反推：

```text
vx_t = (x_{t+1} - x_t) * fps
vy_t = (y_{t+1} - y_t) * fps
desired_vx_norm = vx_t / max_speed_cm_s
desired_vy_norm = vy_t / max_speed_cm_s
```

Player_05 第一版参数：

```text
max_speed_cm_s = 540
max_acceleration_cm_s2 = 950
```

### 5.6 数据检查

新增检查入口：

```text
Content/FutsalMOT/code/rl_01b_check_demos.py
```

检查内容：

```text
1. demo_index.json 是否存在
2. 每个 npz 是否可读
3. obs 是否存在 NaN / Inf
4. actions 是否存在 NaN / Inf
5. action 是否大部分位于 [-1, 1]
6. 每个 episode 长度是否为 299 或合理值
7. 随机抽样生成 2D 轨迹图和示范视频
```

输出：

```text
Saved/FutsalMOT_RL/reports/demo_check_report.json
Saved/FutsalMOT_RL/videos/demos/demo_sample_<seq_id>.mp4
Saved/FutsalMOT_RL/eval/demo_sample_<seq_id>.png
```

---

## 6. 阶段 2：行为克隆 / 模仿学习

### 6.1 目标

训练一个神经网络策略模仿规则控制器：

```text
policy(obs) -> [desired_vx_norm, desired_vy_norm]
```

新增入口：

```text
Content/FutsalMOT/code/rl_02_train_bc.py
```

核心实现：

```text
futsalmot_rl/training/train_bc.py
futsalmot_rl/models/mlp_policy.py
```

### 6.2 模型结构

第一版使用 MLP：

```text
Input: obs_dim
Hidden: 128
Hidden: 128
Output: 2
Activation: ReLU
Output activation: tanh
```

损失函数：

```text
MSE(policy(obs), expert_action)
```

### 6.3 数据划分

按 episode 划分：

```text
train: 80%
val: 10%
test: 10%
```

不要按 frame 随机划分，避免相邻帧泄漏。

### 6.4 输出

```text
Saved/FutsalMOT_RL/models/
├─ defender_follow_bc_v1.pt
├─ defender_follow_bc_v1_config.json
├─ defender_follow_bc_v1_metrics.json
└─ defender_follow_bc_v1_best.pt
```

训练日志：

```text
Saved/FutsalMOT_RL/train_logs/bc/
├─ train_log.jsonl
├─ loss_curve.png
└─ bc_summary.json
```

### 6.5 训练中途二维视频保存要求

必须实现训练中途可视化视频保存。

视频保存策略：

```text
每隔 eval_interval_epochs 进行一次策略 rollout
默认 eval_interval_epochs = 5
每次选择固定 eval episodes
保存 2D 比赛视频
```

输出目录：

```text
Saved/FutsalMOT_RL/videos/bc/
├─ epoch_0005_<seq_id>.mp4
├─ epoch_0010_<seq_id>.mp4
├─ epoch_0015_<seq_id>.mp4
└─ final_bc_<seq_id>.mp4
```

每个视频内容：

```text
1. 2D 球场俯视图
2. 所有球员位置
3. Ball_01 位置
4. Player_05 使用醒目样式标出
5. Player_01 target 使用另一种样式标出
6. Player_05 到 Player_01 的连线
7. Player_05 的速度箭头
8. 当前 frame / time
9. 当前 action
10. 当前距离 target 的距离
11. 当前 possession owner
12. 当前事件类型
13. 如发生 out_of_bounds / collision，用文字警告
```

视频建议：

```text
格式：mp4
fps：15 或 30
分辨率：1280 × 720
背景：2D futsal pitch
```

实现建议：

```text
使用 matplotlib + imageio
或 matplotlib + OpenCV VideoWriter
```

### 6.6 BC 验收标准

```text
1. test action MSE 稳定下降
2. BC rollout 轨迹与规则轨迹接近
3. Player_05 平均位置误差 < 50 cm
4. out_of_bounds_rate = 0
5. collision_rate 不高于规则 baseline
6. A2.5a 轨迹验证无 ERROR
7. final_bc 视频能清楚看到 Player_05 的跟防行为
```

---

## 7. 阶段 3：Python RL 环境

### 7.1 目标

建立一个与 UE 独立的 2D RL 环境。

新增核心文件：

```text
futsalmot_rl/envs/defender_follow_env.py
```

环境名：

```text
FutsalDefenderFollowEnv
```

### 7.2 环境运行逻辑

```text
reset():
    选择一个 source rule episode
    读取所有对象的规则轨迹
    初始化 Player_05 位置
    其他对象进入 replay 模式
    返回 obs

step(action):
    action -> desired velocity
    限速
    限加速度
    更新 Player_05 位置
    读取其他对象当前帧规则位置
    计算 reward
    计算 collision/out_of_bounds
    返回 next_obs, reward, done, info
```

### 7.3 运动约束

RL 策略输出：

```text
a = [a_vx, a_vy] in [-1, 1]
```

反归一化：

```text
desired_vx = a_vx * max_speed_cm_s
desired_vy = a_vy * max_speed_cm_s
```

加速度限制：

```text
dv = desired_v - current_v
if ||dv|| > max_acceleration * dt:
    dv = normalize(dv) * max_acceleration * dt
```

速度限制：

```text
if ||v|| > max_speed:
    v = normalize(v) * max_speed
```

位置更新：

```text
pos_next = pos + v * dt
```

边界：

```text
x in [-1950, 1950]
y in [-950, 950]
```

如果出界：

```text
clip 到边界
info["out_of_bounds"] = True
reward += out_of_bounds_penalty
```

### 7.4 Reward 设计

第一版 dense reward：

```text
reward =
  r_marking_point
+ r_distance_band
+ r_goal_side
+ r_smoothness
+ r_boundary
+ r_collision
```

建议公式：

```text
r_marking_point = -0.004 * distance_to_marking_point
r_distance_band = -0.003 * abs(distance_to_target - ideal_distance_cm)
r_goal_side = +0.2 if defender is between target and own_goal else -0.1
r_smoothness = -0.001 * acceleration_norm
r_boundary = -2.0 if out_of_bounds else 0
r_collision = -2.0 if distance_to_any_player < collision_distance_cm else 0
```

默认参数：

```text
ideal_distance_cm = 180
collision_distance_cm = 50
```

### 7.5 环境 sanity check

新增入口：

```text
Content/FutsalMOT/code/rl_03_env_sanity_check.py
```

测试三种策略：

```text
zero_policy：不动
random_policy：随机动作
rule_replay_policy：复现规则 action
```

预期：

```text
rule_replay_policy reward > zero_policy reward > random_policy reward
```

如果不是这个结果，应先修 reward，不要开始 PPO。

输出：

```text
Saved/FutsalMOT_RL/reports/env_sanity_report.json
Saved/FutsalMOT_RL/videos/rl_eval/sanity_rule_replay_<seq_id>.mp4
Saved/FutsalMOT_RL/videos/rl_eval/sanity_random_<seq_id>.mp4
```

---

## 8. 阶段 4：PPO 强化学习微调

### 8.1 目标

在 BC 初始化基础上进行 PPO 微调。

新增入口：

```text
Content/FutsalMOT/code/rl_04_train_ppo.py
```

核心实现：

```text
futsalmot_rl/training/train_ppo.py
futsalmot_rl/training/callbacks.py
```

### 8.2 算法

第一版使用 PPO。

动作空间：

```text
Box(low=-1, high=1, shape=(2,))
```

观察空间：

```text
Box(shape=(obs_dim,))
```

初始化：

```text
加载 Saved/FutsalMOT_RL/models/defender_follow_bc_v1_best.pt
作为 PPO 初始策略
```

### 8.3 训练输出

```text
Saved/FutsalMOT_RL/models/
├─ defender_follow_ppo_v1_latest.pt
├─ defender_follow_ppo_v1_best.pt
├─ defender_follow_ppo_v1_config.json
└─ defender_follow_ppo_v1_metrics.json
```

训练日志：

```text
Saved/FutsalMOT_RL/train_logs/ppo/
├─ train_log.jsonl
├─ reward_curve.png
├─ loss_curve.png
└─ ppo_summary.json
```

### 8.4 PPO 训练中途视频保存要求

必须实现 callback：

```text
RLVideoEvalCallback
```

默认保存频率：

```text
每 eval_interval_steps = 25000 steps 保存一次
```

输出目录：

```text
Saved/FutsalMOT_RL/videos/rl_train/
├─ step_000025000_<seq_id>.mp4
├─ step_000050000_<seq_id>.mp4
├─ step_000075000_<seq_id>.mp4
└─ ...
```

每次保存内容：

```text
1. 至少保存 1 个固定 eval episode 的视频
2. 可选保存 3 个 eval episode
3. 视频中显示 reward、距离 target、collision、out_of_bounds、frame
4. 视频中同时显示规则轨迹 ghost 或 trail，用于对比
```

推荐视频信息：

```text
标题：PPO step 25000 | seq_id | episode reward
左上：frame/time/reward
右上：distance_to_target/collision/out_of_bounds
球场内：Player_05 RL 实线轨迹，Rule Player_05 半透明虚线轨迹
```

### 8.5 最终视频保存

训练结束后必须保存最终评估视频：

```text
Saved/FutsalMOT_RL/videos/final/
├─ final_rl_<seq_id>_episode_001.mp4
├─ final_rl_<seq_id>_episode_002.mp4
└─ final_rl_summary_grid.mp4
```

如实现困难，至少保存每个 eval episode 的单独视频。

### 8.6 PPO 验收标准

```text
1. episode reward 相比 BC 有提升，或至少不明显下降
2. out_of_bounds_rate = 0
3. collision_rate 不高于规则 baseline
4. mean_marking_distance 不差于规则 baseline
5. A2.5a 轨迹验证无 ERROR
6. 视频中 Player_05 行为不抖、不乱跑、不出界
7. final_rl 视频能直观看出跟防效果
```

---

## 9. 阶段 5：RL Rollout 导出为 A3.3 兼容文件

### 9.1 目标

将 RL 控制的 Player_05 轨迹合并回 A3.3 兼容 JSON。

新增入口：

```text
Content/FutsalMOT/code/rl_06_export_rl_a33.py
```

核心实现：

```text
futsalmot_rl/rollout/export_to_a33.py
```

### 9.2 输入

```text
Saved/FutsalMOT_RL/models/defender_follow_ppo_v1_best.pt
源规则 A3.3 文件
```

### 9.3 输出

不要写入原 `configs/runs/`。

输出到独立目录：

```text
Saved/FutsalMOT_RL/exported_a33/
├─ rl_<seq_id>_Player_05_a33.json
├─ rl_<seq_id>_export_report.json
└─ rl_<seq_id>_trajectory_validation_report.json
```

### 9.4 合并策略

```text
Player_01~Player_04：保留原规则轨迹
Player_06~Player_08：保留原规则轨迹
Ball_01：保留原规则轨迹
Player_05：替换为 RL rollout 轨迹
```

更新内容：

```text
Player_05 keyframes
Player_05 yaw_deg
Player_05 action_timeline
object_stats
trajectory metadata
```

第一版可以保持原事件不变，因为 Player_05 不改变球权，不直接参与 pass/receive/shot。

### 9.5 验证

导出后必须运行独立验证：

```text
1. 使用现有 validate_trajectory.py 的逻辑，但以独立方式调用
2. 不更新 pipeline_current.json
3. 报告写入 Saved/FutsalMOT_RL/reports/
```

输出：

```text
Saved/FutsalMOT_RL/reports/rl_a33_validation_<seq_id>.json
```

---

## 10. 可选 UE 接入方式

由于要求 RL 管线独立，默认不自动触发 UE。

如果用户后续需要把 RL 轨迹渲染成 UE 数据，有两种安全方式：

### 10.1 手动复制方式

将：

```text
Saved/FutsalMOT_RL/exported_a33/rl_<seq_id>_Player_05_a33.json
```

手动复制到一个新的 run 目录或手动指定给 UE。

### 10.2 环境变量方式

修改或复用 UE 脚本时，只通过环境变量显式指定：

```text
FUTSALMOT_CONFIG_PATH = Saved/FutsalMOT_RL/exported_a33/rl_<seq_id>_Player_05_a33.json
```

要求：

```text
不自动修改 pipeline_current.json
不覆盖原规则 run
不覆盖原 images_clean
```

UE 输出也建议使用独立 seq_id：

```text
seq_id = rl_<original_seq_id>_p05
```

这样输出不会覆盖已有图像和标注。

---

## 11. 2D 可视化视频设计规范

### 11.1 视频类型

本方案要求至少生成以下几类视频：

```text
1. demo_sample 视频
   展示规则示范轨迹

2. BC 训练中途视频
   展示行为克隆不同 epoch 的策略效果

3. BC 最终视频
   展示 BC policy 最终表现

4. PPO 训练中途视频
   展示强化学习不同 step 的策略效果

5. PPO 最终视频
   展示最终 RL policy 表现

6. Rule vs BC vs RL 对比视频
   可选但强烈建议
```

### 11.2 视频内容元素

每帧必须绘制：

```text
1. 五人制足球场边界
2. 中线
3. 中圈
4. 球门方向或进攻方向
5. 8 名球员
6. 足球
7. Player_05 高亮
8. Player_01 target 高亮
9. Player_05 到 Player_01 连线
10. Player_05 速度箭头
11. Player_05 历史轨迹 trail
12. Ball 历史轨迹 trail
13. 当前 frame / time
14. 当前 reward 或累计 reward
15. 当前 distance_to_target
16. 当前 collision / out_of_bounds 状态
17. 当前 possession owner
18. 当前 event type
```

### 11.3 颜色建议

颜色无需和 UE 一致，但必须稳定：

```text
A 队：蓝色系
B 队：红色系
Player_05：亮红 / 特殊边框
Player_01：亮蓝 / 特殊边框
Ball：黑色或橙色
规则轨迹 ghost：灰色半透明虚线
RL 轨迹：高亮实线
```

### 11.4 视频实现建议

文件：

```text
futsalmot_rl/viz/pitch_drawer.py
futsalmot_rl/viz/video_recorder.py
```

推荐库：

```text
matplotlib
imageio
opencv-python
numpy
```

如没有 ffmpeg，则优先使用：

```text
imageio-ffmpeg
```

也可以输出 PNG 序列作为 fallback：

```text
Saved/FutsalMOT_RL/videos/.../<video_name>_frames/
```

再单独合成 MP4。

### 11.5 视频文件命名规范

```text
demo_<seq_id>.mp4
bc_epoch_<epoch>_<seq_id>.mp4
bc_final_<seq_id>.mp4
ppo_step_<step>_<seq_id>.mp4
ppo_final_<seq_id>.mp4
compare_rule_bc_rl_<seq_id>.mp4
```

---

## 12. 评估体系

### 12.1 学习指标

```text
BC:
  train_action_mse
  val_action_mse
  test_action_mse
  mean_position_error_cm

PPO:
  episode_reward_mean
  episode_reward_std
  reward_component_marking
  reward_component_goal_side
  reward_component_collision
  reward_component_boundary
```

### 12.2 轨迹指标

```text
out_of_bounds_count
collision_count
minimum_player_distance_cm
mean_marking_distance_cm
std_marking_distance_cm
speed_warning_count
turn_angle_warning_count
trajectory_error_count
trajectory_warning_count
```

### 12.3 数据集可用性指标

```text
A3.3 文件可读
关键帧覆盖完整
track_id 不变
class_id 不变
Player_05 trajectory 合理
bbox 后续可导出
可选 UE render 成功
可选 layout_check 成功
```

### 12.4 对比对象

至少比较：

```text
Rule baseline
BC policy
RL policy
```

对比报告输出：

```text
Saved/FutsalMOT_RL/reports/compare_rule_bc_rl_<seq_id>.json
Saved/FutsalMOT_RL/reports/compare_rule_bc_rl_summary.csv
```

---

## 13. 配置文件建议

### 13.1 `configs/rl/rl_demo_export_v1.json`

```json
{
  "schema_version": "RL_DEMO_EXPORT_V1",
  "agent_id": "Player_05",
  "target_id": "Player_01",
  "source_runs_dir": "Content/FutsalMOT/code/configs/runs",
  "output_dir": "Saved/FutsalMOT_RL/demos",
  "fps": 30,
  "court": {
    "x_min_cm": -1950,
    "x_max_cm": 1950,
    "y_min_cm": -950,
    "y_max_cm": 950
  },
  "normalization": {
    "x_scale": 1950,
    "y_scale": 950,
    "player_speed_scale": 750,
    "distance_scale": 2200
  }
}
```

### 13.2 `configs/rl/bc_defender_follow_v1.json`

```json
{
  "schema_version": "BC_DEFENDER_FOLLOW_V1",
  "agent_id": "Player_05",
  "demo_index": "Saved/FutsalMOT_RL/demos/demo_index.json",
  "model_out": "Saved/FutsalMOT_RL/models/defender_follow_bc_v1.pt",
  "train": {
    "epochs": 100,
    "batch_size": 512,
    "learning_rate": 0.0003,
    "weight_decay": 0.00001,
    "train_split": 0.8,
    "val_split": 0.1,
    "test_split": 0.1,
    "eval_interval_epochs": 5
  },
  "network": {
    "hidden_sizes": [128, 128],
    "activation": "relu",
    "output_activation": "tanh"
  },
  "video": {
    "enabled": true,
    "fps": 15,
    "width": 1280,
    "height": 720,
    "output_dir": "Saved/FutsalMOT_RL/videos/bc"
  }
}
```

### 13.3 `configs/rl/ppo_defender_follow_v1.json`

```json
{
  "schema_version": "PPO_DEFENDER_FOLLOW_V1",
  "agent_id": "Player_05",
  "target_id": "Player_01",
  "bc_init_model": "Saved/FutsalMOT_RL/models/defender_follow_bc_v1_best.pt",
  "model_out": "Saved/FutsalMOT_RL/models/defender_follow_ppo_v1.pt",
  "env": {
    "fps": 30,
    "episode_length_frames": 300,
    "controlled_agents": ["Player_05"],
    "replay_agents": [
      "Player_01",
      "Player_02",
      "Player_03",
      "Player_04",
      "Player_06",
      "Player_07",
      "Player_08",
      "Ball_01"
    ],
    "max_speed_cm_s": 540,
    "max_acceleration_cm_s2": 950
  },
  "reward": {
    "marking_point_weight": -0.004,
    "distance_band_weight": -0.003,
    "goal_side_bonus": 0.2,
    "goal_side_penalty": -0.1,
    "acceleration_penalty": -0.001,
    "out_of_bounds_penalty": -2.0,
    "collision_penalty": -2.0,
    "ideal_mark_distance_cm": 180,
    "collision_distance_cm": 50
  },
  "train": {
    "algorithm": "ppo",
    "total_timesteps": 500000,
    "learning_rate": 0.0001,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "eval_interval_steps": 25000
  },
  "video": {
    "enabled": true,
    "fps": 15,
    "width": 1280,
    "height": 720,
    "output_dir": "Saved/FutsalMOT_RL/videos/rl_train"
  }
}
```

---

## 14. 建议执行顺序

Agent 应按以下顺序实现，不要跳步：

```text
Step 1：创建独立目录结构 futsalmot_rl/
Step 2：实现 rl_paths.py 和基础 IO
Step 3：实现 A3.3 只读解析器 a33_reader.py
Step 4：实现 obs_builder.py 和 action_builder.py
Step 5：实现 rl_01_export_demos.py
Step 6：实现 rl_01b_check_demos.py
Step 7：实现 pitch_drawer.py 和 video_recorder.py
Step 8：导出 demo_sample 视频
Step 9：实现 MLP policy 和 demo_dataset
Step 10：实现 rl_02_train_bc.py
Step 11：BC 训练中每 5 epoch 保存 2D 视频
Step 12：实现 rl_03_eval_bc.py
Step 13：实现 defender_follow_env.py
Step 14：实现 rl_03_env_sanity_check.py
Step 15：实现 PPO 训练
Step 16：PPO 训练中每 25000 steps 保存 2D 视频
Step 17：实现 final RL evaluation 和最终视频
Step 18：实现 RL rollout 导出 A3.3
Step 19：实现 Rule / BC / RL 对比报告
Step 20：可选接入 UE 渲染
```

---

## 15. 最终交付物

Agent 完成后应交付：

```text
1. 独立 RL 代码包 futsalmot_rl/
2. RL 入口脚本 rl_01 到 rl_06
3. demo_index.json 和若干 npz 示范数据
4. BC 模型和训练日志
5. PPO 模型和训练日志
6. 训练中途二维视频
7. 最终二维视频
8. Rule vs BC vs RL 对比视频
9. RL 生成的 A3.3 兼容 JSON
10. RL 评估报告 JSON / CSV
11. 使用说明 README_FutsalMOT_RL.md
```

---

## 16. 成功标准

第一版成功标准不是“完全踢得像真实球员”，而是：

```text
1. 不破坏现有 FutsalMOT 主管线
2. RL 管线能独立运行
3. 能从现有规则 A3.3 导出示范数据
4. BC 能学会基本跟防
5. PPO 能在 BC 基础上微调
6. 训练中途能保存二维比赛视频
7. 最终能保存二维结果视频
8. RL rollout 能导出 A3.3 兼容文件
9. 导出的轨迹能通过轨迹验证器
10. 可选接入 UE 后仍能渲染和标注
```

---

## 17. 论文表述建议

可在论文中写成：

```text
本文在既有事件驱动的 FutsalMOT 合成数据生成管线之外，构建了一套独立的学习型角色控制实验管线。该管线不直接修改原始数据生成流程，而是以规则控制器生成的可验证轨迹作为专家示范，通过行为克隆训练得到初始神经网络控制器，并进一步利用 PPO 在二维五人制足球环境中进行强化学习微调。为提高训练过程的可解释性，系统在训练过程中定期导出二维俯视比赛视频，用于观察策略在不同训练阶段的行为变化。训练完成后，学习型控制器生成的轨迹被导出为与原 A3.3 格式兼容的密集轨迹文件，从而可选择性接入 Unreal Engine 渲染和多格式标注导出流程。
```

英文：

```text
We build an independent learning-based control pipeline alongside the existing event-driven FutsalMOT synthetic data generation framework. Instead of modifying the original generation pipeline, rule-generated and validated trajectories are used as expert demonstrations to pretrain a neural controller via behavioral cloning. The controller is then fine-tuned with PPO in a lightweight 2D futsal environment. To improve interpretability during training, the system periodically exports top-down 2D match videos that visualize the intermediate and final policy behavior. The learned controller outputs dense trajectories compatible with the original A3.3 format, allowing optional integration with Unreal Engine rendering and automatic multi-format annotation export.
```

---

## 18. Agent 注意事项

Agent 执行时必须遵守：

```text
1. 不改现有 01/02/03 主脚本。
2. 不覆盖 pipeline_current.json。
3. 不覆盖已有 runs。
4. 不覆盖 Saved/FutsalMOT 下已有数据。
5. 所有 RL 输出写入 Saved/FutsalMOT_RL。
6. 所有 RL 代码放入 futsalmot_rl。
7. 每个阶段必须有独立入口脚本。
8. 每个训练阶段必须保存二维视频。
9. 出现视频合成失败时，至少保存 PNG 帧序列。
10. 任何导出的 RL A3.3 文件必须先通过验证，再考虑 UE 渲染。
```

---

## 19. 最小可运行闭环

### 19.1 第一阶段最小闭环

```text
A3.3 规则轨迹
  ↓
rl_01_export_demos.py
  ↓
demo_index.json + npz
  ↓
rl_01b_check_demos.py
  ↓
demo_sample_2d.mp4
```

### 19.2 第二阶段最小闭环

```text
demo_index.json
  ↓
rl_02_train_bc.py
  ↓
defender_follow_bc_v1.pt
  ↓
每 5 epoch 保存一个 bc_epoch_xxxx.mp4
  ↓
rl_03_eval_bc.py
  ↓
bc_final_<seq_id>.mp4
```

### 19.3 第三阶段最小闭环

```text
BC policy
  ↓
FutsalDefenderFollowEnv
  ↓
rl_03_env_sanity_check.py
  ↓
rule replay reward > zero policy reward > random policy reward
  ↓
rl_04_train_ppo.py
  ↓
每 25000 steps 保存一个 ppo_step_xxxxxx.mp4
  ↓
final_rl_<seq_id>.mp4
```

### 19.4 第四阶段最小闭环

```text
PPO policy
  ↓
rl_06_export_rl_a33.py
  ↓
Saved/FutsalMOT_RL/exported_a33/rl_<seq_id>_Player_05_a33.json
  ↓
独立轨迹验证通过
  ↓
Rule / BC / RL 指标对比报告生成
```

完成以上四个阶段后，FutsalMOT-RL 第一版视为完成。
