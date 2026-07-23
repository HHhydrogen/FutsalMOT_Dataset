# FutsalMOT-RL 强化学习管线说明

> 在既有 FutsalMOT 合成数据生成管线基础上新增的独立角色控制实验管线。
>
> **核心路线**：规则示范数据 → 行为克隆 (BC) → 强化学习微调 (PPO) → 2D 可视化视频 → A3.3 兼容轨迹导出
>
> **当前版本**：v1 — 控制 `Player_05`（B 队主防守者）执行跟防 `Player_01` 任务

---

## 1. 与主管线关系

本管线是完全独立的，**不修改**任何现有 FutsalMOT 主管线文件：

```
主管线 (不变):                        RL 管线 (新增):
code/                                 code/
├─ 01_generate_trajectories.py        ├─ rl_01_export_demos.py
├─ 02_run_unreal.py                   ├─ rl_01b_check_demos.py
├─ 03_check_labels.py                 ├─ rl_02_train_bc.py
├─ futsalmot/ (核心管线)               ├─ rl_03_env_sanity_check.py
│  └─ scripts/ (不动)                  ├─ rl_03_eval_bc.py
│                                     ├─ rl_04_train_ppo.py
├─ futsalmot_rl/ (RL 独立包)           ├─ rl_05_eval_rl.py
│  ├─ core/ (路径/IO/种子)             └─ rl_06_export_rl_a33.py
│  ├─ data/ (A3.3解析/示范导出/Dataset)
│  ├─ envs/ (RL环境)
│  ├─ features/ (obs/action/normalization)
│  ├─ rewards/ (奖励函数)
│  ├─ models/ (MLP策略网络)
│  ├─ training/ (BC/PPO训练)
│  ├─ rollout/ (策略执行/A3.3导出)
│  ├─ viz/ (2D球场绘制/视频录制)
│  └─ evaluation/ (评估/对比)
```

所有 RL 输出写入独立目录 `Saved/FutsalMOT_RL/`，不污染主管线的 `Saved/FutsalMOT/`。

---

## 2. 架构流程

```
已有规则 A3.3 轨迹 (configs/runs/)
        │
        ▼
┌─ rl_01_export_demos.py ──────────────────────┐
│  从 A3.3 提取 Player_05 的 observation-action │
│  输出: demo_index.json + 12个 .npz 文件        │
└──────────────────────┬────────────────────────┘
                       ▼
┌─ rl_01b_check_demos.py ──────────────────────┐
│  校验数据完整性, 生成轨迹图和示范视频           │
└──────────────────────┬────────────────────────┘
                       ▼
┌─ rl_02_train_bc.py ──────────────────────────┐
│  行为克隆: MLP(obs→action) MSE损失           │
│  输出: BC模型 + loss曲线 + 训练视频            │
└──────────────────────┬────────────────────────┘
                       ▼
┌─ rl_03_eval_bc.py ───────────────────────────┐
│  BC评估: test MSE, 位置误差, 最终视频          │
└──────────────────────┬────────────────────────┘
                       ▼
┌─ rl_03_env_sanity_check.py ──────────────────┐
│  环境完整性检查: rule > zero > random         │
└──────────────────────┬────────────────────────┘
                       ▼
┌─ rl_04_train_ppo.py ─────────────────────────┐
│  PPO微调(从BC初始化)  + 每25k步视频           │
│  输出: PPO模型 + reward曲线 + 训练视频         │
└──────────────────────┬────────────────────────┘
                       ▼
┌─ rl_05_eval_rl.py ───────────────────────────┐
│  最终评估: reward/出界/碰撞/标记距离 + 视频    │
└──────────────────────┬────────────────────────┘
                       ▼
┌─ rl_06_export_rl_a33.py ─────────────────────┐
│  合并RL轨迹→A3.3 JSON + 独立轨迹验证          │
│  输出: exported_a33/rl_*.json                │
└──────────────────────────────────────────────┘
```

---

## 3. 技术细节

### 3.1 Observation 设计 (38维)

| 特征 | 维数 | 归一化 |
|------|------|--------|
| Player_05 自身位置/速度/yaw | 6 | x/1950, y/950, v/750 |
| Player_01 目标位置/速度 | 4 | x/1950, y/950, v/750 |
| Ball 位置/速度 | 4 | x/1950, y/950, v/3000 |
| 自家球门位置 | 2 | x/1950, y/950 |
| 距离特征 (目标/球/球门) | 3 | /2200 |
| 角度特征 (目标/球 sin+cos) | 4 | 三角函数 |
| 边界距离 (左/右/上/下) | 4 | /1950, /950 |
| 球权状态 (目标持球/队友持球/自由球) | 3 | 0/1 |
| 剩余步数 | 1 | /300 |
| 事件类型 one-hot (7种) | 7 | 0/1 |

### 3.2 Action 设计

```
动作空间: Box(low=-1, high=1, shape=(2,))
  action[0] → desired_vx_norm  (-1 ~ 1 → -540 ~ 540 cm/s)
  action[1] → desired_vy_norm  (-1 ~ 1 → -540 ~ 540 cm/s)
```

运动约束:
- 加速度限制: ≤ 950 cm/s² (逐帧)
- 速度限制: ≤ 540 cm/s (Player_05 最大速度)
- 位置更新: `pos += vel * dt` (dt = 1/30s)
- 边界: clip 到球场范围 [-1950,1950] × [-950,950] + 10cm 容差

### 3.3 奖励函数 (v2 优化版)

| 组件 | 权重 | 说明 |
|------|------|------|
| `r_marking_point` | -0.004 × 距离 | 到最佳标记位置的距离 |
| `r_distance_band` | -0.003 × 偏差 | 与目标距离偏离理想值(180cm) |
| `r_goal_side` | +0.5 / -0.5 | 是否在目标与本方球门之间 |
| `r_smoothness` | -0.002 × 加速度/100 | 平滑运动惩罚 |
| `r_boundary` | **-10.0** | 出界重罚 |
| `r_boundary_proximity` | -0.02 × 接近度² | 接近边界时渐进惩罚(300cm内) |
| `r_collision` | **-5.0** | 碰撞重罚 |

### 3.4 模型结构

```text
BC: MLPPolicy
  Input(38) → Linear(128) → ReLU → Linear(128) → ReLU → Linear(2) → Tanh
  参数: 12,802

PPO: MLPActorCritic
  Actor: 同 MLPPolicy (从BC初始化)
  Critic: Input(38) → Linear(128) → ReLU → Linear(128) → ReLU → Linear(1)
  参数: 26,241 (含 log_std)
```

### 3.5 环境实现

**`FutsalDefenderFollowEnv`** (Gymnasium API)

- 控制 `Player_05`，其余7名球员+球回放规则轨迹
- `reset()`: 从 A3.3 加载全部对象位置到 frame 0
- `step(action)`: 更新 Player_05 → 读取其他对象规则位置 → 计算 reward → 检测碰撞/出界
- 每 episode 300 帧 (10秒 @ 30FPS)

---

## 4. 训练结果

### 4.1 行为克隆 (BC)

| 指标 | 值 |
|------|------|
| 训练数据 | 12 episodes (3588 transitions) |
| 训练 epochs | 20 |
| 最终 train loss | 0.0013 (MSE) |
| 最佳 val loss | 0.0008 |
| 测试 MSE | 0.0012 |
| 平均位置误差 | 0.63 cm |
| 训练时间 | 0.6 秒 |

### 4.2 PPO 强化学习 (v1 — 原始 reward)

| 指标 | 训练时 | 评估时 |
|------|--------|--------|
| Best mean reward | -288.95 | -548 |
| 出界次数 | - | 30 / ep |
| 碰撞次数 | - | 15 / ep |

### 4.3 PPO 强化学习 (v2 — 优化 reward) ★ 推荐

| 指标 | 训练时 | 评估时 |
|------|--------|--------|
| Best mean reward | -476.31 | -509 |
| **出界次数** | - | **0** ✅ |
| **碰撞次数** | - | **0** ✅ |
| 平均标记距离 | - | **208 cm** |
| 轨迹验证 ERROR | - | **0** ✅ |
| 训练时间 | 668 秒 | - |

### 4.4 三策略对比

| 指标 | 规则 baseline | BC | PPO v2 |
|------|-------------|----|--------|
| 出界 | 0 | 0 | **0** |
| 碰撞 | 0 | 23 | **0** |
| 标记距离 | ~180 cm | ~220 cm | **208 cm** |
| A3.3 验证 ERROR | 0 | 0 | **0** |

---

## 5. 文件结构

### 5.1 代码文件 (42个)

```
futsalmot_rl/
├─ __init__.py
├─ core/
│  ├─ __init__.py
│  ├─ rl_paths.py       # 独立路径定义
│  ├─ rl_io.py           # JSON/NPZ 原子读写
│  └─ rl_seed.py         # 确定性种子工具
├─ data/
│  ├─ __init__.py
│  ├─ a33_reader.py      # A3.3 只读解析器
│  ├─ demo_exporter.py   # 示范数据导出
│  └─ demo_dataset.py    # PyTorch Dataset
├─ envs/
│  ├─ __init__.py
│  └─ defender_follow_env.py  # Gymnasium 环境
├─ features/
│  ├─ __init__.py
│  ├─ obs_builder.py     # 38维 observation
│  ├─ action_builder.py  # 连续速度 action
│  └─ normalization.py   # 坐标/速度归一化
├─ rewards/
│  ├─ __init__.py
│  └─ defender_rewards.py  # 密集奖励函数
├─ models/
│  ├─ __init__.py
│  ├─ mlp_policy.py      # MLP 策略网络 + ActorCritic
│  └─ policy_io.py       # 模型保存/加载
├─ training/
│  ├─ __init__.py
│  ├─ train_bc.py        # BC 训练循环
│  ├─ train_ppo.py       # PPO 训练循环 (纯 PyTorch)
│  └─ callbacks.py       # 训练回调 + 视频录制
├─ rollout/
│  ├─ __init__.py
│  ├─ policy_rollout.py  # 策略 rollout
│  └─ export_to_a33.py   # RL → A3.3 导出 + 验证
├─ viz/
│  ├─ __init__.py
│  ├─ pitch_drawer.py    # 2D 球场绘制
│  ├─ video_recorder.py  # MP4 视频录制
│  └─ plot_metrics.py    # 训练曲线绘制
└─ evaluation/
   ├─ __init__.py
   ├─ evaluate_policy.py  # 策略评估 + BC视频回调
   └─ compare_rule_bc_rl.py  # 三路对比
```

### 5.2 入口脚本 (8个)

| 脚本 | 功能 | 典型用法 |
|------|------|----------|
| `rl_01_export_demos.py` | 导出规则示范数据 | `python rl_01_export_demos.py` |
| `rl_01b_check_demos.py` | 校验示范数据 | `python rl_01b_check_demos.py` |
| `rl_02_train_bc.py` | 训练 BC 策略 | `python rl_02_train_bc.py --epochs 100` |
| `rl_03_env_sanity_check.py` | 环境完整性检查 | `python rl_03_env_sanity_check.py` |
| `rl_03_eval_bc.py` | 评估 BC 策略 | `python rl_03_eval_bc.py` |
| `rl_04_train_ppo.py` | 训练 PPO 策略 | `python rl_04_train_ppo.py` |
| `rl_05_eval_rl.py` | 评估 RL 策略 | `python rl_05_eval_rl.py` |
| `rl_06_export_rl_a33.py` | 导出 A3.3 轨迹 | `python rl_06_export_rl_a33.py` |

### 5.3 输出目录结构

```
Saved/FutsalMOT_RL/
├─ demos/               # 示范数据 (.npz + index.json)
├─ models/              # 训练好的模型 (.pt)
│  ├─ defender_follow_bc_v1.pt
│  ├─ defender_follow_bc_v1_best.pt
│  ├─ defender_follow_ppo_v1_best.pt
│  └─ defender_follow_ppo_v1_latest.pt
├─ train_logs/          # 训练日志
│  ├─ bc/ (train_log.jsonl + loss_curve.png)
│  └─ ppo/ (train_log.jsonl + reward_curve.png)
├─ reports/             # 评估报告
│  ├─ demo_export_report.json
│  ├─ demo_check_report.json
│  ├─ env_sanity_report.json
│  ├─ bc_eval_report.json
│  ├─ rl_eval_report.json
│  ├─ rl_a33_validation_*.json
│  └─ validation/ (轨迹验证详细报告)
├─ eval/                # 轨迹图
├─ videos/              # 2D 比赛视频
│  ├─ demos/             (示范轨迹视频)
│  ├─ bc/               (BC 各 epoch 视频 + 最终视频)
│  ├─ rl_train/         (PPO 训练过程视频, 每25k步)
│  ├─ rl_eval/          (环境 sanity check 视频)
│  └─ final/            (最终 RL 视频)
├─ rollouts/            # rollout 数据 (备用)
└─ exported_a33/        # A3.3 兼容轨迹
   └─ rl_*_Player_05_a33.json
```

---

## 6. 运行指南

### 环境要求

```bash
# Python >= 3.10 (推荐 conda 环境)
pip install torch gymnasium imageio imageio-ffmpeg matplotlib numpy
```

### 快速开始 (全流程)

```bash
cd D:/projects/FustalMOT_UEDataset/Content/FutsalMOT/code

# 1. 导出示范数据 (需要先有规则 A3.3 轨迹)
python rl_01_export_demos.py

# 2. 检查示范数据
python rl_01b_check_demos.py

# 3. 训练 BC
python rl_02_train_bc.py --epochs 20

# 4. 评估 BC
python rl_03_eval_bc.py

# 5. 环境完整性检查
python rl_03_env_sanity_check.py

# 6. PPO 微调 (v2 优化版)
python rl_04_train_ppo.py --total-timesteps 500000

# 7. 评估 PPO
python rl_05_eval_rl.py

# 8. 导出 A3.3 兼容轨迹
python rl_06_export_rl_a33.py
```

### 查看训练过程

```bash
# loss 曲线
open Saved/FutsalMOT_RL/train_logs/bc/loss_curve.png

# reward 曲线
open Saved/FutsalMOT_RL/train_logs/ppo/reward_curve.png

# 观看视频
open Saved/FutsalMOT_RL/videos/final/final_rl_*.mp4
```

---

## 7. 扩展指南

### 支持更多模板

`demo_exporter.py` 和 `defender_follow_env.py` 当前支持所有模板（template 1/2/3），只要 A3.3 文件中有 `Player_05` 和 `Player_01`。直接运行 `rl_01_export_demos.py` 会自动扫描所有可用的 run 目录。

### 训练更多球员

- 修改 `rl_01_export_demos.py` 的 `--agent` 和 `--target` 参数
- 调整 `obs_builder.py` 中的 `build_observation()` 参数
- 在 `defender_follow_env.py` 中修改 `agent_id` 和 `target_id`

### 接入 UE 渲染

导出的 A3.3 轨迹文件可以通过环境变量方式接入 UE：

```bash
set FUTSALMOT_CONFIG_PATH=Saved/FutsalMOT_RL/exported_a33/rl_episode_random_0001_t1_Player_05_a33.json
```

然后在 UE Python 控制台运行 `02_run_unreal.py`。注意：**不自动更新** `pipeline_current.json`，不会覆盖原始数据。

---

## 8. 验收清单

- [x] 不修改现有 FutsalMOT 主管线
- [x] RL 管线独立运行
- [x] 能从规则 A3.3 导出示范数据
- [x] BC 能学会基本跟防 (MSE=0.001)
- [x] PPO 能在 BC 基础上微调
- [x] 训练中每 25k 步保存 2D 视频
- [x] 最终 2D 结果视频
- [x] RL rollout 导出 A3.3 兼容 JSON
- [x] 导出轨迹通过轨迹验证器 (0 ERROR)
- [x] 所有输出在独立目录 Saved/FutsalMOT_RL/

---

*生成日期：2026-07-23*
