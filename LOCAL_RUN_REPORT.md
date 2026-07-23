# FutsalMOT-RL Local Run Report

> 生成日期: 2026-07-24
> 最后提交: 9c808fc

---

## 1. 环境信息

| 项目 | 值 |
|------|-----|
| Python | C:\Python314\python.exe — 3.14.4 |
| pip | 26.0.1 |
| 安装方式 | `pip install -e .` (editable) |
| UE 工程 | D:\projects\FustalMOT_UEDataset |
| .uproject | FustalMOT_UEDataset.uproject |
| 仓库位置 | D:\projects\FustalMOT_UEDataset\Content\FutsalMOT\code |

### 依赖版本

| 包 | 版本 |
|----|------|
| torch | 2.13.0+cpu |
| gymnasium | 1.3.0 |
| numpy | 2.4.4 |
| matplotlib | 3.11.1 |

---

## 2. 删除的 CI 文件

- `.github/workflows/ci.yml` — GitHub Actions workflow 已删除
- `.github/` 目录已删除

---

## 3. 删除的 stub 命令

- `futsalmot_rl/commands/benchmark.py` — 未实现，删除
- `futsalmot_rl/commands/export.py` — 未实现，删除
- `futsalmot_rl/commands/comparison.py` — 未实现，删除
- `futsalmot_rl/commands/ue.py` — 未实现，删除
- `futsalmot_rl/commands/train.py` — 保留但 CLI 不再注册，通过 `scripts/train_bc.py` 使用

当前 CLI 只保留 `demos` 和 `evaluate` 两个真实命令。

---

## 4. 修改文件

| 文件 | 变更 |
|------|------|
| `futsalmot_rl/cli.py` | 使用 `local_config` 替代 `ProjectPaths`，移除 stub |
| `futsalmot_rl/commands/demos.py` | 使用 project_root 字符串，移除 ProjectPaths 依赖 |
| `futsalmot_rl/commands/evaluate.py` | 同上 |
| `futsalmot_rl/core/local_config.py` | **新增** — 读取 `configs/local_paths.json` |
| `futsalmot_rl/models/mlp_policy.py` | **新增** `get_value()` 方法 |
| `futsalmot_rl/training/train_ppo.py` | 修复 explained_variance 计算 |
| `futsalmot_rl/training/train_ppo.py` | 去除 unused `all_value_preds` |
| `configs/local_paths.example.json` | **新增** — 本地配置模板 |
| `configs/local_paths.json` | **新增** — 当前机器配置，已 gitignore |
| `scripts/train_bc.py` | **新增** — BC 训练 thin wrapper |
| `scripts/local_check.ps1` | **新增** — 本地检查脚本 |
| `scripts/local_rl_smoke.ps1` | **新增** — RL smoke 测试脚本 |
| `.gitignore` | 排除 `local_paths.json` |

---

## 5. 本地路径配置

配置在 `configs/local_paths.json`：

```json
{
  "ue_project_root": "D:/projects/FustalMOT_UEDataset",
  "uproject_file": "D:/projects/FustalMOT_UEDataset/FustalMOT_UEDataset.uproject"
}
```

程序优先读取该文件，CLI `--project-root` 参数可覆盖。

---

## 6. 执行的命令与结果

### 6.1 本地检查

```bash
python -m compileall futsalmot futsalmot_rl -q
→ (no output, PASS)

python -m pytest tests/unit -q --tb=short
→ 46 passed in 2.14s

python -c "from futsalmot_rl.core.local_config import load_local_paths; ..."
→ OK
```

### 6.2 Demo 导出

```bash
python tools/legacy/rl_01_export_demos.py --max-episodes 2
→ Exported 2 demos, 598 transitions
```

### 6.3 环境测试

```bash
python -c "from futsalmot_rl.envs.defender_follow_env import ..."
→ Obs shape: (38,), dtype: float32
→ 30 env steps OK
```

### 6.4 BC smoke training (2 epochs)

```bash
python scripts/train_bc.py --epochs 2 --no-video
→ Epoch 1: train_loss=0.137 val_loss=0.099
→ Epoch 2: train_loss=0.082 val_loss=0.059
→ Training complete (0.0s)
→ Best val loss: 0.059
```

模型: `Saved/FutsalMOT_RL/models/defender_follow_bc_v1.pt`

### 6.5 PPO smoke training (2048 steps)

```bash
python -c "from futsalmot_rl.training.train_ppo import PPOTrainer; ..."
→ Iter 1: step=2048 mean_reward=-1334 pi_loss=-0.0009 v_loss=6160
→ Training complete (0.9s)
→ Best mean reward: -1334
```

模型: `Saved/FutsalMOT_RL/models/defender_follow_ppo_v1_best.pt`

### 6.6 A3.3 导出

```bash
python -c "from futsalmot_rl.rollout.export_to_a33 import ..."
→ A3.3 exported, 300 frames
```

导出: `Saved/FutsalMOT_RL/exported_a33/rl_episode_random_0001_t1_Player_05_a33.json`

---

## 7. PPO 正确性验证

| 要求 | 状态 |
|------|------|
| GAE 中禁止 env.reset() | ✅ `compute_gae` 是纯函数，无 env 访问 |
| 每个 transition 保存 real next_obs | ✅ 在 reset 前计算 next_value |
| 分离 terminated/truncated | ✅ 分别存储，GAE 分别处理 |
| return 使用 un-normalized advantage | ✅ `returns = raw_advantages + values` |
| rollout 跨批次延续 | ✅ Trainer 维护 `_current_obs` |
| episode reward 独立累加器 | ✅ `current_episode_reward += reward` |
| 单样本标准差 unbiased=False | ✅ |
| 无双重 tanh | ✅ `MLPPolicy.forward()` 返回 raw mean |
| observation 使用实际 agent 状态 | ✅ `self_vel` 来自 `self.agent_vel` |
| observation 无 NaN | ✅ 30 env steps 验证通过 |

---

## 8. UE 部分 — 需人工执行的步骤

普通 Python 无法操作 Unreal Editor。按以下步骤在 UE 中验证：

### 步骤 1: 准备 RL A3.3 文件

文件已在: `Saved/FutsalMOT_RL/exported_a33/rl_episode_random_0001_t1_Player_05_a33.json`

### 步骤 2: 在 UE Python 控制台执行

打开 Unreal Editor，在 Python Console 运行：

```python
py "D:/projects/FustalMOT_UEDataset/Content/FutsalMOT/code/rl_07b_ue_render_rl.py"
```

该脚本内置了 RL A3.3 路径，不需要外部环境变量。

### 步骤 3: Movie Render Queue 渲染

- Output Directory: `Saved\FutsalMOT_RL\ue_closed_loop\images\rl_episode_random_0001_t1_p05`
- File Name Format: `{frame_number}`
- Image Format: PNG
- Resolution: 1920 × 1080

### 步骤 4: Windows 后处理

```cmd
cd /d D:\projects\FustalMOT_UEDataset\Content\FutsalMOT\code
python 03_check_labels.py --annotation "Saved\FutsalMOT\annotations\objects_bbox_2d_clean_rl_episode_random_0001_t1_p05.json" --step 5
```

### 步骤 5: 验证

```cmd
python rl_07_validate_ue_closed_loop.py --check --seq-id rl_episode_random_0001_t1_p05
```

---

## 9. 尚未验证的问题

- UE 渲染（需要 Unreal Editor，当前环境无 `unreal` 模块）
- UE bbox 标注导出
- layout_check（依赖 UE 渲染出的图像）
- 多模板（当前仅验证 template 1）
- 多 seed RL 训练
