# LOCAL_CODE_REVIEW_REPORT

> 生成日期: 2026-07-24
> 当前 commit: d463c97

---

## 1. 环境

| 项目 | 值 |
|------|-----|
| Python | C:\Python314\python.exe — 3.14.4 |
| 安装方式 | `pip install -e .` |
| UE 工程 | D:\projects\FustalMOT_UEDataset |
| .uproject | FustalMOT_UEDataset.uproject |
| 仓库位置 | D:\projects\FustalMOT_UEDataset\Content\FutsalMOT\code |

## 2. GitHub Actions

- ✅ `./github/` 已删除

## 3. 路径常量

- `train_ppo.py`: `train()` 方法签名移除 `MODELS_DIR`/`TRAIN_LOGS_DIR` 默认值，改为 `log_dir`/`model_dir` 必选参数
- `train_bc.py`、`ablation_runner.py` 等调用方已传入对应参数

## 4. 模型 Device 处理

`MLPActorCritic` 和 `MLPPolicy`：
- `get_action()`: 使用 `_get_device(self)` 获取模型设备，`torch.as_tensor(obs, device=device)` 转换输入
- `get_value()`: 同上
- 返回给环境的 action 转回 CPU NumPy
- 新增 `test_model_device.py` (6 cases)

## 5. Shared Backbone

- 新增 `_raw_mean_and_value()` 统一方法
- `forward()`、`get_action_and_value()`、`get_value()`、`get_action()` 全部通过该内部方法
- `shared_backbone=True` 时不再直接访问 `self.actor` / `self.critic`
- 新增 `test_shared_backbone.py` (8 cases)

## 6. Yaw 持久化

- 新增 `_update_yaw(vx, vy)` 方法，使用实际 agent velocity
- 低于 `_yaw_speed_threshold` (5cm/s) 时保持上次朝向
- `reset()` 时初始化 `_last_yaw = 0.0`
- 不再读取规则 ghost 轨迹作为 agent yaw
- 新增 `test_yaw_persistence.py` (6 cases)

## 7. Frame / Transition 语义

- terminal observation 中 `steps_left == 0`
- `steps_left` 随帧推进单调递减
- 新增 `test_episode_boundary.py` (6 cases)

## 8. PPO Rollout / GAE

- `compute_gae()` 为纯函数，无 env 访问
- `collect_rollout()` 返回类型已修正为 `tuple[dict[str, Tensor], list[float]]`
- `bootstrap_mask = 1.0 - terminated` (terminated 不 bootstrap)
- `continuation_mask = 1.0 - (terminated|truncated)` (停止 GAE 递归)
- 每个 transition 保存真实 `next_value`（在 reset 之前计算）

## 9. 最后一轮采样

```python
remaining = total_timesteps - global_step
steps_to_collect = min(n_steps, remaining)
```

`global_step` 不超过 `total_timesteps`。

## 10. Best Model 判断

- 无完成 episode 时 `mean_episode_reward = None`
- 不更新 best model
- partial episode reward 保留到下一 rollout

## 11. CLI / 入口

- 只保留 `futsalmot-rl demos` 和 `futsalmot-rl evaluate`
- 训练/导出使用 `scripts/train_bc.py`、`scripts/train_ppo.py` 等
- 所有 `stub` 命令已删除

## 12. 测试

```
======================== 72 passed in 2.41s ==========================

tests/unit/test_paths.py ............... 15
tests/unit/test_gae.py .................  8
tests/unit/test_actor_distribution.py ..  7
tests/unit/test_observation.py ......... 13
tests/unit/test_shared_backbone.py .....  8  (new)
tests/unit/test_model_device.py ........  6  (new)
tests/unit/test_yaw_persistence.py .....  6  (new)
tests/unit/test_episode_boundary.py ....  6  (new)
```

## 13. Smoke 结果

| 步骤 | 结果 |
|------|------|
| BC train (2 epochs) | loss 0.251 → 0.126 |
| PPO train (2048 steps) | reward -2504, done |
| A3.3 export | 300 frames, OK |

## 14. 未验证的 UE 部分

UE 渲染需要 Unreal Editor，按 `UE_LOCAL_STEPS.md` 操作：

```python
py "D:/projects/FustalMOT_UEDataset/Content/FutsalMOT/code/tools/legacy/rl_07b_ue_render_rl.py"
```
