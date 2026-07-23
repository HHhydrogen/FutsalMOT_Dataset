# FutsalMOT-Dataset 重构报告

> 生成日期: 2026-07-24
> 提交: 58e400e

---

## 1. 修改前问题

### PPO 数学错误

| 问题 | 严重性 | 根因 |
|------|--------|------|
| GAE bootstrap 调用 `env.reset()` | P0 | `_compute_gae()` 用新 episode 初始状态价值代替末端状态价值 |
| Critic 使用归一化 advantage 做回归目标 | P0 | advantage 归一化在 GAE 内部完成，污染了 returns |
| Double tanh | P0 | `MLPPolicy.forward()` 返回 `tanh(net)`，PPO 又做第二次 `tanh` |
| Episode reward 跨 episode | P0 | 固定窗口 `rewards_list[-300]` 在非 300 步 episode 时混入上局奖励 |
| 缺少 terminated/truncated 分离 | P0 | 单一 `dones` 无法区分自然结束和截断 |

### 路径与配置问题

| 问题 | 严重性 | 根因 |
|------|--------|------|
| `set_project_root()` 在模块常量初始化后才执行 | P1 | 模块级路径常量依赖可变全局状态 |
| `pipeline_current.json` 在 git 中 | P1 | 含本机绝对路径和运行信息 |
| 路径推断只有 `parents[n]` 硬编码 | P1 | 无法在 CI 或临时目录中工作 |

### 工程化问题

| 问题 | 严重性 | 根因 |
|------|--------|------|
| 17 个 `rl_*.py` 散落根目录 | P2 | 没有统一 CLI |
| 无测试 | P2 | `tests/` 目录不存在 |
| `sys.path.insert()` 多处 | P2 | 项目未正确安装为 package |
| 宽泛 `except Exception` | P2 | 隐藏真实 bug |
| 无 CI | P3 | 无质量门禁 |

---

## 2. 修改后架构

### 路径系统

```
ProjectPaths (dataclass frozen)
├── repo_root      # pyproject.toml 所在目录
├── ue_project_root # *.uproject 所在目录
├── saved_rl_dir   # Saved/FutsalMOT_RL
├── models_dir     # Saved/FutsalMOT_RL/models
├── demos_dir      # Saved/FutsalMOT_RL/demos
├── configs_dir    # code/configs
├── runs_dir       # code/configs/runs
└── ...
```

解析优先级:
1. `--project-root` CLI 参数
2. `FUTSALMOT_PROJECT_ROOT` 环境变量
3. 从 cwd 向上搜索 `*.uproject`
4. 从包位置推断（最后手段）

### PPO 数学语义

```
terminated=True  → bootstrap_mask=0  (不 bootstrap)
truncated=True   → 使用 critic bootstrap
rollout 边界     → 使用 last_obs bootstrap
episode_ended    = terminated OR truncated (阻止 GAE 跨 episode 累积)
```

GAE 计算:

```python
delta = reward + gamma * next_value * bootstrap_mask - value
gae = delta + gamma * gae_lambda * continuation_mask * next_gae
```

Actor 分布:

```
raw_mean = actor(obs)        # 无界输出
std = exp(log_std)
u ~ N(raw_mean, std)         # 在无界空间采样
a = tanh(u)                  # 单次 tanh 压缩
log pi(a|s) = log N(u) - sum log(1 - a²)  # Jacobian 修正
```

### CLI

```
futsalmot-rl [--project-root PATH] <command> [args]

Commands:
  demos export      导出示范数据
  demos check       校验示范数据
  train bc          行为克隆训练 (legacy)
  train ppo         PPO 训练 (legacy)
  evaluate bc       评估 BC 策略
  evaluate rl       评估 RL 策略
  evaluate sanitize-env  环境完整性检查
  benchmark build   构建基准表格 (stub)
  export a33        导出 A3.3 (stub)
  ue verify         UE 验证 (stub)
```

---

## 3. 修改文件列表

### 新增

| 文件 | 说明 |
|------|------|
| `futsalmot_rl/core/paths.py` | `ProjectPaths` 不可变路径对象 |
| `futsalmot_rl/core/exceptions.py` | 异常层次结构 |
| `futsalmot_rl/commands/__init__.py` | CLI 命令包 |
| `futsalmot_rl/commands/demos.py` | `demos` 命令实现 |
| `futsalmot_rl/commands/evaluate.py` | `evaluate` 命令实现 |
| `futsalmot_rl/commands/train.py` | `train` 命令 stub |
| `tests/conftest.py` | 共享 fixture（tmp_project, mini_a33） |
| `tests/unit/test_paths.py` | 路径解析测试（15） |
| `tests/unit/test_actor_distribution.py` | Actor 分布测试（7） |
| `tests/unit/test_observation.py` | 观测契约测试（13） |
| `tests/unit/test_gae.py` | GAE 数值测试（8） |
| `.github/workflows/ci.yml` | GitHub Actions CI |
| `configs/pipeline_current.example.json` | pipeline_current 模板 |

### 修改

| 文件 | 说明 |
|------|------|
| `futsalmot_rl/training/train_ppo.py` | 重写 collect_rollout/GAE/terminated-truncated |
| `futsalmot_rl/models/mlp_policy.py` | 修复 double tanh, raw mean |
| `futsalmot_rl/envs/defender_follow_env.py` | 修复 observation 使用实际速度/yaw |
| `futsalmot_rl/cli.py` | 重写，使用 ProjectPaths + commands 包 |
| `futsalmot_rl/core/rl_paths.py` | 薄封装层，转调 ProjectPaths |
| `futsalmot/core/paths.py` | 统一路径解析逻辑 |
| `.gitignore` | 全面更新 |
| `pyproject.toml` | hatch build config, wheel 包声明 |

### 移动

| 文件 | 原位置 | 新位置 |
|------|--------|--------|
| 17 个 `rl_*.py` | `code/` | `tools/legacy/` |

### 删除

| 文件 | 原因 |
|------|------|
| `configs/pipeline_current.json` | 含本机路径，改为 `.example` 模板 |

---

## 4. 测试结果

```
$ pytest tests/ -v --tb=short
======================= 46 passed in 2.16s ========================

tests/unit/test_paths.py ............... 15 passed
tests/unit/test_actor_distribution.py ..  7 passed
tests/unit/test_observation.py ......... 13 passed
tests/unit/test_gae.py .................  8 passed

$ python -m compileall futsalmot futsalmot_rl tests -q
(no output = all OK)

$ python -m build
Successfully built futsalmot-0.2.0.tar.gz and futsalmot-0.2.0-py3-none-any.whl

$ pip install dist/futsalmot-0.2.0-py3-none-any.whl
$ python -c "import futsalmot_rl; print('OK')"
OK
```

---

## 5. 尚未验证的 UE 部分

以下部分需要 Unreal Editor 才能验证，已在 CI 中通过 `@pytest.mark.ue` 标记跳过：

- UE preflight 检查
- UE build sequences
- Movie Render Queue 渲染
- bbox 标注导出
- layout_check

---

## 6. 行为兼容性变化

| 变更 | 影响 |
|------|------|
| `rl_*.py` 移至 `tools/legacy/` | 直接调用 `python rl_*.py` 不再工作，需使用 `python tools/legacy/rl_*.py` 或 CLI |
| CLI 改为 `futsalmot-rl <command>` | 新入口，旧用法仍然可通过 `python -m futsalmot_rl.cli` 使用 |
| `MLPPolicy.forward()` 不再包含 `tanh` | 旧 checkpoint 的 actor 权重可以直接迁移（`load_state_dict` 按名称匹配），但 forward 输出范围变了 |
| 路径解析 | 不再硬编码 `parents[n]`，支持 `.uproject` 搜索和环境变量 |

---

## 7. 下一阶段建议

1. **FA-2 Goal-side Defense 训练** — reward v3 配置和指标已定义在 `futsalmot_rl/academy/fa2_goal_side_defense.py`
2. **完整命令迁移** — `train/benchmark/export/ue` 命令从 stub 升级为完整实现
3. **可复现性** — episode 输出拆为 `episode.json`（确定性数据）+ `run_manifest.json`（运行信息）
4. **多智能体** — FA-4 Two Defenders 和后续 FA-5~8
5. **mypy 全量通过** — 当前仅配置了 mypy，未实际运行
