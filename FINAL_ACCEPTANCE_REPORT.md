FINAL_ACCEPTANCE_REPORT.md

## 1. 环境

| 项目 | 值 |
|---|---|
| commit | 9589eb9 (+ PPO checkpoint model_type fix) |
| Python | C:\Python314\python.exe — 3.14.4 |
| Source episode | configs/runs/production_run/episode_random_0001_t1_a33.json |

## 2. 各步骤结果

| 步骤 | 结果 |
|------|------|
| Demo (12 demos) | PASS |
| BC train (2 epochs) | PASS — loss 0.167 → 0.052 |
| BC load + evaluate (2 eps) | PASS |
| PPO train (1024 steps) | PASS — reward -1355 |
| PPO load + evaluate (2 eps) | PASS — type=MLPActorCritic, action finite |
| A3.3 export (300 frames) | PASS — 478 KB, schema OK |
| Formal models unchanged | PASS (0 models before, 0 after) |

## 3. 修复的问题

1. **`policy_io.py`**: `model_type` 存在 `save_dict` 顶层而非 `architecture` 内，导致含 architecture 字段的 PPO checkpoint 被错误加载为 MLPPolicy。改为优先读取 `save_dict["model_type"]`。

## 4. 未通过的步骤

无。

## 5. UE 中需要人工执行的唯一下一步

```python
py "D:/projects/FustalMOT_UEDataset/Content/FutsalMOT/code/tools/legacy/rl_07b_ue_render_rl.py"
```
