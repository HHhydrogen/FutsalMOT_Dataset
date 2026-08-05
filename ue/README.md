# UE 侧脚本

## 文件清单

| 文件 | 说明 |
|------|------|
| `run_task.py` | **统一 UE 入口（推荐）** — 读取 resolved task，调用既有导入/渲染逻辑 |
| `import_grf_episode.py` | 主脚本 — 建 Sequence / 导标注 / MRQ 渲染（`--config` 为 legacy 模式） |
| `actor_mapping.example.json` | 实体 ID (L0~L4, R0~R4, BALL) → UE Actor 标签 映射 |
| `render_episode.py` | MRQ 异步渲染 RGB + Instance-ID Mask（含 CV GT preset 应用） |
| `render_preset.py` | CV GT deterministic render preset（纯配置，pytest 可测） |
| `recover_render.py` | 从已有 `render/` 恢复 `img1/`（`--resolved-task`） |

## 要求

- Unreal Engine 5.x 项目，关卡中放置 11 个 Actor（10 名球员 + 1 个足球）
- **无需** gfootball、.venv、GRF_MARL — 纯 UE Python + stdlib
- Actor 标签与 ue profile 的 `actor_mapping` 指向的 JSON 一致

## 推荐用法（resolved task）

1. 在 P1 生成 resolved task（机器路径由 `.futsalmot.local.json` / 环境变量提供）：

```powershell
uv run grf-ue task resolve configs/tasks/soak_300frames_4cam.example.json
```

2. 获取 UE 命令并复制到 **Unreal Editor Python Console**：

```powershell
uv run grf-ue task ue-command configs/tasks/soak_300frames_4cam.example.json
```

输出形如：

```python
py "D:/.../code/ue/run_task.py" --resolved-task "D:/.../.futsalmot/runtime/soak_300frames_4cam/resolved-task.json"
```

`ue/run_task.py` 与 P1 读取**同一个** resolved task（schema `futsalmot_resolved_task` v1），
不再隐式读取根目录 `ue_import_config.json`；episode / mapping / dataset 路径由 resolver 填充。

## Legacy 用法（已弃用）

`import_grf_episode.py --config <legacy-config>` 仍可用但打印 deprecation warning。
删除计划见 `docs/migration/TASK_CONFIG_MIGRATION.md`。

## 脚本模式

| 模式 | 说明 |
|------|------|
| `preview` | 直接在关卡中设置 Actor 变换（不创建资产） |
| `sequence` | 创建/覆盖 Level Sequence 资产 |
| `both`（默认） | 先 preview 后 sequence |
| `annotations` | 只导几何标注（fallback bbox） |
| `full` | 建 Sequence + 导标注 + MRQ 渲染（RGB + Object ID EXR） |

## 球滚动旋转

在 ue profile（`configs/ue/*.json`）的 `ball_rolling` 段配置：
- `radius_m` — 球半径（默认 0.11）
- `roll_sign` — 滚反了设为 `-1.0`
- `enabled: false` 可完全禁用

## 注意事项

- 球员 Z = 90cm（角色中枢在地面以上；地面 Z=0）
- 球 Z = GRF 数据 × 100 + 2cm 偏移
- Yaw 通过位置增量计算，低速保持先前朝向
- 需要人工在 UE 视口中验证效果
