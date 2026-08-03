# UE Import Script — `import_grf_episode.py`

## 文件清单

| 文件 | 说明 |
|------|------|
| `import_grf_episode.py` | **主脚本** — 在 Unreal Editor Python Console 中执行 |
| `actor_mapping.example.json` | 实体 ID (L0~L4, R0~R4, BALL) → UE Actor 标签 映射 |
| `render_episode.py` | MRQ 异步渲染 RGB + Instance-ID Mask（含 CV GT preset 应用） |
| `render_preset.py` | CV GT deterministic render preset（纯配置，pytest 可测） |

## 要求

- Unreal Engine 5.x 项目，关卡中放置了 11 个 Actor（10 名球员 + 1 个足球）
- **无需** gfootball、.venv、GRF_MARL — 纯 UE Python + stdlib
- Actor 标签与 mapping 文件一致（默认 `Player_L0`~`Player_L4`, `Player_R0`~`Player_R4`, `Ball_01`）

## 使用方式

### 方式一：一键导入（推荐）

确保 `ue_import_config.json`（位于 `code/` 根目录）已配置好，然后在 UE Python Console 中：

```python
py "D:/projects/FustalMOT_UEDataset/Content/FutsalMOT/code/ue/import_grf_episode.py"
```

脚本自动从 `ue_import_config.json` 加载所有参数。

### 方式二：命令行参数覆盖

```python
# 覆盖 episode 目录和 mapping
py "D:/.../import_grf_episode.py" --episode "D:/.../outputs/episode_0001" --mapping "D:/.../actor_mapping.example.json" --replace-existing

# 仅预览模式（直接在关卡中设置 Actor 变换，不创建 Sequence 资产）
py "D:/.../import_grf_episode.py" --mode preview
```

## 脚本模式

| 模式 | 说明 |
|------|------|
| `preview` | 直接在关卡中设置 Actor 变换（不创建资产） |
| `sequence` | 创建/覆盖 Level Sequence 资产 |
| `both`（默认） | 先 preview 后 sequence |

## 球滚动旋转

脚本自动为足球添加滚动旋转，基于位移量计算。在 `ue_import_config.json` 的 `ball_rolling` 段中配置：
- `radius_m` — 球半径（默认 0.11）
- `roll_sign` — 滚反了设为 `-1.0`
- `enabled: false` 可完全禁用

## 注意事项

- 球员 Z = 90cm（角色中枢在地面以上；地面 Z=0）
- 球 Z = GRF 数据 × 100 + 2cm 偏移
- Yaw 通过位置增量计算，低速保持先前朝向
- 需要人工在 UE 视口中验证效果
