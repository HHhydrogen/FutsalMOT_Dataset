# 动画架构审计报告

## 一、当前修改检查

### 实际修改的文件

| 文件 | 类型 | 作用 | 是否符合架构 |
|------|------|------|-------------|
| `docs/animation_architecture_check.md` | 文档 | 现有资产检查报告 | ✅ 只读 |
| `docs/animation_implementation_guide.md` | 文档 | UE 操作指南 | ✅ 只读 |
| `docs/mannequin_unarmed_animations.md` | 文档 | 动画资源统计 | ✅ 只读 |
| `ue/import_grf_episode.py` | Python | Sequence 生成 + 动画轨道 | ⚠️ 见下方 |
| `ue_import_config.json` | 配置 | Sequence 路径/命名/序列 | ✅ |
| `ue/animation_config.local.json` | 配置 | 动画资产路径 | ⚠️ 路径需修正 |

### 不正确架构：已创建

**0 个**不符架构的 UE 资产。

```
FutsalMOT/Animations/        ← 目录不存在
BP_FutsalPlayer              ← .uasset 不存在
ABP_Futsal_Player            ← .uasset 不存在
BS_Futsal_Locomotion         ← .uasset 不存在
```

所有 UE 资产操作仅存在于**文档中**，未实际创建。保留已有 GRF 控制链完好无损。

### 不正确架构：文档中存在但被删除/废弃

| 设计 | 状态 |
|------|------|
| 新建 BP_FutsalPlayer | ❌ 文档已写，实际未创建，现废弃 |
| 替换场景 Actor | ❌ 仅在文档中提到，未执行 |
| CharacterMovement 驱动 | ❌ 未使用 |
| Root Motion | ❌ 所有使用 In-Place 动画 |
| FootballMovesPack 路径 | ⚠️ animation_config.local.json 仍引用，需修正为 UE 自带 Mannequin 路径 |

---

## 二、正确架构（修正后）

```
GRF trajectory (JSONL)
        ↓
import_grf_episode.py
        ↓
Level Sequence + Transform Track  ← 唯一位置来源
        ↓
现有 Player Actor_XX (SetActorLocation)
        ↓
Mesh → Anim Class → ABP_GRF_Player
        ↓
BS_GRF_Locomotion (1D Blend Space, Speed 0-600)
        ↓
Idle → Walk → Run (In-Place)
```

### 关键约束

| 约束 | 措施 |
|------|------|
| 位置来源唯一 | Transform Track 保持 SetActorLocation，不启用 CharacterMovement |
| Root Motion 禁止 | In-Place 动画 |
| 不新建 Character BP | 只在现有 Actor 的 Mesh 上挂 ABP |
| 不新建场景 Actor | 保持现有 Player_L0~R4 |
| MOT 不受影响 | 位置、bbox、keypoints 输出不变 |

---

## 三、需要修正的内容

### 1. animation_config.local.json 路径修正

现有：`/Game/FootballMovesPack/...`
修正为：`/Game/Characters/Mannequins/Anims/Unarmed/...`

注意：Mannequins 路径下**没有**独立的 Walk 和 Run 动画序列，只有 `BS_Idle_Walk_Run` Blend Space（1D, Speed 驱动）。
因此 GRF import 脚本的 `_add_animation_tracks` Sequencer 动画轨道方式需要替换为：
→ 由场景中 Actor 的 AnimBP 实时驱动，而非写死在 Sequence 中。

### 2. import_grf_episode.py 动画策略修正

当前：在 Level Sequence 中写入 `MovieSceneSkeletalAnimationTrack`
问题：Sequencer 写入的动画段是静态的，无法根据运行时速度变化切换

修正：移除或废弃 `_add_animation_tracks` 函数中创建 Sequencer 动画轨道的逻辑
改用：Actor 自身的 AnimBP 实时读取 Velocity 驱动 Blend Space

### 3. 文档废弃

`docs/animation_implementation_guide.md` 中新建 BP_FutsalPlayer 的设计已废弃。
需要替换为：只在已有 Actor 上挂 AnimBP。

---

## 四、完整检查结果

| 检查项 | 结果 |
|--------|------|
| GRF trajectory 是否被修改 | ✅ 未修改 |
| episode 导入是否被修改 | ✅ 未修改 |
| Player ID 系统是否被修改 | ✅ 未修改 |
| bbox/keypoint 导出是否被修改 | ✅ 未修改 |
| Sequence 生成流程是否被修改 | ⚠️ 仅添加了可移除的动画轨道代码 |
| 是否有新 Character BP 创建 | ✅ 无 |
| 是否有新 Actor 替换场景 | ✅ 无 |
| 是否有 CharacterMovement 介入 | ✅ 无 |
| 是否有 Root Motion 引入 | ✅ 无 |

## 五、结论

当前架构 **未损坏**。动画相关修改局限于：

1. `import_grf_episode.py` 的 `_add_animation_tracks()` — 可安全移除或保留
2. `animation_config.local.json` — 路径需修正
3. 文档 — 需更新以匹配正确架构

下一版正确流程：

```
Actor Mesh (已有 Player_L0)
        ↓
Anim Class = ABP_GRF_Player（新建，只挂载不新建 Actor）
        ↓
Event Graph: Pawn Velocity → Speed
        ↓
BS_GRF_Locomotion: Speed 0~600 → Idle/Walk/Run
```
