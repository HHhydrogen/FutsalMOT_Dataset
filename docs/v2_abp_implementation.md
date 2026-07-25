# ABP_GRF_Player 实施步骤 v2

所有操作在 UE Editor 内完成。不创建新 Blueprint，不替换 Actor。

---

## Step 1: 创建 BS_GRF_Locomotion (Blend Space 1D)

1. UE Content Browser → `FutsalMOT/` → 右键 → `Animation` → `Blend Space`
2. 选择 Skeleton: `SK_Mannequin` (`/Game/Characters/Mannequins/Meshes/SK_Mannequin`)
3. 命名: `BS_GRF_Locomotion`
4. 保存路径: `/Game/FutsalMOT/Animations/`

### Axis 设置

| 参数 | 值 |
|------|-----|
| Axis Name | `Speed` |
| Axis Range | `0` ~ `600` |
| Axis Unit | `cm/s` |

### 放入动画

| Speed | 动画路径 |
|-------|---------|
| 0 | `/Game/Characters/Mannequins/Anims/Unarmed/MM_Idle` |
| 250 | `/Game/Characters/Mannequins/Anims/Unarmed/Walk/MF_Unarmed_Walk_Fwd` |
| 500 | `/Game/Characters/Mannequins/Anims/Unarmed/Jog/MF_Unarmed_Jog_Fwd` |

右键 Blend Space 图表上的坐标格 → `Add Animation Sample(s)` → 输入速度值 → 选择对应动画。

确认每个动画的 **Root Motion Lock** 为关闭状态。

---

## Step 2: 创建 ABP_GRF_Player (Animation Blueprint)

1. UE Content Browser → `FutsalMOT/Animations/` → 右键 → `Animation` → `Animation Blueprint`
2. Target Skeleton: `SK_Mannequin`
3. 命名: `ABP_GRF_Player`

### 变量

| 名称 | 类型 | 默认值 | 分类 |
|------|------|--------|------|
| `Speed` | float | 0.0 | Instance Editable |
| `Direction` | float | 0.0 | Instance Editable |

### Event Graph

```
Event Blueprint Update Animation
        ↓
Try Get Pawn Owner → Is Valid?
        ↓ Valid
Get Velocity
        ↓
VectorLength(Velocity) → → → Set Speed
        ↓
CalculateDirection(Velocity, Actor Rotation) → → → Set Direction
```

节点详细路径：
1. 右键 → `Event Blueprint Update Animation` → 添加
2. 从引脚拉出 → `Try Get Pawn Owner`
3. 从 Pawn Owner 拉出 → `Cast To BP_ThirdPersonCharacter` (或你实际使用的 Character BP)
4. As Cast → `Get Character Movement` → `Get Velocity`
5. 从 Velocity 拉出 → `V(ectorLength)` → 输出到 `Set Speed`
6. 从 Velocity + Actor Rotation 拉出 → `Calculate Direction` → 输出到 `Set Direction`

### Anim Graph

1. 打开 Anim Graph 选项卡
2. 删除默认的 `BlendSpacePlayer` + `Result` 之间的连接
3. 添加 `BlendSpacePlayer` 节点：
   - 搜索并添加
   - 双击进入详情
4. 设置 BlendSpacePlayer 属性：
   - Blend Space: `BS_GRF_Locomotion`
5. 将 `BlendSpacePlayer` 输出连接到 `Result` (Output Animation Pose)

### BlendSpacePlayer 输入

| 参数 | 来源 |
|------|------|
| Speed | 拖动 `Speed` 变量到图表 → `Get Speed` → 连接到 BlendSpacePlayer 的 `Speed` |

6. 点击 `Compile` → 保存

---

## Step 3: 挂载到已有 Actor

不要创建新 Actor。直接修改场景中已存在的 `Player_L0`：

1. 选中 `Player_L0`（世界大纲中）
2. Details 面板 → Mesh Component
3. 找到 `Anim Class` 属性
4. 改为 `ABP_GRF_Player`
5. 验证: 播放 Sequence 时 Player_L0 应显示走跑动画

对 `Player_L1` ~ `Player_R4` 各执行一次。

或者批量操作：在世界大纲中多选所有 `Player_XX` → 右键 → 批量设置 Anim Class。

---

## Step 4: 校准

运行 `episode_0001` 导入 → 在 Sequencer 播放：

| 现象 | 解决 |
|------|------|
| 不动（Idle） | ABP 中 Speed 变量一直为 0 → 检查 Event Graph 中 Velocity 读取 |
| 永远 Walk | Speed 阈值不对 → 调小 Speed 范围或检查 Velocity 单位 |
| 永远 Run | Speed 值太大 → Blend Space 的 Speed 网格点不匹配 |
| 不切换 Blend | 检查 BlendSpacePlayer 的 Speed 输入是否连到变量 |

ABP 调试方法：
1. 在 ABP 中打断点（Breakpoint）在 `Set Speed` 节点
2. 运行 Sequence
3. 观察每一帧的 Speed 值

---

## Step 5: 验证

| 检查 | 方法 |
|------|------|
| 位置不变 | 对比修改前和修改后的 bbox/keypoints 输出 |
| 动画播放 | Sequencer 中观察人物是否做走跑动作 |
| 速度对应 | 慢速走、快速跑 |
| 不漂移 | Actor 位置完全由 Transform Track 控制，动画 In-Place |

---

## 资产清单

| 资产 | 路径 | 类型 |
|------|------|------|
| `BS_GRF_Locomotion` | `/Game/FutsalMOT/Animations/BS_GRF_Locomotion` | Blend Space 1D |
| `ABP_GRF_Player` | `/Game/FutsalMOT/Animations/ABP_GRF_Player` | Animation Blueprint |

已有未改动的资产：

| 资产 | 路径 |
|------|------|
| SK_Mannequin | `/Game/Characters/Mannequins/Meshes/SK_Mannequin` |
| MM_Idle | `/Game/Characters/Mannequins/Anims/Unarmed/MM_Idle` |
| MF_Unarmed_Walk_Fwd | `/Game/Characters/Mannequins/Anims/Unarmed/Walk/MF_Unarmed_Walk_Fwd` |
| MF_Unarmed_Jog_Fwd | `/Game/Characters/Mannequins/Anims/Unarmed/Jog/MF_Unarmed_Jog_Fwd` |
| BP_ThirdPersonCharacter | `/Game/ThirdPerson/Blueprints/BP_ThirdPersonCharacter` |
