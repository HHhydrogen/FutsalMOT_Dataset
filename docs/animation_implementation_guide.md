# 动画系统实施指南

所有 UE 资产创建操作必须在 **Unreal Editor** 内完成。本文档为 UE Editor 内操作的完整步骤。

---

## 架构总览

```
GRF trajectory (JSONL)
        ↓
Level Sequence + Transform Track (import_grf_episode.py)
        ↓
BP_FutsalPlayer (Character) ← 位置由 Sequencer 驱动
        ↓
Event Tick → 计算 Velocity → 传递给 AnimBP
        ↓
ABP_Futsal_Player (Animation Blueprint)
        ↓
BS_Futsal_Locomotion (Blend Space 1D 或 2D)
        ↓
Walk / Jog / Idle In-Place Animation
```

---

## Step 1: 创建 BS_Futsal_Locomotion (Blend Space)

**路径**: `/Game/FutsalMOT/Animations/BS_Futsal_Locomotion`

### 方案 A — Blend Space 1D (推荐快速上手)

1. UE Content Browser → `FutsalMOT/` → 右键 → `Animation` → `Blend Space`
2. 选择 Skeleton: `SK_Mannequin`
3. 命名: `BS_Futsal_Locomotion`
4. Axis Settings:
   - Horizontal Axis: `Speed`, 范围 `0 ~ 600`
5. 放入动画 (右键插入点 → 选择动画):

```
Speed 0:    MM_Idle
Speed 150:  MF_Unarmed_Walk_Fwd
Speed 350:  MF_Unarmed_Jog_Fwd
Speed 600:  MF_Unarmed_Jog_Fwd (拉长采样速度)
```

### 方案 B — Blend Space 2D (完整方向支持)

1. 创建: 同上但选 `Blend Space` (不是 1D)
2. Axis:
   - Horizontal: `Direction`, `-180 ~ 180`
   - Vertical: `Speed`, `0 ~ 600`
3. 网格填入动画:

| Sp\Dir | -180(退) | -90(左) | 0(前) | 90(右) | 180(退) |
|--------|----------|---------|-------|--------|---------|
| 0 | MM_Idle | MM_Idle | MM_Idle | MM_Idle | MM_Idle |
| 150 | Walk_Bwd | Walk_Left | Walk_Fwd | Walk_Right | Walk_Bwd |
| 350 | Jog_Bwd | Jog_Left | Jog_Fwd | Jog_Right | Jog_Bwd |

---

## Step 2: 创建 ABP_Futsal_Player (Animation Blueprint)

**路径**: `/Game/FutsalMOT/Animations/ABP_Futsal_Player`

1. UE Content Browser → `FutsalMOT/Animations/` → 右键 → `Animation` → `Animation Blueprint`
2. Target Skeleton: `SK_Mannequin`
3. 命名: `ABP_Futsal_Player`

### 变量

打开 ABP → My Blueprint 面板 → Variables → `+`:

| 变量名 | 类型 | 默认值 | 分类 | 描述 |
|--------|------|--------|------|------|
| Speed | float | 0.0 | Instance Editable | 当前移动速度 (转换后) |
| Direction | float | 0.0 | Instance Editable | 移动方向 -180~180 |
| AnimationSpeedScale | float | 0.015 | Instance Editable | 速度转换系数 |

### Event Graph

1. 打开 Event Graph
2. 右键空白处 → 输入 `Event Blueprint Update Animation` → 点生成
3. 连线:

```
Event Blueprint Update Animation
        ↓
Try Get Pawn Owner → Is Valid?
        ↓ (Valid)
Get Movement Component → 获取 Velocity
        ↓
VectorLength(Velocity) → * AnimationSpeedScale → Set Speed
        ↓
CalculateDirection(Velocity, ActorRotation) → Set Direction
```

具体节点:

| 节点 | 位置 | 连接 |
|------|------|------|
| Event Blueprint Update Animation | 自动生成 | → |
| Try Get Pawn Owner | 搜索 | 输出到 IsValid branch |
| Get Movement Component | 从 Pawn Owner 引脚拉出搜索 | → |
| Get Velocity | 从 Movement Comp 拉出 | → |
| VectorLength | 搜索 | 输入 Velocity → 输出到 * |
| * (Multiply) | 搜索 | 输入 = VectorLength 结果, AnimationSpeedScale |
| Set Speed | 从变量 Speed 引脚拖出 | → |
| CalculateDirection | 搜索 | Velocity, Actor Rotation → 输出 |
| Set Direction | 从变量 Direction 引脚拖出 | → |

### Anim Graph

1. 打开 Anim Graph 选项卡
2. 删除默认的 `BlendSpacePlayer` 和 `OutputPose`
3. 添加:

```
Result (Anim Pose)
    ↓ 连到
BlendSpacePlayer (BS_Futsal_Locomotion)
    ↓ 连到
Output Animation Pose
```

4. 选择 BlendSpacePlayer → Details 面板:
   - Blend Space: `BS_Futsal_Locomotion`
   - Speed: 变量 `Speed`
   - Direction (如用 2D Blend Space): 变量 `Direction`

5. 编译 → 保存

---

## Step 3: 创建 BP_FutsalPlayer (Character Blueprint)

**路径**: `/Game/FutsalMOT/Blueprints/BP_FutsalPlayer`

1. UE Content Browser → `FutsalMOT/Blueprints/` → 右键 → `Blueprint Class` → 选 `Character`
2. 命名: `BP_FutsalPlayer`

### Mesh 设置

1. 打开 BP → Mesh 组件 (在 Components 面板中)
2. Details:
   - Skeletal Mesh: `SK_Mannequin`
   - Anim Class: `ABP_Futsal_Player`

### Event Graph (Velocity 计算)

1. Event Graph 中添加变量:
   - `PreviousLocation` (Vector, 公开, 默认 (0,0,0))
   - `DeltaSecondsClamp` (float, 默认 0.001)

2. Event Tick 连线:

```
Event Tick → Delta Seconds
        ↓
Get Actor Location
        ↓
Branch → Is Valid? 判断 PreviousLocation ≠ (0,0,0)
        ↓ Valid
(CurrentLocation - PreviousLocation) / Max(Delta Seconds, 0.001)
        ↓ 结果 = Velocity
VSize(Velocity) * 0.015 → Set 变量 VelocityScaled (float, 临时局部)
        ↓
CalculateDirection(Velocity, Actor Rotation) → Set 变量 DirectionScaled (float, 临时局部)
        ↓
Get Anim Instance → Cast To ABP_Futsal_Player
        ↓
Set Speed = VelocityScaled
Set Direction = DirectionScaled
```

4. 结尾:

```
Set PreviousLocation = CurrentLocation
```

---

## Step 4: 替换场景 Actor

在世界大纲 (World Outliner) 中:

1. 选中 `Player_L0`
2. 右键 → `Replace Actor with` → `BP_FutsalPlayer`
3. 确认保留 Transform（位置保持一致）
4. 重复 L1~L4, R0~R4

或删除全部，拖入 10 个新 `BP_FutsalPlayer`，分别命名。

---

## Step 5: 校准 AnimationSpeedScale

运行 `import_grf_episode.py` 生成 Sequence → 打开 Sequencer 播放:

| 现象 | 解决 |
|------|------|
| 球员动画比实际慢 | 增大 `AnimationSpeedScale` (0.015→0.02) |
| 球员动画比实际快 | 减小 `AnimationSpeedScale` (0.015→0.01) |
| 静止时仍有动画 | BP Tick 中检查 PreviousLocation 初始化 |
| 方向相反 | 检查 CalculateDirection 的 Velocity/ActorRotation 参数 |

在 `ABP_Futsal_Player` 变量的 `AnimationSpeedScale` Default Value 直接改，或者 BP Tick 里改乘数。

---

## Step 6: 验证

| 检查项 | 方法 |
|--------|------|
| 球员不滑行 | 播放 Sequence → 观察脚步是否踩地 |
| 速度对应 | 慢速 Idle→Walk, 高速 Walk→Jog |
| 方向对应 | 横移/后退播放对应动画 |
| MOT 不变 | 运行 YOLO bbox 验证脚本 |
| Keypoints 不变 | 运行 keypoint 验证脚本 |

---

## 关联文件

| 文件 | 作用 |
|------|------|
| `ue/animation_config.local.json` | GRF import 动画配置 |
| `ue/import_grf_episode.py` | GRF → UE Level Sequence |
| `BP_FutsalPlayer` | 足球 Character Blueprint (新建) |
| `ABP_Futsal_Player` | 足球 Animation Blueprint (新建) |
| `BS_Futsal_Locomotion` | 走跑 Blend Space (新建) |

---

## 下一步可扩展

| 功能 | 所需 |
|------|------|
| 射门动画 | 添加 Kick 动画 → SyncGroup 打断 locomotion |
| 传球动画 | Pass 动画 + Event 触发 |
| 守门员专用 | 独立 Blend Space + ABP |
| 动作标注 | AnimBP 输出 state→JSON |
