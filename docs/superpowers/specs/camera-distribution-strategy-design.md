# P2-5 Camera Distribution Strategy Design

状态：设计阶段完成，本文档不包含实现。

## 1. 当前 Camera 系统分析

当前相机配置沿用既有任务和运行时链路：

```text
Task Spec
  -> simulation.camera.profiles
  -> ResolvedTask
  -> camera_distribution.py
  -> run_task.py
  -> UE Camera / Sequence / MRQ
```

当前已经完成的能力：

- P2-1：C1-C5 Static Surveillance Anchor Camera。
- P2-2：resolution、focal length、sensor/filmback 和 FOV 参数模型。
- P2-3：Coverage Distribution Model。
- P2-4：P01 Partial Static Camera 的 placement、resolver、UE runtime 和最小 MRQ 闭环。

当前 Camera 类别如下：

| 类别 | ID 示例 | 当前状态 | 当前职责 |
| --- | --- | --- | --- |
| Static Surveillance Anchor | `C1`-`C5` | 已完成 | 固定全场基线、多相机稳定观察 |
| Partial Static Surveillance | `P01` | 已完成首个原型 | 左半场局部观察、增加 MOT 难度 |
| Dynamic Broadcast | `B01` 等 | 未来方向 | 暂不进入当前主分布 |

Camera Distribution 的问题是决定一个 episode 需要哪些 Camera。它不负责生成相机位置、不负责创建 UE Actor、不负责 Sequence 或 MRQ。

## 2. Camera Category 定义

### 2.1 Static Surveillance Anchor

Static Surveillance Anchor 是当前数据集的核心类别。

其特征：

- 使用固定 canonical ID：`C1`、`C2`、`C3`、`C4`、`C5`。
- 固定存在于 anchor baseline 中。
- 当前语义为 `full_field`。
- 位置、旋转、height、lens 和 resolution 由已有 profile 明确描述。
- 适合跨 episode 比较和稳定 benchmark。

### 2.2 Partial Static Surveillance

Partial Static Surveillance 是 Static Surveillance 的受控扩展，而不是独立的 Camera 系统。

其特征：

- 使用 `P` 前缀，例如 `P01`。
- 作为 C1-C5 之外的可选附加 Camera。
- 通过现有 Coverage Model 描述目标区域。
- 通过已有 placement resolver 产生实际 position、rotation、height 和 lens 参数。
- 生成结果必须进入 ResolvedTask 后才能进入 UE runtime。
- 当前已实现的唯一实例是：
  ```text
  P01 -> partial_field -> left_half
  ```

### 2.3 Dynamic Broadcast

Dynamic Broadcast 是未来增强方向，不属于当前主分布模型。

未来可能使用 `B` 前缀，例如 `B01`，但本阶段不定义其数量、比例、行为参数或生成流程。其后续实现必须继续复用 Task Spec、ResolvedTask 和现有 UE runtime 边界，不能创建独立的 Broadcast Camera Manager 或动态相机服务。

## 3. Static Surveillance 优先级说明

Static Surveillance 的优先级高于 Dynamic Broadcast，原因是当前数据集主要服务于多目标追踪和球员统计任务：

- 稳定视角便于建立连续轨迹。
- 固定坐标系便于跨 episode 比较。
- 长时间连续观察有利于计算轨迹完整性。
- 固定相机之间的空间关系更容易复现和审计。
- 生成结果更容易与相机标定、Mask 和 MOT annotation 对齐。

Dynamic Broadcast 的运动、变焦和跟随行为会带来额外变量。它适合未来研究视角变化、短时遮挡和动态重识别，但不应在当前阶段稀释 Static Surveillance 的数据集基线。

## 4. C1-C5 Anchor Policy

C1-C5 始终组成固定的 canonical anchor baseline：

```text
C1 C2 C3 C4 C5
```

Policy：

- 每个使用 canonical camera system 的 episode 都必须保留 C1-C5。
- 不随机删除 C1-C5。
- 不使用 Partial Camera 替换 C1-C5。
- 不因为 Partial Camera 数量变化而改变 C1-C5 的语义。
- C1-C5 保持 `static_surveillance + full_field`。
- C1-C5 的 UE mapping、Sequence、Camera Cut 和既有参数保持独立稳定。

这样可以提供一个跨 episode 一致的 benchmark：同一组 anchor 能够用于比较不同轨迹、seed、lens、resolution 或 Partial 扩展带来的影响。

Anchor baseline 的默认 Camera Count 为：

```text
5
```

## 5. Partial Camera 扩展策略

Partial Camera 只能作为附加视角加入：

```text
Anchor baseline + selected Partial Cameras
```

当前可用：

```text
P01 -> partial_field -> left_half
```

未来扩展可以按照相同规则继续定义：

```text
P02 -> partial_field -> right_half
P03 -> partial_field -> goal_area_left/right
```

这些只是未来 ID 和语义示例，本阶段不实现 P02/P03。

### 5.1 Partial Camera 是否可选

Partial Camera 是可选的。建议支持以下 episode profile：

```text
anchor_only
anchor_plus_one_partial
anchor_plus_two_partial
```

当前实现阶段至少支持：

```text
anchor_only
anchor_plus_one_partial(P01)
```

Partial Camera 不应默认强制加入所有旧任务，因为旧任务需要保持 C1-C5 baseline 行为和可比性。

### 5.2 Partial Camera 最大数量

建议当前策略将 Partial Camera 数量限制为 `0..2`：

- `0`：只运行 C1-C5 baseline。
- `1`：加入一个 Partial Camera，例如 P01。
- `2`：加入两个受控 Partial Camera，未来可用于左右区域互补。

`2` 是当前建议的软上限，不是新的调度系统或数据库约束。它的作用是限制早期实验的渲染成本、profile 复杂度和跨相机分析规模。未来若有明确实验需要，可以在同一 profile contract 中调整，而不是新增 Camera Manager。

### 5.3 Partial Camera 采样范围

Partial Camera 的选择应来自有限、显式的 profile 集合，而不是任意 ID 或复杂概率系统：

- 当前允许 `P01`。
- 未实现的 P02/P03 不应被配置或隐式生成。
- 每个 Partial Camera 必须有合法 Coverage intent。
- 每个 Partial Camera 必须经过 placement resolver 生成完整 profile。
- 每个 Partial Camera 必须在进入 ResolvedTask 前完成确定性验证。

本阶段不定义精确生产比例，也不定义连续概率分布。任务级的离散 profile 比隐藏的 episode 内随机采样更容易复现、审计和控制成本。

### 5.4 Partial Camera 与 Anchor 的关系

Partial Camera 与 Anchor 是互补关系：

- Anchor 提供全场稳定观察。
- Partial Camera 提供局部观察和更高的轨迹中断概率。
- 两者可以共同生成同一 episode 的多视角数据。
- Partial Camera 的存在不改变 Anchor 的 ID、参数或职责。
- Partial Camera 不应被视为 Anchor 的别名。

## 6. Camera Count Strategy

推荐采用固定 Anchor 加显式 Partial 扩展：

```text
episode camera count = 5 + partial_camera_count
partial_camera_count ∈ {0, 1, 2}
```

推荐的三档任务级策略：

| Episode Profile | Cameras | 目的 | 成本 |
| --- | --- | --- | --- |
| `anchor_only` | C1-C5 | 稳定 benchmark、完整基础视角 | 最低基线成本 |
| `anchor_plus_one_partial` | C1-C5 + P01 | 验证局部覆盖与轨迹中断 | 中等成本 |
| `anchor_plus_two_partial` | C1-C5 + P01 + future P02 | 左右局部互补实验 | 较高成本 |

当前不实现 `anchor_plus_two_partial` 的具体 P02，只保留策略边界。

### 6.1 选择固定数量而非类别概率

当前推荐通过显式 Camera profile 集合或离散 episode profile 选择 Camera，而不是在运行时使用复杂概率系统，原因包括：

- 每个 episode 的 Camera 列表清晰可读。
- ResolvedTask 可以完整记录最终选择。
- 任务重复执行时 Camera 集合不变。
- 渲染帧数、磁盘成本和 MRQ 工作量可预估。
- 更容易比较 `anchor_only` 与 Partial augmentation 的 MOT 差异。

未来如果需要统计分布，可以在任务生成层批量创建不同的离散 episode profiles；不需要引入动态任务调度服务。

### 6.2 数据规模与渲染成本

每增加一个 Camera，通常会增加：

- RGB 输出。
- 可能的 Object-ID/Mask 输出。
- Camera annotation。
- MOT/YOLO 后处理工作量。
- 磁盘空间和 MRQ 时间。

固定 C1-C5 能提供稳定的最低成本基线。Partial 数量上限 `2` 能在保留空间互补价值的同时，避免早期实验无限扩张 Camera 数量。

## 7. Camera Identity Strategy

继续使用同一个 Camera ID namespace，不创建第二套身份系统。

ID 前缀语义：

| 前缀 | 类别 | 当前状态 |
| --- | --- | --- |
| `C` | Canonical Static Anchor | `C1`-`C5` 已完成 |
| `P` | Partial Static Extension | `P01` 已完成，其他待后续 |
| `B` | Dynamic Broadcast | 未来方向 |

Identity 规则：

- Camera ID 属于模型和 ResolvedTask 层。
- UE Actor 名称不是 Camera ID。
- Camera ID 在一个 ResolvedTask 中必须唯一。
- 未知前缀或未批准的 ID 必须拒绝。
- P01 等 Partial ID 不能进入 C1-C5 anchor 列表。
- B 系列在未实现前不能被当前主分布接受。

示例：

```text
C1-C5 -> fixed canonical anchors
P01   -> implemented partial static camera
B01   -> future dynamic broadcast camera
```

## 8. Future Dynamic Broadcast Boundary

Dynamic Broadcast 当前只作为未来扩展，不放入当前主分布比例，也不影响 Static Surveillance 默认策略。

未来 B 系列如果实现，应满足：

- 使用同一个 Task Spec -> ResolvedTask 入口。
- 使用同一个 Camera ID namespace。
- 由 profile 明确描述动态行为和参数约束。
- 使用已有 Sequence/MRQ 机制表达动态状态。
- 由确定性 seed 产生可重放结果。
- 不创建独立 Camera Manager、Broadcast Manager 或 runtime service。

未来 Broadcast 实现需要单独定义：

- 动态 transform 轨迹。
- 跟随目标和速度限制。
- pan/tilt/zoom 行为。
- 动态视野与 Coverage 的关系。
- 帧级状态验证。

这些问题不属于当前 P2-5 Camera Distribution Strategy 的实现范围。

## 9. 与现有 Pipeline 的关系

Camera Distribution 只负责决定一个 episode 需要哪些 Camera。其边界如下：

```text
Camera Distribution
  -> selected camera IDs / profile set
  -> ResolvedTask
```

之后继续复用现有模块：

- Coverage：`simulation.camera.profiles.<camera_id>.coverage` 和现有 Coverage Model。
- Placement：现有 Partial placement resolver，例如 `P01` 的 `left_half_sideline_high_v1`。
- Resolution/Lens：现有 profile 参数和 focal length authority。
- Resolver：`config/resolver.py` 传递最终 profile 到 ResolvedTask。
- Runtime：`run_task.py` 选择已映射的 Camera 并应用现有参数。
- UE Mapping：将模型 Camera ID 映射到已有或未来明确提供的 UE Actor/Sequence。
- Sequence：现有 Level Sequence 和 Camera Cut 机制。
- MRQ：现有异步渲染流程。
- Annotation：现有 camera calibration、geometry annotation 和后处理流程。

Camera Distribution 不负责：

- Camera 位置生成。
- Coverage 几何验证算法本身。
- UE Actor 创建。
- Sequence 创建。
- Camera Cut 创建或修改。
- MRQ 队列控制。
- Camera 标定 artifact。
- MOT 或 Re-ID 标签生成。

不改变以下边界：

- `ValidationResult`
- Pipeline State
- Run Manifest
- Audit/Cleanup
- `camera_state.jsonl`

## 10. 推荐分布模型

当前推荐模型为：

```text
Primary category:
  Static Surveillance

Mandatory baseline:
  C1 C2 C3 C4 C5

Optional extension:
  selected P* cameras

Current available extension:
  P01

Future only:
  B* Dynamic Broadcast
```

推荐的当前任务模式：

```text
anchor_only:
  C1 C2 C3 C4 C5

anchor_plus_one_partial:
  C1 C2 C3 C4 C5 P01
```

不把 `Static Surveillance : Dynamic Broadcast` 设计成当前运行时概率比例。当前阶段的重点是确定性、稳定性和 MOT 数据质量；Broadcast 等未来类别不应提前增加主模型复杂度。

## 11. 设计取舍

### 固定 Anchor + 可选 Partial

优点：

- 保留稳定 benchmark。
- 明确区分全场和局部覆盖价值。
- Camera 数量可预估。
- 实现和审计简单。
- 与当前 P01 闭环直接兼容。

代价：

- 不会自动模拟所有真实场馆的摄像机部署差异。
- Partial Camera 覆盖范围需要后续逐个提供 placement template。
- 当前不能通过随机采样快速扩大视角种类。

### 不采用复杂概率分布

优点：

- 避免隐藏随机性。
- 不需要新调度系统。
- 更容易定位渲染和 MOT 差异来源。
- 能直接比较不同 Camera Count profile。

代价：

- 大规模数据集的类别频率需要在外层任务批量规划。
- 不会自动根据目标比例生成 episode。

当前阶段选择确定性优先，接受上述代价。

## 12. 非目标范围

本设计不实现：

- Camera 生成代码。
- 随机 Camera Generator。
- Camera 优化算法。
- 自动 UE Camera Actor 创建。
- 自动 Sequence 或 Camera Cut 创建。
- Dynamic Broadcast 系统。
- P02/P03 或其他未批准 Partial Camera。
- Camera Manager。
- Coverage Manager。
- Camera Database。
- Camera Registry。
- 动态任务调度系统。
- Polygon Coverage System。
- 新的 Geometry System。
- MRQ pipeline 改造。
- ValidationResult 修改。
- Pipeline State 修改。
- Run Manifest 修改。
- Audit/Cleanup 修改。
- `camera_state.jsonl` 接入或修改。
- MOT、Re-ID 或 annotation schema 改造。

## 13. 后续实现建议

后续如果继续扩展 Camera Distribution，建议按以下顺序：

1. 在同一个 `simulation.camera.profiles` 和现有 mapping contract 中明确表示 episode Camera 集合。
2. 保持 C1-C5 mandatory，并增加受控 Partial ID 集合校验。
3. 让 P01 作为当前唯一可执行 Partial profile 继续通过既有 placement/runtime 路径。
4. 先以 `anchor_only` 和 `anchor_plus_one_partial` 做 MOT 对比实验。
5. 只有在 P01 的数据质量和成本都被验证后，才定义 P02 等新的 Partial profile。
6. Dynamic Broadcast 另行设计，不提前纳入当前主分布。

每个后续 Camera 扩展都应先获得完整 profile、mapping、placement 验证和最小 UE smoke，再进入生产级数据生成。
