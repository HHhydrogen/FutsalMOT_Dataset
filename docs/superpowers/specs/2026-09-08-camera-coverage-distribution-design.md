# P2-3 Camera Coverage Distribution Design

状态：设计阶段完成，本文档不包含实现。

## 1. 当前 Camera 系统

当前项目使用单文件 Task Spec 作为任务入口，相机配置沿用既有运行时链路：

```text
Task Spec
  -> simulation.camera.profiles
  -> ResolvedTask
  -> camera_distribution.py
  -> run_task.py
  -> UE CineCameraComponent
```

P2-1 已建立五个固定 Static Surveillance canonical anchor：

| Canonical ID | UE Camera Actor | UE Level Sequence | 当前语义 |
| --- | --- | --- | --- |
| `C1` | `CineCam_01` | `LS_Cam_01` | 固定全场机位 |
| `C2` | `CineCam_02` | `LS_Cam_02` | 固定全场机位 |
| `C3` | `CineCam_03` | `LS_Cam_03` | 固定全场机位 |
| `C4` | `CineCam_04` | `LS_Cam_04` | 固定全场机位 |
| `C5` | `CineCam_Main` | `LS_Cam_Main` | 固定全场主机位 |

P2-2 已在现有 Camera Profile 中支持 resolution、focal length 和 sensor/filmback 相关参数。Coverage 只描述摄像机的目标空间观察范围，不改变已有相机参数、UE mapping、Sequence 或 MRQ 责任边界。

## 2. Coverage 设计目标

Coverage 的定义是：

> 一个摄像机能够观察球场哪些区域。

Coverage 是 `simulation.camera.profiles.<camera_id>` 中的 Camera Profile 属性。它描述目标语义，不直接负责求解相机摆放、旋转、镜头或 UE 资产。

本设计目标：

- 为当前固定全场 anchor 提供显式、可审计的 Coverage 语义。
- 为未来 Static Partial Camera 提供最小的区域分类和输入参数。
- 保持现有 flat profile 结构，不建立第二套 Coverage schema。
- 让未来生成流程能够从 Coverage 目标产生完整 resolved camera profile。
- 避免把 Coverage 混入 Pipeline State、渲染、标注或 artifact 管理。

Coverage 不负责：

- 计算相机位置和旋转。
- 选择 focal length、resolution 或 sensor。
- 创建 UE Camera Actor 或 Level Sequence。
- 控制 Camera Cut 或 MRQ。
- 描述可见像素、bbox、MOT 或 Re-ID 结果。
- 作为独立数据库、注册表或运行时服务存在。

## 3. Coverage 分类

### 3.1 `full_field`

`full_field` 表示目标是观察整个标准 Futsal 球场。它是当前 C1-C5 的唯一 Coverage 类型。

Profile 形状：

```json
{
  "coverage": "full_field"
}
```

规则：

- 必须显式声明 `coverage: "full_field"`。
- 不允许提供 `region`。
- 不允许提供 `coverage_ratio`。
- C1-C5 必须使用该值，不依赖默认 Coverage。
- `full_field` 是目标语义，不替代最终 UE 实际标定或视觉验收。

### 3.2 `partial_field`

`partial_field` 表示目标只观察球场的一部分。它适用于未来 Static Surveillance 扩展相机，不改变相机类型为静态监控的事实。

Profile 形状：

```json
{
  "coverage": "partial_field",
  "region": "left_half",
  "coverage_ratio": 0.5
}
```

规则：

- 必须显式声明 `coverage: "partial_field"`。
- 必须提供 `region`。
- `coverage_ratio` 可选。
- `region` 是主要语义，决定相机应关注的场地区域。
- `coverage_ratio` 只是辅助的离散目标描述，不等于经过投影、遮挡和裁剪后测量的真实几何面积。
- `partial_field` 不得通过过宽 FOV、过高机位或其他参数退化成没有区域限制的全场机位。

### 3.3 初始 Region 枚举

首版只定义以下六个区域：

| Region | 语义 |
| --- | --- |
| `left_half` | 球场左半场目标区域 |
| `right_half` | 球场右半场目标区域 |
| `goal_area_left` | 左侧球门及其附近目标区域 |
| `goal_area_right` | 右侧球门及其附近目标区域 |
| `sideline_left` | 左侧边线及其附近目标区域 |
| `sideline_right` | 右侧边线及其附近目标区域 |

这些值是有限的语义枚举，不是自由几何描述。首版不支持 polygon、bounding box、任意点集或用户自定义区域。

### 3.4 `coverage_ratio`

`coverage_ratio` 只作为 partial profile 的可选辅助字段。建议首版仅允许离散值：

```text
0.25 | 0.50 | 0.75
```

这些值表达生成目标的粗粒度覆盖级别，不宣称真实可见面积精确等于 25%、50% 或 75%。

如果未来 profile 不提供 `coverage_ratio`，生成流程仍必须依据 `region` 生成可验证的部分覆盖相机；不能因为字段缺失而默认为 full field。

## 4. 数据结构设计

Coverage 继续使用当前 Camera Profile 的 flat 字段，不改成嵌套 `coverage.type` 结构，也不增加独立配置系统。

概念结构：

```json
{
  "camera_id": "C1",
  "type": "static_surveillance",
  "coverage": "full_field",
  "resolution": [1920, 1080],
  "lens": {
    "focal_length_mm": 15.0
  },
  "position_m": [25.0, 12.5, 7.0],
  "rotation_deg": [-29.39, -140.19, 0.0],
  "height_m": 7.0,
  "distortion": null
}
```

未来 partial profile 的概念结构：

```json
{
  "camera_id": "P01",
  "type": "static_surveillance",
  "coverage": "partial_field",
  "region": "left_half",
  "coverage_ratio": 0.5,
  "resolution": [1920, 1080],
  "lens": {
    "focal_length_mm": 15.0
  },
  "position_m": [0.0, 10.0, 8.0],
  "rotation_deg": [-25.0, -90.0, 0.0],
  "height_m": 8.0,
  "distortion": null
}
```

### 4.1 字段要求

| 字段 | `full_field` | `partial_field` | 说明 |
| --- | --- | --- | --- |
| `coverage` | 必填，值为 `full_field` | 必填，值为 `partial_field` | Coverage 主类型 |
| `region` | 禁止 | 必填 | partial 的主要空间语义 |
| `coverage_ratio` | 禁止 | 可选 | 离散目标级别，不是真实面积 |
| `type` | `static_surveillance` | `static_surveillance` | 本设计只覆盖静态监控 |
| `position_m` | 由既有 profile 提供 | 由未来生成流程解析 | 不由 Coverage 字段直接替代 |
| `rotation_deg` | 由既有 profile 提供 | 由未来生成流程解析 | 不由 Coverage 字段直接替代 |
| `resolution` | 由既有 profile 提供 | 由未来生成流程解析 | 复用 P2-2 参数 |
| `lens` | 由既有 profile 提供 | 由未来生成流程解析 | focal length 仍是镜头 authority |

### 4.2 验证规则

未来实现应在现有 Task/Profile 验证边界内增加以下规则：

- `coverage` 只能是 `full_field` 或 `partial_field`。
- `full_field` 必须没有 `region` 和 `coverage_ratio`。
- `partial_field` 必须有合法 `region`。
- `partial_field.coverage_ratio` 如果存在，只能是 `0.25`、`0.50` 或 `0.75`。
- `partial_field` 不允许 `coverage_ratio >= 1.0`。
- `region` 只能使用初始六项枚举。
- C1-C5 必须是 `type: "static_surveillance"` 和 `coverage: "full_field"`。
- Coverage 验证不得推导或修改 position、rotation、lens、resolution。
- 生成完成后，resolved profile 必须包含完整可执行的相机参数；不能把未解析的 Coverage 目标交给 UE runtime。

## 5. C1-C5 关系

C1-C5 是固定 canonical anchor，不是随机生成结果：

- 始终使用同一个 Canonical Camera ID 体系。
- 始终属于 Static Surveillance。
- 始终显式声明 `coverage: "full_field"`。
- 不因为未来 partial camera 的存在而改变位置、旋转、镜头、resolution 或 Sequence 关系。
- 不由 Coverage distribution ratio 随机替换或删除。

未来 Static Partial Camera 继续使用同一个 camera ID namespace。例如：

```text
C1-C5  -> 固定 full_field anchors
P01/P02 -> 未来生成的 partial static cameras
```

`P01`、`P02` 是 Canonical/Runtime camera ID 的扩展命名，不是第二套身份系统。它们仍然通过既有 resolved task 和 UE mapping 进入同一条运行时链路。UE Actor 名称不能成为模型层 camera ID。

## 6. Future Partial Camera 边界

本阶段只定义目标和边界，不实现生成算法、UE Actor 创建或 Sequence 处理。

未来流程建议如下：

```text
Task Spec profile
  -> coverage=partial_field + region
  -> 可选 coverage_ratio
  -> 受控候选生成
  -> 解析 position / rotation / focal length / resolution
  -> 实际覆盖范围验证
  -> 生成完整 resolved camera profile
  -> 复用现有 UE mapping / run_task.py / UE Camera runtime
```

### 6.1 输入

输入只包含既有 profile 中的 Coverage 目标字段，以及已有 camera 参数约束：

- `coverage: "partial_field"`。
- 一个合法 `region`，例如 `left_half`。
- 可选 `coverage_ratio`。
- 既有 resolution、focal length、sensor 和静态相机约束。
- Task seed 或其他已有可复现输入，若未来需要受控采样。

### 6.2 输出

未来生成流程必须输出完整的 resolved camera profile，至少包括：

- camera ID。
- `type`。
- `coverage`。
- `region`。
- 可选 `coverage_ratio`。
- position。
- rotation。
- focal length/lens。
- resolution。
- 与现有 UE runtime 所需的 mapping 信息。

Coverage 只表达“想看哪里”，不负责解决“相机放在哪里”。摆放算法属于未来的受控生成实现，并且必须在生成后通过实际几何/可见范围检查验证结果。

### 6.3 生成门禁

未来实现不应只检查配置字段是否填写，而应检查候选相机的实际空间结果：

- 指定 region 是否进入相机有效视野。
- 相机是否仍然覆盖了目标区域，而不是完全偏离目标。
- partial 相机是否意外退化为 full-field。
- position、rotation、focal length 和 resolution 是否形成可执行 profile。
- 生成结果是否确定性可复现，并写入 ResolvedTask。

如果候选不满足目标，应按受控规则重新生成或 fail fast；不得静默接受不符合 Coverage 语义的结果。

## 7. MOT 数据价值分析

### 7.1 Full Field

优点：

- 球员在较长时间内保持可见。
- 更容易建立完整轨迹。
- 适合全局战术分析、队形分析和跨区域运动分析。
- C1-C5 之间可以提供稳定的多摄像机互补基线。

限制：

- 单相机 MOT 难度相对较低。
- 出视野和重新出现事件较少。
- 对 Re-ID、短时遮挡和跨区域恢复能力的压力较小。
- 不能单独代表真实场馆中局部机位的观测缺失。

### 7.2 Partial Field

优点：

- 更容易产生球员出视野和重新出现事件。
- 增加 track interruption 和 fragment。
- 提升 Re-ID 难度，尤其是左右半场或边线区域之间的切换。
- 与 full-field anchor 形成空间互补。
- 适合研究局部观察、区域统计和跨摄像机关联。

限制：

- 单个相机不能提供完整全局信息。
- 轨迹中断可能来自视野边界，而不是目标真正消失或遮挡。
- 目标区域定义不清时，容易把 Coverage 语义和实际几何可见性混淆。
- 如果没有真实覆盖验证，partial 配置值可能只是标签，不能代表画面事实。

### 7.3 Region 差异

- `left_half` / `right_half`：适合制造半场级的轨迹断裂和跨半场重识别问题。
- `goal_area_left` / `goal_area_right`：适合局部高密度事件和球门附近遮挡，但全局轨迹信息最不完整。
- `sideline_left` / `sideline_right`：适合边线运动、出入视野和侧向观察差异，但需要避免把边线目标误解成任意 polygon 区域。

这些类别用于控制数据生成目标，不直接决定 MOT 标签内容。最终标签仍由实际渲染、Mask 和现有 annotation pipeline 产生。

## 8. Future Distribution Strategy

未来分布策略应以 Static Surveillance 为优先：

1. 固定保留 C1-C5 full-field anchors，作为每个兼容任务的稳定基线。
2. 按任务需要增加零个或多个 `partial_field` profile。
3. Partial profile 通过 `region` 表达区域语义，通过可选 `coverage_ratio` 表达受控目标级别。
4. 生成候选必须经过实际 Coverage 验证后才能进入 ResolvedTask。
5. 暂不规定生产数据中的精确相机数量或比例；比例应在后续实验设计中根据 MOT 任务目标、存储成本和视觉验收结果确定。

该策略不要求每个最小任务都包含 partial camera，也不允许用 partial camera 随机替换 C1-C5 anchor。Full-field baseline 与 partial extension 的职责应保持可区分。

## 9. Future Implementation Boundary

未来实现可以修改或扩展现有 `simulation.camera.profiles` 相关模型和解析逻辑，但应保持以下边界：

- 不新增 Camera Manager、Coverage Manager、Region Database、Camera Database 或独立 registry。
- 不新增与 Task Spec 平行的配置入口。
- 不改变 `ResolvedTask` 作为运行时任务契约的角色。
- 不把 Coverage 放入 Pipeline State、Run Manifest、ValidationResult 或 Audit/Cleanup。
- 不让 Coverage 直接创建或管理 UE Actor、Level Sequence 或 MRQ。
- 不把 coverage ratio 当作真实面积标注或替代实际相机标定。
- 不把 coverage region 写成自由 polygon、bounding box 或复杂几何系统。
- 不改变 `camera_state.jsonl` 的职责；未来若需要记录实际状态，仍应沿用现有 artifact 约束。

## 10. 非目标范围

本设计明确不实现：

- 随机 Camera 生成。
- Static Partial Camera 的 UE Actor 创建。
- 新的 Camera Actor。
- Level Sequence 创建或修改。
- Camera Cut 修改。
- MRQ 配置变化。
- Camera Manager、Coverage Manager 或数据库。
- polygon、bounding box、自由区域或复杂 Geometry System。
- 动态 Broadcast Camera。
- distortion model。
- 新的 annotation、MOT、Re-ID 或 artifact system。
- `camera_state.jsonl` 修改或生成。
- ValidationResult、Pipeline State、Run Manifest、Audit/Cleanup 修改。

## 11. 设计结论

Coverage 采用最小 flat 模型：

```text
coverage: full_field

coverage: partial_field
region: left_half
coverage_ratio: 0.5  # 可选辅助信息
```

`coverage` 是主分类，`region` 是 partial 的主要语义，`coverage_ratio` 只是离散辅助目标。C1-C5 明确固定为 Static Surveillance full-field anchors；未来 P01/P02 等 partial camera 继续使用同一 camera ID 体系和既有 Task Spec -> ResolvedTask -> UE runtime 链路。
