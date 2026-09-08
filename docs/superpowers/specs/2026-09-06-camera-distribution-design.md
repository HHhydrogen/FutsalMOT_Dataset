# P2-1 Camera Distribution Model Design

状态：设计阶段完成，等待确认后编码

基线：`3b0e476aa483771b396ca4ed22381236989d4090`

## Current Camera Pipeline Analysis

### Dataset repository

当前正式任务入口是单文件 Task Spec。`DatasetTaskConfig` 在 `src/grf_ue_bridge/config/models.py` 中定义顶层 `export`、`ue`、`postprocess` 和 `audit`；其中 `ue` 的 `sequences` 与 `annotation_export` 是现有相机相关配置。当前 schema 没有独立的 `simulation` 字段，也没有相机分布模型。

`resolve_task()` 在 `src/grf_ue_bridge/config/resolver.py` 中将 Task Spec 解析为 `ResolvedTask`，并把 `task.ue.model_dump()` 原样放入 `ResolvedTask.ue_profile`。它只补充 `annotation_export.playback_fps`，不会生成相机状态或改变相机 Actor。`task_export` 会把 sanitized task、`ue-profile.json` 等 provenance 写入 episode，但 P1 轨迹导出不消费 UE 相机参数。

当前 P2 正式入口是 `ue/run_task.py`：

1. `full` 模式读取 resolved task。
2. `create_sequence()` 根据 `ue_profile.sequences` 创建或覆盖 Level Sequence。
3. `export_annotations()` 根据 `annotation_export.cameras` 查找 UE Camera Actor，读取其当前 CineCameraComponent 标定和世界变换。
4. `render_sequences()` 使用这些 Sequence 提交异步 MRQ RGB 和 Object-ID Mask 渲染。
5. MRQ 完成后把对齐帧写入每个相机的 `img1/`，并写 `render_summary.json`。

现有 episode artifact 中，每个相机已有 `camera.json`、`annotations.jsonl`、`img1/`、可选 `mask/` 和 MOT/YOLO 产物。`camera.json` 是从实际 UE Camera Actor 读取的标定结果，包含图像尺寸、内参、焦距、FOV、世界位置、旋转和方向基向量。当前没有 `camera_state.jsonl` 的生成器、读取器或验证器。

### UE repository

当前外层 UE 项目为 Unreal Engine 5.8，正式地图是 `/Game/FutsalMOT/Maps/L_FutsalCourt`。`Config/DefaultEngine.ini` 仍指向不存在的 `L_Futsal_Demo`，因此相机任务不能依赖默认启动地图，必须在执行前确认当前 Editor Level。

可读配置和资产引用只暴露现有 `CineCam_01..04` 的 Sequence 关系。Camera Actor 的实际标签、位置和旋转存储在二进制 `L_FutsalCourt.umap` 中，不能从 Git 文本 diff 得到完整事实；设计不能假设这些 Actor 已经语义上等同于 `C1..C4`，也不能假设 `C5` 已存在。

`ue/import_grf_episode.py` 的 `create_sequence()` 当前将 `sequences` 中的 `camera_actor` 查找为 Level Actor，使用 `seq.add_possessable(cam_actor)`，添加 Camera Cut，并添加相机 transform track。相机 transform track 会把当前 Actor 变换烘焙到 Sequence，避免首帧退回关卡默认视角。

`ue/render_episode.py` 按相同的 `sequences` 列表加载 Level Sequence，并将其提交给 `MoviePipelineQueueSubsystem` / MRQ。MRQ 的相机选择来自 Sequence 的 Camera Cut，而不是单独的 Camera Manager。现有实现支持在提交 MRQ 前通过 Python 设置 CineCameraComponent 的后处理属性，也支持通过 UE Python 设置 Actor transform；尚未实现按模型参数创建、选择或随机化相机。

### Current pipeline conclusion

现有相机管线是“Task Spec 指定 UE 名称 -> resolved task 传递 -> Sequence 绑定 Camera Actor -> MRQ 按 Camera Cut 渲染 -> UE 读取实际标定写 `camera.json`”。P2-1 应扩展这个边界，而不是另建相机系统：模型负责决定 episode 的相机分布和生成参数，UE mapping 负责把模型相机绑定到可执行的 Actor/Sequence。

## Architecture Boundary

Camera Distribution Model 属于 Task Spec / simulation configuration 层，职责是描述一个 episode 应该拥有哪些相机、各相机的语义类型、分布比例、参数约束和随机策略。

它不负责：

- 管理长期存在的 Camera Actor；
- 保存相机数据库或跨 episode 资产注册表；
- 维护 Camera Manager、Simulation Framework 或运行时服务；
- 改变 Pipeline State、ValidationResult 或 Run Manifest 的职责；
- 替代 `camera.json` 的实际 UE 标定结果。

数据流保持如下：

```text
Task Spec
  -> simulation.camera
  -> Runtime Config / task model
  -> ResolvedTask.ue_profile.simulation.camera
  -> UE camera distribution application
  -> existing sequences / MRQ / annotation export
  -> camera.json + camera_state.jsonl
```

`ResolvedTask` 仍然是 P1/P2 之间唯一的运行时任务契约。不得创建 `camera_config.json`、`camera_database` 或与 task 平行的配置入口。

## Canonical Camera ID Design

模型层使用稳定、与 UE 资产解耦的 Canonical Camera ID：`C1`、`C2`、`C3`、`C4`、`C5`。

语义定义如下：

| ID | 语义 | 默认类型 | 默认覆盖 |
| --- | --- | --- | --- |
| `C1` | 左上角 corner full court | `static_surveillance` | `full_field` |
| `C2` | 右上角 corner full court | `static_surveillance` | `full_field` |
| `C3` | 左下角 corner full court | `static_surveillance` | `full_field` |
| `C4` | 右下角 corner full court | `static_surveillance` | `full_field` |
| `C5` | 中线 main full court | `static_surveillance` | `full_field` |

`CineCam_01..04` 不直接等同于 `C1..C4`。它们是当前 UE 资产名称，必须经过显式 mapping 才能参与某个 episode。模型、标注和 artifact 使用 Canonical ID；UE 资产绑定只在运行时配置中保存 UE 名称。

首版的硬约束是：一个启用的 static surveillance anchor profile 必须解析出完整的 `C1..C5` 五个 slot。若某个 slot 没有 UE mapping、Actor 不存在或无法建立 Camera Cut，任务应在 UE 执行前失败，而不是静默减少相机数。

## UE Mapping Strategy

建议把 mapping 放在既有 `ue` 块内，而不是单独文件。概念结构如下：

```json
{
  "ue": {
    "camera_mapping": {
      "C1": {"actor": "CineCam_01", "sequence": "LS_Cam_01"},
      "C2": {"actor": "CineCam_02", "sequence": "LS_Cam_02"},
      "C3": {"actor": "CineCam_03", "sequence": "LS_Cam_03"},
      "C4": {"actor": "CineCam_04", "sequence": "LS_Cam_04"},
      "C5": {"actor": "CineCam_Main", "sequence": "LS_Cam_Main"}
    }
  }
}
```

这只是 mapping 形状，不代表当前地图已有 `CineCam_Main`。首版可以把 `C1..C4` 映射到待核验的现有 Actor，`C5` 映射到未来新增的 UE Camera Actor。模型层不因 Actor 更名、资产迁移或 C5 的创建方式改变。

现有 `ue.sequences` 是列表结构，后续实现应保持兼容或一次性明确迁移为以 Canonical ID 为键的结构。推荐的长期语义是：Sequence entry 持有 `camera_id`，并通过 mapping 解析 `camera_actor`；不允许把 UE Actor 名称重新提升为模型 ID。

Mapping 的最小校验包括：

- 每个启用的 Canonical ID 恰好一个 UE Actor 和一个 Sequence；
- 一个 UE Actor 不被两个 active Canonical ID 复用，除非未来显式允许同一视角别名；
- Sequence 的 Camera Cut 与对应 Actor 可建立；
- `C1..C5` 的模型语义不会由 Actor 标签推断；
- 运行时最终使用的 Actor 名称写入 `camera_state.jsonl`。

## Camera Distribution Model

最终采用 Dual Camera System：

- `static_surveillance`：核心，占 episode 相机样本目标的 70% 至 80%；
- `dynamic_broadcast`：辅助，占 20% 至 30%。

比例是生成分布目标，不要求每个只有五个 anchor 的最小 episode 恰好满足百分比。固定 anchor 属于 static surveillance 基线；额外 partial camera 和 broadcast camera 由 task 中的数量或采样策略决定。任务必须能在 resolved task 中得到确定的相机列表和参数，不能把未决的随机选择留给 MRQ。

空间策略采用 Hybrid Strategy：

1. 先建立固定的 `C1..C5` canonical anchors。
2. 再按 task 的 random profile 生成零个或多个扩展相机。
3. 每个扩展相机必须符合其 type constraint，并生成稳定的 runtime camera ID。

建议扩展相机 ID 使用 `P01`, `P02` 和 `B01` 等明确前缀，避免与 `C1..C5` 混淆。扩展相机的模型语义仍由 Task Spec 定义，UE 侧只负责绑定或创建实现所需的 Actor/Sequence。

## Static Surveillance Design

### Fixed full-court anchors

五个 anchor 必须固定位置、固定方向和全场覆盖。固定定义属于模型语义，不做 episode 内随机化：

- `C1`：左上角；
- `C2`：右上角；
- `C3`：左下角；
- `C4`：右下角；
- `C5`：中线主视角。

具体 UE 坐标、旋转、相机高度和镜头数值必须显式进入 task 的 runtime camera configuration 或由受控 preset 解析得到，不能依赖 CineCamera 的隐式默认值。设计阶段不提交地图资产，也不声称当前 `L_FutsalCourt` 已满足五个视角的视觉覆盖要求；实现阶段需要在 UE Editor 中核验。

### Static partial cameras

Partial camera 是可选的 static surveillance 扩展，用于增加遮挡、出入视野和 re-identification 难度。第一版只支持离散 coverage 值：`25`, `50`, `75`，表示目标覆盖比例，而不是任意连续百分比。

区域枚举首版建议为：`left_half`、`right_half`、`penalty_area`、`corner_area`。区域与 coverage 必须同时约束相机候选空间。`coverage` 不得为 `100`，也不得通过过宽 FOV、过高位置或退回 full-field preset 使 partial camera 实际退化为全场相机。

实现时应使用几何/可见区域检查作为生成门禁：相机候选若覆盖超过 partial 上限或无法覆盖指定区域，则重采样或失败；不能仅记录配置值而不检查实际画面。

## Dynamic Broadcast Design

只保留一个 dynamic broadcast profile：`main_broadcast`。首版不实现 replay camera、handheld camera 或 tactical moving camera。

Broadcast camera 的语义是电视主摄像机，支持以下行为参数：

- `smooth_follow`：平滑跟随目标点或场地关注点；
- `pan`：水平转动范围与速度约束；
- `tilt`：垂直转动范围与速度约束；
- `zoom`：焦距或 FOV 的平滑变化。

动态行为必须由确定的 episode seed 和显式 profile 生成，使 camera state 能被重放。第一版不需要建立通用轨迹编辑器；可以使用 runtime camera transform / focal parameters 在序列生成阶段写入已有 Sequence。Broadcast 的跟随目标、速度限制和变焦范围属于 simulation.camera 参数，不能从 UE 默认蓝图行为隐式获得。

## Camera Parameter Model

每个 Canonical 或扩展 camera 都应解析为一个完整 camera profile。建议字段如下：

```json
{
  "camera_id": "C1",
  "type": "static_surveillance",
  "coverage": "full_field",
  "region": "corner_north_west",
  "resolution": [1920, 1080],
  "lens": {
    "focal_length_mm": 24.0,
    "horizontal_fov_deg": 72.0
  },
  "position_m": [0.0, 0.0, 8.0],
  "rotation_deg": [0.0, 0.0, 0.0],
  "height_m": 8.0,
  "distortion": null
}
```

参数优先级固定为：

1. `resolution`；
2. `lens.focal_length_mm` 或 `lens.horizontal_fov_deg`；
3. `position_m`；
4. `rotation_deg`；
5. `height_m`。

这里的优先级表示模型必须显式拥有并验证这些参数，不表示后面的字段可以覆盖前面的字段。`height_m` 是位置语义的一部分，若同时提供 `position_m[2]`，实现必须检查二者一致或按明确规则拒绝冲突。焦距和水平 FOV 至少一个必须给出；若同时给出，应由实现校验二者与 sensor/分辨率关系一致。

`distortion` 只作为预留字段，第一版固定不实现复杂失真。正式 CV GT preset 默认仍关闭镜头失真，以保持 RGB 与 Object-ID Mask 的空间边界契约。

## Resolution/Lens/FOV Design

支持的分辨率档位为 `720p`、`1080p` 和 `4K`，也允许 runtime schema 直接保存 `[width, height]`，但实现应对档位做规范化。分辨率最终必须同时用于 MRQ 输出与 `camera.json` 标定，继续遵守现有 `render_rgb.output_resolution_x/y` 与 calibration 的一致性校验。

相机类型的镜头分布建议：

- static full court：较宽视场，优先保证全场覆盖与球员可检测尺寸；
- static partial：按区域和 coverage 选择中等到较宽视场，不得扩大到 full-field；
- dynamic broadcast：较长焦、较窄视场，焦距/FOV 随 broadcast profile 平滑变化。

第一版只允许数值范围或离散 preset，不引入镜头数据库。任何随机采样都必须在 resolved task 中固定为可审计的结果，至少使用 task seed 派生，不依赖 UE 默认随机状态。

## Coverage Model

Coverage 是模型级离散变量：

```text
full_field = 100%
partial = 25% | 50% | 75%
```

`full_field` 只允许 anchor 或明确的 full-field static profile。Partial camera 必须带 `region`，并满足 `coverage < 100%`。Broadcast camera 的 coverage 不使用 partial 枚举强行描述；它使用 `broadcast_profile` 和实际 camera state 描述当前视野，必要时可以记录估计覆盖区域，但不把动态视野伪装成静态 partial。

## Multi-Camera Relationship

第一版只保留轻量关系 metadata，不实现 graph system。每个 camera 可选记录：

```json
{
  "overlap_group": "full_court_baseline",
  "overlap_with": ["C2", "C5"],
  "topology_role": "anchor"
}
```

这些字段用于未来分析 camera overlap、topology 和 cross-camera association，不参与当前 camera 生成决策，也不建立独立关系数据库。关系引用必须使用 Canonical/runtime camera ID，不能使用 UE Actor 名称。

## camera_state.jsonl Artifact Design

`camera_state.jsonl` 是 episode 级 dataset artifact，不是管理系统、数据库或运行时服务。建议位置：

```text
<dataset_root>/<episode_name>/camera_state.jsonl
```

每一行记录一个 camera 在一个采样帧或状态采样点的解析状态。静态相机可以每个 frame 记录相同的 transform；实现也可在第一版选择按状态变化写记录，但必须在 schema 中固定采样语义。推荐首版按输出 frame 写，以便与 `annotations.jsonl`、MRQ 帧号和动态 broadcast 对齐。

建议结构：

```json
{
  "schema": "futsalmot_camera_state",
  "version": 1,
  "frame": 100,
  "source_step": 100,
  "time_seconds": 10.0,
  "camera_id": "C1",
  "ue_actor": "CineCam_01",
  "sequence": "LS_Cam_01",
  "type": "static_surveillance",
  "coverage": "full_field",
  "region": "corner_north_west",
  "resolution": [1920, 1080],
  "focal_length_mm": 24.0,
  "horizontal_fov_deg": 72.0,
  "position_m": [1.0, -8.0, 8.0],
  "rotation_deg": [0.0, 45.0, 0.0],
  "height_m": 8.0,
  "distortion": null,
  "relationship": {
    "overlap_group": "full_court_baseline",
    "overlap_with": ["C2", "C5"],
    "topology_role": "anchor"
  }
}
```

`frame` 使用数据集输出帧的 0 基编号，`source_step` 保留与现有 frames/annotation 的对应关系；如最终采用 1 基 artifact frame，则必须在实现前统一命名并写入契约，不能与现有 `annotations.jsonl` 混用。建议继续使用 `frame` 0 基、`annotations.jsonl.frame_index` 1 基的现有边界。

`camera_state.jsonl` 与每相机 `camera.json` 的职责不同：

- `camera_state.jsonl`：模型 ID、UE mapping、类型、coverage、关系以及逐帧/逐状态的生成结果；
- `camera.json`：UE 当前 CineCameraComponent 的实际 calibration 和外参快照，用于投影和标注。

实现必须在写 artifact 前确认 state 中的 resolution、focal/FOV、position/rotation 与实际 UE Actor/Component 一致；否则应失败或显式标记不一致，不能生成看似合法的 metadata。

## Task Spec Extension Proposal

建议在现有 Task Spec 内新增 `simulation.camera`，不新增实体文件：

```json
{
  "simulation": {
    "camera": {
      "distribution": {
        "system": "dual",
        "static_surveillance_ratio": [0.70, 0.80],
        "dynamic_broadcast_ratio": [0.20, 0.30],
        "anchors": ["C1", "C2", "C3", "C4", "C5"],
        "partial": {
          "enabled": true,
          "count": 1,
          "coverage": [25, 50, 75],
          "regions": ["left_half", "right_half", "penalty_area", "corner_area"]
        },
        "broadcast": {
          "enabled": true,
          "count": 1,
          "profile": "main_broadcast"
        }
      },
      "profiles": {
        "C1": {
          "type": "static_surveillance",
          "coverage": "full_field",
          "resolution": [1920, 1080],
          "lens": {"horizontal_fov_deg": 72.0},
          "position_m": [1.0, -8.0, 8.0],
          "rotation_deg": [0.0, 45.0, 0.0],
          "height_m": 8.0,
          "distortion": null
        }
      }
    }
  },
  "ue": {
    "camera_mapping": {
      "C1": {"actor": "CineCam_01", "sequence": "LS_Cam_01"}
    }
  }
}
```

上述字段是设计 proposal，不修改当前 schema。建议实现时将 `simulation` 作为 `DatasetTaskConfig` 的显式模型字段，而非继续把所有内容塞入 `ue.annotation_export`；`ue.annotation_export` 继续负责输出相机列表、标注和 render 设置。`simulation.camera` 负责“应生成什么相机”，`ue.camera_mapping` 负责“使用哪个 UE 资产承载它”，两者不得合并。

Resolver 应把 `simulation` 和 `ue.camera_mapping` 归一化复制到 `ResolvedTask`，建议最终形态为：

```text
ResolvedTask.simulation.camera
ResolvedTask.ue_profile.camera_mapping
ResolvedTask.ue_profile.sequences
```

为保持单一运行时契约，不建议新增顶层 `ResolvedCameraConfig` 文件。Resolved task 必须包含已确定的 camera IDs、采样参数、随机种子派生结果和 UE mapping，UE 脚本不读取原始 Task Spec。

## UE Integration Points

首版实现接入点按现有职责划分：

1. **Task resolver**：校验 camera distribution、解析 Canonical ID、保留随机 seed 结果和 mapping。
2. **UE sequence creation**：在 `create_sequence()` 前验证 mapping；按 Canonical ID 找到 UE Actor，生成或更新已有 Sequence 的 Camera Cut 与 transform track。
3. **Runtime transform application**：在 Sequence 创建/渲染前通过 UE Python 设置 Actor transform 和 CineCameraComponent 参数。静态 anchor 使用固定参数；partial 使用 resolved candidate；broadcast 将动态轨迹写入已有 Sequence track。
4. **Annotation export**：按 Canonical ID 遍历相机，但仍从实际 UE Actor 读取 `camera.json`，保证投影使用真实状态。
5. **MRQ**：继续由现有 `render_sequences()` 按 Sequence 的 Camera Cut 选择视角；不新增 MRQ 相机服务。
6. **Artifact output**：在相机状态已经应用并验证后，由 UE 侧写 episode 级 `camera_state.jsonl`。

关于 C5，最佳接入方式是新增一个普通 CineCamera Actor 并由 mapping 指向它，再沿用现有 Sequence possessable、Camera Cut 和 transform track 流程。是否在地图中手工放置、由 Editor Python 创建，或由已有资产复制，属于 UE 资产实施选择；这些选择不泄漏到 `C5` 模型 ID。首版验收必须通过 Unreal MCP/Editor 检查实际 Actor、Sequence、Camera Cut、分辨率、FOV、位置和旋转。

## Implementation Phases

### Phase 1: Schema and resolver contract

- 在 Task Spec 增加 `simulation.camera` 与 `ue.camera_mapping` 的模型和校验。
- 将 camera distribution 与 mapping 传入 ResolvedTask。
- 明确随机 seed 派生规则和 canonical/runtime camera ID 规则。
- 增加纯 Python schema/resolver 测试，不触碰 UE 资产。

### Phase 2: Anchor mapping and static profiles

- 为 `C1..C5` 建立五个 anchor slot 的显式 mapping。
- 核验现有 `CineCam_01..04` 的实际位置、旋转、覆盖和 Sequence 关系。
- 为缺失的 C5 选择并实施 UE Camera Actor 接入方式。
- 让 UE Sequence 创建和 annotation export 使用 Canonical ID 到 Actor 的解析结果。

### Phase 3: Runtime camera application and artifact

- 实现静态 anchor、partial candidate 和显式 resolution/lens/transform 应用。
- 在 MRQ 前验证实际 CineCamera 状态与 resolved profile。
- 写 `camera_state.jsonl`，并在每行记录 Canonical ID 与 UE mapping。
- 复用现有 `camera.json`、annotation、MRQ 和 audit 流程；不修改 Pipeline State、ValidationResult 或 Run Manifest。

### Phase 4: Dynamic broadcast

- 只实现 `main_broadcast` profile。
- 生成确定性 smooth follow、pan、tilt、zoom 轨迹并写入已有 Sequence。
- 验证动态 camera state 与 MRQ 帧一一对应。
- 通过小帧数和可视化 overlay 验收平滑性、视野边界和标定一致性。

### Phase 5: Partial distribution and relationship metadata

- 加入 partial 25/50/75 coverage 和区域约束的生成门禁。
- 写 overlap/topology 轻量 metadata，不实现 graph 运算。
- 扩展 audit 或独立 artifact 检查只到必要程度，避免扩大当前验证系统职责。

## Risks

- **UE 地图事实不完整**：Actor 位置、标签和 C5 是否存在位于二进制 `.umap`，静态代码检查无法证明；必须在 UE Editor 中核验。
- **默认地图错误**：`DefaultEngine.ini` 仍引用不存在的 `L_Futsal_Demo`，自动化执行前必须显式加载或确认 `L_FutsalCourt`。
- **Sequence 覆盖风险**：现有 `replace_existing=true` 会删除并重建 Sequence；实现必须避免把 mapping 错误扩散到无关 Sequence。
- **Camera Cut 失败**：UE API 版本差异可能导致 Camera Cut 创建失败，MRQ 可能渲染错误视角；应在提交 MRQ 前做 binding 和 cut 校验。
- **标定与实际渲染不一致**：分辨率、焦距、filmback、FOV 或 transform 若只写入配置而未应用到 Actor，会造成 `camera.json` 与 RGB/Mask/Pose 错位。
- **Partial 退化**：仅记录 `coverage=25/50/75` 不能证明实际视野是 partial，需要几何或渲染后的覆盖检查。
- **动态相机不可复现**：broadcast 若使用 UE 隐式时间或随机状态，无法重建 camera state；必须由 resolved seed 和显式轨迹参数驱动。
- **artifact 版本兼容**：`camera_state.jsonl` 是新 artifact，必须定义 schema/version、frame convention 和 canonical ID 规则，避免与旧 episode 的 camera discovery 混淆。
- **比例解释错误**：70/30 是跨 episode 的目标分布，不应要求每个只有 5 个 anchor 的 episode 精确满足比例。
- **范围膨胀**：camera relationship、distortion、复杂动态行为很容易演变为独立系统；首版必须保持为 task 参数和 artifact metadata。

## Investigation Summary

- 当前 Dataset Task Spec 的相机入口是 `ue.sequences`、`ue.annotation_export.cameras` 和 `render_rgb`，没有 Camera Distribution Model。
- `ResolvedTask` 是现有 P1/P2 运行时契约，适合承载 `simulation.camera` 的 resolved 结果。
- UE 正式入口是 `ue/run_task.py`；Sequence 创建、annotation export 和 MRQ 渲染均已存在。
- Sequence 通过 `add_possessable()`、Camera Cut 和 transform track 引用 Camera Actor。
- MRQ 选择 Sequence 的 Camera Cut；没有独立 Camera Manager。
- UE Python 支持运行前读取和修改 Actor transform 以及 CineCameraComponent 参数，但尚未实现 camera generation。
- 当前每相机已有 `camera.json`，没有 `camera_state.jsonl`。
- 当前可读资产只暴露 `CineCam_01..04` 的 Sequence 引用；`L_FutsalCourt.umap` 中的 Actor 状态需要 Editor/MCP 实测。
- 当前 `DefaultEngine.ini` 的默认地图路径错误，执行 UE 任务时不能依赖默认地图。

## Recommended Design

采用“Canonical Camera ID + UE Mapping”的三层架构：

1. Dataset Camera Model 使用 `C1..C5` 及扩展 runtime IDs 表达研究语义。
2. Runtime Camera Configuration 放在 `simulation.camera`，显式保存 type、coverage、resolution、lens/FOV、position、rotation、height、seed 和关系 metadata。
3. UE Asset Binding 放在现有 `ue` 块，通过 mapping 把 Canonical ID 绑定到 Actor 和 Sequence。

固定五个 static surveillance anchor，采用 Dual Camera System 和 Hybrid Strategy；partial 作为受约束的随机扩展，dynamic 只保留一个 main broadcast profile。`camera_state.jsonl` 作为 episode artifact，记录每帧相机状态、Canonical ID 和 UE mapping；`camera.json` 继续作为真实 UE calibration artifact。

## Alternative Options Considered

### Option A: 直接使用 UE Actor 名称

例如把 `CineCam_01` 直接作为模型 ID。实现改动最少，但数据集语义绑定到地图资产命名，C5 新增、资产迁移和跨地图复用都会破坏历史含义。拒绝。

### Option B: Canonical Camera ID + UE Mapping

使用 `C1..C5` 作为稳定模型 ID，通过 `ue.camera_mapping` 映射 Actor/Sequence。增加少量解析和校验，但保持研究语义与 UE 资产解耦，支持当前四个 Actor 和未来 C5。采用。

### Option C: 独立 Camera Database / Camera Manager

把相机 preset、资产和关系集中到独立数据库或运行时管理器。可扩展性强，但违反任务约束，增加状态来源、生命周期和验证边界，也无法解决当前 `.umap` 资产事实需要 Editor 验收的问题。拒绝。

## Files To Modify In Future

以下是后续编码阶段的候选文件，不在 P2-1 设计阶段修改：

- Dataset `src/grf_ue_bridge/config/models.py`：Task Spec / ResolvedTask 模型。
- Dataset `src/grf_ue_bridge/config/resolver.py`：simulation camera 与 mapping 的解析和校验。
- Dataset `src/grf_ue_bridge/ue/run_task.py`：读取 resolved camera contract 并传递给 UE workflow。
- Dataset `ue/import_grf_episode.py`：Canonical ID 到 Camera Actor 的 Sequence、Camera Cut 和 transform track 接入。
- Dataset `ue/annotation_exporter.py`：Canonical ID、camera state 与真实 camera calibration 的关联。
- Dataset `ue/render_episode.py`：MRQ 前的状态应用/验证和 camera state 完成处理。
- Dataset `ue/scene_apply.py` 或新增极小的 camera profile helper：仅在现有职责无法容纳时使用，不创建 Camera Manager。
- Dataset `tests/test_task_config.py`、`tests/test_task_resolver.py`、`tests/test_ue_resolved_task.py`：schema/resolver contract 测试。
- Dataset 新增相机模型纯函数测试：coverage、ID、seed、参数一致性和 frame mapping。
- Dataset `docs/DATA_CONTRACT.md`：实现后补充新 Task/Artifact 契约。
- Dataset `docs/VALIDATION_AND_LIMITATIONS.md`：实现后记录实际验证边界。
- UE `Content/FutsalMOT/Maps/L_FutsalCourt.umap`：仅在实现阶段按确认后的资产方案处理 C5 或 anchor slot；不在设计阶段改动。

## Files Not To Touch

- 不创建 `camera_config.json`、`camera_database` 或独立 camera configuration repository。
- 不新增 Camera Manager、Simulation Framework 或 Dataset Management System。
- 不修改 Blueprint、MCP 插件或插件 toolset 作为本设计的一部分。
- 不修改 `ValidationResult`、Pipeline State 或 Run Manifest 的核心 schema/职责。
- 不把 Canonical ID 替换为 `CineCam_01..04`。
- 不修改已有 `camera_projection.py` 的坐标约定，除非实现阶段发现实际 UE 标定错误且另行确认。
- 不在设计阶段修改 `.umap`、`.uasset`、Sequence asset 或默认地图配置。
- 不把动态 broadcast、partial coverage 或 relationship metadata 做成独立服务。

## Implementation Roadmap

1. 用户确认本设计和 `simulation.camera` / `ue.camera_mapping` 的字段命名。
2. 在 Dataset 子仓库实现 schema/resolver contract 和纯 Python 测试。
3. 生成一个只含 `C1..C5` 的 static anchor smoke task，验证 resolved task 中的 Canonical ID 与 UE mapping。
4. 通过 Unreal MCP 核验 `L_FutsalCourt`、现有 `CineCam_01..04` 和 C5 slot 的实际状态。
5. 实现 anchor state application、Sequence Camera Cut 和 `camera_state.jsonl`，运行小帧数 UE smoke。
6. 对比每相机 `camera.json`、`camera_state.jsonl`、MRQ RGB/Mask 和 annotation frame mapping。
7. 加入 partial camera 的 coverage/region 门禁。
8. 加入单一 main broadcast profile 及可复现的动态轨迹。
9. 复用现有 audit/postprocess/manifest 流程做兼容性验证；只在确有必要时补充相机 artifact 检查。
10. 由用户确认实现结果后，再分别提交 Dataset 子仓库和外层 UE gitlink；本设计阶段不自动 commit/push。

## Final Decision

P2-1 采用 Dual Camera System、Hybrid Strategy 以及 Canonical Camera ID + UE Mapping。

- Canonical IDs：`C1..C5`，属于 Dataset semantic layer。
- Static anchors：固定五个 full-court cameras，不随机化。
- Partial cameras：可选、随机、coverage 仅为 25/50/75，必须满足区域和非 full-field 约束。
- Dynamic cameras：只保留一个 main broadcast profile，支持 smooth follow、pan、tilt、zoom。
- Task extension：复用 Task Spec，在 `simulation.camera` 中描述分布，在 `ue.camera_mapping` 中描述 UE binding。
- Runtime contract：通过 `ResolvedTask` 传递 resolved camera distribution 和 mapping。
- Existing pipeline：继续使用现有 Sequence、Camera Cut、MRQ、annotation exporter 和 `camera.json`。
- New artifact：增加设计上的 `camera_state.jsonl`，作为 dataset artifact，不作为管理系统。
- C5 policy：设计不假设 C5 当前存在；首版实现必须提供 C5 anchor slot，并通过显式 mapping 接入未来或新增 UE Camera Actor。
- Scope boundary：不新增 Camera Manager、Simulation Framework、Database、MCP 或独立配置体系。

P2-1 Design Phase COMPLETE

等待设计确认后再进入编码阶段。
