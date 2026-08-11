# configs/ — 数据集任务配置（单 config）参数参考

`configs/` 下每个 `.json` 文件描述**一次完整的数据集（单集）输出任务**：导出参数、
UE 相机/渲染参数、机器路径全部内联在一个文件里，直接入库。解析器（resolver）
据此生成 resolved task（`.futsalmot/runtime/`，git 忽略）供 P1 与 UE 共用。

- 完整模板：`configs/example.json`
- 现有配置：`configs/pose_smoke_3frames_1cam.json`（冒烟/demo：3 步 × 1 相机，yolo_pose 已启用，含 anti-teleport 参数）

## 快速上手（新建一个数据集任务）

1. 复制模板：`Copy-Item configs/example.json configs/<task_id>.json`
2. 编辑 `configs/<task_id>.json`：替换 `<DATASET_ROOT>` / `<UE_PROJECT_ROOT>` 为真实机器路径，
   修改 `task_id` / `episode_name`，再按需调导出、UE、后处理参数。
3. 校验并解析：

```powershell
uv run grf-ue task validate configs/<task_id>.json
uv run grf-ue task resolve configs/<task_id>.json
```

4. 后续：`grf-ue task export` → `grf-ue task ue-command`（UE Console 运行）→
   `grf-ue task postprocess` / `audit`（见根 README）。

## 顶层字段

| 字段 | 类型 | 必填 | 默认 | 说明 |
|------|------|------|------|------|
| `schema` | string | 是 | — | 固定 `"futsalmot_dataset_task"` |
| `version` | int | 是 | — | 固定 `2` |
| `task_id` | string | 是 | — | 任务唯一 ID，`^[A-Za-z0-9_-]+$`；决定 resolved task 运行时目录名 |
| `episode_name` | string | 是 | — | episode 目录名，`^[A-Za-z0-9_]+$`（无路径分隔符）；产出目录即 `<dataset_root>/<episode_name>/` |
| `dataset_root` | string | 是 | — | 数据集输出根目录（**绝对路径**，直接入库）；所有轨迹与相机数据落其下 |
| `ue_project_root` | string | 是 | — | Unreal 项目根目录（含 `.uproject`，**绝对路径**，直接入库） |
| `export` | object | 是 | — | GRF 导出参数（见下） |
| `ue` | object | 是 | — | UE 相机/Sequence/渲染参数（见下） |
| `postprocess` | object | 否 | 默认值 | 后处理参数（见下） |
| `audit` | object | 否 | 默认值 | 审计预期（见下） |

> `repo_root` 无需配置：自动按 `pyproject.toml` 向上探测。

## `export` 块（GRF 导出）

| 字段 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `scenario` | string | —（必填） | GRF 场景名，如 `"5_vs_5"` |
| `seed` | int | `42` | 根随机种子。同 seed 独立进程导出 `frames.jsonl` 完全一致（可复现） |
| `num_steps` | int | `300` | GRF 仿真步数（GRF 原生 10 FPS） |
| `target_fps` | int | `0` | 目标导出帧率：`0`/`10` = 保持原生 10fps（每 GRF 步一条标注）；`30` = 插值到 30fps 导出。须为 10 的倍数 |
| `playback_fps` | int | `30` | UE 目标回放帧率（影响 Sequence 时间轴与 MRQ 渲染帧率） |
| `field_length_m` | float | `40.0` | UE 场地长度（米），坐标转换用 |
| `field_width_m` | float | `20.0` | UE 场地宽度（米），坐标转换用 |
| `render` | bool | `false` | 是否渲染 GRF 游戏画面（通常关，画面由 UE 渲染） |
| `write_video` | bool | `false` | 是否录制视频 |
| `dump_full_raw_observation` | bool | `false` | 是否额外导出完整原始观测（调试用） |
| `number_of_left_players_agent_controls` | int | `0` | 由 agent 控制的左队球员数（`0` = 全部内置 AI） |
| `number_of_right_players_agent_controls` | int | `0` | 由 agent 控制的右队球员数（`0` = 全部内置 AI） |
| `game_duration` | int/null | `null` | 单个回合的引擎帧数（场景默认 `3000`）。**设大**避免采集步数耗尽回合→`env.reset()` 瞬移；`null` = 用场景默认 |
| `left_team_difficulty` | float/null | `null` | 左队 AI 难度（0~1，场景默认 `0.05`）。调高→球更少出界→减少 set-piece 重摆阵型瞬移；`null` = 用场景默认 |
| `right_team_difficulty` | float/null | `null` | 右队 AI 难度（0~1，场景默认 `0.05`）；`null` = 用场景默认 |

**导出帧数 = `num_steps × max(1, target_fps/10)`**，需与 `audit.expected_frames_per_camera` 一致。

### 避免球员瞬移（回合长度 + AI 难度）

数据中「球员瞬移」的两个来源，需要对应参数才能根治：

1. **回合耗尽重置**：GRF 单个回合按 `game_duration` 帧计时（`steps_left` 每决策步 -1）。
   当采集步数 `num_steps` 接近 `game_duration` 时，回合结束触发 `env.reset()`，球员回到开球位。
   → 调大 `game_duration`（如 `10000`），让 `num_steps` 远小于回合长度。
2. **出界/死球重摆阵型**：AI 难度过低时球频繁出界，引擎把球员重摆到 set-piece 阵型（game_mode
   切换处可见跳位）。→ 调高 `left_team_difficulty` / `right_team_difficulty`（如 `0.3`~`0.6`），
   提高 AI 控球能力，减少死球与阵型跳变。

> 两参数都在 `export` 块，写入单 config 入库；`null` = 沿用场景默认。

## `ue` 块（UE 相机 / Sequence / 渲染）

| 字段 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `actor_mapping` | string | `"ue/actor_mapping.example.json"` | 实体 ID → UE Actor 标签映射 JSON，**相对仓库根** |
| `sequence_package_path` | string | `"/Game/FutsalMOT/Sequences"` | 生成的 Level Sequence 资产包路径 |
| `sequences` | array | `[]` | Sequence/相机列表：每项 `{ "name": "<Sequence 名>", "camera_actor": "<相机 Actor 标签>" }` |
| `replace_existing` | bool | `true` | 覆盖已存在的 Sequence 资产 |
| `ball_rolling` | object | `{}` | 球滚动旋转（见下） |
| `annotation_export` | object | `{}` | 标注/渲染配置（见下） |

### `ball_rolling`（球滚动旋转）

| 字段 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `enabled` | bool | `true` | 是否按帧累加球的自转 |
| `radius_m` | float | `0.11` | 球半径（米） |
| `minimum_move_distance_m` | float | `0.0001` | 位移小于该值不更新旋转（避免抖动） |
| `roll_sign` | float | `1.0` | 滚反了设 `-1.0` |

### `annotation_export`（CV 标注 + 渲染）

| 字段 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `enabled` | bool | `true` | 是否导出标注/渲染 |
| `image_width` / `image_height` | int | `1920`/`1080` | 输出分辨率（像素） |
| `cameras` | array | `[]` | 相机 Actor 标签列表（**与 `sequences` 一致**）；数量须等于 `audit.expected_cameras` |
| `frame_start` / `frame_end` | int/null | `0`/`null` | 标注/渲染帧范围（`null` = 全部） |
| `export_internal_jsonl` | bool | `true` | 导出内部 JSONL 标注 |
| `export_mot` | bool | `true` | 导出 MOT 格式 |
| `include_ball` | bool | `false` | 是否标注球（几何侧开关；`postprocess.include_ball` 也需配合） |
| `mot_visibility_mode` | string | `"unoccluded"` | MOT 可见性判定模式 |
| `ball_scale` | float/null | `null` | 球 bbox 缩放（`null` = 用 `ball_radius_m`） |
| `ball_radius_m` | float | `0.11` | 球半径（米），用于 bbox 尺寸 |
| `player_bbox` | object | — | 球员 bbox 数据源（见下） |
| `render_rgb` | object | — | RGB 渲染配置（见下） |
| `instance_mask` | object | — | Instance-ID Mask 渲染配置（见下） |

#### `player_bbox`（球员 bbox 数据源）

| 字段 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `source` | string | `"capsule"` | `mesh`（SkeletalMesh 参考姿势边界，贴合）或 `capsule`（胶囊组件缩放） |
| `width_scale` | float | `0.7` | `capsule` 模式宽度缩放 |
| `height_scale` | float | `1.0` | `capsule` 模式高度缩放 |

#### `render_rgb`（MRQ RGB 渲染）

| 字段 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `enabled` | bool | `true` | 是否渲染 RGB |
| `preset` | string | `"cv_gt"` | 渲染 preset 名（`cv_gt` = CV GT 确定性设置） |
| `cv_gt` | object | — | preset 细节：`motion_blur` / `depth_of_field` / `chromatic_aberration` / `lens_distortion`（关），`anti_aliasing`（如 `"taa"`），`temporal_sampling` |
| `output_resolution_x` / `output_resolution_y` | int | `1920`/`1080` | 渲染分辨率 |
| `frame_rate` | int | `30` | MRQ 渲染帧率 |
| `file_name_format` | string | `"{frame_number}"` | 输出文件名模板 |
| `zero_pad_frame_numbers` | int | `6` | 帧号补零位数（如 `000001`） |

#### `instance_mask`（Instance-ID Mask → Cryptomatte）

| 字段 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `enabled` | bool | `true` | 是否渲染 Instance-ID Mask（Object ID Pass，multilayer EXR Cryptomatte） |
| `mask_source` | string | `"object_id_pass"` | Mask 来源（当前仅 `object_id_pass`） |
| `mask_channel` | string | `"r"` | Cryptomatte 实体 ID 所在通道 |
| `id_scale` / `id_offset` | float | `1.0`/`0.0` | ID → mask_id 线性映射参数 |
| `background_value` | int | `0` | 背景像素值 |
| `polygon_tolerance_px` | float | `1.0` | 多边形简化容差（像素） |
| `max_polygon_points` | int | `64` | 单多边形最大点数 |
| `post_process_material` | null | `null` | stencil→颜色材质（本 UE 5.8 实测不可用，保持 `null`） |

## `postprocess` 块（后处理：cryptomatte → annotate → validate）

| 字段 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `include_ball` | bool | `true` | 是否把球纳入标注（mask-primary bbox/分割/MOT/YOLO） |
| `workers` | int | `4` | 并行 worker 数（1–32） |
| `chunk_size` | int | `50` | 帧分块大小（`0` = 自动） |
| `png_compress_level` | int | `1` | mask PNG 压缩级别（0–9） |
| `formats` | array | 全选 | 输出格式：`json` / `mot` / `yolo-det` / `yolo-seg` |
| `clean_stale` | bool | `true` | 重跑前清理陈旧产物 |
| `validation_level` | string | `"full"` | 标注校验级别：`full` / `quick` |
| `yolo_pose` | object | 默认关闭 | YOLO Pose 人体关键点导出（见下） |

### `yolo_pose`（YOLO Pose COCO 17 点标注）

| 字段 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `enabled` | bool | `false` | 是否导出 YOLO Pose。开启后 **UE 侧**（`ue/run_task.py` 读同一 resolved task）导出每相机 `pose_keypoints.jsonl`（世界 3D 关键点 + occluded），**P1** `task postprocess` 自动生成 `labels_pose/` + `futsal_pose.yaml` 并校验 |
| `visibility_neighborhood_radius` | int | `2` | Instance-ID Mask 邻域判定半径（像素），用于关键点遮挡判定（避免轮廓边缘误判） |
| `write_dataset_yaml` | bool | `true` | 是否在 episode 根生成 `yolo_pose/` 可训练暂存目录（images 硬链接 + labels 副本）与 `futsal_pose.yaml` |
| `occlusion_trace` | bool | `true` | UE 侧是否对每个关键点做遮挡 trace（自遮挡 / 球 / 围挡等非 mask 几何） |
| `trace_tolerance_cm` | float | `20.0` | 遮挡 trace 容差（cm）：命中距离 < 关键点距离 − 容差即判遮挡 |
| `bone_overrides` | object | `{}` | UE bone 名覆盖：`{COCO 关键点名: UE bone 名}`。默认映射见 `ue/pose_bones.py`（SKM_Quinn_Simple 已实测确认） |
| `head_offsets_cm` | object | `{}` | 脸部五点相对 head 骨骼的局部偏移覆盖：`{脸部 COCO 名: [x, y, z] cm}`（骨骼中无眼/鼻/耳时使用） |

> 启用后执行链路：`task export` → `task ue-command`（UE 运行，含 pose 关键点导出）→
> `task postprocess`（annotate-masks → annotate-pose → validate-annotations → validate-pose）。

## `audit` 块（完整性审计预期）

| 字段 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `expected_cameras` | int | `4` | 期望相机数（须等于 `ue.annotation_export.cameras` 数量） |
| `expected_frames_per_camera` | int | `300` | 每相机期望标注帧数（须等于「导出帧数」） |

## 一致性校验（`grf-ue task validate`）

- `ue.annotation_export.cameras` 数量 == `audit.expected_cameras`
- 导出帧数（`num_steps × max(1, target_fps/10)`）== `audit.expected_frames_per_camera`
- `dataset_root` / `ue_project_root` 必填（绝对路径）
- 产出目录 `<dataset_root>/<episode_name>/` 由 `episode_name` 决定（UE 端按 episode_id 定位）
