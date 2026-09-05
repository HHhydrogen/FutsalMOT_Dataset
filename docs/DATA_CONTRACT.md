# FutsalMOT 数据契约

本文档描述当前代码读写的 JSON/JSONL、图像和标签格式。字段名称以 `src/grf_ue_bridge/schema.py`、`ue/dataset_export.py`、`ue/annotation_exporter.py`、`ue/pose_annotator.py` 和相关验证器为准。

## 目录和阶段

单文件 task 的 `dataset_root` 和 `episode_name` 决定 episode 目录：

```text
<dataset_root>/<episode_name>/
  meta.json
  frames.jsonl
  raw_observations.jsonl       # 可选
  provenance/
    export_config.json         # 旧式 exporter 可能写入
    task.json                  # task 工作流写入
    resolved-task.sanitized.json
    export-profile.json
    ue-profile.json
    actor-mapping.json
    external_sources.lock.json
  <camera_id>/
    camera.json
    annotations.jsonl
    seqinfo.ini                # export_mot=true 时由 UE 侧写入
    gt/gt.txt                  # export_mot=true 或 annotate-masks 选择 mot 时写入
    img1/*.png                 # 对齐后的 canonical RGB
    render/*.png               # MRQ 原始 RGB，可能在完成后被 zero-waste 删除
    render_mask/*.exr          # Object-ID Cryptomatte 原始输出
    mask/*.png                 # P1 从 EXR 解码的 Instance-ID Mask
    mask_config.json
    labels/det/*.txt
    labels/seg/*.txt
    labels_pose/*.txt
    debug/
  pose_capture.jsonl           # Runtime Pose 原始持久化
  pose_session.json
  coco17_3d.jsonl
  futsal_pose.yaml
  yolo_pose/labels/<camera>/*.txt
  dataset_manifest.json
  audit/soak_audit_report.json
  audit/soak_audit_report.md
```

P1 轨迹输出和 P2 数据集输出在当前 resolver 中都指向同一个 episode 目录。P1 先写 `meta.json`、`frames.jsonl` 和 provenance，UE 再在相同目录下写 camera 子目录和渲染产物。

## Task 与 resolved task

task 文件的顶层 schema 是：

```json
{
  "schema": "futsalmot_dataset_task",
  "version": 2,
  "task_id": "...",
  "episode_name": "...",
  "dataset_root": "绝对路径",
  "ue_project_root": "绝对路径",
  "export": {},
  "ue": {},
  "postprocess": {},
  "audit": {},
  "artifact_policy": {"profile": "research_minimal"}
}
```

`export` 和 `ue` 必须内联在同一个 JSON 中；`ue.actor_mapping` 默认相对 Python 仓库根目录解析为 `ue/actor_mapping.example.json`。`postprocess.formats` 的合法值是 `json`、`mot`、`yolo-det`、`yolo-seg`，`yolo_pose.enabled` 和 `debug.enabled` 默认关闭。

`task resolve` 产生的运行时 schema 是 `futsalmot_resolved_task`，版本为 `1`。它包含绝对的 `repo_root`、`dataset_root`、`ue_project_root`、`trajectory_output`、`dataset_episode_dir`、`actor_mapping`，以及归一化后的 `export_profile`、`ue_profile`、`postprocess` 和 `audit`。该文件在 `.futsalmot/runtime/<task_id>/resolved-task.json`，属于被忽略的运行时文件，不是提交契约。

## Episode 轨迹

### `meta.json`

`meta.json` 的 schema 为 `grf_ue_episode`，版本为 `1`，主要字段为：

- `episode_id`：输出目录名。
- `source`：`environment`、`scenario`、`control_mode`、`seed`、外部仓库 commit，以及可选的 `game_duration` 和双方 AI 难度。
- `randomness`：`policy`、`root_seed`、`grf_game_engine_seed`、`python_seed`、`numpy_seed`、`ue_visual_seed`。当前 policy 为 `futsalmot_seed_v1`；`ue_visual_seed` 当前只记录，不被 UE 视觉随机化消费。
- `timing`：`source_step_seconds`、`playback_fps`、`num_steps`。
- `field`：长度、宽度、中心原点和 `x_range_m`/`y_range_m`。
- `entities`：10 名球员和 `BALL` 的定义，包括 `id`、队伍、源索引、角色和 `is_goalkeeper`。
- `coordinate_transform`：导出时使用的坐标变换说明。

实体 ID 固定为 `L0` 至 `L4`、`R0` 至 `R4` 和 `BALL`。角色优先从 GRF 观测读取，缺失或异常时使用默认角色序列。

### `frames.jsonl`

每个非空行是一个 frame 对象：

```json
{
  "step": 0,
  "time_seconds": 0.0,
  "score": [0, 0],
  "ball": {
    "position_m": [0.0, 0.0, 0.11],
    "source_grf_position": [0.0, 0.0, 0.11],
    "velocity_mps": [0.0, 0.0, 0.0]
  },
  "players": [
    {
      "id": "L0",
      "position_m": [0.0, 0.0, 0.0],
      "velocity_mps": [0.0, 0.0],
      "speed_mps": 0.0,
      "movement_heading_deg": null,
      "active": true,
      "has_ball": false
    }
  ],
  "ball_owned_team": -1,
  "ball_owned_player": -1,
  "game_mode": 0
}
```

字段规则：

- `step` 从 `0` 开始，导出器正常情况下按行递增。
- `time_seconds` 单位为秒。
- 每帧恰好 10 名球员，ID 必须覆盖两队的 `L0..L4` 和 `R0..R4`，且不重复。
- 球员 `position_m` 单位为米，Z 固定为 `0`。球的位置 Z 原样保留；旧 episode 可能没有速度、持球或 game mode 字段。
- `velocity_mps` 和 `speed_mps` 单位为米/秒；`movement_heading_deg` 使用 `atan2(vy, vx)`，静止时可能为 `null`。
- `ball_owned_team` 的约定是 `-1` 无持球、`0` 左队、`1` 右队；`ball_owned_player` 是队内下标，`-1` 表示无持球。

### 坐标和时间重采样

默认场地为 `40m x 20m`，原点在中心：

- GRF X 的 `[-1, 1]` 映射到米坐标 `[-field_length/2, field_length/2]`。
- GRF Y 的实际可用半范围约为 `1/2.25`，代码将它拉伸到 `[-field_width/2, field_width/2]`。
- 球 Z 使用 `Z_FIELD_SCALE=1` 原样透传，通常已经近似为米；球员 Z 固定为 `0`。
- GRF 原生采样步长为 `0.1s`，即 10fps。

当 `target_fps` 为 `0` 或 `10` 时保持 GRF 原生帧数。当 `target_fps=30` 时，普通导出输出 `num_steps * 3` 帧，使用位置插值。`trajectory_time_scale > 1` 时，数据集时间 `k / target_fps` 对应源轨迹时间 `k / target_fps * trajectory_time_scale`，位置使用速度感知的 Hermite 重采样，离散状态使用 hold/nearest，不做数值插值。

图像标注使用 1 基帧号，而轨迹和 Sequence 使用 0 基源步：

```text
annotation frame_index = source_step + 1
Sequence/MRQ frame = round(source_step * source_step_seconds * playback_fps)
img1/000001.png      <-> annotation frame_index=1 <-> source_step=0
```

这里的 `playback_fps` 是 Sequence display rate。MRQ 的输出 `frame_rate` 不能替换这个映射参数。

## Actor、相机和几何标注

默认映射文件 `ue/actor_mapping.example.json` 为：

| entity_id | UE Actor 标签 | track_id | mask_id |
| --- | --- | ---: | ---: |
| `L0..L4` | `Player_L0..Player_L4` | `1..5` | `1..5` |
| `R0..R4` | `Player_R0..Player_R4` | `6..10` | `6..10` |
| `BALL` | `Ball_01` | `100` | `11` |

Mask 背景值为 `0`。MOT 和 Mask 的 ID 映射是固定的，不能按相机或帧改变。

`camera.json` 由 UE 侧从 `CineCameraComponent` 当前状态生成，包含：

- `image_width`、`image_height`。
- `intrinsics.width/height/fx/fy/cx/cy`，单位为像素。
- `extrinsics.world_location_m`、`world_rotation_deg` 和 `forward/right/up`。
- `focal_length_mm`、`sensor_size_mm`、水平/垂直 FOV 和单位说明。

世界到相机的计算是 `camera = R @ (world - location)`，其中 R 的行向量依次是相机 forward、right、up。图像原点在左上角，投影公式为：

```text
u = cx + fx * y_cam / x_cam
v = cy - fy * z_cam / x_cam
```

`x_cam <= 0` 的点在相机后方，视为投影失败。几何标注将 Actor 世界 AABB 的 8 个角点投影到图像并裁剪到图像边界。球可通过 `ball_radius_m` 使用指定半径。球员仅在 task 提供非空 `player_bbox` 时进入 `player_world_bounds()`：该函数的 `source` 缺省值为 `mesh`，也可显式选择 `capsule`；若 task 完全省略 `player_bbox`，代码会走 `get_world_bounds()`，其优先级是 Character Capsule、网格组件 bounds、Actor bounds fallback。

`annotations.jsonl` 每行包含：

```json
{
  "episode_id": "episode_name",
  "camera_id": "CineCam_01",
  "frame_index": 1,
  "source_step": 0,
  "time_seconds": 0.0,
  "objects": []
}
```

几何阶段的 object 至少包含 `entity_id`、`track_id`、`class`、`team`、`role`、`is_goalkeeper`、`world_position`、`raw_bbox_xyxy`、`raw_bbox_xywh`、`bbox_xyxy`、`bbox_xywh`、`in_frame`、`truncated` 和 `visibility`。`visibility` 不表示真实遮挡，几何导出默认写 `null`。

## Instance-ID Mask 和可见像素 GT

当前正式 mask 来源是 UE MRQ 的 `MoviePipelineObjectIdRenderPass`，输出 multilayer EXR 到 `render_mask/`。`ue/render_episode.py` 对 EXR 只统计与标注对齐的帧，不在 UE 中解码；P1 的 `grf-ue cryptomatte-to-mask` 使用 `openexr` 和 NumPy 完成转换。

Cryptomatte 解码契约：

- manifest 位于 EXR header 的 `cryptomatte/<hash>/manifest`。
- UE 5.8 当前实现优先读取 `RGBA` 层 R 通道的 float32 Actor ID。
- manifest 的十六进制 ID 按 float32 位模式精确匹配，不使用宽松的浮点误差比较。
- 输出 `mask/{frame_index:06d}.png` 为单通道 PNG，背景为 `0`，实体为 `1..11`。

`grf-ue annotate-masks` 读取 `mask/` 和几何 `annotations.jsonl`，覆盖写回 `annotations.jsonl` 并写入 `mask_config.json`。有 mask 时，object 的语义为：

| 条件 | `bbox_source` | 可见 bbox | 是否进入 MOT/YOLO |
| --- | --- | --- | --- |
| 实体像素数大于 0 | `instance_mask` | 由 mask 像素 min/max 得到的 pixel-tight bbox | 是，按 `include_ball` 过滤球 |
| 实体像素数为 0 | `not_visible` | `bbox_xyxy/xywh` 和 raw bbox 为 `null` | 否 |
| 没有 mask 目录或缺某帧 mask | 保持几何字段 | 保留几何 fallback | 不作为 mask-primary YOLO 对象 |

几何 bbox 不会被回填到可见 bbox；有 mask 的情况下保存在 `geometry_bbox_xyxy` 和 `geometry_bbox_xywh`。`visible_pixel_count` 是可见像素数，`visibility` 仍为 `null`。完全不可见对象的几何 bbox 只用于参考。

默认会提取分割轮廓。多连通域时尝试桥接为一个 ring，并检查栅格化面积：额外面积比例不超过 `0.10`、缺失面积比例不超过 `0.05` 且 IoU 不低于 `0.75` 才接受桥接；失败时回退到最大连通域，并记录 `segmentation_fallback` 和原因。`--no-segmentation` 会跳过 polygon 计算，但仍计算 bbox、可见像素、MOT 和 YOLO Detect。

`postprocess.formats` 控制派生产物。`annotations.jsonl` 和 `mask_config.json` 始终写入；`mot` 控制 `gt/gt.txt`，`yolo-det` 控制 `labels/det/`，`yolo-seg` 控制 `labels/seg/`。默认 `clean_stale=true` 会删除本次未选择的旧派生产物目录或 `gt/gt.txt`。

## MOT 和 YOLO

### MOTChallenge

`gt/gt.txt` 每行 9 列：

```text
frame,track_id,x,y,w,h,confidence,class_id,visibility
```

帧号从 `1` 开始；`x/y/w/h` 为图像内整数像素；confidence 固定为 `1`。球员 `class_id=1`，球 `class_id=100`。默认 `visibility=1.00`，表示当前版本没有把真实遮挡程度写入 MOT visibility；图像边界截断的另一种模式在序列化器中存在，但 mask-primary 后处理固定使用 `unoccluded`。

只有 `in_frame=true` 且 bbox 合法的对象进入 MOT；`not_visible` 对象不会进入。`include_ball` 由 UE 标注配置和 P1 后处理配置分别控制，使用时要保持两侧意图一致。

`seqinfo.ini` 使用 `img1` 作为图像目录，记录 episode 名、帧率、帧数、宽高和 `.png` 扩展名。

### YOLO Detect 和 Segment

YOLO Detect 的每行是：

```text
class_id center_x center_y width height
```

坐标归一化到 `[0,1]`，类别为 `0=player`、`1=ball`。YOLO Segment 使用类别 ID 加归一化 polygon 坐标。两种标签只从 `bbox_source="instance_mask"` 的可见对象写出；没有 mask 的 legacy 几何对象和 `not_visible` 对象不会进入训练标签。

## Runtime Pose、COCO17 和 YOLO Pose

正式 `run_task.py --mode full` 使用 5 个 C4 Recorder，每个 Recorder 负责 2 名球员：

| Recorder | 球员 |
| --- | --- |
| G0 | `L0`, `L1` |
| G1 | `L2`, `L3` |
| G2 | `L4`, `R0` |
| G3 | `R1`, `R2` |
| G4 | `R3`, `R4` |

每名球员采样 13 个骨骼：`head`、左右 `upperarm`、`lowerarm`、`hand`、`thigh`、`calf` 和 `foot`。BurnIn 的 `OnOutputFrameStarted` 每个输出帧调用 5 个 Recorder；每个 Recorder 写入带 episode 和主相机名的动态 SaveGame slot。

`pose-finalize` 读取第一个配置相机的 5 个 slot：

- `pose_capture.jsonl`：每行一个 actor/bone sample，位置为 UE cm，包含 `root`、`shot`、`game_time`、`actor_id`、`bone`、`x/y/z` 和四元数。
- `pose_session.json`：保存主相机、期望帧数、捕获帧数、完整性标志、样本数和 root 帧范围。`capture_complete` 不为 true 时禁止生成正式 COCO17。
- `coco17_3d.jsonl`：episode 级，每行一个 actor/frame，`keypoints_3d_m` 为按 COCO 顺序排列的 17 个世界坐标点，单位米。
- `<camera>/coco17_2d.jsonl`：每相机一份，每行包含 `keypoints_2d_px` 和 `visible`；投影坐标单位为像素。
- `<camera>/pose_keypoints.jsonl`：`run_task.py` 把 COCO17 3D 结果桥接为 P1 Pose 标注器的输入，`keypoints_world` 单位米。Runtime Pose 桥接不写 `occluded`，P1 主要依据 mask 判定 visibility。

COCO 17 点顺序固定为：

```text
nose, left_eye, right_eye, left_ear, right_ear,
left_shoulder, right_shoulder, left_elbow, right_elbow,
left_wrist, right_wrist, left_hip, right_hip,
left_knee, right_knee, left_ankle, right_ankle
```

资产骨架没有眼、鼻、耳骨骼，脸部五点由 `head` 骨骼局部偏移推导；12 个肢体点使用候选骨骼的世界 transform。`pose_bones.py` 是两侧共用的映射常量来源。

`grf-ue annotate-pose` 读取 Pose、相机、mask-primary annotations 和 mask，写入 `labels_pose/{frame_index:06d}.txt`。每行固定 56 个字段：

```text
class xc yc w h (x y visibility) * 17
```

Pose 类别当前只写 `0=player`；关键点坐标归一化到 `[0,1]`，visibility 为 `0=无效`、`1=遮挡`、`2=可见`。Pose bbox 直接复用同一对象的 mask-primary bbox，与 YOLO Detect 一致。只有可见球员且至少有一个有效关键点时才写行。`futsal_pose.yaml` 和 `yolo_pose/labels/<camera>/` 是可再生的训练暂存结构，RGB 不复制到 `yolo_pose/images/`，训练应引用各 camera 的 `img1/`。

## provenance、manifest 和清理

task export 会保存 task、sanitized resolved task、export profile、UE profile、actor mapping 和外部锁文件快照。sanitized resolved task 用 `${REPO_ROOT}`、`${UE_PROJECT_ROOT}`、`${DATASET_ROOT}` 占位符替换绝对路径。

顶层 `grf-ue build-manifest` 生成数据集根目录的 `dataset_manifest.json`，对明确纳入的 episode 统计产物，按 `metadata`、`final` 或 `all` profile 写 SHA-256 JSONL 校验文件，并计算 trajectory hash、canonical hash、重复 seed/轨迹组和 dataset fingerprint。`verify-manifest` 的退出码为 `0=通过`、`1=内容缺失或不匹配`、`2=manifest/schema/参数错误`。

task 级 `manifest` 使用 `src/grf_ue_bridge/workflows/artifact_cleanup.py` 的 episode manifest builder，和顶层 `build-manifest` 不是同一个入口。不要用其中一个命令的输出字段去推断另一个命令的完整性结论。
