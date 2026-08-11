# YOLO Pose（COCO 17 点）标注导出设计

> 状态：已实现（2026-08-11）。
> 范围：仅标准 COCO 17 点人体关键点 → Ultralytics YOLO Pose 标签，集成进现有
> mask-primary / dataset task pipeline。不包含 23/25 点、足球专用关键点、3D Pose
> dataset 格式、手指 / Face Mesh / SMPL 等扩展。

## 1. 目标与数据流

```
GRF 轨迹（P1）→ UE Level Sequence + MRQ 渲染（RGB + Object ID Mask）
   → UE 关键点导出：pose_keypoints.jsonl（世界 3D 17 点 + occluded 标志，每相机）
   → P1 postprocess：annotate-masks → annotate-pose
       ├── labels_pose/{frame:06d}.txt（YOLO Pose，每行 56 字段）
       ├── futsal_pose.yaml（可直接 yolo pose train）
       └── debug/pose/（overlay 可视化，可选）
   → validate-pose
```

## 2. 骨骼名确认（实测，非猜测）

球员 Actor 为 `BP_ThirdPersonCharacter`，Mesh 组件使用 **`SKM_Quinn_Simple`**
（UE5 Manny/Quinn Mannequin 骨架，`SKM_Manny_Simple` 同骨架）。已读取该
SkeletalMesh 资产确认存在以下骨骼：

`root`、`pelvis`、`spine_01..spine_05`、`neck_01`、`neck_02`、`head`、
`clavicle_l/r`、`upperarm_l/r`、`lowerarm_l/r`、`hand_l/r`、
`thigh_l/r`、`calf_l/r`、`foot_l/r`、`ball_l/r`（及手指骨骼）。

**不存在**眼 / 鼻 / 耳等脸部骨骼（`b_eye_l` 等均无）→ 脸部五点由 head 骨骼的
局部偏移推导。

## 3. COCO 17 点映射表

### 3.1 肢体点（真实骨骼 transform，子骨骼原点 = 关节）

| COCO 关键点 | 索引 | UE bone | 依据 |
|-------------|------|---------|------|
| nose / left_eye / right_eye / left_ear / right_ear | 0–4 | `head` + 局部偏移 | 骨架无脸部骨骼，见 §3.2 |
| left_shoulder | 5 | `clavicle_l` | 锁骨肩端即肩关节 |
| right_shoulder | 6 | `clavicle_r` | 同上（右） |
| left_elbow | 7 | `lowerarm_l` | 前臂骨原点 = 肘 |
| right_elbow | 8 | `lowerarm_r` | 同上（右） |
| left_wrist | 9 | `hand_l` | 手骨原点 = 腕 |
| right_wrist | 10 | `hand_r` | 同上（右） |
| left_hip | 11 | `thigh_l` | 大腿骨原点 = 髋 |
| right_hip | 12 | `thigh_r` | 同上（右） |
| left_knee | 13 | `calf_l` | 小腿骨原点 = 膝 |
| right_knee | 14 | `calf_r` | 同上（右） |
| left_ankle | 15 | `foot_l` | 足骨原点 = 踝 |
| right_ankle | 16 | `foot_r` | 同上（右） |

左右点严格对应**角色 anatomical 左右**（`_l` ↔ left、`_r` ↔ right），与图像左右无关。
映射集中在 `ue/pose_bones.py`（`LIMB_BONE_CANDIDATES` + `resolve_limb_bone_map`），
可用 `postprocess.yolo_pose.bone_overrides` 覆盖（骨骼命名不同的资产）。

### 3.2 脸部五点（head 骨骼局部偏移）

骨骼中无眼 / 鼻 / 耳，故在 head 骨骼**局部空间**定义偏移，再经 head 骨骼 world
rotation 转世界坐标（`apply_head_offsets`）。假设 head 局部轴：**+X=角色前向、
+Y=角色右向、+Z=角色上向**（UE 约定）。

默认偏移（cm，`HEAD_OFFSET_CM`，按 Mannequin 头部尺寸解剖学估算，**可配置/需验证**）：

| 关键点 | 局部偏移 [x, y, z] | 说明 |
|--------|-------------------|------|
| nose | [7.5, 0, 9.5] | 面部中央 |
| left_eye | [7.0, -3.2, 12.5] | 角色左侧 = 局部 −Y |
| right_eye | [7.0, 3.2, 12.5] | 角色右侧 = 局部 +Y |
| left_ear | [0.5, -8.0, 11.0] | 头侧 |
| right_ear | [0.5, 8.0, 11.0] | 头侧 |

> **必须用 overlay 验证**：`grf-ue pose-overlay <cam_dir> --frames 1,2,3` 检查脸部五点
> 是否落在角色头部正确位置；如需微调，改 `postprocess.yolo_pose.head_offsets_cm`。
> UE 侧可运行 `import pose_export; pose_export.dump_actor_bones('Player_L0')` 复核实际骨骼名。

## 4. 可见性判定（v = 0 / 1 / 2）

统一约定：`v=0` 无效、`v=1` 被遮挡、`v=2` 可见。**基于真实渲染/几何**，不用 bbox 近似。

### 判定顺序（`pose_annotator._keypoint_visibility`）

1. **v=0**：3D 坐标无效 / 在相机后方（`x_cam <= 0`）/ 投影非有限值 / 投影点明显位于
   图像有效区域之外。
2. **遮挡（v=1）**，满足其一：
   - **其他实例遮挡**：投影点邻域（默认 5×5，`visibility_neighborhood_radius=2`）内
     其他实例（球员 / 球）的 mask 像素数 ≥ 该球员自身像素数。该信号来自
     **Instance-ID Mask（真实渲染结果）**，是 inter-object 遮挡的 ground truth。
   - **几何遮挡（UE trace）**：UE 侧对每个关键点做射线检测（相机→关键点），任一
     blocking 命中且命中距离 < 关键点距离 − `trace_tolerance_cm`（默认 20cm）即视为被
     几何遮挡（自遮挡、球、围挡等非 mask 几何）。记录在 `pose_keypoints.jsonl` 的
     `occluded` 字段。
3. **v=2**：其余（投影点落在自身 mask 邻域或自由空间，且无遮挡）。

邻域采样避免「关节点恰好落在人体轮廓边缘附近」产生大量错误 v=1（边缘容差：own ≥ other
时不判遮挡）。关节在身体表面内部的深度差由 trace 容差吸收。

## 5. 坐标与投影约定

- **世界坐标**：UE 左手系，X 前、Y 右、Z 上。`pose_keypoints.jsonl` 存**米**
  （UE cm / 100），与 `camera.json` 的 `world_location_m` / `frames.jsonl` 一致。
- **投影**：`camera = R @ (world - location)`（R 行向量 = forward/right/up）；
  `u = cx + fx·(y_cam / x_cam)`、`v = cy − fy·(z_cam / x_cam)`；`x_cam <= 0` 视为在相机
  后方。内参/外参取自 `camera.json`（UE 导出的真实 CineCamera 标定），不手工估算。
- **图像坐标**：原点左上，x 右 y 下。归一化 `x = u / width`、`y = v / height`。
- 复用 `camera_projection.project_world_to_image`（与几何 bbox 同一套投影）。

## 6. BBox 与 track 对齐

- YOLO Pose 行的 bbox 复用 `annotations.jsonl` 的 **mask-primary bbox**（`bbox_source ==
  "instance_mask"`），与 YOLO det 完全一致（同一 `det_xyxy_to_yolo_norm` 归一化），
  **不根据关键点重新生成**。
- 只对 `bbox_source=="instance_mask"` 且 `in_frame` 的球员产生 pose 行（与 YOLO det
  导出集合一致）；球不产生关键点标签。
- 关键点数据按 `(frame_index, track_id)` 与 annotations 对齐，确保
  player / track / mask_id / bbox / 17 keypoints 始终属于同一球员。

## 7. 输出布局

```
<episode_root>/
├── <camera>/
│   ├── pose_keypoints.jsonl      # UE 导出：世界 3D 关键点 + occluded（每相机）
│   ├── annotations.jsonl         # mask-primary bbox（annotate-masks）
│   ├── labels_pose/000001.txt    # YOLO Pose 标签（新，不覆盖 labels/det、labels/seg）
│   ├── labels/det/ ... labels/seg/ ... gt/gt.txt   # 既有检测/分割/MOT（不受影响）
│   └── debug/pose/000001.png     # overlay 可视化（grf-ue pose-overlay）
├── yolo_pose/                    # 可训练暂存：images/（img1 硬链接）+ labels/（副本）
└── futsal_pose.yaml              # dataset YAML（kpt_shape [17,3] + flip_idx）
```

## 8. 已知风险 / 未验证项

- **UE API 兼容性（2026-08 实跑已修复两处）**：
  - `SkeletalMesh.get_ref_skeleton()` 在 UE 5.8 Python 不存在 → 原 `_list_actor_bones` 返回 0 骨骼、
    肢体点全空。已改为 **probe 方式**：`_resolve_bone_map_by_probe` 逐候选骨读 world transform，
    用「非全零四元数」识别真实骨骼（缺失骨骼的 `get_bone_transform` 返回零值 transform），
    不依赖列出骨骼名；`_list_actor_bones` 降级为仅诊断（用 Skeleton 资产 `bone_tree`）。
  - `unreal.ETraceTypeQuery` 在 UE 5.8 Python 不存在 → 原 `_line_trace` 崩溃。已改为**全防御式**：
    枚举/params/函数任意缺失或调用失败都返回 None，P1 退化为仅 Instance-ID Mask 判定，
    绝不中断导出。
- **姿势对齐**：关键点取自**导出时**关卡中角色的实际骨骼姿势。若角色的 AnimBlueprint
  （`ABP_Unarmed`）在 MRQ PIE 渲染期间播放动画，编辑器姿势与渲染姿势可能不同，
  导致关键点与渲染图轻微错位。验证/规避：用 `pose-overlay` 对照确认；必要时把 mesh
  动画置空（参考姿势）保证导出与渲染一致。
- **脸部偏移**：默认 head 局部偏移为解剖学估算，**必须**经 overlay 人工确认后微调。
- **遮挡 trace 语义**：trace 仅作为 mask 之外的自遮挡/非 mask 几何补充信号；若 trace 命中
  地面/围挡等场景几何距离关键点过近，可能把个别可见关键点误判 v=1——用 overlay 抽查，
  必要时调大 `trace_tolerance_cm`。
- 本机环境无法启动 Unreal Editor：UE 侧代码需在 UE 内按 README 步骤执行并核对 overlay。
