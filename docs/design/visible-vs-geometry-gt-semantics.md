# 可见像素 GT 与几何投影 GT 语义分离设计

- 日期：2026-08-03
- 状态：已实现（annotation schema 语义收紧）
- 范围：只收紧 annotations.jsonl 的 bbox_source 语义与 validator 一致性检查；不改 Instance-ID Mask 渲染、多连通域 YOLO Seg bridge、MRQ、GRF 管线。

## 1. 背景与问题

### 1.1 现状

`annotate-masks` 生成 mask-primary 标注时，把「可见像素 GT」与「几何投影 GT」混在了一起：

- mask 中有像素 → `bbox_source="instance_mask"`，bbox 由 mask min/max 计算（正确）。
- mask 中实体像素为 0 → `bbox_source="geometry"`，**且把几何 bbox 回填到 `bbox_xyxy`/`bbox_xywh`**。

即「完全不可见」的实体仍带着一个"可见"的 bbox，且 `bbox_source="geometry"` 同时被用作「无 mask 数据」与「mask 存在但不可见」两种互斥情形，语义含糊。

### 1.2 目标

严格区分两种 GT：

1. **可见像素 GT**（primary）：`instance_mask`，来自 Instance-ID Mask 可见像素。
2. **几何投影 GT**（debug/fallback）：`geometry_bbox_*`，不参与训练标签。

并明确「无 mask 数据」与「mask 中不可见」是两种不同的状态。

## 2. 方案：bbox_source 三值化

`bbox_source` 取值：

| 取值 | 含义 | in_frame | bbox_xyxy/xywh | visible_pixel_count | MOT/YOLO |
|------|------|----------|----------------|---------------------|----------|
| `"instance_mask"` | mask 可见像素派生 bbox（primary GT） | `true` | mask 可见 bbox | ≥1 | 导出 |
| `"not_visible"` | 有有效 mask 帧但实体像素为 0（遮挡/离屏） | `false` | `null` | 0 | 不导出 |
| `"geometry"` / 无字段 | legacy 几何标注（仅无 mask 数据时保留） | 依几何 | 几何投影 bbox | 无 | 仅几何 MOT |

### 2.1 关键约束

- `mask_annotator._upgrade_object` 不可见分支：`bbox_source="not_visible"`、`in_frame=false`、`bbox_xyxy/xywh = null`、`raw_bbox_* = null`、`segmentation = null`、`visible_pixel_count = 0`；几何只保留在 `geometry_bbox_*`，**绝不回填**。
- 无 `mask/` 目录或缺某帧 mask PNG → 该帧保持 UE 几何标注原样（legacy fallback），不写 `bbox_source`。
- MOT/YOLO 只导出 `bbox_source="instance_mask"` 的对象（不可见对象天然被 `in_frame=false` 排除）。
- 常量定义在 `ue/annotation_utils.py`：`BBOX_SOURCE_GEOMETRY / BBOX_SOURCE_INSTANCE_MASK / BBOX_SOURCE_NOT_VISIBLE`（UE 与 P1 共享）。

### 2.2 validator 新增一致性检查（`annotation_validator.py`）

- `bbox_source` 取值必须合法。
- `bbox_source="instance_mask"` → `in_frame=true` 且 `visible_pixel_count>0`。
- `visible_pixel_count==0` → 可见 bbox/raw/segmentation 必须为 null，且 `bbox_source` 应为 `not_visible`（旧格式 `geometry`+回填会报错，需重跑 `annotate-masks`）。
- `bbox_source="not_visible"` → `in_frame=false`、`visible_pixel_count==0`。
- `in_frame=false` → `bbox_xyxy/xywh` 必须为 null。
- `geometry_bbox_*` 为独立字段：存在时须为合法 4 元素 bbox。
- MOT 交叉校验：gt.txt 行的对象不得为不可见（`not_visible` / `visible_pixel_count==0`）。
- YOLO 计数校验：每帧行数 ∈ [instance_mask 球员数, 球员数+球数]。
- mask 校验：`not_visible` 实体在 mask 里必须真的无像素（反向一致性）。

## 3. 行为变更（破坏性提示）

旧版 `annotate-masks` 输出的「不可见但回填几何 bbox、`bbox_source="geometry"`」标注将不再通过 validator——需对存量数据重跑一次 `annotate-masks` 以升级为新语义。MOT/YOLO 输出内容不变（不可见对象本来就不导出）。

## 4. 非目标

- 不改 Instance-ID Mask 渲染 / Cryptomatte。
- 不改多连通域 YOLO Seg bridge 与面积 gate。
- 不改 MRQ、GRF pipeline。
- 不做 COCO、FPS 优化、motion blur。
