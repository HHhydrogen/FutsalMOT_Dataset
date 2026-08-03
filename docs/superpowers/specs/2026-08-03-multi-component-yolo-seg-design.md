# 多连通域 Instance Mask → YOLO Segmentation 转换设计

- 日期：2026-08-03
- 状态：已评审（用户确认面积检查口径）
- 范围：只解决一个实例的 visible mask 含多个 disconnected components 时，YOLO Segment 转换的正确性问题；不做其他功能。

## 1. 背景与问题

### 1.1 现状

- `ue/instance_mask.py` 的 `mask_to_polygons()` 已按 8 连通域为每个分量提取外轮廓并做轻量 RDP 简化，返回**每分量一个多边形**（像素坐标）。
- `polygon_to_yolo_flat()` 只是把所有分量的点**简单串联**成一个 flat 列表。

### 1.2 缺陷

当同一实例（如被另一球员遮挡后分开的头部与躯干）的 visible mask 含多个 disconnected components 时，简单串联会把各分量的点直接首尾相接：

- 在背景上拉出**跨区域连线**；
- 产生**自交/非简单多边形**；
- 下游栅格化时填出**错误的大块背景分割区**。

### 1.3 格式限制

YOLO Segment 每实例每行只能是一个多边形（`class x1 y1 x2 y2 ...`）。**一个实例无法无损表达多个不相交区域**，因此任何转换都是**派生近似**，必须以不损坏 raw Instance-ID Mask 为前提并记录。

## 2. 目标 / 非目标

### 目标

1. 明确支持一个实例包含多个 disconnected visible regions。
2. 采用明确、可测试的 multi-component merge 策略，YOLO Seg 不产生错误跨区域连接。
3. raw Instance-ID Mask 保持 canonical GT，定义不变。
4. bbox 严格等于 mask min/max（不受 YOLO 多边形近似影响）。
5. 单连通域行为完全不变。
6. contour simplification 仅轻量（默认 RDP tolerance=1.0px、max_points=64 不变），不改变人体边界。

### 非目标

- 不修改 bbox、MRQ、GRF、RGB pipeline。
- 不改 `annotation_validator`（YOLO 坐标 ∈[0,1]/行格式校验已覆盖；跨区域自交由本设计在生成侧保证）。
- 不新增依赖（无 cv2/shapely）。

## 3. 方案：最近点桥接合并 + 面积膨胀检查 + 最大连通域回退

用户已选定此策略。流水线：

```
多连通域 mask
→ 逐分量轮廓 + 轻量 RDP（mask_to_polygons，保持现状）
→ 最近边界点桥接成单 ring（weak-simple）
→ 合法性检查（无 proper crossing）
→ 面积膨胀检查（rasterize 回 mask，ROI 内，extra / refined_missing / IoU）
→ 通过 → 输出 YOLO Seg（单多边形，记录为派生近似）
→ 任一失败 → 回退只保留最大连通域 → 记录 fallback_reason
```

### 3.1 桥接合并（几何层，纯函数）

新增 `merge_to_single_ring(polygons, areas=None) -> (ring, meta)`：

1. `len(polygons) == 1` → 原样返回（`merged=False`）。
2. 按各多边形 bbox 中心 (x, y) 排序（确定性）。
3. 逐次桥接：`ring = _bridge_splice(ring, next)`：
   - 取两轮廓**最近边界点对** (a ∈ ring, b ∈ next)（欧氏距离；平局取最小索引，确定性）。
   - 各自沿原方向（顺时针）走满一圈后拼接，连接处为**往返零宽桥** `(a→b)` 与闭合边 `(b→a)`（同段反向、共线重叠）。结果为一**weak-simple 单 ring**：内部 = 各分量内部 ∪ 零宽连接，不包络背景。
4. **合法性检查** `_ring_has_proper_crossing(ring)`：任意非相邻边是否存在**严格交叉**（两条开线段在各自内部点处相交；共线重叠/端点相接不算——零宽桥因此不被误判）。有交叉 → 回退。
5. 返回 `(ring, meta)`；`meta = {"n_components", "merged", "fallback", "crossing_detected"}`。`fallback` 为 `None` 或 `"largest_component"`（此时 ring = `areas` 最大者的多边形，`areas` 缺省用 shoelace 面积）。

### 3.2 面积膨胀检查（语义层，运行时）

用户要求直接 rasterize polygon 与原始 mask 对比（离散像素语义，shoelace 连续面积不采用）。

- **栅格化器**：新增 `polygon_to_mask(points, width, height) -> bool ndarray`，纯 numpy **even-odd 扫描线**栅格化。像素 `(px,py)` 中心在多边形内即填充。
  - 必要性：PIL `ImageDraw.polygon` 实测对 contour ring 严重欠填充（3×3 方块轮廓只填 1px），不可用；项目无 cv2。自研保证确定性、可测。
  - 轮廓多边形天然欠填一圈边界像素（3×3 块→4/9，20×20 块→≈361/400）。这是**单连通域已存在**的固有特性，不是 merge 引入。
- **ROI**：只在实例 bbox（= ring bbox）局部栅格化，不创建整张 1920×1080 mask。
- **指标**（`raster` = 栅格结果，`truth` = ROI 内原始 mask 切片）：
  - `extra_ratio = |raster ∧ ¬truth| / |truth|` — polygon 把背景填成前景（桥仅为 O(gap) 薄线，真实 mask 上 ≈1–3%）。
  - `refined_missing_ratio = |truth ∧ ¬dilate8(raster, radius=1)| / |truth|` — **8 邻域精化 missing**：只排除天然欠填的 1px 边界环，抓**被丢弃的完整碎片**；容差**仅 1 pixel**，不扩大到多像素。
  - `iou = |inter| / |union|`。
  - `raw_missing_ratio = |truth ∧ ¬raster| / |truth|` — 仅诊断，不参与 fallback。
- **Gate**（任一失败或栅格化异常 → 回退最大连通域，记录 `fallback_reason`）：
  - `extra_ratio ≤ 0.10`
  - `refined_missing_ratio ≤ 0.05`
  - `iou ≥ 0.75`
- **单连通域完全绕过此检查**（`n_components == 1` 时行为与现状逐字节一致）。

### 3.3 阈值默认值

- `AREA_TOL_EXTRA_RATIO = 0.10`
- `AREA_TOL_MISSING_RATIO = 0.05`
- `AREA_TOL_IOU = 0.75`

作为 `mask_annotator` 模块常量，**不进 CLI**（保持命令面不变）；随 `mask_config.json` 的 note 记录。

## 4. 数据流与记录（`mask_annotator.py`）

`_upgrade_object` 流程（mask 可见分支）：

1. `polys, areas = mask_to_polygons_with_areas(binary, tolerance, max_points)`（新增，返回每分量像素数；`mask_to_polygons` 签名与行为保持不变）。
2. `ring, meta = merge_to_single_ring(polys, areas)`（几何层：桥接 + 交叉检查 + 最大连通域回退）。
3. 若 `n_components > 1`：ROI 内 `polygon_to_mask` + 指标；任一 gate 失败 → 用 `areas` 最大者 polygon 覆盖 `ring`，`fallback_reason` 记录具体指标（如 `"extra_ratio=0.42>0.10"`）。
4. `seg_flat = ring_to_yolo_flat(ring, width, height)`。
5. object 新增字段：
   - `segmentation_components`（int，可见碎片数）
   - `segmentation_merged`（bool，是否桥接合并）
   - `segmentation_fallback`（None / `"largest_component"`）
   - `segmentation_fallback_reason`（None / 具体指标字符串；记录派生近似与丢弃信息）

`_write_yolo_labels` 不变（仍读 `obj["segmentation"]` 写 `labels/seg/`）。

## 5. API 变更汇总（`ue/instance_mask.py`）

| 函数 | 类型 | 说明 |
|------|------|------|
| `merge_to_single_ring(polygons, areas=None)` | 新增 | 多连通域→单 ring；纯函数，几何层 |
| `polygon_to_mask(points, width, height)` | 新增 | even-odd 扫描线栅格化（测试与运行时面积检查共用） |
| `dilate8(mask)` | 新增 | 8 邻域膨胀（radius=1） |
| `raster_quality_metrics(raster, truth)` | 新增 | 计算 extra/refined_missing/iou/raw_missing |
| `mask_to_polygons_with_areas(...)` | 新增 | 返回 (polys, areas) |
| `ring_to_yolo_flat(ring, width, height, precision)` | 新增 | ring → YOLO 归一化 flat |
| `polygon_to_yolo_flat(...)` | 行为修改 | 内部先 merge 再 flatten；单连通域输出不变 |
| `mask_to_polygons(...)` | 不变 | 保持签名与行为 |

## 6. 测试计划（TDD）

### 6.1 `tests/test_instance_mask.py`

- **单连通域**：`polygon_to_yolo_flat` 输出结构不变、坐标 ∈[0,1]。
- **两 disconnected components**：合并为单个 ring、无 proper crossing、覆盖两分量、坐标 ∈[0,1]。
- **多个 disconnected components（3）**：单 ring、无 proper crossing。
- **遮挡人体**：头+躯干被遮挡带分开 → 2 分量 → 单 ring 覆盖两碎片。
- **小目标 BALL**：小分量 ring、归一化 ∈[0,1]。
- **rasterize 回 mask**（用户要求）：`polygon_to_mask(merged_ring)` 覆盖 `Σ polygon_to_mask(component)`（两分量皆被覆盖，边界环约定两侧一致），且其面积不显著超过原始 mask 像素数。
- **远距分量 → 面积检查失败 → 回退最大连通域**（`merge_to_single_ring` 或指标层）。
- **交叉检测**：`_ring_has_proper_crossing` 对已知自交 ring（naive 串联）为真、对桥接 ring 为假。
- **`polygon_to_mask` 正确性**：矩形/三角形已知面积。
- 更新现有 `test_multi_polygon_merged`（原断言串联 12 点即错误行为，需改为断言合并后的正确结构）。

### 6.2 `tests/test_mask_annotator.py`

- 合成 camera 目录：L0 被 L1 遮挡带**一分为二** → `segmentation_components=2`、`segmentation_merged=True`、YOLO seg 单行有效、raster 面积正常、validator 仍通过。
- 单连通域路径：字段 `segmentation_merged=False`、无面积检查介入。
- 回退路径：构造远距/病态情形 → `segmentation_fallback="largest_component"` 且 `fallback_reason` 记录。

## 7. 验收标准

- raw mask 不变。
- bbox 仍严格等于 mask min/max。
- YOLO Seg 不产生错误跨区域连接（桥接 ring 无 proper crossing；raster 面积 gate 通过）。
- 所有现有测试 + 新测试通过。
- 完成后提交一个独立 git commit（commit message 用简体中文）。

## 8. 文档同步

- `README.md`「CV Dataset Annotation Export」：
  - `segmentation` 字段说明改为：多连通域时是**派生近似**（YOLO 单多边形限制），最近点桥接合并为单 ring，raw mask 为 canonical GT。
  - 新增字段 `segmentation_components` / `segmentation_merged` / `segmentation_fallback` 说明。
- `CLAUDE.md` 架构段 `instance_mask.py` / `mask_annotator.py` 描述同步（一句级别）。
