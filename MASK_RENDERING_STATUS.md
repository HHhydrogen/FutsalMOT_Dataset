# Instance-ID Mask 渲染问题：状态总结

> 状态：**✅ RESOLVED（2026-08-03）**
> 最终方案：**UE MRQ Object ID Pass → multilayer EXR / Cryptomatte → entity instance mask → bbox / segmentation**
> 说明：本文件最初用于跟踪「Instance-ID Mask 全黑」这一阻塞问题；该问题已解决。CustomStencil 排查过程作为**历史记录**保留（第 4、5 节），不再代表当前阻塞。当前正式流程见第 3 节。

---

## 1. 目标

在渲染 RGB 的同时，为每个 Camera / 每帧输出一张 **Instance-ID Mask**：

- 每个实体在 mask 图像中显示为稳定唯一的编号（`L0..L4→1..5`、`R0..R4→6..10`、`BALL→11`，背景 0）。
- 由 mask 可见像素计算 pixel-tight bbox、可见像素数、模态分割多边形，统一生成 MOT / YOLO detect / YOLO segment。

整套 P1 侧逻辑已完成并测试通过，渲染链路已由 Object ID Pass + Cryptomatte EXR 打通。

## 2. 最终方案（已跑通，当前正式流程）

```
UE MRQ Object ID Pass（MoviePipelineObjectIdRenderPass, id_type=ACTOR）
→ multilayer EXR（Cryptomatte 编码）
→ P1 `grf-ue cryptomatte-to-mask`：EXR manifest + RGBA.R float32 → mask/{frame}.png（像素 = mask_id 1..11，背景 0）
→ `grf-ue annotate-masks`：可见像素 → pixel-tight bbox / visible_pixel_count / 模态分割多边形
→ `grf-ue validate-annotations`：最终验收
```

关键实现（UE 5.8 实测）：

- mask job = `MoviePipelineObjectIdRenderPass`（`id_type=ACTOR`）+ **multilayer EXR** 输出。
- manifest 在 EXR header 的 `cryptomatte/<hash>/manifest`（`{actor_label: "hex_id"}`）；实体 ID 是 float32，存于 `RGBA` 层 R 通道，`hex_id` = 该 float32 位模式的**大端**十六进制。
- `cryptomatte.py` 逐实体按 `R 通道 == float32(hex_id)` 精确匹配生成 mask（背景 0）。
- 帧对齐与 RGB 一致：`frame_index = step + 1 ↔ 渲染帧 round(step × step_seconds × fps)`。

端到端验证（episode_smoke，3 帧，Camera_01）已通过：mask 灰度 `[0,1,...,11]`（11 实体全部命中）、`annotate-masks` 3/3 帧 bbox_source=instance_mask、`validate-annotations` PASSED。

## 3. 完整标注链路（当前正式流程）

1. UE：`import_grf_episode.py --mode full`（建 Sequence + 几何标注 + MRQ 渲染 RGB 与 Object ID EXR）。
2. P1：`grf-ue cryptomatte-to-mask <dataset_dir>` → `mask/*.png`。
3. P1：`grf-ue annotate-masks <dataset_dir> [--include-ball]` → mask-primary bbox / 分割 / MOT / YOLO。
4. P1：`grf-ue validate-annotations <dataset_dir>` → 最终验收（含 dataset regression 一致性检查）。

## 4. 历史：CustomStencil 黑屏排查（已解决，非当前阻塞）

> ⚠️ 以下为当时排查过程的历史记录。**custom stencil + post-process 材质在 UE 5.8/MRQ 下不可用（mask 全 0），已放弃**，改用第 2 节方案。保留此段仅作备查。

当时尝试的方案（全部失败）：

- 材质读 `SceneTexture(PPI_CUSTOM_STENCIL).R / 255 → Emissive`，结构经探针确认正确。
- `r.CustomDepth=3`（Enable with Stencil）在 job 级 `MoviePipelineConsoleVariableSetting` 设置。
- 编辑器为 31 个组件设 `render_custom_depth=True` + `custom_depth_stencil_value=mask_id`，读回 31/31 通过，且 stencil 值确实带入 PIE。
- 关卡已保存持久化。

核心矛盾：

```
PIE 里有 stencil 值(1~11) + custom depth 开启(r.CustomDepth=3) + 材质读 PPI_CUSTOM_STENCIL
                              ↓
               渲染出的 mask 仍然是全 0（stencil 缓冲为空）
```

根因结论：UE 5.8 中 MRQ 的 custom depth/stencil 路线在本项目不可用——`additional_post_process_materials` 输出的 mask 全 0；`MoviePipelinePostProcessPass` 在 5.8 是结构体（非独立 render pass），无法作为独立 pass 使用。

## 5. 历史：已确认的事实（CustomStencil 阶段，仅备查）

1. 材质读 stencil 正确：`M_StencilToID` 的 SceneTexture 节点 `scene_texture_id = PPI_CUSTOM_STENCIL`。
2. 材质输出路径正确：`CustomStencil.R / 255 → Emissive`。
3. `r.CustomDepth` 官方定义：`2 = Enabled On Demand（stencil 禁用）`、`3 = Enabled With Stencil`；job 级已用 `3`。
4. 编辑器 actor stencil 值已设置并读回验证。
5. stencil 值确实带入 PIE 世界。
6. 关卡已保存持久化。

## 6. Git 记录

- `08fc2a9` feat: 新增 Instance-ID Mask 像素级 GT 标注管线
- `d0e2046` docs: 记录 Instance-ID Mask 全黑问题排查状态 + PIE stencil 诊断
- `643d297` feat: 解决 Instance-ID Mask 全 0——改用 Object ID Pass + Cryptomatte EXR
- `97e70c3` docs: 文档同步为 Object ID Pass + Cryptomatte 实际方案 + 3 步 smoke episode 配置
- `e6ccad1` refactor: 清理调试残留（本文件曾临时删除，现按最终方案重建为 RESOLVED）
