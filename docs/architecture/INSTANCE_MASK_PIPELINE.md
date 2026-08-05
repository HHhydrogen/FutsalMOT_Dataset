# Instance-ID Mask 渲染管线

> 状态：已采用（2026-08-03 起）。
> 最终方案：**UE MRQ Object ID Pass → multilayer EXR / Cryptomatte → entity instance mask → bbox / segmentation**。
> 本页合并原 `MASK_RENDERING_STATUS.md` 的最终方案与历史排查结论。

## 1. 目标

渲染 RGB 的同时，为每个 Camera / 每帧输出一张 **Instance-ID Mask**：

- 每个实体在 mask 图像中为稳定唯一编号（`L0..L4→1..5`、`R0..R4→6..10`、`BALL→11`，背景 0）。
- 由 mask 可见像素计算 pixel-tight bbox、可见像素数、模态分割多边形，统一生成 MOT / YOLO detect / YOLO segment。

## 2. 最终方案（正式流程）

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
- `render_summary.json` 对 Object ID EXR 的 mask 状态：统计对齐帧数并记 `mask_source="object_id_exr"`（mask/*.png 由 P1 生成），不恒报 partial。

## 3. 完整标注链路（正式流程）

1. UE：`uv run grf-ue task ue-command <task>` → `ue/run_task.py --resolved-task ...`（建 Sequence + 几何标注 + MRQ 渲染 RGB 与 Object ID EXR）。
2. P1：`grf-ue task postprocess <task>` 内 cryptomatte 阶段 → `mask/*.png`。
3. P1：annotate 阶段 → mask-primary bbox / 分割 / MOT / YOLO。
4. P1：validate 阶段 → 最终验收（含 dataset regression 一致性检查）。

## 4. 历史：为何放弃 CustomStencil（备查，非当前阻塞）

> 以下为当时排查过程的历史记录。**custom stencil + post-process 材质在 UE 5.8/MRQ 下不可用（mask 全 0），已放弃**，改用第 2 节方案。

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

已确认事实（仅备查）：材质读 stencil 正确、输出路径正确、`r.CustomDepth` 定义（`3 = Enabled With Stencil`）、actor stencil 值设置正确、关卡已持久化。
