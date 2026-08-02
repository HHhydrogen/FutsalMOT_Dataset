# Instance-ID Mask 渲染问题：状态总结

> 更新：2026-08-03
> 目的：记录"Instance-ID Mask 全黑"这一阻塞问题的完整排查过程、已确认事实与剩余假设，供后续继续开发或换方案时参考。

---

## 1. 目标

在渲染 RGB 的同时，为每个 Camera / 每帧输出一张 **Instance-ID Mask**：

- 每个实体在 mask 图像中显示为稳定唯一的编号（`L0..L4→1..5`、`R0..R4→6..10`、`BALL→11`，背景 0）。
- 由 mask 可见像素计算 pixel-tight bbox、可见像素数、模态分割多边形，统一生成 MOT / YOLO detect / YOLO segment。

整套 **P1 侧逻辑已完成并测试通过**（134 个测试全绿），唯一阻塞在 **UE/MRQ 渲染出 mask 图这一步**。

## 2. 已完成的架构（已提交，工作正常）

| 部分 | 状态 |
|------|------|
| `ue/instance_mask.py`：mask 解码/量化、pixel-tight bbox、连通域、轮廓、RDP 简化、YOLO 归一化 | ✅ 纯函数，测试通过 |
| `src/grf_ue_bridge/mask_annotator.py` + `grf-ue annotate-masks`：mask → annotations.jsonl / MOT / YOLO det / YOLO seg | ✅ 幂等，端到端可用 |
| `annotation_validator.py`：RGB/mask 帧一致、mask ID 合法、bbox==mask、YOLO 坐标范围 | ✅ |
| `ue/render_episode.py`：MRQ **异步**渲染 RGB + mask job（delegate + watchdog，不阻塞主线程） | ✅ RGB 正常；mask job 能跑、能出 10 帧文件 |
| 材质 `M_StencilToID`：post-process，`SceneTexture(PPI_CUSTOM_STENCIL).R / 255 → Emissive` | ✅ 结构已确认正确 |

## 3. 当前阻塞问题

**渲染出的 mask 全黑（R/G/B/A 全部为 0），没有实体编号。**

即使所有可检查的环节都确认无误，mask 仍然是全 0。

## 4. 已确认的事实（带证据）

1. **材质读 stencil 正确**：`M_StencilToID` 的 SceneTexture 节点 `scene_texture_id = <SceneTextureId.PPI_CUSTOM_STENCIL: 25>`（探针读回确认）。
2. **材质输出路径正确**：`CustomStencil.R / 255 → Emissive`（Divide + Constant 节点；直接输出会在 LDR/PNG 饱和成白色，已改为 ÷255）。
3. **r.CustomDepth 值正确**：UE 5.8 官方定义 `2 = Enabled On Demand（stencil 禁用）`、`3 = Enabled With Stencil`。job 级已用 `r.CustomDepth=3`（之前误写成 2 会关掉 stencil）。
4. **编辑器 actor stencil 值已设置**：31 个组件 `render_custom_depth=True` + `custom_depth_stencil_value=mask_id`，读回验证 31/31 通过。
5. **stencil 值确实带入 PIE 世界**：MRQ 渲染期间探针检查 `PIE Player_L0 mesh: render_custom_depth=True, stencil=1`。
6. **关卡已保存**：设值后 `_save_current_level()` 持久化。

## 5. 核心矛盾

```
PIE 里有 stencil 值(1~11) + custom depth 开启(r.CustomDepth=3) + 材质读 PPI_CUSTOM_STENCIL
                              ↓
               渲染出的 mask 仍然是全 0（stencil 缓冲为空）
```

值带进了 PIE、开关正确、材质读的纹理正确，但**实际渲染时 stencil 缓冲里没有数据**。

## 6. 剩余假设（下一步排查方向）

1. **MRQ 的 post-process pass 是否真的渲染 custom depth/stencil？**
   - 当前 mask job 用 `MoviePipelineDeferredPassBase` + `additional_post_process_materials`（材质 `FinalImageM_StencilToID.*` 输出全 0）。
   - 疑点：该附加 post-process 输出 pass 在 UE 5.8 里可能不会先渲染 custom depth/stencil（stencil 缓冲为空）。
   - 可试：改用 `MoviePipelinePostProcessPass`（该类存在，成员 `enabled`/`material`）作为独立 pass；或给 DeferredPass 显式开启 custom depth 相关设置。
2. **材质在 pass 里实际采样到的是空的 stencil**：与 1 同根因。
3. **兜底方案**：放弃 stencil，改用「每实体独立纯色材质 + `MoviePipelineDeferredPass_Unlit`」渲染 mask，P1 按颜色查表识别（不依赖 stencil 缓冲）。

## 7. 附注：ObjectId / Cryptomatte

- `MoviePipelineObjectIdRenderPass` 在本 UE 5.8 只暴露 `id_type`（无 stencil 模式），输出为 `ActorHitProxyMask00.*` 层，这是 **Cryptomatte 的正常输出层**，正确消费方式是 multilayer EXR + Cryptomatte 解析，不能当普通逐实体 ID PNG 用。当前不做该路径。

## 8. Git 状态

已提交并推送：
- `08fc2a9` feat: 新增 Instance-ID Mask 像素级 GT 标注管线
- `db364b6` fix: 修正 mask stencil 渲染关键问题（r.CustomDepth=3 / 材质÷255 / 关卡持久化 / smoke 配置）
- 另有 `ue_import_config.smoke.json`（单 Camera smoke test 配置，`--config` 支持相对路径）

---

## ✅ 已解决（2026-08-03 更新）

**根因**：UE 5.8 里 MRQ 的 custom depth/stencil 路线在本项目不可用——即使 PIE 里 stencil 值正确、`r.CustomDepth=3`、材质读 `PPI_CUSTOM_STENCIL`，`additional_post_process_materials` 输出的 mask 仍全 0；且 `MoviePipelinePostProcessPass` 在 UE 5.8 是结构体（非独立 render pass），无法作为独立 pass 使用。

**解决方案（已跑通）**：改用 UE 原生 **Object ID Pass + multilayer EXR/Cryptomatte**。
- `mask_source="object_id_pass"`：mask job = `MoviePipelineObjectIdRenderPass`（`id_type=ACTOR`）+ EXR(multilayer)。
- 实测编码：manifest 在 EXR header `cryptomatte/*/manifest`（`{actor_label: "hex_id"}`）；实体 ID 是 float32，存于 `RGBA` 层 R 通道，`hex_id` = 该 float32 位模式的大端十六进制。
- 新增 `grf-ue cryptomatte-to-mask`：把 `render_mask/*.exr` 转成 `mask/{frame}.png`（每个实体像素 = mask_id 1~11，背景 0），与现有 `annotate-masks` / validator 契约一致。

**端到端验证（episode_smoke，3 帧，Camera_01）**：
- mask 灰度值 = `[0,1,2,...,11]`（11 个实体全部命中）。
- `annotate-masks` → 3/3 帧 bbox_source=instance_mask；每实体 bbox 由 mask 可见像素算出（R0 最近 77×160、球最小 6×12）。
- `validate-annotations` → PASSED。
- MOT gt.txt / YOLO det / YOLO seg / overlay 均正常。

**后续正式流程**：UE `--mode full`（object_id_pass 渲 EXR）→ P1 `grf-ue cryptomatte-to-mask` → `grf-ue annotate-masks` → `grf-ue validate-annotations`。

---

### 一句话结论

custom stencil + post-process 材质在 UE 5.8/MRQ 不可用（mask 全 0）；改用 **Object ID Pass + Cryptomatte EXR** 已完全跑通，P1 转换器 `cryptomatte-to-mask` 与现有标注管线无缝衔接。
