# 第二阶段浸泡测试（Soak Test）：300 步 × 4 相机

> 本文记录 `episode_0001`（300 GRF 步 / 4 相机 / 30 FPS MRQ / 1920×1080 / RGB + Object ID Cryptomatte）的完整浸泡测试结果：真实 UE 渲染 + Windows 后处理 + full validation + 审计 + 基准 + 故障恢复。

## 1. 环境与版本

| 项 | 值 |
|----|-----|
| Git commit | `d7423be`（本浸泡测试基于该 HEAD + 工作区修复；修复提交见文末） |
| Branch | `main` |
| Unreal Engine | 5.8 |
| Python | 3.9.25（`.venv`，`uv` 管理） |
| CPU | Intel Core i5-14600KF（14 核 / 20 线程） |
| GPU | NVIDIA GeForce RTX 4070 Ti SUPER |
| RAM | 32 GB |
| 存储 | 数据集盘 G:（运行前空闲 1825 GB；输出共 25 GB） |

## 2. 测试配置

| 项 | 值 |
|----|-----|
| GRF 场景 | `5_vs_5`，seed=42，300 步（`configs/mvp_builtin_5v5.json`） |
| 最终标注帧 | 300 帧 / 相机（10 FPS 标注，`source_step_seconds=0.1`） |
| 相机 | 4（CineCam_01~04） |
| 分辨率 | 1920×1080 |
| 轨迹 FPS | 10（GRF 原生） |
| MRQ 渲染 FPS | 30 |
| 帧映射 | `select_rendered_frame_indices` → `keep = round(i×0.1×30)` = {0,3,…,897}，每 3 帧取 1 |
| Object ID | `MoviePipelineObjectIdRenderPass` → multilayer EXR Cryptomatte（`render_mask/*.exr`） |
| 渲染预设 | `cv_gt`（关 motion_blur/DoF/CA/distortion，单时刻采样，engine warm-up） |
| 输出 | `G:/FutsalMOT_Dataset/episode_0001/` |
| UE 配置 | `ue_import_config.soak.json` + `--mode full` |

## 3. UE 渲染结果

MRQ 提交 8 个 job（4 相机 × [RGB + Object ID]），每相机渲 **900 帧**（0..899）。

| 相机 | RGB render | Object ID EXR | img1 抽取 | 状态 |
|------|-----------:|--------------:|----------:|:----:|
| CineCam_01 | 900（0..899） | 900（0..899） | 300（1..300） | ✅ |
| CineCam_02 | 900 | 900 | 300 | ✅ |
| CineCam_03 | 900 | 900 | 300 | ✅ |
| CineCam_04 | 900 | 900 | 300 | ✅ |

- 首末帧齐全（render 帧 0 与 899、img1 帧 1 与 300 均存在），无缺帧、重复、零字节。
- `render_summary.json`：**`status: success`**（修复后不再恒 partial），每相机 `img1_frames=300`、`mask_frames=300`、`mask_source=object_id_exr`、`ok=True`。

> 注：UE 阶段逐相机开始/结束时间与内存/GPU 采样未记录——渲染由用户在编辑器内执行，`scripts/monitor_soak_resources.py` 未运行（见「尚存限制」）。

## 4. Windows 后处理（正式运行，workers=4）

| 阶段 | 命令 | 耗时 | 峰值树 RSS | 结果 |
|------|------|-----:|-----------:|------|
| Cryptomatte→Mask | `cryptomatte-to-mask --workers 4 --png-compress-level 1` | 57.5 s | 578 MB | 4/4 相机各 300 mask（1200） |
| 标注 | `annotate-masks --include-ball --workers 4 --chunk-size 50 --formats all --clean-stale` | 19.1 s | 368 MB | 1200 camera-frame |
| Full 验证 | `validate-annotations --workers 4 --validation-level full` | 38.9 s | 401 MB | **PASSED（rc=0）** |

> Cryptomatte 首次运行冷缓存偏慢（57.5s）；基准中的同 worker 数热缓存为 37.0s。EXR 用 PIZ 压缩、整帧解压为主。

## 5. 最终文件数量

| 产物 | 每相机 | 合计 |
|------|------:|-----:|
| img1/*.png | 300 | 1200 |
| mask/*.png | 300 | 1200 |
| annotations.jsonl 行 | 300 | 1200 |
| labels/det/*.txt | 300 | 1200 |
| labels/seg/*.txt | 300 | 1200 |
| gt/gt.txt（MOT 行） | 3296–3299 | 13189 |
| render/*.png（原始 RGB） | 900 | 3600 |
| render_mask/*.exr（原始 EXR） | 900 | 3600 |
| camera.json / seqinfo.ini / mask_config.json / gt.txt | 各 1 | 各 4 |

标注统计：不可见对象 11（不进入 MOT/YOLO）；多连通域对象 3651（合并 1867、面积门回退最大连通域 1784）。

## 6. 同步与正确性（`scripts/audit_soak_episode.py`）

| 检查 | 结果 |
|------|------|
| 相机数量 / 期望帧数 | 4 / 300 ✓ |
| render/EXR 覆盖 keep_indices | 全部覆盖 ✓ |
| img1/mask/ann/det/seg 缺帧、重复、零字节 | 无 ✓ |
| 跨相机时间同步（time/source_step/episode_id） | 300 帧一致 ✓ |
| track_id / mask_id 稳定映射（13200 对象） | 全部一致 ✓ |
| camera.json 标定（分辨率一致、外参互异、无 NaN） | 合法 ✓ |
| render_summary | success ✓ |
| 进程内 validate（quick） | rc=0 ✓ |

审计报告：`G:/FutsalMOT_Dataset/episode_0001/audit/soak_audit_report.{json,md}`。

## 7. 视觉抽查

每相机生成 bbox 叠加图 + mask 彩色图（`debug/{frame}_{bbox,mask_color}.png`）与 bbox 标注视频（`video_30fps.mp4`，300 帧 @30fps）。

- 固定帧：1、2、3、50、100、150、200、250、298、299、300。
- 自动选择帧（每相机）：极小可见球帧（球 1–5 px）、遮挡最严重帧、多连通域最多帧（30–31 components）、可见对象最多/最少帧。索引见 `G:/FutsalMOT_Dataset/episode_0001/audit/spot_check_index.md`。
- 程序化核对：末三帧球位置连续变化（5.204→5.078→4.959）、无停顿/重复；RGB↔mask 像素对齐；bbox==mask min/max（full validator 复验）。人工目视结论待用户按索引确认。

## 8. 稳定性

- 后处理峰值进程树 RSS：crypto 578 MB / annotate 368 MB / validate 401 MB（workers=4）。
- 基准峰值树 RSS 随 worker 数以可解释方式增长（148→317→518→892 MB，每 worker 加载 numpy/PIL/OpenEXR ~100 MB）。
- 子进程在命令结束后全部退出（`peak_child_count` 与 worker 数一致，命令返回后无残留）。
- 磁盘总量 25 GB（render 14.1 + img1 4.7 + debug 4.1 + render_mask 2.8 + 其余 <0.1 GB）。
- **未捕获项**：UE 编辑器渲染期间内存/GPU 采样缺失（用户未运行监控脚本）。

## 9. 故障恢复测试（受控副本，未动真实数据）

流程：复制 CineCam_01 完整渲染 → 删除非首尾 keep-index 帧 `render/000150.png` 与 `render_mask/000240.exr`。

| 步骤 | 结果 |
|------|------|
| 审计缺帧检测 | `FAIL`（render 缺 150、EXR 缺 240），**退出码 1** |
| `recover_render_to_img1` | **`partial`**：img1=299、mask=299、`ok=False` |
| render_summary | **`status: partial`**（不写 success） |
| 补回缺失帧后恢复 | `success`：img1=300、mask=300 |

> 删除非 keep-index 的中间 EXR（如 250）不会触发缺帧——它们是渲染冗余帧，不参与标注，属正确行为。

## 10. 发现并修复的问题

| # | 问题 | 根因 | 修改文件 | 修复 |
|---|------|------|---------|------|
| 1 | render_summary 对 Object ID EXR 恒报 `partial` | `find_mask_files`/`copy_mask_frames` 只认 PNG，EXR job 恒 `mask_frames=0` | `ue/render_episode.py` | 支持 EXR 发现与对齐统计；EXR 源只统计不复制（P1 转换）；记录 `mask_source` |
| 2 | watchdog 30 分钟硬超时对长渲染误收尾 | `elapsed > 1800` 无条件收尾，不区分渲染是否仍在推进 | `ue/render_episode.py` | 超时**且**文件数长时间无变化才收尾 |
| 3 | `total_img1_frames` 混入 mask 对齐计数（2400） | finalize 把所有 job 复制/对齐数累加 | `ue/render_episode.py` | 分列 `total_img1_frames`（RGB）与 `total_mask_frames` |
| 4 | 审计单相机标定误报 | 「四相机外参相同」检查未排除单相机场景 | `scripts/audit_soak_episode.py` | ≥2 相机才检查 |

回归测试：`tests/test_render_export.py` 新增 5 例（EXR 发现、EXR 只统计不复制、EXR 部分对齐、恢复 EXR success/partial）。

## 11. 性能基准（staging 副本，1200 camera-frame，repeat 1，full validation）

| workers | Cryptomatte | Annotate | Validate | Pipeline | 加速比 | Peak tree RSS |
| ------: | ----------: | -------: | -------: | -------: | ------: | ------------: |
| 1 | 145.38 s | 55.71 s | 168.97 s | 370.07 s | 1.00× | 148 MB |
| 2 | 63.97 s | 26.94 s | 67.86 s | 158.78 s | 2.33× | 317 MB |
| 4 | 37.02 s | 20.52 s | 41.78 s | 99.31 s | 3.73× | 518 MB |
| 8 | 35.86 s | 14.12 s | 35.50 s | 85.48 s | 4.33× | 892 MB |

**结论**：w4→w8 端到端仅 +16% 提速但内存 +72%（892 MB）；本机 20 线程推荐默认 **w4**（性价比拐点），内存允许时 w8。瓶颈以 EXR 整帧解压（Cryptomatte）与 mask 多边形提取（annotate）为主，非 NVMe。

## 12. 测试结果

- pytest：**304 passed**（新增 5 例 EXR 回归）。
- `uv build`：通过（见提交记录）。
- full validator：`validate-annotations --validation-level full` 退出码 0，4 相机全部 PASSED。
- soak 审计：0 失败项，退出码 0。

## 13. 尚存限制

- **EXR 解码**：PIZ 整帧解压 + openexr 无按通道 API，Cryptomatte 为最大耗时项（靠并行摊薄）。
- **磁盘占用**：单数据集 25 GB（含原始 render/EXR 与 debug 可视化）；debug/ 可删以节省 4 GB。
- **内存随 worker 线性增长**：w8 峰值树 RSS 892 MB；低内存机器建议 w1–4。
- **UE 渲染资源采样缺失**：`scripts/monitor_soak_resources.py` 已提供，但本次渲染期间未运行；如需 UE 侧内存/GPU 趋势需补跑。
- **恢复粒度**：恢复仅统计缺帧并标 partial，不能从 EXR 重新渲染缺失 job（需重跑 MRQ）。

## 14. 推荐命令（复现）

```powershell
# 1) 导出 episode
uv run grf-ue export --config configs/mvp_builtin_5v5.json --output outputs/episode_0001
uv run grf-ue validate outputs/episode_0001

# 2) UE 一键全流程（Unreal Editor Python Console）
py "D:/projects/FustalMOT_UEDataset/Content/FutsalMOT/code/ue/import_grf_episode.py" --config ue_import_config.soak.json --mode full

# 3) 渲染完成后 P1 后处理
uv run grf-ue cryptomatte-to-mask G:/FutsalMOT_Dataset/episode_0001 --workers 4 --png-compress-level 1 --episode outputs/episode_0001 --mapping ue/actor_mapping.example.json
uv run grf-ue annotate-masks G:/FutsalMOT_Dataset/episode_0001 --include-ball --workers 4 --chunk-size 50 --formats all --clean-stale
uv run grf-ue validate-annotations G:/FutsalMOT_Dataset/episode_0001 --workers 4 --validation-level full

# 4) soak 审计
uv run python scripts/audit_soak_episode.py --input G:/FutsalMOT_Dataset/episode_0001 --expected-cameras 4 --expected-frames-per-camera 300 --episode outputs/episode_0001 --validation-level quick

# 5) 渲染期间资源监控（可选）
uv run python scripts/monitor_soak_resources.py --input G:/FutsalMOT_Dataset/episode_0001 --interval 30
```
