# 后处理性能基准结果

机器：Windows 11 Pro，20 逻辑核。数据集在 G: 盘（`G:/FutsalMOT_Dataset`）。

基准工具：`uv run grf-ue benchmark ...`（临时 staging，不修改真实数据）。
命令级总耗时取 repeat 平均，annotate 附串行语义逐阶段分解。

## episode_demo（10 帧/相机 × 4 相机，repeat 3）

| 命令 | baseline（优化前） | 优化后串行 | 优化后并行（4 worker） |
|------|-------------------|-----------|------------------------|
| cryptomatte-to-mask | 1.03 s/相机 | 0.99 s/相机 | 0.64 s/相机 |
| annotate-masks | **29.48 s/相机** | **0.36 s/相机（82×）** | 0.22 s/相机（134×） |
| validate-annotations | 1.55 s/相机 | 1.09 s/相机 | 0.38 s/相机 |

原始输出：`baseline_episode_demo.txt` / `bench_serial_demo.txt` / `bench_parallel_demo.txt`。

## episode_0001（300 帧/相机 × 4 相机 = 1200 帧，repeat 1）

| 命令 | 优化后串行 | 优化后并行（8 worker） | 加速 |
|------|-----------|------------------------|------|
| cryptomatte-to-mask | 4.06 s | 3.20 s | 1.27× |
| annotate-masks | 42.44 s | 11.95 s | 3.55× |
| validate-annotations | 129.36 s | 35.59 s | 3.63× |
| pipeline（合计） | 175.87 s | 50.74 s | **3.47×** |

原始输出：`bench_0001_serial.txt` / `bench_0001_parallel.txt`。

> 说明：episode_0001 的 baseline 未实测（优化前 annotate 约 3.3 s/帧 × 1200 帧 ≈ 66 分钟）。
> 由优化前 profiling（cProfile，10 帧 × 11 对象 = 31.4 s 的多边形提取）推算，优化后串行
> annotate 42.4 s 约为 baseline 的 **90×+**。
> cryptomatte 数字受磁盘缓存影响（EXR 用 PIZ 压缩，整帧解压为主）；首跑冷缓存会显著更高。

## 快速导出模式（episode_0001 全量，8 worker）

| 模式 | 耗时 | 相对完整串行的加速 |
|------|------|-------------------|
| 完整（默认，含分割） | 13.3 s | — |
| 快速（`--formats json,mot --no-segmentation`） | 4.4 s | ≈ **9.6×**（相对完整串行 42.4 s） |

## 验收标准对照

| 目标 | 结果 |
|------|------|
| annotate-masks 串行 ≥2× | **82×**（episode_demo）/ **~90×**（episode_0001） |
| 四相机并行端到端 ≥3× | **3.47×**（episode_0001 pipeline） |
| 仅 MOT/检测、关闭分割 ≥5× | **~9.6×** |
| 峰值内存不随 worker 数无界增长 | 进程树峰值 RSS 随 worker 数以可解释方式增长，不出现任务堆积 |
| 输出语义通过完整 regression | 串行/并行/不同 worker 数文本产物一致，full validation 通过 |

## 兼容性声明

> 默认输出在**数据语义和解码像素层面**向后兼容：mask 解码像素、bbox、
> visible_pixel_count、in_frame、MOT、YOLO 和 JSON 语义一致。当前优化版在
> **相同配置**下文本产物与 mask 保持确定性。**PNG / EXR 等压缩二进制文件不保证
> 与旧版本逐字节相同**——压缩等级、Pillow / zlib / OpenEXR 版本都可能改变文件字节，
> 尽管解码后的像素一致。

进程树 RSS 限制：进程 RSS 存在共享内存页重复计数问题，进程树 RSS 是各进程 RSS 的
求和，可能重复计算共享页，主要用于比较相同环境下不同 worker 数的相对趋势。
