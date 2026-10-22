# 导出移交与 P1 后处理性能验证

本轮只修改原始 Python 仓库，不修改正在恢复的 episode、UE 会话或 `Saved/WeatherPushReview/dataset` 运行副本。硬链接只用于 MRQ 完成后不再改写的帧；删除 scratch 名称后，canonical 目标仍持有完整数据。

## RGB / EXR 移交

`ue/render_episode.py` 的 `_link_or_copy_frame_atomic` 在目标目录内建立唯一 staging，先尝试硬链接，失败时回退 `copy2`，最后使用 `os.replace` 原子覆盖。复制或覆盖失败时保留源及旧目标，清理 staging。源与目标确为同一路径时仍抛出 `SameFileError`；中断后重跑时，不同路径但已经正确硬链接的帧直接复用。RGB 帧映射、EXR 渲染帧号和源文件清理时机保持原契约。

真实文件回归覆盖同 inode、SHA/字节一致、源删除后目标存活、copy/replace 中断、mtime、部分完成后重入以及只删除已完成 EXR。测试文件为 `tests/test_render_frame_handoff.py`。

小型性能工具不读取现有数据，可显式选择两个卷验证真实跨卷回退：

```powershell
.venv/Scripts/python.exe tests/perf_render_frame_handoff.py `
  --scratch-parent D:/projects/FutsalMOT_UEDataset/Saved/PaperPilot/20261022_cpu `
  --cross-volume-parent C:/Users/20113/AppData/Local/Temp `
  --output D:/projects/FutsalMOT_UEDataset/Saved/PaperPilot/20261022_cpu/handoff_benchmark.json
```

2026-10-22，8 个 8 MiB 临时文件，各测 3 轮：

| 移交方式 | 中位耗时 | inode 验证 |
| --- | ---: | --- |
| 同卷 copy2 | 0.0340 秒 | 24/24 不同 |
| 同卷原子硬链接 | 0.0102 秒 | 24/24 相同 |
| D → C 跨卷原子复制 | 0.0435 秒 | 24/24 不同 |

所有 SHA256 一致。计时不含准备和校验，数据可命中系统缓存，不能外推 SATA HDD 的持续吞吐。硬链接确定消除的是同卷每份 payload 的一次读取和一次写入，仍有目录和元数据操作。以实测代表帧 RGB 14,597,311 字节、EXR 40,337,963 字节估算，4×902 帧若全部同卷，少搬运约 198.21 GB payload，即约 396.41 GB 的读写总量；若 EXR 输出通过 junction 跨到 SSD，这部分仍需真实复制，只有同卷 RGB 可少约 105.33 GB 的读写总量。上述体积是代表帧估算，不是全 episode 的实测汇总。

## P1 Cryptomatte 读取实验

生产解码器维持现有完整 uint32 位模式比较、最多 manifest 命中选层和同分 RGBA 优先语义。本轮没有修改 `cryptomatte.py`。原始与 frozen 解码器 SHA256 均为 `d6c92dc8a4b70f2cf8426009d6b79e4029f36afa4a7a0c8f070c9ab69de97730`。

`tests/perf_cryptomatte_readers.py` 只读既有 EXR/golden，在独立目录输出 PNG 和证据，并拒绝把实验输出放在 episode 内。它比较生产 `OpenEXR.File` grouped 读取、实验 planar 读取，以及每帧独占 File/缓冲区的 1 或 2 线程/进程。planar 复用 grouped 的层顺序、尺寸和 dtype，任何不支持的通道布局立即失败，不进行隐式精度转换。

```powershell
.venv/Scripts/python.exe tests/perf_cryptomatte_readers.py `
  --episode G:/FutsalMOT_Dataset/_staging/4k_v1/episode_paper_4k_joint_gate_retry2 `
  --output D:/projects/FutsalMOT_UEDataset/Saved/PaperPilot/20261022_cpu/cryptomatte_readers_repeated `
  --frames-per-camera 3 --repeats 2
```

12 张原生 3840×2160、4 个机位，每方案 2 轮并轮换执行顺序，共 96 次解码。所有方案的选层、ID uint32 SHA256、mask 每个像素、PNG 字节、实体计数和 pixel-tight bbox 均与 grouped 基线一致；每个 mask 和可见 bbox 也逐一与已有 golden 比对。墙钟包含逐像素和 bbox 验证：

| 方案 | 第 1 轮 | 第 2 轮 |
| --- | ---: | ---: |
| grouped / 1 线程 | 10.728 秒 | 10.865 秒 |
| grouped / 2 线程 | 5.875 秒 | 5.899 秒 |
| planar / 1 线程 | 10.539 秒 | 10.400 秒 |
| planar / 2 线程 | 5.669 秒 | 5.631 秒 |

当前 OpenEXR 版本是 3.4.13，没有新版文档的 `set_global_thread_count` / `num_threads` 接口；不为此升级固定的 P1 环境。官方说明 `File` 在 Python 中整帧分配数组，`separate_channels=True` 只改变布局，不能选择性省掉解压：[OpenEXR Python 文档](https://openexr.com/en/latest/python.html)。

planar 相对 grouped 的收益仅约 3–4%，不足以承担新的分组兼容性边界，保留为实验。2 线程在本次短样本含校验流程中约快 1.83 倍。随后用 `--executor process --readers grouped` 单独检查 Windows 进程池，12 帧各 2 轮，共 48 次解码，同样全部精确一致：

| 方案 | 第 1 轮 | 第 2 轮 |
| --- | ---: | ---: |
| grouped / 串行 | 10.817 秒 | 10.697 秒 |
| grouped / 2 进程 | 6.012 秒 | 5.919 秒 |

2 进程的含校验墙钟约快 1.80 倍，证据在 `cryptomatte_process_repeated/result.json`；这仍是短样本，不能代替 902 帧全序列的吞吐和内存验证。正式入口已经支持 `workers=2`，应在 MRQ 完全结束、内存空余充足后验证它，并以全帧 annotation validator 和 task audit 为准。不要把 `workers=0` 的按核心数自动并发当作 4K 安全配置。

证据目录为外层 `Saved/PaperPilot/20261022_cpu/`。`render_handoff_only.patch` 只含 `os/tempfile` 导入、helper 和 RGB/EXR 两处调用；`render_episode_handoff_candidate.py` 是供审查的候选文件，并未覆盖 frozen。候选独立通过 14 项真实文件移交测试，四个移交函数 AST 与原始实现一致。候选基于 frozen SHA256 `42ebdc234308328b2bfdee39e814934e4ba43253139fa13bbf756f0eb7588b00`，应用前应再次核对状态和 hash，不能把原始文件里的号码系统改动一起复制进去。

## 取消请求与失败恢复的时序

UE 5.8 的 `executor.cancel_all_jobs()` 可能在 PIE 拆卸调用栈内同步触发 `finished(False)`，同时没有可读的 `is_cancelled` 属性。外部自动化应调用当前 `_AsyncRenderPipeline` 实例的 `request_cancel()`：它先登记 `_cancel_wait`，再调用 native 取消 API。返回 `True` 只表示 native 请求已调用，不证明 MRQ 已停止或 restore 已完成；重复请求返回 `False`。

已有等待时，`finished(False)` 或迟到的 `finished(True)` 都不得绕过等待。普通失败也等待目标 Editor World、PIE inactive、MRQ inactive 和 executor 停止连续两个 tick 后才调用恢复，但保持 `failed` 及原始原因。没有明确取消证据的原生 UI 取消仍只能报告 `failed`，不能根据 `success=False` 猜成取消。watchdog 错误也使用同一等待。注册失败或 120 秒超时保留快照，报告 `MANUAL_REQUIRED` 和 `restore_deferred=true`，禁止恢复回调；过程中不阻塞 tick、不保留 UWorld 引用。

CPU 回归在 `tests/test_render_cancel_request.py`，覆盖无取消属性下的同步重入、迟到 success、重复请求、native API 缺失/异常、无提交地图、注册失败、真实失败原因保留和超时。新增复现先出现 9 个预期失败，修复后相关渲染/时序测试 152 项通过；独立 frozen 候选通过 25 项新取消和既有生命周期/轮询测试。这些证据不能替代真实 UE 取消 smoke。

本轮候选是 `render_episode_cancel_candidate.py` 和 `render_cancel_only.patch`，基于已移植 handoff 的 frozen 实际 SHA256 `247f5982c93a7de4bf684d66c8a3d742f29669b274bbfe65e8ae2fe56d10b874`。回执 `render_cancel_candidate.json` 的 hash 始终按 `read_bytes()` 计算；补丁只新增 3 个方法、修改 4 个已有方法，其余方法和顶层 AST 保持不变。正在运行 MRQ 时不覆盖 frozen，须等整批结束并核对执行 profile 后再移植。

## 无损 EXR 压缩的小样本比较

`tests/perf_exr_lossless_compression.py` 仅接受 2–3 张独立 EXR，拒绝在源 episode 内写入。2026-10-22 使用既有 4K golden 的 CineCam_01/02/04 各 1 张，逐张串行比较 PIZ、ZIP、ZIPS，并轮换编码次序；不测 RLE 或有损格式。P1 OpenEXR 3.4.13 的 9 份输出全部回读通过：20 个通道的名称、尺寸、dtype、采样、pLinear、完整 HALF uint16/FLOAT uint32 位模式，以及全部 header metadata（除预期改变的 compression）和 Cryptomatte 元数据均一致。

| 压缩 | 平均写入墙钟 | 平均回读墙钟 | 3 帧合计大小 |
| --- | ---: | ---: | ---: |
| PIZ（当前源格式） | 1.684 秒 | 0.801 秒 | 120.20 MB |
| ZIP | 1.970 秒 | 0.397 秒 | 107.71 MB |
| ZIPS | 2.139 秒 | 0.360 秒 | 107.08 MB |

ZIP 和 ZIPS 少约 10–11% 文件体积、回读更快，但 P1 写入分别慢约 17% 和 27%。这不支持“换 ZIP 可直接提高当前 UE 编码吞吐”的结论；若为后处理或容量优化，可在下一轮 UE 中单变量小试，不能直接迁移生产。计时包含 `File.write` 编码和 D 盘缓存写入，不包含 fsync；机器同时运行 UE MRQ，且 P1 绑定版本、默认压缩级别、线程实现与 UE 实际 encoder 未证明相同。官方将这三种格式列为无损，ZIP 使用 16 行一块，ZIPS 使用单行：[OpenEXR 技术说明](https://openexr.com/en/latest/TechnicalIntroduction.html)。

成功证据在 `Saved/PaperPilot/20261022_cpu/exr_lossless_compression_retry1/result.json`。首次仪器运行因 `OpenEXR.File.__exit__` 清空传入字典而在第二个编码前中断，已通过独立 2×2 最小复现确认，并改为复制字典容器、保持原像素缓冲区后重跑；首次目录不计入结果。源 EXR、现有数据和生产配置未改变。
