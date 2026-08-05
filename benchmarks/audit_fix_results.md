# 审计修复后的基准结果

机器：Windows 11 Pro，20 逻辑核。数据在 G: 盘。基准工具 `uv run grf-ue benchmark ...`
（repeat 1，严格校验阶段状态与帧数；任一阶段不完整即失败）。

## episode_demo（10 帧/相机 × 4 相机 = 40 camera-frames，staged validate）

| workers | cryptomatte | annotate | validate | peak_tree_rss |
| ------: | ----------: | -------: | -------: | -------------: |
| 1 | 4.30 s | 1.83 s | 5.46 s | 122 MB |
| 2 | 3.12 s | 1.16 s | 2.51 s | 305 MB |
| 4 | 2.58 s | 0.96 s | 1.53 s | 527 MB |
| 8 | 3.01 s | 1.00 s | 1.54 s | 823 MB |

> annotate 串行 1.83s → 并行 4 worker 0.96s（1.9×）；validate 5.46s → 1.53s（3.6×）。
> 进程树 RSS 随 worker 数以可解释方式增长（每个 worker 进程加载 numpy/PIL/OpenEXR）。

## episode_0001（300 帧/相机 × 4 相机 = 1200 camera-frames，validate-on-input，--episode outputs/episode_0001）

| workers | cryptomatte | annotate | validate | pipeline | peak_tree_rss |
| ------: | ----------: | -------: | -------: | -------: | -------------: |
| 1 | 147.28 s | 52.74 s | 163.39 s | 363.41 s | 149 MB |
| 2 | 63.10 s | 27.04 s | 67.11 s | 157.25 s | 308 MB |
| 4 | 36.85 s | 17.98 s | 35.77 s | 90.60 s | 576 MB |
| 8 | 28.74 s | 12.58 s | 35.74 s | 77.06 s | 905 MB |

> 端到端（pipeline）workers=8 vs 串行 = **4.7×**；annotate 4.2×、validate 4.6×、
> cryptomatte 5.1×。每阶段都校验：cryptomatte 处理 1200 帧 == staged_total 1200、
> annotate/validate 退出码 0。进程树 RSS 随 worker 数以可解释方式增长
> （每个 worker 进程加载 numpy/PIL/OpenEXR ~100 MB）。

## 单相机 300 帧帧级并行（CLI，episode_0001/CineCam_01，staged 副本）

| workers | annotate 耗时 |
| ------: | ------------: |
| 1 | 11 s |
| 4 | 5 s（2.2×） |
| 8 | 4 s（2.75×） |

> 单相机 + workers>1 确实启用帧级分块并行，不再退回串行。

## 进程树内存

`PeakMemory` 周期采样完整进程树（root + recursive children）：报告
`peak_root_rss` / `peak_children_rss` / `peak_process_tree_rss` / `peak_child_count`。
进程树 RSS 是各进程 RSS 的求和，共享内存页可能重复计数，仅用于比较相同环境下
不同 worker 数的相对趋势。
