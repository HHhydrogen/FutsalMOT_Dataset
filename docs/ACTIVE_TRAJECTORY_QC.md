# 多 seed 的连续活跃轨迹筛选

更新：2026-10-21。选段模块为 `src/grf_ue_bridge/active_trajectory.py`，只读源轨迹并提供选段计划；裁剪模块为 `src/grf_ue_bridge/active_episode.py`，新建 episode 并重建 manifest，两个模块均不修改原始源文件或 UE 资产。

目标是保留有效运动、跳过后期长期停滞或小范围抖动，再增加其他 seed；没有固定 900 帧要求。QC 阈值不等于 GRF 的比赛语义标签。球员记录的 `active=true` 也不是运动证据，所有运动指标均由相邻 `position_m` 平面坐标差分计算。

10秒上限仅用于最初快速渲染门禁，不能作为正式 MOT 数据集的固定长度上限。正式选段应将 `max_duration_s` 设为完整源时长，并保留尽可能长的合格连续窗。

## 默认门禁

| 参数 | 默认值 | 目的 |
| --- | ---: | --- |
| 最短/最长采样时长 | 5 / 10 秒 | 首批快速生成短片段，实际长度可变 |
| 活跃速度 | 0.5 m/s | 复用已有 `motion_quality.ACTIVE_MPS` |
| 团队活跃 | 至少6名外场球员，覆盖至少90%的帧间隔 | 复用已有团队门槛 |
| 连续低运动上限 | 1秒 | 容许短暂正常停顿，排除长停滞 |
| 真实空间跨度 | 至少4名外场球员的平面包围盒对角线 ≥1.5m | 排除小区域快速抖动 |
| 不合理位置跳变 | 任意球员速度 >12m/s 的窗拒绝 | 避开回合重置等瞬移 |
| 默认窗口数 | 每 seed 1段 | 可配置更多互不重叠的连续窗 |
| 局部运动邻域 | 默认0（兼容最初短窗）；长窗显式设3秒 | 防止全窗先跑动后长期快速小抖动 |
| 局部真实跨度 | 至少4名外场球员在邻域内跨度≥1米 | 不把高速往返抖动当作持续运动 |
| 连续局部低运动上限 | 3秒 | 长窗启用局部门禁时生效 |

门将不参与团队活跃人数，也不要求每名外场球员始终运动。空间跨度与路径长度分别记录：真正跑动后返回起点的球员可以有效；只靠累计路径变长的小区域抖动不能通过跨度门禁。超过速度阈值属于需要检查的源轨迹异常，不能把它视为更活跃的数据。

5/10秒以采样帧数计：10FPS下为50–100帧，首末观测的真实时间跨度为4.9–9.9秒。报告同时保存 `sample_duration_s` 和 `trajectory_span_s`，避免论文中的时长混淆。

## 确定性和 provenance

先尝试最长窗口，符合条件的同长度窗口按团队覆盖、实际空间运动和较早起点排序。不消费 Python/NumPy RNG，不循环轨迹，不逐帧删除后拼接。多个窗口仍是独立连续片段，同 seed 的所有窗口和 camera 必须留在同一数据划分中。

输入必须显式提供 `dt_s`，或通过 episode helper 从 `meta.timing.source_step_seconds` 读取，不能隐式猜成30FPS。每帧 roster、坐标有限性和存在的时间戳连续性均检查。错误输入失败，不生成“勉强有效”的窗口。

长源实现用前缀计数检查活跃覆盖、瞬移和连续低运动，稀疏极值表检查任意候选区间空间跨度；只对最终选段计算逐人完整路径等指标。预处理复杂度为 O(人数×n log n)，候选最坏为 O(人数×n²)，空间 O(人数×n log n)，不再对每个候选重复扫描全窗形成 O(n³)。1800帧实测约0.35–0.5秒/seed/策略；极端全局小范围高速抖动、且关闭局部门禁时约10.86秒。持续真实运动1800帧约0.084秒，长静止尾约0.284秒；这些为本机 CPU 测量，不是硬实时保证。

长窗启用 `local_motion_window_s=3`、`min_local_spatial_extent_m=1`、`max_local_low_motion_streak_s=3`，对完整源中居中邻域计算局部真实移动人数，在源两端平移邻域。连续局部移动人数不足4超过3秒的窗口拒绝。报告保存 `source_local_motion` 和每个选段的 `local_motion.coverage/max_low_motion_streak_s`。该邻域依赖完整源上下文，不能在裁剪后重新分类并要求边界指标与源报告相同。

建议并行比较两个可审计 QC 候选，不把阈值直接写成“有效比赛”的真值：严格档为≥6人活跃覆盖90%、低运动≤1秒；较宽 MOT 候选为≥4人活跃覆盖80%、低运动≤3秒。两档都保留至少4人全窗空间跨度≥1.5米、任意球员速度≤12m/s，以及上述局部抗抖动门禁。不能将两个档的结果混合后隐藏 policy；同 seed/camera 也不能跨 train/val/test。

```python
long_policy = ActiveWindowPolicy(
    min_duration_s=5,
    max_duration_s=180,
    min_active_outfields=4,
    min_team_active_coverage=0.8,
    max_low_motion_streak_s=3,
    local_motion_window_s=3,
    min_local_spatial_extent_m=1,
    max_local_low_motion_streak_s=3,
)
```

报告 schema 为 `futsalmot_active_trajectory_qc` v1，记录完整 policy 和 SHA256、源帧数、dt、门将集合、原轨迹运动指标、候选拒绝原因计数和每个连续窗。窗口范围为原始帧数组的半开边界 `[start_index,end_index_exclusive)`，并保留源 step/time。每个球员记录路径长度、净位移、空间跨度、活跃比例、静止比例和最大速度。

调用示例：

```python
from grf_ue_bridge.active_trajectory import (
    ActiveWindowPolicy, inspect_episode_active_windows,
)

report = inspect_episode_active_windows(
    episode_dir, policy=ActiveWindowPolicy(max_windows=1),
)
for window in report["selected_windows"]:
    start = window["start_index"]
    end = window["end_index_exclusive"]
```

实际 crop 应新建输出，保留原始数据与 QC 报告，重新编号本地 step/time，并按实际保留帧数更新 meta、resolved task、appearance/environment manifest 和轨迹指纹。禁止直接截短已带指纹的文件后继续使用旧 provenance。

## 生成裁剪 episode

```python
from grf_ue_bridge.active_episode import prepare_active_episode
from grf_ue_bridge.config.resolver import load_resolved_task

entry = prepare_active_episode(
    load_resolved_task(resolved_path),
    source_episode=archived_source_dir,
    selection_report=report,
    window_index=0,
)
```

`resolved_path` 对应裁剪后的渲染任务。在本批10FPS任务中，`export_profile.num_steps` 必须等于窗口帧数，不能保留源任务的300步。目标帧率、playback FPS、seed、scenario 必须与源轨迹一致；不允许在裁剪任务重复使用时间缩放。高帧率输入按 `num_steps × (target_fps/10)` 检查输出帧数；窗口不能整除该倍数时必须调整窗口或配置。

裁剪工具先验证源 episode，再用报告 policy 重新选段并严格比对窗口，防止报告与源轨迹不一致。只复制 `[start_index,end_index_exclusive)` 的连续原始帧，重置 `step=0..N-1`、`time_seconds=index×dt`；球员、球、ownership、score 等其余数据保持原样。目标 meta 保留源随机种子、实体与坐标定义，只更新 episode ID 和实际帧数。

输出首先写入同盘临时目录，执行标准 `_write_provenance`、appearance/environment manifest 构建及验证、`validate_episode` 后发布。输出路径不能逃逸 `dataset_root` 或与源目录重叠；已有不同内容拒绝覆盖。相同输入可以幂等复用，并重新核对 frames/meta 与 manifest。失败临时目录保留以便诊断。

`provenance/active_window.json` 记录完整源 frames/meta 的 SHA256、原始 step 半开范围、原始时间半开范围、QC policy SHA256、每人运动指标、dt、裁剪后 SHA256 和输入指纹；`active-selection.json` 保存重新计算的全源 QC；`source-meta.json` 保存原始 meta；`source-capture.json` 保存完整源采集要求。`source_capture_steps` 定义为完整源 JSONL 的采样帧数，本批为300；它不表示裁剪后的帧数，也不应与高帧率插值前 GRF 引擎步数混淆。

复现步骤必须为：先按原配置采集完整300帧源轨迹，核对源 SHA256，然后应用本窗口裁剪和时间归零。直接运行100步渲染 task 的 `task export` 不会得到从第20帧等非零起点开始的裁剪结果，不能声称复现。工具自身不启动 GRF。若归档只有 frames/meta，可以忠实裁剪，但无法从这两个文件恢复所有 GRF 导出参数；此时明确记录 `source_export_profile_available=false`。若源目录保留 `provenance/export-profile.json`，工具会将原采集 profile 写入 crop provenance。应在 seed 公共信息中保留原完整采集 config、版本锁和复现证据。

CLI 可读单份报告，或用 `--seed` 从报告列表中选择一项：

```powershell
python tools/prepare_active_episode.py --resolved resolved-task.json --source raw_sources/seed_1001 --selection-json active_selection.json --seed 1001 --output-json entry.json
```

返回值为 index 用 dict，含目标目录、episode_name、root_seed、frame_count、num_steps、dt、source_window、source_capture_steps、源 SHA256、policy SHA256、input_fingerprint 和 active_provenance 路径。CLI stdout 包含验证日志，机器消费结果请使用 `--output-json`。

## 本机实测

源为 pilot seed1001–1008 的300帧10FPS轨迹。正式筛选均找到100帧有效窗，团队≥6人活跃覆盖91.9%–100%；具体范围与完整指标保存在本机 `Saved/PaperPilot/20261021/active_selection.json`。

原300帧全段的≥4人活跃覆盖在 seed1002 和1006 中仅52.7%和59.3%，而筛出的短窗均满足更严格的≥6人门禁。该现象支持按 seed 筛选而不固定输出长度；它不证明其他 seed 或更长源片段均有足够有效数据。没有合格连续窗时应跳过 seed，或继续采集其他 seed。

验证：`python -m pytest tests/test_active_trajectory.py -q`。测试覆盖静止、小范围高速抖动、真实多人运动、门将及个别外场静止、短暂停顿、长停滞不拼接、返回起点的正常运动、瞬移拒绝、可变长度、时间/roster错误及确定性。

裁剪验证：`python -m pytest tests/test_active_episode.py tests/test_active_trajectory.py tests/test_motion_quality.py -q`。覆盖连续裁剪、归零、完整源不变、原始300帧要求、manifest 指纹重建、同输入幂等、不同输出拒绝、路径逃逸、配置/报告不匹配、manifest 篡改拒绝及 CLI 入口。

2026-10-21，本机同 GRF/代码版本实际重新采集 seed1001 的完整300帧，SHA256 为 `a2a94a7bbbf7d74bdfba436a1873c28e4959da77420fafd11e15c7f3e76ae222`，与归档逐字节一致。证据在本机 `Saved/PaperPilot/20261021_active/source_reproduction_check.json`。这仅证明该 seed 在本机当前版本下的完整源重跑一致，不能外推所有 seed、平台或 GRF 版本。

## 180秒长源比较

2026-10-21，seed1001–1008 各扩至1800帧10FPS，`max_duration_s=180/min_duration_s=5`；两档均启用局部抗抖动门禁。下表为采样时长，首末观测跨度需减0.1秒：

| seed | 严格最长秒数 | 较宽 MOT 候选范围 | 较宽秒数 |
| --- | ---: | --- | ---: |
| 1001 | 16.5 | [0,307) | 30.7 |
| 1002 | 14.3 | [0,185) | 18.5 |
| 1003 | 20.2 | [0,286) | 28.6 |
| 1004 | 22.5 | [0,245) | 24.5 |
| 1005 | 26.8 | [0,451) | 45.1 |
| 1006 | 11.2 | [0,202) | 20.2 |
| 1007 | 30.3 | [541,974) | 43.3 |
| 1008 | 37.1 | [449,865) | 41.6 |

报告为本机 `Saved/PaperPilot/20261021_long/long_policy_comparison_local.json`，另保留关闭局部门禁的基线 `long_policy_comparison.json`；不覆盖最初短窗或30秒源的报告。`continuity_bound_only` 档只取消活跃覆盖要求以诊断连续低运动/瞬移边界，不应直接用于正式产出。加入局部检查使 seed1003 的最优片段从 [432,723) 改为 [0,286)，并缩短数个 seed 的尾段；仅用全窗空间跨度不足以排除混合长窗中的后期小抖动。

180秒源最长连续局部低运动为85.6–164.5秒。增加源时长不能保证同 seed 获得更长有效 MOT 片段，应同时扩充 seed，并结合 GRF 比赛状态和原始观测诊断长期低运动原因；这些运动统计本身不证明具体引擎原因。

## 40 seed 源池与天气分桶

`tools/scan_active_source_pool.py` 增量读取已导出源索引，根据源 SHA256 与完整 policy 复用已完成报告；`--watch-seconds` 可短期等待后续追加源。每个源的最长时长上限动态取 `meta.timing.num_steps × dt`，不固定为180秒。扫描只读取完成的源，不运行 GRF 或 UE。

```powershell
python tools/scan_active_source_pool.py --index source_pool_index.json --output-dir pool_qc
```

输出 `active_pool_selection.json` 保存每 seed 完整 QC 和运动诊断；`active_pool_ranking.json` 按8个天气桶列出全部候选，按窗长降序、seed升序排序。天气由 `(root_seed-1001)%8` 固定，0–5为 train、6为 validation、7为 synthetic_test。排序不读取任何训练或模型指标；同 seed 的所有片段/镜头仍须留在同一划分。

2026-10-21，seed1001–1040 共40个180秒源完成扫描。各桶最长候选如下，不能把合成测试桶称为真实测试集：

| 天气桶 | seed | 连续源范围 | 秒数 | 划分 |
| --- | ---: | --- | ---: | --- |
| daylight_clear_v1 | 1025 | [0,430) | 43.0 | train |
| overcast_soft_v1 | 1034 | [0,789) | 78.9 | train |
| warm_evening_v1 | 1019 | [55,497) | 44.2 | train |
| cool_evening_v1 | 1036 | [353,860) | 50.7 | train |
| light_haze_v1 | 1021 | [0,902) | 90.2 | train |
| cloudy_variation_v1 | 1022 | [379,825) | 44.6 | train |
| dawn_cool_v1 | 1007 | [541,974) | 43.3 | validation |
| night_floodlit_v1 | 1024 | [208,722) | 51.4 | synthetic_test |

8组为每相机4463帧，4机位共17852帧。完整本机报告在 `Saved/PaperPilot/20261021_long/pool_qc/`，表中候选只是按运动/长度的排名，不表示渲染或 annotation 已通过。

40个源中38个出现最终≥4名外场球员运动不足的尾段，尾段13.1–166.7秒；另外两个源末尾没有此类连续尾段。38段内 `game_mode` 都保持0，比分与持球者不变，球始终位于名义场地内；球平面平均速度最高0.040077m/s，平面包围盒对角线最高0.908544m。全部球员近乎完全静止的帧占各尾段54.2%–98.1%。本机 GRF 的 `doc/observation.md` 将0定义为 `e_GameMode_Normal`。

这些可见事实不支持将后期停滞简单等同于球出界或固定球模式。源 JSONL 未保存 `StepSnapshot.done/info/steps_left`，不能据此判定回合是否结束、内部 AI 状态或引擎停止原因。诊断报告分别保存球速度、球位置/跨度、出界比例、game_mode、比分、持球者及未导出状态的限制；未据此修改 GRF。
