# P3-2 Valid Trajectory Cohort

## Goal

从预先确定的 candidate seeds 中筛选 Motion Quality 合格的 trajectory，供后续 P2-8 Camera Distribution Stability Validation 使用。

本阶段只进行 trajectory generation 和现有 Motion Quality Audit。cohort selection 在任何 Camera A/B 实验之前完成，不读取 Camera、P01 或 UE 结果。

## Candidate Seeds

固定 candidate set：

```text
42, 43, 44, 45, 46, 47, 48, 49,
50, 51, 52, 53, 54, 55, 56, 57, 58, 59
```

统一设置：

```text
scenario: 5_vs_5
frames: 300
FPS: 10
field: 40m x 20m
seed policy: futsalmot_seed_v1
```

所有 seed 独立生成，未执行 retry、自动重采样或 seed-specific exception。

## Motion Quality Results

判定使用现有 Motion Quality 语义：

- 外场 active ratio `>= 0.75`；
- 外场最长 stationary streak `<= 2.0s`；
- team active outfield coverage `>= 0.90`；
- GK stationary streak 作为现有 Audit 指标记录，不改变历史 strict cohort 纳入语义。

| Seed | Min Active Ratio | Max Outfield Stationary | Max GK Stationary | Team Active Coverage | Longest Low-Motion Plateau | Decision |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 42 | 0.390 | 13.5s | 22.0s | 0.390 | 15.3s | invalid |
| 43 | 0.587 | 2.6s | 13.4s | 0.760 | 1.3s | invalid |
| 44 | 0.763 | 1.1s | 12.2s | 0.913 | 0.7s | valid |
| 45 | 0.727 | 1.3s | 11.6s | 0.890 | 0.8s | invalid |
| 46 | 0.713 | 1.9s | 20.8s | 0.910 | 0.4s | invalid |
| 47 | 0.590 | 5.2s | 20.5s | 0.630 | 8.4s | invalid |
| 48 | 0.797 | 1.3s | 4.8s | 0.840 | 1.5s | invalid |
| 49 | 0.727 | 1.9s | 12.4s | 0.880 | 0.5s | invalid |
| 50 | 0.357 | 10.5s | 23.3s | 0.297 | 18.1s | invalid |
| 51 | 0.697 | 2.2s | 14.1s | 0.853 | 0.7s | invalid |
| 52 | 0.677 | 0.9s | 20.5s | 0.887 | 0.9s | invalid |
| 53 | 0.777 | 2.4s | 16.7s | 0.890 | 0.6s | invalid |
| 54 | 0.797 | 1.1s | 8.3s | 0.883 | 0.8s | invalid |
| 55 | 0.697 | 2.1s | 14.9s | 0.790 | 0.9s | invalid |
| 56 | 0.330 | 10.9s | 23.5s | 0.290 | 18.6s | invalid |
| 57 | 0.787 | 2.1s | 10.6s | 0.827 | 0.8s | invalid |
| 58 | 0.573 | 8.9s | 14.9s | 0.627 | 9.2s | invalid |
| 59 | 0.650 | 2.0s | 12.6s | 0.790 | 1.8s | invalid |

每个 candidate 的 `frame_count` 均为 `300`。

## Valid Cohort

```text
total candidates: 18
strict-valid count: 1
invalid count: 17
valid ratio: 1 / 18 = 5.6%
strict-valid seeds: [44]
```

只有 seed `44` 同时满足当前 strict cohort 纳入条件。Seed `44` 的 GK stationary streak 为 `12.2s`，但这与 P2-8 已实际采用的 valid decision 一致：GK 指标被记录，而没有作为历史 strict cohort 的阻断条件。没有新增或放宽阈值。

## Cohort Selection Rule

确定性选择规则为：

1. 只保留 Motion Quality strict-valid 的 seed；
2. 按 seed 数字升序排序；
3. 选择排序后的前 `5` 个 seed；
4. 如果 strict-valid 少于 `5`，返回 `INSUFFICIENT`，不选择部分 cohort，不扩大 candidate 范围。

本次排序结果只有 `[44]`，因此无法形成目标大小为 5 的 cohort：

```text
selection status: INSUFFICIENT
selected P2-8 cohort: none
```

候选筛选在 Camera A/B 之前完成，且 selection 只读取 trajectory Motion Quality 结果。没有根据 P01 visibility、Camera annotation、UE 或 MRQ 结果反向挑选 seed。

## Determinism Check

按要求重复生成一个 strict-valid seed 和一个 strict-invalid seed：

| Seed | Status | First frames hash | Repeat frames hash | Identical |
| ---: | --- | --- | --- | --- |
| 44 | strict-valid | `62e0703cb78e2a572d47088d330911d7c31be92761c6f7f05774929226484f57` | `62e0703cb78e2a572d47088d330911d7c6f7f05774929226484f57` | yes |
| 42 | strict-invalid | `9aef1323c3866beac9dcc4bde41dfe01345d2f96c0089db58feef547dd59afe3` | `9aef1323c3866beac9dcc4bde41dfe01345d2f96c0089db58feef547dd59afe3` | yes |

两次重复生成都得到 `300` 帧，且 `frames.jsonl` 字节级 SHA-256 完全一致。没有修改 root seed contract 或 `futsalmot_seed_v1` 派生规则。

## Limitations

- cohort selection 只过滤 trajectory quality，不表示 GRF generator 已经稳定。
- invalid seed 没有被修复，也没有通过 retry 或自动换 seed 隐藏失败。
- 18 个 candidate 中只有 1 个 strict-valid，样本量不足以说明 generator 的普遍质量。
- valid cohort 不代表 P01 已经跨 seed 稳定有效。
- 本阶段没有运行 UE annotation、MRQ、Camera Distribution 或 P01 分析。
- 本阶段没有扩大 candidate seed 范围，也没有修改 generator 或 Motion Quality Gate。
- `.futsalmot/p3_2_valid_trajectory_cohort/` 下的 trajectory、重复生成和 JSON 结果是本地验证产物，不应提交。

## Final Decision

1. **Seeds 42-59 中有多少 strict-valid trajectory？**

   `1 / 18`，只有 seed `44`。

2. **是否达到至少 5 个 valid seeds？**

   没有。结果为 `1 < 5`，因此 P3-2 状态为：

   ```text
   INSUFFICIENT
   ```

3. **最终冻结的 P2-8 cohort 是哪些 seeds？**

   没有冻结可用 cohort。按照预定规则，valid 少于 5 时不选择部分 cohort；seed `44` 只能记录为唯一 strict-valid candidate，不能冒充满足目标的 cohort。

4. **cohort 是否在 Camera A/B 之前确定？**

   是。筛选只基于 trajectory generation 和现有 Motion Quality Audit，未读取或运行 Camera A/B、P01、UE 或 MRQ。

5. **是否保持 deterministic？**

   是。Seed `44` 和 strict-invalid seed `42` 的重复 `frames.jsonl` 均字节级一致，未修改 seed derivation contract。

6. **是否可以重新启动 P2-8？**

   不可以。由于没有达到至少 5 个 strict-valid seeds，本阶段不能提供满足要求的 valid trajectory cohort。按任务结束条件停止，不扩大 seed 范围、不修改 gate、不修改 generator、不进入 P2-8。
