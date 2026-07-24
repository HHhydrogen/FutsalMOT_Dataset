# FutsalMOT Dataset — GRF Reboot

## 项目目标

使用 Google Research Football 和后续 GRF_MARL 预训练策略生成足球比赛轨迹，将轨迹导出为独立的 GRF-UE Episode 格式，再由 Unreal Engine 读取和回放。

## 当前阶段

GRF → JSONL → Unreal Engine 最小回放验证。

## 当前不包含

- 旧规则轨迹生成器
- A3.3
- 自研行为克隆
- 自研 PPO
- 事件系统
- MOT 标注生成
- 批量 episode
- GRF_MARL 预训练策略接入

## 旧版本

旧管线已冻结在：

- Tag：`legacy-a33-final-a9b657e`
- Branch：`archive/legacy-a33-a9b657e`
