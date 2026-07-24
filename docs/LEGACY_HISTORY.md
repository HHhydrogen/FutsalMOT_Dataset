# 旧版本说明 — FutsalMOT Legacy Pipeline

## 冻结信息

| 项目 | 值 |
|------|-----|
| 冻结 commit | `a9b657eaa7fe5f2920f149ed93b1b17c5b7d071c` |
| Tag | `legacy-a33-final-a9b657e` |
| Archive branch | `archive/legacy-a33-a9b657e` |
| 冻结日期 | 2026-07-24 |

## 删除的主要模块

- **`futsalmot/`** — 规则轨迹生成器管线（core、pipeline、scripts、UE 桥接）
- **`futsalmot_rl/`** — 自研 BC/PPO 强化学习管线（academy、benchmark、commands、data readers、models、wrappers）
- **`tools/`** — 旧版 CLI 入口脚本（rl_01~rl_07）
- **`scripts/`** — BC/PPO 训练脚本、PowerShell 冒烟测试
- **`configs/`** — 管线配置和运行产物
- **`tests/`** — 旧版测试套件

## 旧系统曾完成的能力

- 4v4 无守门员五人制足球规则轨迹生成（基于预设路径和简单物理）
- A3.3 标注格式定义和输出
- 生成轨迹到 Unreal Engine 渲染管线的完整桥接
- 三维空间标注（tight bbox、骨骼关键点、场地关键点）
- 事件/帧状态标注、球权信息
- 自研行为克隆（BC）训练和评估
- 自研 PPO 强化学习训练和评估
- A3.3 格式 RL 轨迹导出
- UE 回放渲染和布局检查
- Smoke check 和集成测试

## 更换轨迹来源的原因

旧规则轨迹生成器使用硬编码路径和简单物理，轨迹多样性和真实性有限。使用 Google Research Football (GRF) 及其预训练 MARL 策略可以获得更真实、更多样的足球比赛轨迹，且无需手动编写运动规则。

## 如何查看旧版本

在仓库根目录使用 PowerShell：

```powershell
git switch --detach legacy-a33-final-a9b657e
```

查看后可切换回当前分支：

```powershell
git switch grf-reboot
```
