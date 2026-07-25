# Mannequin Unarmed Animations

路径：`/Game/FootballMovesPack/Demo/Mannequins/Anims/Unarmed/`

适用骨骼：UE Mannequin（SK_Mannequin）

## Idle

| 文件名 | 大小 | 说明 |
|--------|------|------|
| `MM_Idle` | 1.4 MB | 空手站立 Idle，男性体型 |

## Walk（走）

`Walk/` 目录，共 8 个动画，MF 前缀 = 女性体型

| 文件名 | 大小 | 方向 |
|--------|------|------|
| `MF_Unarmed_Walk_Fwd` | 476 KB | ✅ **向前走** |
| `MF_Unarmed_Walk_Fwd_Left` | 412 KB | 前左 45° |
| `MF_Unarmed_Walk_Fwd_Right` | 412 KB | 前右 45° |
| `MF_Unarmed_Walk_Left` | 487 KB | 左平移（strafe） |
| `MF_Unarmed_Walk_Right` | 492 KB | 右平移（strafe） |
| `MF_Unarmed_Walk_Bwd` | 607 KB | 后退 |
| `MF_Unarmed_Walk_Bwd_Left` | 538 KB | 后左 45° |
| `MF_Unarmed_Walk_Bwd_Right` | 538 KB | 后右 45° |

## Jog（慢跑/跑）

`Jog/` 目录，共 8 个动画

| 文件名 | 大小 | 方向 |
|--------|------|------|
| `MF_Unarmed_Jog_Fwd` | 532 KB | ✅ **向前跑** |
| `MF_Unarmed_Jog_Fwd_Left` | 467 KB | 前左 45° |
| `MF_Unarmed_Jog_Fwd_Right` | 468 KB | 前右 45° |
| `MF_Unarmed_Jog_Left` | 488 KB | 左平移（strafe） |
| `MF_Unarmed_Jog_Right` | 522 KB | 右平移（strafe） |
| `MF_Unarmed_Jog_Bwd` | 484 KB | 后退 |
| `MF_Unarmed_Jog_Bwd_Left` | 418 KB | 后左 45° |
| `MF_Unarmed_Jog_Bwd_Right` | 418 KB | 后右 45° |

## 其他

| 文件名 | 大小 | 类型 | 说明 |
|--------|------|------|------|
| `BS_Idle_Walk_Run` | 54 KB | Blend Space | Idle→Walk→Run 混合空间 |
| `ABP_Unarmed` | 394 KB | Animation Blueprint | 空手动画蓝图 |

## 当前配置建议

如 `ue/animation_config.local.json` 使用：

```json
"animations": {
  "idle": "/Game/FootballMovesPack/Demo/Mannequins/Anims/Unarmed/MM_Idle",
  "walk": "/Game/FootballMovesPack/Demo/Mannequins/Anims/Unarmed/Walk/MF_Unarmed_Walk_Fwd",
  "run": "/Game/FootballMovesPack/Demo/Mannequins/Anims/Unarmed/Jog/MF_Unarmed_Jog_Fwd"
}
```

所有动画均为 **In-Place（原地）** 动画，不包含 Root Motion，适合直接用于 GRF 轨迹驱动的人物位移。
