# FutsalMOT 4v4 无守门员数据集生成管线

版本：`8PLAYER_PIPELINE_V1`

本版本将场上对象从 **4 名球员 + 1 个球** 扩展为：

- Team A：`Player_01`–`Player_04`
- Team B：`Player_05`–`Player_08`
- 足球：`Ball_01`
- 守门员：无
- 每帧目标数：9
- Track ID：球员 1–8，足球 101

当前回合仍为 10 秒、30 FPS、300 帧、4 个固定相机。

---

## 0. 安装

在解压目录中运行：

```powershell
powershell -ExecutionPolicy Bypass -File .\INSTALL_8PLAYER_PIPELINE.ps1
```

默认安装到：

```text
D:\projects\FustalMOT_UEDataset\Content\FutsalMOT\code
```

安装器会备份被替换的同名文件，但不会修改：

```text
configs/
Saved/
关卡
材质
动画资产
```

---

## 1. 一次性扩展 UE 场景到 8 人

### 1.1 运行前

打开原来包含以下 Actor 的关卡：

```text
Player_01
Player_02
Player_03
Player_04
Ball_01
CineCam_01
CineCam_02
CineCam_03
CineCam_04
```

### 1.2 在 Unreal Editor Python 控制台运行

```python
py "D:/projects/FustalMOT_UEDataset/Content/FutsalMOT/code/futsalmot/scripts/ue_setup_8_players.py"
```

脚本会：

- 以 `Player_01`–`Player_04` 为模板；
- 创建缺失的 `Player_05`–`Player_08`；
- 已存在的 Actor 自动跳过；
- 不创建守门员；
- 不自动保存关卡；
- 失败时回滚本次新建的 Actor。

报告位置：

```text
Content/FutsalMOT/code/_agent_test_outputs/ue_setup_8_players_report.json
```

确认报告为 `PASS`，并在场景中检查 8 名球员后，**手动保存当前关卡**。

---

## 2. Windows：从 seed 生成 8 人回合

```powershell
cd D:\projects\FustalMOT_UEDataset\Content\FutsalMOT\code
py .\01_generate_trajectories.py --seed 1 --template 1
```

模板：

| ID | 内容 |
|---:|---|
| 1 | 4v4 单人带球射门，其他队员进行宽度支援、锚点保护和盯防 |
| 2 | 4v4 带球—传球—接球，包含接应跑位和防守跟随 |
| 3 | 4v4 传球—接球—带球—射门，包含弱侧支援和纵深保护 |

Windows 主链路：

```text
futsalmot/scripts/generate_random_episode.py
→ futsalmot/scripts/validate_episode.py
→ futsalmot/scripts/compile_trajectory.py
→ futsalmot/scripts/enhance_trajectory.py
→ futsalmot/scripts/validate_trajectory.py
→ futsalmot/scripts/generate_event_annotations.py
```

成功后应看到：

```text
[OK] Windows pipeline complete
Accepted attempt: N
```

生成的 A3.3 配置必须满足：

```text
players = 8
objects = 9
frames = 300
每个对象 keyframes = 300
每帧 player_actions = 8
```

当前 episode 指针：

```text
Content/FutsalMOT/code/configs/pipeline_current.json
```

---

## 3. 球员移动优化说明

### 3.1 初始阵型

生成器使用 4v4 角色阵型：

Team A：

- `Player_01`：持球推进
- `Player_02`：接应/第二持球点
- `Player_03`：弱侧宽度
- `Player_04`：后方锚点

Team B：

- `Player_05`：第一盯防人
- `Player_06`：接球队员盯防
- `Player_07`：弱侧盯防
- `Player_08`：纵深保护

防守球员直接初始化在各自的 **goal-side marking lane**，避免先站得过深，再穿过进攻球员去寻找盯防位置。

### 3.2 防守移动

`defend_follow` 使用：

- 目标未来位置预测；
- goal-side 纵向偏移；
- 侧向盯防偏移；
- 最大速度限制；
- 最大加速度限制；
- 制动距离限制；
- 局部球员分离力；
- 确定性计算，不使用运行时随机状态。

### 3.3 带球视觉运动

已移除会导致足球沿前进方向来回倒退的纵向正弦摆动。

当前只保留：

- 与球员速度关联的轻微横向触球；
- 与球员速度关联的轻微垂直起伏；
- 低速和起步阶段自动衰减。

传球、接球、射门接触帧允许足球发生较大方向变化，但会保留为可审计 WARNING，不再误判为普通轨迹 ERROR。

---

## 4. UE Preflight + 构建 Sequencer 与导出 bbox

在 Unreal Editor Python 控制台运行：

```python
py "D:/projects/FustalMOT_UEDataset/Content/FutsalMOT/code/02_run_unreal.py"
```

`02_run_unreal.py` 会先执行 read-only preflight，再构建 Sequencer 并导出 bbox 标注。

必须检查：

```text
Player_01–Player_08 全部存在
Ball_01 存在
CineCam_01–CineCam_04 全部存在
8 名球员骨架兼容
配置 objects = 9
每对象 300 keyframes
每名球员有完整 yaw 和 action_timeline
```

首次构建前 Level Sequence 不存在可显示 WARNING，但 Actor 缺失属于 ERROR。

预期：

```text
OBJECT_COUNT = 9
Records: 1200
Expected records: 1200
```

说明：

- bbox record 数仍为 `4 相机 × 300 帧 = 1200`；
- 每条 record 现在应包含 9 个对象，而不是 5 个；
- 现有 Idle/Jog fallback 继续使用；
- 足球专项动画仍为 HOLD。

---

## 5. MRQ 渲染

每个相机渲染 300 帧：

| 参数 | 值 |
|---|---|
| Start Frame | 0 |
| End Frame | 299 |
| Resolution | 1920 × 1080 |
| Output | PNG |
| Warm Up | 0 |

输出目录：

```text
Saved/FutsalMOT/images_clean/<seq_id>/cam_01/000000.png ... 000299.png
Saved/FutsalMOT/images_clean/<seq_id>/cam_02/000000.png ... 000299.png
Saved/FutsalMOT/images_clean/<seq_id>/cam_03/000000.png ... 000299.png
Saved/FutsalMOT/images_clean/<seq_id>/cam_04/000000.png ... 000299.png
```

总图像数仍为 1200。

---

## 6. 后处理 YOLO / MOT / Overlay

```powershell
py .\03_check_labels.py --annotation "D:/projects/FustalMOT_UEDataset/Saved/FutsalMOT/annotations/objects_bbox_2d_clean_<seq_id>.json"
```

预期核心结果：

```text
records = 1200
yolo_files = 1200
expected_objects_per_record = 9
CHECK PASSED
ALL DONE
```

人工检查时至少查看两个相机和所有触球关键帧，确认：

- 8 名球员和足球 bbox 均存在；
- Track ID 不重复；
- 防守球员没有穿过被盯防球员；
- 没有明显瞬移、180°抖动或身体重叠；
- 图像与标注逐帧一致。

---

## 7. 回滚

安装器会创建：

```text
Content/FutsalMOT/code/_backup_8player_pipeline_<timestamp>/
```

需要回滚时，将该目录中的文件复制回 `code/`。

`futsalmot/scripts/ue_setup_8_players.py` 不自动保存关卡。如果尚未手动保存，只需关闭关卡并放弃修改；如果已经保存，需要手动删除 `Player_05`–`Player_08` 或恢复关卡备份。

---

## 8. 当前验证范围

已完成：

- Python 语法检查；
- 3 个模板完整 Windows 链路；
- 8 players / 9 objects / 300 frames 检查；
- 事件标注 8 个 `player_actions` 检查；
- 多 seed smoke test；
- 确定性重试逻辑保留。

仍需用户在项目内完成：

- `futsalmot/scripts/ue_setup_8_players.py` 的首次 UE 运行；
- `02_run_unreal.py` 的 8 人现场检查和 Sequencer/bbox 验证；
- MRQ 1200 图像和 9-object overlay 验收。
