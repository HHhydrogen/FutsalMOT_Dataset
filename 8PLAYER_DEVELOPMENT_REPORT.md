# FutsalMOT 8 人扩展与移动逻辑开发报告

## 目标

将当前 2v2 场上球员扩展为 4v4 无守门员，并在不破坏既有 A3.1–A3.3c 管线接口的前提下，改善防守跟随、无球跑位和低速带球视觉稳定性。

## 主要改动

### 数据模型

- 球员：`Player_01`–`Player_08`
- 球：`Ball_01`
- 队伍：A/B 各 4 名场上球员
- 守门员：空列表
- Track ID：1–8 和 101
- 每帧对象数：9

### 随机回合生成器

`11_generate_random_episode.py`

- 新增 4v4 角色阵型；
- 三个模板均补充无球支援和四人防线；
- 初始位置执行全体最小距离约束；
- 防守者按目标球员的 goal-side marking lane 初始化；
- seed + attempt 继续使用确定性派生 RNG。

### 轨迹编译器

`12_compile_trajectory.py`

- `defend_follow` 新增 `positioning=goal_side`；
- 未来帧目标预测；
- 速度和加速度上限；
- 制动速度约束，减少越过目标后反向；
- 局部分离力同时考虑被盯防球员和已编译球员；
- 输出保留 team、role、roster 和 movement metadata。

### A3.3 增强

`13_enhance_trajectory.py`

- 移除纵向正弦回摆；
- 横向/垂直带球视觉量随球员速度缩放；
- 防止低速起步时足球局部倒退并产生人工 180°转向。

### 轨迹验证

`14_validate_trajectory.py`

- 仍严格检查玩家速度、转向和间距；
- 足球在 pass/receive/shot contact frame 的大转向降级为可审计 WARNING；
- 非接触帧大转向仍为 ERROR。

### UE

- `20_build_sequences.py`：动态支持 9 个对象并导出 team/role；
- `21_preflight.py`：动态配置读取，当前阶段要求 8 名球员；
- `23_ue_setup_8_players.py`：幂等创建 Player_05–08，不自动保存关卡。

### 事件标注

`31_generate_event_annotations.py`

- 每帧输出 8 名球员 action；
- 支持 `target=possession_owner` 的动态选择器；
- 输出 roster 和 movement metadata。

## Windows 测试结果

最终版本基准测试：

| seed | template | 结果 | accepted attempt |
|---:|---:|---|---:|
| 1 | 1 | PASS | 1 |
| 1 | 2 | PASS | 1 |
| 1 | 3 | PASS | 1 |
| 2 | 1 | PASS | 1（早期最终移动模型） |
| 2 | 2 | PASS | 1（早期最终移动模型） |
| 2 | 3 | PASS | 1 |
| 3 | 1 | PASS | 1（早期最终移动模型） |
| 3 | 2 | PASS | 1（早期最终移动模型） |
| 3 | 3 | PASS，允许确定性重试 | 2 |
| 4 | 2 | PASS | 1 |

结构验收：

```text
players = 8
objects = 9
frames = 300
keyframes per object = 300
player_actions per frame = 8
event annotation status = PASS
```

## 未在本环境执行的内容

本环境没有 Unreal Editor，因此以下文件仅完成语法和静态检查：

- `20_build_sequences.py`
- `21_preflight.py`
- `22_scan_animations.py`
- `23_ue_setup_8_players.py`

第一次 UE 运行后应重点检查：

1. `Player_05`–`Player_08` 是否正确复制骨架和材质；
2. 9 个对象是否均进入 Sequence 和 bbox；
3. 每条 bbox record 是否包含 9 个对象；
4. 第二次运行是否幂等；
5. MRQ overlay 中防守移动是否符合预期。
