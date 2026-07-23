# UE Local Steps — RL 轨迹渲染

> 以下步骤需要在 Unreal Editor 中手动完成。

## 前提

- RL A3.3 文件已导出：`Saved/FutsalMOT_RL/exported_a33/rl_episode_random_0001_t1_Player_05_a33.json`
- 该文件中 Player_05 的轨迹由 PPO 策略生成，其余球员和球保持规则回放
- seq_id: `rl_episode_random_0001_t1_p05`

## 步骤 1：在 UE Python 控制台执行

打开 Unreal Editor，在 Python Console 输入：

```python
py "D:/projects/FustalMOT_UEDataset/Content/FutsalMOT/code/rl_07b_ue_render_rl.py"
```

这个脚本内置了 RL A3.3 文件路径，不需要预先设置环境变量。

**如果 `py` 命令不可用**，改用：

```python
import unreal
unreal.SystemLibrary.execute_console_command(None, 'py "D:/projects/FustalMOT_UEDataset/Content/FutsalMOT/code/rl_07b_ue_render_rl.py"')
```

预期输出：
- Config 加载信息（指向 RL A3.3 文件）
- Preflight 检查通过
- Level Sequence 构建完成
- 标注导出成功

## 步骤 2：Movie Render Queue 渲染

| 设置 | 值 |
|------|-----|
| Output Directory | `D:\projects\FustalMOT_UEDataset\Saved\FutsalMOT_RL\ue_closed_loop\images\rl_episode_random_0001_t1_p05` |
| File Name Format | `{frame_number}` |
| Image Format | PNG |
| Resolution | 1920 × 1080 |

## 步骤 3：Windows 后处理

```cmd
cd /d D:\projects\FustalMOT_UEDataset\Content\FutsalMOT\code
python 03_check_labels.py --annotation "D:\projects\FustalMOT_UEDataset\Saved\FutsalMOT\annotations\objects_bbox_2d_clean_rl_episode_random_0001_t1_p05.json" --step 5
```

## 步骤 4：验证

```cmd
python rl_07_validate_ue_closed_loop.py --check --seq-id rl_episode_random_0001_t1_p05
```

## 检查清单

- [ ] RL A3.3 配置文件可被 UE 读取
- [ ] Player_05 按 RL 轨迹移动，无跳帧
- [ ] 其余球员保持规则回放
- [ ] bbox 标注正确
- [ ] 渲染图像完整（4 相机 × 300 帧 = 1200 张）
- [ ] 未覆盖原始规则数据
