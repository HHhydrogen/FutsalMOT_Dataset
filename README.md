# FutsalMOT 数据集代码

这是 FutsalMOT Unreal Engine 合成数据集管线的代码仓库，存放位置约定为：

```text
Content/FutsalMOT/code
```

仓库只包含 Python 代码和少量配置，不包含完整 UE 工程资产、渲染图片或最终数据集产物。

## 当前范围

当前管线生成的是 4v4 无守门员五人制足球回合：

- A 队：`Player_01` 到 `Player_04`
- B 队：`Player_05` 到 `Player_08`
- 足球：`Ball_01`
- 每帧对象数：9
- 相机数：4 个固定 CineCamera
- 时间轴：10 秒，30 FPS，300 帧
- 预期标注记录数：`4 * 300 = 1200`

管线会导出同步的 RGB 元数据、tight bbox、动作时间轴、事件/帧状态标注、球权信息，以及球员骨骼 2D 关键点。

## 公开入口

根目录只保留三个公开 `.py` 文件：

```text
01_generate_trajectories.py
02_run_unreal.py
03_check_labels.py
```

真正的实现脚本都在 `futsalmot/scripts/` 下，属于内部实现。

## 工作流

### 第 1 步 — Windows 生成轨迹

修改 `configs/pipeline_config.json` 中的 `seed` 等参数，然后运行：

```powershell
cd D:\projects\FustalMOT_UEDataset\Content\FutsalMOT\code
py .\01_generate_trajectories.py
```

运行完成后会打印：
- **MRQ Output Directory**：UE 渲染时要填的路径
- **File Name Format**：`{frame_number}`
- **第 2 步 UE 命令**：需要在 UE Python 控制台完整粘贴运行
- **第 3 步 Windows 命令**：渲染完成后在终端运行

所有生成的 JSON 都会存入 `configs/runs/<唯一run_id>/`。

### 第 2 步 — UE 渲染

1. 在 UE Python 控制台运行第 1 步打印的 UE 命令
2. 打开 Movie Render Queue，按第 1 步打印的信息设置：
   - **Output Directory** = 第 1 步打印的路径
   - **File Name Format** = `{frame_number}`
   - **Image Format** = PNG
   - **Resolution** = 1920 × 1080
3. 渲染

### 第 3 步 — Windows 布局检查

渲染完成后，在终端运行第 1 步打印的第 3 步命令（已包含 `--step 5 --draw-keypoints`）。

输出到 `Saved/FutsalMOT/layout_check/<seq_id>/`，每 5 帧绘制 1 帧。

每张布局检查图包含：
- 所有目标的 bounding box
- 球员骨骼关键点（黄色圆点）
- 场地 41 个关键点（红色圆点 + 名称）
- 场地边界线（蓝色线条）

## 8 人场景初始化

如果 UE 场景里还只有 `Player_01` 到 `Player_04`，可以在 Unreal Editor Python 控制台中运行一次：

```python
py "D:/projects/FustalMOT_UEDataset/Content/FutsalMOT/code/futsalmot/scripts/ue_setup_8_players.py"
```

它会从已有球员模板补齐 `Player_05` 到 `Player_08`，不会自动保存关卡。

## 目录结构

```text
code/
├─ 01_generate_trajectories.py
├─ 02_run_unreal.py
├─ 03_check_labels.py
├─ README.md
├─ configs/
│  ├─ pipeline_config.json
│  ├─ pipeline_current.json
│  └─ runs/
├─ futsalmot/
│  ├─ core/
│  │  ├─ paths.py
│  │  ├─ io.py
│  │  ├─ hashing.py
│  │  └─ process.py
│  ├─ pipeline/
│  │  └─ constants.py
│  ├─ scripts/          (内部实现)
│  └─ ue/
│  ├─ core/
│  ├─ pipeline/
│  ├─ scripts/
│  └─ ue/
└─ pyproject.toml
```

## 主要模块

- `futsalmot/core/paths.py`：统一代码、配置、项目和输出路径。
- `futsalmot/core/io.py`：原子 JSON / 文本读写。
- `futsalmot/core/process.py`：带日志的子进程执行工具。
- `futsalmot/pipeline/constants.py`：模板名和内部脚本路径。
- `futsalmot/scripts/`：三个公开入口背后的内部实现。

## 当前验证状态

代码层面的 smoke check 已通过：

```text
compileall: PASS
episode validation: PASS, warnings=0, errors=0
trajectory validation: WARNING, warnings=60, errors=0
```

这些轨迹 warning 是当前基线的一部分，不是错误。

UE 运行结果必须在 Unreal Editor 中实际验证，因为普通 Windows Python 不提供 `unreal` 模块。
