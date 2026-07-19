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

## 第 1 步：Windows 生成轨迹

先修改总配置文件：

```text
configs/pipeline_config.json
```

这个总配置只保留少量关键参数：

```json
{
  "seed": 1,
  "template_id": 1,
  "max_attempts": 10,
  "timeout_sec": 300,
  "strict_warnings": false,
  "skip_trajectory_validation": false,
  "allow_trajectory_errors": false,
  "update_current_pointer": true,
  "run_id_prefix": "run"
}
```

默认运行：

```powershell
py .\01_generate_trajectories.py
```

如果需要临时覆盖，也可以传命令行参数：

```powershell
py .\01_generate_trajectories.py --seed 1 --template 1
```

可用模板：

| ID | 内容 |
|---:|---|
| 1 | 4v4 单人带球射门，其他队员进行宽度支援、锚点保护和盯防 |
| 2 | 4v4 带球-传球-接球，包含接应跑位和防守跟随 |
| 3 | 4v4 传球-接球-带球-射门，包含弱侧支援和纵深保护 |

这一阶段会依次执行：

```text
生成事件回合
→ 验证事件
→ 编译密集轨迹
→ 增强 yaw / action / ball state / contact 信息
→ 验证密集轨迹
→ 生成事件与逐帧状态标注
```

输出会进入唯一 run 目录：

```text
configs/runs/<run_id>/
```

示例：

```text
configs/runs/run_20260719_120102_seed0001_t1/
```

其中会保存：

```text
<seq_id>.json
<seq_id>_a32.json
<seq_id>_a33.json
event_annotations/
pipeline_run_report.json
```

如果后续进入第 2 步，MRQ 的图片会保存到：

```text
Saved/FutsalMOT/images_clean/<seq_id>/cam_01/
Saved/FutsalMOT/images_clean/<seq_id>/cam_02/
Saved/FutsalMOT/images_clean/<seq_id>/cam_03/
Saved/FutsalMOT/images_clean/<seq_id>/cam_04/
```

图片命名格式为 6 位补零帧号：

```text
Saved/FutsalMOT/images_clean/<seq_id>/cam_01/000000.png
Saved/FutsalMOT/images_clean/<seq_id>/cam_01/000001.png
...
Saved/FutsalMOT/images_clean/<seq_id>/cam_01/000299.png
```

其他相机同理：`cam_02/000000.png` 到 `cam_02/000299.png`，`cam_03/000000.png` 到 `cam_03/000299.png`，`cam_04/000000.png` 到 `cam_04/000299.png`。

也就是从第 1 步生成出来的同一个 `seq_id` 目录去接第 2 步渲染结果。

如果 `update_current_pointer=true`，还会更新：

```text
configs/pipeline_current.json
```

## 第 2 步：在 Unreal Editor 中运行

在 Unreal Editor 的 Python 控制台中执行：

```python
py "D:/projects/FustalMOT_UEDataset/Content/FutsalMOT/code/02_run_unreal.py"
```

这个步骤会先做只读 preflight，再构建 Sequencer 并导出标注。

核心输出：

```text
Saved/FutsalMOT/annotations/objects_bbox_2d_clean_<seq_id>.json
Saved/FutsalMOT/annotations/objects_bbox_2d_clean_<seq_id>.jsonl
```

对每个球员对象，标注中会包含：

```text
bbox_2d_clean
bbox_xyxy_clean
keypoints_2d
keypoints_2d_yolo
```

`keypoints_2d_yolo` 采用 YOLO pose 风格的扁平格式：

```text
x_norm, y_norm, visibility, x_norm, y_norm, visibility, ...
```

visibility 定义：

```text
0 = 缺失或在相机后方
1 = 在相机前方但不在图像内
2 = 在图像内
```

## 第 3 步：Windows 检查标注

MRQ 渲染完成后，执行：

```powershell
py .\03_check_labels.py --annotation "D:/projects/FustalMOT_UEDataset/Saved/FutsalMOT/annotations/objects_bbox_2d_clean_<seq_id>.json"
```

如果当前 PowerShell 位于 UE 项目根目录 `D:\projects\FustalMOT_UEDataset`，请使用完整脚本路径：

```powershell
py "D:/projects/FustalMOT_UEDataset/Content/FutsalMOT/code/03_check_labels.py" --annotation "D:/projects/FustalMOT_UEDataset/Saved/FutsalMOT/annotations/objects_bbox_2d_clean_<seq_id>.json"
```

如果要在 overlay 上画出关键点：

```powershell
py .\03_check_labels.py --draw-keypoints
```

预期输出：

```text
Saved/FutsalMOT/overlay_objects_bbox_<seq_id>/
Saved/FutsalMOT/labels_yolo_clean/<seq_id>/
Saved/FutsalMOT/labels_mot_clean/<seq_id>/
Saved/FutsalMOT/annotations/manifest_<seq_id>.json
```

预期检查结果：

```text
records = 1200
yolo_files = 1200
expected_objects_per_record = 9
CHECK PASSED
ALL DONE
```

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
