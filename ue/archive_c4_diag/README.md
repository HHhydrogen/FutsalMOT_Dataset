# archive_c4_diag（C4 诊断/中间尝试归档）

本目录存放 **非正式 pipeline** 的一次性诊断、API 探针、中间实现与被替代的修复脚本。
**不要**把它们当作 C4 正式流程的一部分运行；正式 C4 管线位于 `ue/` 根目录的
`build_*` / `mrq_*` / `tag_players_c4.py` / `setup_c4_level.py` / `read_sg_c4.py` /
`export_sg_c4.py` / `build_coco17_c4.py` / `overlay_coco17_c4.py` / `c4_overlay_check.py`。

## 内容分类

- **一次性诊断**（`diag_*` / `phase8_*`）：Phase A 资产反射、socket 节点结构、Phase 8 单帧
  投影 trace 等，均为定位问题时的一次性输出，问题已解决。
- **API 探针**（`probe_*`）：`create_node_from_name` 探测等，已被可靠的
  `add_call_function_node("<GeneratedClass>:Function")` 软路径方案替代。
- **中间尝试**（`build_bp_recorder_c4a/c4b`）：SampleActor 函数化方案，被正式版
  `build_bp_recorder_c4.py` 替代。
- **已被正式实现替代的修复脚本**：
  - `fix_socket_pins_c4.py`：一次性外科补 `InSocketName`，已并入 `build_bp_recorder_c4.py`。
  - `fix_camera_json_c4.py`：一次性把 camera.json 内参改为 1920×1080。**C5 必修项要求
    camera calibration 按任务真实输出分辨率自动生成 fx/fy/cx/cy，不再依赖本脚本。**

保留原因：保留历史以便回溯，若确认不再需要可整体删除本目录。