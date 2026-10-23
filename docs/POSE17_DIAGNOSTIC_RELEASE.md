# 已验联合小样的CPU诊断导出

`src/grf_ue_bridge/tools/export_pose17_diagnostic.py`接收联合通路已经完成的四机位×十帧；仅在显式`location_pilot_contract=true`且共享`validate_location_pilot_contract`通过时，接收冻结四seed之一的120帧。普通诊断仍只允许十帧，不能靠修改frame_count扩容。它不是生产publisher，不修改现有v1，也不把unknown升级成COCO可见性或训练准入。真实导出由主agent显式执行；CPU tests只使用16×8自绘图和假EXR解码器，不能代替真实输出验收。

2026-10-23主agent已实际执行CLI，完成`Saved/Pose17Development/diagnostic_pose17_export_a01`：40张原生4K RGB复制、40张mask、400条人物记录，每份EXR解码一次。随后独立只读CPU核对通过：四相机全部400份world17与本次原始native逐值完全相同；400人物各有独立17bool位置mask，6800点均为unknown且location_valid=true，没有升级可见性；40RGB/40mask相对0..9文件名集合完整。仅抽读每相机相对5（源105）的四张mask，uint8、3840×2160及合法0..10实例ID通过；没有重解码EXR或扫描媒体hash。新鲜收据为`Saved/Pose17Development/diagnostic_pose17_export_a01_cpu_acceptance_a01.json`。这些结果证明本次诊断导出保真，`not_training=true`和正式可见性/训练准入限制均不变。

## 执行

在内层Python3.9环境，输入三份已有来源及全新workspace Saved诊断目录：

```powershell
.venv/Scripts/python.exe -m grf_ue_bridge.tools.export_pose17_diagnostic <joint_attempt> <alignment_report.json> <native_acceptance_report.json> <workspace_Saved/diagnostic_pose17_export_a01> --copy-rgb --workers 6
```

所有路径须在当前外层项目`Saved`中；拒绝符号链接、junction、已有输出、非`diagnostic_`输出名及写入源attempt内部。默认RGB只保留外部Saved来源引用；`--copy-rgb`才逐字节复制PNG，不重编码、不创建硬链接、不缩放。EXR每帧只调用一次已有Cryptomatte decoder，40份EXR生成40张uint8 player实例mask及各自十实体bbox/area。这里是生成所需数据，不额外生成逐角色审计图、重新渲染或重复媒体SHA扫描。

120帧位置试产沿用同一CLI，动态生成每seed480 RGB/480 mask/4800人物记录，解码480份EXR各一次；四seed范围固定为1042的[100,220)、1044的[307,427)、1048的[100,220)、1102的[100,220)。前三组train、1102为synthetic_dev，不能将邻帧随机拆分或改变来源窗口。要求实际RGB/EXR相对帧集合精确为0..119，导出的timeline源帧不重基；group额外保留独立location_pilot契约。十实体/七模型/四机位实际身份与恢复继续走已有联合验证，不重复头点或图像审查。该扩展仍输出`diagnostic_only=true/not_training=true`，并非正式生产发布或GPU训练授权。

CLI默认`--workers 6`，仅允许1..6；Python API `export_attempt(..., workers=1)`默认仍串行。多进程采用Windows spawn，每相机最多workers个在途帧，每帧独立解码一次EXR、保存mask并返回尺寸/bbox/area小结果，父进程按帧序汇总原Pose/world/UV及复制RGB，不跨进程传4K数组。不设置未经验证的OpenEXR线程参数；不追加预览或媒体hash。API若显式使用workers>1，调用脚本也须使用`if __name__ == '__main__'`保护。失败不自动重解码或覆盖已生成mask，保留partial且不写完成dataset.json，使用新目录重试。串并行CPU测试使用自绘16×8真EXR，核对mask字节、bbox/area及结果顺序相同；这不等于真实4K性能收益已经测量。

开始前必须存在通过的结构配对报告和原生事务/恢复报告，二者attempt身份正确且仍标记`MRQ_sync_verified=false`。工具重新核对冻结plan SHA、十份manifest及四份native的实际身份/帧/fence/恢复和Editor前后状态；原始native的世界点/源帧须与已验配对报告完全一致，不接受验收之后被修改的点。尚未成功恢复、streaming需要重启、记录不齐或配对报告失败时不开始输出。

## 布局

```text
diagnostic_pose17_export_a01/
  export_started.json
  dataset.json                         # 仅全部写入完成后创建
  groups/seed_00001048/
    group.json
    timeline.jsonl                     # 10个源时间索引，共享一次
    provenance/
      plan.json, state.json
      preflight.json, editor_after.json
      alignment_report.json
      native_acceptance_report.json
      source_build_receipt.json         # 输入声明时保存
      CineCam_01_native.json ...
      landmarks/<manifest_sha>.json     # 相同manifest只保存一次
    cameras/CineCam_01/
      view.json                         # 原相机静态内外参
      pose3d_world.jsonl
      pose2d.jsonl
      rgb/000000.png ...                # 仅--copy-rgb
      mask/000000.png ...
```

其余三机位相同。group声明`world_storage=per_camera_native`、`cross_camera_world_shared=false`和`world_values_modified=false`；每相机保存原native十实体×17点世界米坐标，不平均、不用epsilon去重、不复用首相机。微米级差异完整保留。每相机`view.json`只保存实际内外参；最小导出遇标定随帧变化会失败并保留partial，不静默舍弃。

`timeline`仅表示真实GRF源frame/FPS时钟，当前relative0..9对应source100..109。各相机原生世界时钟、MRQ cached时钟、engineframe和pose revision分别保留在本机位3D记录，不强行合成一个采样时刻。所有world对象仍保留，即使该视角mask为空；完全不可见对象的visible bbox为null、area为0。

`pose2d.jsonl`每行引用本camera的world帧，包含17个原始投影UV、独立17bool `location_valid`、前向深度和原投影状态。真实有限3D、正深度且画内的位置可以`location_valid=true`同时`visibility_state=unknown`；画外/相机后方的位置为false，不把其UV裁到边缘伪装有效。`world_location_valid`表示已采样有限世界坐标，不代表可见。

mask背景/非球员为0，L0..L4为1..5，R0..R4为6..10。球等非球员不写入player mask。bbox是实例mask可见像素的tight半开xywh，area是可见像素数；它们不是amodal人体面积或关节可见性证据。

输出始终`diagnostic_only=true`、`not_training=true`、`production_accepted=false`。目前只导出RGB/mask/Pose/world/provenance；不写YOLO训练配置、不宣称真实合成收益、不修改研究封存test，也不生成全unknown的COCO pose标签冒充可训练监督。训练JSONL桥接和正式小批发布须另行实现明确入口，不能编辑这些标志绕过。

失败时保留partial和`export_started.json`，没有最终`dataset.json`就不能视为完成；重试使用新目录，不覆盖失败输出。
