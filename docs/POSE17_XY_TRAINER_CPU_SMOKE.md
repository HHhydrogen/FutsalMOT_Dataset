# 位置头完整Trainer CPU兼容小样

`tools/smoke_pose_xy_trainer_cpu.py`只验证本机Ultralytics 8.4.60、Torch 2.6.0的单进程CPU兼容性。它从已核验SHA的官方`yolo11n-pose.pt`开始，走`pose_xy_trainer_class()`的严格迁移、实际数据加载器、损失、SGD、验证器、checkpoint保存和官方resume API。它不是研究训练入口。

## 输入与固定边界

工具在全新输出目录中自画4张128×128、明确写有`TOY CPU FIXTURE`的图片，train/val各2张，使用单类39列YOLO位置标签，即bbox与17×2坐标。训练和验证夹具可以相同；这里没有泛化评估。它不读取v1媒体、MRQ诊断帧、真实数据或封存测试集。所有fixture文件SHA记录在收据中。

必须显式CPU，`CUDA_VISIBLE_DEVICES=-1`、workers=0、amp=false、plots=false、compile=false。禁用网络集成和无图场景的字体检查，并以socket连接拦截器阻止网络；尝试记录留在收据中。使用batch=2、nbs=2、单线程、SGD、lr0=0.001、momentum=0.9、weight_decay=0、关闭warmup和增强。未覆盖AdamW、GPU/AMP、DDP或完整研究数据性能。

## 何谓通过

声明2个epoch，但首次在epoch0保存后停止，因此恰好1次优化器更新；随后以epoch0的未剥离checkpoint调用真正的`resume`，从epoch1开始，再完成1次更新。首次及续载后均断言：

- 三尺度隐藏容量和最终物理卷积输出保持51，暴露的关键点输出是17×2。
- 位置输出行梯度非零、参数实际变化；visibility行梯度为零，且在零weight decay下不变化。
- visibility BCE安装一旦调用便失败的哨兵；实际`kobj_loss=0`。
- 所有参数有限，完整Trainer验证、final_eval正常返回。
- 续载前，优化器全部保存tensor与加载后tensor逐项比较；允许官方保存FP16到优化器dtype的明确转换，EMA更新计数同时恢复。

当前本机首次收据位于`Saved/ResearchDiagnostics/pose_xy_trainer_cpu_a01/receipt.json`。279个SGD momentum tensor恢复相等，EMA更新计数1→2，首次/续载位置参数变化量均大于零，网络尝试为空。运行日志为同级`pose_xy_trainer_cpu_a01.log`。配置可直接读取该目录`trainer_run/args.yaml`；续载会复用同一run目录，首阶段未剥离checkpoint也保留了其原始train_args。

## 保存/续载限制

官方8.4.60 `save_model()`保存EMA半精度模型和FP16优化器状态；`final_eval()`会从last/best剥离优化器。工具因此在`on_model_save`时保留`initial_one_update_unstripped.pt`与`resume_one_update_unstripped.pt`，不会从已strip的文件声称恢复了状态。源官方checkpoint不修改，输出目录必须不存在。

这一结果证明该环境的续载API和下一步计算兼容，不证明中断与不中断训练轨迹逐位一致。EMA/FP16量化与随机状态恢复均可能影响轨迹，正式对照应固定相同初始化、优化器、更新预算及中断策略。日志中的toy AP没有研究意义，不能据其声称模型进步。

17×2使用本机默认位置损失的uniform sigma（1/17），并非17×3标准COCO sigma配置。真实与真实加合成两臂必须使用相同17×2损失配置；保留预训练网络容量不代表自动保留COCO损失定义。未知可见性仍然未知，不训练visibility head，也不能将位置标签称为COCO v1/v2可见性标签。

## 运行

```powershell
D:/projects/FutsalMOT/.venv/Scripts/python.exe -X utf8 tools/smoke_pose_xy_trainer_cpu.py --output D:/projects/FutsalMOT_UEDataset/Saved/ResearchDiagnostics/pose_xy_trainer_cpu_a02
```

完整兼容小样已经实际运行，不需要为文档检查重复运行；仅修改计算或续载路径后再开新attempt。夹具契约测试`tests/test_pose_xy_trainer_smoke_contract.py`不启动训练。
