# JSONL独立位置监督：图像加载与CPU Trainer兼容验证

状态：2026-10-23，独立CPU toy工程验收。未下载或读取真实训练图像、未读取封存test、未启动GPU、未冻结样本量/真实合成比例/成功门槛，未改变现有协议。本文补充此前候选协议中尚未完成的图像加载门槛，不把工程检查称为合成收益。

## 实现边界

`src/grf_ue_bridge/pose_location_dataset_v2.py`新增：

- `LocationPoseDataset`：读取独立JSONL与图像、核对尺寸/图像SHA/人物ID覆盖、保留所有人物bbox；不会先转换为YOLO txt。
- `letterbox_geometry`和`transform_location_instance`：同步图像几何、bbox、xy、location_valid；原false点永不恢复，画外点新设false并零占位，元数据明确记录原因。
- 自定义`collate_fn`：图像stack；xy、独立bool mask、bbox、cls按实例cat；人物ID、source可见性状态同步保留。
- `location_pose_trainer_class()`：继承既有`pose_xy_trainer_class()`，保持严格输出适配、hidden51/physical final51，只重载数据/损失/验证相关入口，不修改已验共享函数。

首次只支持确定性中心LetterBox，图像使用OpenCV INTER_LINEAR与114边框。四组横/竖与奇数尺寸测试实际对比本机Ultralytics原生`LetterBox`，图像数组逐位相同。xy使用同一比例与整数padding，false点不被padding平移成有效点。通用仿射纯函数另外测试了画外mask、bbox裁剪、完全画外bbox返回None及镜像COCO17联合置换；**当前实际Dataset没有启用随机仿射或镜像**。LetterBox不会删除已合法的原图bbox，若发生则报错。后续若启用裁剪须同步记录被删人物ID并核验背景语义，不能静默复用当前覆盖声明。

当前候选Trainer在初始化前就要求显式`device='cpu'`，拒绝GPU/自动设备；要求workers0、amp=false、plots=false，禁用未实现的几何/颜色/自定义增强和multi-scale、rect/cache/compile/fraction/类别加权。只接受`usage_scope=cpu_toy_compatibility_only`的独立配置。这是刻意设置的研究训练门禁，不能通过把真实数据标成toy来绕过。root以后另行设计、冻结并验收研究入口。

## 数据格式

配置是JSON，不走Ultralytics的自动下载YAML检查：

```json
{
  "schema": "pose_location_jsonl_cpu_toy_v1",
  "usage_scope": "cpu_toy_compatibility_only",
  "train": "train.jsonl",
  "val": "val.jsonl"
}
```

每行图像记录包含`schema=separated_pose_location_frame_v1`、frame_key、image_file（相对JSONL或绝对路径）、image_sha256、原图width/height、`bbox_coverage=all_noncrowd_person_instances`、`crowd_present=false`、expected_person_annotation_ids和instances。实例使用`pose_location_supervision_v2.py`的转换结果：pixel bbox、17×2归一化xy、独立17 bool location_valid、source可见性元数据。

真实COCO必须保留source_coco_v及其v0/1/2语义：v0不能标有效，也不能在LetterBox后重新出现；num_keypoints=0人物照常保留bbox和cls。合成source_coco_v必须null；unknown可见性和有效位置可以同时存在，outside/behind_camera/missing不能同时声称位置有效。本文toy使用这些**来源语义夹具**，不声称自绘图像来自COCO或UE。

crowd仍fail-closed，需要独立ignore-aware实现或跑前元数据图级筛选；重复/遗漏人物ID、坏SHA、尺寸不符、非法坐标和状态冲突均拒绝。图像SHA全读只在当前微型toy兼容路径使用，正式规模若复用冻结校验收据应另设明确定义，不能不记录就静默跳过。

## Loader到损失/续载的真实验证

新增工具`tools/smoke_pose_location_jsonl_cpu.py`只创建两个写有TOY字样的自绘图，原图分别192×96和96×192。train/val引用同一toy图片用于API兼容，不做泛化评价；没有任何正式/真实媒体输入。

一个batch有2图3人：真实源语义夹具8个已标点（含遮挡点）、真实源语义全缺点人物、合成源语义17个unknown可见性且位置有效点。实际Loader输出shape `[2,3,128,128]`；3个人物bbox均在，位置有效点数分别8/0/17；全部无效xy保持零占位。

先从已核验官方`yolo11n-pose.pt`严格适配，完整Trainer完成一次SGD更新，保存epoch0未剥离checkpoint，再用官方resume API恢复并完成epoch1一次更新。权重衰减为0，便于严格验证无visibility监督行不变化。当前加固后收据：

`Saved/ResearchDiagnostics/pose_location_jsonl_trainer_cpu_a02/receipt.json`

实际结果：

| 检查 | 首次 | resume后 |
| --- | ---: | ---: |
| 实际优化器更新数 | 1 | 1 |
| hidden/physical final通道 | 51/51 | 51/51 |
| xy权重绝对变化和 | 0.0063099442 | 0.0072606939 |
| visibility梯度/权重变化和 | 0/0 | 0/0 |
| kobj loss | 0 | 0 |
| EMA更新计数 | 1 | 2 |

279个优化器tensor已逐项与checkpoint保存值（显式dtype转换后）比较恢复相等；原始checkpoint保持SHA不变，网络attempts为空。收据记录Python/Torch/Ultralytics版本、完整配置、fixture SHA和实现SHA。a01首次收据保留，a02对应初始化CPU门禁、状态冲突拒绝等加固后的当前实现，不覆盖旧结果。两次使用的toy坐标/像素和计算结果一致；这不是正式数据重复实验。

新6项Dataset/几何测试以及此前严格xy/缺点/关联契约，共28项unittest实际通过。实际随机模型loss测试还证明：全缺点人物bbox仍有非零梯度，缺点pose输出梯度严格为零。完整toy smoke中安装visibility BCE失败哨兵，未触发。

## 为什么不输出AP

默认PoseValidator会把GT第三维、全缺点人物与检测目标共享的统计混在一起；不能直接拿内部location mask冒充COCO可见性标签或默认AP分母。本候选使用明确的`LocationLossOnlyValidator`，计算相同mask桥接的检测/位置loss，fitness只用于Trainer保存接口；不报告AP、不选择研究最优模型。

`final_eval`显式记录`loss_only_cpu_compatibility_no_AP_no_optimizer_strip`，保留可续载优化器。没有宣称本次走过默认AP/final strip流程。此前独立严格xy Trainer smoke已验证默认Trainer路径；本次验证新Dataset和独立mask路径，两者门槛不同。

官方保存仍使用EMA半精度模型和FP16优化器状态；resume成功仅证明当前API/下一步兼容，不证明中断与不中断轨迹逐位相同。当前只验CPU、单线程、SGD、微型batch；GPU/AMP/DDP、多worker、AdamW、正式COCO评估、完整crowd ignore、采样预算与Pose关联实验均未由本次验证覆盖。

## 独立复现

只有计算/契约有变化时才需新attempt，不为文档收尾重复优化器小样：

```powershell
D:/projects/FutsalMOT/.venv/Scripts/python.exe -X utf8 tools/smoke_pose_location_jsonl_cpu.py --output D:/projects/FutsalMOT_UEDataset/Saved/ResearchDiagnostics/pose_location_jsonl_trainer_cpu_a03
```

工具拒绝已有输出目录、阻止socket网络、固定CPU、只读取指定官方本地权重和自绘fixture。它不会下载图像或自动选择GPU，也不会调用UE或Git。
