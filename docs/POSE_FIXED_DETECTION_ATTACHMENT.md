# 固定足球检测的Pose附件候选

`experiments/sim2real/attach_fixed_detection_pose.py`只为固定raw检测缓存附加模型预测Pose，不运行关联器，不读取GT选择Pose，不改变任何固定检测框、分数、顺序或ID。计算模块是`src/grf_ue_bridge/fixed_detection_pose_attach_v2.py`。本次只进行了CPU夹具；真实450帧缓存与附件推理仍由主agent单独安排，尚无此工具产生的追踪收益。

## 输入与身份

共用`prepare_pose_association_dev.load_fixed_cache`，要求完整`cache_manifest.json`、调用者明确指定的manifest SHA、450行完整`detections.jsonl`及其SHA。范围仅为SNMOT151/152/153的各1..150帧，`frame_number`一基、`frame_index`零基，camera为broadcast。这三个开发片段均game9，不是三个独立比赛。

每个固定检测的ID为`SNMOT-151:000001:000000`，最后一段是原模型输出序号；必须包含原`source_index`、class0、原图`bbox_xywh_px`和score。旧ByteTrack输出已丢失未关联检测并舍入，不能代替raw输入。现有官方`det.txt`已被独立源审计识别为GT框复制，不能作为本主线固定检测输入。

附件工具复验cache→源登记SHA和全部帧/图像路径对应，不读取源登记中的GT文件。图像只在显式`--execute-inference`时读入，读取字节同时计算SHA后解码一次；没有额外媒体扫描。封存SNMOT116/187在路径stat之前就拒绝。所有输出必须是新目录，失败保留partial与failure，不写完成manifest。

## 整帧推理与匹配

主线仅整帧640中心LetterBox，整数padding、填充114、BGR→RGB/255，与当前位置训练的预处理一致。明确使用CPU或cuda:0，不自动选GPU；GPU前拒绝运行中的UnrealEditor，执行者仍需安排与其他GPU作业串行。没有GT crop或固定框crop。实际模型必须是保留物理51通道再索引的17×2普通Pose；加载调用者钉住SHA的可信本地checkpoint，不从YAML重建、不隐式回退model/best。默认选择实际EMA；官方严格xy参考权重需显式`--model-member model`。已知正式位置对照checkpoint必须完成其计划步数且EMA计数一致。

同一显式配置用于所有模型。`candidate_configuration()`提供尚未冻结的候选：模型person阈值0.001、NMS IoU0.7、最多100个Pose预测；固定框与预测框IoU至少0.3，按IoU降序贪心一对一，同分按固定ID和模型输出序号。它不以预测Pose与GT的吻合程度选人，也不以模型新框替换固定框。贪心匹配不是全局最优，密集相邻球员仍可能误配；必须同时报告匹配失败比例，不将其当作真实身份对应证明。100预测上限是本附件独立设置，与COCO AP入口的20上限不同，不能把两类结果混写。

坐标逆变换使用实际LetterBox比例与整数left/top，记录原图到输入矩阵。模型框截到图像边界；关键点不截到边界。固定框保持原始浮点值，禁止先转xyxy再减回而产生新的舍入值。

## 独立位置有效性

模型只有xy，**没有可用关节confidence**。`geometry_valid`仅表示点有限、画内、位于固定框扩边范围内；候选margin为宽高各0.25。它不代表准确、可见或无遮挡，`visibility_state`始终unknown，`joint_confidence=null`。person score另存为`model_person_score`，不能冒充关节置信度。

至少6个肢体点（COCO索引5..16）通过几何规则才令`prediction_valid=true`。匹配失败、肢体点不足或显式全局门槛失败时，保留原检测并输出17个false/零坐标，供bbox-only回退。原始有限坐标与失败原因另存；不能因零占位恰好落在画内而恢复有效。

配置另有`minimum_bbox_height_input_px`和`minimum_limb_spread_bbox_diagonal_fraction`。当前候选均为0，明确未启用尺度/塌点筛选；不能声称这些质量检查已经过滤异常姿态。若主agent之后启用，必须独立登记配置SHA并对所有arm保持相同，不能按有利结果调门槛。即使启用，几何代理也不是物理visibility或独立精度验收。

## 输出与关联器对接

`attachment_manifest.json`的schema为`futsalmot_fixed_detection_pose_attachment_manifest_v1`，绑定固定cache manifest SHA、checkpoint SHA/member、配置SHA和值、附件JSONL SHA、计数及源码SHA。状态`complete_pose_attachment_not_tracking`仅表示附件完整。

`attachments.jsonl`每帧一行，schema为`futsalmot_fixed_detection_pose_frame_v1`，包含sequence/camera/frame身份和`attachments`列表。每个原固定检测恰好一条；空检测帧也保留。每条附件与`pose_association_contract_v2`兼容：

- `source=model_prediction_xy`、`coordinate_space=image_pixels`、`validity_method=finite_bbox_relative_geometry`；
- 原`detection_id`、`bbox_xywh_px`、`frame_index`及模型/cache SHA；
- 17个有限`xy_px`、独立bool `geometry_valid`和整人`prediction_valid`；
- 模型框匹配IoU、原预测坐标、失败原因及坐标变换。

`validate_attachment_frame`拒绝数量、顺序、框、模型、帧和mask错配，以及伪visibility。无附件/全false时Pose成本应回退bbox。本附件CPU测试包含3帧全false附件接实际Pose ByteTrack后，与原ByteTrack输出、均值、协方差和生命周期逐值相同；这不替代450帧真实缓存的lambda0验收。lambda0还必须在独立关联器中逐行及状态证明等价。固定检测AP在Pose关联比较中应保持不变。

## 执行格式

先把`candidate_configuration()`的JSON显式登记到独立配置文件，后续是否冻结由主agent确定。以下不是实际已执行的450帧实验：

```powershell
& D:/projects/FutsalMOT/.venv/Scripts/python.exe experiments/sim2real/attach_fixed_detection_pose.py `
  --cache-manifest <完整raw-cache>/cache_manifest.json --cache-sha256 <固定cache SHA> `
  --configuration <登记配置.json> --configuration-sha256 <配置SHA> `
  --checkpoint <明确模型>/last.pt --checkpoint-sha256 <权重SHA> --model-member ema `
  --device cpu --output <全新附件目录> --execute-inference
```

CPU测试只使用数值/自绘图及450行元数据夹具，不解码真实足球图像、不读真实GT/test、不使用GPU。
