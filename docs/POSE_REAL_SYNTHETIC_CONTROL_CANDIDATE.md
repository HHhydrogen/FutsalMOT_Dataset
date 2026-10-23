# COCO真实位置监督与新合成Pose对照：候选协议

状态：2026-10-23候选协议与CPU工程准备，未冻结、未下载COCO、未开展研究数据训练、未改已有实验协议。独立JSONL图像/LetterBox/严格xy Trainer的自绘toy更新与续载已完成；正式COCO评价器和Pose-aware Tracker尚未验收。本文不代表已有AssA提升。

## 当前真实开发边界

只读检查了现有sim2real协议、研究目录顶层清单和来源元数据。当前真实训练是SNMOT-060/game4、097/game6各150帧；151/game9为原开发段，152/153也是game9的开发扩展。三段联合450帧只能称同一比赛开发验证，不能当三场独立比赛。116/187维持封存，本次没有打开其图像或标签文件。已有bbox协议、R5结果和固定ByteTrack配置均不改写。

研究目录public/datasets顶层尚未出现COCO数据目录；这不是遍历全磁盘得出的数据不存在证明。未取得COCO清单、未核验可用样本数量，因此现在不能填写真实image_id列表或写已冻结manifest。现有v1八seed的Pose关闭，绝不能把其RGB/bbox冒充带Pose训练集；新的S必须来自独立验收后的Pose输出。

## 为什么普通17×2标签不够

[COCO官方格式](https://github.com/cocodataset/cocodataset.github.io/blob/master/dataset/format-data.htm)规定v0未标注、v1已标但遮挡、v2已标且可见。v0的原始坐标为零；位置训练应使用v>0，而不是仅v2。保留原始v与坐标作来源证据。

本机Ultralytics8.4.60 `data/utils.py:verify_image_label`对17×2只用负坐标判断无效，会将(0,0)自动变成有效。用负数哨兵也会触发标签合法性、增强和边界逻辑，不选这条隐式通道。直接删去所有缺点人物则会丢掉大量部分标注，并把它们从检测目标变成背景。

内部候选记录显式分开：

```json
{
  "annotation_id": 7,
  "source": "coco_real",
  "bbox_xywh_px": [10, 20, 100, 180],
  "xy_normalized": "17×2有限坐标数组，缺失点规范占位0",
  "location_valid": "17个bool，与可见性无关",
  "visibility_state": "17个source状态",
  "source_coco_v": "仅真实COCO保留原始17个0/1/2；合成为null"
}
```

真实v1/v2且位置画内时location_valid=true，v0为false；原始v1继续是occluded，不改成visible。合成有可信世界位置、投影和画内检查即可为true，visibility_state仍可unknown，绝不将其导出成伪COCO v1。坐标占位只便于有限tensor运算，不是位置标签。

使用COCO instances和person_keypoints原始注释，按image_id/annotation_id严格联结，保留选中图片中**全部已标注noncrowd person bbox**。包括没有任何关键点的人：检测cls/bbox正常监督，17个位置mask全false。join缺失/重复、类别或bbox不一致要报错，不按关联后的pose列表重新定义完整人物集合。官方未标注的真实人物不能仅凭本工具证明已覆盖。

COCO crowd不是一个普通人的bbox。首轮候选可在抽样前由元数据剔除含person crowd的整图，单列剔除比例和拥挤度偏差；不根据模型表现筛选。不能只删除crowd注释后照常把那片区域作为负样本。若保留crowd图，须先实现并CPU验证crowd区域内负anchor分类损失忽略、正常person正样本优先的ignore-aware分配/损失；当前原型尚未实现此门槛。

## 已实现的CPU桥接与独立toy图像加载集成

`src/grf_ue_bridge/pose_location_supervision_v2.py`实现真实/合成记录转换、完整人物ID覆盖门禁、目标collate和位置loss桥接。batch始终保存`keypoints[N,17,2]`、独立`location_valid[N,17] bool`、source可见性元数据，模型输出仍17×2。

桥接只在调用本机`v8PoseLoss`时，建立临时`[x,y,location_valid]`目标tensor；这个第三维是内部位置mask，既非训练输出维，也不是COCO visibility标签。复用原目标分配器与位置loss，pred维数2因此不调用visibility BCE。所有缺点仍保留bbox；mask全false的人pose loss为零。当前默认uniform sigma=1/17及各人有效点归一化均沿用本机位置loss，真实/合成两臂必须一致；全缺点实例可能稀释batch位置loss权重，需要记录每batch有标点实例数/有效点数，不按结果切换归一化。

第一阶段纯CPU契约测试验证部分点/全缺点没有无效点梯度，遮挡但已标位置和合成unknown位置能共同整理到一个batch。实际随机初始化的YOLO11n严格适配模型执行2图batch的原生目标分配/损失/backward时，全缺点人物的bbox输出仍有非零梯度，所有缺点的pose输出梯度严格为零，visibility BCE哨兵未触发；这一阶段未执行优化器更新。

后续独立`pose_location_dataset_v2.py`已实现`LocationPoseDataset`读取JSONL和图像，不经YOLO txt的隐式mask生成；自定义collate对location_valid按实例cat。真正Ultralytics loader→loss→严格xy Trainer在自绘横/竖图上完成首次1次SGD更新、checkpoint保存、官方resume后再1次更新。三个toy人物的有效点数8/0/17、全部bbox和unknown可见性元数据均保留，位置权重变化非零、visibility损失/梯度/权重变化均零，279个优化器tensor恢复、EMA计数1→2。当前加固后收据为`Saved/ResearchDiagnostics/pose_location_jsonl_trainer_cpu_a02/receipt.json`，完整范围见[JSONL CPU Dataset说明](POSE_LOCATION_JSONL_CPU_DATASET.md)。这不是COCO或正式合成集上的研究训练。

当前实际Dataset关闭Mosaic、MixUp、CopyPaste和随机裁剪，只启用确定性中心LetterBox。图像、xy、bbox和mask同步已做toy验收，四种长宽/取整下图像数组与本机原生LetterBox逐位相同；原false不恢复，画外新设false并零占位。纯函数另测镜像COCO17联合置换，但实际Dataset没有启用镜像。若以后加裁剪或其他增强，仍须对xy、location_valid、source元数据同步变换/置换，bbox丢弃/顺序变化时按annotation_id同删同排，并新增对应验收。已有toy链路不替代真实数据完整性和正式评价门禁。

## 候选R/RS预算，不是已冻结清单

以下数值仅供root选择；不能据此自动下载、训练或修改现有协议。

| 项目 | 候选定义 |
| --- | --- |
| R固定子集 | COCO train2017目标1000幅；至少有一名person含v>0，允许部分标点，保留图内全部person框；crowd规则先定 |
| S固定子集 | 新Pose批次目标1000幅；来源组、相机、时间、天气/外观分层，整seed同split，包含真实17点/投影/完整person bbox/同步证据 |
| 选择 | 元数据资格筛选后，以固定salt+image_id或源frame_key的SHA排序选样，不看模型结果；取得数据后才冻结精确ID、源SHA和排除统计 |
| 初始化 | 同一已核验官方yolo11n-pose.pt、同一严格xy适配、同一hidden51及物理final51、同一loss桥接 |
| 训练seed | 候选20261024/25/26三次配对；root确认后再冻结 |
| 固定更新 | 候选1000次实际更新、batch=nbs=4、无warmup/early-stop改变预算、评最终last、不选best |
| R | 每次4个真实：4000真实样本曝光 |
| RS | 每次3个相同配对真实槽+1合成：3000真实+1000合成曝光，合成25% |
| 固定其余 | SGD/lr日程、weight_decay、尺寸640、增强、AMP策略、梯度累积、优化器重建/续载策略完全相同 |

主问题是**相同计算/更新预算下合成能否替代部分真实重复曝光而有收益**。两臂拥有同一真实不同图像/标注集合，但真实曝光次数不同，必须报告，不能声称同时控制了真实曝光。第一1000个配对槽的精确顺序、重复次数、真实不同图像覆盖和合成不同图像覆盖需冻结；若1000更新无法覆盖预定全部R，应在跑前调整采样表，不能训练后重写。不要把添加合成而额外增加更新的结果当成相同步数效应。需要固定真实曝光的另一研究问题应另登记额外计算控制。

官方Pose模型已用COCO训练：这不是“从未见过这1000张标注”的低标注训练设定，而是COCO预训练后固定子集的继续训练。不能声称总真实标注预算仅1000。COCO train不含跟踪ID，因此本阶段提供位置表征监督，不能直接声称从COCO学到了身份关联；关联器应固定而非在这两个臂各自训练调参。

新S还必须单列COCO手工关节与源骨中心/表面锚点的语义差异、bbox尺寸/扩框策略、人物像素高度、点缺失密度和遮挡位置比例。不能为了提高训练数把未验动画/投影的静态候选加入S；旧近景诊断几帧也不是S。两臂框策略与位置loss权重要先定，不在AP下降后隐式扩框。

## 固定检测与Pose关联接口

现有ByteTrack不使用Pose，单纯训练Pose头并不会让它使用姿态。真实跟踪主实验要固定同一检测器、检测图像、NMS/分数/类别、检测框与分数清单SHA和顺序；两模型只替换预测Pose来源。不得各用自己的YOLO框/置信度作为Tracker输入。

一种可落地的附着方案：按同一固定检测框裁人，经过同一固定上下文与LetterBox后分别运行R/RS Pose模型，用跑前固定的candidate-to-crop选择规则（例如最高IoU，固定IoU门槛及deterministic tie-break）附着到原detection_id，并逆变换到**整图像素空间**。模型内部框只用于选择Pose候选，不能替换跟踪检测框/分数。匹配失败/多义/位置越界按同一规则标无Pose并回退bbox-only，记录两臂附着覆盖率；这仍需实现和实际smoke，不是当前已有能力。不可用GT框/GT Pose为真实推理补缺。

候选接口`pose_association_contract_v2.py`只接收`source=model_prediction_xy`、COCO17整图xy、由预先固定有限性/相对检测框规则计算的`geometry_valid`，不接收GT可见性。身份包含sequence/camera/frame、模型SHA和固定检测集SHA；跨帧顺序、相机、模型、检测清单错配拒绝。geometry_valid只表示预测的几何可用性，不能解释成可见性置信度。

候选成本先以Kalman预测中心位移补偿上一匹配帧的预测关节：`p'=p+(当前Kalman中心-上次检测框中心)`。对共同几何有效关节求`C_pose=1-mean(exp(-d²/(2*(2sigma)²*A)))`，A为两框平均面积。暂给出躯干/四肢索引5..16、至少6点和sigma=.1作为代码默认的**未冻结候选参数**，不称标准COCO OKS评价，也不代表最好参数。

固定既有Kalman、运动/IoU gating、候选配对集合、ByteTrack高低分阶段及阈值，仅在既有gating内使用`C=(1-lambda)*C_bbox+lambda*C_pose`；缺Pose时回退原C_bbox，不能给缺Pose一个有利的零成本。lambda=0必须精确复现原bbox-only分支，作为必要对照。root以后选择并冻结lambda/关节集/sigma/最少点数/附着规则；可以在开发集选一次，但需要记录所试有限候选和选择依据，此后R/RS统一使用，不按训练seed或片段重调。当前CPU成本函数不是完整Tracker，也未证明lambda=0追踪输出逐位重现。

建议最小报告矩阵：同一固定detections上的bbox-only(lambda0)、R Pose+固定lambda、RS Pose+相同lambda，三个训练seed配对；额外官方pretrained Pose可作为固定参考。若没有Pose-aware tracker实际接口与同输入验收，暂时只能报告位置模型工程进展，不能用bbox-only AssA为Pose论文故事背书。

## 开发指标与最终封存

primary建议为151/152/153同game9联合TrackEval COMBINED_SEQ的RS相对R之AssA配对变化；同时逐段完整报告。候选“可定义为开发进步”门槛：三seed平均AssA至少+0.5pp、至少2/3配对seed为正，平均HOTA不下降、IDF1最多下降0.5pp。具体门槛尚待root选择冻结，不由当前工具自动激活。报告全部seed、DetA、HOTA、AssA、IDF1、IDSW、Pose附着覆盖率及失败率，不能只筛有利片段/seed。

COCO val2017的真实位置质量可作为secondary sanity check，按原始v>0评价，不能把SoccerNet没有真实Pose标签的问题隐藏掉。[COCO官方评估器](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py)会忽略num_keypoints=0目标进行关键点评分，这不等于训练应删除其检测框。无可见性头的预测格式和score规则仍需单独固定并核验官方评价接口。

反复使用game9开发会产生选择偏差，三个训练seed不能代替三场独立比赛。最终116/187继续封存，既不用于位置质量/关联门槛设计，也不在本候选工具里路由；最终一次评测须由root在独立冻结后授权。当前虽有独立JSONL加载/LetterBox/Trainer的CPU toy收据，但没有新COCO冻结清单、新S正式Pose集、正式缺点/ignore评价器或Pose-aware跟踪器验收收据，因此尚不能启动正式这轮研究训练，更不能声称合成已有收益。

## 本次证据与下一门槛

第一阶段11项契约测试覆盖部分/全缺点、合成unknown独立位置、全人物bbox覆盖、crowd拒绝、同batch有效性、内部mask不改变原数据、无效点无梯度、实际随机模型全缺点人物检测梯度、位移补偿、缺Pose回退及GT/错配拒绝；这些测试不读取真实数据或运行优化器。一次宽泛discover误选旧pytest测试，研究环境未装pytest导致2个import错误；明确指定两个新unittest模块后通过，未安装或升级环境。后续加入独立图像Dataset及严格xy迁移/夹具回归，当前相关28项unittest已全部通过；另外的自绘toy完整Trainer smoke执行了前述实际SGD更新与resume，不能混称所有测试都未发生优化器更新。

独立图像Dataset、确定性LetterBox、实际混合batch位置loss/更新/续载的CPU toy门槛已完成；当前Trainer刻意只允许CPU toy配置，验证为loss-only，不输出AP。进入正式对照前仍需：root选定候选参数；取得官方COCO元数据并冻结合规子集/源SHA；正式缺点/crowd-ignore与评价器验收；新S通过逐角色动画/投影/同步/人物完整性门禁；固定检测Pose附着与lambda0原追踪复现；安全关闭UE后再按统一预算启动GPU。若扩大增强、设备/优化器或数据范围，须对应新增兼容验收。这里只完成CPU协议与工程准备，不更改当前开发/生产任务，也不冻结样本量、比例或成功阈值。
