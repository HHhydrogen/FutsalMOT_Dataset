# 固定足球检测与Pose辅助关联候选

本入口只登记既有开发片段并准备原始检测缓存，不训练、不运行UE、不接触封存测试116/187。正式冻结须由主agent在lambda=0等价、预测Pose接口及几何规则就绪后签署。本候选不是已经观察到的模型提升。

## 现有来源的可用边界

开发151来自`public/soccernet_tracking/train`，152/153来自`public/development_extension_v1/train`。三者均game9，源seqinfo各750帧、1920×1080、25FPS，本候选严格使用帧1–150，共450帧；帧序不拼接，每序列重置Tracker。比赛内片段及三个训练seed不能当作独立比赛或统计显著性证据。

`pipeline.track`以`YOLO.track(conf=.1,iou=.7,imgsz=640,classes=[0])`调用ByteTrack，只写有track ID的结果，bbox被格式化成4位小数。旧`tracking/*/data`缺少被丢弃/未匹配的检测，因此不能恢复完整原始输入，不能用于固定检测对照。旧R4/R5结果只作为历史背景，新缓存基线不与其数值直接比较。

本机三份官方目录`det/det.txt`分别10626、15950、10079行；全文`(frame,bbox)`多重集合与相应全部GT精确相同，score全部1。因此它们在本研究中标为`gt_bbox_oracle_do_not_use`，不进入检测主线；不根据文件名将其当成独立检测器输出。

主agent指定现有官方COCO预训练`weights/yolo11n.pt`，实际SHA256为`0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1`。缓存时还检查80类、class0为person，排除项目微调player头。该检测器固定，与初始化/R/RS三个Pose臂独立，不随Pose模型或其开发结果改变。

派生GT复用`round4_combined_eval/datasets/real_player_pilot_v1/mot_gt`，登记时逐文本验证其框和ID集合等于源GT的既有过滤：gameinfo轨迹描述以`player `或`goalkeeper`开头，帧1–150。保持原框，派生mark/class/visibility均为1，TrackEval `DO_PREPROC=False`。这是player/goalkeeper派生开发任务，不是官方全类别benchmark。非player检测不借GT删除，其FP由评测自然计入。

## 两步执行与不覆盖

第一步不导入Torch/Ultralytics，只读JSON/INI/MOT文本、检测权重SHA及图像路径存在性。图像不解码、不重扫媒体哈希。输出目录必须全新且无联接：

```powershell
.venv/Scripts/python.exe experiments/sim2real/prepare_pose_association_dev.py prepare --output D:/projects/FutsalMOT_UEDataset/Saved/Pose17Development/association_dev_candidate_a01
```

已经生成的a01含`source_manifest.json`和`protocol_candidate.json`，不能再次覆盖。第二步仅在UE安全退出、当前GPU训练结束且主agent签限定准入后运行。推理必须用研究Python3.10环境，不能在P1安装或升级Torch。

准入JSON最小字段：

```json
{
  "schema": "futsalmot_fixed_detection_cache_admission_v1",
  "inference_authorized": true,
  "source_manifest_sha256": "由主agent绑定实际文件",
  "protocol_candidate_sha256": "由主agent绑定实际文件",
  "detector_sha256": "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1",
  "test_split_used": false
}
```

工具不能自行产出此授权。缓存命令必须显式`export --execute-inference --device cpu`或`--device 0`，另指定`--source-manifest`、`--protocol`、`--admission`及全新`--output`。现在未运行推理。缓存一次解码450张源图，不训练、不追踪、不读取Pose/GT决定检测。固定conf=.1、IoU=.7、imgsz640、classes0、max_det300、rect=true、half/augment/agnostic_nms=false、逐帧batch1及Ultralytics8.4.60；设备与运行库版本进入完成收据。失败保留partial，不伪造完成manifest，也不隐式重命名或续写。

## Matcher与Pose附件的接口

`detections.jsonl`一行一帧，严格151/152/153、各1–150顺序，空检测帧同样保留。`frame_number`为MOT的1基，`frame_index`为0基；`camera_id`固定`broadcast`；`frame_key`例如`SNMOT-151:000001`。`image_file/width/height`指原1920×1080图像。

检测列表每项包含`detection_id`（例如`SNMOT-151:000001:000000`）、`source_index`、`class_id:0`、`bbox_xywh_px`、`score`。ID仅源模型输出行号，不取GT/Tracker ID；保留native float序列化值，不做小数位舍入、不删未关联框、不按Pose置信度二次筛选。

`cache_manifest.json`的schema为`futsalmot_fixed_detection_cache_manifest_v1`，完成状态`complete_fixed_raw_detections`；绑定`detections_jsonl:{path,sha256,frames:450}`、`source_manifest:{path,sha256}`、detector身份、推理配置/设备/库版本、准入SHA，并明确`tracking_applied/pose_applied/GT_used_in_prediction/test_split_used`均为false。整个完成manifest的SHA即`fixed_detection_set_sha256`，不循环写进JSONL本身。

纯CPU消费者`load_fixed_cache(manifest_path,expected_manifest_sha256)`返回`(manifest,frames)`，检查完整身份、JSONL SHA、450帧顺序、检测ID/数值/像素域，保持原数值，不打开媒体。Pose附件必须额外绑定该缓存SHA、Pose模型SHA及原检测ID，保留固定bbox与score；几何有效性只来自预测，不借GT visibility或GT框筛选。主线采用整帧640预测；候选匹配/几何阈值尚待接口验证后冻结。

## 候选比较和验收

先用相同数组输入验证lambda=0标准ByteTrack等价，包括空帧、低分、多阶段匹配、丢失与恢复。不能拿旧`model.track`的截断缓存作为等价参照。只有此门槛和Pose附件错配拒绝通过，主agent才能冻结唯一lambda与整帧匹配/几何规则。

Pose成本候选复用`pose_association_contract_v2`：COCO关节5–16、至少6个共同几何有效点、sigma=.1、最多30帧间隔；缺Pose退回bbox。lambda候选0/.1/.25仍未冻结，不可看RS-R结果后只汇报最有利值。检测阈值、Kalman、ByteTrack高低分/新轨迹阈值、buffer及其他关联设置全部共享。

新主指标为三序列TrackEval COMBINED_SEQ的AssA，附HOTA、DetA、IDF1、MOTA、IDSW及每片段/seed完整结果。候选进步门槛为RS−R AssA均值至少+.5pp、至少2/3训练seed正向，HOTA均值不降、IDF1均值下降不超.5pp，正式采用前仍需主agent冻结。共享检测意味着AP不变，关联收益不能写成检测AP改善。保留负向变化，不重新选择片段、模型或lambda；开发收益不等于封存测试、跨比赛或真实五人制结论。
