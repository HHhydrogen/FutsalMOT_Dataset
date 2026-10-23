# COCO位置监督子集准备

本工具只进行CPU数据准备，不包含训练或UE调用。当前1000幅train2017和200幅val2017是固定图像ID候选；训练协议仍待root冻结，不能据此启动研究训练。SNMOT-116/187不在工具输入或路由中，继续封存。

官方来源为[COCO下载页](https://cocodataset.org/#download)及[下载页原始文件](https://github.com/cocodataset/cocodataset.github.io/blob/master/dataset/download.htm)。只下载官方annotations_trainval2017.zip，解出instances/person_keypoints的train/val四份JSON。逐图下载选中的1200个JPEG，不下载18GB train ZIP或完整val ZIP。官网images域名在本机HTTPS证书不匹配；使用相同官方S3桶的path-style HTTPS并保持TLS验证，记录实际URL；不关闭证书验证，也不从第三方镜像补标注。本地SHA是下载观测摘要，不冒称官方发布摘要。

已只读查找本项目和Research目录，未找到COCO2017标注或图像缓存，未遍历全磁盘。若已有缓存可传`--local-archive`、`--reuse-images`；它们必须是普通路径而非目录联接。本机外部数据写入需正常权限升级；所有数据写入限定在Research/pose下独立目录。

```powershell
.venv/Scripts/python.exe -B experiments/sim2real/prepare_coco_pose_subset.py annotations
.venv/Scripts/python.exe -B experiments/sim2real/prepare_coco_pose_subset.py select --train-count 1000 --val-count 200 --selection-seed 20261024
.venv/Scripts/python.exe -B experiments/sim2real/prepare_coco_pose_subset.py images --workers 4
```

选择先按image_id/annotation_id严格核对instances和keypoints的完整person集合、bbox、类别和iscrowd。任何联结缺失/重复报错；不把缺失person当背景。实际官方注释的bbox全部精确相等，但area大量存在浮点尾数差异；分别保留keypoints原area和source_instance_area_by_annotation_id，记录差异计数，不以epsilon吞差异，不改官方关键点评估area。先排除含person crowd的整图，再要求至少一人有一个画内v>0关键点，以固定salt、selection_seed、split、image_id的canonical JSON SHA排序取前N幅。crowd图剔除比例明确记录，这个子集不能称全COCO人群分布。v1虽然遮挡仍是已标位置，v0无效且规范零占位；选中图片的全部noncrowd框都保留，包括num_keypoints=0的人。格式依据[COCO官方注释格式](https://github.com/cocodataset/cocodataset.github.io/blob/master/dataset/format-data.htm)。

原始官方注释也含少量零宽/零高或与图像无交集的person框。工具在排序前排除含这类框的整图，记录反例及数量，不只删除一个人物、不猜测补框；入选图仍完整保留全部人物框。

目录：

```text
D:/projects/FutsalMOT_Research/pose/coco2017_location_a01/
  annotations_receipt.json
  annotations/{instances,person_keypoints}_{train2017,val2017}.json
  cache/annotations_trainval2017.zip
  subset/selection_manifest.json
  subset/{train2017,val2017}_person_annotations.json
  subset/{train2017,val2017}.location.jsonl
  images/{train2017,val2017}/<12位image_id>.jpg
  images/{train2017,val2017}/<12位image_id>.jpg.source.json
  images_receipt.json
```

选择清单记录源SHA、ID顺序、资格/排除统计和全缺点人物数。每张图记录传输来源、原coco_url、license ID、尺寸与SHA，读取JPEG verify。位置JSONL直接适配独立LocationPoseDataset，保留source_coco_v与location_valid，不走YOLO txt隐式mask。

已有成功图片只有来源sidecar、SHA和身份全部匹配才复用。输出清单与完成收据不覆盖；下载截断保留.part并停止，不把partial当成功。首轮失败后可继续images动作复用已经有成功sidecar的图片；若失败文件留有.part或图片无sidecar，应先明确核对，不静默覆盖。不得仅删除某一crowd注释后继续把其区域作为负样本。

候选协议为`pose_location_coco_candidate_protocol.json`：同一官方已COCO预训练的YOLO11n-pose检查点，严格保留塔宽的17×2位置适配；SGD、constant lr0=.001、momentum=.9、weight_decay=0、batch=nbs4、1000次实际更新、LetterBox唯一增强、评last、无warmup/AMP/early stop。以上是待root冻结的数值，不修改既有检测/追踪协议。R为4真实/步，共4000真实曝光；RS25为3真实+1合成/步，共3000真实+1000合成。两臂不同真实图片集合相同、更新预算相同，但真实曝光不同，不能声称同时固定。精确采样表仍需在正式训练前冻结并记录所有有效点/人物/前景anchor密度。

官方检查点已在COCO上预训练，因此不能声称总共仅使用1000幅真实标注，公开val也只是开发检查。新合成S需来自独立location pilot验收，未完成前不创建伪S清单；unknown visibility不能变COCO可见标签。COCO关键点评估的xy预测分数适配和正式Trainer尚需独立门槛，本工具准备成功不证明AP或HOTA有提升。COCO没有跟踪ID，MOT收益还须固定detections与Pose-aware关联器验证；封存足球test不会被本工具打开。
