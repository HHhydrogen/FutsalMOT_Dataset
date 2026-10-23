# 固定检测的Pose辅助ByteTrack候选

`pose_bytetrack_v2.py`与`experiments/sim2real/run_pose_association_dev.py`提供最小可执行CPU关联入口。只消费独立raw检测缓存与模型Pose附件，不运行检测/Pose推理，不打开图像或GT，不修改当前训练/评价/协议。真实缓存与附件仍由各自入口在root授权后生成。

## 基线与有限范围

固定输入是`prepare_pose_association_dev.load_fixed_cache`核验的SNMOT-151/152/153各150帧，包含空帧、全部raw检测及原始float框/分数和稳定detection_id。旧model.track输出只保留有ID轨迹且已做小数舍入，不能反推raw检测或作为lambda0数值基线。官方det文件被发现与GT框复制相符，也不能作为推理缓存。

lambda0基线是**同一新固定raw数组**直接送入本机Ultralytics8.4.60 BYTETracker，参数与`bytetrack_fixed.yaml`相同。候选lambda=0完全绕过Pose读取和混合算术、使用原STrack类型，自动逐值比较每帧输出及tracked/lost/removed的ID、mean/covariance、score、状态和生命周期。代码、上游ByteTrack源码、LAP版本、tracker配置及固定检测集SHA绑定到一次lambda0收据，后续Pose臂复用该证据，无需每模型重复基线。

这不声称与历史各自模型的端到端model.track结果逐位相同；检测来源已经不同。真实dev的lambda0验收尚待缓存完成后root执行；当前证据是实际ByteTrack状态机上的纯CPU数值夹具。

## Pose成本与回退

只覆盖上游`get_dists`的高分第一关联和unconfirmed关联，分别保留原.8/.7门槛；低分第二阶段继续原bbox路径。上游Kalman、分数融合、分阶段过滤、丢失/重激活和删除状态机均复用，不复制整个update、不猴子补丁全局matching函数。

只在原bbox成本不超过该阶段门槛的边上混合`(1-lambda)*C_bbox+lambda*C_pose`；门槛外成本原样保留，Pose不能复活原本不可行的候选。Pose代价沿用已发布接口：上一匹配预测关节加Kalman当前中心相对上一原检测中心位移，再求尺度归一的平均相似性；这不是COCO官方OKS评价。

缺附件、附件失败、共同有效点不足或超过最大间隔均退回原bbox成本，不能给缺点优惠的零成本。新匹配没有Pose时清除旧Pose，避免把陈旧点当成当前观测。

附件必须逐detection_id、原bbox、sequence/camera/frame、模型SHA与固定检测集SHA精确绑定，只接收`source=model_prediction_xy`、`coordinate_space=image_pixels`及`validity_method=finite_bbox_relative_geometry`。有效点再次核对有限性、画内及固定相对框范围；geometry mask不是物理可见性，不读取GT Pose或unknown当visible。整帧附件生成与纯校验由`fixed_detection_pose_attach_v2`提供，模型内部框只做一对一附着，不能改变Tracker原框/分数。

目前候选bbox_margin=.25、肢体5..16、最少6点、sigma=.1/max_gap30，以及lambda0/.1/.25仅登记为待root冻结参数；不自动扫参，不按R/RS结果或片段换lambda。

## CPU消费入口

真实缓存完成后先运行一次lambda0：

```powershell
D:/projects/FutsalMOT/.venv/Scripts/python.exe experiments/sim2real/run_pose_association_dev.py `
  --cache-manifest <cache_manifest.json> --cache-sha256 <实际SHA> `
  --protocol <protocol_candidate.json> --protocol-sha256 <实际SHA> `
  --pose-lambda 0 --output <全新研究pose或Saved子目录>
```

非零Pose臂另加`--attachment-manifest`、`--attachment-sha256`和`--lambda0-receipt`。收据/配置/450帧附件必须与原缓存一致，逐帧复用附件纯校验；模型推理不会在此入口启动。

输出`data/SNMOT-151.txt`等三份10列MOT和`receipt.json`，坐标不主动截断成4位小数。未完成时不写最终收据。既有本机LAP位于`D:/projects/FutsalMOT_Research/runtime_dependencies`，与旧pipeline相同；入口只加此路径，不自动安装依赖。CLI固定CUDA不可见；直接Python API调用仍只做CPU NumPy关联，不加载模型。

TrackEval主指标预定AssA，并报告HOTA/DetA/IDF1/MOTA/IDSW及附着覆盖率；不能由本工具的结构/等价收据宣称改善。三个片段均来自game9且已用于开发，三训练seed不是三场比赛；正式lambda、质量规则、进步阈值及最终test授权仍由root独立冻结，116/187保持封存。

## 定向证据

`test_pose_bytetrack_v2.py`和`test_run_pose_association_dev.py`本次12项CPU通过：真实ByteTrack库的lambda0成本矩阵/轨迹/Kalman状态逐值一致；缺附件和全false mask在非零lambda仍逐值退回bbox；包含空帧、低分、丢失重激活、unconfirmed路径；GT/伪visibility/误ID/越界拒绝。另用450帧自绘数值检测完成三序列MOT写出合同，不使用真实媒体/GT/检查点或GPU，也不计算HOTA/AssA。
