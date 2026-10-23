# 真实诊断导出到位置JSONL的CPU工程检查

`check_pose17_location_cpu.py`是独立检查入口，不改旧检测/追踪协议、研究Trainer或UE代码。它读取已经完成的`diagnostic_pose17_export_a01`，生成同样保持`not_training=true`的工程JSONL，通过已发布的`LocationPoseDataset`、确定性LetterBox、严格`map_pose_xy`及`separated_location_loss_class`完成一个实际CPU batch的forward/backward。没有优化器更新、训练循环、AP评价或GPU。

真实数据不能标为toy绕过候选Trainer的限制。本入口不实例化只允许`cpu_toy_compatibility_only`的Trainer，直接复用其已发布Dataset、同一严格xy模型映射和同一loss桥接；工程检查通过不开放研究Trainer或改写源准入。

## 运行

研究环境使用已存在的Python3.10/Torch2.6.0+cu124/Ultralytics8.4.60，P1 Python3.9不安装Torch。项目根执行：

```powershell
D:/projects/FutsalMOT/.venv/Scripts/python.exe Content/FutsalMOT/code/experiments/sim2real/check_pose17_location_cpu.py --release Saved/Pose17Development/diagnostic_pose17_export_a01 --acceptance Saved/Pose17Development/diagnostic_pose17_export_a01_cpu_acceptance_a01.json --output Saved/Pose17Development/diagnostic_location_cpu_a01 --imgsz 128
```

本地官方`yolo11n-pose.pt`默认位于研究目录，SHA256必须为`869e83fcdffdc7371fa4e34cd8e51c838cc729571d1635e5141e3075e9319dc0`。不下载模型；CLI禁网络、隐藏CUDA并显式将模型/张量放CPU，CPU单线程。只允许128或256输入尺寸，输出须为全新workspace Saved下`diagnostic_`目录，不能覆盖或写入v1。

默认将40图整理成一个batch，实际原图只经过确定性LetterBox，不新增随机增强。若128下目标分配没有实际位置梯度，会报失败，不能将零pose loss称为通路通过；之后由主agent决定是否使用新的256尝试，不自动重跑。

## 源证据与完整人物

适配前实际读取并核对诊断dataset/group、CPU导出验收内容及其attempt身份、冻结plan、manifest、原生采样/恢复与结构配对报告。复用原生验收器当前校验完整四相机/十实体/帧/fence/恢复，再将导出每相机的400对象world17逐值绑定本次原始native；UV和17bool有效性必须与本相机真实标定重新投影一致。仅填写一个非空`evidence_ref`不能通过。

所有mask面积非零的可见人物bbox都保留，不要求其17点全部有效；缺点人物仍有检测cls/bbox目标。完全不可见（area0、bbox=null）的实体不伪造图像检测框，保留在原world文件与当前行`nonvisible_entities_retained_in_world`。bbox必须保持源可见mask的tight半开xywh，不允许默默裁剪或改成amodal框。

JSONL保留每图完整人物ID集合、源dataset/验收/结构报告引用和每实例native/world引用，来源证据收据另记录实际小JSON/manifest SHA。世界值仍逐相机独立，不平均或用容差去重。为了接已发布Dataset当前图像身份契约，这个40图工程检查在JSONL准备时计算图像SHA，Dataset初始化按其默认接口再次核验；这不是零媒体hash检查，也不是正式大规模发布策略。它不重解码EXR或重建mask。

## 位置与可见性分离

每人物输出17×2归一化xy和独立17bool `location_valid`，`source=synthetic`、`source_coco_v=null`。unknown可见性与有效位置同时存在；画外/相机后方/缺失位置为false，张量中无效xy规范占位零，源UV/world仍在原诊断文件中。数据不转为YOLO txt，也不把内部bool解释成COCO v1/v2。

检查保留已验证的Pose塔hidden51及末端物理51通道，严格选择xy暴露34维输出。实际batch必须有有限且非零位置loss/xy梯度；visibility loss与物理visibility行梯度须为0，并安装visibility BCE调用失败哨兵。

本次真实数据全部位置可能有效，因此不能谎称真实batch直接覆盖了缺点梯度。另运行一个明确标记`counterfactual_probe_not_source_GT=true`的纯loss反例：两人中一人仅一点有效、另一人全缺点，验证有效点梯度非零、无效点及全缺点人物pose梯度严格为零、visibility loss为零。该探针不篡改JSONL、源标注或真实有效性mask。

## 当前验证

主agent已实际执行真实40图：`Saved/Pose17Development/diagnostic_location_cpu_a01/receipt.json`为passed，40图/400可见人物/6800有效位置，pose loss11.9853，xy行梯度绝对和735.5164；visibility loss、visibility行梯度及独立无效点梯度探针均0。优化器更新0、GPU=false、not_training及原MRQ状态保持。这补充了下文开发时toy证据，不代表研究训练收益。

新适配器8项tiny输入/反例通过，覆盖人物遗漏、伪造非空ref、失败验收、world变化、非bool mask、可见性升级及bbox越界。另在研究Python中对两张自绘TOY图实际运行同一CPU batch入口，3人物有效点数17/0/8，检测框均保留；pose loss为11.546092、xy梯度绝对和103.442501、visibility loss及visibility行梯度为0，无效点/全缺点人物pose梯度探针为0。收据位于`Saved/Pose17Development/diagnostic_location_tool_tiny_selfcheck_a01/receipt.json`。仅该小样是本工具开发时的实际运行，不将其当作真实40图已检查。

主agent后续执行真实40图后，新`receipt.json`才证明这份真实诊断数据的CPU加载/损失工程兼容。它始终记录`not_training=true`、`source_not_training_preserved=true`、`MRQ_sync_verified=false`、`training_admission=false`、`optimizer_updates=0`。没有真实receipt或中途失败仅留下JSONL时，都不能声称通过。
