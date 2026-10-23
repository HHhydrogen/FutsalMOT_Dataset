# 十二关节局部表面支持：真实像素候选诊断

本工具保持十二个COCO肢体点的native骨中心3D和投影2D位置不变，只为其构造解剖表面支持代理。输出`candidate_state`与覆盖失败原因；每一点`visibility_state`恒为`unknown`、`visibility_method=not_measured`，不写COCO或公开标注。它不是已验收的关节可见性真值。

入口为`src/grf_ue_bridge/joint_surface_support_v2.py`及`python -m grf_ue_bridge.tools.inspect_joint_surface_support`。只做P1 CPU计算，不调用UE/GPU、不改原数据、不生成逐角色图。四张既有中间帧EXR各打开一次，其他输入为小JSON和已重建NPZ。

## 支持区定义

采用既有`ue/joint_visibility_support_v2.py::prepare_support_region`相同的候选规则：三角顶点平均骨链权重不小于阈值，三角中心到该关节参考中心的距离不大于半径。本次实现向量化选择，并有独立toy与既有函数对照。具体骨链左右分开：

| 关节原骨中心 | 支持骨链 |
| --- | --- |
| shoulder / upperarm | clavicle、upperarm |
| elbow / lowerarm | upperarm、lowerarm |
| wrist / hand | lowerarm、hand |
| hip / thigh | pelvis、thigh |
| knee / calf | thigh、calf |
| ankle / foot | calf、foot |

候选参数预先声明为参考半径0.08m、平均链权重0.25，尚未冻结为正式语义协议，没有用渲染像素拟合。参考关节中心取真实资产inverse-reference float32矩阵的逆矩阵平移，是支持区准备候选，不替换native关节中心。权重来自已核对原字节的真实UE固定槽u16，按链求和/65535，不用Blender权重、最近空间点或固定骨偏移。

每角色/每关节支持区保存实际有序三角索引、骨链、参数、参考中心、mesh identity、weights SHA及规范JSON SHA，状态为`candidate_unreviewed`。七角色共84区；这些区域的解剖语义仍需独立验收。离线skin解码模块仅新增返回已经验证的`raw_weights_u16`数组，没有改变原矩阵/LBS计算。

## 每点九个真实像素中心

先由实际EXR相机投影原native骨中心，最近像素采用已校准的`floor(uv+0.5)`，cx/cy保持`(size-1)/2`。其3×3九个整数像素中心分别反投影射线，与该关节动态支持区的真实三角做双面Möller–Trumbore最近命中；无交点保持unknown，不用closest-point补点。不会沿camera→精确骨中心的一条射线，去比较邻近九个不同像素的深度。

各命中点转为camera-forward Z，与同像素`FinalImageWorldDepth.R`线性前向厘米除100得到的米比较。只使用现有真实平面深度编码校准；每行保留native/EXR相机、源帧/相对帧、fence/revision、skin/mesh/manifest门禁和NPZ身份、dtype、单位、世界变换核对。

九个像素都要有支持三角交点、有限正深度、同一首实例token、首rank coverage=1且其余rank coverage=0，才可能产生候选。固定5mm残差阈值；本版额外保留原头深度诊断的raw场景深度spread≤10mm保守门禁，混合coverage/ID、边界、无支持面或残差不一致均unknown。

- 支持表面九像素残差均在5mm内，且首实例为目标：可见候选。
- 九像素首深度都比局部支持面近超过5mm，且首实例仍为目标：自遮挡候选。
- 同样更近，首实例为另一个已映射Actor：外遮挡候选；报告原Actor标签，不把未映射几何编成人物遮挡。
- 首深度无映射token、不同实例近似同深度、残差混合或支持面在渲染表面之前：unknown。

所有候选依然unknown，不调用旧prototype中“只传非空validated_evidence_ref字符串就提升”的路径。也不将内部骨中心Z、bbox或mask命中当作遮挡真值。

## 真实首跑结果

输入`offline_lbs_joint1048_a02/report.json`绑定的400条原始矩阵SHA全部精确一致，实际只读frame5的40个紧凑表面。七geometry/skin文件SHA、NPZ的实际mesh/weights/matrix/camera/entity/frame/变换均核对；消费原`joint_seed1048_a01_alignment_a01/report.json`和`depth_encoding_a02.json`。没有新增渲染。

结果为`Saved/Pose17Development/joint_support_frame5_a01/report.json`及`support_regions.json`：四EXR各打开一次，480点，errors为空，84个支持区均非空，约9.2秒CPU。

| 仅候选分类 | 数量 |
| --- | --- |
| 可见候选 | 83 |
| 遮挡候选 | 9（自身4、其他Actor 5） |
| unknown | 388 |

unknown原因为326点raw深度spread超过10mm，57点九条支持射线未全部命中，5点混合coverage/ID。hips占未全命中的39/57，提示后续需审查髋的支持区域语义/半径，不能直接扩大半径去增加标签。

只读此报告现有九像素样本、没有重读EXR的补充诊断：326个raw-spread拒绝中，212个其实九个逐像素支持面残差都在5mm内且为目标纯实例，54个均有更近first hit，60个残差混合。说明raw Z spread可能保守拒绝可由表面斜率/曲率解释的情况。本版没有改变门禁或提升这326点；若后续改用预测残差/几何边界门禁，应另做显式版本，复用已存像素样本，不自动把统计量变成真值。

## 可复现入口

从内层根目录执行，输出目录必须全新：

```powershell
.venv/Scripts/python.exe -m grf_ue_bridge.tools.inspect_joint_surface_support --surface-report D:/projects/FutsalMOT_UEDataset/Saved/Pose17Development/offline_lbs_joint1048_a02/report.json --paired-report D:/projects/FutsalMOT_UEDataset/Saved/Pose17Development/smoke/joint_seed1048_a01_alignment_a01/report.json --depth-calibration D:/projects/FutsalMOT_UEDataset/Saved/Pose17Development/depth_encoding_a02.json --output D:/projects/FutsalMOT_UEDataset/Saved/Pose17Development/joint_support_frame5_new
.venv/Scripts/python.exe -m pytest tests/test_joint_surface_support_v2.py tests/test_pose17_offline_lbs_v2.py -q
```

17项新支持区toy测试覆盖骨链左右/参数身份、既有区域规则等价、九像素独立斜面深度、局部三角最近命中、self/external候选、边界/无命中/coverage/ID/无效深度/未映射几何和非有限值的unknown及合法JSON输出。真实480点结果与这些toy分开记录。后续仍需支持区语义、小样独立几何/遮挡检查与明确正式契约，才能考虑写训练可见性标签。
