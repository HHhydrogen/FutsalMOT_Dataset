# 离线RefToLocal与动态表面重建：严格SHA诊断

入口为`src/grf_ue_bridge/pose17_offline_lbs_v2.py`和独立CLI `python -m grf_ue_bridge.tools.reconstruct_pose17_lbs`。只使用P1固定Python3.9/NumPy，不导入unreal，不读RGB/EXR、训练数据或最终test，不启动GPU，也不修改旧native/资产/计划。输出仅限新诊断目录，已有目录拒绝覆盖。

## 已有真实证据

主agent真实UE导出`Saved/Pose17Development/asset_skin_data_a02/player003.json`至`player009.json`，直接snake字段、无wrapper。资产基础权重raw SHA与历史component有效权重逐角色完全相同；这个条件不把静态资产导出变为当前component采样。

本工具首次在joint_seed1048四机位全部400条native记录上重建RefToLocal，400/400原始SHA精确命中，0 mismatch、0 error。七模型参考position float32及有序triangle uint32原始SHA也全部命中。结果为`Saved/Pose17Development/offline_lbs_joint1048_a02/report.json`。较早`offline_lbs_joint1048_a01`只有首机位100条矩阵诊断；它没有被覆盖。

只为输出帧5保存四机位×十实体共40个NPZ，合计8,609,200字节，没有逐帧巨大网格JSON。对这40组重建表面的五个已签名头锚点做CPU重心插值及各自原native世界变换，与历史native五头坐标共200点比较，最大误差0m；额外小收据为`head_anchor_numeric_crosscheck.json`。这是同一蒙皮算法/数据的数值交叉核对，不是独立图像定位精度或GPU最终表面真值。

不平均、合并或按容差共享不同机位的世界轨迹，四机位分别保留自身native变换。所有输出仍为`visibility_state=unknown`、`render_surface_equivalence_verified=false`，不生成COCO可见性或正式Pose发布标记。

## 固定数学路径

当前只实现`ue58_win64_transform_f64_to_f32_sse_order_non_fma_v1`：本机UE5.8源码Win64默认FMA3=0。先按`FTransform<double>.ToMatrixWithScale()`的独立double运算构造矩阵，再cast float32；每个LOD.ActiveBoneIndex按`VectorMatrixMultiply(inverse_reference, current_component_matrix)`的顺序，四项分别float32乘/加、每步舍入。未活跃骨保持Identity，不能把inverseRef×current强加给所有骨。没有使用`np.dot`/BLAS，也不搜索不同舍入方式挑一个“接近”的结果。

完整矩阵数组按骨序、row-major float32小端原字节计算`ue_ref_to_local_rowmajor_f32_le_v1` SHA，保留signed zero。实际历史骨名/顺序、同MPFBVisual骨来源、mesh identity和有效weights SHA均须先一致。leader映射未落盘时拒绝猜测；骨隐藏、ref-pose override、额外required bones、不同编译浮点路径或JSON精度不足可能造成不匹配。本工具不会用容差代替原SHA。

任一请求记录不命中或原生门禁失败，整次请求只写错误/候选报告，不输出任何表面文件。只有所有请求SHA完全一致后，才按固定影响槽序逐项float32计算LBS。零权重槽位保留，不重归一化、不合并影响。世界坐标以double按原UE `Q.Rotate(Scale*P)+Translation`顺序变换，最后厘米转米。

## 输入和紧凑输出

静态skin读取独立C++ API的直接UStruct；若调用者另加包装层，必须显式指定`--skin-snapshot-field`，不自动猜`data/snapshot`。读取Base64 raw weights/inverse-reference matrices，验证原始SHA/长度、固定stride和索引/权重宽度；8-bit权重经引擎getter的u16语义对应×257，JSON每槽局部骨、section映射的全局骨、raw_u16、float32权重与Base64交叉核对。逆参考矩阵JSON16数值也逐字节对照Base64。

旧native通过已发布normalizer和现有GPU/LBS sampling_context/逐样本门禁，并核对未截断完整frame×entity集合。此工具不重新读EXR配对，因此仍须引用原先相同attempt通过的配对报告，不能拿本工具替代RGB同步验证。

可选参考几何来自真实UE导出的`production_mesh_references_a01/Player_003.json`至`Player_009.json`。仅该历史格式允许LOD/vertex_count十进制字符串转整数；其他identity字段逐字段相同，position与有序triangle重新按原编码算SHA。

每个NPZ均可用`np.load(path, allow_pickle=False)`读取：

| 字段 | 语义 |
| --- | --- |
| `positions_cm` | N×3 float32，实际mesh component空间，单位cm |
| `mesh_identity_json` | 实际LOD0完整identity的JSON标量字符串 |
| `coordinate_space` / `unit` | `mesh_component` / `cm` |
| `visual_world_transform_json` | 本相机本帧原native显式translation_cm/quaternion_xyzw/scale_xyz |
| `native_ref_to_local_sha256` / `skin_weight_sha256` | 本条历史native矩阵/有效权重SHA |
| `entity_id` / `camera_id` | 显式实体与相机 |
| `output_frame` / `source_frame` | int64相对输出帧与原Sequence源帧 |

有序三角继续由report引用的同identity参考geometry读取，不在每条NPZ复制相同拓扑。`--surface-frame-index`仅减少表面输出，不减少全部native矩阵SHA检查；可以从这些完整但紧凑的单中间帧表面再筛十二关节支持三角，而不用重新渲染。关节中心坐标与表面支持区域仍是不同语义。

## 本机可复现命令

从内层代码仓库运行，输出路径必须全新：

```powershell
$poseOfflineArgs = @('-m','grf_ue_bridge.tools.reconstruct_pose17_lbs','--surface-frame-index','5','--output','D:/projects/FutsalMOT_UEDataset/Saved/Pose17Development/offline_lbs_joint1048_new')
foreach ($poseCameraId in 1..4) {
  $poseOfflineArgs += @('--native-json',('D:/projects/FutsalMOT_UEDataset/Saved/Pose17Development/smoke/joint_seed1048_a01_gpu_lbs/CineCam_{0:D2}_native_pose_restore_receipt.json' -f $poseCameraId))
}
foreach ($posePlayerId in 3..9) {
  $poseOfflineArgs += @('--skin-json',('D:/projects/FutsalMOT_UEDataset/Saved/Pose17Development/asset_skin_data_a02/player{0:D3}.json' -f $posePlayerId),'--reference-json',('D:/projects/FutsalMOT_UEDataset/Saved/Pose17Development/production_mesh_references_a01/Player_{0:D3}.json' -f $posePlayerId))
}
& .venv/Scripts/python.exe @poseOfflineArgs
.venv/Scripts/python.exe -m pytest tests/test_pose17_offline_lbs_v2.py -q
```

省略所有`--reference-json`只生成矩阵SHA报告；不读取/写入表面。CLI返回0表示全部请求原矩阵SHA匹配；不匹配返回2。CPU toy回归29项通过，覆盖逐项舍入、inactive Identity、signed zero、单ULP拒绝、leader/跨mesh/weights、Base64/JSON/section错配、几何原SHA、完整native门禁、整批表面拒绝、帧子集NPZ/dtype/单位与原世界变换。真实400条结果与toy测试分别记录，不把toy当实际采样。
