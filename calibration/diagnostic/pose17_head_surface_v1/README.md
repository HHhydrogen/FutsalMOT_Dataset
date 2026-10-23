# Pose17五个头部表面锚点：诊断校准快照

本目录只保存Player003–009当前静态候选，不是正式生产校准或COCO可见性真值。七份`manifests/*.json`从项目`Saved/ResearchDiagnostics`按原始字节复制；内容、换行、SHA、实际UE mesh身份和全部candidate/static字段保持不变。003的逐锚点`candidate_unreviewed`也原样保留，不能用顶层静态审查状态覆盖它。

`registry.json`是新的相对路径入口，所有`candidate_path`及`bundled_evidence[].path`相对于本目录。计划生成器应先核对registry SHA，再把相对路径解析为本机完整路径；不要修改manifest正文去替换历史绝对路径，否则其SHA就会变化。本次没有修改任何runner或正式计划。

## 文件和SHA

| 角色 | 原始候选SHA256 |
| --- | --- |
| 003 | 7264fbaf474a589456dd45358d2932441c302f1a6969c1532e6ba4f2f35e1ec4 |
| 004 | 0bdb0e688cc3fbf830ee1d1e904d3166ad489536515221b9b043915e289bff0a |
| 005 | c890e9eb1b0749d6de36fe794ef1d1d640639a2865ba80cd5068732c72d9e44b |
| 006 | e9978583d822657191584ec61f8435a4629c00df63a600d81db14bb6c08a7554 |
| 007 | 2b35df1a91320156edfe5c3c9757559c5ff2a9f7b15cebc4fb11dc5abb316bbe |
| 008 | 80f04a6c9d546c71fa525a0ab958df9744377e25bca22d9bd991b8bb9f6454ce |
| 009 | e15f62f28e9883e63f1f0c688176120fde96c8721eccf1896c2303cac4771d02 |

registry还保存每角色LOD0顶点数、拓扑/参考几何签名、精确UE资产路径和原状态。003路径不含`Mesh/`子目录，004–009含该子目录。不能因共享骨架、同MakeHuman头部或相同局部锚点就混用角色identity。

本目录`.gitattributes`把原manifest与evidence JSON标记为`-text`，防止Git按平台转换换行后破坏原始字节SHA；它不改变JSON内容。

`evidence/`保存七份原始严格对应收据和原registry的字节快照。严格收据中两耳的`REJECTED`保留；当前双耳方法是独立的`recalibrated_on_ue_retriangulated_polygon`，源点与UE重新三角化表面点偏移约0.0458724mm，不能称五点严格完全同点。历史原registry里的绝对`candidate_path`只用于来源追溯；运行时入口为本目录新的registry。004–009 manifest里的绝对`strict_correspondence_receipt`不改写，新registry提供原引用到本地`evidence/`的明确映射。

## 生成和核对

从内层代码仓库根目录执行，P1固定Python3.9，只使用标准库，不需要UE、Blender、Torch或网络：

```powershell
.venv/Scripts/python.exe calibration/diagnostic/pose17_head_surface_v1/verify_calibration.py

# 可选：与原Saved七manifest再次比较字节；只读取小JSON。
.venv/Scripts/python.exe calibration/diagnostic/pose17_head_surface_v1/verify_calibration.py --source-root D:/projects/FutsalMOT_UEDataset/Saved/ResearchDiagnostics

# 在一个新目录重建数据清单，不覆盖既有不同文件。
.venv/Scripts/python.exe calibration/diagnostic/pose17_head_surface_v1/package_from_saved.py --source-root D:/projects/FutsalMOT_UEDataset/Saved/ResearchDiagnostics --output D:/projects/FutsalMOT_UEDataset/Saved/Pose17Development/calibration_copy_new
```

打包器确认原registry的七角色集合、各manifest SHA及身份，复制manifest、严格对应JSON和原registry；新registry按固定顺序生成。已有文件只允许字节完全相同，发现不同文件立即拒绝覆盖。此命令只复现快照打包，不重做原几何标定。`verify_calibration.py`独立校验七份SHA、角色资产/identity、五个真实索引与重心、原始静态未验标志、原registry对应和严格失败收据。它不读取媒体，也不将JSON检查称作UE动态验证。

## 几何生成来源与未打包依赖

原标定流程为：只读Blender的ShapeKeys evaluated几何，保留原拓扑语义；由正交源图的像素射线定位鼻尖、左右眼/耳表面；验证实际源FBX和UE LOD0参考几何，用完整表面覆盖确定`diag(100,-100,100,1)`米到厘米变换，兼容重排及拆点，不猜vertex顺序。鼻眼为严格唯一实际UE三角，双耳因重三角化严格失败后独立重标定，失败收据没有放宽阈值。实际三角顶点和重心绑定各角色LOD0签名。

完整重做几何标定仍需要项目UE资产、外部`D:/projects/FutsalMOT_blender/characters/Player_00N/`下的Blender源和`export/MPFB_Player_00N_SOP.fbx`、对应源/UE参考几何导出及历史目视审查图。这些资产/图像不在这个小清单中。manifest的`source_blend`绝对路径是来源记录；FBX SHA与21张审查图SHA来自历史收据，不代表本次重算过。registry逐项记录外部路径/原图basename、历史SHA及打包时存在性；所有七个Blend、七个FBX及21张图本机均存在，但迁移机器后不能假定可用。

本包的manifest/strict JSON消费者不需要打开上述外部资产；原始标定和独立视觉审查无法仅凭此包完整重做。这一限制不会被隐藏为“完全自包含”。未打包临时图、UE资产或媒体，不复制外部工程，不改原Saved文件。

本次联合MRQ的后续动态证据应存独立运行收据并绑定本manifest SHA。不要因联合400行结构检查或四张总览通过就回写本静态候选的`not_animated`、`not_MRQ_verified`或`precision_not_measured`；不声称所有模型动画精度、遮挡或正式Pose数据集已验收。
