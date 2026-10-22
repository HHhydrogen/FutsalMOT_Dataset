# 验证和当前限制

## 当前正式批次

2026-10-22已完成8个seed、八天气、32视角序列，20,116对原生3840×2160 RGB/mask，含逐帧bbox；生产与平铺迁移记录见PRODUCTION_V1.md及SEED_DATASET.md。

基准组为full，后续按用户授权采用quick/basic。clip保留真实准入等级、场景生命周期和full基准引用。全部生成mask/bbox不等于全部逐帧像素审计，structure检查也不能发现媒体内部字节损坏。

## 测试与两环境边界

默认pytest跳过7项GRF集成，真实GRF可显式使用-m grf_integration。P1固定Python3.9；UE脚本只能在Editor Python中执行，不在P1导入unreal或在UE导入NumPy/OpenEXR。

无号码发布工作树完整回归1710 passed、7 deselected。测试证明纯代码契约，不替代UE资产、动画、相机、实际渲染或视觉验收。新GPU实验的103项结构/协议回归不等于训练结果已经改善。

## 数据与运动限制

- GRF轨迹来自双方内置AI，不是人类球员采集；源坐标与40×20m场地映射固定。
- 按固定seed采集完整1800帧，运动QC选最长连续活跃窗口；不同seed不强制900帧，不补静止尾段或拼接片段。
- 活跃QC只证明轨迹运动指标，不证明所有足球战术真实、所有肢体动画或碰撞合理。
- 可见像素bbox不同于amodal人体框；完全遮挡时bbox为空，未知visibility不伪造成1。
- 同seed全部相机同split；不能随机分邻帧或机位制造独立验证集。本轮8组均train，真实数据按比赛隔离。

## UE与生命周期

- 当前UE5.8与正式地图L_FutsalCourt；原生MCP优先，真实脚本使用FutsalMOTTools。
- 地图切换仅通过原生SceneTools.load_level，不在切图时持有UWorld引用。
- MRQ异步，提交不等于成功；必须读取真实终态与完整帧集。PNG过滤只保留FinalImage，多层Object-ID EXR走原生路径。
- 通用task要求appearance/environment恢复；持久生产使用独立retain收据，绑定真实文件SHA和checkpoint，不能伪造restored。
- 逐机位释放executor及delegate，活动MRQ期间不重建线程池。写队列目标12/12，实际依提交内存回退并跨seed继承，6GiB最终保护保留。
- 未知dirty包或checkpoint变化时停止写操作，不强制结束Editor；只可放弃经核对的任务自有场景状态。
- FootIK、Retarget和动画几何无法由静态测试证明；角色资产仍需SOP及真实UE验收。
- 不同UE/GPU的PNG不保证逐字节一致。

## 发布、清理与归档

verify_clip默认full重读SHA；metadata读文本SHA和图像头；structure读文本和文件覆盖，显式复用已有发布证据。发布完整SHA只生成一次，用户授权的快速清理不重复扫描媒体。

layout迁移只允许每seed单片段，同卷移动目录、修改引用和文本SHA，媒体不复制或编码。旧生产回执保留其原布局；迁移回执单独记录，不能改写历史来源。

临时uasset通过UE资产API删除，先确认没有范围外引用，不能用文件系统强删。正式地图、共享模型、骨架和材质不因零引用自动删。Windows递归删除严格校验边界，不跟随研究目录junction到G盘源图像。

Git不包含生产媒体、原始轨迹、权重或本机Saved。清理缓存和测试产物时保留源轨迹、冻结运行代码、关键回执及号码开发证据。AutoSave默认关闭，重要修改先主动保存再提交版本。

## GPU实验限制

本轮只检验固定真实标注/计算预算下，合成数据对公开足球检测和固定追踪器效果的影响。无真实五人制标注，不能外推目标场景收益；单个小样本val和单训练seed不足以支持稳健论文结论。真实final test保持封存。训练启动或完成不等于证明合成数据有效，必须报告实际指标及下降项。

号码系统未完成，不属于本轮发布或训练输入；其贴合、遮挡、mask同步和身份恢复收益以后独立验收。
