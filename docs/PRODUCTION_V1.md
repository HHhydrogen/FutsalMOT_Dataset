# 无号码4K正式批次

2026-10-22本轮已完成。用户最终目标8组，原16/32组清单只作历史归档。

| seed | 天气 | 帧/机位 |
| --- | --- | ---: |
| 1021 | light_haze_v1 | 902 |
| 1025 | daylight_clear_v1 | 430 |
| 1007 | dawn_cool_v1 | 433 |
| 1042 | overcast_soft_v1 | 463 |
| 1044 | cool_evening_v1 | 822 |
| 1048 | night_floodlit_v1 | 952 |
| 1102 | cloudy_variation_v1 | 414 |
| 1131 | warm_evening_v1 | 613 |

合计20,116 RGB及同数mask，4机位、原生3840×2160、10FPS，连续41.4–95.2秒。本轮8组全部train，合成val/test暂缓，无号码和Pose。

根目录G:/FutsalMOT_Dataset/v1_no_numbers_4k；平铺为groups/seed_XXXXXXXX/cameras/，保留clip.json描述连续窗。结构见[SEED_DATASET.md](SEED_DATASET.md)。

## 生产流程与优化

固定seed采集1800帧源轨迹，运动QC选择最长连续活跃窗，不拼接或补尾段。UE按绝对manifest设置球员和天气，复用Editor/场景；四机位串行MRQ。

原始PNG/EXR直接SSD，只将FinalImage送入PNG编码，多层Object-ID EXR保持原生。逐机位解绑delegate、释放executor/job和UE GC；RGB移交G盘，EXR同盘归档。6进程CPU生成全部mask/bbox。发布计算一次媒体SHA，清理检查文件身份，不重复读媒体。

写线程和队列目标12/12；提交余量低于8.5GiB回退10、低于7.5GiB回退8，6GiB最终保护保留。同一Editor跨seed继承实际回退，避免重新置12造成内存峰值；最终12线程/8容量。

1048按用户要求废弃a01，用a02全部重渲染。1102首机位57帧因程序内存保护取消，失败帧不复用，明确续跑后发布并清理。已发布组不覆盖。

## 准入与复现

基准full，其余按授权quick/basic；clip记录真实等级及full基准，不伪造像素审计。无逐组预览或重复媒体SHA。通用task默认恢复，生产完成契约为retain。

本机Saved/Production/v1_no_numbers_4k保存冻结代码、原始轨迹、计划与发布/清理回执。本轮原收据对应生成时布局；之后的平铺变更记录于Saved/DatasetMaintenance及正式shared/layout_migration.json，媒体未复制或重编码。UE/GPU跨平台不保证PNG逐字节一致。

已结束的164个临时Sequence通过UE API清理，正式地图和共享资产保留；AutoSave关闭。Editor核对自有checkpoint后正常退出，为GPU实验让出资源。

研究入口为experiments/sim2real/run_serial_pilot.ps1：Prepare建立派生标签与冻结预算，Gpu进行统一设备的A/B评测、C训练及检测/追踪评测。图像通过Windows目录联接直接引用G盘，正式数据不可原地改写。真实test不自动运行。
