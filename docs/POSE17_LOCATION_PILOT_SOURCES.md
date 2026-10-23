# 四seed位置监督候选的源Sequence

`ue/pose17_location_pilot_sources.py`只创建新的源Sequence，不准备队列、不提交MRQ，也不启动训练。它消费本机Saved中的冻结候选；候选本身不随Git分发，也不是正式Pose数据发布契约。

| seed | 原冻结活动源窗 | 候选渲染源窗 | 导入前缀 | 本批split |
| --- | --- | --- | --- | --- |
| 1042 | [0,463) | [100,220) | [0,220) | train |
| 1044 | [207,1029) | [307,427) | [0,427) | train |
| 1048 | [0,952) | [100,220) | [0,220) | train |
| 1102 | [0,414) | [100,220) | [0,220) | synthetic_dev |

每组120帧、10FPS、四个原机位、3840×2160；意向总量1920对RGB/mask。窗口在原活动窗内，不补帧、不拼接，也不把1044的源307误写成100。导入完整零帧前缀，沿用原导入器的运动历史及速度轨道，最终只渲染候选窗口；首尾样本跨度11.9秒，名义时长12秒。

本次固定七模型十实体roster，使用当前场景，计划固定已验证的蓝地板，优先隔离位置监督链路。它不是原seed天气/外观的复现，也不是地板多样性实验。各seed的独立地板流自然选择另外记录；`paired_override`和`planned_not_applied`如实保留，建源入口不应用地板或角色override。

新资产根为`/Game/FutsalMOT/Development/Pose17/location_pilot_4seed_a01`，每seed创建四个源Sequence。只调用原`import_grf_episode.create_sequence`，显式`replace_existing=False`、`open_after_create=False`、`run_api_smoke=False`。不复制轨迹插值算法，不打开Sequencer，不创建/删除_TEMP_SMOKE。Actor快照与Level/PIE/MRQ/dirty门禁复用已验证的联合源入口；每seed导入后核对十五Actor不变、dirty为空，最终核对地图及四个canonical Sequence文件SHA未变。

通过真实Unreal MCP执行文件前，设置`FUTSAL_POSE17_LOCATION_SOURCE_ACTION`：默认`preflight`只读UE；`prepare`仅保存十六个自有新Sequence。两动作分别写候选目录中的`source_preflight.json`、`source_prepare.json`。目录/任一目标资产/本动作收据已存在就拒绝，失败保留partial及失败收据，不自动重试、删除或覆盖。

入口核对候选SHA、真实GRF源/meta SHA、v1 clip来源、冻结生产计划、角色manifest及导入器SHA。它不调用短诊断`validate_plan`来绕过3..30帧范围。120帧必须由另一个显式`location_pilot_contract=true`门禁准入，限本四seed/窗口/split，不能顺便放宽所有joint诊断。source准备成功也不能代替实际120帧MRQ终态、连续帧与全员覆盖检查。

visibility继续unknown，位置有效性独立；不将unknown改成COCO可见标签。若跨相机同timestamp world存在微小差异，就保留逐机位原值，不平均或用epsilon制造精确共享。1102只是本批合成dev，不是已封存的真实final test；v1原train归属不改。已有初始化若用过1102必须披露污染，不能再声称独立合成泛化验证。

本入口不修改共享runner、analyzer或C++。完成渲染并安全退出Editor后，才另行冻结真实/合成比例、真实曝光及更新预算并开始GPU对照；本文件不授权训练，也不预设收益。
