# FutsalMOT GRF-UE 数据集代码

本目录是独立Python仓库，分支main；外层UE项目通过submodule gitlink引用。

## 正式数据

8个seed、八天气、32相机序列、原生3840×2160、10FPS，共20,116张RGB和同数mask，含逐帧bbox。每机位连续414–952帧，全部保留原train归属。正式入口：G:/FutsalMOT_Dataset/v1_no_numbers_4k/dataset.json。

每seed单片段，已移除clips/clip_000。直接读取groups/seed_XXXXXXXX/cameras/；保留group.json、clip.json、trajectory.jsonl和checksums.json。消费者按group.json.clips[].path读取，不推断父目录层数。

## 环境与入口

| 阶段 | 环境 | 入口 |
| --- | --- | --- |
| P1轨迹及后处理 | Python3.9固定，.venv | grf-ue task |
| P2序列与MRQ | UE5.8 Editor Python | ue/run_task.py，由Unreal MCP执行 |
| 研究训练及评测 | 独立Python3.10环境 | experiments/sim2real/pipeline.py |

P1不导入unreal；UE不导入NumPy/OpenEXR，JSON/JSONL是两环境接口。不要升级P1 Python。

```powershell
uv sync
uv run grf-ue task validate configs/weather_seeded_smoke_3frames_4cam.json
uv run grf-ue task export configs/weather_seeded_smoke_3frames_4cam.json
uv run grf-ue task ue-command configs/weather_seeded_smoke_3frames_4cam.json
uv run pytest
```

本机uv包装器异常时可使用.venv/Scripts/python.exe -m pytest。MRQ异步，提交后保持Editor tick；成功终态后执行P1。通用恢复流程与正式生产持久场景有不同完成契约，不能将retained伪装成restored。

## 权威文档

- [DATA_CONTRACT.md](docs/DATA_CONTRACT.md)：轨迹、相机和标注字段。
- [VALIDATION_AND_LIMITATIONS.md](docs/VALIDATION_AND_LIMITATIONS.md)：验证与限制。
- [SEED_DATASET.md](docs/SEED_DATASET.md)：平铺发布结构和迁移。
- [PRODUCTION_V1.md](docs/PRODUCTION_V1.md)：本轮规模、生产优化及来源。
- [ACTIVE_TRAJECTORY_QC.md](docs/ACTIVE_TRAJECTORY_QC.md)：连续活跃窗口。
- [FLOOR_DIVERSITY.md](docs/FLOOR_DIVERSITY.md)：独立seed地板因子与UE资产只读验收入口。
- [实验协议](experiments/sim2real/README.md)：真实足球检测和追踪对照。

基准full，其余quick/basic如实记录；全部生成mask/bbox不等于全部做像素审计。号码尚未验收，正式数据关闭号码和Pose。

## Pose研究开发

四seed位置试产已进入实际采集，采用显式[120帧契约及独立源序列](docs/POSE17_LOCATION_PILOT_SOURCES.md)，后处理CPU默认6进程且不重复生成总览。已新增[真实40图位置损失接入](experiments/sim2real/POSE17_LOCATION_CPU_CHECK.md)、[严格矩阵SHA离线重建](docs/POSE17_OFFLINE_LBS_DIAGNOSTIC.md)及[十二关节表面支持候选](docs/POSE17_JOINT_SUPPORT_DIAGNOSTIC.md)。几何和候选证据不提升unknown可见性，也不等于训练收益。

七模型十实体四机位联合短诊断已完成40张4K RGB/40份EXR/400条采样，完整[诊断导出](docs/POSE17_DIAGNOSTIC_RELEASE.md)包含实例mask、bbox、Pose2D与逐机位原始world17。见[联合入口](docs/POSE17_JOINT_DIAGNOSTIC.md)和[七角色校准快照](calibration/diagnostic/pose17_head_surface_v1/README.md)。472项针对CPU回归及真实MRQ证据通过；所有visibility仍unknown，输出不是正式训练数据，不能称已证明合成收益。

新Pose17与v1分开，当前尚无正式Pose批次。已发布的研究工程入口包括[保留预训练容量的位置头](docs/POSE17_XY_HEAD_TRANSFER.md)、[独立缺点mask与JSONL图像CPU链](docs/POSE_LOCATION_JSONL_CPU_DATASET.md)及[未冻结对照候选](docs/POSE_REAL_SYNTHETIC_CONTROL_CANDIDATE.md)。unknown可见性与位置有效性分别存储，全缺点人物仍保留检测框；toy更新和续载只证明工程兼容，不能当作AP或关联提升。P1不安装Torch，研究依赖保持独立；GPU研究训练和UE渲染串行，最终test116/187封存。

内层提交和push成功后外层才可更新gitlink。用户已明确要求提交时无需重复确认；无授权不提交。未完成号码开发保留在工作树但排除提交。数据、权重、.venv、.external、.futsalmot和Saved等本机产物不入Git。
