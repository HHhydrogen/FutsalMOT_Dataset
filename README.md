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

内层提交和push成功后外层才可更新gitlink。用户已明确要求提交时无需重复确认；无授权不提交。未完成号码开发保留在工作树但排除提交。数据、权重、.venv、.external、.futsalmot和Saved等本机产物不入Git。
