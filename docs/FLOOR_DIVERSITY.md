# 可复现地板颜色因子

本功能提供独立于GRF运动、天气和角色外观的地板选择流。四种颜色共用中性灰色细纹基底：蓝、青绿、红棕和中性灰。地板变化用于视觉条件对照，不代表已经证明训练收益。

## P1选择与记录

标准库模块`src/grf_ue_bridge/floor_diversity.py`提供`select_floor_id`、`build_floor_manifest`和`paired_floor_manifests`。相同seed及catalog得到相同选择，不改变Python全局RNG。manifest需要真实轨迹、固定场景上下文和纹理的SHA，不接受伪造来源。

配对实验必须保持运动、相机、角色、天气及采样不变，仅覆盖floor_id；同一运动的全部变体属于同split。普通小批seed不能保证四颜色数量均衡，需要均衡时预先登记分层名单及override。manifest的`planned_not_applied`不是UE应用成功收据。

## P2资产创建与只读验收

通过真实Unreal MCP执行`ue/create_floor_assets_v1.py`。默认检查源素材与目标资产；六个目标已全部存在时只读验收，不覆盖；目标均不存在且显式设置`FUTSALMOT_FLOOR_CREATE_APPLY=1`时才创建。部分目标已存在则拒绝隐式补齐，须明确新attempt。

纹理源由外层UE仓库分发：`SourceAssets/Floors/V1/T_FloorNeutral_Albedo_v1.png`及`source_receipt.json`。SHA固定为`81bfaa6a9e0e826446649d251d638c8a6c11319c135173f7768b3a009f5166eb`；来源记录保留生成方式及提示词，不依赖历史Saved目录。

创建资产位于`/Game/FutsalMOT/Environment/Floors/V1/`：一张Texture2D、母材质`M_FloorTint_v1`及四个`MI_sport_*_v1`。纹理使用sRGB、Wrap；材质以绝对世界XY厘米坐标除以`FloorTileSizeCM=200`采样，粗糙度0.72、金属度0，纹理调制限制为±4%。仅适用于水平XY场地。

脚本不替换地图材质，不改变球员或灯光，不自动启动渲染。真实地板绑定由独立生产事务管理；当前此功能尚未接入正式批量生产。UE5.8参数setter的bool存在恒false行为，入口通过真实参数读回验收。

## 验证与边界

```powershell
.venv/Scripts/python.exe -m pytest tests/test_floor_diversity.py tests/test_floor_asset_entrypoint.py -q
```

19项CPU测试通过；它们验证选择、配对、来源约束及入口不依赖历史恢复器，不能证明渲染效果。四种地板已在实际球场作静态同姿势预览并恢复原槽，独立白线保留、非地板场景状态未变。此预览包含Editor辅助图标及未冻结的动态渲染状态，不是严格像素配对消融。

正式生成前仍需MRQ中核验地板在各天气下的曝光、周期和接缝，并与RGB/mask/Pose同步门禁一起验收。既有v1八组数据不修改。
