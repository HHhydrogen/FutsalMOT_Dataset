# Validation、Audit、Cleanup 统一契约设计

## 目标

统一 Dataset/Python 管线中 Validation、Task Audit 和 Cleanup 对任务成功的判断：task/resolved-task 声明的功能决定 required checks；未启用的功能只能产生 `skipped`，不能导致失败；required check 失败或 errors 非空必须导致失败；只有 warnings 时任务仍通过。

## 当前根因

`task postprocess` 根据 `postprocess.formats` 生成派生产物，但调用 `validate_annotation_dir()` 时没有传入 task 要求，validator 因此无条件要求每个 camera 都有 `gt/gt.txt`。另一方面，cleanup 的 `_validation_gate()` 无条件要求 `render_summary.json` 和 `pose_session.json`，并且读取 audit 时只识别旧的 `ok/failed_checks`，而 `task_audit` 正式写出的是 `passed/exit_code/errors/warnings`。Audit 也以 `pose_session.json` 是否存在推断 Pose 是否启用，违反了 requirement 应由 task 配置决定的原则。

## 设计

### 1. TaskRequirements

新增纯 Python 的 `task_requirements` 模块，提供小型不可变 requirement 对象及解析 helper。helper 接受 resolved task、resolved task 字典或独立配置字典，统一读取：

- `ue.annotation_export.render_rgb.enabled`：`requires_render`，缺省按 UE 渲染代码的 `enabled=False` 语义视为未要求；
- `ue.annotation_export.instance_mask.enabled`：`requires_instance_mask`；
- `postprocess.formats` 和 `ue.annotation_export.export_mot`：`requires_mot`。明确 `export_mot=false` 且格式只有 `json` 时不要求 MOT；若 postprocess 明确选择 `mot`，则该 task 要求 MOT；
- `postprocess.formats`：`requires_yolo_det`、`requires_yolo_seg`；
- `postprocess.yolo_pose.enabled`：`requires_pose`，文件是否存在不能改变该值。

helper 只表达 requirement，不检查文件，也不根据已有 artifact 反推功能开关，避免 Audit/Postprocess/Cleanup 各自解释配置。

### 2. ValidationResult

新增纯 Python `ValidationResult`，公开字段为：

```text
passed: bool
exit_code: int
errors: list[str]
warnings: list[str]
checks: dict[str, dict]
```

每个 check 至少有 `status`，取值为 `passed`、`failed` 或 `skipped`，并可包含 `required`、`message` 和 detail。结果由统一 finalize 逻辑计算：errors 非空或 required check 为 failed 时 `passed=False`、`exit_code=1`；否则 `passed=True`、`exit_code=0`。warning 不参与失败判断，skipped 不参与失败判断。提供 JSON dict 转换和从 audit report 读取的 fail-safe helper，供 cleanup 使用。

### 3. Annotation validation

`validate_annotation_dir()` 增加可选 `require_mot` 参数，默认保持 standalone `validate-annotations` 的旧行为（没有 task 时仍要求 MOT）。task postprocess 从共享 requirements 传入 `require_mot=False/True`。MOT 未 required 时缺少 `gt/gt.txt` 写入 skipped detail，不加入 errors；若文件存在仍验证其内容。MOT required 时保持缺失或非法即失败。整数退出码 API 保持兼容，并补充结构化结果入口给 workflow/audit 使用。

### 4. Task Audit

`task_audit` 保留现有 `cameras`、`sync`、`mapping`、`calibration`、`render_summary`、`pose_coco17` 等 detail，以及 top-level `passed`、`exit_code`、`errors`、`warnings`。新增 canonical `checks`，并由 `ValidationResult` 统一计算最终结论。

Render、Mask、MOT、YOLO 和 Pose 的存在性/完整性依据 requirements 判断。Pose required 时严格检查 session 存在、`capture_complete=true`、raw capture、COCO17 和 camera 侧结果；Pose disabled 时产生 `runtime_pose: skipped`，即使残留 pose 文件也不提升 requirement。MOT disabled 时报告 `mot_export: skipped`。现有 detail 中的 `ok` 字段继续保留作为局部统计兼容字段，但不再作为正式 top-level 成功契约。

### 5. Cleanup

`_validation_gate()` 接受 resolved task/requirements 并复用同一 requirement helper。只对 required render、Pose 和其它已有 gate 执行阻塞检查；未 required 的功能返回 skipped detail。存在 audit JSON 时优先读取并严格尊重 canonical `passed`：`passed=false`、`exit_code!=0`、errors 非空或 required failed check 都阻止 cleanup。旧格式仅在没有 canonical 字段时读取 `ok/failed_checks`；旧格式中明确失败、无法解析或字段不确定均 fail safe。warnings-only 不阻止 cleanup。

`plan_cleanup()` 继续默认 dry-run，`apply_cleanup()` 只在 gate 通过时删除现有 transient 集合。保持 `img1`、camera.json、annotations、MOT、Pose final labels、provenance、audit 和 manifest 等 canonical artifacts 不进入删除集合。

### 6. CLI、兼容性和文档

保留现有 CLI 名称和参数。standalone `validate-annotations` 默认语义不变；task workflow 显式传入 requirements。Audit 新报告不生成 `ok/failed_checks` 契约，只保留已有 detail 字段；cleanup 对旧 report 提供只读兼容。更新 `docs/VALIDATION_AND_LIMITATIONS.md`，描述 JSON-only/MOT disabled、Pose disabled cleanup 和 canonical audit gate 的真实行为。

## 测试范围

- `ValidationResult`：errors、required failed、warnings-only、skipped 的四种结论；
- requirements：MOT、Pose、render/mask/YOLO 的配置解析和残留文件不影响 requirement；
- annotation validator：MOT disabled 缺失、MOT enabled 缺失、MOT enabled 合法；
- task postprocess：formats 传递到 validator；
- audit：canonical checks、Pose/MOT skipped/required 和 warnings-only；
- cleanup：canonical passed/failed、legacy report、Pose 矩阵、dry-run、gate fail apply、gate pass apply 和 canonical artifact 保留。

不修改 Unreal 资产、Blueprint、地图或 UE Python；不引入依赖；保持 Python 3.9 兼容。
