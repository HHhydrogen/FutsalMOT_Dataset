# Task 5 报告：清理 Config v3 文档并验证

## 完成内容

- 更新 `README.md`、`configs/README.md`、`CLAUDE.md` 和 `docs/REPRODUCIBILITY_AND_MANIFEST.md` 的 Config v3 示例，统一使用 `C03 -> FrontCamera`、`C07 -> RearCamera` 显式映射。
- 删除 README 中被误标为 Config v3 的旧 `postprocess`/`yolo_pose`/`debug` 伪 JSON 示例；v2 legacy 参数章节保持兼容说明。
- 补充 resolved task 驱动的动态 manifest/status 能力声明、条件化 cleanup gate 和不变的 public GT 结构说明。
- 新增 `tests/test_documentation.py`，解析所有目标文档中的完整 v3 JSON 示例并用 `TaskConfigV3` 校验，同时防止 v3 示例重新出现 legacy `postprocess` 字段。

## 验证

- `uv run pytest tests/test_documentation.py -q`: 2 passed
- `uv run pytest`: 749 passed, 7 deselected, 62 warnings
- `uv run pytest tests/test_repository_hygiene.py -q`: 9 passed, 28 warnings
- `git diff --check`: 通过

## Git

提交信息：`清理Config v3公开文档契约`

提交：`3341e1e`
