# Task 4: Preserve v2 and Existing Pipeline Regression Coverage

Read this first: exact task requirements.

Work in `C:\Users\20113\AppData\Local\Temp\opencode\futsalmot-single-episode-public-output`. Modify only tests and loader if required: `tests/test_task_config.py`, `tests/test_task_resolver.py`, `tests/test_task_cli.py`, `tests/test_ue_resolved_task.py`, and `src/grf_ue_bridge/config/loader.py` only for compatibility fixes.

Guarantee existing v2 `futsalmot_dataset_task` configs continue to load, resolve, and feed old fields to validate/export/ue-command/postprocess/audit without local config. Loading v2 emits clear deprecated warning; v3 does not. Verify v3 resolved objects still contain `export_profile`, `ue_profile`, `actor_mapping`, `postprocess`, `audit`, and `artifact_policy`; `validate_resolved_task()` passes. Do not change public output contract or add unrelated features.

Add targeted tests for `pytest.warns(DeprecationWarning, match="Config v2")`, v2 resolve without local config, v3 resolved old shape, and existing CLI/UE resolved behavior. Run `uv run pytest tests/test_task_config.py tests/test_task_resolver.py tests/test_task_cli.py tests/test_ue_resolved_task.py -q` and then `uv run pytest`; append exact outputs to `.superpowers/sdd/2026-09-03-config-v3/task-4-report.md`. Commit with concise Simplified Chinese message. Do not dispatch agents/reviewers or push. Return only status/commit/tests/concerns.
