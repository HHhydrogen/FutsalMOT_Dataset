# Task 3: Add CLI Local-config Option, Warning, and Summary

Read this first: exact task requirements.

Work in `C:\Users\20113\AppData\Local\Temp\opencode\futsalmot-single-episode-public-output`. Modify `src/grf_ue_bridge/cli.py`, `config/loader.py`, and `tests/test_task_cli.py`.

Expose:

```text
grf-ue task validate TASK --local-config PATH
grf-ue task resolve TASK --local-config PATH
```

Both commands use `--local-config` first, then `FUTSALMOT_LOCAL_CONFIG`, then fail. Never auto-search neighboring local config. v3 validation/resolution errors must be nonzero and must not create/overwrite `resolved-task.json` or partial runtime output. v2 remains usable without local config and emits a clear `DeprecationWarning`/deprecated CLI warning; v3 should not warn.

`task resolve` must print concise summary including episode, seed/steps, FPS, resolution, cameras with `C## -> UE Actor`, expected frames, annotations, classes, and public sequence names, using the resolver summary data.

Follow existing Typer CLI patterns and preserve all existing commands/flags. Add tests for CLI explicit path, env fallback, explicit-over-env priority, missing local config, summary labels/values, v2 deprecation warning, and no runtime file on failure. Follow TDD. Run `uv run pytest tests/test_task_cli.py -q` and `uv run pytest`, append exact outputs to `.superpowers/sdd/2026-09-03-config-v3/task-3-report.md`, commit concise Simplified Chinese message. Do not dispatch agents/reviewers or push.
