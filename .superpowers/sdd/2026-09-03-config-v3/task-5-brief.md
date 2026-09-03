# Task 5: Documentation and Final Verification

Read this first: exact task requirements.

Work in `C:\Users\20113\AppData\Local\Temp\opencode\futsalmot-single-episode-public-output`. Modify `README.md`, `CLAUDE.md`, and `docs/REPRODUCIBILITY_AND_MANIFEST.md` only for Config v3/local config documentation. Do not change code or public GT contract.

Document the exact v3 example, local machine example, ignored real local config, fixed `--local-config` > `FUTSALMOT_LOCAL_CONFIG` > missing-error priority, no automatic task-directory search, path/.uproject validation gates, no partial resolved output, all resolver-derived legacy fields, complete resolve summary, v2 deprecated warning/compatibility, and unchanged public GT contract. Do not claim batch/split/assembler.

Run `uv run pytest`, `git diff --check`, and verify no real `configs/local.machine.json`, secrets or generated runtime files are tracked. If a valid local config exists, run `uv run grf-ue task resolve configs/episode_0001.json --local-config configs/local.machine.json`; otherwise report the concrete limitation. Write report to `.superpowers/sdd/2026-09-03-config-v3/task-5-report.md`. Commit only documentation if appropriate with concise Simplified Chinese message; do not push. Do not dispatch agents/reviewers. Return status, commit, tests, and concerns.
