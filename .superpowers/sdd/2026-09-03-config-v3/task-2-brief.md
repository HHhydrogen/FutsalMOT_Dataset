# Task 2: Load Local Config and Resolve v3 Into Legacy Runtime Fields

Read this first: exact task requirements.

Work in `C:\Users\20113\AppData\Local\Temp\opencode\futsalmot-single-episode-public-output`. Modify `config/loader.py`, `config/resolver.py`, `config/models.py`, `config/paths.py`, and `tests/test_task_resolver.py`.

Implement:

- `load_local_machine_config(path: Path) -> LocalMachineConfig`
- `resolve_local_config(cli_path: Optional[Path], env: Mapping[str, str]) -> Path`
- `resolve_task(task_file: Path, local_config: Optional[Path] = None) -> ResolvedTask`
- `validate_task(task_file: Path, local_config: Optional[Path] = None) -> List[str]`
- `resolved_task_summary(resolved: ResolvedTask) -> Dict`

Priority is exactly explicit CLI local path, then `FUTSALMOT_LOCAL_CONFIG`, then error. Never auto-search a neighboring file. For v3, validate local config exists, legal JSON/fields, dataset_root exists/is directory, ue_project_root exists/is directory and contains `.uproject`; fail before creating output/runtime or overwriting an existing resolved task. Local config has only machine paths/environment fields.

Resolve v3 into existing legacy fields: output.fps -> export target/playback, annotation playback and MRQ frame rate; resolution -> annotation/render dimensions; cameras C## -> UE sequence entries, camera actors/list/count/public names; steps/fps -> expected frames; annotations/classes -> canonical dependencies and include_ball; audit expected cameras/frames. Preserve advanced simulation fields/defaults. Do not alter public GT contract.

Add `resolved.config_v3` summary containing episode, seed, steps, fps, resolution, cameras, annotations, classes, expected frames and public sequence names. Preserve `export_profile`, `ue_profile`, `actor_mapping`, `postprocess`, `audit`, `artifact_policy` old fields. Existing v2 path must remain usable without local config.

Follow TDD, run `uv run pytest tests/test_task_resolver.py -q` and `uv run pytest`, append exact outputs to `.superpowers/sdd/2026-09-03-config-v3/task-2-report.md`, commit concise Simplified Chinese message. Do not dispatch agents/reviewers or push. Return only status/commit/tests/concerns.
