# Task 1: Add Config v3 and Local Machine Models

Read this first: this is the exact task contract.

Modify the inner Python repository at `C:\Users\20113\AppData\Local\Temp\opencode\futsalmot-single-episode-public-output`. Work on `src/grf_ue_bridge/config/paths.py`, `models.py`, `.gitignore`, and add `configs/local.machine.example.json`; update `tests/test_task_config.py`.

Add strict user Config v3 models:

```json
{
  "schema": "futsalmot_task",
  "version": 3,
  "episode_id": "0001",
  "simulation": {"scenario": "5_vs_5", "seed": 42, "steps": 300},
  "cameras": {"C01": "CineCam_01"},
  "output": {"fps": 30, "resolution": [1920, 1080], "annotations": ["mot", "pose", "mots"], "classes": ["player", "ball"]},
  "debug": false
}
```

Required constants: `TASK_V3_SCHEMA = "futsalmot_task"` and `LOCAL_CONFIG_ENV = "FUTSALMOT_LOCAL_CONFIG"`.

Validate v3 schema/version, safe episode ID, positive simulation steps, supported scenario/advanced optional fields, C## camera keys with nonempty UE actor names, positive `[width,height]`, supported nonempty annotations (`mot`, `pose`, `mots`) and classes (`player`, `ball`), and reject legacy duplicate top-level blocks/fields (`dataset_root`, `ue_project_root`, `export`, `ue`, `postprocess`, `audit`, `artifact_policy`, `task_id`, `episode_name`). Use Pydantic patterns already in models.py and keep v2 models working.

Add `LocalMachineConfig` accepting only machine path/environment fields, at minimum `dataset_root` and `ue_project_root`; reject task fields such as episode/cameras/fps/annotations/classes/debug. Add `configs/local.machine.json` to .gitignore and create `configs/local.machine.example.json` with placeholders only.

Do not change resolver/CLI yet beyond model constants needed by later tasks. Keep comments/docstrings in Simplified Chinese. Write tests first and run them to observe expected failure before implementation. Run `uv run pytest tests/test_task_config.py -q`, append exact results to `.superpowers/sdd/2026-09-03-config-v3/task-1-report.md`, commit with concise Simplified Chinese message, and return only status/commit/tests/concerns. Do not dispatch agents/reviewers or push.
