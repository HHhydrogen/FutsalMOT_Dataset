# Task 3: Add Public Output Validator and Audit Integration

Read this first — exact requirements.

Create `src/grf_ue_bridge/public_validator.py`, `tests/test_public_validator.py`, and modify `src/grf_ue_bridge/workflows/task_audit.py` plus relevant tests.

Required interface:

`validate_public_episode(episode_dir: Path) -> ValidationResult`, where result exposes `ok`, `errors`, `stats` and a CLI-compatible success/failure interpretation.

Validate:

- episode_manifest relationship, sequence naming, required files and seqinfo fields;
- JPG frame continuity/readability/size, no zero-byte files;
- MOT nine comma-separated fields and bbox bounds;
- Pose player records have exactly 17 `[x,y,v]` triples and valid finite coordinates/visibility; ball records have `keypoints: null`;
- MOTS six whitespace-separated fields; RLE decodes to declared dimensions and non-negative area;
- exact per-frame `(frame_id, track_id)` set equality across MOT, Pose, MOTS;
- public IDs/classes: players 1..10/class 1, ball 100/class 100.

Integrate `task_audit` so an episode with `episode_manifest.json` uses public validation and only requires canonical files after cleanup. EXR, render/, mask/, YOLO and internal JSONL are optional diagnostics. Preserve existing legacy annotation audit when no public manifest exists.

Add tests for a valid player+ball fixture and cross-modal mismatch, plus malformed MOT/Pose/MOTS/RLE, missing JPG, manifest mismatch, and ball keypoints errors. Do not remove old validator behavior. Keep comments/docstrings in Simplified Chinese and do not touch batch/split/assembler/config refactors.

Run `uv run pytest tests/test_public_validator.py tests/test_audit_fixes.py tests/test_annotation_validator.py -q` and report exact result in `.superpowers/sdd/2026-09-02-single-episode-public-output/task-3-report.md`. Commit with concise Simplified Chinese message. Do not dispatch subagents/reviewers or push.
