# Task 5: End-to-End Smoke Verification and Documentation Updates

Read this first — exact requirements.

Update Chinese documentation (`README.md`, `docs/REPRODUCIBILITY_AND_MANIFEST.md`, and `CLAUDE.md` only if commands/output contract require it) to document the actual single-episode public/temporary/debug lifecycle.

Document the exact canonical output tree and structures: JPG RGB quality 95, MOT nine columns, Pose COCO 17 players plus football `keypoints:null`, MOTS six fields with COCO compressed RLE, football `track_id=100`/class 100, and `trajectory_id=episode_id`. State that default output excludes PNG masks, YOLO, debug, and duplicate internal labels; existing converters remain explicit capabilities. Do not document batch/split/assembler as implemented.

Run `uv run grf-ue task validate configs/pose_smoke_3frames_1cam.json`, then the configured export and existing UE task workflow. Unreal Python must be executed through configured Unreal MCP FutsalMOTTools when available; do not ask the user to run it manually. Run `uv run grf-ue task audit configs/pose_smoke_3frames_1cam.json` and inspect actual output structure. Use no commit/push unless repository policy permits; this repository requires no automatic commit. Write report to `.superpowers/sdd/2026-09-02-single-episode-public-output/task-5-report.md`. Return only status, any commits, test/audit summary, actual output path if generated, and concerns.
