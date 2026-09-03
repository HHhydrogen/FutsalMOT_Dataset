# Task 1: Build Canonical RLE and Public GT Writer

Read this first — it is the exact task requirements.

Implement the first component of the approved single-episode public output plan in the inner Python repository. Create `src/grf_ue_bridge/public_episode.py` and `tests/test_public_episode.py`.

Required interfaces:

- `encode_coco_rle(mask: numpy.ndarray) -> dict`
- `decode_coco_rle(rle: dict, height: int, width: int) -> numpy.ndarray`
- `public_track_id(entity_id: str) -> int`
- `write_public_episode(episode_dir: Path, *, mapping: dict, sequence_configs: list[dict], jpeg_quality: int = 95) -> dict`
- `build_public_manifest(...) -> dict`

Requirements:

- Public objects are 10 players plus ball: `L0..L4 -> 1..5`, `R0..R4 -> 6..10`, `BALL -> 100`; mask ID mapping remains `BALL -> 11`.
- Use COCO compressed RLE, column-major/Fortran scan order, with deterministic encode/decode and validation of dimensions and run totals.
- Consume existing camera metadata, internal `annotations.jsonl`, `pose_keypoints.jsonl`, actor mapping, and Cryptomatte EXR data. Reuse existing pure modules where useful; do not write `mask/*.png`.
- Produce canonical per-sequence `gt/gt.txt`, `gt/gt_pose.json`, `gt/gt_mots.txt`, `seqinfo.ini`, and episode-root `episode_manifest.json`.
- MOT rows have nine comma-separated columns: frame,id,x,y,w,h,mark,class,visibility. Players class 1, ball class 100, mark 1. Bboxes come from visible mask pixels.
- Pose JSON contains player COCO 17 `[x,y,v] * 17` records and ball records with `class: "ball"`, bbox, and `keypoints: null`. Use `v=0/1/2`; preserve a player record when its object is visible even if individual projections fail.
- MOTS rows have six whitespace-separated fields: frame track_id class_id image_height image_width RLE. Use compact JSON for the RLE field if needed.
- Only objects with visible mask pixels are emitted, and all three canonical files must have the same `(frame_id, track_id)` set per sequence. Sort records deterministically.
- Manifest includes schema version, episode_id, trajectory_id equal to episode_id, sequence list, frame count, dimensions, modalities, public classes, and track policy.
- Use existing atomic writing helpers. Keep comments/docstrings in simplified Chinese. Do not modify config systems, batch/split/assembler, animation, Camera Manager, or delete existing converter capabilities.

Tests must cover RLE round trip, stable IDs including ball, player+ball canonical output, invisible/empty frames, invalid keypoints, non-square masks, bbox clipping, manifest trajectory ID, and cross-modal identity-set equality. Use stubs/mocks for EXR if necessary so tests run without Unreal.

Run `uv run pytest tests/test_public_episode.py -q` and include the command and result in the report.

Write the complete implementation report to `.superpowers/sdd/2026-09-02-single-episode-public-output/task-1-report.md`; return only status, commits, one-line test summary, and concerns. Do not dispatch subagents or reviewers. Do not push.
