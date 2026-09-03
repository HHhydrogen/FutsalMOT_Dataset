# Task 4: Wire Postprocess, Cleanup, and Dataset Manifest

Read this first — exact requirements.

Modify `src/grf_ue_bridge/workflows/task_postprocess.py`, `artifact_cleanup.py`, `dataset_manifest.py`, and relevant tests.

Requirements:

- Default task postprocess uses `public_episode.write_public_episode` from EXR source and `public_validator.validate_public_episode`; it must not require or create `mask/*.png`, YOLO det/seg/pose, debug overlay/video, or duplicate internal public labels.
- Preserve explicit legacy annotate/YOLO/Mask/Cryptomatte/debug commands and opt-in behavior.
- Default output is canonical JPG `img1`, `gt/gt.txt`, `gt/gt_pose.json`, `gt/gt_mots.txt`, `seqinfo.ini`, and `episode_manifest.json`.
- Cleanup never deletes those canonical files. It may delete EXR/render scratch, internal JSONL, and explicit duplicate labels only after public validation/audit passes. Dry-run remains default; failed validation blocks apply.
- `dataset_manifest.py` must count JPG as final RGB, account for canonical public modalities, carry trajectory/sequence/public class policy where compatible, and exclude temporary/debug from final checksum profiles. Do not implement multi-episode assembly.
- Existing tests and legacy cleanup behavior must remain compatible where no public manifest exists.

Add tests proving default postprocess/public output does not create duplicate mask/YOLO/debug artifacts, cleanup preserves canonical JPG/GT and removes transient EXR/render, cleanup blocks on failed public validation, and dataset manifest counts canonical JPG/public files.

Run `uv run pytest tests/test_task_cli.py tests/test_dataset_manifest.py tests/test_audit_fixes.py -q` and then `uv run pytest`; append exact results to `.superpowers/sdd/2026-09-02-single-episode-public-output/task-4-report.md`. Commit with concise Simplified Chinese message. Do not dispatch agents/reviewers or push.
