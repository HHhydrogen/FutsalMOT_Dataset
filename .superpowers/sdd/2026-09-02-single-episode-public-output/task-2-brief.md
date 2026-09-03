# Task 2: Switch RGB Rendering and Sequence Helpers to JPEG

Read this first — it is the exact task requirements.

Modify `ue/render_episode.py` and `ue/dataset_export.py`; update `tests/test_render_export.py` as needed.

Requirements:

- Public RGB destination is `img1/{frame:06d}.jpg`.
- Frame discovery accepts `.png`, `.jpg`, and `.jpeg`; preserve FinalImage prefix preference.
- If source is JPG/JPEG, copy directly. If source is PNG during legacy recovery, convert once to RGB JPEG at quality 95; use a temporary file and `os.replace` so conversion is atomic.
- `build_seqinfo` defaults to `imExt=.jpg`.
- Configure Unreal Engine 5.8 MRQ RGB output with the JPEG image-sequence output setting where available, quality default 95, while leaving Object-ID EXR output unchanged. Keep defensive error behavior for unavailable API names/properties.
- Existing PNG render recovery remains supported.
- Update render counters/discovery to handle all RGB suffixes, while public `img1` checks expect `.jpg`.
- Do not modify config systems or remove existing converters.

Tests must cover JPEG discovery, direct JPEG copy to `img1`, PNG legacy conversion to JPEG, seqinfo `.jpg`, and retain existing render tests.

Run `uv run pytest tests/test_render_export.py tests/test_render_preset.py -q`; include command/result in the report.

Write report to `.superpowers/sdd/2026-09-02-single-episode-public-output/task-2-report.md`; return only status, commits, one-line tests, and concerns. Do not dispatch subagents or reviewers. Do not push.
