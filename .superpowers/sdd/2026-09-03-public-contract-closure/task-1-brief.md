# Task 1: Preserve Explicit Camera Identity

Read the plan first: `docs/superpowers/plans/2026-09-03-public-contract-closure.md`.

In the current worktree, make v3 camera mapping explicit end-to-end. Preserve three concepts separately: public `camera_id` (C##), UE `camera_actor`, and `public_sequence_name`. Given `{"C03":"FrontCamera","C07":"GoalCamera"}`, resolver/writer must create sequence names `FutsalMOT_<episode>_C03` and `FutsalMOT_<episode>_C07`; never infer public IDs from actor names or config order. Reject duplicate UE actor values.

Write failing tests first, run them, then implement minimal changes in `config/models.py`, `config/resolver.py`, `public_episode.py` and relevant tests. Keep v3 schema shape and frozen public GT fields unchanged. Run focused camera/resolver/public tests and full `uv run pytest`; write report to `.superpowers/sdd/2026-09-03-public-contract-closure/task-1-report.md`; commit with Simplified Chinese message. Do not push or dispatch further agents.
