# Task 2: Add Resolved-task Public Audit Contract

Read the plan first: `docs/superpowers/plans/2026-09-03-public-contract-closure.md`.

Strengthen public audit so it validates the public manifest/output against the resolved v3 task contract, not only internally. The audit must compare exact expected camera IDs, public sequence names, expected frames per camera, requested annotations, requested classes, FPS, and resolution. Any missing or mismatched value must return audit FAIL. Preserve legacy audit fallback when no public v3 contract exists.

Use the existing `config_v3` block in resolved_task.json or an equivalent explicit contract; do not create a second user schema. Add a resolved-task input/CLI path only as needed, preserving existing task audit behavior. Add failing tests for one missing camera, short frame set, annotation mismatch, class mismatch, FPS mismatch and resolution mismatch; then implement and run focused/full tests. Write report to `.superpowers/sdd/2026-09-03-public-contract-closure/task-2-report.md`, commit Simplified Chinese message, do not push or dispatch agents.
