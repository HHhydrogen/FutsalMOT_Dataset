# Task 4: Remove Hardcoded Capability Declarations

Read the plan first: `docs/superpowers/plans/2026-09-03-public-contract-closure.md`.

Update `artifact_cleanup.py`, `dataset_manifest.py`, `task_status.py`, and resolver/manifest/status tests so manifest/status capabilities reflect resolved task `config_v3.annotations` and `config_v3.classes`, not hardcoded player+ball+pose. Preserve frozen public GT schemas and v2 fallback behavior.

Add failing tests for dynamic mot-only/player-only and mot+mots/player+ball declarations. Implement minimal propagation of requested annotations/classes, actual public modalities and class declarations, camera mapping, and cleanup status. Run focused and full tests, write report to SDD directory, commit Simplified Chinese message, do not push or dispatch agents.
