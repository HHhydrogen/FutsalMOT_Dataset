# Task 3: Make Cleanup Gates Conditional

Read the plan first: `docs/superpowers/plans/2026-09-03-public-contract-closure.md`.

Update `workflows/artifact_cleanup.py` and cleanup tests so gates derive from requested annotations/classes in resolved `config_v3`/manifest. Only tasks whose requested annotations include `pose` require `pose_session.json` and pose capture completion. Valid `mot-only` and `mot+mots` tasks must be cleanable without Pose files. Other render/mask/annotation gates should likewise be conditional on requested annotations/classes; preserve canonical protection and existing failed gates.

Write failing mot-only and mot+mots tests, run them, implement minimal changes, run cleanup-focused/full tests, write report to the SDD directory, commit Simplified Chinese message, and do not push or dispatch agents.
