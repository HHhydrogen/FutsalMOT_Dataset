# Task 5: Clean Config v3 Documentation and Verify

Read the plan first: `docs/superpowers/plans/2026-09-03-public-contract-closure.md`.

Update README.md, configs documentation, CLAUDE.md and related public-output documentation to remove Config v3-labeled legacy postprocess/ball/repeated-field examples. Ensure v3 examples validate against current schema and document explicit camera mapping, dynamic manifest/status capabilities, conditional cleanup gates, and unchanged public GT format. Do not modify Config v3 main schema or public GT format.

Add a test that parses/validates the documented v3 example if practical. Run `uv run pytest`, `git diff --check`, and repository hygiene checks. Write report to the SDD directory and commit Simplified Chinese message. Do not push or dispatch agents.
