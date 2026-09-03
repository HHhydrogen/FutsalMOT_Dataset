# SDD ledger — plan: docs/superpowers/plans/2026-09-03-public-contract-closure.md

## Preflight scan

| Item | Shared files/interfaces | Finding | Ruling |
|---|---|---|---|
| Task 1 / Task 2 | resolved `config_v3` camera/sequence summary | Task 2 consumes explicit identity produced by Task 1. | Proceed in order. |
| Task 1 / Task 4 | manifest/status camera and capability fields | Task 4 consumes resolver-derived fields. | Proceed in order. |
| Task 2 / Task 3 | requested annotations/classes and audit result | Cleanup gate depends on audit/requested modalities. | Proceed in order. |
| Task 2 / Task 5 | audit CLI/docs | Task 5 documents Task 2 behavior. | Proceed in order. |
| Task 3 / Task 4 | cleanup gates and dynamic capabilities | Both touch lifecycle declarations; keep conditional logic centralized. | Proceed in order. |
| Task 4 / Task 5 | manifest/status output documentation | Task 5 documents dynamic declarations. | Proceed in order. |
| Task 1 | camera identity tests vs resolver/writer | Required C03/C07 and duplicate actor behaviors are explicit. | Self-consistent. |
| Task 2 | audit tests vs resolved contract | Required mismatch cases are explicit. | Self-consistent. |
| Task 3 | cleanup tests vs requested annotations | mot-only/mot+mots cases are explicit. | Self-consistent. |
| Task 4 | manifest/status tests vs resolved values | Dynamic annotations/classes are explicit. | Self-consistent. |
| Task 5 | docs vs final code | Documentation follows final behavior. | Self-consistent. |

## Rulings

- Ruling: use the existing `config_v3` summary as the public audit contract rather than introducing a second config schema — it already crosses resolver/CLI boundaries and minimizes changes; cost if wrong: future resolved schema may need a dedicated contract block.
- Ruling: preserve legacy audit fallback when no public manifest/resolved v3 metadata exists — required for v2 compatibility; cost if wrong: legacy output may not receive new strict checks until migrated.
Task 1: fix round 1/5 started — reviewer found the camera test did not exercise the explicit public_sequence_name writer input.
Task 1: fix round 1/5 (1 addressed, 0 open; commits 4fb769d..2385bc5)
Task 1: complete (commits 94dc99e..2385bc5, review clean)
Task 2: minor (deferred): malformed manifest sequence entries could crash resolved-contract audit; add CLI-level v3 contract mismatch regression test if touched later.
Task 2: complete (commits 2385bc5..e0ace9a, review clean with deferred minors)
Task 3: fix round 1/5 started — reviewer found v2 cleanup gates bypassed when empty config_v3 was passed and synthetic v3 cleanup tests.
Task 3: fix round 1/5 (2 addressed, 0 open; commits ccbe242..a467be9)
Task 3: fix round 2/5 started — re-review found real public v3 mot/mot+mots fixture coverage missing.
Task 3: fix round 2/5 (1 addressed, 0 open; commits a467be9..2fdfc72)
Task 3: complete (commits e0ace9a..2fdfc72, review clean)
Task 4: fix round 1/5 started — reviewer found dataset_manifest not consuming resolved capabilities, player-only false ball policy, weak tests, and incomplete partial-v3 fallback.
Task 4: fix round 1/5 (5 addressed, 0 open; commits 4d8b02f..dec66f5)
Task 4: fix round 2/5 started — re-review found missing-value partial-v3 capability fallback incorrectly defaulting to player-only.
Task 4: fix round 2/5 (1 addressed, 0 open; commits dec66f5..0f325bb)
Task 4: complete (commits 2fdfc72..0f325bb, review clean)
Task 5: complete (commits 0f325bb..934fa53, review clean with deferred minors)
Task 2: minor (deferred): malformed manifest sequence entries could crash resolved-contract audit; add CLI-level v3 contract mismatch regression test if touched later.
Task 5: minor (deferred): documentation test does not assert all prose invariants; report omits the unavailable real local-config resolve limitation.
Task 4: fix round 1/5 started — reviewer found dataset_manifest not consuming resolved task capabilities, player-only false ball policy, weak manifest/status sequence/cleanup tests, and incomplete v3 capability fallback.
Task 4: fix round 1/5 (5 addressed, 1 open; commits 4d8b02f..dec66f5)
Task 4: fix round 2/5 started — re-review found missing-value partial-v3 capability fallback incorrectly defaults to player-only.
Task 3: fix round 1/5 started — reviewer found CLI v2 cleanup bypasses legacy render/Pose gates when empty config_v3 is passed; conditional gate tests are too synthetic.
