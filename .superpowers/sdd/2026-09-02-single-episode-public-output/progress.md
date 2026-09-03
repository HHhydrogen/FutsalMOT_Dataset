# SDD ledger — plan: docs/superpowers/plans/2026-09-02-single-episode-public-output.md

## Preflight scan

| Item | Shared files/interfaces | Finding | Ruling |
|---|---|---|---|
| Task 1 / Task 2 | public writer consumes rendered images; JPEG helpers are independent | No direct code dependency; Task 1 can use existing image paths and Task 2 later changes output suffixes. | Proceed in order; writer accepts the final image directory contract. |
| Task 1 / Task 3 | `write_public_episode` produces files consumed by `validate_public_episode` | Interface is explicit and consistent. | Proceed. |
| Task 1 / Task 4 | canonical writer is called by postprocess; cleanup protects its outputs | Task 4 depends on Task 1 output names and manifest fields. | Proceed in order. |
| Task 2 / Task 3 | JPG output is validated by public validator | Task 3 must validate `.jpg`; Task 2 establishes it. | Proceed in order. |
| Task 2 / Task 4 | postprocess and cleanup count/copy JPG | Same public suffix contract. | Proceed in order. |
| Task 3 / Task 4 | public audit gates cleanup | Task 4 consumes validator result through audit. | Proceed in order. |
| Task 4 / Task 5 | workflow output and documentation | Task 5 verifies the integrated contract. | Proceed in order. |
| Task 1 | writer tests against writer files | Planned tests cover RLE, IDs, visibility, and canonical output. | Self-consistent. |
| Task 2 | render tests against render helpers | Planned tests cover JPEG discovery and destination; UE API requires runtime smoke. | Self-consistent. |
| Task 3 | validator tests against validator files | Planned fixture helpers must create all required canonical files. | Self-consistent. |
| Task 4 | workflow tests against cleanup/manifest files | Planned fixture requires public validation to pass before cleanup. | Self-consistent. |
| Task 5 | smoke commands against integrated output | Uses existing config and UE MCP workflow. | Self-consistent. |

## Rulings

- Ruling: Pose JSON keeps football records with `class="ball"` and `keypoints=null` — the approved spec requires football in all three canonical identity sets while COCO 17 points apply only to players — cost if wrong: downstream Pose consumers may need a class-aware parser.
- Ruling: `trajectory_id` is `episode_id` — explicitly confirmed by the user — cost if wrong: future multi-episode grouping would need a migration field.
- Ruling: worktree is external under the approved temporary directory because `.worktrees` is not ignored and automatic gitignore commits are prohibited — cost if wrong: the branch must be integrated manually later.

Task 1: fix round 1/5 started — reviewer found package import failure, incorrect EXR ordinal frame selection, off-screen Pose visibility, missing public mapping enforcement, non-deterministic sequence order, non-atomic JPEG writes, and insufficient edge-case tests.
Task 1: fix round 1/5 (7 addressed, 0 open; commits 46e5b36..e439482)
Task 1: complete (commits 8b7ed46..e439482, review clean)
Task 2: fix round 1/5 started — reviewer found PNG-only public counters/audits/regression/resolution validation and non-recursive RGB cleanup.
Task 2: fix round 1/5 (3 addressed, 1 open; commits 4f1b80c..281abfe)
Task 2: fix round 2/5 started — re-review found legacy annotation/pose validators and cleanup still PNG-only, stale manifest docs, and 9 full-suite regressions from making legacy regression JPG-only.
Task 2: fix round 2/5 (5 addressed, 0 open; commits 281abfe..f860441)
Task 2: fix round 3/5 started — re-review found stale task_audit Markdown statistic keys and task_status img1 legacy suffix reporting.
Task 2: fix round 3/5 (2 addressed, 0 open; commits f860441..2b08b50)
Task 2: complete (commits e439482..2b08b50, review clean)
Task 3: fix round 1/5 started — reviewer found validator crash paths, missing MOTS/image dimension binding, unbounded frame IDs, weak manifest/numeric validation, noncanonical JPG names, and misleading public audit camera statistics.
Task 3: fix round 1/5 (5 addressed, 3 open; commits 80fbd05..f97f413)
Task 3: fix round 2/5 started — re-review reproduced malformed manifest frame_count and malformed seqinfo parser crashes, plus empty public audit camera statistics.
Task 3: fix round 2/5 (3 addressed, 0 open; commits f97f413..1a7e2fb)
Task 3: complete (commits 2b08b50..1a7e2fb, review clean)
Task 4: fix round 1/5 started — reviewer found public cleanup bypassing existing audit failures, stale PNG/JPEG duplicates left by canonical writer, and insufficient real-path/legacy-opt-in tests.
Task 4: fix round 1/5 (4 addressed, 0 open; commits b6392cd..80178f8)
Task 4: fix round 2/5 started — re-review found normalized JPEG frame-name collision could silently overwrite an existing canonical frame.
Task 4: fix round 2/5 (0 addressed, 1 open; commits 80178f8..d45e327)
Task 4: fix round 3/5 started — final re-review found collision detection occurs after GT files are overwritten, leaving a partially modified episode.
Task 4: fix round 3/5 (1 addressed, 0 open; commits d45e327..5104906)
Task 4: complete (commits 1a7e2fb..5104906, review clean)
Task 5: fix round 1/5 started — reviewer found missing cleanup lifecycle documentation and ambiguous wording between public postprocess omission and cleanup deletion; documentation changes are working-tree-only by policy.
Task 5: fix round 1/5 (0 addressed, 2 open; documentation gate wording reviewed)
Final review: fix round 1/5 started — critical public layout/manifest schema mismatch; important cross-camera partial writes, invalid player Pose null keypoints, and missing exact sequence/track-policy validation; minor JPEG source re-encoding.
Final review: fix round 1/5 (5 addressed, 1 open; commits 5104906..a3646d1)
Final review: fix round 2/5 started — final re-review found manifest camera_id/modalities/track_id_policy mismatch and validator rejection of a conforming minimal manifest; docs also lacked exact schema.
Final review: fix round 2/5 (5 addressed, 0 open; commits a3646d1..01898bb)
Final review: parked — validator accepts extra manifest fields — Ruling: the approved specification says the manifest must record “at least” the listed fields, so forward-compatible optional fields are allowed; rejecting extras would unnecessarily prevent future aggregation metadata. Cost if wrong: strict consumers may need their own schema allowlist.
Task 5: complete (commits 5104906..01898bb, one parked non-blocking finding)
