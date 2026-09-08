# P3-3 Trajectory Source Feasibility and GRF Replacement Design

## Status

Approved in chat by the project owner before implementation planning.

## Problem Statement

FutsalMOT currently uses Google Research Football `5_vs_5` with GRF `builtin_ai` as its primary trajectory source. P3-1 established that the observed low-motion behavior is present in GRF output and that the tested single-agent control-wiring intervention does not improve it. P3-2 evaluated candidate root seeds `42-59` with 300 frames at 10 FPS and found only `1/18` strict-valid trajectories under the existing Motion Quality selection semantics.

The project must therefore evaluate whether GRF builtin_ai is still suitable as the default trajectory source, without assuming that the existing dependency should be retained merely because the current pipeline already consumes it.

This phase evaluates source feasibility and produces an architecture decision. It does not perform a full GRF replacement or modify the production trajectory generator.

## Goals

- Characterize P3-2 failure frequency and failure combinations without changing the Motion Quality Gate.
- Evaluate the current GRF builtin_ai source as a baseline.
- Verify whether a materially different GRF-native source is available in the pinned external repositories and can be exercised with bounded cost.
- Evaluate UE-native deterministic generation as an architectural option without implementing a full UE football simulator.
- Evaluate a lightweight project-owned trajectory prototype as a possible replacement source without building a tactical AI framework.
- Compare candidates using motion quality, determinism, controllability, futsal plausibility, integration cost, contract compatibility, reproducibility, maintenance risk and dataset value.
- Produce one explicit decision: `KEEP_GRF`, `GRF_OPTIONAL`, `REPLACE_GRF` or `INSUFFICIENT`.

## Non-Goals

- No modification to `grf_runner.py`, current production generator, action wiring, difficulty, scenario, field dimensions, seed derivation or Motion Quality thresholds.
- No seed-specific hacks, automatic retry, invalid-seed hiding, artificial velocity injection or post-hoc frame repair.
- No complete football tactical AI, behavior-tree framework, simulation engine, trajectory manager, registry, scheduler or database.
- No simultaneous refactor of GRF, UE, Camera, annotation or pipeline contracts.
- No UE annotation, MRQ or Camera Distribution validation unless a candidate cannot be evaluated without a minimal UE-only probe.
- No full migration even if the decision is `REPLACE_GRF`.

## Current Contract Boundary

The trajectory source is separated from downstream playback and dataset generation by existing artifacts:

```text
trajectory source
    -> meta.json + frames.jsonl
    -> existing UE playback / Level Sequence generation
    -> existing Camera / annotation / MOT pipeline
```

The preferred replacement boundary is the source layer only. The following should remain reusable unless experiments provide concrete evidence otherwise:

- `frames.jsonl` episode and frame representation;
- fixed entity IDs `L0..L4`, `R0..R4`, `BALL`;
- `meta.json` timing, field and entity metadata;
- `futsalmot_seed_v1` or an explicitly compatible deterministic seed contract;
- actor mapping;
- UE actor-state playback and Level Sequence generation;
- Camera Distribution;
- annotation and MOT GT generation;
- Run Manifest, Pipeline State, ValidationResult and Audit/Cleanup contracts.

## Candidate Sources

### Current GRF builtin_ai

Use the existing `5_vs_5` path as the baseline. Reuse P3-2 results for the initial failure characterization. The baseline is not granted default-source status by historical use.

### GRF-native Alternatives

Inspect only capabilities actually present in the pinned Google Research Football and GRF_MARL repositories:

- GRF `bot` player path;
- replay player path;
- PPO/checkpoint player path, only if a usable checkpoint and required runtime are already available;
- GRF multi-agent policy paths;
- GRF_MARL pretrained or rollout policy paths, only if they can run in the current environment without introducing a large unplanned dependency migration.

The evaluation must distinguish a real runnable source from a documented but unavailable option. No action shape, checkpoint, policy or behavior may be inferred without API and artifact evidence.

### UE-native deterministic source

Inspect the current UE project for existing AIController, navigation, CharacterMovement, behavior-tree or deterministic trajectory facilities. Assess whether they could generate the required 5v5 state, ball interaction and reproducible episode. No full implementation is required in this phase.

### Lightweight project-owned source

Assess a throwaway Python-only prototype that emits the existing trajectory contract. It may use deterministic seeded target/movement rules, team separation, bounded possession and ball/player interaction. It must not be reduced to independent random walks and must not become a new general simulation framework.

### External simulator or policy source

Record only clearly reusable sources already present in the repository or existing dependencies. Do not perform a broad third-party migration.

## Evaluation Criteria

Each candidate is evaluated against the same criteria:

1. Motion Quality: ability to satisfy the unchanged existing Motion Quality Gate over the bounded experiment.
2. Determinism: identical seed and settings produce identical trajectory artifacts.
3. Controllability: player movement, activity level, possession behavior and episode duration are controllable without invalid-seed filtering.
4. Futsal plausibility: 5v5 spatial separation, ball/player interaction, attacking/defending movement and non-random behavior.
5. Integration cost: changes required to trajectory representation, `frames.jsonl`, actor mapping, UE playback, annotation and Camera pipeline.
6. Contract compatibility: ability to reuse the existing artifact and downstream contracts.
7. Reproducibility: suitability for paper and dataset regeneration.
8. Maintenance risk: dependence on opaque, difficult-to-debug or unmaintained behavior.
9. Dataset value: diversity and sustained MOT observations, not merely gate passage.

## Failure Characterization

Use the existing P3-2 candidate results for seeds `42-59`. Report, without changing thresholds:

- failure frequency for outfield active ratio;
- failure frequency for outfield stationary streak;
- failure frequency for team active coverage;
- severe global low-motion cases using the observed plateau metric;
- combinations of failed criteria;
- GK stationary streak as a recorded diagnostic metric only, not a newly blocking criterion.

Classify failures as severe global low-motion, multi-criterion failure, single-criterion near miss or other. The classification is descriptive and does not alter validity.

## Feasibility Experiments

The experiment set is limited to:

- baseline: current GRF builtin_ai;
- at most one runnable GRF-native alternative;
- at most one lightweight project-owned prototype if it is justified after the GRF-native probe.

The bounded experiment uses:

```text
scenario concept: 5_vs_5
frames: 300
FPS: 10
seeds: 42, 43, 44, 45
```

Each runnable candidate must produce or be mapped to the existing `frames.jsonl` contract and be evaluated with the existing Motion Quality Audit. Record strict-valid count, motion metrics, deterministic repeat result, generation time and contract mapping status. Do not run Camera, annotation or MRQ for Python-side candidates.

If a candidate cannot be run because its checkpoint, runtime or dependency is unavailable, record `INFEASIBLE_IN_CURRENT_ENVIRONMENT` with evidence and do not manufacture a result.

## Decision Rules

### KEEP_GRF

Choose only if an existing, verified GRF path can materially improve stability without large patches or unacceptable maintenance cost, while retaining determinism, controllability and current contract compatibility.

### GRF_OPTIONAL

Choose if GRF retains research or comparison value but current builtin_ai is not sufficiently reliable as the default dataset source, and a replacement is not yet sufficiently validated for an immediate migration.

### REPLACE_GRF

Choose only if a candidate is materially better than GRF on motion quality, determinism, controllability and contract compatibility, with controlled implementation complexity and credible futsal plausibility. Output only a minimal migration design; do not execute the migration.

### INSUFFICIENT

Choose if no candidate has enough evidence to support a responsible source decision.

## Expected Report

Create:

```text
docs/p3-3-trajectory-source-feasibility-and-grf-replacement.md
```

The report must contain:

- Problem Statement;
- P3-2 Failure Characterization;
- Candidate Trajectory Sources;
- Evaluation Criteria;
- Feasibility Experiments;
- Results;
- GRF Assessment;
- Replacement Assessment;
- Contract Impact;
- Decision;
- Recommended Next Phase.

It must explicitly answer whether current GRF builtin_ai is suitable as the default, whether a reliable GRF-native path exists, whether UE-native generation is realistic, whether a lightweight project-owned generator is realistic, whether GRF should be formally replaced, which contracts can be retained, and whether the next phase should be GRF repair or Replacement Prototype.

## Verification and Stop Conditions

- Verify all reported experiment results from fresh outputs.
- Do not claim a candidate is feasible based only on documentation.
- Do not modify production defaults during this phase.
- If the decision is `REPLACE_GRF`, stop after the migration design.
- Stop after the report and decision; do not automatically begin the next phase.
