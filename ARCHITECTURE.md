# Code Architecture

The code directory is organized around stable numbered entry scripts plus a shared `futsalmot` package.

## Stable Entry Scripts

- `00_run_pipeline.py`: Windows-side single seed/template orchestration.
- `10_validate_episode.py`: validates event semantics and possession rules.
- `11_generate_random_episode.py`: generates 4v4 outfield random episodes.
- `12_compile_trajectory.py`: compiles event configs into dense trajectories.
- `13_enhance_trajectory.py`: adds yaw, actions, ball state, and contact metadata.
- `14_validate_trajectory.py`: validates dense trajectory safety.
- `20_build_sequences.py`: Unreal-side Sequencer and annotation export.
- `21_preflight.py`: Unreal-side read-only preflight.
- `22_scan_animations.py`: Unreal-side animation asset scan.
- `23_ue_setup_8_players.py`: Unreal-side one-time 8-player scene setup.
- `30_convert_and_check.py`: postprocess, YOLO/MOT conversion, overlay, integrity check.
- `31_generate_event_annotations.py`: event and frame-state annotation export.

## Shared Package

- `futsalmot/core/paths.py`: canonical code/project/config/output paths.
- `futsalmot/core/io.py`: atomic text/JSON writing and JSON loading.
- `futsalmot/core/hashing.py`: artifact hashing.
- `futsalmot/core/process.py`: subprocess execution with logs.
- `futsalmot/pipeline/constants.py`: reusable pipeline script names and template names.
- `futsalmot/ue/`: reserved for Unreal helpers after runtime compatibility is proven.

## Refactor Rules

- Keep numbered files as public entry points unless all docs and UE console commands are updated together.
- Move pure-Python shared logic into `futsalmot/` when it has at least two callers or clarifies a stable boundary.
- Avoid moving Unreal runtime logic aggressively; import behavior inside Unreal Editor must be validated before relying on package modules there.
- Generated data belongs under `configs/events/generated/`, `Saved/FutsalMOT/`, or `_agent_test_outputs/`, not inside shared modules.
