# Code Architecture

The code directory is organized around three public root entry scripts plus a shared `futsalmot` package.

## Public Entry Scripts

- `01_generate_trajectories.py`: Windows-side generation and validation of trajectory configs.
- `02_run_unreal.py`: Unreal-side preflight plus Sequencer/annotation export.
- `03_check_labels.py`: Windows-side image/annotation integrity check and YOLO/MOT export.

## Internal Script Implementations

Implementation scripts live under `futsalmot/scripts/`. They are not the public API, but they remain directly executable for debugging.

## Shared Package

- `futsalmot/core/paths.py`: canonical code/project/config/output paths.
- `futsalmot/core/io.py`: atomic text/JSON writing and JSON loading.
- `futsalmot/core/hashing.py`: artifact hashing.
- `futsalmot/core/process.py`: subprocess execution with logs.
- `futsalmot/pipeline/constants.py`: reusable pipeline script names and template names.
- `futsalmot/ue/`: reserved for Unreal helpers after runtime compatibility is proven.

## Refactor Rules

- Keep only the three public root entry scripts in the code root.
- Move pure-Python shared logic into `futsalmot/` when it has at least two callers or clarifies a stable boundary.
- Avoid moving Unreal runtime logic aggressively; import behavior inside Unreal Editor must be validated before relying on package modules there.
- Generated data belongs under `configs/events/generated/`, `Saved/FutsalMOT/`, or `_agent_test_outputs/`, not inside shared modules.
