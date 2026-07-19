# FutsalMOT Code Package

This package contains shared Python code used by the three public pipeline entry scripts in the parent directory.

The root entry scripts remain the stable public API because project docs, Unreal Python console commands, and existing runbooks call them directly.

## Layout

- `core/`: filesystem paths, atomic IO, hashing, and logged subprocess execution.
- `pipeline/`: pipeline constants and orchestration-facing helpers.
- `scripts/`: internal implementations behind the public root entry scripts.
- `ue/`: reserved for Unreal-specific shared helpers. UE scripts should adopt this gradually because Unreal Python has stricter import/runtime constraints.

## Entry Point Policy

- Windows CLI users should call `01_generate_trajectories.py` and `03_check_labels.py`.
- Unreal Editor users should run `02_run_unreal.py` from the UE Python console.
- Shared code should move into `futsalmot/` only when it does not make UE execution less reliable.
