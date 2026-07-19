# FutsalMOT Code Package

This package contains shared Python code used by the numbered pipeline entry scripts in the parent directory.

The numbered scripts remain the stable public entry points because project docs, Unreal Python console commands, and existing runbooks call them directly.

## Layout

- `core/`: filesystem paths, atomic IO, hashing, and logged subprocess execution.
- `pipeline/`: pipeline constants and orchestration-facing helpers.
- `ue/`: reserved for Unreal-specific shared helpers. UE scripts should adopt this gradually because Unreal Python has stricter import/runtime constraints.

## Entry Point Policy

- Windows CLI users should keep calling `00_run_pipeline.py`, `30_convert_and_check.py`, and related numbered scripts.
- Unreal Editor users should keep running `20_build_sequences.py`, `21_preflight.py`, `22_scan_animations.py`, and `23_ue_setup_8_players.py` from the UE Python console.
- Shared code should move into `futsalmot/` only when it does not make UE execution less reliable.
