# FutsalMOT Dataset Code

This repository contains the Python code for the FutsalMOT Unreal Engine synthetic dataset pipeline. It is intended to live under an Unreal project at:

```text
Content/FutsalMOT/code
```

The full Unreal project assets, rendered images, and generated dataset outputs are not part of this code repository.

## Current Scope

The current pipeline generates short 4v4 outfield futsal episodes with no goalkeepers:

- Team A: `Player_01` to `Player_04`
- Team B: `Player_05` to `Player_08`
- Ball: `Ball_01`
- Objects per frame: 9
- Cameras: 4 fixed CineCameras
- Timeline: 10 seconds, 30 FPS, 300 frames
- Expected annotation records: `4 cameras * 300 frames = 1200`

The pipeline exports synchronized RGB metadata, tight player/ball bbox annotations, player actions, event/frame state annotations, possession metadata, and player skeleton 2D keypoints for pose-style training data.

## Public Entry Points

Only three Python files in the repository root are public entry points:

```text
01_generate_trajectories.py
02_run_unreal.py
03_check_labels.py
```

All implementation scripts live under `futsalmot/scripts/` and are considered internal.

## Step 1: Generate Trajectories On Windows

Run from this `code` directory:

```powershell
py .\01_generate_trajectories.py --seed 1 --template 1
```

Available templates:

| ID | Description |
|---:|---|
| 1 | 4v4 solo dribble and shot, with support, anchor, and marking movement |
| 2 | 4v4 dribble, pass, receive, with support run and defensive follow |
| 3 | 4v4 pass, receive, dribble, shot, with weak-side support and cover |

This step runs the Windows-side pipeline:

```text
generate random episode
-> validate episode
-> compile dense trajectory
-> enhance yaw/action/ball state/contact metadata
-> validate dense trajectory
-> generate event/frame-state annotations
-> update configs/pipeline_current.json
```

Key outputs:

```text
configs/events/generated/<seq_id>.json
configs/events/generated/<seq_id>_a32.json
configs/events/generated/<seq_id>_a33.json
configs/pipeline_current.json
_agent_test_outputs/pipeline_<seq_id>/
```

## Step 2: Run In Unreal Editor

Run from the Unreal Editor Python console:

```python
py "D:/projects/FustalMOT_UEDataset/Content/FutsalMOT/code/02_run_unreal.py"
```

This step runs read-only preflight checks first, then builds/updates the Level Sequence and exports bbox/keypoint annotations.

Expected core output:

```text
Saved/FutsalMOT/annotations/objects_bbox_2d_clean_<seq_id>.json
Saved/FutsalMOT/annotations/objects_bbox_2d_clean_<seq_id>.jsonl
```

For each player object, annotation records include:

```text
bbox_2d_clean
bbox_xyxy_clean
```

`keypoints_2d_yolo` uses the YOLO pose-style flat format:

```text
x_norm, y_norm, visibility, x_norm, y_norm, visibility, ...
```

Visibility values:

```text
0 = missing or behind camera
1 = in front of camera but outside image
2 = inside image
```

## Step 3: Check Labels On Windows

After MRQ renders images to `Saved/FutsalMOT/images_clean/<seq_id>/`, run:

```powershell
py .\03_check_labels.py --annotation "D:/projects/FustalMOT_UEDataset/Saved/FutsalMOT/annotations/objects_bbox_2d_clean_<seq_id>.json"
```

To draw keypoints on overlay images:

```powershell
py .\03_check_labels.py --draw-keypoints
```

Expected outputs:

```text
Saved/FutsalMOT/overlay_objects_bbox_<seq_id>/
Saved/FutsalMOT/labels_yolo_clean/<seq_id>/
Saved/FutsalMOT/labels_mot_clean/<seq_id>/
Saved/FutsalMOT/annotations/manifest_<seq_id>.json
```

Expected checks:

```text
records = 1200
yolo_files = 1200
expected_objects_per_record = 9
CHECK PASSED
ALL DONE
```

## One-Time 8 Player Scene Setup

If the UE level still contains only `Player_01` to `Player_04`, run this once from the Unreal Editor Python console:

```python
py "D:/projects/FustalMOT_UEDataset/Content/FutsalMOT/code/futsalmot/scripts/ue_setup_8_players.py"
```

The script creates missing `Player_05` to `Player_08` from existing player templates. It does not auto-save the level; inspect the result and save manually.

## Repository Layout

```text
code/
├─ 01_generate_trajectories.py
├─ 02_run_unreal.py
├─ 03_check_labels.py
├─ configs/
├─ futsalmot/
│  ├─ core/
│  ├─ pipeline/
│  ├─ scripts/
│  └─ ue/
├─ pyproject.toml
└─ README.md
```

Important modules:

- `futsalmot/core/paths.py`: canonical code, config, project, and output paths.
- `futsalmot/core/io.py`: atomic JSON/text writes and JSON reading.
- `futsalmot/core/process.py`: subprocess execution with log capture.
- `futsalmot/pipeline/constants.py`: template names and internal script paths.
- `futsalmot/scripts/`: internal implementations behind the three public entry points.

## Validation Status

Current smoke checks pass at the code level:

```text
compileall: PASS
trajectory validation: WARNING, warnings=60, errors=0
```

The trajectory warnings are expected for the current seed/template baseline and are not validation errors.

UE runtime validation must be performed inside Unreal Editor because the `unreal` Python module is not available in normal Windows Python.
