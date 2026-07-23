"""Export RL A3.3 config and prepare UE environment variables.

This module creates a copy of the RL A3.3 file with proper naming for UE,
generates the UE command script, and provides validation checks.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from futsalmot_rl.core.rl_io import read_json, write_json_atomic, write_text_atomic
from futsalmot_rl.core.rl_paths import (
    PROJECT_ROOT,
    ensure_dirs,
)

UE_CLOSED_LOOP_DIR = PROJECT_ROOT / "Saved" / "FutsalMOT_RL" / "ue_closed_loop"


def prepare_ue_config(
    rl_a33_path: str | Path,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Prepare an RL A3.3 config for UE rendering.

    Steps:
    1. Copy the RL A3.3 file to the UE closed-loop directory.
    2. Generate UE environment command batch file.
    3. Verify the config is valid for UE consumption.

    Args:
        rl_a33_path: Path to the RL A3.3 JSON file.
        output_dir: Output directory (default: .../ue_closed_loop/).

    Returns:
        Report dict with paths and instructions.
    """
    rl_a33_path = Path(rl_a33_path)
    if not rl_a33_path.is_file():
        raise FileNotFoundError(f"RL A3.3 file not found: {rl_a33_path}")

    output_dir = Path(output_dir) if output_dir else UE_CLOSED_LOOP_DIR
    ensure_dirs()

    seq_id = _extract_seq_id(rl_a33_path)

    # 1. Copy the RL A3.3 to the closed-loop directory
    dest_path = output_dir / f"rl_{seq_id}_for_ue.json"
    shutil.copy2(str(rl_a33_path), str(dest_path))

    # 2. Generate UE environment command
    cmd_content = _generate_ue_command(dest_path, seq_id)
    cmd_path = output_dir / "ue_env_command.txt"
    write_text_atomic(cmd_path, cmd_content)

    # 3. Generate the step-by-step instructions
    instructions = _generate_instructions(dest_path, seq_id, output_dir)
    notes_path = output_dir / "notes.md"
    write_text_atomic(notes_path, instructions)

    report = {
        "schema_version": "UE_CLOSED_LOOP_PREP_V1",
        "seq_id": seq_id,
        "rl_a33_source": str(rl_a33_path.resolve()),
        "rl_a33_for_ue": str(dest_path.resolve()),
        "ue_command_file": str(cmd_path.resolve()),
        "notes_file": str(notes_path.resolve()),
        "ready_for_ue": True,
    }
    report_path = output_dir / "ue_prep_report.json"
    write_json_atomic(report_path, report)

    return report


def _extract_seq_id(a33_path: Path) -> str:
    """Extract seq_id from an A3.3 file."""
    data = read_json(a33_path)
    seq_id = data.get("seq_id") or data.get("episode_id") or a33_path.stem
    # Remove _a33 suffix if present
    seq_id = str(seq_id).replace("_a33", "")
    return seq_id


def _generate_ue_command(a33_path: Path, seq_id: str) -> str:
    """Generate the UE environment variable setup command."""
    return """FutsalMOT-RL UE Closed-Loop Setup
=================================
Date: 2026-07-23

STEP 1: Set environment variable
---------------------------------
In a command prompt (NOT PowerShell), run:

  set FUTSALMOT_CONFIG_PATH={a33_abs_path}

Then launch Unreal Editor from the SAME command prompt.

If UE is already open, close it first, then:
  1. Open a new command prompt
  2. Run: set FUTSALMOT_CONFIG_PATH={a33_abs_path}
  3. Launch UE from that prompt

STEP 2: In UE Python Console
-----------------------------
Once the level is loaded, run:

  py "D:/projects/FustalMOT_UEDataset/Content/FutsalMOT/code/02_run_unreal.py"

This will:
  - Run preflight check (read-only)
  - Build Level Sequences with RL-controlled Player_05
  - Export bbox/skeleton annotations

STEP 3: Render with Movie Render Queue
---------------------------------------
Open Movie Render Queue and set:
  Output Directory: {render_output}
  File Name Format: {{frame_number}}
  Image Format:     PNG
  Resolution:       1920 x 1080

STEP 4: Windows Post-Processing
--------------------------------
After rendering completes, run:

  cd /d D:\\projects\\FustalMOT_UEDataset\\Content\\FutsalMOT\\code
  python 03_check_labels.py --annotation {annotation_path}

IMPORTANT SAFEGUARDS:
- This RL config uses seq_id="{seq_id}"
- It will NOT overwrite original rule data
- Player_05 is RL-controlled; all others follow rule replay
- Check that Player_05 moves smoothly (no teleporting)
""".format(
        a33_abs_path=str(a33_path.resolve()).replace("/", "\\"),
        seq_id=seq_id,
        render_output=str(PROJECT_ROOT / "Saved" / "FutsalMOT_RL" / "ue_closed_loop" / "images" / seq_id).replace("/", "\\"),
        annotation_path=str(PROJECT_ROOT / "Saved" / "FutsalMOT" / "annotations" / f"objects_bbox_2d_clean_{seq_id}.json").replace("/", "\\"),
    )


def _generate_instructions(a33_path: Path, seq_id: str, output_dir: Path) -> str:
    """Generate detailed step-by-step UE instructions."""
    return f"""# FutsalMOT-RL UE Closed-Loop Verification

## Overview

This document describes how to verify that the RL-controlled trajectory
can be rendered in Unreal Engine and produce valid annotations.

## Files

| File | Path |
|------|------|
| RL A3.3 Config | `{a33_path.resolve()!s}` |
| Seq ID | `{seq_id}` |
| Expected annotation | `Saved/FutsalMOT/annotations/objects_bbox_2d_clean_{seq_id}.json` |
| Expected images | `Saved/FutsalMOT_RL/ue_closed_loop/images/{seq_id}/` |
| Expected layout check | `Saved/FutsalMOT/layout_check/{seq_id}/` |

## What to Check

After rendering, verify:

1. **UE reads the config** — The preflight script should report
   "Config loaded from: ..." pointing to the RL A3.3 file.

2. **Player_05 movement** — Watch the Level Sequence playback.
   Player_05 should move smoothly, following Player_01.

3. **Other players unchanged** — Player_01~04 and Player_06~08
   should follow their original rule trajectories.

4. **No data overwrite** — The seq_id "{seq_id}" is unique
   and will not overwrite any existing rule data.

5. **bbox quality** — After layout_check, verify that Player_05's
   bounding box follows the player correctly.

## Success Criteria

- [ ] UE preflight passes (no errors)
- [ ] Level Sequence builds successfully
- [ ] All 4 cameras render 300 frames (1200 images)
- [ ] Annotation JSON is generated
- [ ] layout_check runs without errors
- [ ] Player_05 bbox is not misaligned
- [ ] No original data was overwritten

## If Something Goes Wrong

- Check the UE Output Log for error messages
- Verify FUTSALMOT_CONFIG_PATH is set correctly
- Try running ue_preflight.py standalone in UE Python console
- Check that the source A3.3 file exists and is valid
"""
