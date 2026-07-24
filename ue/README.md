# UE Import Script — GRF Replay Preview

## Files

| File | Description |
|------|-------------|
| `actor_mapping.example.json` | Maps entity IDs (L0–L4, R0–R4, BALL) to UE actor names |
| `animation_config.example.json` | Example animation config — copy to `animation_config.local.json` and fill in your asset paths |
| `import_grf_episode.py` | UE Python script that imports episode data into a Level Sequence |

## Requirements

- Unreal Engine 5.x project with 10 player actors + 1 ball actor placed in the level
- No `gfootball`, no `.venv`, no GRF_MARL — pure UE Python + stdlib

## Quick Start (Transform Only)

1. Place actors in the UE level matching names in your mapping file.
2. Export a GRF episode:
   ```powershell
   uv run grf-ue export --config configs/mvp_builtin_5v5.json --output outputs/episode_0001
   ```
3. In Unreal Editor Python Console:
   ```python
   py "D:/projects/FustalMOT_UEDataset/Content/FutsalMOT/code/ue/import_grf_episode.py" --episode "D:/projects/FustalMOT_UEDataset/Content/FutsalMOT/code/outputs/episode_0001" --mapping "D:/projects/FustalMOT_UEDataset/Content/FutsalMOT/code/ue/actor_mapping.example.json" --replace-existing
   ```
4. Script creates a Level Sequence with Transform Tracks for all 10 players + ball.

## Locomotion Animation + Ball Rolling

To add automatic Idle/Walk/Run animation and ball rolling rotation:

### 1. Prepare In-Place Animations

Create three **In-Place Animation Sequences** compatible with your character's Skeleton:

- `Idle` — standing still
- `Walk_Fwd` — forward walk
- `Run_Fwd` — forward run

These must be **In-Place** animations (no Root Motion). Root Motion will cause the character to drift away from the Transform Track position.

### 2. Create Local Config

Copy `animation_config.example.json` to `animation_config.local.json`:

```powershell
cp ue/animation_config.example.json ue/animation_config.local.json
```

Edit `animation_config.local.json` and fill in the actual asset paths:

```json
{
  "animations": {
    "idle": "/Game/YourProject/Animations/Idle",
    "walk": "/Game/YourProject/Animations/Walk_Fwd",
    "run": "/Game/YourProject/Animations/Run_Fwd"
  }
}
```

### 3. Run with Animation

```python
py "D:/projects/FustalMOT_UEDataset/Content/FutsalMOT/code/ue/import_grf_episode.py" --episode "D:/projects/FustalMOT_UEDataset/Content/FutsalMOT/code/outputs/episode_0001" --mapping "D:/projects/FustalMOT_UEDataset/Content/FutsalMOT/code/ue/actor_mapping.example.json" --animation-config "D:/projects/FustalMOT_UEDataset/Content/FutsalMOT/code/ue/animation_config.local.json" --replace-existing
```

## Locomotion Configuration

The `locomotion` section controls speed thresholds:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `idle_max_speed_mps` | 0.20 | Speed below this → Idle |
| `run_min_speed_mps` | 2.50 | Speed above this → Run |
| `smoothing_window` | 5 | Moving average window (odd number) |
| `minimum_segment_frames` | 6 | Minimum animation section length (30 FPS frames) |
| `idle_play_rate` | 1.0 | Idle animation play rate |
| `walk_reference_speed_mps` | 1.4 | Speed at which walk animation looks natural |
| `run_reference_speed_mps` | 4.0 | Speed at which run animation looks natural |
| `minimum_play_rate` | 0.75 | Clamp lower bound |
| `maximum_play_rate` | 1.50 | Clamp upper bound |

Play rate for walk/run = `mean_speed / reference_speed`, clamped to [minimum, maximum].

## Ball Rolling Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `radius_m` | 0.11 | Ball radius in meters |
| `minimum_move_distance_m` | 0.0001 | Ignore tiny displacements |
| `roll_sign` | 1.0 | Set to `-1.0` if ball rolls backward visually |

If the ball rolls backward visually, change `roll_sign` to `-1.0`. Do NOT modify GRF data.

## Actor Mapping

Edit `actor_mapping.example.json` to match your UE level's actor labels.
Default mapping assumes:
- `Player_L0` through `Player_L4` for left team
- `Player_R0` through `Player_R4` for right team
- `Ball_01` for the football

## Script Modes

| Mode | Description |
|------|-------------|
| `preview` | Set actor transforms directly in the level (no asset creation) |
| `sequence` | Create/overwrite a Level Sequence asset |
| `both` (default) | Both preview + sequence |

## Notes

- Player Z = 90cm (character pivot at waist level; ground is Z=0)
- Ball Z = GRF data × 100 + 2cm offset
- Yaw is computed from position deltas
- **Transform Tracks** control all world positions; animation is purely in-place
- If characters drift or slide, your animations likely contain Root Motion — switch to In-Place animations
- No CharacterMovement, NavMesh, collision correction, or IK
- Manual verification required in UE viewport
