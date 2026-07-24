# UE MINI-MVP — Manual Verification Steps

## Prerequisites

- [ ] Unreal Engine project with same coordinate system as export (X forward, Y right, Z up)
- [ ] 10 player actors placed: `Player_L0` to `Player_L4` (left), `Player_R0` to `Player_R4` (right)
- [ ] 1 ball actor placed: `Ball_01`
- [ ] (Optional) 3 In-Place animation sequences: Idle, Walk_Fwd, Run_Fwd

## Steps

### 1. Export a fresh episode
```powershell
uv run grf-ue export --config configs/mvp_builtin_5v5.json --output outputs/episode_0001
```

### 2. Transform-only import (no animation)
```python
py "D:/projects/FustalMOT_UEDataset/Content/FutsalMOT/code/ue/import_grf_episode.py" --episode "D:/projects/FustalMOT_UEDataset/Content/FutsalMOT/code/outputs/episode_0001" --mapping "D:/projects/FustalMOT_UEDataset/Content/FutsalMOT/code/ue/actor_mapping.example.json" --replace-existing
```

### 3. (Optional) With locomotion animation + ball rolling
Create `ue/animation_config.local.json` from `ue/animation_config.example.json`, fill in your animation asset paths, then:
```python
py "D:/projects/FustalMOT_UEDataset/Content/FutsalMOT/code/ue/import_grf_episode.py" --episode "D:/projects/FustalMOT_UEDataset/Content/FutsalMOT/code/outputs/episode_0001" --mapping "D:/projects/FustalMOT_UEDataset/Content/FutsalMOT/code/ue/actor_mapping.example.json" --animation-config "D:/projects/FustalMOT_UEDataset/Content/FutsalMOT/code/ue/animation_config.local.json" --replace-existing
```

### 4. Verify in viewport
- Open the created Level Sequence in Sequencer
- Press Play to preview
- Players should move across the field with appropriate idle/walk/run animation
- Ball should roll continuously when moving, stay still when stationary
- No character should drift/slide from Root Motion

### 5. Troubleshooting

| Symptom | Fix |
|---------|-----|
| Actors not found | Check labels in `actor_mapping.example.json` against actual level actors |
| Ball in ground | Adjust `BALL_Z_OFFSET_CM` in script |
| Ball rolls backward | Set `roll_sign: -1.0` in animation config |
| Character slides/drifts | Animation contains Root Motion — use In-Place animations |
| Wrong animation state | Tune `idle_max_speed_mps` / `run_min_speed_mps` in config |
| "Animation asset not found" | Check paths in `animation_config.local.json` |

## Status

**READY_FOR_MANUAL_UE_VALIDATION**

The Python code and episode data are ready. A human must step through the UE Editor steps above.
