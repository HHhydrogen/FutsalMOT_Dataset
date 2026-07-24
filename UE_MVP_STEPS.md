# UE MINI-MVP — Manual Verification Steps

## Prerequisites

- [ ] Unreal Engine project with same coordinate system as export (X forward, Y right, Z up)
- [ ] 10 player actors placed: `Player_L0` to `Player_L4` (left), `Player_R0` to `Player_R4` (right)
- [ ] 1 ball actor placed: `Ball_01`

## Steps

1. **Export a fresh episode**
   ```powershell
   uv run grf-ue export --config configs/mvp_builtin_5v5.json --output outputs/episode_0001
   ```

2. **Open Unreal Editor**
   - Load your level with 10 players + 1 ball
   - Open Python Console: Window > Developer Tools > Python Console

3. **Run import script**
   ```python
   py "D:/projects/FustalMOT_UEDataset/Content/FutsalMOT/code/ue/import_grf_episode.py" --episode "D:/projects/FustalMOT_UEDataset/Content/FutsalMOT/code/outputs/episode_0001" --mapping "D:/projects/FustalMOT_UEDataset/Content/FutsalMOT/code/ue/actor_mapping.example.json"
   ```

4. **Verify in viewport**
   - Camera follows ball or overview
   - Players should be spread across the field (left team near x=-20, right team near x=+20)
   - Ball approximately at center after kickoff (kickoff -> movement)
   - Manually scrub or run play to see frame-by-frame updates

5. **Troubleshooting**
   - Actors not found: check labels in `actor_mapping.example.json` against actual level actors
   - Ball in ground: adjust Z offset in `import_grf_episode.py` line `+ 50.0`
   - Wrong orientation: the script uses Unreal `Rotator(pitch, yaw, roll)`; yaw is computed from movement

## Status

**READY_FOR_MANUAL_UE_VALIDATION**

The Python code and episode data are ready. A human must step through the UE Editor steps above.
