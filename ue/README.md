# UE Import Script — GRF Replay Preview

## Files

| File | Description |
|------|-------------|
| `actor_mapping.example.json` | Maps entity IDs (L0–L4, R0–R4, BALL) to UE actor names |
| `import_grf_episode.py` | UE Python script that reads episode data and positions actors |

## Requirements

- Unreal Engine 5.x project with 10 player actors + 1 ball actor placed in the level
- No `gfootball`, no `.venv`, no GRF_MARL — pure UE Python + stdlib

## Usage

1. Place actors in the UE level matching names in your mapping file.
2. Export a GRF episode using:
   ```powershell
   uv run grf-ue export --config configs/mvp_builtin_5v5.json --output outputs/episode_0001
   ```
3. In Unreal Editor, open Python Console (Window > Developer Tools > Python Console).
4. Run:
   ```python
   py "C:/path/to/your/project/ue/import_grf_episode.py" --episode "C:/path/to/outputs/episode_0001" --mapping "C:/path/to/ue/actor_mapping.example.json"
   ```
5. Script applies transforms to actors frame-by-frame.

## Actor Mapping

Edit `actor_mapping.example.json` to match your UE level's actor labels.
Default mapping assumes:
- `Player_L0` through `Player_L4` for left team
- `Player_R0` through `Player_R4` for right team
- `Ball_01` for the football

## Notes

- Yaw is computed from position deltas (not from GRF data)
- Player Z is always 0 (ground level)
- No CharacterMovement, NavMesh, collision correction, or animation
- Manual verification required in UE viewport
