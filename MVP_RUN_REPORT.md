# MVP Run Report — GRF → JSONL → UE

## 1. Windows 版本

| Item | Value |
|------|-------|
| Windows Product | Windows 10 Pro |
| Windows Version | 2009 |
| OS Build | 26200 |
| Architecture | 64-bit |

## 2. PowerShell / uv / Python

| Tool | Version |
|------|---------|
| PowerShell | 5.1 (built-in) |
| uv | 0.11.8 |
| Python | 3.9.25 |
| Interpreter | `D:\projects\FustalMOT_UEDataset\Content\FutsalMOT\code\.venv\Scripts\python.exe` |

## 3. Build Tools

| Tool | Version |
|------|---------|
| Git | 2.53.0.windows.2 |
| CMake | 4.4.0 (not needed — PyPI install succeeded) |
| Visual Studio | 2022 BuildTools (18) with MSVC 14.50, Windows SDK 10 |

## 4. GRF Installation

- **Method**: PyPI (`uv add gfootball`)
- **Version**: 2.10.2
- **Engine commit**: `3d9e754720a95621bba6475c4d3b0d56fe919014`
- **Source build**: Not needed (PyPI binary wheel installed successfully)

## 5. GRF_MARL

- **Commit**: `6cf67a509dc204f5f413adaa57619652580c80f1`
- **Status**: Cloned but not loaded (no pretrained policies used in MVP)

## 6. Scenario & Control

| Item | Value |
|------|-------|
| Scenario | `5_vs_5` |
| Control mode | `builtin_AI_vs_builtin_AI` (1 left player nominally controlled, `action_builtin_ai` sent every step) |
| Steps | 300 |
| Seed | 42 |

## 7. Reset & 300-step Result

- **Reset**: PASS
- **300 steps**: PASS (no episode termination during 300 steps)
- **Final score**: (0, 0)

## 8. Output Files

| File | Size | SHA256 |
|------|------|--------|
| `outputs/episode_0001/meta.json` | 2834 bytes | `890A5D8CB6A843914459F164D687CB5C12F866EF5171E3DEBC7695923B897017` |
| `outputs/episode_0001/frames.jsonl` | 278088 bytes | `FAC8D3AFF207F99F3599CA1B2C4DB40BEEF4700AA821CE1BB4E01864EC9B8769` |

- **Frames**: 300 lines (0..299)
- **Players per frame**: 10 (exactly L0-L4, R0-R4)
- **Ball per frame**: present with position + source_grf_position

## 9. Validator Result

```
VALIDATOR: Episode PASSED all checks
  Steps: 300
  Source step seconds: 0.1
  Field: 40.0m x 20.0m
```

## 10. pytest Result

```
31 passed in 0.20s
```

## 11. UE Execution

- **Status**: `READY_FOR_MANUAL_UE_VALIDATION`
- **UE not actually opened or run** — human must follow `UE_MVP_STEPS.md`

## 12. Remaining Manual Steps

1. Place 10 player actors + 1 ball in UE level
2. Run `ue/import_grf_episode.py` in UE Python Console
3. Verify player and ball positions in viewport
