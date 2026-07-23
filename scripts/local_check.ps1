param(
    [switch]$Verbose = $false
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

function Check-Step {
    param($Name, $Command)
    Write-Host "`n=== $Name ===" -ForegroundColor Cyan
    Write-Host "  $Command" -ForegroundColor Gray
    try {
        if ($Verbose) {
            Invoke-Expression $Command
        } else {
            $result = Invoke-Expression $Command 2>&1
            if ($LASTEXITCODE -and $LASTEXITCODE -ne 0) {
                Write-Host "  FAILED (exit $LASTEXITCODE)" -ForegroundColor Red
                Write-Host $result
                exit $LASTEXITCODE
            }
        }
        Write-Host "  PASS" -ForegroundColor Green
    } catch {
        Write-Host "  FAILED: $_" -ForegroundColor Red
        exit 1
    }
}

Write-Host "============================================" -ForegroundColor Yellow
Write-Host "  FutsalMOT-RL Local Check" -ForegroundColor Yellow
Write-Host "  Repo: $RepoRoot" -ForegroundColor Yellow
Write-Host "  Python: $((python --version 2>&1))" -ForegroundColor Yellow
Write-Host "============================================" -ForegroundColor Yellow

Check-Step "Python version" "python --version"
Check-Step "Dependencies" "python -c 'import torch; import gymnasium; import numpy; print(f\"torch {torch.__version__}\")'"
Check-Step "Local config" "python -c 'from futsalmot_rl.core.local_config import load_local_paths; c = load_local_paths(); print(c.get(\"ue_project_root\",\"(not set)\"))'"
Check-Step "Uproject check" "python -c 'from pathlib import Path; import json; cfg = json.load(open(\"configs/local_paths.json\")); p = Path(cfg[\"uproject_file\"]); assert p.is_file(), f\"Missing: {p}\"; print(f\"OK: {p.name}\")'"
Check-Step "Compileall" "python -m compileall futsalmot futsalmot_rl -q"
Check-Step "Unit tests" "python -m pytest tests/unit -q --tb=short"

Write-Host "`n============================================" -ForegroundColor Yellow
Write-Host "  ALL CHECKS PASSED" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Yellow
