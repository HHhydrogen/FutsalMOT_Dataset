param([string]$ConfigFile = "configs/local_paths.json")

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

# Find python
$PythonExe = "python"
if (Test-Path $ConfigFile) {
    $cfg = Get-Content $ConfigFile | ConvertFrom-Json
    if ($cfg.python_exe) { $PythonExe = $cfg.python_exe }
}

function Invoke-Checked {
    param([string]$Label, [scriptblock]$Cmd)
    Write-Host "`n=== $Label ===" -ForegroundColor Cyan
    try {
        & $Cmd
        if ($LASTEXITCODE -and $LASTEXITCODE -ne 0) {
            throw "Exit code: $LASTEXITCODE"
        }
        Write-Host "  [PASS]" -ForegroundColor Green
    } catch {
        Write-Host "  [FAIL] $_" -ForegroundColor Red
        exit 1
    }
}

Write-Host "============================================" -ForegroundColor Yellow
Write-Host "  FutsalMOT-RL Local Check" -ForegroundColor Yellow
Write-Host "  Repo: $RepoRoot"
Write-Host "  Python: $(& $PythonExe --version 2>&1)"
Write-Host "============================================" -ForegroundColor Yellow

Invoke-Checked "Python version" { & $PythonExe --version }
Invoke-Checked "Dependencies" { & $PythonExe -c "import torch; import gymnasium; import numpy; print(f'torch {torch.__version__}')" }
Invoke-Checked "Local config" { & $PythonExe -c "from futsalmot_rl.core.local_config import load_local_paths; c = load_local_paths(); print(c.get('ue_project_root','(not set)'))" }
Invoke-Checked "Uproject" { & $PythonExe -c "
from pathlib import Path; import json
c = json.load(open('$ConfigFile'))
p = Path(c['uproject_file'])
assert p.is_file(), f'Missing: {p}'
print(f'OK: {p}')
" }
Invoke-Checked "Compileall" { & $PythonExe -m compileall futsalmot futsalmot_rl -q }
Invoke-Checked "Unit tests" { & $PythonExe -m pytest tests/unit -q --tb=short }

Write-Host "`n============================================" -ForegroundColor Yellow
Write-Host "  ALL CHECKS PASSED" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Yellow
