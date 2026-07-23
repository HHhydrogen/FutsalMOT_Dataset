param(
    [string]$SmokeDir = "",
    [string]$ConfigFile = "configs/local_paths.json"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

# Determine output directory
if (-not $SmokeDir) {
    # Try to read from config
    if (Test-Path $ConfigFile) {
        $cfg = Get-Content $ConfigFile | ConvertFrom-Json
        $ueRoot = $cfg.ue_project_root
        if ($ueRoot) {
            $SmokeDir = "$ueRoot/Saved/FutsalMOT_RL/local_smoke"
        }
    }
}
if (-not $SmokeDir) {
    $SmokeDir = "$RepoRoot/Saved/FutsalMOT_RL/local_smoke"
}

# Find python
$PythonExe = "python"
if (Test-Path $ConfigFile) {
    $cfg = Get-Content $ConfigFile | ConvertFrom-Json
    if ($cfg.python_exe) { $PythonExe = $cfg.python_exe }
}

# Record formal model hashes BEFORE any smoke operations
$FormalModelDirs = @(
    "$RepoRoot/Saved/FutsalMOT_RL/models"
)
$beforeHashes = @{}
$beforeFiles = @()
foreach ($dir in $FormalModelDirs) {
    if (Test-Path $dir) {
        Get-ChildItem $dir -Recurse -File -ErrorAction SilentlyContinue |
            Where-Object { $_.Extension -in ".pt", ".pth", ".ckpt" } |
            ForEach-Object {
                $hash = (Get-FileHash $_.FullName -Algorithm SHA256).Hash
                $beforeHashes[$_.FullName] = $hash
                $beforeFiles += $_.FullName
            }
    }
}
Write-Host "Formal models tracked: $($beforeFiles.Count) files"

$SourceFile = "$RepoRoot/configs/runs/production_run/episode_random_0001_t1_a33.json"
if (-not (Test-Path $SourceFile)) {
    Write-Host "[FAIL] Source file not found: $SourceFile" -ForegroundColor Red
    exit 1
}

# Clean local_smoke only
if (Test-Path $SmokeDir) {
    Remove-Item -Recurse -Force "$SmokeDir/*" -ErrorAction SilentlyContinue
} else {
    New-Item -ItemType Directory -Force -Path $SmokeDir | Out-Null
}

$SmokeDemos = "$SmokeDir/demos"
$SmokeModels = "$SmokeDir/models"
$SmokeLogs = "$SmokeDir/logs"
$SmokeExports = "$SmokeDir/exports"
$SmokeEval = "$SmokeDir/evaluation"

foreach ($d in @($SmokeDemos, $SmokeModels, $SmokeLogs, $SmokeExports, $SmokeEval)) {
    New-Item -ItemType Directory -Force -Path $d | Out-Null
}

function Invoke-Checked {
    param([string]$Label, [scriptblock]$Cmd)
    Write-Host "`n>>> $Label" -ForegroundColor Cyan
    try {
        & $Cmd
        if ($LASTEXITCODE -and $LASTEXITCODE -ne 0) {
            throw "Exit code: $LASTEXITCODE"
        }
        Write-Host "  [PASS] $Label" -ForegroundColor Green
    } catch {
        Write-Host "  [FAIL] $Label : $_" -ForegroundColor Red
        exit 1
    }
}

Write-Host "`n============================================" -ForegroundColor Yellow
Write-Host "  FutsalMOT-RL Smoke Test"
Write-Host "  Output: $SmokeDir"
Write-Host "  Python: $PythonExe"
Write-Host "============================================" -ForegroundColor Yellow

# 1. Export demo
$SmokeDemoIndex = "$SmokeDemos/demo_index.json"
Invoke-Checked "Export demo" {
    & $PythonExe tools/legacy/rl_01_export_demos.py --max-episodes 2 --output-dir $SmokeDemos
}

# 2. Check demo
Invoke-Checked "Check demo" {
    & $PythonExe -c "
import numpy as np
import json
idx = json.load(open('$SmokeDemoIndex'))
for d in idx.get('demos', []):
    data = np.load(d['path'])
    assert np.all(np.isfinite(data['obs'])), 'NaN in obs'
assert len(idx['demos']) > 0, 'No demos'
print(f'Checked {len(idx[\"demos\"])} demos, OK')
"
}

# 3. Env sanity
Invoke-Checked "Env sanity" {
    & $PythonExe -c "
from futsalmot_rl.envs.defender_follow_env import FutsalDefenderFollowEnv
import numpy as np
env = FutsalDefenderFollowEnv(source_episode_path='$SourceFile')
obs, _ = env.reset()
print(f'Obs: {obs.shape} {obs.dtype}')
for i in range(30):
    a = env.action_space.sample()
    obs, r, term, trunc, _ = env.step(a)
    if term or trunc: obs, _ = env.reset()
assert np.all(np.isfinite(obs)), 'NaN'
env.close()
print('30 steps OK')
"
}

# 4. BC train
$SmokeBcModel = "$SmokeModels/bc_smoke.pt"
Invoke-Checked "BC train (2 epochs)" {
    & $PythonExe -c "
import sys
sys.path.insert(0, '.')
from futsalmot_rl.training.train_bc import train_bc
summary = train_bc(
    demo_index_path='$SmokeDemoIndex',
    model_out='$SmokeBcModel',
    config={'epochs': 2, 'batch_size': 512, 'train_split': 0.5, 'val_split': 0.25},
)
assert summary.get('best_val_loss', 1) < 1, 'BC loss too high'
print(f'BC done, loss={summary[\"best_val_loss\"]:.4f}')
"
}

# 5. BC eval
Invoke-Checked "BC eval" {
    & $PythonExe -m futsalmot_rl.cli evaluate bc `
        --source $SourceFile `
        --model $SmokeBcModel `
        --output-dir $SmokeEval/bc `
        --n-episodes 3 `
        --device cpu `
        --seed 42
    if (-not (Test-Path "$SmokeEval/bc/evaluation_summary.json")) {
        throw "evaluation_summary.json not found"
    }
}

# 6. PPO smoke (1024 steps)
$SmokePpoModel = "$SmokeModels/ppo_smoke_best.pt"
$SmokePpoLatest = "$SmokeModels/ppo_smoke_latest.pt"
Invoke-Checked "PPO smoke (1024 steps)" {
    & $PythonExe -c "
import sys
sys.path.insert(0, '.')
from futsalmot_rl.envs.defender_follow_env import FutsalDefenderFollowEnv
from futsalmot_rl.training.train_ppo import PPOTrainer
env = FutsalDefenderFollowEnv(source_episode_path='$SourceFile')
trainer = PPOTrainer(env, config={'total_timesteps': 1024, 'n_steps': 1024, 'n_epochs': 3, 'batch_size': 64})
trainer.load_pretrained('$SmokeBcModel')
summary = trainer.train(total_timesteps=1024, log_dir='$SmokeLogs/ppo', model_dir='$SmokeModels', run_name='ppo_smoke')
print(f'PPO done, reward={summary.get(\"best_mean_reward\", \"N/A\")}')
env.close()
"
    if (-not (Test-Path $SmokePpoLatest)) {
        throw "PPO latest checkpoint not found at $SmokePpoLatest"
    }
}

# 7. PPO eval
Invoke-Checked "PPO eval" {
    $model = if (Test-Path $SmokePpoModel) { $SmokePpoModel } else { "$SmokeModels/ppo_smoke_latest.pt" }
    & $PythonExe -m futsalmot_rl.cli evaluate ppo `
        --source $SourceFile `
        --model $model `
        --output-dir $SmokeEval/ppo `
        --n-episodes 3 `
        --device cpu `
        --seed 42
    if (-not (Test-Path "$SmokeEval/ppo/evaluation_summary.json")) {
        throw "evaluation_summary.json not found"
    }
}

# 8. A3.3 export
Invoke-Checked "Export A3.3" {
    $model = $SmokePpoLatest
    if (-not (Test-Path $model)) { throw "PPO model not found: $model" }
    & $PythonExe -c "
from futsalmot_rl.models.policy_io import load_policy
from futsalmot_rl.rollout.export_to_a33 import export_rl_a33
policy, _, _ = load_policy('$model')
report = export_rl_a33('$SourceFile', lambda o, **kw: policy.get_action(o, deterministic=True), output_dir='$SmokeExports')
assert report.get('n_frames', 0) > 0, 'No frames'
print(f'A3.3 exported: {report[\"output_path\"]} ({report[\"n_frames\"]} frames)')
"
}

# 9. Verify formal models not overwritten
Invoke-Checked "Formal model integrity" {
    $violations = @()
    $newFiles = @()
    $afterFiles = @()

    foreach ($dir in $FormalModelDirs) {
        if (Test-Path $dir) {
            Get-ChildItem $dir -Recurse -File -ErrorAction SilentlyContinue |
                Where-Object { $_.Extension -in ".pt", ".pth", ".ckpt" } |
                ForEach-Object {
                    $afterFiles += $_.FullName
                    if ($beforeFiles -contains $_.FullName) {
                        $hashAfter = (Get-FileHash $_.FullName -Algorithm SHA256).Hash
                        if ($beforeHashes[$_.FullName] -ne $hashAfter) {
                            $violations += "Modified: $($_.FullName)"
                        }
                    } else {
                        $newFiles += $_.FullName
                    }
                }
        }
    }

    $deletedFiles = @()
    foreach ($beforeFile in $beforeFiles) {
        if (-not (Test-Path $beforeFile)) {
            $deletedFiles += $beforeFile
        }
    }

    if ($deletedFiles.Count -gt 0) {
        foreach ($d in $deletedFiles) { Write-Host "  DELETED: $d" -ForegroundColor Red }
        throw "Formal model files were deleted"
    }
    if ($violations.Count -gt 0) {
        foreach ($v in $violations) { Write-Host "  VIOLATION: $v" -ForegroundColor Red }
        throw "Formal model files were modified"
    }
    if ($newFiles.Count -gt 0) {
        Write-Host "  New files in formal dir: $($newFiles.Count)" -ForegroundColor Yellow
        foreach ($n in $newFiles) { Write-Host "    NEW: $n" -ForegroundColor Yellow }
        throw "Smoke created files in formal model directory"
    }
    if ($beforeFiles.Count -eq 0) {
        Write-Host "  (No formal models existed before smoke — none created now)" -ForegroundColor Yellow
    } else {
        Write-Host "  All $($beforeFiles.Count) formal model(s) unchanged" -ForegroundColor Green
    }
    Write-Host "  Formal models unchanged: PASS" -ForegroundColor Green
    Write-Host "  Smoke output isolation: PASS" -ForegroundColor Green
}

Write-Host "`n============================================" -ForegroundColor Yellow
Write-Host "  ALL SMOKE TESTS PASSED" -ForegroundColor Green
Write-Host "  Output: $SmokeDir" -ForegroundColor Yellow
Write-Host "============================================" -ForegroundColor Yellow
