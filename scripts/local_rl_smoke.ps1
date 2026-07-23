$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$SmokeDir = "$RepoRoot/Saved/FutsalMOT_RL/local_smoke"
$null = New-Item -ItemType Directory -Force -Path $SmokeDir

Write-Host "============================================" -ForegroundColor Yellow
Write-Host "  FutsalMOT-RL Local Smoke Test" -ForegroundColor Yellow
Write-Host "  Output: $SmokeDir" -ForegroundColor Yellow
Write-Host "============================================" -ForegroundColor Yellow

function Step {
    param($Name, $Command)
    Write-Host "`n>>> $Name" -ForegroundColor Cyan
    Write-Host "  $Command" -ForegroundColor Gray
    try {
        Invoke-Expression $Command 2>&1 | ForEach-Object { Write-Host "  $_" }
        if ($LASTEXITCODE -and $LASTEXITCODE -ne 0) {
            throw "Exit code: $LASTEXITCODE"
        }
        Write-Host "  [PASS] $Name" -ForegroundColor Green
    } catch {
        Write-Host "  [FAIL] $Name: $_" -ForegroundColor Red
        exit 1
    }
}

# 1. Export minimal demo
Step "Export demo" "python tools/legacy/rl_01_export_demos.py --max-episodes 2"

# 2. Check demo
Step "Check demo" "python tools/legacy/rl_01b_check_demos.py"

# 3. Create env and step with random actions
Step "Env sanity" "python -c '
from futsalmot_rl.envs.defender_follow_env import FutsalDefenderFollowEnv
from futsalmot_rl.core.rl_paths import RUNS_DIR
import numpy as np
source = str(RUNS_DIR / ""production_run"" / ""episode_random_0001_t1_a33.json"")
env = FutsalDefenderFollowEnv(source_episode_path=source)
obs, _ = env.reset()
print(f""Obs shape: {obs.shape}, dtype: {obs.dtype}"")
for i in range(30):
    a = env.action_space.sample()
    obs, r, term, trunc, _ = env.step(a)
    if term or trunc:
        obs, _ = env.reset()
assert np.all(np.isfinite(obs)), ""NaN in obs""
print(""30 steps OK"")
env.close()
'"

# 4. BC training (tiny, 2 epochs)
Step "BC train" "python scripts/train_bc.py --epochs 2 --no-video"

# 5. BC eval
Step "BC eval" "python -c '
from futsalmot_rl.models.policy_io import load_policy
from futsalmot_rl.envs.defender_follow_env import FutsalDefenderFollowEnv
from futsalmot_rl.core.rl_paths import RUNS_DIR, MODELS_DIR
source = str(RUNS_DIR / ""production_run"" / ""episode_random_0001_t1_a33.json"")
policy, _, _ = load_policy(str(MODELS_DIR / ""defender_follow_bc_v1.pt""))
env = FutsalDefenderFollowEnv(source_episode_path=source)
obs, _ = env.reset()
done = False
while not done:
    a = policy.get_action(obs, deterministic=True)
    obs, _, term, trunc, _ = env.step(a)
    done = term or trunc
env.close()
print(""BC eval: episode complete"")
'"

# 6. PPO smoke (2048 steps)
Step "PPO smoke" "python -c '
from futsalmot_rl.envs.defender_follow_env import FutsalDefenderFollowEnv
from futsalmot_rl.training.train_ppo import PPOTrainer
from futsalmot_rl.core.rl_paths import RUNS_DIR, MODELS_DIR
import torch
source = str(RUNS_DIR / ""production_run"" / ""episode_random_0001_t1_a33.json"")
env = FutsalDefenderFollowEnv(source_episode_path=source)
trainer = PPOTrainer(env, config={""total_timesteps"": 2048, ""n_steps"": 2048, ""n_epochs"": 3, ""batch_size"": 64})
# Load BC init
bc_model = str(MODELS_DIR / ""defender_follow_bc_v1.pt"")
trainer.load_pretrained(bc_model)
summary = trainer.train(total_timesteps=2048)
print(f""PPO smoke done. Best reward: {summary.get(\""best_mean_reward\"", \""N/A\"")}"")
'"

# 7. PPO eval
Step "PPO eval" "python -c '
from futsalmot_rl.models.policy_io import load_policy
from futsalmot_rl.envs.defender_follow_env import FutsalDefenderFollowEnv
from futsalmot_rl.core.rl_paths import RUNS_DIR, MODELS_DIR
source = str(RUNS_DIR / ""production_run"" / ""episode_random_0001_t1_a33.json"")
policy, _, _ = load_policy(str(MODELS_DIR / ""defender_follow_ppo_v1_best.pt""))
env = FutsalDefenderFollowEnv(source_episode_path=source)
obs, _ = env.reset()
done = False
while not done:
    a = policy.get_action(obs, deterministic=True)
    obs, _, term, trunc, _ = env.step(a)
    done = term or trunc
env.close()
print(""PPO eval: episode complete"")
'"

# 8. Export A3.3
Step "Export A3.3" "python -c '
from futsalmot_rl.models.policy_io import load_policy
from futsalmot_rl.rollout.export_to_a33 import export_rl_a33
from futsalmot_rl.core.rl_paths import RUNS_DIR, MODELS_DIR
source = str(RUNS_DIR / ""production_run"" / ""episode_random_0001_t1_a33.json"")
policy, _, _ = load_policy(str(MODELS_DIR / ""defender_follow_ppo_v1_best.pt""))
report = export_rl_a33(source, lambda obs, det: policy.get_action(obs, det))
print(f""A3.3 exported: {report.get(\""output_path\"", \""?\"")}"")
'"

Write-Host "`n============================================" -ForegroundColor Yellow
Write-Host "  ALL SMOKE TESTS PASSED" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Yellow
