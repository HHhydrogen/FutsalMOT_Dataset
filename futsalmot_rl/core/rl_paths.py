"""FutsalMOT-RL independent path definitions.

All RL outputs go to Saved/FutsalMOT_RL/ — never touch Saved/FutsalMOT/.
"""

from __future__ import annotations

from pathlib import Path

# ── Project anchor ──────────────────────────────────────────────
# We locate the project root from this file's own location:
#   .../code/futsalmot_rl/core/rl_paths.py
#   → parents[2] = code/
#   → parents[5] = project root (D:/projects/FustalMOT_UEDataset)
_CODE_DIR = Path(__file__).resolve().parents[2]
# Walk up from code/ to find the UE project root
# Path: code/ → Content/FutsalMOT/ → Content/ → project root
_CONTENT_FUTSALMOT_DIR = _CODE_DIR.parent
_CONTENT_DIR = _CONTENT_FUTSALMOT_DIR.parent
_PROJECT_ROOT = _CONTENT_DIR.parent

# ── RL output root (completely independent from main pipeline) ──
RL_ROOT = _PROJECT_ROOT / "Saved" / "FutsalMOT_RL"

# ── Sub-directories ────────────────────────────────────────────
DEMOS_DIR = RL_ROOT / "demos"
MODELS_DIR = RL_ROOT / "models"
TRAIN_LOGS_DIR = RL_ROOT / "train_logs"
EVAL_DIR = RL_ROOT / "eval"
VIDEOS_DIR = RL_ROOT / "videos"
ROLLOUTS_DIR = RL_ROOT / "rollouts"
EXPORTED_A33_DIR = RL_ROOT / "exported_a33"
REPORTS_DIR = RL_ROOT / "reports"
BENCHMARK_DIR = RL_ROOT / "benchmark"
ABLATIONS_DIR = RL_ROOT / "ablations"
PAPER_TABLES_DIR = RL_ROOT / "paper_tables"

# ── Source data paths (read-only references to main pipeline) ──
CODE_DIR = _CODE_DIR
CONFIG_DIR = _CODE_DIR / "configs"
RUNS_DIR = CONFIG_DIR / "runs"
PROJECT_ROOT = _PROJECT_ROOT

# ── Shared output constants ────────────────────────────────────
COURT_X_MIN, COURT_X_MAX = -1950.0, 1950.0
COURT_Y_MIN, COURT_Y_MAX = -950.0, 950.0
FPS = 30
EPISODE_FRAMES = 300
PLAYER_MAX_SPEED_CM_S = 750.0
BALL_MAX_SPEED_CM_S = 3000.0


def ensure_dirs() -> None:
    """Create all RL output directories — safe to call multiple times."""
    for directory in (
        DEMOS_DIR,
        MODELS_DIR,
        TRAIN_LOGS_DIR / "bc",
        TRAIN_LOGS_DIR / "ppo",
        EVAL_DIR,
        VIDEOS_DIR / "demos",
        VIDEOS_DIR / "bc",
        VIDEOS_DIR / "rl_train",
        VIDEOS_DIR / "rl_eval",
        VIDEOS_DIR / "final",
VIDEOS_DIR / "comparison",
        ROLLOUTS_DIR,
        EXPORTED_A33_DIR,
        REPORTS_DIR,
        BENCHMARK_DIR,
        PAPER_TABLES_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
