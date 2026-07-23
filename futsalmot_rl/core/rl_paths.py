"""FutsalMOT-RL independent path definitions.

All RL outputs go to Saved/FutsalMOT_RL/ — never touch Saved/FutsalMOT/.

Project root resolution priority:
  1. Explicit override via set_project_root() (called from CLI --project-root)
  2. Environment variable FUTSALMOT_PROJECT_ROOT
  3. Search upward from cwd for *.uproject file
  4. Directory-structure inference from this file's location
"""

from __future__ import annotations

import os
from pathlib import Path

# ── Mutable project root (set by CLI or env var) ──────────────
_project_root_override: Path | None = None


def resolve_project_root() -> Path:
    """Resolve the UE project root directory.

    Priority: explicit override → env var → .uproject search → inference.
    """
    if _project_root_override is not None:
        return _project_root_override

    env_value = os.environ.get("FUTSALMOT_PROJECT_ROOT")
    if env_value:
        return Path(env_value).resolve()

    # Search upward from cwd for a .uproject file
    for parent in Path.cwd().resolve().parents:
        uproject_files = list(parent.glob("*.uproject"))
        if uproject_files:
            return parent

    # Fall back to directory inference from this file
    code_dir = Path(__file__).resolve().parents[2]
    return code_dir.parent.parent.parent


def set_project_root(path: str | Path) -> None:
    """Override the project root (call from CLI before any path access)."""
    global _project_root_override
    _project_root_override = Path(path).resolve()


# ── Resolved paths ────────────────────────────────────────────
_CODE_DIR = Path(__file__).resolve().parents[2]
_PROJECT_ROOT = resolve_project_root()

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
