"""FutsalMOT path definitions with explicit project-root resolution.

Project root resolution priority:
  1. Explicit override via set_project_root()
  2. Environment variable FUTSALMOT_PROJECT_ROOT
  3. Search upward from cwd for *.uproject file
  4. Directory-structure inference from this file's location
"""

from __future__ import annotations

import os
from pathlib import Path


# ── Mutable project root ──────────────────────────────────────
_project_root_override: Path | None = None


def resolve_project_root() -> Path:
    """Resolve the UE project root directory."""
    if _project_root_override is not None:
        return _project_root_override

    env_value = os.environ.get("FUTSALMOT_PROJECT_ROOT")
    if env_value:
        return Path(env_value).resolve()

    for parent in Path.cwd().resolve().parents:
        uproject_files = list(parent.glob("*.uproject"))
        if uproject_files:
            return parent

    # Fallback: .../code → .../Content/FutsalMOT/ → .../Content/ → project root
    code_dir = Path(__file__).resolve().parents[2]
    return code_dir.parent.parent.parent


def set_project_root(path: str | Path) -> None:
    """Override the project root (call from CLI before any path access)."""
    global _project_root_override
    _project_root_override = Path(path).resolve()


# ── Derived paths ─────────────────────────────────────────────
_CODE_DIR = Path(__file__).resolve().parents[2]
_PROJECT_ROOT = resolve_project_root()

CODE_DIR = _CODE_DIR
CONTENT_FUTSALMOT_DIR = CODE_DIR.parent
CONTENT_DIR = CONTENT_FUTSALMOT_DIR.parent
PROJECT_ROOT = _PROJECT_ROOT
CONFIG_DIR = CODE_DIR / "configs"
GENERATED_EVENT_DIR = CONFIG_DIR / "events" / "generated"
RUNS_DIR = CONFIG_DIR / "runs"
PIPELINE_CONFIG_PATH = CONFIG_DIR / "pipeline_config.json"
CURRENT_RUN_POINTER = CONFIG_DIR / "pipeline_current.json"
AGENT_OUTPUT_DIR = CODE_DIR / "_agent_test_outputs"


def resolve_code_relative(path: str | Path) -> Path:
    """Resolve a path relative to the code directory."""
    path = Path(path).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (CODE_DIR / path).resolve()


def saved_futsalmot_dir(project_root: Path | None = None) -> Path:
    """Return Saved/FutsalMOT under the project root."""
    root = project_root.resolve() if project_root is not None else PROJECT_ROOT.resolve()
    return root / "Saved" / "FutsalMOT"
