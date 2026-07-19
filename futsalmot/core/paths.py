from __future__ import annotations

from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[2]
CONTENT_FUTSALMOT_DIR = CODE_DIR.parent
CONTENT_DIR = CONTENT_FUTSALMOT_DIR.parent
PROJECT_ROOT = CONTENT_DIR.parent
CONFIG_DIR = CODE_DIR / "configs"
GENERATED_EVENT_DIR = CONFIG_DIR / "events" / "generated"
CURRENT_RUN_POINTER = CONFIG_DIR / "pipeline_current.json"
AGENT_OUTPUT_DIR = CODE_DIR / "_agent_test_outputs"


def resolve_code_relative(path: str | Path) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (CODE_DIR / path).resolve()


def saved_futsalmot_dir(project_root: Path | None = None) -> Path:
    root = project_root.resolve() if project_root is not None else PROJECT_ROOT.resolve()
    return root / "Saved" / "FutsalMOT"
