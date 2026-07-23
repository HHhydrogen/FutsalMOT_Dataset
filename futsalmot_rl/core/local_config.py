"""Local path configuration — reads configs/local_paths.json."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def _find_config() -> Path | None:
    """Search for local_paths.json relative to this file or cwd."""
    candidates = [
        Path(__file__).resolve().parents[2] / "configs" / "local_paths.json",
        Path.cwd() / "configs" / "local_paths.json",
    ]
    for c in candidates:
        if c.is_file():
            return c
    return None


def load_local_paths() -> dict[str, Any]:
    """Load local machine paths from config file.

    Priority: local_paths.json → env var → empty dict (caller handles defaults).
    """
    config_path = _find_config()

    if config_path is not None:
        with open(config_path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data

    # Fallback: env var
    env_root = os.environ.get("FUTSALMOT_PROJECT_ROOT")
    if env_root:
        return {"ue_project_root": env_root}

    return {}


def get_repo_root() -> Path:
    """Return the repo root (where this file lives: code/futsalmot_rl/core/)."""
    return Path(__file__).resolve().parents[2]


def get_ue_project_root(cfg: dict[str, Any] | None = None) -> Path:
    """Return resolved UE project root."""
    if cfg is None:
        cfg = load_local_paths()

    explicit = cfg.get("ue_project_root") or cfg.get("project_root")
    if explicit:
        p = Path(explicit).resolve()
        if p.is_dir():
            return p
        raise FileNotFoundError(f"UE project root not found: {p}")

    raise FileNotFoundError(
        "No UE project root configured. Set ue_project_root in configs/local_paths.json"
    )
