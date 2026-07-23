"""ProjectPaths — immutable path context for FutsalMOT-RL.

Resolution priority:
  1. Explicitly passed --project-root
  2. Environment variable FUTSALMOT_PROJECT_ROOT
  3. Search cwd (and parents) for *.uproject
  4. Infer from package location (last resort)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from futsalmot_rl.core.exceptions import ConfigurationError


@dataclass(frozen=True)
class ProjectPaths:
    """Immutable path context. Created once at CLI entry, passed everywhere."""

    repo_root: Path
    """Python project root (contains pyproject.toml, configs/, futsalmot_rl/)."""

    ue_project_root: Path
    """Unreal Engine project root (contains *.uproject and Saved/)."""

    # ── Derived RL output paths ─────────────────────────────────
    @property
    def saved_rl_dir(self) -> Path:
        return self.ue_project_root / "Saved" / "FutsalMOT_RL"

    @property
    def demos_dir(self) -> Path:
        return self.saved_rl_dir / "demos"

    @property
    def models_dir(self) -> Path:
        return self.saved_rl_dir / "models"

    @property
    def train_logs_dir(self) -> Path:
        return self.saved_rl_dir / "train_logs"

    @property
    def videos_dir(self) -> Path:
        return self.saved_rl_dir / "videos"

    @property
    def exported_a33_dir(self) -> Path:
        return self.saved_rl_dir / "exported_a33"

    @property
    def reports_dir(self) -> Path:
        return self.saved_rl_dir / "reports"

    @property
    def benchmark_dir(self) -> Path:
        return self.saved_rl_dir / "benchmark"

    @property
    def ablations_dir(self) -> Path:
        return self.saved_rl_dir / "ablations"

    # ── Source data (read-only references to main pipeline) ─────
    @property
    def configs_dir(self) -> Path:
        return self.repo_root / "configs"

    @property
    def runs_dir(self) -> Path:
        return self.configs_dir / "runs"

    # ── Convenience ─────────────────────────────────────────────
    def ensure_all(self) -> None:
        """Create all RL output directories."""
        for d in (
            self.saved_rl_dir,
            self.demos_dir,
            self.models_dir,
            self.train_logs_dir / "bc",
            self.train_logs_dir / "ppo",
            self.videos_dir / "demos",
            self.videos_dir / "bc",
            self.videos_dir / "rl_train",
            self.videos_dir / "rl_eval",
            self.videos_dir / "final",
            self.videos_dir / "comparison",
            self.exported_a33_dir,
            self.reports_dir,
            self.benchmark_dir,
            self.ablations_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)


# ── Resolution logic ──────────────────────────────────────────

_REPO_ROOT_CACHE: Path | None = None
_REPO_ROOT_LOCKED: bool = False


def _discover_repo_root() -> Path:
    """Find the repo root (where pyproject.toml lives)."""
    global _REPO_ROOT_CACHE, _REPO_ROOT_LOCKED
    if _REPO_ROOT_CACHE is not None and _REPO_ROOT_LOCKED:
        return _REPO_ROOT_CACHE

    # Start from this file's location: futsalmot_rl/core/paths.py
    candidate = Path(__file__).resolve()
    for parent in (candidate, *candidate.parents):
        if (parent / "pyproject.toml").is_file():
            _REPO_ROOT_CACHE = parent
            _REPO_ROOT_LOCKED = True
            return parent
    raise ConfigurationError(
        f"Cannot find repo root: no pyproject.toml found in any parent of {candidate}"
    )


def _reset_repo_root_cache() -> None:
    """Clear the repo root cache (testing only)."""
    global _REPO_ROOT_CACHE, _REPO_ROOT_LOCKED
    _REPO_ROOT_CACHE = None
    _REPO_ROOT_LOCKED = False


def resolve_project_root(explicit: str | Path | None = None) -> Path:
    """Resolve the UE project root directory.

    Args:
        explicit: CLI --project-root override.

    Returns:
        Absolute path to the UE project root.

    Raises:
        ConfigurationError: if the root cannot be found.
    """
    # 1. Explicit override
    if explicit is not None:
        try:
            p = Path(explicit).resolve(strict=True)
        except (FileNotFoundError, OSError) as exc:
            raise ConfigurationError(f"--project-root path does not exist: {explicit}") from exc
        if not p.is_dir():
            raise ConfigurationError(f"--project-root is not a directory: {p}")
        return p

    # 2. Environment variable
    env_val = os.environ.get("FUTSALMOT_PROJECT_ROOT")
    if env_val:
        try:
            p = Path(env_val).resolve(strict=True)
        except (FileNotFoundError, OSError) as exc:
            raise ConfigurationError(f"FUTSALMOT_PROJECT_ROOT={env_val} does not exist") from exc
        if not p.is_dir():
            raise ConfigurationError(f"FUTSALMOT_PROJECT_ROOT={env_val} is not a directory")
        return p

    # 3. Search upward from cwd for *.uproject
    cwd = Path.cwd().resolve()
    for candidate in (cwd, *cwd.parents):
        uproject = list(candidate.glob("*.uproject"))
        if uproject:
            return candidate

    # 4. Infer from repo root (or cwd-based pyproject discovery)
    try:
        repo = _discover_repo_root()
        inferred = repo.parent.parent.parent
        if inferred.is_dir():
            return inferred
    except ConfigurationError:
        pass

    raise ConfigurationError(
        "Cannot find UE project root. Pass --project-root or set FUTSALMOT_PROJECT_ROOT."
    )


def create_paths(project_root: str | Path | None = None) -> ProjectPaths:
    """Create a ProjectPaths instance.

    Args:
        project_root: Explicit UE project root. If None, auto-detect.
    """
    ue_root = resolve_project_root(project_root)
    repo_root = _discover_repo_root()
    return ProjectPaths(repo_root=repo_root, ue_project_root=ue_root)
