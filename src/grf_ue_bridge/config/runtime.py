"""Machine-local runtime path 配置与环境变量展开。"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Mapping, Optional, Tuple


_VAR_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def load_local_runtime_config(repo_root: Path, path: Optional[Path] = None) -> dict:
    """读取 machine-local 配置；未提供文件时返回空配置。"""
    config_path = Path(path) if path is not None else repo_root / ".futsalmot" / "local.json"
    if not config_path.is_file():
        return {}
    try:
        with config_path.open(encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"local runtime config 读取失败: {config_path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"local runtime config 顶层必须是 JSON 对象: {config_path}")
    paths = value.get("paths", {})
    if not isinstance(paths, dict):
        raise ValueError(f"local runtime config.paths 必须是 JSON 对象: {config_path}")
    for name in ("dataset_root", "ue_project_root"):
        if name not in paths or not isinstance(paths[name], str) or not paths[name].strip():
            raise ValueError(f"local runtime config 缺少 required path: {name}")
    return value


def expand_runtime_path(value: str, env: Mapping[str, str]) -> str:
    """展开 `${NAME}`；变量缺失时明确失败。"""
    def replace(match: re.Match) -> str:
        name = match.group(1)
        if name not in env or not env[name].strip():
            raise ValueError(f"missing runtime path variable: {name}")
        return env[name]

    return _VAR_RE.sub(replace, value)


def resolve_runtime_paths(
    task_file: Path,
    task,
    repo_root: Path,
    dataset_root: Optional[str] = None,
    ue_project_root: Optional[str] = None,
    local_config: Optional[Path] = None,
) -> Tuple[Path, Path]:
    """按 CLI > local > environment > task placeholder > legacy 解析路径。"""
    config_path = local_config
    if config_path is None:
        for candidate_root in (task_file.parent, *task_file.parent.parents):
            candidate = candidate_root / ".futsalmot" / "local.json"
            if candidate.is_file():
                config_path = candidate
                break
            if candidate_root == repo_root:
                break
    local = load_local_runtime_config(repo_root, config_path)
    local_paths = local.get("paths", {})
    env = os.environ

    def choose(name: str, override: Optional[str], env_names: Tuple[str, ...]) -> Path:
        source = override
        if source is None:
            source = local_paths.get(name)
        if source is None:
            for env_name in env_names:
                source = env.get(env_name)
                if source:
                    break
        if source is None:
            source = getattr(task, name, None)
            if source is not None and Path(source).is_absolute():
                import warnings
                warnings.warn(
                    f"Using deprecated absolute path from task config: {name}",
                    RuntimeWarning,
                    stacklevel=3,
                )
        if not isinstance(source, str) or not source.strip():
            raise ValueError(f"missing runtime path: {name}")
        expanded = expand_runtime_path(source.strip(), env)
        path = Path(expanded).expanduser()
        if not path.is_absolute():
            path = repo_root / path
        return path.resolve()

    return (
        choose("dataset_root", dataset_root, ("FUTSALMOT_DATASET_ROOT",)),
        choose(
            "ue_project_root",
            ue_project_root,
            ("FUTSALMOT_UE_ROOT", "FUTSALMOT_UE_PROJECT_ROOT"),
        ),
    )
