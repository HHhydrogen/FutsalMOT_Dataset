"""配置加载：本地配置、task、export/UE profile。

加载优先级（机器路径）：
    CLI 临时覆盖 > 环境变量 FUTSALMOT_* > .futsalmot.local.json > 可移植默认值。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

from grf_ue_bridge.config import models as m
from grf_ue_bridge.config.paths import (
    ENV_DATASET_ROOT,
    ENV_REPO_ROOT,
    ENV_UE_PROJECT_ROOT,
    default_repo_root,
    find_repo_root,
    resolve_local_config_path,
)
from grf_ue_bridge.config import paths as _paths


# ── 本地机器配置 ────────────────────────────────────────────────────────

def load_local_config(
    env: Optional[dict] = None, cwd: Optional[Path] = None
) -> Optional[m.LocalConfig]:
    """读取 .futsalmot.local.json（不存在返回 None）。"""
    cfg_path = resolve_local_config_path(env, cwd)
    if not cfg_path.is_file():
        return None
    with open(cfg_path, encoding="utf-8") as f:
        data = json.load(f)
    if data.get("schema") not in (None, _paths.LOCAL_CONFIG_SCHEMA):
        raise ValueError(f"local 配置 schema 非法: {data.get('schema')!r}")
    return m.LocalConfig(**data)


def resolve_local_paths(
    env: Optional[dict] = None,
    cwd: Optional[Path] = None,
    allow_missing_local: bool = False,
) -> Dict[str, Path]:
    """解析机器根路径（绝对），优先级 env > local 文件 > 默认。

    repo_root 有可移植默认；ue_project_root 与 dataset_root 缺失时报清晰错误
    （不猜测盘符）。返回键：repo_root / ue_project_root / dataset_root /
    cache_dir / log_dir（后两者可为 None）。
    """
    e = env if env is not None else os.environ
    local = load_local_config(env, cwd)

    repo_root = Path(
        e.get(ENV_REPO_ROOT)
        or (local.repo_root if local and local.repo_root else None)
        or default_repo_root()
    ).expanduser().resolve()

    ue_root = e.get(ENV_UE_PROJECT_ROOT) or (local.ue_project_root if local else None)
    ds_root = e.get(ENV_DATASET_ROOT) or (local.dataset_root if local else None)

    missing = []
    if not ue_root:
        missing.append("ue_project_root")
    if not ds_root:
        missing.append("dataset_root")
    if missing and not allow_missing_local:
        raise ValueError(
            "缺少必需的本地路径: " + ", ".join(missing)
            + "。请在 .futsalmot.local.json 或 FUTSALMOT_UE_PROJECT_ROOT / "
              "FUTSALMOT_DATASET_ROOT 环境变量中提供。"
        )

    out = {
        "repo_root": repo_root,
        "ue_project_root": Path(ue_root).expanduser().resolve() if ue_root else repo_root,
        "dataset_root": Path(ds_root).expanduser().resolve() if ds_root else repo_root,
        "cache_dir": None,
        "log_dir": None,
    }
    if local and local.cache_dir:
        out["cache_dir"] = Path(local.cache_dir).expanduser().resolve()
    if local and local.log_dir:
        out["log_dir"] = Path(local.log_dir).expanduser().resolve()
    return out


def apply_task_path_overrides(
    task: m.DatasetTaskConfig, local: Dict[str, Path]
) -> Dict[str, Path]:
    """在解析后的本地路径之上应用 task 内可选机器路径（单一 config 用法）。

    task 内字段优先级最高：task.dataset_root > 环境变量 > .futsalmot.local.json。
    """
    out = dict(local)
    if task.dataset_root:
        out["dataset_root"] = Path(task.dataset_root).expanduser().resolve()
    if task.ue_project_root:
        out["ue_project_root"] = Path(task.ue_project_root).expanduser().resolve()
    if task.repo_root:
        out["repo_root"] = Path(task.repo_root).expanduser().resolve()
    return out


def resolve_paths_for_task(
    task: m.DatasetTaskConfig,
    env: Optional[dict] = None,
    cwd: Optional[Path] = None,
) -> Dict[str, Path]:
    """按优先级解析机器路径：task 内字段 > env > local 文件 > 默认。"""
    base = resolve_local_paths(env, cwd)
    return apply_task_path_overrides(task, base)


# ── task 与 profile 加载 ────────────────────────────────────────────────

def _read_json(path: Path, label: str) -> dict:
    if not path.is_file():
        raise ValueError(f"{label} 不存在: {path}")
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"{label} 不是合法 JSON ({path}): {e}")


def load_task_config(path: Path) -> m.DatasetTaskConfig:
    """加载并校验一个 task 配置。"""
    data = _read_json(path, "task 配置")
    if data.get("schema") not in (None, _paths.TASK_SCHEMA):
        raise ValueError(f"task schema 非法: {data.get('schema')!r}")
    task = m.DatasetTaskConfig(**data)
    task.postprocess.validate_formats()
    return task


def load_export_profile(task: m.DatasetTaskConfig, task_dir: Path):
    """按 task 的 export_profile 引用加载导出 profile。

    返回 ExportConfig 实例（复用现有模型做字段校验），无业务复制。
    """
    from grf_ue_bridge.config import ExportConfig

    path = task_dir / task.export_profile
    data = _read_json(path, "export profile")
    if "schema" in data and data["schema"] not in (None, _paths.EXPORT_PROFILE_SCHEMA):
        raise ValueError(f"export profile schema 非法: {data['schema']!r}")
    # ExportConfig 不接受 schema 键 → 剥离
    data = {k: v for k, v in data.items() if k not in ("schema", "version")}
    return ExportConfig(**data)


def load_ue_profile(task: m.DatasetTaskConfig, task_dir: Path) -> dict:
    """按 task 的 ue_profile 引用加载 UE profile（episode 无关 dict）。"""
    path = task_dir / task.ue_profile
    data = _read_json(path, "UE profile")
    if data.get("schema") not in (None, _paths.UE_PROFILE_SCHEMA):
        raise ValueError(f"UE profile schema 非法: {data.get('schema')!r}")
    return data


def resolve_profile_refs(
    task: m.DatasetTaskConfig, task_dir: Path
) -> Tuple[Path, Path]:
    """解析 task 引用的 profile 绝对路径（仅校验存在，不读取）。"""
    export_p = (task_dir / task.export_profile).resolve()
    ue_p = (task_dir / task.ue_profile).resolve()
    if not export_p.is_file():
        raise ValueError(f"export profile 不存在: {export_p}")
    if not ue_p.is_file():
        raise ValueError(f"UE profile 不存在: {ue_p}")
    return export_p, ue_p
