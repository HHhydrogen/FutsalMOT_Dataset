"""路径解析、安全与可移植性工具。

task 文件默认禁止盘符/UNC 绝对路径与 `..` 逃逸；resolved task（运行时，
被 gitignore）使用绝对路径；provenance 快照用 `${REPO_ROOT}` 等占位符
做可移植替换。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

ENV_REPO_ROOT = "FUTSALMOT_REPO_ROOT"
ENV_UE_PROJECT_ROOT = "FUTSALMOT_UE_PROJECT_ROOT"
ENV_DATASET_ROOT = "FUTSALMOT_DATASET_ROOT"
ENV_LOCAL_CONFIG = "FUTSALMOT_LOCAL_CONFIG"

LOCAL_CONFIG_FILENAME = ".futsalmot.local.json"
LOCAL_CONFIG_SCHEMA = "futsalmot_local_config"
TASK_SCHEMA = "futsalmot_dataset_task"
UE_PROFILE_SCHEMA = "futsalmot_ue_profile"
EXPORT_PROFILE_SCHEMA = "futsalmot_export_profile"
RESOLVED_TASK_SCHEMA = "futsalmot_resolved_task"

# 可移植 provenance 占位符
PLACEHOLDER_REPO_ROOT = "${REPO_ROOT}"
PLACEHOLDER_UE_PROJECT_ROOT = "${UE_PROJECT_ROOT}"
PLACEHOLDER_DATASET_ROOT = "${DATASET_ROOT}"


def find_repo_root(start: Optional[Path] = None) -> Path:
    """向上查找含 pyproject.toml 的仓库根。"""
    cur = Path(start or Path.cwd()).resolve()
    for p in [cur, *cur.parents]:
        if (p / "pyproject.toml").is_file():
            return p
    return cur


def default_repo_root() -> Path:
    """默认仓库根：从本包文件位置向上找 pyproject.toml；找不到回退 cwd。"""
    here = Path(__file__).resolve()
    for p in [here, *here.parents]:
        if (p / "pyproject.toml").is_file():
            return p
    return Path.cwd()


def resolve_local_config_path(env: Optional[dict] = None, cwd: Optional[Path] = None) -> Path:
    """local 配置路径：env FUTSALMOT_LOCAL_CONFIG > <cwd>/.futsalmot.local.json。"""
    e = env if env is not None else os.environ
    if e.get(ENV_LOCAL_CONFIG):
        return Path(e[ENV_LOCAL_CONFIG]).expanduser().resolve()
    return Path(cwd or Path.cwd()) / LOCAL_CONFIG_FILENAME


def is_absolute_unsafe(value: str) -> bool:
    """是否为盘符/UNC 绝对路径（task 内默认拒绝）。"""
    v = value.strip()
    if v.startswith("\\\\") or v.startswith("//"):
        return True  # UNC
    if "://" in v:
        return True  # scheme://
    p = Path(v)
    return p.is_absolute()  # 含盘符 C:\ 或 POSIX /


def resolve_task_relative(rel: str, base: Path) -> Path:
    """把 task 内相对路径解析到 base 之下；默认拒绝绝对路径与 `..` 逃逸。

    Args:
        rel: task 中声明的相对路径（如 "outputs/ep_s42"）。
        base: 允许的根目录（repo_root / dataset_root）。

    Returns:
        base 下的绝对路径（已 resolve）。

    Raises:
        ValueError: 路径为绝对路径，或逃逸出 base。
    """
    rel = rel.strip()
    if not rel:
        raise ValueError("路径为空")
    if is_absolute_unsafe(rel):
        raise ValueError(f"task 内默认禁止绝对路径: {rel!r}")
    base_r = base.resolve()
    cand = (base_r / rel).resolve()
    try:
        cand.relative_to(base_r)
    except ValueError:
        raise ValueError(f"路径逃逸允许根目录 {base_r}: {rel!r}")
    return cand


def resolve_with_allow_absolute(rel: str, base: Path, allow_absolute: bool = False) -> Path:
    """允许时把绝对路径原样使用；否则走 resolve_task_relative。"""
    rel = rel.strip()
    if allow_absolute and is_absolute_unsafe(rel):
        return Path(rel).expanduser().resolve()
    return resolve_task_relative(rel, base)


def sanitize_path(value: str, repo_root: Path, ue_project_root: Path, dataset_root: Path) -> str:
    """把绝对路径替换为可移植占位符（provenance 用）。"""
    v = Path(value).as_posix()
    for placeholder, root in (
        (PLACEHOLDER_REPO_ROOT, Path(repo_root)),
        (PLACEHOLDER_UE_PROJECT_ROOT, Path(ue_project_root)),
        (PLACEHOLDER_DATASET_ROOT, Path(dataset_root)),
    ):
        r = root.resolve().as_posix()
        if v == r:
            return placeholder
        if v.startswith(r + "/"):
            return placeholder + v[len(r):]
    return v
