"""路径解析与可移植性工具。

task 单 config 直接含绝对机器路径（入库）；resolved task（运行时，被 gitignore）
使用绝对路径；provenance 快照用 `${REPO_ROOT}` 等占位符做可移植替换。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

TASK_SCHEMA = "futsalmot_dataset_task"
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


def resolve_task_relative(rel: str, base: Path) -> Path:
    """把 task 内路径解析为绝对路径。

    绝对路径原样使用（单 config 可含机器路径）；相对路径解析到 base 之下并
    拒绝 `..` 逃逸。

    Args:
        rel: task 中声明的路径（相对如 "ue/actor_mapping.example.json"，或绝对）。
        base: 相对路径的允许根目录（repo_root 等）。

    Returns:
        绝对路径（已 resolve）。

    Raises:
        ValueError: 路径为空，或相对路径逃逸出 base。
    """
    rel = rel.strip()
    if not rel:
        raise ValueError("路径为空")
    p = Path(rel).expanduser()
    if p.is_absolute():
        return p.resolve()
    base_r = base.resolve()
    cand = (base_r / p).resolve()
    try:
        cand.relative_to(base_r)
    except ValueError:
        raise ValueError(f"路径逃逸允许根目录 {base_r}: {rel!r}")
    return cand


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
