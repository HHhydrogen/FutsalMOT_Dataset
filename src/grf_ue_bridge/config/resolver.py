"""task 解析：把 task + profile + 本地配置合并为 resolved task。

resolved task 是唯一运行时契约：普通 Python CLI 与 UE Python（ue/run_task.py）
都读取它，不再分别实现两套路径解析。
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

from grf_ue_bridge.config import loader
from grf_ue_bridge.config import models as m
from grf_ue_bridge.config import paths as _paths
from grf_ue_bridge.config.paths import (
    PLACEHOLDER_DATASET_ROOT,
    PLACEHOLDER_REPO_ROOT,
    PLACEHOLDER_UE_PROJECT_ROOT,
    sanitize_path,
)


# ── 校验（只读，不写文件）───────────────────────────────────────────────

def validate_task(
    task_file: Path,
    env: Optional[dict] = None,
    local: Optional[dict] = None,
    allow_absolute_paths: bool = False,
) -> List[str]:
    """只读校验一个 task，返回问题列表（空 = 通过）。不生成任何文件。"""
    problems: List[str] = []
    task_file = task_file.resolve()

    try:
        task = loader.load_task_config(task_file)
    except Exception as e:  # noqa: BLE001
        return [f"task 解析失败: {e}"]

    task_dir = task_file.parent

    # profile 存在
    try:
        loader.resolve_profile_refs(task, task_dir)
    except ValueError as e:
        problems.append(str(e))

    # 本地配置可解析（必须字段存在）
    try:
        paths = local if local is not None else loader.resolve_local_paths(env)
    except ValueError as e:
        problems.append(str(e))
        return problems  # 路径无法继续解析

    repo_root, dataset_root = paths["repo_root"], paths["dataset_root"]

    # 路径可解析（含逃逸/绝对路径拒绝）
    try:
        traj = _resolve_output(task, repo_root, "trajectory", allow_absolute_paths)
    except ValueError as e:
        problems.append(f"trajectory_output: {e}")
        traj = None
    try:
        ds = _resolve_output(task, dataset_root, "dataset", allow_absolute_paths)
    except ValueError as e:
        problems.append(f"dataset_output: {e}")
        ds = None

    if traj and ds and traj == ds:
        problems.append("trajectory_output 与 dataset_episode_dir 冲突（相同路径）")
    if ds and ds.name != task.episode_name:
        problems.append(
            f"dataset_output 目录名 {ds.name!r} != episode_name {task.episode_name!r}"
            "（UE 端按 episode_id 定位数据集目录，两者必须一致）"
        )

    # 相机数量
    try:
        ue = loader.load_ue_profile(task, task_dir)
        cam_ids = (ue.get("annotation_export") or {}).get("cameras") or []
        expected_cams = task.audit.expected_cameras
        if cam_ids and len(cam_ids) != expected_cams:
            problems.append(
                f"UE profile 相机数 {len(cam_ids)} != audit.expected_cameras {expected_cams}"
            )
    except ValueError as e:
        problems.append(str(e))

    # 期望帧数 vs 导出步数
    try:
        export_cfg = loader.load_export_profile(task, task_dir)
        steps = export_cfg.num_steps
        factor = max(1, (export_cfg.target_fps or 10) // 10)
        expected_frames = steps * factor
        if expected_frames != task.audit.expected_frames_per_camera:
            problems.append(
                f"导出帧数 {expected_frames}（num_steps={steps}×factor={factor}）"
                f" != audit.expected_frames_per_camera {task.audit.expected_frames_per_camera}"
            )
    except ValueError as e:
        problems.append(str(e))

    return problems


def _resolve_output(
    task: m.DatasetTaskConfig, base: Path, kind: str, allow_absolute: bool
) -> Path:
    rel = (
        task.paths.trajectory_output if kind == "trajectory"
        else task.paths.dataset_output
    )
    if not rel:
        rel = f"outputs/{task.episode_name}" if kind == "trajectory" else task.episode_name
    return _paths.resolve_with_allow_absolute(rel, base, allow_absolute)


# ── 解析为 resolved task ────────────────────────────────────────────────

def resolve_task(
    task_file: Path,
    env: Optional[dict] = None,
    local: Optional[dict] = None,
    allow_absolute_paths: bool = False,
) -> m.ResolvedTask:
    """把 task 解析为运行时 resolved task（含绝对路径）。"""
    task_file = task_file.resolve()
    task = loader.load_task_config(task_file)
    task_dir = task_file.parent
    paths = local if local is not None else loader.resolve_local_paths(env)

    repo_root = paths["repo_root"]
    ue_project_root = paths["ue_project_root"]
    dataset_root = paths["dataset_root"]

    traj = _resolve_output(task, repo_root, "trajectory", allow_absolute_paths)
    ds = _resolve_output(task, dataset_root, "dataset", allow_absolute_paths)

    export_cfg = loader.load_export_profile(task, task_dir)
    if task.seed is not None:
        export_cfg = export_cfg.model_copy(update={"seed": task.seed})
    export_dict = export_cfg.model_dump()

    ue = loader.load_ue_profile(task, task_dir)
    actor_rel = str(ue.get("actor_mapping") or "ue/actor_mapping.example.json")
    actor_mapping = _paths.resolve_with_allow_absolute(
        actor_rel, repo_root, allow_absolute_paths
    )

    return m.ResolvedTask(
        task_id=task.task_id,
        episode_name=task.episode_name,
        source_task_file=str(task_file),
        repo_root=str(repo_root),
        ue_project_root=str(ue_project_root),
        dataset_root=str(dataset_root),
        trajectory_output=str(traj),
        dataset_episode_dir=str(ds),
        export_profile=export_dict,
        ue_profile=ue,
        actor_mapping=str(actor_mapping),
        postprocess=task.postprocess.model_dump(),
        audit=task.audit.model_dump(),
    )


def validate_resolved_task(resolved: m.ResolvedTask) -> List[str]:
    """校验 resolved task 结构（schema/version/路径包含）。"""
    problems: List[str] = []
    if resolved.schema_ != _paths.RESOLVED_TASK_SCHEMA:
        problems.append(f"resolved task schema 非法: {resolved.schema_!r}")
    if resolved.version != 1:
        problems.append(f"resolved task version 非法: {resolved.version!r}")

    def _contained(value: str, root: str, label: str) -> None:
        try:
            Path(value).resolve().relative_to(Path(root).resolve())
        except ValueError:
            problems.append(f"{label} 逃逸其根目录: {value}")

    _contained(resolved.trajectory_output, resolved.repo_root, "trajectory_output")
    _contained(resolved.dataset_episode_dir, resolved.dataset_root, "dataset_episode_dir")
    _contained(resolved.actor_mapping, resolved.repo_root, "actor_mapping")
    return problems


def runtime_dir(repo_root: Path) -> Path:
    """resolved task 运行时目录（gitignore）。"""
    return repo_root / ".futsalmot" / "runtime"


def save_resolved_task(resolved: m.ResolvedTask, repo_root: Path) -> Path:
    """原子写入 .futsalmot/runtime/<task_id>/resolved-task.json。"""
    out_dir = runtime_dir(repo_root) / resolved.task_id
    out_dir.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".resolved.", suffix=".json", dir=str(out_dir))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(resolved.model_dump_json(by_alias=True, indent=2) + "\n")
        final = out_dir / "resolved-task.json"
        os.replace(tmp, final)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return final


def load_resolved_task(path: Path) -> m.ResolvedTask:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if data.get("schema") != _paths.RESOLVED_TASK_SCHEMA:
        raise ValueError(f"resolved task schema 非法: {data.get('schema')!r}")
    if data.get("version") != 1:
        raise ValueError(f"resolved task version 非法: {data.get('version')!r}")
    return m.ResolvedTask(**data)


def sanitize_resolved_task(resolved: m.ResolvedTask) -> Dict:
    """把 resolved task 转为可移植 provenance 字典（绝对路径 → ${占位符}）。"""
    repo = Path(resolved.repo_root)
    ue_root = Path(resolved.ue_project_root)
    ds_root = Path(resolved.dataset_root)

    def _san(s: str) -> str:
        return sanitize_path(s, repo, ue_root, ds_root)

    out = resolved.model_dump(by_alias=True)
    out["source_task_file"] = _san(resolved.source_task_file)
    out["repo_root"] = _san(resolved.repo_root)
    out["ue_project_root"] = _san(resolved.ue_project_root)
    out["dataset_root"] = _san(resolved.dataset_root)
    out["trajectory_output"] = _san(resolved.trajectory_output)
    out["dataset_episode_dir"] = _san(resolved.dataset_episode_dir)
    out["actor_mapping"] = _san(resolved.actor_mapping)
    return out


# ── active task（可选便利）──────────────────────────────────────────────

ACTIVE_TASK_FILENAME = ".futsalmot/active-task.json"


def active_task_path(repo_root: Path) -> Path:
    return repo_root / ACTIVE_TASK_FILENAME


def save_active_task(task_file: Path, repo_root: Path) -> Path:
    """写入 active task（只保存 task 文件路径：仓库内用相对，仓库外用绝对）。"""
    p = active_task_path(repo_root)
    p.parent.mkdir(parents=True, exist_ok=True)
    tf = task_file.resolve()
    try:
        rel = tf.relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        rel = tf.as_posix()  # task 在仓库外：保存绝对路径
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        json.dump({"schema": "futsalmot_active_task", "version": 1, "task_file": rel},
                  f, ensure_ascii=False, indent=2)
    return p


def load_active_task(repo_root: Path) -> Optional[Path]:
    """读取 active task 文件路径（不存在返回 None）。"""
    p = active_task_path(repo_root)
    if not p.is_file():
        return None
    try:
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    rel = data.get("task_file")
    if not rel:
        return None
    cand = (repo_root / rel).resolve()
    if not cand.is_file():
        return None
    return cand


def clear_active_task(repo_root: Path) -> None:
    p = active_task_path(repo_root)
    if p.exists():
        p.unlink()
