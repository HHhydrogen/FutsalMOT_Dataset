"""task 解析：把单 config（含绝对机器路径）归一化为 resolved task。

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

def validate_task(task_file: Path) -> List[str]:
    """只读校验一个 task，返回问题列表（空 = 通过）。不生成任何文件。"""
    problems: List[str] = []
    task_file = task_file.resolve()

    try:
        task = loader.load_task_config(task_file)
    except Exception as e:  # noqa: BLE001
        return [f"task 解析失败: {e}"]

    # 机器路径（必填）
    if not task.dataset_root:
        problems.append("缺少 dataset_root")
    if not task.ue_project_root:
        problems.append("缺少 ue_project_root")

    # 相机数量
    cam_ids = (task.ue.annotation_export or {}).get("cameras") or []
    expected_cams = task.audit.expected_cameras
    if cam_ids and len(cam_ids) != expected_cams:
        problems.append(
            f"相机数 {len(cam_ids)} != audit.expected_cameras {expected_cams}"
        )

    # 期望帧数 vs 导出步数
    steps = task.export.num_steps
    factor = max(1, (task.export.target_fps or 10) // 10)
    expected_frames = steps * factor
    if expected_frames != task.audit.expected_frames_per_camera:
        problems.append(
            f"导出帧数 {expected_frames}（num_steps={steps}×factor={factor}）"
            f" != audit.expected_frames_per_camera {task.audit.expected_frames_per_camera}"
        )

    # source duration 必须满足 output_frames × tempo（fail-fast，禁止截断/重复/补末帧）
    # source_time = dataset_time × tempo；GRF 引擎需运行到 source_last 时刻。
    # compute_source_steps 已用 +2 步（0.2s）为 Hermite 后继 sample 留余量，
    # 故此处只需保证 game_duration 覆盖到 source_last = (output_frames-1)/fps × tempo。
    tempo = float(task.export.trajectory_time_scale or 1.0)
    if tempo >= 1.0 and task.export.game_duration:
        output_frames = expected_frames
        source_last = (output_frames - 1) / float(task.export.target_fps or 10) * tempo
        available_source_s = float(task.export.game_duration)
        if available_source_s < source_last:
            problems.append(
                f"source duration 不足：需要 ≥ {source_last:.1f}s（output {output_frames}帧 "
                f"× tempo {tempo} 的 source_last），但 game_duration={available_source_s:.0f}s。"
                f"请增大 export.game_duration 或降低 num_steps/tempo，禁止截断/重复/补末帧。"
            )

    return problems


# ── 解析为 resolved task ────────────────────────────────────────────────

def resolve_task(task_file: Path) -> m.ResolvedTask:
    """把单 config 解析为运行时 resolved task（含绝对路径）。"""
    task_file = task_file.resolve()
    task = loader.load_task_config(task_file)

    repo_root = _paths.default_repo_root()
    dataset_root = Path(task.dataset_root).expanduser().resolve()
    ue_project_root = Path(task.ue_project_root).expanduser().resolve()
    episode_dir = dataset_root / task.episode_name

    actor_mapping = _paths.resolve_task_relative(task.ue.actor_mapping, repo_root)

    # 把 export.playback_fps 注入 annotation_export，让 render_episode 帧映射正确
    # （render_episode 用 annotation_export.playback_fps 做 Sequence display rate 帧映射）
    ue_profile = task.ue.model_dump()
    ann_export = ue_profile.get("annotation_export") or {}
    if "playback_fps" not in ann_export:
        ann_export["playback_fps"] = task.export.playback_fps
    ue_profile["annotation_export"] = ann_export

    return m.ResolvedTask(
        task_id=task.task_id,
        episode_name=task.episode_name,
        source_task_file=str(task_file),
        repo_root=str(repo_root),
        ue_project_root=str(ue_project_root),
        dataset_root=str(dataset_root),
        trajectory_output=str(episode_dir),
        dataset_episode_dir=str(episode_dir),
        export_profile=task.export.model_dump(),
        ue_profile=ue_profile,
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

    _contained(resolved.trajectory_output, resolved.dataset_root, "trajectory_output")
    _contained(resolved.dataset_episode_dir, resolved.dataset_root, "dataset_episode_dir")
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
