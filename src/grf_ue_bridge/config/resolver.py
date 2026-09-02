"""task 解析：把单 config（含绝对机器路径）归一化为 resolved task。

resolved task 是唯一运行时契约：普通 Python CLI 与 UE Python（ue/run_task.py）
都读取它，不再分别实现两套路径解析。
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Mapping, Optional

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

def validate_supported_fps(fps: int) -> None:
    """校验现有导出管线支持的目标帧率。"""
    if fps < 10 or fps % 10:
        raise ValueError("output.fps 必须为正的 10 的倍数且至少为 10")

def resolve_local_config(cli_path: Optional[Path], env: Mapping[str, str]) -> Path:
    """按 CLI、环境变量顺序选择 local config，不自动搜索。"""
    selected = cli_path or (Path(env["FUTSALMOT_LOCAL_CONFIG"]) if env.get("FUTSALMOT_LOCAL_CONFIG") else None)
    if selected is None:
        raise ValueError("缺少 local config：请提供 --local-config 或 FUTSALMOT_LOCAL_CONFIG")
    return selected.expanduser().resolve()


def _local_paths(path: Path) -> tuple[Path, Path]:
    local = loader.load_local_machine_config(path)
    if not local.dataset_root.strip() or not local.ue_project_root.strip():
        raise ValueError("local config 路径为空")
    dataset_root = Path(local.dataset_root).expanduser().resolve()
    ue_root = Path(local.ue_project_root).expanduser().resolve()
    if not dataset_root.is_dir():
        raise ValueError(f"dataset_root 不存在或不是目录: {dataset_root}")
    if not ue_root.is_dir():
        raise ValueError(f"ue_project_root 不存在或不是目录: {ue_root}")
    projects = [path for path in ue_root.glob("*.uproject") if path.is_file()]
    if len(projects) != 1:
        raise ValueError(f"ue_project_root 必须恰好包含一个 .uproject: {ue_root}")
    return dataset_root, ue_root


def validate_task(task_file: Path, local_config: Optional[Path] = None) -> List[str]:
    """只读校验一个 task，返回问题列表（空 = 通过）。不生成任何文件。"""
    problems: List[str] = []
    task_file = task_file.resolve()

    try:
        task = loader.load_task_config(task_file)
        if isinstance(task, m.TaskConfigV3):
            validate_supported_fps(task.output.fps)
            _local_paths(resolve_local_config(local_config, os.environ))
            return []
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

def resolve_task(task_file: Path, local_config: Optional[Path] = None) -> m.ResolvedTask:
    """把单 config 解析为运行时 resolved task（含绝对路径）。"""
    task_file = task_file.resolve()
    task = loader.load_task_config(task_file)

    repo_root = _paths.default_repo_root()
    config_v3: Dict = {}
    if isinstance(task, m.TaskConfigV3):
        dataset_root, ue_project_root = _local_paths(resolve_local_config(local_config, os.environ))
        episode_name = task.episode_id
        fps = task.output.fps
        validate_supported_fps(fps)
        expected_frames = task.simulation.steps * max(1, fps // 10)
        sequences = [{"name": f"FutsalMOT_{episode_name}_{camera_id}", "camera_id": camera_id,
                      "public_sequence_name": f"FutsalMOT_{episode_name}_{camera_id}",
                      "camera_actor": actor}
                     for camera_id, actor in task.cameras.items()]
        annotations = set(task.output.annotations)
        include_ball = "ball" in task.output.classes
        ann_export = {
            "enabled": True, "playback_fps": fps,
            "image_width": task.output.resolution[0], "image_height": task.output.resolution[1],
            "cameras": list(task.cameras.values()), "camera_mapping": dict(task.cameras),
            "export_mot": "mot" in annotations,
            "export_mots": "mots" in annotations, "export_pose": "pose" in annotations,
            "camera_actors": list(task.cameras.values()), "camera_count": len(sequences),
            "public_sequence_names": [s["name"] for s in sequences],
            "include_ball": include_ball,
            "instance_mask": {"enabled": True, "mask_source": "object_id_pass"},
            "render_rgb": {"enabled": True, "output_resolution_x": task.output.resolution[0],
                           "output_resolution_y": task.output.resolution[1], "frame_rate": fps},
        }
        export_profile = m.ExportConfig(
            scenario=task.simulation.scenario, seed=task.simulation.seed,
            num_steps=task.simulation.steps, target_fps=fps, playback_fps=fps,
            trajectory_time_scale=task.simulation.trajectory_time_scale,
            game_duration=task.simulation.game_duration,
            left_team_difficulty=task.simulation.left_team_difficulty,
            right_team_difficulty=task.simulation.right_team_difficulty,
            number_of_left_players_agent_controls=task.simulation.number_of_left_players_agent_controls,
            number_of_right_players_agent_controls=task.simulation.number_of_right_players_agent_controls,
        ).model_dump()
        ue_profile = m.UeProfile(
            sequences=sequences, annotation_export=ann_export,
            ball_rolling=task.simulation.ball_rolling,
        ).model_dump()
        postprocess = m.PostprocessTaskConfig(
            include_ball=include_ball,
            formats=[a for a in ("mot", "mots") if a in annotations],
            yolo_pose={"enabled": "pose" in annotations},
        ).model_dump()
        audit = {"expected_cameras": len(sequences), "expected_frames_per_camera": expected_frames}
        config_v3 = {"episode": episode_name, "seed": task.simulation.seed,
                     "steps": task.simulation.steps, "fps": fps,
                     "resolution": list(task.output.resolution), "cameras": dict(task.cameras),
                     "annotations": list(task.output.annotations), "classes": list(task.output.classes),
                     "expected_frames": expected_frames,
                     "public_sequence_names": [s["name"] for s in sequences]}
        episode_name = task.episode_id
    else:
        dataset_root = Path(task.dataset_root).expanduser().resolve()
        ue_project_root = Path(task.ue_project_root).expanduser().resolve()
        episode_name = task.episode_name
        export_profile = task.export.model_dump()
        ue_profile = task.ue.model_dump()
        postprocess = task.postprocess.model_dump()
        audit = task.audit.model_dump()
    episode_dir = dataset_root / episode_name

    actor_mapping = _paths.resolve_task_relative(
        task.ue.actor_mapping if not isinstance(task, m.TaskConfigV3) else "ue/actor_mapping.example.json",
        repo_root,
    )
    if isinstance(task, m.TaskConfigV3) and not actor_mapping.is_file():
        raise ValueError(f"actor mapping 不存在或不是文件: {actor_mapping}")

    # 把 export.playback_fps 注入 annotation_export，让 render_episode 帧映射正确
    # （render_episode 用 annotation_export.playback_fps 做 Sequence display rate 帧映射）
    ann_export = ue_profile.get("annotation_export") or {}
    if not isinstance(task, m.TaskConfigV3) and "playback_fps" not in ann_export:
        ann_export["playback_fps"] = task.export.playback_fps
    ue_profile["annotation_export"] = ann_export

    return m.ResolvedTask(
        task_id=task.task_id if not isinstance(task, m.TaskConfigV3) else task.episode_id,
        episode_name=episode_name,
        source_task_file=str(task_file),
        repo_root=str(repo_root),
        ue_project_root=str(ue_project_root),
        dataset_root=str(dataset_root),
        trajectory_output=str(episode_dir),
        dataset_episode_dir=str(episode_dir),
        export_profile=export_profile,
        ue_profile=ue_profile,
        actor_mapping=str(actor_mapping),
        postprocess=postprocess,
        audit=audit,
        artifact_policy=dict(getattr(task, "artifact_policy", None) or {"profile": "research_minimal"}),
        config_v3=config_v3,
    )


def resolved_task_summary(resolved: m.ResolvedTask) -> Dict:
    """返回 CLI 与诊断使用的 resolved task 摘要。"""
    if resolved.config_v3:
        return dict(resolved.config_v3)
    return {"episode": resolved.episode_name, "fps": resolved.export_profile.get("playback_fps"),
            "expected_frames": resolved.audit.get("expected_frames_per_camera")}


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
