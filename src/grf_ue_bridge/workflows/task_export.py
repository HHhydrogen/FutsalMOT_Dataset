"""task 工作流：导出轨迹 + 写 provenance（复用 run_episode/export_episode）。"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Callable

from grf_ue_bridge.config import models as m
from grf_ue_bridge.config import resolver as _resolver


def _write_provenance(resolved: m.ResolvedTask, traj: Path) -> None:
    """按 task 写 episode 级 provenance（可移植，无本机路径）。"""
    prov = traj / "provenance"
    prov.mkdir(parents=True, exist_ok=True)

    src = Path(resolved.source_task_file)
    if src.is_file():
        shutil.copy2(src, prov / "task.json")

    (prov / "resolved-task.sanitized.json").write_text(
        json.dumps(_resolver.sanitize_resolved_task(resolved), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (prov / "export-profile.json").write_text(
        json.dumps(resolved.export_profile, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (prov / "ue-profile.json").write_text(
        json.dumps(resolved.ue_profile, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    mp = Path(resolved.actor_mapping)
    if mp.is_file():
        shutil.copy2(mp, prov / "actor-mapping.json")
    lock = Path(resolved.repo_root) / "external_sources.lock.json"
    if lock.is_file():
        shutil.copy2(lock, prov / "external_sources.lock.json")


def run_export(
    resolved: m.ResolvedTask, *, print_fn: Callable[[str], None] = print
) -> int:
    """执行导出（复用现有 run_episode/export_episode，不复制 exporter 实现）。"""
    from grf_ue_bridge.config.models import ExportConfig
    from grf_ue_bridge.exporter import export_episode
    from grf_ue_bridge.grf_runner import run_episode

    export_cfg = ExportConfig(**resolved.export_profile)
    traj = Path(resolved.trajectory_output)

    print_fn(f"Export task: {resolved.task_id}  episode={resolved.episode_name}")
    print_fn(f"  root seed: {export_cfg.seed}  scenario: {export_cfg.scenario}  steps: {export_cfg.num_steps}")
    print_fn(f"  trajectory output: {traj}")

    result = run_episode(
        scenario=export_cfg.scenario,
        seed=export_cfg.seed,
        num_steps=export_cfg.num_steps,
        render=export_cfg.render,
        number_of_left_players_agent_controls=export_cfg.number_of_left_players_agent_controls,
        number_of_right_players_agent_controls=export_cfg.number_of_right_players_agent_controls,
    )
    export_episode(export_cfg, result, traj)
    _write_provenance(resolved, traj)
    print_fn(f"Provenance written: {traj / 'provenance'}")
    return 0
