"""配置加载：数据集 task（单 config）。

task 文件为自包含 JSON：导出 + UE + 机器路径内联，直接加载校验即可。
"""

from __future__ import annotations

import json
import copy
import sys
from pathlib import Path

from grf_ue_bridge.config import models as m
from grf_ue_bridge.config import paths as _paths


def _read_json(path: Path, label: str) -> dict:
    if not path.is_file():
        raise ValueError(f"{label} 不存在: {path}")
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"{label} 不是合法 JSON ({path}): {e}")


def load_task_config(path: Path) -> m.DatasetTaskConfig:
    """加载并校验一个数据集 task 配置（单 config，含内联 export/ue）。"""
    data = _read_json(path, "task 配置")
    if data.get("schema") not in (None, _paths.TASK_SCHEMA):
        raise ValueError(f"task schema 非法: {data.get('schema')!r}")
    raw_simulation = data.get("simulation")
    raw_mapping = (data.get("ue") or {}).get("camera_mapping")
    if isinstance(raw_simulation, dict):
        profiles = ((raw_simulation.get("camera") or {}).get("profiles") or {})
        partial = profiles.get("P01")
        if isinstance(partial, dict):
            if partial.get("coverage") != "partial_field":
                raise ValueError("P01 必须使用 coverage=partial_field")
            if partial.get("region") != "left_half":
                raise ValueError("P01 必须使用 region=left_half")
            if (
                partial.get("placement_template") is not None
                and partial.get("placement_template") != "left_half_sideline_high_v1"
            ):
                raise ValueError(
                    "P01 必须使用 placement_template=left_half_sideline_high_v1"
                )
            ue_dir = Path(__file__).resolve().parent.parent.parent.parent / "ue"
            if str(ue_dir) not in sys.path:
                sys.path.insert(0, str(ue_dir))
            from camera_distribution import resolve_partial_static_camera_profile

            data = copy.deepcopy(data)
            normalized_profiles = data["simulation"]["camera"]["profiles"]
            normalized_profiles["P01"] = resolve_partial_static_camera_profile("P01", partial)
    if raw_simulation is not None and isinstance(raw_mapping, dict):
        for camera_id, entry in raw_mapping.items():
            if not isinstance(entry, dict):
                raise ValueError(f"Canonical ID {camera_id} 的 camera_mapping 必须为对象")
            for field in ("actor", "sequence"):
                value = entry.get(field)
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(
                        f"Canonical ID {camera_id} 的 camera_mapping.{field} 不能为空"
                    )
    task = m.DatasetTaskConfig(**data)
    task.postprocess.validate_formats()
    if task.simulation is not None:
        required_camera_ids = {"C1", "C2", "C3", "C4", "C5"}
        mapping = task.ue.camera_mapping
        if mapping is None:
            raise ValueError(
                "simulation.camera 存在时必须提供 ue.camera_mapping，且完整包含 C1..C5"
            )
        profile_ids = set(task.simulation.camera.profiles)
        mapping_ids = set(mapping)
        allowed_mapping_ids = required_camera_ids | {"P01"}
        p01_required_but_missing = "P01" in profile_ids and "P01" not in mapping_ids
        if (
            not required_camera_ids.issubset(mapping_ids)
            or not mapping_ids.issubset(allowed_mapping_ids)
            or p01_required_but_missing
        ):
            missing = sorted(required_camera_ids - mapping_ids)
            unknown = sorted(mapping_ids - allowed_mapping_ids)
            if "P01" in profile_ids and "P01" not in mapping_ids:
                missing.append("P01")
            details = []
            if missing:
                details.append(f"缺少 {', '.join(missing)}")
            if unknown:
                details.append(f"未知 {', '.join(unknown)}")
            raise ValueError(
                "ue.camera_mapping 必须恰好包含 C1,C2,C3,C4,C5（" + "；".join(details) + "）"
            )
        if "P01" in mapping_ids and "P01" not in profile_ids:
            raise ValueError("ue.camera_mapping.P01 只有在 P01 profile 存在时才允许")
        episode_profile = task.simulation.camera.distribution.episode_profile
        if episode_profile == "anchor_plus_one_partial":
            if "P01" not in profile_ids or "P01" not in mapping_ids:
                raise ValueError(
                    "anchor_plus_one_partial 要求同时提供 P01 profile 和 ue.camera_mapping.P01"
                )
        elif episode_profile == "anchor_plus_two_partial":
            raise ValueError(
                "anchor_plus_two_partial 当前需要第二个已实现 Partial Camera，例如 P02"
            )
        task.simulation.camera.validate_camera_contract()
    return task
