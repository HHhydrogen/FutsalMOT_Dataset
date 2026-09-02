"""配置加载：数据集 task（单 config）。

task 文件为自包含 JSON：导出 + UE + 机器路径内联，直接加载校验即可。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from grf_ue_bridge.config import models as m
from grf_ue_bridge.config.models import TaskConfig
from grf_ue_bridge.config import paths as _paths


def _read_json(path: Path, label: str) -> dict:
    if not path.is_file():
        raise ValueError(f"{label} 不存在: {path}")
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"{label} 不是合法 JSON ({path}): {e}")


def load_task_config(path: Path) -> TaskConfig:
    """加载并校验一个数据集 task 配置（单 config，含内联 export/ue）。"""
    data = _read_json(path, "task 配置")
    if data.get("schema") == _paths.TASK_V3_SCHEMA and data.get("version") != 3:
        raise ValueError(f"Config v3 version 非法: {data.get('version')!r}")
    if data.get("schema") == _paths.TASK_V3_SCHEMA:
        return m.TaskConfigV3(**data)
    if data.get("schema") not in (None, _paths.TASK_SCHEMA):
        raise ValueError(f"task schema 非法: {data.get('schema')!r}")
    task = m.DatasetTaskConfig(**data)
    task.postprocess.validate_formats()
    return task


def load_local_machine_config(path: Path) -> m.LocalMachineConfig:
    """加载并校验仅包含本机路径的配置。"""
    data = _read_json(path, "local config")
    return m.LocalMachineConfig(**data)
