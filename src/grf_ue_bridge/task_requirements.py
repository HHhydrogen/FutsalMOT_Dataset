"""从 task 或 resolved task 确定验证所需功能。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Set


@dataclass(frozen=True)
class TaskRequirements:
    """任务功能开关的只读快照。"""

    requires_render: bool = False
    requires_instance_mask: bool = False
    requires_mot: bool = False
    requires_yolo_det: bool = False
    requires_yolo_seg: bool = False
    requires_pose: bool = False


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    return {}


def _source_mapping(source: Any) -> Mapping[str, Any]:
    if isinstance(source, Mapping):
        return source
    model_dump = getattr(source, "model_dump", None)
    if callable(model_dump):
        try:
            dumped = model_dump()
        except Exception:
            return {}
        return _as_mapping(dumped)
    return {}


def _enabled(section: Any) -> bool:
    return bool(_as_mapping(section).get("enabled", False))


def _formats(postprocess: Mapping[str, Any]) -> Set[str]:
    raw_formats = postprocess.get("formats", [])
    if isinstance(raw_formats, str):
        values: Iterable[Any] = raw_formats.split(",")
    elif isinstance(raw_formats, (list, tuple, set, frozenset)):
        values = raw_formats
    else:
        values = ()
    return {str(value).strip() for value in values if str(value).strip()}


def _annotation_export(source: Mapping[str, Any]) -> Dict[str, Any]:
    """合并 task 和 resolved task 的 annotation_export，resolved 值优先。"""
    annotation_export: Dict[str, Any] = {}
    for ue_key in ("ue", "ue_profile"):
        ue = _as_mapping(source.get(ue_key))
        annotation = _as_mapping(ue.get("annotation_export"))
        annotation_export.update(annotation)
    return annotation_export


def resolve_task_requirements(source: Any) -> TaskRequirements:
    """从配置对象或字典解析 requirements，不读取任何 episode 产物。"""
    data = _source_mapping(source)
    annotation_export = _annotation_export(data)
    postprocess = _as_mapping(data.get("postprocess"))
    formats = _formats(postprocess)

    render_block = annotation_export.get("render_rgb")
    requires_render = False if render_block is None else _enabled(render_block)
    requires_instance_mask = _enabled(annotation_export.get("instance_mask"))

    formats_explicit = "formats" in postprocess
    export_mot = bool(annotation_export.get("export_mot", False))
    requires_mot = "mot" in formats or (not formats_explicit and export_mot)

    return TaskRequirements(
        requires_render=requires_render,
        requires_instance_mask=requires_instance_mask,
        requires_mot=requires_mot,
        requires_yolo_det="yolo-det" in formats,
        requires_yolo_seg="yolo-seg" in formats,
        requires_pose=_enabled(postprocess.get("yolo_pose")),
    )
