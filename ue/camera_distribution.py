"""固定锚点相机的实际状态归一化和 JSONL 写入。"""

import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Dict, Iterable


_CAMERA_ORDER = {"C1": 0, "C2": 1, "C3": 2, "C4": 3, "C5": 4}
_REQUIRED_ACTUAL_FIELDS = ("resolution", "position_m", "rotation_deg")
_SCHEMA = "futsalmot_camera_state"
_VERSION = 1
_ANCHOR_IDS = ("C1", "C2", "C3", "C4", "C5")
_PARTIAL_CAMERA_IDS = {"P01"}
_EPISODE_CAMERA_IDS = {
    "anchor_only": ["C1", "C2", "C3", "C4", "C5"],
    "anchor_plus_one_partial": ["C1", "C2", "C3", "C4", "C5", "P01"],
}
LEFT_HALF_SIDELINE_HIGH_V1 = "left_half_sideline_high_v1"
_LEFT_HALF_TEMPLATE = {
    "position_m": [-10.0, -25.0, 16.0],
    "rotation_deg": [-32.47119229084849, 90.0, 0.0],
    "height_m": 16.0,
    "focal_length_mm": 12.0,
    "resolution": [1920, 1080],
}


def _is_left_half_profile(profile: Dict[str, Any]) -> bool:
    return (
        profile.get("coverage") == "partial_field"
        and profile.get("region") == "left_half"
        and profile.get("placement_template") == LEFT_HALF_SIDELINE_HIGH_V1
    )


def _validate_p01_profile(profile: Dict[str, Any]) -> None:
    if profile.get("coverage") != "partial_field":
        raise ValueError("P01 必须使用 coverage=partial_field")
    if profile.get("region") != "left_half":
        raise ValueError("P01 必须使用 region=left_half")
    if profile.get("placement_template") != LEFT_HALF_SIDELINE_HIGH_V1:
        raise ValueError(
            "P01 必须使用 placement_template=left_half_sideline_high_v1"
        )


def resolve_partial_static_camera_profile(
    camera_id: str,
    intent: Dict[str, Any],
) -> Dict[str, Any]:
    """把当前唯一 partial intent 解析为固定、可复现的 camera profile。"""
    if camera_id not in _PARTIAL_CAMERA_IDS:
        raise ValueError("partial static camera 当前只允许 P01")
    if intent.get("type") != "static_surveillance":
        raise ValueError("partial static camera 的 type 必须为 static_surveillance")
    if intent.get("coverage") != "partial_field" or intent.get("region") != "left_half":
        raise ValueError("P01 当前只支持 partial_field + left_half")
    profile = dict(intent)
    profile.update({
        "position_m": list(_LEFT_HALF_TEMPLATE["position_m"]),
        "rotation_deg": list(_LEFT_HALF_TEMPLATE["rotation_deg"]),
        "height_m": _LEFT_HALF_TEMPLATE["height_m"],
        "resolution": list(_LEFT_HALF_TEMPLATE["resolution"]),
        "lens": {"focal_length_mm": _LEFT_HALF_TEMPLATE["focal_length_mm"]},
    })
    profile["placement_template"] = LEFT_HALF_SIDELINE_HIGH_V1
    profile["distortion"] = None
    validate_partial_camera_coverage(profile)
    return profile


def validate_partial_camera_coverage(profile: Dict[str, Any]) -> bool:
    """验证当前 partial template 的 transform 和水平 FOV 覆盖 left_half。"""
    if not _is_left_half_profile(profile):
        raise ValueError("partial camera placement template 不匹配")
    for field in ("position_m", "rotation_deg", "height_m", "resolution"):
        expected = _LEFT_HALF_TEMPLATE[field]
        actual = profile.get(field)
        if isinstance(expected, list):
            if list(actual or []) != expected:
                raise ValueError("partial camera placement template transform 不匹配")
        elif actual != expected:
            raise ValueError("partial camera placement template 参数不匹配")
    import math
    from camera_projection import focal_length_to_fov_deg

    focal = float((profile.get("lens") or {}).get("focal_length_mm"))
    if focal != _LEFT_HALF_TEMPLATE["focal_length_mm"]:
        raise ValueError("partial camera placement template lens 参数不匹配")
    horizontal_fov = math.radians(focal_length_to_fov_deg(focal, 23.76))
    position_x, position_y, _ = profile["position_m"]
    target_points = [(-20.0, -10.0), (-20.0, 10.0), (0.0, -10.0), (0.0, 10.0)]
    yaw = math.radians(float(profile["rotation_deg"][1]))
    forward_x, forward_y = math.cos(yaw), math.sin(yaw)
    for target_x, target_y in target_points:
        dx = target_x - position_x
        dy = target_y - position_y
        angle = abs(math.atan2(forward_x * dy - forward_y * dx, forward_x * dx + forward_y * dy))
        if angle > horizontal_fov / 2.0:
            raise ValueError("partial camera FOV 未覆盖 left_half 目标区域")
    return True


def resolve_episode_camera_ids(resolved_task: Dict[str, Any]) -> list[str]:
    """按已验证的 episode profile 确定性选择相机 ID。"""
    simulation = resolved_task.get("simulation") or {}
    camera = simulation.get("camera") or {}
    distribution = camera.get("distribution") or {}
    if distribution.get("anchors") != list(_ANCHOR_IDS):
        raise ValueError("simulation.camera.distribution.anchors 必须完整包含 C1..C5")
    episode_profile = distribution.get("episode_profile", "anchor_only")
    if episode_profile == "anchor_plus_two_partial":
        raise ValueError("anchor_plus_two_partial 当前需要第二个已实现 Partial Camera，例如 P02")
    try:
        selected_ids = _EPISODE_CAMERA_IDS[episode_profile]
    except KeyError:
        raise ValueError("不支持的 episode_profile: {}".format(episode_profile))

    profiles = camera.get("profiles") or {}
    mapping = (resolved_task.get("ue_profile") or {}).get("camera_mapping") or {}
    missing_profiles = [camera_id for camera_id in selected_ids if camera_id not in profiles]
    if missing_profiles:
        raise ValueError("selected camera profile 缺失: {}".format(", ".join(missing_profiles)))
    if "P01" in selected_ids:
        _validate_p01_profile(profiles["P01"])
    missing_mappings = [camera_id for camera_id in selected_ids if camera_id not in mapping]
    if missing_mappings:
        raise ValueError("selected camera mapping 缺失: {}".format(", ".join(missing_mappings)))
    return list(selected_ids)


def _require_non_empty(row: Dict[str, Any], field: str) -> Any:
    value = row.get(field)
    if value is None or value == "":
        raise ValueError("camera state 缺少必需字段: {}".format(field))
    return value


def _require_vector(row: Dict[str, Any], field: str, length: int) -> list:
    value = _require_non_empty(row, field)
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError("camera state 字段 {} 必须为 {} 维 transform".format(field, length))
    return list(value)


def _require_resolution(value: Any) -> list:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence) or len(value) != 2:
        raise ValueError("camera state 字段 resolution 必须为两个有限正整数")
    if any(
        isinstance(item, bool)
        or not isinstance(item, int)
        or item <= 0
        for item in value
    ):
        raise ValueError("camera state 字段 resolution 必须为两个有限正整数")
    return list(value)


def _require_non_negative_int(row: Dict[str, Any], field: str) -> None:
    value = _require_non_empty(row, field)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError("camera state 字段 {} 必须为非负整数".format(field))


def _require_non_negative_finite(row: Dict[str, Any], field: str) -> None:
    value = _require_non_empty(row, field)
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < 0:
        raise ValueError("camera state 字段 {} 必须为有限非负数".format(field))


def _require_finite_number(value: Any, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("camera state 字段 {} 必须为有限数值".format(field))


def _require_finite_vector(row: Dict[str, Any], field: str) -> None:
    values = _require_vector(row, field, 3)
    for value in values:
        _require_finite_number(value, field)


def _validate_numeric_state(state: Dict[str, Any]) -> None:
    _require_finite_vector(state, "position_m")
    _require_finite_vector(state, "rotation_deg")
    _require_finite_number(state.get("height_m"), "height_m")
    for field in ("focal_length_mm", "horizontal_fov_deg"):
        if state.get(field) is not None:
            _require_finite_number(state[field], field)


def _compare_value(actual: Any, requested: Any, field: str) -> None:
    if isinstance(requested, (list, tuple)):
        if list(actual) != list(requested):
            raise ValueError("camera profile/state 不匹配: {}".format(field))
    elif actual != requested:
        raise ValueError("camera profile/state 不匹配: {}".format(field))


def _compare_profile(profile: Dict[str, Any], actual: Dict[str, Any]) -> None:
    requested_lens = profile.get("lens", profile)
    for field in ("resolution", "position_m", "rotation_deg", "height_m"):
        if field in profile:
            _compare_value(actual.get(field), profile[field], field)
    for field in ("focal_length_mm", "horizontal_fov_deg"):
        if field in requested_lens and requested_lens[field] is not None:
            _compare_value(actual.get(field), requested_lens[field], field)


def _validate_state_row(row: Dict[str, Any]) -> None:
    if row.get("schema") != _SCHEMA:
        raise ValueError("camera state 字段 schema 必须为 {}".format(_SCHEMA))
    if row.get("version") != _VERSION:
        raise ValueError("camera state 字段 version 必须为 {}".format(_VERSION))
    camera_id = _require_non_empty(row, "camera_id")
    _require_non_empty(row, "ue_actor")
    _require_non_empty(row, "sequence")
    _require_non_empty(row, "type")
    _require_non_empty(row, "coverage")
    _require_non_empty(row, "height_m")
    _require_non_negative_int(row, "frame")
    _require_non_negative_int(row, "source_step")
    _require_non_negative_finite(row, "time_seconds")
    if camera_id not in _CAMERA_ORDER:
        raise ValueError("camera state 的 camera_id 必须为 C1..C5: {}".format(camera_id))
    _require_resolution(row.get("resolution"))
    _require_finite_vector(row, "position_m")
    _require_finite_vector(row, "rotation_deg")
    _require_finite_number(row["height_m"], "height_m")
    for field in ("focal_length_mm", "horizontal_fov_deg"):
        if row.get(field) is not None:
            _require_finite_number(row[field], field)
    if row.get("focal_length_mm") is None and row.get("horizontal_fov_deg") is None:
        raise ValueError("camera state 缺少必需字段: focal_length_mm 或 horizontal_fov_deg")


def normalize_camera_state(
    camera_id: str,
    ue_actor: str,
    sequence: str,
    profile: Dict[str, Any],
    actual: Dict[str, Any],
    frame: int,
    source_step: int,
    time_seconds: float,
) -> Dict[str, Any]:
    """将 UE 读回的实际状态整理为一行 camera_state.jsonl。"""
    for field in ("position_m", "rotation_deg"):
        _require_non_empty(actual, field)
    resolution = _require_resolution(actual.get("resolution"))
    _validate_numeric_state(actual)
    row = {
        "schema": _SCHEMA,
        "version": _VERSION,
        "frame": frame,
        "source_step": source_step,
        "time_seconds": time_seconds,
        "camera_id": camera_id,
        "ue_actor": ue_actor,
        "sequence": sequence,
        "type": profile.get("type"),
        "coverage": profile.get("coverage"),
        "resolution": resolution,
        "focal_length_mm": actual.get("focal_length_mm"),
        "horizontal_fov_deg": actual.get("horizontal_fov_deg"),
        "position_m": list(actual["position_m"]),
        "rotation_deg": list(actual["rotation_deg"]),
        "height_m": actual.get("height_m"),
        "distortion": actual.get("distortion"),
    }
    _compare_profile(profile, actual)
    _validate_state_row(row)
    return row


def write_camera_state_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    """按帧和 C1..C5 顺序写入实际相机状态。"""
    ordered_rows = list(rows)
    frame_cameras = set()
    for row in ordered_rows:
        _validate_state_row(row)
        key = (row["frame"], row["camera_id"])
        if key in frame_cameras:
            raise ValueError("camera state 存在 duplicate (frame, camera_id): {}".format(key))
        frame_cameras.add(key)
    frames = {row["frame"] for row in ordered_rows}
    for frame in frames:
        cameras = {row["camera_id"] for row in ordered_rows if row["frame"] == frame}
        if cameras != set(_CAMERA_ORDER):
            raise ValueError("camera state 每个 frame 必须完整包含 C1..C5: {}".format(frame))
    ordered_rows.sort(key=lambda row: (row["frame"], _CAMERA_ORDER[row["camera_id"]]))
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in ordered_rows:
            handle.write(json.dumps(row, ensure_ascii=True, separators=(",", ":")))
            handle.write("\n")


def resolve_anchor_sequence_entries(simulation: Dict[str, Any], mapping: Dict[str, Any]) -> list:
    """将 resolved camera contract 转换为现有 Sequence 列表契约。"""
    camera = (simulation or {}).get("camera") or {}
    distribution = camera.get("distribution") or {}
    anchors = distribution.get("anchors")
    profiles = camera.get("profiles") or {}
    if anchors != list(_ANCHOR_IDS):
        raise ValueError("simulation.camera.distribution.anchors 必须完整包含 C1..C5")
    if not set(_ANCHOR_IDS).issubset(set(profiles)):
        raise ValueError("simulation.camera.profiles 必须完整包含 C1..C5")
    mapping_ids = set(mapping or {})
    if not set(_ANCHOR_IDS).issubset(mapping_ids) or not mapping_ids.issubset(set(_ANCHOR_IDS) | {"P01"}):
        raise ValueError("ue.camera_mapping 必须包含 C1..C5，且只允许额外 P01")
    if "P01" in mapping_ids and "P01" not in profiles:
        raise ValueError("ue.camera_mapping.P01 只有在 P01 profile 存在时才允许")
    entries = []
    for camera_id in _ANCHOR_IDS:
        entry = mapping[camera_id]
        actor = entry.get("actor") if isinstance(entry, dict) else None
        sequence = entry.get("sequence") if isinstance(entry, dict) else None
        if not isinstance(actor, str) or not actor.strip():
            raise ValueError(f"Canonical ID {camera_id} 的 actor mapping 不能为空")
        if not isinstance(sequence, str) or not sequence.strip():
            raise ValueError(f"Canonical ID {camera_id} 的 sequence mapping 不能为空")
        entries.append({
            "camera_id": camera_id,
            "camera_actor": actor.strip(),
            "name": sequence.strip(),
            "profile": profiles[camera_id],
        })
    actors = [entry["camera_actor"] for entry in entries]
    if len(set(actors)) != len(actors):
        raise ValueError("C1..C5 不允许复用同一个 UE Actor")
    return entries


def _resolve_canonical_sequence_entries(
    simulation: Dict[str, Any],
    mapping: Dict[str, Any],
    selected_ids: Iterable[str],
) -> list:
    profiles = ((simulation or {}).get("camera") or {}).get("profiles") or {}
    entries = []
    actors = []
    for camera_id in selected_ids:
        entry = mapping[camera_id]
        actor = entry.get("actor") if isinstance(entry, dict) else None
        sequence = entry.get("sequence") if isinstance(entry, dict) else None
        if not isinstance(actor, str) or not actor.strip():
            raise ValueError(f"Canonical ID {camera_id} 的 actor mapping 不能为空")
        if not isinstance(sequence, str) or not sequence.strip():
            raise ValueError(f"Canonical ID {camera_id} 的 sequence mapping 不能为空")
        entries.append({
            "camera_id": camera_id,
            "camera_actor": actor.strip(),
            "name": sequence.strip(),
            "profile": profiles[camera_id],
        })
        actors.append(actor.strip())
    if len(set(actors)) != len(actors):
        raise ValueError("selected cameras 不允许复用同一个 UE Actor")
    return entries


def resolve_mapped_camera_entries(
    simulation: Dict[str, Any],
    mapping: Dict[str, Any],
    camera_ids: Iterable[str],
) -> list:
    """按显式 camera ID 子集解析已映射的 runtime camera entries。"""
    profiles = ((simulation or {}).get("camera") or {}).get("profiles") or {}
    selected_ids = list(camera_ids)
    if selected_ids != ["P01"]:
        raise ValueError("当前非 canonical runtime selection 只支持 P01")
    if "P01" not in profiles or "P01" not in (mapping or {}):
        raise ValueError("P01 profile 和 ue.camera_mapping.P01 必须同时存在")
    _validate_p01_profile(profiles["P01"])
    entry = mapping["P01"]
    actor = entry.get("actor") if isinstance(entry, dict) else None
    sequence = entry.get("sequence") if isinstance(entry, dict) else None
    if not isinstance(actor, str) or not actor.strip() or not isinstance(sequence, str) or not sequence.strip():
        raise ValueError("Canonical ID P01 的 actor/sequence mapping 不能为空")
    return [{
        "camera_id": "P01",
        "camera_actor": actor.strip(),
        "name": sequence.strip(),
        "profile": profiles["P01"],
    }]


def resolve_runtime_camera_selection(
    resolved_task: Dict[str, Any],
    *,
    legacy_sequences: list,
    legacy_cameras: list,
    camera_ids: Iterable[str] = None,
) -> Dict[str, Any]:
    """解析 UE 运行时相机选择，canonical contract 优先，旧字段作 fallback。"""
    simulation = resolved_task.get("simulation") or {}
    ue_profile = resolved_task.get("ue_profile") or {}
    mapping = ue_profile.get("camera_mapping")
    if simulation and mapping is not None:
        if camera_ids is not None:
            entries = resolve_mapped_camera_entries(simulation, mapping, camera_ids)
        else:
            selected_ids = resolve_episode_camera_ids(resolved_task)
            if selected_ids == list(_ANCHOR_IDS):
                entries = resolve_anchor_sequence_entries(simulation, mapping)
            else:
                entries = _resolve_canonical_sequence_entries(simulation, mapping, selected_ids)
        resolutions = {tuple(entry["profile"].get("resolution", (1920, 1080))) for entry in entries}
        if len(resolutions) != 1:
            formatted = ", ".join("{}x{}".format(*resolution) for resolution in sorted(resolutions))
            raise ValueError("canonical camera resolution 必须统一以供当前 MRQ 配置使用: {}".format(formatted))
        resolution = list(next(iter(resolutions)))
        sequences = [
            {
                "name": entry["name"],
                "camera_actor": entry["camera_actor"],
                "camera_id": entry["camera_id"],
            }
            for entry in entries
        ]
        return {
            "entries": entries,
            "sequences": sequences,
            "cameras": [entry["camera_actor"] for entry in entries],
            "resolution": resolution,
            "canonical": True,
        }
    return {
        "entries": [],
        "sequences": list(legacy_sequences or []),
        "cameras": list(legacy_cameras or []),
        "resolution": None,
        "canonical": False,
    }
