"""MPFB Player_N 视觉 Profile 的纯函数和 UE 适配入口。

本模块的 Profile 规范化和 provenance 函数可在 P1 Python 进程中运行。
只有真正访问 Unreal Editor 时才允许导入 ``unreal``。
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence


FORMAL_LEVEL = "/Game/FutsalMOT/Maps/L_FutsalCourt"
DRIVER_VISIBILITY_TICK_OPTION = "ALWAYS_TICK_POSE_AND_REFRESH_BONES"
DEFAULT_PLAYER_ENTITY_IDS = (
    "L0",
    "L1",
    "L2",
    "L3",
    "L4",
    "R0",
    "R1",
    "R2",
    "R3",
    "R4",
)


def required_driver_visibility_tick_option() -> str:
    """返回 Runtime Retarget 隐藏驱动网格所需的骨骼刷新策略。"""
    return DRIVER_VISIBILITY_TICK_OPTION


def json_default(value: Any) -> Any:
    """把 Unreal Python 的 Array、Text、Name 等值转换为 JSON 基础类型。"""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    try:
        return list(value)
    except TypeError:
        return str(value)

_PROFILE_FIELDS = ("mesh", "anim_class", "skeleton")
_SNAPSHOT_FIELDS = (
    "entity_id",
    "actor_path",
    "label",
    "tags",
    "transform",
    "component_name",
    "old_mesh",
    "old_anim_class",
    "old_skeleton",
    "visual_component",
    "visual_parent",
    "visual_mesh",
    "visual_anim_class",
    "visual_transform",
)


def is_generated_class_path(path: Any) -> bool:
    """判断 UE 对象路径是否已经指向 GeneratedClass。"""

    return isinstance(path, str) and "." in path and path.rsplit(".", 1)[1].endswith("_C")


def _require_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} 必须是非空字符串")
    value = value.strip()
    if not value.startswith("/Game/"):
        raise ValueError(f"{field} 必须是 /Game/ 资产路径: {value}")
    return value


def normalize_player_visuals(
    player_visuals: Optional[Mapping[str, Any]],
    *,
    entity_ids: Sequence[str] = DEFAULT_PLAYER_ENTITY_IDS,
) -> Optional[Dict[str, Any]]:
    """规范化角色视觉配置，并为未显式映射的实体填充默认 Profile。"""

    if not player_visuals:
        return None
    if not isinstance(player_visuals, Mapping):
        raise ValueError("player_visuals 必须是对象")

    profiles_raw = player_visuals.get("profiles")
    if not isinstance(profiles_raw, Mapping):
        raise ValueError("player_visuals.profiles 必须是对象")

    profiles: Dict[str, Dict[str, str]] = {}
    for profile_id_raw, profile_raw in profiles_raw.items():
        if not isinstance(profile_id_raw, str) or not profile_id_raw.strip():
            raise ValueError("profile ID 必须是非空字符串")
        profile_id = profile_id_raw.strip()
        if not isinstance(profile_raw, Mapping):
            raise ValueError(f"profile {profile_id} 必须是对象")
        profile = {
            field: _require_text(profile_raw.get(field), f"profile {profile_id}.{field}")
            for field in _PROFILE_FIELDS
        }
        profile["visual_component"] = str(
            profile_raw.get("visual_component", "MPFBVisual")
        )
        profile["visual_parent"] = str(
            profile_raw.get("visual_parent", "CharacterMesh0")
        )
        profile["visual_relative_location_cm"] = list(
            profile_raw.get("visual_relative_location_cm", [0.0, 0.0, 0.0])
        )
        profile["visual_relative_rotation_deg"] = list(
            profile_raw.get("visual_relative_rotation_deg", [0.0, 0.0, 0.0])
        )
        profile["visual_relative_scale"] = list(
            profile_raw.get("visual_relative_scale", [1.0, 1.0, 1.0])
        )
        for field, expected in (
            ("visual_relative_location_cm", 3),
            ("visual_relative_rotation_deg", 3),
            ("visual_relative_scale", 3),
        ):
            if len(profile[field]) != expected:
                raise ValueError(f"profile {profile_id}.{field} 必须有 {expected} 个数值")
        profiles[profile_id] = profile

    default_profile = player_visuals.get("default_profile")
    if not isinstance(default_profile, str) or not default_profile.strip():
        raise ValueError("default_profile 必须是非空字符串")
    default_profile = default_profile.strip()
    if default_profile not in profiles:
        raise ValueError(f"不存在的 profile: {default_profile}")

    entity_profiles_raw = player_visuals.get("entity_profiles") or {}
    if not isinstance(entity_profiles_raw, Mapping):
        raise ValueError("entity_profiles 必须是对象")

    normalized_entity_profiles: Dict[str, str] = {}
    allowed_entities = tuple(entity_ids)
    for entity_id in allowed_entities:
        profile_id = entity_profiles_raw.get(entity_id, default_profile)
        if not isinstance(profile_id, str) or not profile_id.strip():
            raise ValueError(f"entity_profiles[{entity_id}] 必须是非空字符串")
        profile_id = profile_id.strip()
        if profile_id not in profiles:
            raise ValueError(f"实体 {entity_id} 引用了不存在的 profile: {profile_id}")
        normalized_entity_profiles[entity_id] = profile_id

    unknown_entities = set(entity_profiles_raw) - set(allowed_entities)
    if unknown_entities:
        names = ", ".join(sorted(str(item) for item in unknown_entities))
        raise ValueError(f"entity_profiles 含未知实体: {names}")

    return {
        "default_profile": default_profile,
        "profiles": profiles,
        "entity_profiles": normalized_entity_profiles,
    }


def player_visuals_fingerprint(normalized: Optional[Mapping[str, Any]]) -> Optional[str]:
    """对规范化 Profile 计算不包含机器路径的稳定 SHA-256 指纹。"""

    if normalized is None:
        return None
    canonical = json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def sanitized_player_visuals_provenance(
    normalized: Optional[Mapping[str, Any]],
) -> Optional[Dict[str, Any]]:
    """生成可写入 episode provenance 的角色 Profile 摘要。"""

    if normalized is None:
        return None
    result = copy.deepcopy(dict(normalized))
    result["fingerprint"] = player_visuals_fingerprint(normalized)
    return result


def write_player_visuals_provenance(
    episode_dir: Any,
    normalized: Optional[Mapping[str, Any]],
) -> Optional[Path]:
    """写入不包含本机路径的 episode 角色 Profile provenance。"""

    provenance = sanitized_player_visuals_provenance(normalized)
    if provenance is None:
        return None
    target = Path(episode_dir) / "provenance" / "player-visuals.json"
    _write_json(str(target), provenance)
    return target


def sanitize_snapshot(snapshot: Mapping[str, Any]) -> Dict[str, Any]:
    """限制 rollback 快照字段，避免恢复范围扩散到用户状态。"""

    return {
        field: copy.deepcopy(snapshot[field])
        for field in _SNAPSHOT_FIELDS
        if field in snapshot
    }


def build_restore_payload(snapshot: Mapping[str, Any]) -> Dict[str, Any]:
    """从快照生成仅包含 Mesh/AnimBP 的恢复 payload。"""

    required = ("actor_path", "component_name", "old_mesh", "old_anim_class")
    missing = [field for field in required if field not in snapshot]
    if missing:
        raise ValueError(f"rollback 快照缺少字段: {', '.join(missing)}")
    payload = {
        "actor_path": snapshot["actor_path"],
        "component_name": snapshot["component_name"],
        "skeletal_mesh": snapshot["old_mesh"],
        "anim_class": snapshot["old_anim_class"],
    }
    if "visual_component" in snapshot:
        payload["visual_component"] = snapshot["visual_component"]
    return payload


def select_rollback_payload(
    existing: Optional[Mapping[str, Any]],
    proposed: Mapping[str, Any],
) -> tuple:
    """保留有效的首次快照，避免重复 apply 丢失默认角色恢复点。"""

    if existing is None:
        return dict(proposed), False
    required = {"schema", "version", "level", "actors"}
    if not required.issubset(existing):
        raise ValueError("rollback 快照无效: 缺少顶层字段")
    if existing.get("schema") != "futsalmot_player_visuals_rollback" or existing.get("version") != 1:
        raise ValueError("rollback 快照无效: schema/version 不匹配")
    actors = existing.get("actors")
    if not isinstance(actors, list) or not actors:
        raise ValueError("rollback 快照无效: actors 必须是非空数组")
    actor_ids = set()
    for actor in actors:
        if not isinstance(actor, Mapping):
            raise ValueError("rollback 快照无效: actor 项必须是对象")
        if not actor.get("entity_id") or not actor.get("actor_path"):
            raise ValueError("rollback 快照无效: actor identity 缺失")
        if not actor.get("old_mesh") or not actor.get("old_anim_class"):
            raise ValueError("rollback 快照无效: 原 Mesh/AnimBP 缺失")
        actor_ids.add(actor["entity_id"])
    proposed_ids = {
        actor.get("entity_id")
        for actor in proposed.get("actors", [])
        if isinstance(actor, Mapping)
    }
    if existing.get("level") != proposed.get("level") or actor_ids != proposed_ids:
        raise ValueError("rollback 快照无效: 地图或 Actor 集合不匹配")
    return dict(existing), True


def _json_vector(value: Any) -> list:
    return [float(value.x), float(value.y), float(value.z)]


def _json_rotator(value: Any) -> list:
    return [float(value.pitch), float(value.yaw), float(value.roll)]


def _asset_path(value: Any) -> Optional[str]:
    if value is None:
        return None
    getter = getattr(value, "get_path_name", None)
    return getter() if getter is not None else str(value)


def _package_path(value: Any) -> Optional[str]:
    path = _asset_path(value)
    return path.split(".", 1)[0] if path else None


def _current_level_path(unreal_module: Any) -> Optional[str]:
    world = unreal_module.EditorLevelLibrary.get_editor_world()
    if world is None:
        return None
    return world.get_path_name().split(".", 1)[0]


def _find_player_actors(unreal_module: Any, mapping: Mapping[str, str]) -> tuple:
    editor = unreal_module.get_editor_subsystem(unreal_module.EditorActorSubsystem)
    by_label: Dict[str, list] = {}
    for actor in editor.get_all_level_actors():
        try:
            label = actor.get_actor_label()
        except Exception:
            continue
        by_label.setdefault(label.lower(), []).append(actor)

    found = {}
    errors = []
    for entity_id in DEFAULT_PLAYER_ENTITY_IDS:
        label = mapping.get(entity_id)
        if not isinstance(label, str) or not label.strip():
            errors.append(f"mapping 缺少实体 {entity_id}")
            continue
        matches = by_label.get(label.lower(), [])
        if len(matches) != 1:
            errors.append(f"标签 {label} 应唯一匹配 1 个 Actor，实际 {len(matches)} 个")
            continue
        found[entity_id] = matches[0]
    return found, errors


def _profile_for_entity(normalized: Mapping[str, Any], entity_id: str) -> Mapping[str, str]:
    profile_id = normalized["entity_profiles"][entity_id]
    return normalized["profiles"][profile_id]


def _load_profile_assets(unreal_module: Any, profile: Mapping[str, str]) -> tuple:
    mesh = unreal_module.load_asset(profile["mesh"])
    anim_blueprint = unreal_module.load_asset(profile["anim_class"])
    skeleton = unreal_module.load_asset(profile["skeleton"])
    if mesh is None:
        raise RuntimeError(f"Mesh 资产加载失败: {profile['mesh']}")
    if anim_blueprint is None:
        raise RuntimeError(f"AnimBP 资产加载失败: {profile['anim_class']}")
    if skeleton is None:
        raise RuntimeError(f"Skeleton 资产加载失败: {profile['skeleton']}")
    generated_class = unreal_module.BlueprintEditorLibrary.generated_class(anim_blueprint)
    if generated_class is None:
        raise RuntimeError(f"AnimBP GeneratedClass 加载失败: {profile['anim_class']}")
    actual_skeleton = getattr(mesh, "skeleton", None)
    actual_skeleton_path = _package_path(actual_skeleton)
    if actual_skeleton_path != profile["skeleton"]:
        raise RuntimeError(
            f"Mesh Skeleton 不匹配: expected={profile['skeleton']} actual={actual_skeleton_path}"
        )
    anim_skeleton = anim_blueprint.get_editor_property("target_skeleton")
    anim_skeleton_path = _package_path(anim_skeleton)
    if anim_skeleton_path != profile["skeleton"]:
        raise RuntimeError(
            f"AnimBP Target Skeleton 不匹配: expected={profile['skeleton']} "
            f"actual={anim_skeleton_path}"
        )
    return mesh, generated_class, skeleton


def _cached_profile_assets(unreal_module: Any, cache: Dict[str, tuple], profile_id: str, profile: Mapping[str, str]) -> tuple:
    if profile_id not in cache:
        cache[profile_id] = _load_profile_assets(unreal_module, profile)
    return cache[profile_id]


def _snapshot_actor(unreal_module: Any, entity_id: str, actor: Any) -> Dict[str, Any]:
    components = actor.get_components_by_class(unreal_module.SkeletalMeshComponent)
    named = [c for c in components if c.get_name() == "CharacterMesh0"]
    if len(named) != 1:
        raise RuntimeError(
            f"{actor.get_actor_label()} 的 CharacterMesh0 数量必须为 1，实际 {len(named)}"
        )
    component = named[0]
    mesh = component.get_skeletal_mesh_asset()
    anim_class = component.get_editor_property("anim_class")
    skeleton = getattr(mesh, "skeleton", None) if mesh is not None else None
    location = actor.get_actor_location()
    rotation = actor.get_actor_rotation()
    scale = actor.get_actor_scale3d()
    rel_location = component.get_editor_property("relative_location")
    rel_rotation = component.get_editor_property("relative_rotation")
    rel_scale = component.get_editor_property("relative_scale3d")
    visual_name = "MPFBVisual"
    visual_components = [c for c in components if c.get_name() == visual_name]
    visual_snapshot = None
    if len(visual_components) == 1:
        visual = visual_components[0]
        visual_mesh = visual.get_skeletal_mesh_asset()
        visual_anim_class = visual.get_editor_property("anim_class")
        visual_parent = visual.get_attach_parent()
        visual_snapshot = {
            "name": visual.get_name(),
            "parent": visual_parent.get_name() if visual_parent else None,
            "mesh": _asset_path(visual_mesh),
            "anim_class": _asset_path(visual_anim_class),
            "transform": {
                "location": _json_vector(visual.get_editor_property("relative_location")),
                "rotation": _json_rotator(visual.get_editor_property("relative_rotation")),
                "scale": _json_vector(visual.get_editor_property("relative_scale3d")),
            },
        }
    return sanitize_snapshot(
        {
            "entity_id": entity_id,
            "actor_path": actor.get_path_name(),
            "label": actor.get_actor_label(),
            "tags": [str(tag) for tag in actor.get_editor_property("tags") or []],
            "transform": {
                "location": _json_vector(location),
                "rotation": _json_rotator(rotation),
                "scale": _json_vector(scale),
                "mesh_relative_location": _json_vector(rel_location),
                "mesh_relative_rotation": _json_rotator(rel_rotation),
                "mesh_relative_scale": _json_vector(rel_scale),
            },
            "component_name": component.get_name(),
            "old_mesh": _asset_path(mesh),
            "old_anim_class": _asset_path(anim_class),
            "old_skeleton": _asset_path(skeleton),
            "visual_component": visual_snapshot,
            "visual_parent": visual_snapshot.get("parent") if visual_snapshot else None,
            "visual_mesh": visual_snapshot.get("mesh") if visual_snapshot else None,
            "visual_anim_class": visual_snapshot.get("anim_class") if visual_snapshot else None,
            "visual_transform": visual_snapshot.get("transform") if visual_snapshot else None,
        }
    )


def _actor_detail(unreal_module: Any, entity_id: str, actor: Any) -> Dict[str, Any]:
    components = actor.get_components_by_class(unreal_module.SkeletalMeshComponent)
    named = [c for c in components if c.get_name() == "CharacterMesh0"]
    result = {
        "entity_id": entity_id,
        "actor_path": actor.get_path_name(),
        "label": actor.get_actor_label(),
        "tags": [str(tag) for tag in actor.get_editor_property("tags") or []],
        "skeletal_component_count": len(components),
        "character_mesh_count": len(named),
    }
    if len(named) == 1:
        component = named[0]
        mesh = component.get_skeletal_mesh_asset()
        result.update(
            {
                "mesh": _asset_path(mesh),
                "skeleton": _package_path(getattr(mesh, "skeleton", None)),
                "anim_class": _asset_path(component.get_editor_property("anim_class")),
                "component_name": component.get_name(),
            }
        )
    try:
        capsule = actor.get_component_by_class(unreal_module.CapsuleComponent)
        result["capsule"] = {
            "radius": float(capsule.get_editor_property("capsule_radius")),
            "half_height": float(capsule.get_editor_property("capsule_half_height")),
        }
    except Exception as exc:
        result["capsule_error"] = f"{type(exc).__name__}: {exc}"
    return result


def _ensure_visual_component(
    unreal_module: Any,
    actor: Any,
    profile: Mapping[str, Any],
):
    """获取已存在的 MPFBVisual；组件创建由 Editor MCP 完成。"""

    name = profile["visual_component"]
    components = actor.get_components_by_class(unreal_module.SkeletalMeshComponent)
    matches = [component for component in components if component.get_name() == name]
    if not matches:
        extras = [component for component in components if component.get_name() != "CharacterMesh0"]
        if len(extras) == 1:
            return extras[0]
    if len(matches) != 1:
        raise RuntimeError(
            f"{actor.get_actor_label()} 的 {name} 数量必须为 1，实际 {len(matches)}；"
            "请先通过 Editor MCP ActorTools.add_component 创建组件"
        )
    return matches[0]


def preflight_player_visuals(
    normalized: Mapping[str, Any],
    mapping: Mapping[str, str],
    *,
    expected_level: str = FORMAL_LEVEL,
    require_driver_refresh: bool = False,
) -> Dict[str, Any]:
    """在当前 UE Editor 中只读检查角色 Profile 和 Actor 契约。"""

    import unreal

    errors = []
    warnings = []
    level = _current_level_path(unreal)
    if level != expected_level:
        errors.append(f"当前地图不符合要求: expected={expected_level} actual={level}")
    actors, find_errors = _find_player_actors(unreal, mapping)
    errors.extend(find_errors)
    details = []
    loaded_assets = {}
    for entity_id in DEFAULT_PLAYER_ENTITY_IDS:
        actor = actors.get(entity_id)
        if actor is None:
            continue
        profile_id = normalized["entity_profiles"].get(entity_id)
        profile = _profile_for_entity(normalized, entity_id)
        try:
            assets = _cached_profile_assets(unreal, loaded_assets, profile_id, profile)
            mesh, generated_class, _ = assets
            detail = _actor_detail(unreal, entity_id, actor)
            expected_tag = f"Pose{entity_id}"
            if expected_tag not in detail["tags"]:
                errors.append(f"{detail['label']} 缺少 Pose tag: {expected_tag}")
            if detail["character_mesh_count"] != 1:
                errors.append(f"{detail['label']} CharacterMesh0 数量非法")
            driver = next(
                c
                for c in actor.get_components_by_class(unreal.SkeletalMeshComponent)
                if c.get_name() == "CharacterMesh0"
            )
            driver_anim = _package_path(driver.get_editor_property("anim_class"))
            driver_tick = driver.get_editor_property("visibility_based_anim_tick_option")
            driver_tick_name = getattr(driver_tick, "name", str(driver_tick))
            detail["driver_visibility_tick_option"] = driver_tick_name
            if require_driver_refresh and driver_tick_name != DRIVER_VISIBILITY_TICK_OPTION:
                errors.append(
                    f"{detail['label']} CharacterMesh0 必须刷新骨骼: "
                    f"{driver_tick_name} != {DRIVER_VISIBILITY_TICK_OPTION}"
                )
            expected_driver_anim = "/Game/FutsalMOT/Characters/FutsalPlayerBase/Animation/ABP_FutsalPlayerBase"
            if driver_anim and driver_anim != expected_driver_anim:
                errors.append(
                    f"{detail['label']} CharacterMesh0 必须保留 locomotion AnimBP: {driver_anim}"
                )
            all_visual_candidates = [
                c for c in actor.get_components_by_class(unreal.SkeletalMeshComponent)
                if c.get_name() != "CharacterMesh0"
            ]
            visuals = [
                c for c in all_visual_candidates
                if c.get_name() == profile["visual_component"]
            ]
            if not visuals and len(all_visual_candidates) == 1:
                visuals = all_visual_candidates
            detail["visual_component"] = profile["visual_component"]
            detail["visual_component_present"] = len(visuals) == 1
            if len(visuals) > 1:
                errors.append(f"{detail['label']} 存在多个 {profile['visual_component']} 组件")
            if visuals:
                visual = visuals[0]
                visual_mesh = visual.get_skeletal_mesh_asset()
                visual_anim = visual.get_editor_property("anim_class")
                parent = visual.get_attach_parent()
                detail["visual_mesh"] = _asset_path(visual_mesh)
                detail["visual_anim_class"] = _asset_path(visual_anim)
                detail["visual_parent"] = parent.get_name() if parent else None
                leader = visual.get_editor_property("leader_pose_component")
                detail["visual_leader_pose"] = leader.get_name() if leader else None
                detail["visual_relative_location"] = _json_vector(
                    visual.get_editor_property("relative_location")
                )
                if _package_path(visual_mesh) != profile["mesh"]:
                    errors.append(f"{detail['label']} MPFBVisual Mesh 不匹配")
                if _package_path(visual_anim) != profile["anim_class"]:
                    errors.append(f"{detail['label']} MPFBVisual AnimBP 不匹配")
                if detail["visual_parent"] != profile["visual_parent"]:
                    errors.append(f"{detail['label']} MPFBVisual 父组件不匹配")
                if leader is not None:
                    errors.append(
                        f"{detail['label']} MPFBVisual 不得跨 Skeleton 使用 Leader Pose"
                    )
            else:
                detail["visual_pending_creation"] = True
            detail["target_profile"] = profile_id
            detail["target_mesh"] = _asset_path(mesh)
            detail["target_skeleton"] = profile["skeleton"]
            detail["target_anim_class"] = _asset_path(generated_class)
            details.append(detail)
        except Exception as exc:
            errors.append(f"{entity_id} Profile 预检失败: {type(exc).__name__}: {exc}")
    return {
        "ok": not errors,
        "level": level,
        "actors": details,
        "errors": errors,
        "warnings": warnings,
    }


def _write_json(path: str, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            default=json_default,
        ),
        encoding="utf-8",
    )


def _restore_actor_payload(
    unreal_module: Any,
    payload: Mapping[str, Any],
    *,
    expected_label: Optional[str] = None,
) -> None:
    actor = unreal_module.load_object(None, payload["actor_path"])
    if actor is None:
        raise RuntimeError(f"Actor 加载失败: {payload['actor_path']}")
    if expected_label is not None and actor.get_actor_label() != expected_label:
        raise RuntimeError(f"Actor label 与 rollback 不一致: {payload['actor_path']}")
    components = actor.get_components_by_class(unreal_module.SkeletalMeshComponent)
    named = [c for c in components if c.get_name() == payload["component_name"]]
    if len(named) != 1:
        raise RuntimeError(f"恢复时组件数量非法: {payload['actor_path']}")
    component = named[0]
    mesh = unreal_module.load_asset(payload["skeletal_mesh"])
    if is_generated_class_path(payload["anim_class"]):
        generated_class = unreal_module.load_class(None, payload["anim_class"])
    else:
        anim_blueprint = unreal_module.load_asset(payload["anim_class"])
        generated_class = (
            unreal_module.BlueprintEditorLibrary.generated_class(anim_blueprint)
            if anim_blueprint is not None
            else None
        )
    if mesh is None or generated_class is None:
        raise RuntimeError(f"恢复资产加载失败: {payload['actor_path']}")
    component.set_skeletal_mesh_asset(mesh)
    component.set_editor_property("anim_class", generated_class)
    visual_snapshot = payload.get("visual_component")
    if visual_snapshot is None:
        for visual in list(components):
            if visual.get_name() == "MPFBVisual":
                try:
                    actor.remove_instance_component(visual)
                except Exception:
                    try:
                        visual.destroy_component()
                    except Exception as exc:
                        raise RuntimeError(f"无法移除 MPFBVisual: {exc}")


def apply_player_visuals(
    normalized: Mapping[str, Any],
    mapping: Mapping[str, str],
    *,
    rollback_path: str,
) -> Dict[str, Any]:
    """应用角色视觉 Profile，并保存十个 External Actor 包。"""

    import unreal

    preflight = preflight_player_visuals(normalized, mapping)
    if not preflight["ok"]:
        return {**preflight, "operation": "apply", "saved": [], "rollback_path": rollback_path}
    actors, find_errors = _find_player_actors(unreal, mapping)
    if find_errors:
        return {
            "ok": False,
            "operation": "apply",
            "errors": find_errors,
            "warnings": [],
            "actors": [],
            "saved": [],
            "rollback_path": rollback_path,
        }

    snapshots = []
    for entity_id in DEFAULT_PLAYER_ENTITY_IDS:
        snapshots.append(_snapshot_actor(unreal, entity_id, actors[entity_id]))
    proposed_rollback = {
        "schema": "futsalmot_player_visuals_rollback",
        "version": 1,
        "level": _current_level_path(unreal),
        "actors": snapshots,
    }
    rollback_file = Path(rollback_path)
    existing_rollback = None
    if rollback_file.exists():
        existing_rollback = json.loads(rollback_file.read_text(encoding="utf-8"))
    try:
        rollback_payload, rollback_preserved = select_rollback_payload(
            existing_rollback, proposed_rollback
        )
    except Exception as exc:
        return {
            "ok": False,
            "operation": "apply",
            "errors": [f"拒绝覆盖 rollback 快照: {type(exc).__name__}: {exc}"],
            "warnings": [],
            "actors": snapshots,
            "saved": [],
            "rollback_path": rollback_path,
        }
    if not rollback_preserved:
        _write_json(rollback_path, rollback_payload)

    changed = []
    errors = []
    try:
        loaded_assets = {}
        for entity_id in DEFAULT_PLAYER_ENTITY_IDS:
            actor = actors[entity_id]
            profile_id = normalized["entity_profiles"][entity_id]
            profile = _profile_for_entity(normalized, entity_id)
            mesh, generated_class, _ = _cached_profile_assets(
                unreal, loaded_assets, profile_id, profile
            )
            component = next(
                c for c in actor.get_components_by_class(unreal.SkeletalMeshComponent)
                if c.get_name() == "CharacterMesh0"
            )
            visual = _ensure_visual_component(unreal, actor, profile)
            parent = visual.get_attach_parent()
            if parent is not component:
                visual.attach_to_component(
                    component,
                    unreal.AttachmentRule.KEEP_RELATIVE,
                    unreal.AttachmentRule.KEEP_RELATIVE,
                    unreal.AttachmentRule.KEEP_RELATIVE,
                    True,
                )
            visual.set_skeletal_mesh_asset(mesh)
            visual.set_editor_property("anim_class", generated_class)
            visual.set_editor_property(
                "relative_location",
                unreal.Vector(*profile["visual_relative_location_cm"]),
            )
            visual.set_editor_property(
                "relative_rotation",
                unreal.Rotator(*profile["visual_relative_rotation_deg"]),
            )
            visual.set_editor_property(
                "relative_scale3d",
                unreal.Vector(*profile["visual_relative_scale"]),
            )
            visual.set_editor_property("leader_pose_component", None)
            visual.set_editor_property("visible", True)
            visual.set_editor_property("render_in_main_pass", True)
            visual.set_editor_property("render_in_depth_pass", True)
            component.set_editor_property(
                "visibility_based_anim_tick_option",
                getattr(
                    unreal.VisibilityBasedAnimTickOption,
                    DRIVER_VISIBILITY_TICK_OPTION,
                ),
            )
            changed.append(entity_id)
    except Exception as exc:
        errors.append(f"内存应用失败: {type(exc).__name__}: {exc}")
        for snapshot in snapshots:
            if snapshot.get("entity_id") in changed:
                try:
                    _restore_actor_payload(unreal, build_restore_payload(snapshot))
                except Exception as restore_exc:
                    errors.append(
                        f"内存回滚失败 {snapshot.get('entity_id')}: "
                        f"{type(restore_exc).__name__}: {restore_exc}"
                    )
        return {
            "ok": False,
            "operation": "apply",
            "errors": errors,
            "warnings": [],
            "actors": snapshots,
            "saved": [],
            "rollback_path": rollback_path,
        }

    saved = []
    for entity_id in changed:
        actor = actors[entity_id]
        try:
            result = unreal.EditorAssetLibrary.save_loaded_asset(
                actor, only_if_is_dirty=True
            )
            saved.append({"entity_id": entity_id, "saved": bool(result)})
            if not result:
                errors.append(f"External Actor 保存失败: {entity_id}")
        except Exception as exc:
            errors.append(f"External Actor 保存异常 {entity_id}: {type(exc).__name__}: {exc}")

    postflight = preflight_player_visuals(
        normalized,
        mapping,
        require_driver_refresh=True,
    )
    errors.extend(postflight["errors"])
    return {
        "ok": not errors and postflight["ok"],
        "operation": "apply",
        "errors": errors,
        "warnings": postflight["warnings"],
        "actors": postflight["actors"],
        "saved": saved,
        "rollback_path": rollback_path,
        "rollback_preserved": rollback_preserved,
    }


def restore_player_visuals(rollback_path: str) -> Dict[str, Any]:
    """按 rollback 快照仅恢复 Mesh 和 AnimBP。"""

    import unreal

    path = Path(rollback_path)
    if not path.is_file():
        return {
            "ok": False,
            "operation": "restore",
            "errors": [f"rollback 文件不存在: {rollback_path}"],
            "warnings": [],
            "saved": [],
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    errors = []
    if payload.get("schema") != "futsalmot_player_visuals_rollback":
        errors.append("rollback schema 非法")
    if payload.get("level") != _current_level_path(unreal):
        errors.append(
            f"rollback 地图不匹配: expected={payload.get('level')} actual={_current_level_path(unreal)}"
        )
    if errors:
        return {"ok": False, "operation": "restore", "errors": errors, "warnings": [], "saved": []}

    changed = []
    for snapshot in payload.get("actors", []):
        try:
            _restore_actor_payload(
                unreal,
                build_restore_payload(snapshot),
                expected_label=snapshot.get("label"),
            )
            changed.append(snapshot["entity_id"])
        except Exception as exc:
            errors.append(f"恢复失败 {snapshot.get('entity_id')}: {type(exc).__name__}: {exc}")
            break
    saved = []
    for snapshot in payload.get("actors", []):
        if snapshot.get("entity_id") not in changed:
            continue
        try:
            actor = unreal.load_object(None, snapshot["actor_path"])
            result = unreal.EditorAssetLibrary.save_loaded_asset(actor, only_if_is_dirty=True)
            saved.append({"entity_id": snapshot["entity_id"], "saved": bool(result)})
            if not result:
                errors.append(f"External Actor 恢复保存失败: {snapshot['entity_id']}")
        except Exception as exc:
            errors.append(
                f"External Actor 恢复保存异常 {snapshot['entity_id']}: "
                f"{type(exc).__name__}: {exc}"
            )
    return {
        "ok": not errors and len(changed) == len(payload.get("actors", [])),
        "operation": "restore",
        "errors": errors,
        "warnings": [],
        "saved": saved,
        "restored": changed,
    }
