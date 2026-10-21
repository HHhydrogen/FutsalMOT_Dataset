"""只读检查 episode environment v1.5 的 UE 地图和属性证据。"""

from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple


FORMAL_MAP = "/Game/FutsalMOT/Maps/L_FutsalCourt"
ENVIRONMENT_POLICY = "futsalmot_environment_v1_5"
ENVIRONMENT_SCHEMA = "futsalmot_episode_environment_diagnostic"
ENVIRONMENT_VERSION = 1

REQUIRED_ACTORS: Tuple[Tuple[str, str], ...] = (
    ("DirectionalLight", "DirectionalLight"),
    ("SkyLight", "SkyLight"),
    ("SkyAtmosphere", "SkyAtmosphere"),
    ("VolumetricCloud", "VolumetricCloud"),
    ("ExponentialHeightFog", "ExponentialHeightFog"),
    ("PostProcessVolume", "PostProcessVolume"),
)

# 这些是 Task 4 允许继续验证的 semantic property key。具体 UE 属性名必须
# 由当前 Editor 的 get_editor_property 读回证据确认；本文件不会写入这些字段。
ENVIRONMENT_PROPERTY_FIELDS = {
    "DirectionalLight": (
        "rotation_deg",
        "intensity",
        "light_color_rgba",
        "temperature_enabled",
        "temperature_kelvin",
        "cast_shadows",
    ),
    "SkyLight": (
        "intensity",
        "light_color_rgba",
        "source_type",
        "real_time_capture",
    ),
    "SkyAtmosphere": ("transform_rotation_deg", "multi_scattering"),
    "VolumetricCloud": (
        "visible",
        "layer_bottom_cm",
        "layer_height_cm",
        "cloud_coverage",
        "cloud_density",
    ),
    "ExponentialHeightFog": (
        "visible",
        "fog_density",
        "height_falloff",
        "fog_height_falloff",
        "volumetric_fog_scattering_distribution",
        "volumetric_fog_albedo_rgba",
    ),
    "PostProcessVolume": (
        "infinite_extent",
        "exposure_compensation",
        "white_balance_temperature",
        "white_balance_tint",
        "color_contrast",
        "color_saturation",
    ),
}
VENUE_FILL_PARAMETER_FIELDS = {
    "enabled",
    "intensity",
    "intensity_units",
    "attenuation_radius_cm",
    "source_width_cm",
    "source_height_cm",
    "light_color_rgba",
    "cast_shadows",
    "locations_cm",
}
MANUAL_SPOTLIGHT_ACTOR_LABELS = (
    "SpotLight",
    "SpotLight2",
    "SpotLight3",
    "SpotLight4",
    "SpotLight5",
    "SpotLight6",
    "SpotLight7",
    "SpotLight8",
    "SpotLight9",
    "SpotLight10",
)

_VERIFIED_PROPERTY_SPECS = {
    "DirectionalLight": {
        "intensity": ("component", "intensity"),
        "light_color_rgba": ("component", "light_color"),
        "temperature_enabled": ("component", "use_temperature"),
        "temperature_kelvin": ("component", "temperature"),
        "cast_shadows": ("component", "cast_shadows"),
    },
    "SkyLight": {
        "intensity": ("component", "intensity"),
        "light_color_rgba": ("component", "light_color"),
        "source_type": ("component", "source_type"),
        "real_time_capture": ("component", "real_time_capture"),
    },
    "SkyAtmosphere": {
        "multi_scattering": ("component", "multi_scattering_factor"),
    },
    "VolumetricCloud": {
        "visible": ("component", "visible"),
        "layer_bottom_cm": ("component", "layer_bottom_altitude"),
        "layer_height_cm": ("component", "layer_height"),
        "cloud_coverage": ("material", "Cloud_GlobalCoverage"),
        "cloud_density": ("material", "Cloud_GlobalDensity"),
    },
    "ExponentialHeightFog": {
        "visible": ("component", "visible"),
        "fog_density": ("component", "fog_density"),
        "height_falloff": ("component", "fog_height_falloff"),
        "fog_height_falloff": ("component", "fog_height_falloff"),
        "volumetric_fog_scattering_distribution": (
            "component",
            "volumetric_fog_scattering_distribution",
        ),
        "volumetric_fog_albedo_rgba": ("component", "volumetric_fog_albedo"),
    },
    "PostProcessVolume": {
        "infinite_extent": ("actor", "unbound"),
        "exposure_compensation": ("settings", "auto_exposure_bias"),
        "white_balance_temperature": ("settings", "white_temp"),
        "white_balance_tint": ("settings", "white_tint"),
        "color_contrast": ("settings", "color_contrast"),
        "color_saturation": ("settings", "color_saturation"),
    },
}

# PostProcessSettings 的 override 开关默认关闭；写入前必须显式记录并打开，
# restore 时按 baseline 写回，避免留下生效状态与 baseline 不一致的 Volume。
SETTINGS_OVERRIDE_FLAGS = {
    "exposure_compensation": "override_auto_exposure_bias",
    "white_balance_temperature": "override_white_temp",
    "white_balance_tint": "override_white_tint",
    "color_contrast": "override_color_contrast",
    "color_saturation": "override_color_saturation",
}

_INVALID_EVIDENCE_SOURCES = frozenset(
    {"", "unknown", "placeholder", "not_discovered", "manual", "None"}
)
_ALLOWED_EVIDENCE_SOURCES = frozenset(
    {
        "get_editor_property",
        "get_actor_rotation",
        "get_scalar_parameter_value",
    }
)


def _error_once(errors: list, message: str) -> None:
    if message not in errors:
        errors.append(message)


def _valid_marker(value: Any, expected_source: str) -> bool:
    if not isinstance(value, Mapping):
        return False
    if "readback" not in value or "evidence_source" not in value:
        return False
    source = value.get("evidence_source")
    return (
        isinstance(source, str)
        and source in _ALLOWED_EVIDENCE_SOURCES
        and source not in _INVALID_EVIDENCE_SOURCES
        and source == expected_source
    )


def _property_errors(label: str, record: Mapping[str, Any]) -> list:
    required = set(ENVIRONMENT_PROPERTY_FIELDS[label])
    discovered = record.get("discovered_properties")
    readback = record.get("baseline_readback")
    actor_sources = record.get("property_evidence_source")
    errors = []
    if not isinstance(discovered, Mapping):
        errors.append("{} property discovery 记录不是结构化映射".format(label))
        discovered = {}
    missing_discovered = sorted(required - set(discovered))
    if missing_discovered:
        errors.append(
            "{} property discovery 缺失: {}".format(label, ", ".join(missing_discovered))
        )
    extra_discovered = sorted(set(discovered) - required)
    if extra_discovered:
        errors.append(
            "{} property discovery 含未知 mapping: {}".format(
                label, ", ".join(str(field) for field in extra_discovered)
            )
        )
    if not isinstance(readback, Mapping):
        return errors + ["{} property baseline_readback 缺失或不是对象".format(label)]
    if not isinstance(actor_sources, Mapping):
        errors.append("{} actor-level property_evidence_source 缺失或不是映射".format(label))
        actor_sources = {}
    extra_sources = sorted(set(actor_sources) - required)
    if extra_sources:
        errors.append(
            "{} actor-level property_evidence_source 含未知字段: {}".format(
                label, ", ".join(str(field) for field in extra_sources)
            )
        )
    for field in sorted(required):
        spec = _VERIFIED_PROPERTY_SPECS.get(label, {}).get(field)
        is_rotation = field in ("rotation_deg", "transform_rotation_deg")
        if is_rotation:
            expected = {
                "property_name": "get_actor_rotation",
                "operation": "read",
                "target": "actor",
                "evidence_source": "get_actor_rotation",
            }
        elif spec is None:
            expected = None
        else:
            target, property_name = spec
            evidence_source = (
                "get_scalar_parameter_value"
                if target == "material"
                else "get_editor_property"
            )
            expected = {
                "property_name": property_name,
                "operation": evidence_source,
                "target": target,
                "evidence_source": evidence_source,
            }
        discovery = discovered.get(field)
        discovery_source = None
        if expected is None:
            errors.append(
                "{} property {} mapping 尚未通过 discovery 验证".format(label, field)
            )
        elif not isinstance(discovery, Mapping):
            errors.append(
                "{} property {} discovery 必须是结构化映射".format(label, field)
            )
        else:
            discovery_source = discovery.get("evidence_source")
            for key, expected_value in expected.items():
                if discovery.get(key) != expected_value:
                    errors.append(
                        "{} property {} discovery mapping {} 不匹配".format(
                            label, field, key
                        )
                    )
            if discovery_source not in _ALLOWED_EVIDENCE_SOURCES:
                errors.append(
                    "{} property {} discovery evidence_source 非法".format(label, field)
                )
            if expected is not None and expected["operation"] in {
                "get_editor_property",
                "get_scalar_parameter_value",
            }:
                attempts = discovery.get("attempts")
                if type(attempts) is not int or attempts not in (1, 2):
                    errors.append(
                        "{} property {} discovery attempts 无效".format(label, field)
                    )
            if expected is not None and expected["operation"] == "read":
                if "attempts" in discovery:
                    errors.append(
                        "{} property {} Actor read 不应声明 get_editor_property attempts".format(
                            label, field
                        )
                    )
        evidence = readback.get(field)
        if discovery_source not in _ALLOWED_EVIDENCE_SOURCES or not _valid_marker(
            evidence, discovery_source
        ):
            errors.append(
                "{} property {} 缺少真实 readback/evidence_source".format(label, field)
            )
        actor_source = actor_sources.get(field)
        if (
            not isinstance(actor_source, str)
            or actor_source not in _ALLOWED_EVIDENCE_SOURCES
            or actor_source != discovery_source
        ):
            errors.append(
                "{} property {} actor-level evidence_source 缺失或与 discovery 不一致".format(
                    label, field
                )
            )
    return errors


def validate_runtime_state(
    *, play_session_active: Any, mrq_state: Any
) -> list:
    """验证 PIE/Simulate 与 MRQ 都有明确空闲证据。"""
    errors = []
    if type(play_session_active) is not bool:
        errors.append("PIE/Simulate runtime state 未知，MANUAL_REQUIRED")
    elif play_session_active:
        errors.append("PIE/Simulate 正在运行，环境 preflight 必须停止")
    if mrq_state not in ("inactive", "active"):
        errors.append("MRQ runtime state 未知，MANUAL_REQUIRED")
    elif mrq_state == "active":
        errors.append("MRQ 正在运行，环境 preflight 必须停止")
    return errors


def build_environment_diagnostic_report(
    current_level: Any,
    actor_records: Mapping,
    package_dirty: Sequence[str],
    errors: Sequence[str],
    *,
    policy: str,
    fill_parameters: Optional[Mapping] = None,
    active_fill_actor_labels: Sequence[str] = (),
    play_session_active: Any = None,
    mrq_state: Any = None,
    runtime_evidence: Optional[Mapping] = None,
    manual_spot_lighting: Optional[Mapping] = None,
) -> dict:
    """从只读采集结果生成 fail-closed 的环境诊断报告。"""
    records = actor_records if isinstance(actor_records, Mapping) else {}
    dirty = sorted(
        {path for path in package_dirty if isinstance(path, str) and path}
    )
    collected_errors = [error for error in errors if isinstance(error, str) and error]
    fill_values = (
        deepcopy(dict(fill_parameters))
        if isinstance(fill_parameters, Mapping)
        else None
    )
    if fill_values is not None:
        if set(fill_values) != VENUE_FILL_PARAMETER_FIELDS:
            _error_once(collected_errors, "venue_fill_lighting 参数不完整")
        elif fill_values.get("enabled") is not False:
            _error_once(collected_errors, "UE 事务 apply 前 venue fill lighting 必须 inactive")
        elif fill_values.get("locations_cm") != []:
            _error_once(collected_errors, "disabled venue fill lighting 不应包含位置")
    active_fill_labels = sorted(
        {
            label
            for label in active_fill_actor_labels
            if isinstance(label, str) and label
        }
    )
    if active_fill_labels:
        _error_once(
            collected_errors,
            "上次 venue fill lighting transaction 未恢复: {}".format(
                ", ".join(active_fill_labels)
            ),
        )
    manual_spot = deepcopy(dict(manual_spot_lighting)) if isinstance(
        manual_spot_lighting, Mapping
    ) else None
    if manual_spot is None:
        _error_once(collected_errors, "manual SpotLight 诊断缺失")
    else:
        if (
            manual_spot.get("actor_labels") != list(MANUAL_SPOTLIGHT_ACTOR_LABELS)
            or manual_spot.get("count") != len(MANUAL_SPOTLIGHT_ACTOR_LABELS)
            or manual_spot.get("class") != "SpotLight"
            or manual_spot.get("intensity_units") != "ev"
            or not isinstance(manual_spot.get("intensity"), (int, float))
        ):
            _error_once(collected_errors, "manual SpotLight 必须唯一包含 10 个 EV 光源")
    collected_errors.extend(
        error
        for error in validate_runtime_state(
            play_session_active=play_session_active,
            mrq_state=mrq_state,
        )
        if error not in collected_errors
    )
    actors: Dict[str, dict] = {}
    resolved_paths = []
    resolved_packages = []

    if current_level != FORMAL_MAP:
        _error_once(
            collected_errors,
            "当前 Level 不匹配正式地图：expected={} actual={}".format(
                FORMAL_MAP, current_level
            ),
        )
    if policy != ENVIRONMENT_POLICY:
        _error_once(
            collected_errors,
            "environment policy 不匹配：expected={} actual={}".format(
                ENVIRONMENT_POLICY, policy
            ),
        )
    if dirty:
        _error_once(
            collected_errors,
            "存在 dirty package，环境 preflight 必须停止: {}".format(
                ", ".join(dirty)
            ),
        )

    for label, expected_class in REQUIRED_ACTORS:
        source_record = records.get(label)
        record = dict(source_record) if isinstance(source_record, Mapping) else {}
        source_record_copy = deepcopy(record)
        match_count = record.get("match_count", 0)
        match_count_valid = type(match_count) is int and match_count == 1
        actors[label] = {
            "expected_label": label,
            "expected_class": expected_class,
            "actor_label": record.get("actor_label"),
            "actor_path": record.get("actor_path"),
            "class": record.get("class"),
            "package_path": record.get("package_path"),
            "match_count": match_count,
            "discovered_properties": deepcopy(record.get("discovered_properties")),
            "baseline_readback": deepcopy(record.get("baseline_readback")),
            "property_evidence_source": record.get("property_evidence_source"),
            "source_record": source_record_copy,
        }

        identity_ok = (
            match_count_valid
            and record.get("actor_label") == label
            and record.get("class") == expected_class
            and isinstance(record.get("actor_path"), str)
            and bool(record.get("actor_path"))
            and isinstance(record.get("package_path"), str)
            and bool(record.get("package_path"))
        )
        if not match_count_valid:
            _error_once(
                collected_errors,
                "{} Actor match_count 必须是严格整数 1，实际={!r} type={}".format(
                    label, match_count, type(match_count).__name__
                ),
            )
        if record.get("actor_label") != label:
            _error_once(
                collected_errors,
                "{} Actor label 不匹配：actual={}".format(
                    label, record.get("actor_label")
                ),
            )
        if record.get("class") != expected_class:
            _error_once(
                collected_errors,
                "{} Actor class 不匹配：expected={} actual={}".format(
                    label, expected_class, record.get("class")
                ),
            )
        if not isinstance(record.get("actor_path"), str) or not record.get("actor_path"):
            _error_once(collected_errors, "{} Actor path 缺失".format(label))
        if not isinstance(record.get("package_path"), str) or not record.get("package_path"):
            _error_once(collected_errors, "{} package path 缺失".format(label))
        if isinstance(record.get("package_path"), str) and record["package_path"] in dirty:
            _error_once(collected_errors, "{} package dirty".format(label))
        if identity_ok:
            resolved_paths.append(record["actor_path"])
            resolved_packages.append(record["package_path"])

        for property_error in _property_errors(label, record):
            _error_once(collected_errors, "property evidence {}: {}".format(label, property_error))

    for value, message in (
        (resolved_paths, "Actor path"),
        (resolved_packages, "package path"),
    ):
        duplicates = sorted({item for item in value if value.count(item) > 1})
        if duplicates:
            _error_once(
                collected_errors,
                "环境控制 {} 重复: {}".format(message, ", ".join(duplicates)),
            )

    report = {
        "schema": ENVIRONMENT_SCHEMA,
        "version": ENVIRONMENT_VERSION,
        "current_level": current_level,
        "environment_policy": policy,
        "actors": actors,
        "package_dirty": dirty,
        "runtime_state": {
            "play_session_active": play_session_active,
            "mrq_state": mrq_state,
            "evidence": deepcopy(dict(runtime_evidence))
            if isinstance(runtime_evidence, Mapping)
            else {},
        },
        "errors": collected_errors,
        "preflight_can_start": not collected_errors,
    }
    if manual_spot is not None:
        report["manual_spot_lighting"] = manual_spot
    if fill_values is not None:
        report["venue_fill_lighting"] = {
            "requested": fill_values,
            "active_actor_labels": active_fill_labels,
        }
    return report


def _package_path(value: Any) -> Optional[str]:
    if value is None:
        return None
    getter = getattr(value, "get_path_name", None)
    path = str(getter()) if callable(getter) else str(value)
    return path.split(".", 1)[0] if path else None


def _json_safe(value: Any) -> Any:
    """把 UE struct/enum 转成不依赖 Unreal 类型的 JSON 基础值。"""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _json_safe(child) for key, child in value.items()}
    for names in (
        ("r", "g", "b", "a"),
        ("x", "y", "z", "w"),
        ("pitch", "yaw", "roll"),
    ):
        if all(hasattr(value, name) for name in names):
            return {name: _json_safe(getattr(value, name)) for name in names}
    return str(value)


def _normalize_light_unit(value: Any) -> str:
    name = getattr(value, "name", None)
    if not name:
        name = str(value).split(".")[-1].split(":", 1)[0].strip(" >")
    return str(name).lower()


def _semantic_readback(label: str, property_name: str, value: Any) -> Any:
    """将 UE 原生值归一化为 environment manifest 的便携语义值。"""
    if property_name in ("rotation_deg", "transform_rotation_deg"):
        return [float(value.pitch), float(value.yaw), float(value.roll)]
    if property_name in ("light_color_rgba", "volumetric_fog_albedo_rgba"):
        channels = [float(getattr(value, name)) for name in ("r", "g", "b", "a")]
        if max(channels) > 1.0:
            channels = [channel / 255.0 for channel in channels]
        return channels
    if label == "SkyLight" and property_name == "source_type":
        text = str(value)
        if "SLS_CAPTURED_SCENE" in text or "SLS_SPECIFIED_CUBEMAP" in text:
            return "specified_cube_or_scene"
    if property_name in ("color_contrast", "color_saturation"):
        return [float(getattr(value, name)) for name in ("x", "y", "z", "w")]
    return _json_safe(value)


def _current_level_path(unreal_module: Any) -> Optional[str]:
    world = unreal_module.EditorLevelLibrary.get_editor_world()
    return _package_path(world)


def _all_level_actors(unreal_module: Any) -> list:
    subsystem = unreal_module.get_editor_subsystem(unreal_module.EditorActorSubsystem)
    return list(subsystem.get_all_level_actors())


def _runtime_state(unreal_module: Any) -> Tuple[dict, list]:
    """只读确认当前没有 PIE/Simulate 或 MRQ 会话。"""
    errors = []
    play_session_active = None
    mrq_state = None
    evidence = {
        "play_session_api": "unverified",
        "pie_world_count": None,
        "mrq_api": "unverified",
        "mrq_subsystem_is_rendering": None,
        "active_executor_exists": None,
        "active_executor_is_rendering": None,
    }
    try:
        get_pie_worlds = getattr(unreal_module.EditorLevelLibrary, "get_pie_worlds", None)
        if not callable(get_pie_worlds):
            raise RuntimeError("EditorLevelLibrary.get_pie_worlds 不可用")
        raw_pie_worlds = get_pie_worlds(False)
        if raw_pie_worlds is None or isinstance(raw_pie_worlds, (str, bytes, Mapping)):
            raise TypeError("get_pie_worlds 返回值不是可迭代 world array")
        try:
            pie_worlds = list(raw_pie_worlds)
        except TypeError as exc:
            raise TypeError("get_pie_worlds 返回值不可迭代") from exc
        evidence["play_session_api"] = "EditorLevelLibrary.get_pie_worlds(False)"
        evidence["pie_world_count"] = len(pie_worlds)
        play_session_active = bool(pie_worlds)
    except Exception as exc:
        errors.append(
            "PIE/Simulate 查询失败，MANUAL_REQUIRED: {}: {}".format(
                type(exc).__name__, exc
            )
        )

    try:
        subsystem_class = getattr(unreal_module, "MoviePipelineQueueSubsystem", None)
        if subsystem_class is None:
            raise RuntimeError("MoviePipelineQueueSubsystem 不可用")
        subsystem = unreal_module.get_editor_subsystem(subsystem_class)
        if subsystem is None:
            raise RuntimeError("MoviePipelineQueueSubsystem instance 不可用")
        is_rendering = getattr(subsystem, "is_rendering", None)
        get_active_executor = getattr(subsystem, "get_active_executor", None)
        if not callable(is_rendering) or not callable(get_active_executor):
            raise RuntimeError("MRQ subsystem 缺少已验证的只读状态 API")
        rendering = is_rendering()
        if type(rendering) is not bool:
            raise TypeError("MoviePipelineQueueSubsystem.is_rendering 未返回 bool")
        executor = get_active_executor()
        evidence["mrq_api"] = (
            "MoviePipelineQueueSubsystem.is_rendering/get_active_executor"
        )
        evidence["mrq_subsystem_is_rendering"] = rendering
        evidence["active_executor_exists"] = executor is not None
        if executor is not None:
            executor_is_rendering = getattr(executor, "is_rendering", None)
            if not callable(executor_is_rendering):
                raise RuntimeError("active MRQ executor 缺少 is_rendering")
            executor_rendering = executor_is_rendering()
            if type(executor_rendering) is not bool:
                raise TypeError("active executor is_rendering 未返回 bool")
            evidence["active_executor_is_rendering"] = executor_rendering
            mrq_state = "active"
        else:
            mrq_state = "active" if rendering else "inactive"
    except Exception as exc:
        errors.append(
            "MRQ 查询失败，MANUAL_REQUIRED: {}: {}".format(type(exc).__name__, exc)
        )

    return {
        "play_session_active": play_session_active,
        "mrq_state": mrq_state,
        "evidence": evidence,
    }, errors


def _dirty_package_paths(unreal_module: Any) -> set:
    loading = unreal_module.EditorLoadingAndSavingUtils
    packages = []
    for method_name in ("get_dirty_map_packages", "get_dirty_content_packages"):
        method = getattr(loading, method_name, None)
        if not callable(method):
            raise RuntimeError("UE 缺少 dirty package 查询: {}".format(method_name))
        packages.extend(method())
    return {
        path for path in (_package_path(package) for package in packages) if path
    }


def _class_name(value: Any) -> Optional[str]:
    getter = getattr(value, "get_class", None)
    if not callable(getter):
        return None
    class_object = getter()
    name_getter = getattr(class_object, "get_name", None)
    return str(name_getter()) if callable(name_getter) else None


def _component_for_actor(unreal_module: Any, actor: Any, expected_class: str) -> Any:
    if expected_class == "PostProcessVolume":
        return actor
    get_components = getattr(actor, "get_components_by_class", None)
    if not callable(get_components):
        raise RuntimeError("{} 缺少 get_components_by_class".format(expected_class))
    component_class = getattr(unreal_module, "{}Component".format(expected_class), None)
    if component_class is None:
        raise RuntimeError("UE 未暴露 {}Component".format(expected_class))
    components = list(get_components(component_class))
    if len(components) != 1:
        raise RuntimeError(
            "{} component 必须唯一，实际 {}".format(expected_class, len(components))
        )
    return components[0]


def _read_property(target: Any, property_name: str) -> Any:
    getter = getattr(target, "get_editor_property", None)
    if not callable(getter):
        raise AttributeError("目标没有 get_editor_property")
    return getter(property_name)


def _read_with_one_retry(target: Any, property_name: str) -> Tuple[Any, int, Optional[str]]:
    last_error = None
    for attempt in (1, 2):
        try:
            return _read_property(target, property_name), attempt, None
        except Exception as exc:
            last_error = "{}: {}".format(type(exc).__name__, exc)
    return None, 2, last_error


def _read_method_with_one_retry(
    target: Any, method_name: str, argument: str
) -> Tuple[Any, int, Optional[str]]:
    last_error = None
    method = getattr(target, method_name, None)
    if not callable(method):
        return None, 2, "目标缺少 {}".format(method_name)
    for attempt in (1, 2):
        try:
            return method(argument), attempt, None
        except Exception as exc:
            last_error = "{}: {}".format(type(exc).__name__, exc)
    return None, 2, last_error


def _component_properties(
    unreal_module: Any, actor: Any, label: str
) -> Tuple[dict, dict, dict, list]:
    """只读发现一个环境 Actor 的目标属性，失败两次即停止该 actor。"""
    component = _component_for_actor(unreal_module, actor, label)
    settings = None
    material = None
    if label == "PostProcessVolume":
        settings, _, settings_error = _read_with_one_retry(component, "settings")
        if settings_error is not None:
            return {}, {}, {}, [
                "{} property settings 两次读取失败: {}".format(label, settings_error)
            ]
    if label == "VolumetricCloud":
        material, _, material_error = _read_with_one_retry(component, "material")
        if material_error is not None:
            return {}, {}, {}, [
                "{} property material 两次读取失败: {}".format(label, material_error)
            ]
    discovered = {}
    baseline = {}
    actor_sources = {}
    failures = []
    for property_name in ENVIRONMENT_PROPERTY_FIELDS[label]:
        if property_name in ("rotation_deg", "transform_rotation_deg"):
            try:
                value = actor.get_actor_rotation()
                discovered[property_name] = {
                    "property_name": "get_actor_rotation",
                    "operation": "read",
                    "target": "actor",
                    "evidence_source": "get_actor_rotation",
                }
                baseline[property_name] = {
                    "readback": _semantic_readback(label, property_name, value),
                    "evidence_source": "get_actor_rotation",
                }
                actor_sources[property_name] = "get_actor_rotation"
                continue
            except Exception as exc:
                failures.append(
                    "{} property {} 读取失败: {}".format(
                        label, property_name, exc
                    )
                )
                continue
        target_kind, actual_property = _VERIFIED_PROPERTY_SPECS[label].get(
            property_name, (None, None)
        )
        if target_kind is None or actual_property is None:
            failures.append(
                "{} property {} 尚未通过 UE discovery，禁止猜测属性名".format(
                    label, property_name
                )
            )
            continue
        target = {
            "settings": settings,
            "material": material,
        }.get(target_kind, component)
        if target_kind == "material":
            value, attempts, error = _read_method_with_one_retry(
                target, "get_scalar_parameter_value", actual_property
            )
            evidence_source = "get_scalar_parameter_value"
        else:
            value, attempts, error = _read_with_one_retry(target, actual_property)
            evidence_source = "get_editor_property"
        if error is not None:
            failures.append(
                "{} property {} 两次读取失败: {}".format(
                    label, property_name, error
                )
            )
            continue
        overrides = {}
        if target_kind == "settings":
            override_name = SETTINGS_OVERRIDE_FLAGS.get(property_name)
            if override_name is not None:
                override_value, _, override_error = _read_with_one_retry(
                    settings, override_name
                )
                if override_error is not None:
                    failures.append(
                        "{} property {} override 两次读取失败: {}".format(
                            label, override_name, override_error
                        )
                    )
                    continue
                overrides[override_name] = bool(override_value)
        discovered[property_name] = {
            "property_name": actual_property,
            "operation": evidence_source,
            "target": target_kind,
            "attempts": attempts,
            "evidence_source": evidence_source,
        }
        baseline[property_name] = {
            "readback": _semantic_readback(label, property_name, value),
            "evidence_source": evidence_source,
        }
        if overrides:
            baseline[property_name]["overrides"] = overrides
        actor_sources[property_name] = evidence_source
    return discovered, baseline, actor_sources, failures


def collect_environment_diagnostic(
    unreal_module: Any, fill_parameters: Optional[Mapping] = None
) -> dict:
    """从当前 Editor 只读采集六类环境 Actor 和属性 baseline。"""
    errors = []
    try:
        current_level = _current_level_path(unreal_module)
    except Exception as exc:
        current_level = None
        errors.append("当前 Level 查询失败: {}: {}".format(type(exc).__name__, exc))

    try:
        all_actors = _all_level_actors(unreal_module)
    except Exception as exc:
        all_actors = []
        errors.append("当前 Level Actor 查询失败: {}: {}".format(type(exc).__name__, exc))

    by_label = {}
    for actor in all_actors:
        try:
            by_label.setdefault(actor.get_actor_label(), []).append(actor)
        except Exception as exc:
            errors.append("Actor label 查询失败: {}: {}".format(type(exc).__name__, exc))

    fill_actor_labels = []
    try:
        import episode_environment

        VENUE_FILL_LABEL_PREFIX = episode_environment.VENUE_FILL_LABEL_PREFIX

        fill_actor_labels = sorted(
            label
            for label in by_label
            if isinstance(label, str) and label.startswith(VENUE_FILL_LABEL_PREFIX)
        )
        fill_actor_labels.extend(
            label
            for label in episode_environment.UEEnvironmentTransport._active_fill_lights
            if label not in fill_actor_labels
        )
    except Exception as exc:
        errors.append(
            "venue fill lighting actor 查询失败: {}: {}".format(
                type(exc).__name__, exc
            )
        )

    records = {}
    for label, expected_class in REQUIRED_ACTORS:
        matches = by_label.get(label, [])
        record = {"match_count": len(matches)}
        if len(matches) == 1:
            actor = matches[0]
            record["actor_label"] = label
            record["actor_path"] = actor.get_path_name()
            record["class"] = _class_name(actor)
            record["package_path"] = _package_path(actor.get_outermost())
            try:
                discovered, baseline, actor_sources, property_errors = _component_properties(
                    unreal_module, actor, label
                )
                record["discovered_properties"] = discovered
                record["baseline_readback"] = baseline
                record["property_evidence_source"] = actor_sources
                if property_errors:
                    errors.extend(property_errors)
                    errors.append(
                        "MANUAL_REQUIRED: {} property discovery 未完成".format(label)
                    )
            except Exception as exc:
                errors.append(
                    "{} property discovery 初始化失败: {}: {}".format(
                        label, type(exc).__name__, exc
                    )
                )
                errors.append(
                    "MANUAL_REQUIRED: {} property discovery 不完整".format(label)
                )
        records[label] = record

    manual_matches = [
        actor
        for actor in all_actors
        if actor.get_actor_label() in MANUAL_SPOTLIGHT_ACTOR_LABELS
    ]
    manual_by_label = {actor.get_actor_label(): actor for actor in manual_matches}
    manual_record = {
        "actor_labels": sorted(manual_by_label, key=MANUAL_SPOTLIGHT_ACTOR_LABELS.index),
        "count": len(manual_matches),
        "class": "SpotLight",
        "intensity_units": "unknown",
        "intensity": None,
    }
    if len(manual_matches) == len(MANUAL_SPOTLIGHT_ACTOR_LABELS):
        intensities = []
        units = []
        class_names = []
        for label in MANUAL_SPOTLIGHT_ACTOR_LABELS:
            actor = manual_by_label[label]
            class_names.append(_class_name(actor))
            components = list(actor.get_components_by_class(unreal_module.SpotLightComponent))
            if len(components) != 1:
                errors.append("{} SpotLightComponent 必须唯一".format(label))
                continue
            component = components[0]
            try:
                intensities.append(float(component.get_editor_property("intensity")))
                units.append(_normalize_light_unit(component.get_editor_property("intensity_units")))
            except Exception as exc:
                errors.append("{} manual SpotLight property 读取失败: {}".format(label, exc))
        manual_record["class"] = class_names[0] if len(set(class_names)) == 1 and class_names else None
        manual_record["intensity"] = intensities[0] if len(set(intensities)) == 1 and intensities else None
        manual_record["intensity_units"] = units[0] if len(set(units)) == 1 and units else "unknown"
        if len(set(intensities)) > 1:
            errors.append("manual SpotLight intensity 不一致")
        if len(set(units)) > 1:
            errors.append("manual SpotLight intensity_units 不一致")
    else:
        errors.append("manual SpotLight 必须精确包含 10 个场景 Actor")

    try:
        dirty = sorted(_dirty_package_paths(unreal_module))
    except Exception as exc:
        dirty = []
        errors.append("dirty package 查询失败: {}: {}".format(type(exc).__name__, exc))

    runtime_state, runtime_errors = _runtime_state(unreal_module)
    errors.extend(runtime_errors)

    return build_environment_diagnostic_report(
        current_level,
        records,
        dirty,
        errors,
        policy=ENVIRONMENT_POLICY,
        fill_parameters=fill_parameters,
        active_fill_actor_labels=fill_actor_labels,
        play_session_active=runtime_state["play_session_active"],
        mrq_state=runtime_state["mrq_state"],
        runtime_evidence=runtime_state["evidence"],
        manual_spot_lighting=manual_record,
    )


def _inactive_report(reason: str) -> dict:
    return {
        "schema": ENVIRONMENT_SCHEMA,
        "version": ENVIRONMENT_VERSION,
        "current_level": None,
        "environment_policy": None,
        "actors": {},
        "package_dirty": [],
        "errors": [],
        "preflight_can_start": False,
        "active": False,
        "skipped": True,
        "skip_reason": reason,
    }


def _load_inputs(
    episode_dir: Optional[Path],
    resolved_environment: Optional[Mapping],
) -> Tuple[Optional[Path], Optional[Mapping]]:
    profile_json = os.environ.get("FUTSALMOT_RESOLVED_ENVIRONMENT_JSON")
    if resolved_environment is None and profile_json:
        resolved_environment = json.loads(profile_json)
    resolved_path = os.environ.get("FUTSALMOT_RESOLVED_TASK")
    if resolved_environment is None and resolved_path:
        payload = json.loads(Path(resolved_path).read_text(encoding="utf-8"))
        resolved_environment = (payload.get("ue_profile") or {}).get(
            "environment_profile"
        )
        if episode_dir is None and payload.get("dataset_episode_dir"):
            episode_dir = Path(payload["dataset_episode_dir"])
    if episode_dir is None and os.environ.get("FUTSALMOT_EPISODE_DIR"):
        episode_dir = Path(os.environ["FUTSALMOT_EPISODE_DIR"])
    return episode_dir, resolved_environment


def diagnose_environment(
    episode_dir: Optional[Path] = None,
    resolved_environment: Optional[Mapping] = None,
) -> dict:
    """在真实 UE Editor 中执行 v1.5 的只读环境诊断。"""
    episode_dir, resolved_environment = _load_inputs(episode_dir, resolved_environment)
    if not isinstance(resolved_environment, Mapping):
        return _inactive_report("resolved environment profile 缺失，未激活 v1.5")
    policy = resolved_environment.get("policy")
    if policy != ENVIRONMENT_POLICY:
        return _inactive_report("resolved environment policy 未激活 v1.5")

    fill_parameters = (resolved_environment.get("parameters") or {}).get(
        "venue_fill_lighting"
    )
    if episode_dir is not None:
        manifest_path = episode_dir / "provenance" / "environment.json"
        if not manifest_path.is_file():
            import unreal

            report = build_environment_diagnostic_report(
                None,
                {},
                [],
                ["environment manifest 不存在: {}".format(manifest_path)],
                policy=policy,
                fill_parameters=fill_parameters,
            )
            unreal.log("environment diagnostic manifest missing")
            return report
        # P1/P2 只通过磁盘契约交接；新 Editor 会话没有 P1 包的 import path。
        # 不在 UE 中重新生成参数，以免依赖旧模块缓存或不同版本的 catalog。
        if fill_parameters is None:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            fill_parameters = (manifest.get("parameters") or {}).get("venue_fill_lighting")
    if fill_parameters is None:
        raise ValueError("environment manifest 缺少 venue_fill_lighting 参数，拒绝重新抽样")

    import unreal

    return collect_environment_diagnostic(unreal, fill_parameters)


def main() -> dict:
    """输出结构化诊断 JSON；该入口不执行任何 UE 写操作。"""
    report = diagnose_environment()
    payload = json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2)
    print(payload)
    try:
        import unreal

        unreal.log("futsalmot environment diagnostic: {}".format(payload))
    except Exception:
        pass
    return report


if __name__ == "__main__":
    main()
