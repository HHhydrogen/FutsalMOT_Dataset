"""UE episode environment v1.5 的纯契约和 fail-closed 事务 adapter。"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

ENVIRONMENT_POLICY = "futsalmot_environment_v1_5"
ENVIRONMENT_CATALOG_ID = "futsalmot_indoor_environment_v1_5"
ENVIRONMENT_SCHEMA = "futsalmot_episode_environment"
APPLICATION_REPORT_SCHEMA = "futsalmot_environment_application"
SNAPSHOT_SCHEMA = "futsalmot_episode_environment_rollback"
SNAPSHOT_VERSION = 1
REPORT_VERSION = 1
REPORT_STATES = (
    "not_started",
    "preflight_failed",
    "applied",
    "rendering",
    "restored",
    "restore_failed",
)
FORMAL_MAP = "/Game/FutsalMOT/Maps/L_FutsalCourt"
ACTOR_CLASSES = {
    "DirectionalLight": "DirectionalLight",
    "SkyLight": "SkyLight",
    "SkyAtmosphere": "SkyAtmosphere",
    "VolumetricCloud": "VolumetricCloud",
    "ExponentialHeightFog": "ExponentialHeightFog",
    "PostProcessVolume": "PostProcessVolume",
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
PARAMETER_GROUPS = {
    "DirectionalLight": "directional_light",
    "SkyLight": "sky_light",
    "SkyAtmosphere": "sky_atmosphere",
    "VolumetricCloud": "volumetric_cloud",
    "ExponentialHeightFog": "exponential_height_fog",
    "PostProcessVolume": "post_process",
}
PARAMETER_FIELDS = {
    "directional_light": {
        "rotation_deg", "intensity", "light_color_rgba",
        "temperature_enabled", "temperature_kelvin", "cast_shadows",
    },
    "sky_light": {
        "intensity", "light_color_rgba", "source_type", "real_time_capture",
    },
    "sky_atmosphere": {"transform_rotation_deg", "multi_scattering"},
    "volumetric_cloud": {
        "visible", "layer_bottom_cm", "layer_height_cm", "cloud_coverage", "cloud_density",
    },
    "exponential_height_fog": {
        "visible", "fog_density", "height_falloff", "fog_height_falloff",
        "volumetric_fog_scattering_distribution", "volumetric_fog_albedo_rgba",
    },
    "post_process": {
        "infinite_extent", "exposure_compensation", "white_balance_temperature",
        "white_balance_tint", "color_contrast", "color_saturation",
    },
}
MANUAL_SPOTLIGHT_FIELDS = {
    "enabled", "intensity", "intensity_units", "actor_labels"
}
VENUE_FILL_LABEL_PREFIX = "FutsalMOTEnvironmentFill_"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_LOCAL_PATH_RE = re.compile(
    r"(?:^[A-Za-z]:[\\/]|^\\\\|(?<![A-Za-z0-9_])/[^/\s][^\s,;\"']*)"
)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _manifest_fingerprint(manifest: Mapping[str, Any]) -> str:
    payload = dict(manifest)
    payload.pop("fingerprint", None)
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def _validate_finite_json(value: Any) -> None:
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError("environment parameters 必须是有限值")
    elif isinstance(value, Mapping):
        for child in value.values():
            _validate_finite_json(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _validate_finite_json(child)


def _values_match(expected: Any, actual: Any, *, tolerance: float = 1e-5) -> bool:
    if isinstance(expected, bool) or isinstance(actual, bool):
        return type(expected) is type(actual) and expected == actual
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return math.isclose(float(expected), float(actual), rel_tol=0.0, abs_tol=tolerance)
    if isinstance(expected, Mapping) and isinstance(actual, Mapping):
        return set(expected) == set(actual) and all(
            _values_match(expected[key], actual[key], tolerance=tolerance)
            for key in expected
        )
    if isinstance(expected, (list, tuple)) and isinstance(actual, (list, tuple)):
        return len(expected) == len(actual) and all(
            _values_match(left, right, tolerance=tolerance)
            for left, right in zip(expected, actual)
        )
    return expected == actual


_COLOR_FIELDS = frozenset({"light_color_rgba", "volumetric_fog_albedo_rgba"})
_ROTATION_FIELDS = frozenset({"rotation_deg", "transform_rotation_deg"})
# UE 的 light_color / volumetric_fog_albedo 是 8bit FColor；manifest 保留 0-1 float，
# 回读只能保证 1/255 量化精度，因此这两类字段使用半档容差而不是 1e-5。
_COLOR_TOLERANCE = 0.5 / 255.0 + 1e-6
SETTINGS_OVERRIDE_FLAGS = {
    "exposure_compensation": "override_auto_exposure_bias",
    "white_balance_temperature": "override_white_temp",
    "white_balance_tint": "override_white_tint",
    "color_contrast": "override_color_contrast",
    "color_saturation": "override_color_saturation",
}
_SKYPORT_SOURCE_TYPE = "specified_cube_or_scene"


def _field_tolerance(field: Any) -> float:
    return _COLOR_TOLERANCE if field in _COLOR_FIELDS else 1e-5


def _validate_environment_manifest_inputs(
    manifest: Mapping, profile: Mapping, meta: Mapping, frames_path: Path
) -> list:
    errors = []
    if not isinstance(manifest, Mapping):
        return ["environment manifest 必须是对象"]
    if manifest.get("schema") != ENVIRONMENT_SCHEMA:
        errors.append("environment manifest schema 无效")
    if type(manifest.get("version")) is not int or manifest.get("version") != 1:
        errors.append("environment manifest version 无效")
    for field in ("policy", "catalog_id", "environment_seed", "preset", "catalog_fingerprint"):
        if manifest.get(field) != profile.get(field):
            errors.append("manifest {} 与当前 profile 不一致".format(field))
    if manifest.get("episode_id") != meta.get("episode_id"):
        errors.append("manifest episode_id 与当前 meta 不一致")
    if type(manifest.get("namespace_seed")) is not int:
        errors.append("manifest namespace_seed 无效")
    try:
        actual_hash = hashlib.sha256(Path(frames_path).read_bytes()).hexdigest()
        if manifest.get("trajectory_sha256") != actual_hash:
            errors.append("trajectory_sha256 与当前 frames.jsonl 不一致")
    except (OSError, TypeError, ValueError) as exc:
        errors.append("frames.jsonl 无法读取: {}".format(exc))
    parameters = manifest.get("parameters")
    expected_parameter_groups = set(PARAMETER_FIELDS) | {
        "venue_fill_lighting", "venue_spot_lighting"
    }
    if not isinstance(parameters, Mapping) or set(parameters) != expected_parameter_groups:
        errors.append("manifest parameters 必须精确包含六组环境参数和两组场馆灯光参数")
    else:
        for group, fields in PARAMETER_FIELDS.items():
            values = parameters.get(group)
            if not isinstance(values, Mapping) or set(values) != fields:
                errors.append("manifest parameters.{} 字段不完整".format(group))
        fill = parameters.get("venue_fill_lighting")
        fill_fields = {
            "enabled", "intensity", "intensity_units", "attenuation_radius_cm",
            "source_width_cm", "source_height_cm", "light_color_rgba",
            "cast_shadows", "locations_cm",
        }
        if not isinstance(fill, Mapping) or set(fill) != fill_fields:
            errors.append("manifest parameters.venue_fill_lighting 字段不完整")
        elif (
            type(fill.get("enabled")) is not bool
            or type(fill.get("cast_shadows")) is not bool
            or fill.get("intensity_units") != "lumens"
            or not isinstance(fill.get("locations_cm"), list)
            or (fill.get("enabled") and len(fill.get("locations_cm")) != 4)
            or (not fill.get("enabled") and fill.get("locations_cm") != [])
        ):
            errors.append("manifest venue_fill_lighting 值无效")
        spot = parameters.get("venue_spot_lighting")
        if not isinstance(spot, Mapping) or set(spot) != MANUAL_SPOTLIGHT_FIELDS:
            errors.append("manifest parameters.venue_spot_lighting 字段不完整")
        elif (
            spot.get("enabled") is not True
            or spot.get("intensity_units") != "ev"
            or spot.get("actor_labels") != list(MANUAL_SPOTLIGHT_ACTOR_LABELS)
            or not isinstance(spot.get("intensity"), (int, float))
            or float(spot.get("intensity")) <= 0.0
        ):
            errors.append("manifest venue_spot_lighting 值无效")
        try:
            _validate_finite_json(parameters)
            json.dumps(parameters, allow_nan=False, separators=(",", ":"))
        except (TypeError, ValueError, OverflowError):
            errors.append("manifest parameters 必须是有限且可 JSON 序列化的对象")
    try:
        if not _is_sha256(manifest.get("fingerprint")):
            errors.append("environment manifest fingerprint 无效")
        elif manifest.get("fingerprint") != _manifest_fingerprint(manifest):
            errors.append("environment manifest fingerprint 无效")
    except (TypeError, ValueError, OverflowError):
        errors.append("environment manifest fingerprint 无法计算")
    if _contains_local_path(manifest):
        errors.append("environment manifest 不得包含本机绝对路径")
    return errors


def _contains_local_path(value: Any, key: str = "") -> bool:
    if isinstance(value, Mapping):
        return any(
            _contains_local_path(child, str(child_key))
            for child_key, child in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_local_path(child, key) for child in value)
    if not isinstance(value, str):
        return False
    if "path" in key.lower() and value and not value.startswith("/Game/"):
        return True
    candidate = re.sub(r"/Game/[A-Za-z0-9_./-]+", "UE_ASSET", value)
    return _LOCAL_PATH_RE.search(candidate) is not None


def _validate_actor_identities(actors: Any) -> list:
    errors = []
    if not isinstance(actors, Mapping):
        return ["actors 必须是对象"]
    if set(actors) != set(ACTOR_CLASSES):
        errors.append("actors 必须精确包含六类 environment Actor")
    paths = []
    packages = []
    for label, expected_class in ACTOR_CLASSES.items():
        record = actors.get(label)
        if not isinstance(record, Mapping):
            errors.append("{} Actor identity 缺失".format(label))
            continue
        for field in ("actor_path", "class", "package_path"):
            value = record.get(field)
            if not isinstance(value, str) or not value:
                errors.append("{} Actor {} 缺失".format(label, field))
        if record.get("class") != expected_class:
            errors.append("{} Actor class 与期望不一致".format(label))
        if record.get("actor_label") != label:
            errors.append("{} Actor label identity drift".format(label))
        if type(record.get("match_count")) is not int or record.get("match_count") != 1:
            errors.append("{} Actor match_count 必须为整数 1".format(label))
        actor_path = record.get("actor_path")
        package_path = record.get("package_path")
        if isinstance(actor_path, str) and actor_path:
            paths.append(actor_path)
        if isinstance(package_path, str) and package_path:
            packages.append(package_path)
    if len(paths) != len(set(paths)):
        errors.append("environment Actor path 重复")
    if len(packages) != len(set(packages)):
        errors.append("environment Actor package path 重复")
    return errors


def _actor_identity_fingerprint(actors: Mapping) -> str:
    payload = {
        label: {
            field: actors[label].get(field)
            for field in ("actor_label", "actor_path", "class", "package_path")
        }
        for label in ACTOR_CLASSES
    }
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _expected_field_map(manifest: Mapping[str, Any]) -> dict:
    parameters = manifest.get("parameters")
    if not isinstance(parameters, Mapping):
        return {}
    fields = {}
    for label, group in PARAMETER_GROUPS.items():
        values = parameters.get(group)
        if isinstance(values, Mapping):
            fields[label] = {
                key: deepcopy(value)
                for key, value in values.items()
                if value is not None
            }
        else:
            fields[label] = {}
    return fields


def _manual_spotlight_parameters(manifest: Mapping[str, Any]) -> dict:
    parameters = manifest.get("parameters") or {}
    values = parameters.get("venue_spot_lighting")
    if not isinstance(values, Mapping):
        raise ValueError("manifest venue_spot_lighting 缺失")
    if set(values) != MANUAL_SPOTLIGHT_FIELDS:
        raise ValueError("manifest venue_spot_lighting 字段不完整")
    if values.get("actor_labels") != list(MANUAL_SPOTLIGHT_ACTOR_LABELS):
        raise ValueError("manifest venue_spot_lighting Actor labels 不匹配")
    if values.get("intensity_units") != "ev":
        raise ValueError("manifest venue_spot_lighting intensity_units 必须为 ev")
    return deepcopy(dict(values))


def validate_environment_inputs(
    manifest: Mapping, profile: Mapping, meta: Mapping, frames_path: Path
) -> list[str]:
    """复核 P1 environment manifest 与当前 resolved 输入。"""
    if not isinstance(profile, Mapping):
        return ["environment profile 必须是对象"]
    errors = []
    if profile.get("mode") != "seeded":
        errors.append("environment profile mode 必须为 seeded")
    if profile.get("policy") != ENVIRONMENT_POLICY:
        errors.append("environment profile policy 不匹配")
    if profile.get("catalog_id") != ENVIRONMENT_CATALOG_ID:
        errors.append("environment profile catalog_id 不匹配")
    errors.extend(_validate_environment_manifest_inputs(manifest, profile, meta, Path(frames_path)))
    return list(dict.fromkeys(errors))


def build_environment_snapshot(
    level: Mapping,
    actors: Mapping,
    baseline: Mapping,
    manifest: Mapping,
    transaction_id: str,
    write_plan: Optional[Mapping] = None,
    overrides: Optional[Mapping] = None,
    manual_spot_lighting: Optional[Mapping] = None,
) -> dict:
    """建立包含唯一 Actor identity 和全部目标 baseline 的便携结构。"""
    if not isinstance(level, Mapping):
        raise ValueError("Level identity 必须是对象")
    if not isinstance(manifest, Mapping):
        raise ValueError("environment manifest 必须是对象")
    actor_errors = _validate_actor_identities(actors)
    if actor_errors:
        raise ValueError("environment Actor identity 无效: {}".format("; ".join(actor_errors)))
    expected = _expected_field_map(manifest)
    if set(expected) != set(ACTOR_CLASSES):
        raise ValueError("manifest parameters 缺少六类 environment Actor")
    if not isinstance(baseline, Mapping) or set(baseline) != set(ACTOR_CLASSES):
        raise ValueError("baseline 必须精确包含六类 environment Actor")
    for label, fields in expected.items():
        values = baseline.get(label)
        if not isinstance(values, Mapping) or not set(fields).issubset(values):
            raise ValueError("{} baseline 缺少目标写入字段".format(label))
    if not isinstance(transaction_id, str) or not transaction_id:
        raise ValueError("transaction_id 无效")
    expected_manual = _manual_spotlight_parameters(manifest)
    if manual_spot_lighting is None:
        manual_snapshot = {
            "actor_labels": list(expected_manual["actor_labels"]),
            "baseline": {
                "intensity": float(expected_manual["intensity"]),
                "intensity_units": expected_manual["intensity_units"],
            },
        }
    else:
        manual_snapshot = deepcopy(dict(manual_spot_lighting))
        if manual_snapshot.get("actor_labels") != expected_manual["actor_labels"]:
            raise ValueError("manual SpotLight identity 与 manifest 不匹配")
        baseline_values = manual_snapshot.get("baseline")
        if (
            not isinstance(baseline_values, Mapping)
            or set(baseline_values) != {"intensity", "intensity_units"}
        ):
            raise ValueError("manual SpotLight baseline 无效")
    snapshot = {
        "schema": SNAPSHOT_SCHEMA,
        "version": SNAPSHOT_VERSION,
        "state": "not_started",
        "level": deepcopy(dict(level)),
        "actors": deepcopy(dict(actors)),
        "actor_identity_fingerprint": _actor_identity_fingerprint(actors),
        "baseline": deepcopy(dict(baseline)),
        "write_fields": expected,
        "manifest": deepcopy(dict(manifest)),
        "manifest_fingerprint": manifest.get("fingerprint"),
        "catalog_fingerprint": manifest.get("catalog_fingerprint"),
        "trajectory_sha256": manifest.get("trajectory_sha256"),
        "transaction_id": transaction_id,
        "clean_baseline": {"level_clean": True, "packages": []},
        "transient_fill_lighting": {"enabled": False, "actor_labels": []},
        "venue_fill_parameters": deepcopy(
            manifest.get("parameters", {}).get("venue_fill_lighting", {})
        ),
        "manual_spot_lighting": manual_snapshot,
        "write_plan": deepcopy(dict(write_plan or {})),
        "overrides": deepcopy(dict(overrides or {})),
    }
    errors = validate_environment_snapshot(snapshot)
    if errors:
        raise ValueError("environment snapshot 无效: {}".format("; ".join(errors)))
    return snapshot


def validate_environment_snapshot(snapshot: Mapping) -> list[str]:
    """验证本机 rollback snapshot 的结构、identity、baseline 与路径。"""
    if not isinstance(snapshot, Mapping):
        return ["environment snapshot 必须是对象"]
    required = {
        "schema", "version", "state", "level", "actors", "actor_identity_fingerprint",
        "baseline", "write_fields",
        "manifest", "manifest_fingerprint", "catalog_fingerprint", "trajectory_sha256",
        "transaction_id", "clean_baseline",
        "transient_fill_lighting", "venue_fill_parameters",
        "manual_spot_lighting",
    }
    errors = ["snapshot 缺少字段: {}".format(", ".join(sorted(required - set(snapshot))))] if required - set(snapshot) else []
    if snapshot.get("schema") != SNAPSHOT_SCHEMA:
        errors.append("snapshot schema 无效")
    if type(snapshot.get("version")) is not int or snapshot.get("version") != SNAPSHOT_VERSION:
        errors.append("snapshot version 无效")
    if snapshot.get("state") not in REPORT_STATES:
        errors.append("snapshot state 无效")
    level = snapshot.get("level")
    if (
        not isinstance(level, Mapping)
        or level.get("path") != FORMAL_MAP
        or level.get("package_path") != FORMAL_MAP
        or not isinstance(level.get("class"), str)
        or not level.get("class")
    ):
        errors.append("snapshot Level identity 无效")
    manifest = snapshot.get("manifest")
    if not isinstance(manifest, Mapping):
        errors.append("snapshot manifest 缺失")
    else:
        if manifest.get("schema") != ENVIRONMENT_SCHEMA:
            errors.append("snapshot manifest schema 无效")
        if type(manifest.get("version")) is not int or manifest.get("version") != 1:
            errors.append("snapshot manifest version 无效")
        if manifest.get("policy") != ENVIRONMENT_POLICY:
            errors.append("snapshot manifest policy 无效")
        try:
            if _manifest_fingerprint(manifest) != snapshot.get("manifest_fingerprint"):
                errors.append("snapshot manifest fingerprint drift")
        except (TypeError, ValueError, OverflowError):
            errors.append("snapshot manifest fingerprint 无法验证")
        for field in ("catalog_fingerprint", "trajectory_sha256"):
            if manifest.get(field) != snapshot.get(field):
                errors.append("snapshot {} drift".format(field))
    actors = snapshot.get("actors")
    errors.extend(_validate_actor_identities(actors))
    if isinstance(actors, Mapping) and set(actors) == set(ACTOR_CLASSES):
        try:
            if _actor_identity_fingerprint(actors) != snapshot.get("actor_identity_fingerprint"):
                errors.append("snapshot Actor identity fingerprint drift")
        except (TypeError, ValueError, OverflowError):
            errors.append("snapshot Actor identity fingerprint 无法验证")
    baseline = snapshot.get("baseline")
    write_fields = snapshot.get("write_fields")
    if not isinstance(baseline, Mapping) or not isinstance(write_fields, Mapping):
        errors.append("snapshot baseline/write_fields 必须是对象")
    else:
        if set(baseline) != set(ACTOR_CLASSES) or set(write_fields) != set(ACTOR_CLASSES):
            errors.append("snapshot baseline/write_fields 必须包含六类 Actor")
        for label in ACTOR_CLASSES:
            fields = write_fields.get(label)
            values = baseline.get(label)
            if not isinstance(fields, Mapping) or not isinstance(values, Mapping):
                errors.append("{} snapshot baseline/write_fields 无效".format(label))
            elif not set(fields).issubset(values):
                errors.append("{} snapshot baseline 缺少写入字段".format(label))
    for field in ("manifest_fingerprint", "catalog_fingerprint", "trajectory_sha256"):
        value = snapshot.get(field)
        if not _is_sha256(value):
            errors.append("snapshot {} 无效".format(field))
    if not isinstance(snapshot.get("transaction_id"), str) or not snapshot.get("transaction_id"):
        errors.append("snapshot transaction_id 无效")
    clean = snapshot.get("clean_baseline")
    if not isinstance(clean, Mapping) or clean.get("level_clean") is not True:
        errors.append("snapshot clean baseline 必须确认 Level clean")
    elif clean.get("packages") != sorted(set(clean.get("packages") or [])) or clean.get("packages"):
        errors.append("snapshot dirty package baseline 不允许 apply")
    write_plan = snapshot.get("write_plan")
    if write_plan is not None:
        if not isinstance(write_plan, Mapping):
            errors.append("snapshot write_plan 必须是对象")
        else:
            for label, fields in write_plan.items():
                if label not in ACTOR_CLASSES or not isinstance(fields, Mapping):
                    errors.append("snapshot write_plan {} 无效".format(label))
                    continue
                for field, entry in fields.items():
                    if not isinstance(entry, Mapping):
                        errors.append("snapshot write_plan {}.{} 无效".format(label, field))
                        continue
                    if entry.get("target") not in {"component", "settings", "material", "actor"}:
                        errors.append("snapshot write_plan {}.{} target 无效".format(label, field))
                    property_name = entry.get("property_name")
                    if not isinstance(property_name, str) or not property_name:
                        errors.append(
                            "snapshot write_plan {}.{} property_name 无效".format(label, field)
                        )
                    overrides = entry.get("overrides")
                    if overrides is not None and not (
                        isinstance(overrides, Mapping)
                        and all(isinstance(state, bool) for state in overrides.values())
                    ):
                        errors.append(
                            "snapshot write_plan {}.{} overrides 无效".format(label, field)
                        )
    overrides_snapshot = snapshot.get("overrides")
    if overrides_snapshot is not None and not isinstance(overrides_snapshot, Mapping):
        errors.append("snapshot overrides 必须是对象")
    fill_snapshot = snapshot.get("transient_fill_lighting")
    if (
        not isinstance(fill_snapshot, Mapping)
        or type(fill_snapshot.get("enabled")) is not bool
        or not isinstance(fill_snapshot.get("actor_labels"), list)
        or any(
            not isinstance(label, str) or not label.startswith(VENUE_FILL_LABEL_PREFIX)
            for label in fill_snapshot.get("actor_labels", [])
        )
    ):
        errors.append("snapshot transient_fill_lighting 无效")
    fill_parameters = snapshot.get("venue_fill_parameters")
    if not isinstance(fill_parameters, Mapping) or not isinstance(
        fill_parameters.get("enabled"), bool
    ):
        errors.append("snapshot venue_fill_parameters 无效")
    manual_spot = snapshot.get("manual_spot_lighting")
    if (
        not isinstance(manual_spot, Mapping)
        or manual_spot.get("actor_labels") != list(MANUAL_SPOTLIGHT_ACTOR_LABELS)
        or not isinstance(manual_spot.get("baseline"), Mapping)
        or set(manual_spot.get("baseline") or {})
        != {"intensity", "intensity_units"}
        or manual_spot.get("baseline", {}).get("intensity_units") != "ev"
    ):
        errors.append("snapshot manual_spot_lighting 无效")
    if _contains_local_path(snapshot):
        errors.append("snapshot 不得包含本机绝对路径")
    return list(dict.fromkeys(errors))


def build_environment_application_report(
    manifest: Mapping,
    transaction_id: str,
    state: str,
    actors: Sequence[Mapping],
    errors: Sequence[str],
    warnings: Sequence[str],
) -> dict:
    """建立不含本机路径的 episode environment application report。"""
    if state not in REPORT_STATES:
        raise ValueError("environment application state 无效")
    actor_records = [deepcopy(dict(actor)) for actor in actors]
    report = {
        "schema": APPLICATION_REPORT_SCHEMA,
        "version": REPORT_VERSION,
        "state": state,
        "episode_id": manifest.get("episode_id"),
        "environment_manifest_fingerprint": manifest.get("fingerprint"),
        "catalog_fingerprint": manifest.get("catalog_fingerprint"),
        "trajectory_sha256": manifest.get("trajectory_sha256"),
        "transaction_id": transaction_id,
        "transaction": {"transaction_id": transaction_id},
        "preset": manifest.get("preset"),
        "venue_fill_lighting": {
            "enabled": bool(
                (manifest.get("parameters", {}).get("venue_fill_lighting") or {}).get("enabled")
            ),
            "light_count": len(
                (manifest.get("parameters", {}).get("venue_fill_lighting") or {}).get("locations_cm") or []
            ),
            "restore_status": "not_started",
            "actor_labels": [],
        },
        "venue_spot_lighting": {
            "enabled": True,
            "intensity": (manifest.get("parameters", {}).get("venue_spot_lighting") or {}).get("intensity"),
            "intensity_units": "ev",
            "actor_labels": list(MANUAL_SPOTLIGHT_ACTOR_LABELS),
            "restore_status": "not_started",
        },
        "actors": actor_records,
        "render_result": None,
        "restore_status": "not_started",
        "errors": [str(error) for error in errors],
        "warnings": [str(warning) for warning in warnings],
    }
    validation_errors = validate_environment_application_report(report, manifest)
    if validation_errors:
        raise ValueError("environment application report 无效: {}".format("; ".join(validation_errors)))
    return report


def validate_environment_application_report(report: Mapping, manifest: Mapping) -> list[str]:
    """验证 report 状态、transaction/fingerprint 和结构化记录。"""
    if not isinstance(report, Mapping):
        return ["environment application report 必须是对象"]
    errors = []
    required = {
        "schema", "version", "state", "episode_id",
        "environment_manifest_fingerprint", "catalog_fingerprint",
        "trajectory_sha256", "transaction_id", "preset", "actors",
        "transaction", "render_result", "restore_status", "errors", "warnings",
        "venue_fill_lighting",
        "venue_spot_lighting",
    }
    missing = required - set(report)
    if missing:
        errors.append("report 缺少字段: {}".format(", ".join(sorted(missing))))
    if report.get("schema") != APPLICATION_REPORT_SCHEMA:
        errors.append("report schema 无效")
    if type(report.get("version")) is not int or report.get("version") != REPORT_VERSION:
        errors.append("report version 无效")
    if report.get("state") not in REPORT_STATES:
        errors.append("report state 无效")
    for field, expected in (
        ("episode_id", manifest.get("episode_id")),
        ("environment_manifest_fingerprint", manifest.get("fingerprint")),
        ("catalog_fingerprint", manifest.get("catalog_fingerprint")),
        ("trajectory_sha256", manifest.get("trajectory_sha256")),
        ("preset", manifest.get("preset")),
    ):
        if report.get(field) != expected:
            errors.append("report {} 与 manifest 不匹配".format(field))
    if not isinstance(report.get("transaction_id"), str) or not report.get("transaction_id"):
        errors.append("report transaction_id 无效")
    transaction = report.get("transaction")
    if not isinstance(transaction, Mapping) or transaction.get("transaction_id") != report.get("transaction_id"):
        errors.append("report transaction identity drift")
    actors = report.get("actors")
    if not isinstance(actors, list):
        errors.append("report actors 必须是数组")
    else:
        labels = []
        for actor in actors:
            if not isinstance(actor, Mapping):
                errors.append("report actor record 必须是对象")
                continue
            label = actor.get("actor_label")
            if label not in ACTOR_CLASSES:
                errors.append("report actor_label 无效")
            labels.append(label)
            if actor.get("class") != ACTOR_CLASSES.get(label):
                errors.append("report {} actor class 无效".format(label))
            for key in ("actor_path", "package_path"):
                if not isinstance(actor.get(key), str) or not actor.get(key):
                    errors.append("report actor {} 缺少 {}".format(label, key))
            if actor.get("apply_status") not in {"not_started", "applied", "failed"}:
                errors.append("report {} apply_status 无效".format(label))
            if actor.get("restore_status") not in {"not_started", "restored", "failed"}:
                errors.append("report {} restore_status 无效".format(label))
        if len(labels) != len(set(labels)):
            errors.append("report actors 含重复 actor identity")
        if set(labels) != set(ACTOR_CLASSES):
            errors.append("report actors 必须精确覆盖六类 environment Actor")
    for field in ("errors", "warnings"):
        values = report.get(field)
        if not isinstance(values, list) or any(not isinstance(item, str) or not item for item in values):
            errors.append("report {} 必须是非空字符串数组".format(field))
    if report.get("restore_status") not in {"not_started", "restored", "failed"}:
        errors.append("report restore_status 无效")
    venue_fill = report.get("venue_fill_lighting")
    if (
        not isinstance(venue_fill, Mapping)
        or not isinstance(venue_fill.get("enabled"), bool)
        or type(venue_fill.get("light_count")) is not int
        or venue_fill.get("light_count") < 0
        or venue_fill.get("restore_status") not in {"not_started", "restored", "failed"}
    ):
        errors.append("report venue_fill_lighting 状态无效")
    elif "actor_labels" in venue_fill and (
        not isinstance(venue_fill.get("actor_labels"), list)
        or any(
            not isinstance(label, str) or not label.startswith(VENUE_FILL_LABEL_PREFIX)
            for label in venue_fill.get("actor_labels", [])
        )
    ):
        errors.append("report venue_fill_lighting actor_labels 无效")
    venue_spot = report.get("venue_spot_lighting")
    if (
        not isinstance(venue_spot, Mapping)
        or venue_spot.get("enabled") is not True
        or venue_spot.get("intensity_units") != "ev"
        or venue_spot.get("actor_labels") != list(MANUAL_SPOTLIGHT_ACTOR_LABELS)
        or not isinstance(venue_spot.get("intensity"), (int, float))
        or venue_spot.get("restore_status") not in {"not_started", "restored", "failed"}
    ):
        errors.append("report venue_spot_lighting 状态无效")
    if _contains_local_path(report):
        errors.append("report 不得包含本机绝对路径")
    return list(dict.fromkeys(errors))


def apply_environment_readback_plan(
    manifest: Mapping, snapshot: Mapping, actual_values: Mapping
) -> dict:
    """逐项比较 manifest 写入字段的真实 readback，任何差异都关闭事务。"""
    errors = []
    evidence = {}
    expected_groups = _expected_field_map(manifest)
    if validate_environment_snapshot(snapshot):
        errors.extend(validate_environment_snapshot(snapshot))
    if not isinstance(actual_values, Mapping):
        actual_values = {}
    for label, fields in expected_groups.items():
        actual_group = actual_values.get(PARAMETER_GROUPS[label])
        evidence[label] = {}
        for field, expected in fields.items():
            if not isinstance(actual_group, Mapping) or field not in actual_group:
                errors.append("{} {} readback 缺失".format(label, field))
                continue
            actual = deepcopy(actual_group[field])
            tolerance = _field_tolerance(field)
            evidence[label][field] = {
                "expected": deepcopy(expected),
                "actual": actual,
                "matches": _values_match(expected, actual, tolerance=tolerance),
            }
            if not evidence[label][field]["matches"]:
                errors.append("{} {} readback 与期望不一致".format(label, field))
    return {
        "ok": not errors,
        "state": "applied" if not errors else "restore_failed",
        "evidence": evidence,
        "errors": list(dict.fromkeys(errors)),
    }


def verify_environment_baseline_readback(
    snapshot: Mapping, actual_values: Mapping
) -> dict:
    """按 snapshot baseline 而非 manifest 校验 restore 结果。"""
    errors = []
    evidence = {}
    if not isinstance(actual_values, Mapping):
        actual_values = {}
    plan = snapshot.get("write_plan") or {}
    baseline = snapshot.get("baseline") or {}
    for label, fields in plan.items():
        expected_group = baseline.get(label) or {}
        actual_group = actual_values.get(PARAMETER_GROUPS[label])
        evidence[label] = {}
        for field in fields:
            expected = deepcopy(expected_group.get(field))
            actual = deepcopy(actual_group.get(field)) if isinstance(actual_group, Mapping) else None
            matches = _values_match(
                expected, actual, tolerance=_field_tolerance(field)
            )
            evidence[label][field] = {
                "expected": expected,
                "actual": actual,
                "matches": matches,
            }
            if not matches:
                errors.append("{} {} baseline readback 不一致".format(label, field))
    manual_baseline = (snapshot.get("manual_spot_lighting") or {}).get("baseline") or {}
    actual_spot = actual_values.get("venue_spot_lighting")
    actual_spot = actual_spot if isinstance(actual_spot, Mapping) else {}
    for field in ("intensity", "intensity_units"):
        expected = manual_baseline.get(field)
        actual = actual_spot.get(field)
        matches = _values_match(expected, actual)
        evidence.setdefault("ManualSpotLighting", {})[field] = {
            "expected": deepcopy(expected),
            "actual": deepcopy(actual),
            "matches": matches,
        }
        if not matches:
            errors.append("ManualSpotLighting {} baseline readback 不一致".format(field))
    return {
        "ok": not errors,
        "state": "restored" if not errors else "restore_failed",
        "evidence": evidence,
        "errors": errors,
    }


def build_environment_property_write_plan(
    manifest: Mapping, diagnostic_report: Mapping
) -> dict:
    """把 Task 3 read evidence 映射为可写计划；只读 API 不视为写入证据。"""
    expected = _expected_field_map(manifest)
    actor_records = diagnostic_report.get("actors")
    errors = []
    plan = {}
    if not isinstance(actor_records, Mapping):
        return {"ok": False, "plan": {}, "errors": ["diagnostic actors 无效"]}
    for label, values in expected.items():
        actor_record = actor_records.get(label)
        if not isinstance(actor_record, Mapping):
            errors.append("{} diagnostic actor 缺失".format(label))
            continue
        discovered = actor_record.get("discovered_properties")
        readbacks = actor_record.get("baseline_readback")
        plan[label] = {}
        for field, value in values.items():
            mapping = discovered.get(field) if isinstance(discovered, Mapping) else None
            baseline = readbacks.get(field) if isinstance(readbacks, Mapping) else None
            if not isinstance(mapping, Mapping) or not isinstance(baseline, Mapping):
                errors.append("{} {} 缺少 property mapping/baseline evidence".format(label, field))
                continue
            if "readback" not in baseline or not baseline.get("evidence_source"):
                errors.append("{} {} baseline evidence 无效".format(label, field))
                continue
            if (
                mapping.get("operation")
                not in {"get_editor_property", "get_scalar_parameter_value", "read"}
                or mapping.get("target")
                not in {"component", "settings", "material", "actor"}
                or not isinstance(mapping.get("property_name"), str)
                or not mapping.get("property_name")
                or mapping.get("evidence_source") != baseline.get("evidence_source")
                or mapping.get("evidence_source")
                not in {
                    "get_editor_property",
                    "get_scalar_parameter_value",
                    "get_actor_rotation",
                }
            ):
                errors.append(
                    "MANUAL_REQUIRED: {} {} 没有已验证 UE write mapping".format(
                        label, field
                    )
                )
                continue
            entry = {
                "field": field,
                "target": mapping["target"],
                "property_name": mapping["property_name"],
                "evidence_source": mapping["evidence_source"],
                "baseline": deepcopy(baseline["readback"]),
                "value": deepcopy(value),
            }
            overrides = baseline.get("overrides")
            if isinstance(overrides, Mapping) and overrides:
                entry["overrides"] = {
                    str(name): bool(state) for name, state in overrides.items()
                }
                entry["override_value"] = True
            plan[label][field] = entry
    _plan_duplicate_target_errors(plan, errors)
    return {"ok": not errors, "plan": plan, "errors": errors}


def _plan_duplicate_target_errors(plan: Mapping, errors: list) -> None:
    """同一 UE 属性只能被一个 semantic field 写入，避免静默覆盖。"""
    for label, fields in plan.items():
        seen = {}
        for field, entry in fields.items():
            key = (entry.get("target"), entry.get("property_name"))
            previous_field = seen.get(key)
            if previous_field is not None:
                previous = fields[previous_field]
                if not _values_match(
                    previous.get("value"),
                    entry.get("value"),
                    tolerance=_field_tolerance(field),
                ):
                    errors.append(
                        "MANUAL_REQUIRED: {} {}/{} 映射到同一 UE 属性但值冲突".format(
                            label, previous_field, field
                        )
                    )
            seen[key] = field


def _application_actor_records(actors: Mapping) -> list:
    records = []
    for label in ACTOR_CLASSES:
        source = actors[label]
        records.append(
            {
                "actor_label": label,
                "actor_path": source["actor_path"],
                "class": source["class"],
                "package_path": source["package_path"],
                "apply_status": "not_started",
                "restore_status": "not_started",
            }
        )
    return records


def _disabled_fill_lighting_readback(config: Mapping) -> dict:
    result = deepcopy(dict(config))
    result["enabled"] = False
    result["intensity"] = 0.0
    result["locations_cm"] = []
    return result


def _verify_baseline_and_flush(transport: Any, snapshot: Mapping) -> list:
    errors = []
    try:
        readbacks = transport.read_all(snapshot)
        baseline_check = verify_environment_baseline_readback(snapshot, readbacks)
        if not baseline_check["ok"]:
            return list(baseline_check["errors"])
        flush = getattr(transport, "flush_dirty", None)
        if callable(flush):
            errors.extend(
                "{} dirty flag 清除失败".format(package)
                for package in flush()
            )
        clean = transport.read_clean_state(snapshot)
        if clean != snapshot.get("clean_baseline"):
            errors.append("baseline verified 后 package clean state mismatch")
    except Exception as exc:
        errors.append("baseline verify/dirty flush failed: {}".format(exc))
    return errors


def _execute_apply_transaction(
    manifest: Mapping,
    diagnostic_report: Mapping,
    snapshot_path: Path,
    report_path: Path,
    transport: Any,
) -> dict:
    """通过显式 transport 执行并逐项回读；供真实 adapter 和纯合同测试复用。"""
    actors = _diagnostic_actors(diagnostic_report)
    baseline = _diagnostic_baseline(actors)
    plan_result = build_environment_property_write_plan(manifest, diagnostic_report)
    if not plan_result["ok"]:
        return {
            "ok": False,
            "operation": "apply",
            "state": "preflight_failed",
            "status": "MANUAL_REQUIRED",
            "errors": plan_result["errors"],
            "warnings": [],
            "snapshot_written": False,
            "property_writes": 0,
        }
    clean_packages = diagnostic_report.get("package_dirty")
    level = {
        "path": diagnostic_report.get("current_level"),
        "package_path": diagnostic_report.get("current_level"),
        "class": "World",
    }
    transaction_id = transport.new_transaction_id()
    overrides = {
        label: {
            field: deepcopy(entry["overrides"])
            for field, entry in fields.items()
            if entry.get("overrides")
        }
        for label, fields in plan_result["plan"].items()
    }
    venue_fill_parameters = deepcopy(
        manifest["parameters"]["venue_fill_lighting"]
    )
    manual_spot_parameters = _manual_spotlight_parameters(manifest)
    manual_spot_snapshot = _diagnostic_manual_spot_lighting(diagnostic_report)
    snapshot = build_environment_snapshot(
        level,
        actors,
        baseline,
        manifest,
        transaction_id,
        write_plan=plan_result["plan"],
        overrides=overrides,
        manual_spot_lighting=manual_spot_snapshot,
    )
    snapshot["venue_fill_parameters"] = venue_fill_parameters
    report = build_environment_application_report(
        manifest,
        transaction_id,
        "not_started",
        _application_actor_records(actors),
        [],
        [],
    )
    if clean_packages:
        return {
            "ok": False,
            "operation": "apply",
            "state": "preflight_failed",
            "status": "MANUAL_REQUIRED",
            "errors": ["存在 dirty package，MANUAL_REQUIRED"],
            "warnings": [],
            "snapshot_written": False,
            "property_writes": 0,
        }
    _write_json(snapshot_path, snapshot)
    _write_json(report_path, report)
    writes = 0
    touched = []
    fill_labels = []
    touched_spotlights = []
    try:
        track_actors = getattr(transport, "track_environment_actors", None)
        if callable(track_actors):
            track_actors()
        create_fill = getattr(transport, "create_fill_lighting", None)
        if venue_fill_parameters.get("enabled") and callable(create_fill):
            label_factory = getattr(transport, "fill_lighting_actor_labels", None)
            if not callable(label_factory):
                raise RuntimeError("venue fill lighting identity API 缺失")
            fill_labels = list(
                label_factory(
                    transaction_id,
                    len(venue_fill_parameters.get("locations_cm") or []),
                )
            )
            snapshot["transient_fill_lighting"] = {
                "enabled": True,
                "actor_labels": fill_labels,
            }
            _write_json(snapshot_path, snapshot)
            try:
                created_labels = list(
                    create_fill(venue_fill_parameters, transaction_id)
                )
            except Exception:
                release_fill = getattr(transport, "release_fill_lighting", None)
                if callable(release_fill):
                    release_fill(fill_labels)
                    snapshot["transient_fill_lighting"] = {
                        "enabled": False,
                        "actor_labels": [],
                    }
                    _write_json(snapshot_path, snapshot)
                raise
            if created_labels != fill_labels:
                raise RuntimeError("venue fill light identity 与预先记录不一致")
            if len(created_labels) != len(venue_fill_parameters.get("locations_cm") or []):
                raise RuntimeError("venue fill light 数量与 locations_cm 不匹配")
        for label, fields in plan_result["plan"].items():
            for field, entry in fields.items():
                entry = dict(entry)
                entry["field"] = field
                transport.write(label, entry, entry["value"])
                writes += 1
                touched.append((label, field, entry))
                actual = transport.read(label, entry)
                if not _values_match(
                    entry["value"], actual, tolerance=_field_tolerance(field)
                ):
                    raise RuntimeError(
                        "{} {} readback mismatch".format(label, field)
                    )
        apply_manual_spotlights = getattr(
            transport, "apply_manual_spot_lighting", None
        )
        if not callable(apply_manual_spotlights):
            raise RuntimeError("manual SpotLight apply API 缺失")
        touched_spotlights = list(
            apply_manual_spotlights(manual_spot_parameters)
        )
        if touched_spotlights != list(MANUAL_SPOTLIGHT_ACTOR_LABELS):
            raise RuntimeError("manual SpotLight apply identity/readback 不完整")
        _commit_transport(transport)
        if fill_labels:
            assert_transient_fill = getattr(
                transport, "assert_transient_fill_lighting", None
            )
            if not callable(assert_transient_fill):
                raise RuntimeError("venue fill lighting transient verification API 缺失")
            assert_transient_fill(fill_labels)
            actual_fill = transport.read_fill_lighting(fill_labels)
            expected_fill = deepcopy(venue_fill_parameters)
            actual_for_compare = deepcopy(actual_fill)
            expected_fill.pop("locations_cm", None)
            actual_for_compare.pop("locations_cm", None)
            locations_match = _values_match(
                venue_fill_parameters.get("locations_cm"),
                actual_fill.get("locations_cm"),
                tolerance=0.1,
            )
            if not _values_match(
                expected_fill, actual_for_compare, tolerance=_COLOR_TOLERANCE
            ) or not locations_match:
                raise RuntimeError("venue fill lighting readback mismatch")
        spot_readback = transport.read_manual_spot_lighting(
            MANUAL_SPOTLIGHT_ACTOR_LABELS
        )
        expected_spot = {
            "enabled": True,
            "intensity": manual_spot_parameters["intensity"],
            "intensity_units": manual_spot_parameters["intensity_units"],
            "actor_labels": list(MANUAL_SPOTLIGHT_ACTOR_LABELS),
        }
        if not _values_match(expected_spot, spot_readback):
            raise RuntimeError("manual SpotLight readback mismatch")
        report["state"] = "applied"
        report["venue_fill_lighting"] = {
            "enabled": bool(fill_labels),
            "light_count": len(fill_labels),
            "restore_status": "not_started",
            "readback": (
                transport.read_fill_lighting(fill_labels)
                if fill_labels
                else _disabled_fill_lighting_readback(venue_fill_parameters)
            ),
        }
        if fill_labels:
            report["venue_fill_lighting"]["actor_labels"] = fill_labels
        report["venue_spot_lighting"].update(spot_readback)
        report["venue_spot_lighting"]["restore_status"] = "not_started"
        for actor in report["actors"]:
            actor["apply_status"] = "applied"
        _write_json(report_path, report)
        return {
            "ok": True,
            "operation": "apply",
            "state": "applied",
            "status": None,
            "errors": [],
            "warnings": [],
            "snapshot_written": True,
            "property_writes": writes,
            "snapshot": snapshot,
            "report": report,
        }
    except Exception as exc:
        restore_errors = []
        release_fill = getattr(transport, "release_fill_lighting", None)
        if callable(release_fill) and fill_labels:
            try:
                release_fill(fill_labels)
                restore_fill_packages = getattr(
                    transport, "restore_tracked_package_baseline", None
                )
                if callable(restore_fill_packages):
                    restore_fill_packages()
                snapshot["transient_fill_lighting"] = {
                    "enabled": False,
                    "actor_labels": [],
                }
                _write_json(snapshot_path, snapshot)
            except Exception as fill_exc:
                restore_errors.append("venue fill lighting cleanup failed: {}".format(fill_exc))
        for label, field, entry in reversed(touched):
            try:
                transport.write(label, entry, entry["baseline"])
                actual = transport.read(label, entry)
                if not _values_match(
                    entry["baseline"], actual, tolerance=_field_tolerance(field)
                ):
                    restore_errors.append("{} {} baseline readback mismatch".format(label, field))
            except Exception as restore_exc:
                restore_errors.append(
                    "{} {} baseline restore failed: {}".format(label, field, restore_exc)
                )
        restore_spotlights = getattr(
            transport, "restore_manual_spot_lighting", None
        )
        if callable(restore_spotlights) and touched_spotlights:
            try:
                restore_spotlights(snapshot["manual_spot_lighting"])
            except Exception as restore_exc:
                restore_errors.append(
                    "manual SpotLight baseline restore failed: {}".format(restore_exc)
                )
        try:
            _commit_transport(transport)
        except Exception as commit_exc:
            restore_errors.append("environment commit failed: {}".format(commit_exc))
        restore_errors.extend(_verify_baseline_and_flush(transport, snapshot))
        message = "environment apply/readback failed: {}".format(exc)
        rollback_ok = not restore_errors
        report["state"] = "restored" if rollback_ok else "restore_failed"
        report["restore_status"] = "restored" if rollback_ok else "failed"
        if isinstance(report.get("venue_fill_lighting"), dict):
            report["venue_fill_lighting"]["restore_status"] = (
                "restored" if rollback_ok else "failed"
            )
            if rollback_ok:
                report["venue_fill_lighting"]["enabled"] = False
                report["venue_fill_lighting"]["actor_labels"] = []
        if isinstance(report.get("venue_spot_lighting"), dict):
            report["venue_spot_lighting"]["restore_status"] = (
                "restored" if rollback_ok else "failed"
            )
        report["errors"] = [message] + restore_errors
        for actor in report["actors"]:
            actor["restore_status"] = "restored" if rollback_ok else "failed"
        _write_json(report_path, report)
        return {
            "ok": False,
            "operation": "apply",
            "state": "restored" if rollback_ok else "restore_failed",
            "status": None if rollback_ok else "MANUAL_REQUIRED",
            "errors": report["errors"],
            "warnings": [],
            "snapshot_written": True,
            "property_writes": writes,
            "snapshot": snapshot,
            "report": report,
        }


def _load_unreal():
    import unreal

    return unreal


def _commit_transport(transport: Any) -> None:
    commit = getattr(transport, "commit", None)
    if callable(commit):
        commit()


def _semantic_value(label: str, field: str, raw: Any) -> Any:
    from diagnose_episode_environment import _semantic_readback

    return _semantic_readback(label, field, raw)


class UEEnvironmentTransport:
    """把 environment 写入计划映射到 UE 5.8 Editor 的已验证只读/写入 API。

    动态材质实例只在本进程内存在，restore 时恢复原材质引用；External Actor
    package 的 dirty flag 在 restore 后显式清除，避免留下未保存修改。
    """

    _active_fill_lights = {}

    def __init__(self, unreal_module: Any = None, level_path: str = FORMAL_MAP):
        self._u = unreal_module or _load_unreal()
        self._level_path = level_path
        self._actors = None
        self._mid = None
        self._material_parent = None
        self._touched = {}
        self._touched_labels = set()
        self._fill_lights = {}
        self._package_baseline = None

    def _package_readback(self, actor: Any) -> Any:
        try:
            return self._u.SystemLibrary.get_outermost(actor)
        except Exception:
            return actor.get_package()

    # ---- identity -------------------------------------------------
    def new_transaction_id(self) -> str:
        return "env_tx_" + uuid.uuid4().hex

    def _resolve_actors(self) -> dict:
        if self._actors is not None:
            return self._actors
        subsystem = self._u.get_editor_subsystem(self._u.EditorActorSubsystem)
        all_actors = list(subsystem.get_all_level_actors())
        resolved = {}
        for label, expected_class in ACTOR_CLASSES.items():
            matches = [actor for actor in all_actors if actor.get_actor_label() == label]
            if len(matches) != 1:
                raise RuntimeError(
                    "{} Actor 必须唯一，实际 {}".format(label, len(matches))
                )
            actor = matches[0]
            actual_class = actor.get_class().get_name()
            if actual_class != expected_class:
                raise RuntimeError(
                    "{} Actor class 不匹配: {}".format(label, actual_class)
                )
            resolved[label] = actor
        self._actors = resolved
        return resolved

    def _manual_spot_actors(self, labels: Sequence[str] = MANUAL_SPOTLIGHT_ACTOR_LABELS) -> dict:
        expected = list(labels)
        if expected != list(MANUAL_SPOTLIGHT_ACTOR_LABELS):
            raise RuntimeError("manual SpotLight identity manifest 不匹配")
        subsystem = self._u.get_editor_subsystem(self._u.EditorActorSubsystem)
        actors = list(subsystem.get_all_level_actors())
        resolved = {}
        for label in expected:
            matches = [actor for actor in actors if actor.get_actor_label() == label]
            if len(matches) != 1:
                raise RuntimeError("{} Actor 必须唯一，实际 {}".format(label, len(matches)))
            actor = matches[0]
            if actor.get_class().get_name() != "SpotLight":
                raise RuntimeError("{} Actor class 必须为 SpotLight".format(label))
            components = list(actor.get_components_by_class(self._u.SpotLightComponent))
            if len(components) != 1:
                raise RuntimeError("{} SpotLightComponent 必须唯一".format(label))
            resolved[label] = (actor, components[0])
        return resolved

    def read_manual_spot_lighting(
        self, labels: Sequence[str] = MANUAL_SPOTLIGHT_ACTOR_LABELS
    ) -> dict:
        actors = self._manual_spot_actors(labels)
        intensities = []
        units = []
        for _actor, component in actors.values():
            intensities.append(float(component.get_editor_property("intensity")))
            raw_units = component.get_editor_property("intensity_units")
            unit_name = getattr(raw_units, "name", None)
            if not unit_name:
                unit_name = str(raw_units).split(".")[-1].split(":", 1)[0].strip(" >")
            units.append(str(unit_name).lower())
        if len(set(intensities)) != 1:
            raise RuntimeError("manual SpotLight intensity 不一致")
        if len(set(units)) != 1 or units[0] != "ev":
            raise RuntimeError("manual SpotLight intensity_units 必须统一为 EV")
        return {
            "enabled": True,
            "intensity": intensities[0],
            "intensity_units": units[0],
            "actor_labels": list(actors),
        }

    def apply_manual_spot_lighting(self, config: Mapping) -> list:
        if not isinstance(config, Mapping) or config.get("enabled") is not True:
            raise ValueError("manual SpotLight config 必须 enabled")
        if config.get("intensity_units") != "ev":
            raise ValueError("manual SpotLight intensity_units 必须为 ev")
        if config.get("actor_labels") != list(MANUAL_SPOTLIGHT_ACTOR_LABELS):
            raise ValueError("manual SpotLight actor_labels 不匹配")
        intensity = float(config.get("intensity"))
        if not math.isfinite(intensity) or intensity <= 0.0:
            raise ValueError("manual SpotLight intensity 无效")
        actors = self._manual_spot_actors(config["actor_labels"])
        original = {
            label: {
                "intensity": float(component.get_editor_property("intensity")),
                "intensity_units": component.get_editor_property("intensity_units"),
            }
            for label, (_actor, component) in actors.items()
        }
        touched = []
        try:
            for label, (actor, component) in actors.items():
                component.set_editor_property("intensity_units", self._u.LightUnits.EV)
                component.set_editor_property("intensity", intensity)
                touched.append(label)
                self._touch(actor)
                if not _values_match(intensity, float(component.get_editor_property("intensity"))):
                    raise RuntimeError("{} SpotLight intensity readback mismatch".format(label))
        except Exception:
            for label in reversed(touched):
                _actor, component = actors[label]
                component.set_editor_property("intensity_units", original[label]["intensity_units"])
                component.set_editor_property("intensity", original[label]["intensity"])
            raise
        return list(actors)

    def restore_manual_spot_lighting(self, snapshot: Mapping) -> None:
        if not isinstance(snapshot, Mapping):
            raise ValueError("manual SpotLight snapshot 无效")
        labels = snapshot.get("actor_labels")
        baseline = snapshot.get("baseline")
        if labels != list(MANUAL_SPOTLIGHT_ACTOR_LABELS) or not isinstance(baseline, Mapping):
            raise ValueError("manual SpotLight snapshot identity/baseline 无效")
        if baseline.get("intensity_units") != "ev":
            raise ValueError("manual SpotLight baseline units 必须为 ev")
        actors = self._manual_spot_actors(labels)
        for label, (actor, component) in actors.items():
            component.set_editor_property("intensity_units", self._u.LightUnits.EV)
            component.set_editor_property("intensity", float(baseline["intensity"]))
            self._touch(actor)
            if not _values_match(
                float(baseline["intensity"]),
                float(component.get_editor_property("intensity")),
            ):
                raise RuntimeError("{} SpotLight baseline readback mismatch".format(label))

    def track_environment_actors(self) -> None:
        actors = self._resolve_actors()
        packages = {
            self._package_path(self._package_readback(actor)): actor
            for actor in actors.values()
        }
        spot_actors = self._manual_spot_actors()
        packages.update(
            {
                self._package_path(self._package_readback(actor)): actor
                for actor, _component in spot_actors.values()
            }
        )
        self._touched.update(packages)
        self._package_baseline = set(packages)

    def _component(self, label: str) -> Any:
        actor = self._resolve_actors()[label]
        if label == "PostProcessVolume":
            return actor
        component_class = getattr(self._u, "{}Component".format(label), None)
        if component_class is None:
            raise RuntimeError("UE 未暴露 {}Component".format(label))
        components = list(actor.get_components_by_class(component_class))
        if len(components) != 1:
            raise RuntimeError(
                "{} component 必须唯一，实际 {}".format(label, len(components))
            )
        return components[0]

    def current_identity(self) -> dict:
        actors = {}
        for label, actor in self._resolve_actors().items():
            actors[label] = {
                "actor_label": label,
                "actor_path": actor.get_path_name(),
                "class": actor.get_class().get_name(),
                "package_path": self._package_path(
                    self._package_readback(actor)
                ),
                "match_count": 1,
            }
        level = {
            "path": self.current_level_path(),
            "package_path": self.current_level_path(),
            "class": "World",
        }
        return {"level": level, "actors": actors}

    def current_level_path(self) -> Optional[str]:
        world = self._u.EditorLevelLibrary.get_editor_world()
        return self._package_path(world)

    def _package_path(self, value: Any) -> Optional[str]:
        if value is None:
            return None
        getter = getattr(value, "get_path_name", None)
        path = str(getter()) if callable(getter) else str(value)
        return path.split(".", 1)[0] if path else None

    # ---- value conversion -----------------------------------------
    def _ue_value(self, field: str, value: Any) -> Any:
        if field in _COLOR_FIELDS:
            channels = [min(255, max(0, int(round(float(channel) * 255.0)))) for channel in value]
            # UE 的 Color 位置参数顺序是 (b, g, r, a)；必须用关键字参数避免通道互换。
            return self._u.Color(
                r=channels[0], g=channels[1], b=channels[2], a=channels[3]
            )
        if field in ("color_contrast", "color_saturation"):
            return self._u.Vector4(
                float(value[0]), float(value[1]), float(value[2]), float(value[3])
            )
        if field == "source_type":
            if value != _SKYPORT_SOURCE_TYPE:
                raise RuntimeError("未知 SkyLight source_type: {}".format(value))
            # 语义值同时覆盖 captured scene 与 specified cubemap；正式地图使用
            # captured scene，写入该枚举既满足语义又不引入空 cubemap。
            return self._u.SkyLightSourceType.SLS_CAPTURED_SCENE
        return value

    # ---- material -------------------------------------------------
    def _ensure_material(self, component: Any) -> None:
        if self._mid is not None:
            return
        current = component.get_editor_property("material")
        if current is None:
            raise RuntimeError("VolumetricCloud 缺少可继承的父材质")
        if current.get_class().get_name() == "MaterialInstanceDynamic":
            # 同一 Editor 会话内 apply 后重新进入 restore 时，MID 仍挂在组件上；
            # 直接复用并从 MID 的 parent 恢复原始材质，避免二次嵌套。
            self._mid = current
            self._material_parent = current.get_editor_property("parent")
            if self._material_parent is None:
                raise RuntimeError("MaterialInstanceDynamic 缺少 parent，无法恢复")
            return
        world = self._u.EditorLevelLibrary.get_editor_world()
        mid = self._u.MaterialLibrary.create_dynamic_material_instance(world, current)
        if mid is None:
            raise RuntimeError("无法创建 VolumetricCloud 动态材质实例")
        component.set_editor_property("material", mid)
        self._mid = mid
        self._material_parent = current

    def release_material(self) -> None:
        if self._mid is None:
            return
        component = self._component("VolumetricCloud")
        component.set_editor_property("material", self._material_parent)
        self._mid = None
        self._material_parent = None

    # ---- write / read ---------------------------------------------
    def _touch(self, actor: Any) -> None:
        package = self._package_path(self._package_readback(actor))
        if package:
            self._touched[package] = actor
        label = actor.get_actor_label()
        if label:
            self._touched_labels.add(label)

    def write(self, label: str, entry: Mapping, value: Any) -> None:
        field = entry.get("field") or entry.get("property_name")
        target = entry.get("target")
        actor = self._resolve_actors()[label]
        if target == "actor" and field in _ROTATION_FIELDS:
            rotation = self._u.Rotator(
                roll=float(value[2]), pitch=float(value[0]), yaw=float(value[1])
            )
            if actor.set_actor_rotation(rotation, False) is False:
                raise RuntimeError("{} rotation 写入失败".format(label))
            self._touch(actor)
            return
        if target == "actor":
            actor.set_editor_property(
                entry["property_name"], self._ue_value(field, value)
            )
            self._touch(actor)
            return
        if target == "material":
            component = self._component(label)
            self._ensure_material(component)
            self._mid.set_scalar_parameter_value(
                entry["property_name"], float(value)
            )
            self._touch(actor)
            return
        component = self._component(label)
        if target == "settings":
            settings = component.get_editor_property("settings")
            settings.set_editor_property(
                entry["property_name"], self._ue_value(field, value)
            )
            override_name = SETTINGS_OVERRIDE_FLAGS.get(field)
            if override_name is not None:
                settings.set_editor_property(
                    override_name, bool(entry.get("override_value", True))
                )
            component.set_editor_property("settings", settings)
            self._touch(actor)
            return
        component.set_editor_property(
            entry["property_name"], self._ue_value(field, value)
        )
        self._touch(actor)

    def read(self, label: str, entry: Mapping) -> Any:
        field = entry.get("field") or entry.get("property_name")
        target = entry.get("target")
        actor = self._resolve_actors()[label]
        if target == "actor" and field in _ROTATION_FIELDS:
            raw = actor.get_actor_rotation()
        elif target == "actor":
            raw = actor.get_editor_property(entry["property_name"])
        elif target == "material":
            material = self._mid
            if material is None:
                material = self._component(label).get_editor_property("material")
            if material is None or not hasattr(material, "get_scalar_parameter_value"):
                raise RuntimeError("{} material readback 缺少可用材质".format(label))
            raw = material.get_scalar_parameter_value(entry["property_name"])
        elif target == "settings":
            settings = self._component(label).get_editor_property("settings")
            raw = settings.get_editor_property(entry["property_name"])
        else:
            raw = self._component(label).get_editor_property(entry["property_name"])
        return _semantic_value(label, field, raw)

    def read_all(self, snapshot: Mapping) -> dict:
        plan = snapshot.get("write_plan") or {}
        values = {}
        for label in ACTOR_CLASSES:
            values[PARAMETER_GROUPS[label]] = {
                field: self.read(label, entry)
                for field, entry in (plan.get(label) or {}).items()
            }
        try:
            values["venue_spot_lighting"] = self.read_manual_spot_lighting(
                MANUAL_SPOTLIGHT_ACTOR_LABELS
            )
        except RuntimeError:
            # Some unit transports exercise only the six controlled environment Actors.
            if not isinstance(snapshot.get("manual_spot_lighting"), Mapping):
                raise
            raise
        return values

    # ---- lifecycle ------------------------------------------------
    def commit(self) -> None:
        if "SkyLight" in self._touched_labels:
            self._component("SkyLight").recapture_sky()

    def read_clean_state(self, snapshot: Mapping) -> dict:
        packages = sorted(
            str(package.get_name())
            for package in self._u.EditorLoadingAndSavingUtils.get_dirty_map_packages()
        )
        expected_fill = set(
            (snapshot.get("transient_fill_lighting") or {}).get("actor_labels") or []
        )
        expected_manual = set(
            (snapshot.get("manual_spot_lighting") or {}).get("actor_labels") or []
        )
        remaining_fill = {
            label for label in expected_fill if label in self._active_fill_lights
        }
        remaining_fill.update(
            {
            actor.get_actor_label()
            for actor in self._u.get_editor_subsystem(self._u.EditorActorSubsystem).get_all_level_actors()
            if actor.get_actor_label() in expected_fill
            }
        )
        all_actors = self._u.get_editor_subsystem(
            self._u.EditorActorSubsystem
        ).get_all_level_actors()
        actual_manual = {
            actor.get_actor_label()
            for actor in all_actors
            if actor.get_actor_label() in set(MANUAL_SPOTLIGHT_ACTOR_LABELS)
        }
        clean = {
            "level_clean": (
                self.current_level_path() == self._level_path
                and not remaining_fill
            ),
            "packages": packages,
        }
        if expected_manual:
            clean["level_clean"] = clean["level_clean"] and actual_manual == expected_manual
        return clean

    def flush_dirty(self) -> list:
        if not self._touched:
            return []
        current_dirty = {
            self._package_path(package)
            for package in self._u.EditorLoadingAndSavingUtils.get_dirty_map_packages()
        }
        failures = []
        subsystem = self._u.get_editor_subsystem(self._u.EditorAssetSubsystem)
        packages_to_flush = current_dirty or set(self._touched)
        for package in packages_to_flush:
            actor = self._touched.get(package)
            if actor is None:
                failures.append(package)
                continue
            if not self._u.get_editor_subsystem(self._u.EditorAssetSubsystem).set_dirty_flag(
                actor, False
            ):
                failures.append(package)
        return failures

    def fill_lighting_actor_labels(self, transaction_id: str, count: int) -> list:
        suffix = re.sub(r"[^A-Za-z0-9]", "", str(transaction_id))[-10:]
        if not suffix or type(count) is not int or count < 0:
            raise ValueError("venue fill lighting identity 无效")
        return [
            "{}{}_{}".format(VENUE_FILL_LABEL_PREFIX, suffix, index + 1)
            for index in range(count)
        ]

    def create_fill_lighting(
        self, config: Mapping, transaction_id: Optional[str] = None
    ) -> list:
        if not isinstance(config, Mapping) or config.get("enabled") is not True:
            return []
        locations = config.get("locations_cm")
        if not isinstance(locations, list) or not locations:
            raise ValueError("venue fill lighting locations_cm 缺失")
        transaction_id = transaction_id or self.new_transaction_id()
        labels = self.fill_lighting_actor_labels(transaction_id, len(locations))
        subsystem = self._u.get_editor_subsystem(self._u.EditorActorSubsystem)
        actors = list(subsystem.get_all_level_actors())
        if any(actor.get_actor_label() in labels for actor in actors):
            raise RuntimeError("venue fill light label 已存在，拒绝创建重复灯")
        created = []
        try:
            for index, (label, location) in enumerate(zip(labels, locations)):
                position = self._u.Vector(
                    float(location[0]), float(location[1]), float(location[2])
                )
                target = self._u.Vector(0.0, 0.0, -4.0)
                rotation = self._u.MathLibrary.find_look_at_rotation(position, target)
                actor = subsystem.spawn_actor_from_class(
                    self._u.RectLight, position, rotation, transient=True
                )
                if actor is None or actor.get_class().get_name() != "RectLight":
                    raise RuntimeError("venue fill RectLight 创建失败")
                created.append(actor)
                self._fill_lights[label] = actor
                self._active_fill_lights[label] = actor
                actor.set_actor_label(label)
                components = list(
                    actor.get_components_by_class(self._u.RectLightComponent)
                )
                if len(components) != 1:
                    raise RuntimeError("{} RectLightComponent 必须唯一".format(label))
                component = components[0]
                component.set_editor_property("intensity_units", self._u.LightUnits.LUMENS)
                component.set_editor_property("intensity", float(config["intensity"]))
                component.set_editor_property(
                    "attenuation_radius", float(config["attenuation_radius_cm"])
                )
                component.set_editor_property("source_width", float(config["source_width_cm"]))
                component.set_editor_property("source_height", float(config["source_height_cm"]))
                component.set_editor_property(
                    "light_color", self._ue_value("light_color_rgba", config["light_color_rgba"])
                )
                component.set_editor_property("cast_shadows", bool(config["cast_shadows"]))
        except Exception:
            self.release_fill_lighting(labels)
            raise
        return labels

    def read_fill_lighting(self, labels: Sequence[str]) -> dict:
        labels = list(labels or [])
        if not labels:
            return {
                "enabled": False,
                "intensity": 0.0,
                "intensity_units": "lumens",
                "attenuation_radius_cm": 2600.0,
                "source_width_cm": 350.0,
                "source_height_cm": 350.0,
                "light_color_rgba": [1.0, 1.0, 1.0, 1.0],
                "cast_shadows": False,
                "locations_cm": [],
            }
        subsystem = self._u.get_editor_subsystem(self._u.EditorActorSubsystem)
        actors_by_label = {
            label: self._active_fill_lights[label]
            for label in labels
            if label in self._active_fill_lights
        }
        actors_by_label.update(
            {
                actor.get_actor_label(): actor
                for actor in subsystem.get_all_level_actors()
                if actor.get_actor_label() in labels
                and actor.get_actor_label() not in actors_by_label
            }
        )
        if set(actors_by_label) != set(labels):
            raise RuntimeError("venue fill lighting actor 缺失或重复")
        config = None
        locations = []
        for label in labels:
            actor = actors_by_label[label]
            components = list(actor.get_components_by_class(self._u.RectLightComponent))
            if actor.get_class().get_name() != "RectLight" or len(components) != 1:
                raise RuntimeError("{} fill light identity/class 不匹配".format(label))
            component = components[0]
            color = self._semantic_color(component.get_editor_property("light_color"))
            current = {
                "enabled": True,
                "intensity": float(component.get_editor_property("intensity")),
                "intensity_units": str(component.get_editor_property("intensity_units")).split(".")[-1].lower(),
                "attenuation_radius_cm": float(component.get_editor_property("attenuation_radius")),
                "source_width_cm": float(component.get_editor_property("source_width")),
                "source_height_cm": float(component.get_editor_property("source_height")),
                "light_color_rgba": color,
                "cast_shadows": bool(component.get_editor_property("cast_shadows")),
                "locations_cm": [],
            }
            if current["intensity_units"] != "lumens":
                raise RuntimeError("{} fill light intensity units 非 lumens".format(label))
            if config is None:
                config = current
            elif not _values_match(
                {key: value for key, value in current.items() if key != "locations_cm"},
                {key: value for key, value in config.items() if key != "locations_cm"},
                tolerance=_COLOR_TOLERANCE,
            ):
                raise RuntimeError("venue fill lights 参数不一致")
            location = actor.get_actor_location()
            locations.append([float(location.x), float(location.y), float(location.z)])
        config["locations_cm"] = locations
        return config

    def assert_transient_fill_lighting(self, labels: Sequence[str]) -> None:
        labels = set(labels or [])
        if not labels:
            return
        subsystem = self._u.get_editor_subsystem(self._u.EditorActorSubsystem)
        actors = list(subsystem.get_all_level_actors())
        actors_by_label = {}
        for label in labels:
            actor = self._active_fill_lights.get(label)
            if actor is not None:
                actors_by_label[label] = actor
        for actor in actors:
            label = actor.get_actor_label()
            if label in labels and label not in actors_by_label:
                if label in actors_by_label:
                    raise RuntimeError("venue fill light label 重复: {}".format(label))
                actors_by_label[label] = actor
        if set(actors_by_label) != labels:
            raise RuntimeError("venue fill light identity 与预期不一致")
        for label, actor in actors_by_label.items():
            if (
                actor.get_actor_label() != label
                or actor.get_class().get_name() != "RectLight"
                or len(list(actor.get_components_by_class(self._u.RectLightComponent))) != 1
            ):
                raise RuntimeError("{} transient RectLight identity/class 无效".format(label))
        dirty_packages = {
            self._package_path(package)
            for package in self._u.EditorLoadingAndSavingUtils.get_dirty_map_packages()
        }
        dirty_paths = {path for path in dirty_packages if path}
        if dirty_paths - set(self._touched):
            raise RuntimeError(
                "venue fill lighting 后存在范围外 dirty package: {}".format(
                    sorted(dirty_paths - set(self._touched))
                )
            )

    def restore_tracked_package_baseline(self) -> None:
        failures = self.flush_dirty()
        if failures:
            raise RuntimeError(
                "environment External Actor package baseline 恢复失败: {}".format(
                    sorted(failures)
                )
            )
        remaining = self.read_clean_state(
            {"transient_fill_lighting": {"actor_labels": []}}
        )["packages"]
        if remaining:
            raise RuntimeError(
                "environment package clean baseline 恢复失败: {}".format(
                    sorted(remaining)
                )
            )

    def release_fill_lighting(self, labels: Optional[Sequence[str]] = None) -> None:
        labels = list(labels if labels is not None else self._fill_lights.keys())
        subsystem = self._u.get_editor_subsystem(self._u.EditorActorSubsystem)
        actors = list(subsystem.get_all_level_actors())
        by_label = {}
        for actor in actors:
            label = actor.get_actor_label()
            if label in labels:
                if label in by_label:
                    raise RuntimeError("venue fill light label 重复: {}".format(label))
                by_label[label] = actor
        errors = []
        for label in labels:
            actor = by_label.get(label) or self._fill_lights.get(label)
            if actor is None:
                continue
            if actor.get_class().get_name() != "RectLight":
                errors.append("{} actor class 非 RectLight".format(label))
                continue
            if not subsystem.destroy_actor(actor):
                errors.append("{} transient RectLight 销毁失败".format(label))
                continue
            self._fill_lights.pop(label, None)
            self._active_fill_lights.pop(label, None)
        if errors:
            raise RuntimeError("; ".join(errors))
        remaining = {
            label for label in labels if label in self._active_fill_lights
        }
        remaining.update(
            {
                actor.get_actor_label()
                for actor in subsystem.get_all_level_actors()
                if actor.get_actor_label() in labels
            }
        )
        if remaining:
            raise RuntimeError("venue fill light restore 后仍存在: {}".format(sorted(remaining)))

    @staticmethod
    def _semantic_color(value: Any) -> list:
        channels = [float(getattr(value, name)) for name in ("r", "g", "b", "a")]
        if max(channels) > 1.0:
            channels = [channel / 255.0 for channel in channels]
        return channels


def _write_json(path: Path, value: Mapping) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
    ).encode("utf-8") + b"\n"
    descriptor, temp_name = tempfile.mkstemp(
        prefix=target.name + ".", suffix=".tmp", dir=str(target.parent)
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, str(target))
    except Exception:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


def _read_json(path: Path) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("environment recovery evidence 必须是 JSON object")
    return value


def _diagnostic(manifest: Mapping, profile: Mapping) -> dict:
    from diagnose_episode_environment import diagnose_environment

    episode_dir = profile.get("_episode_dir")
    return diagnose_environment(
        episode_dir=Path(episode_dir) if episode_dir is not None else None,
        resolved_environment=profile,
    )


def _diagnostic_gate_errors(report: Any) -> list:
    if not isinstance(report, Mapping):
        return ["environment diagnostic 缺失或格式无效，MANUAL_REQUIRED"]
    if report.get("schema") != "futsalmot_episode_environment_diagnostic":
        return ["environment diagnostic schema 无效，MANUAL_REQUIRED"]
    if report.get("active") is False or report.get("skipped") is True:
        return ["environment diagnostic inactive，MANUAL_REQUIRED"]
    errors = []
    if report.get("current_level") != FORMAL_MAP:
        errors.append("当前 Level 不匹配正式地图，MANUAL_REQUIRED")
    if report.get("environment_policy") != ENVIRONMENT_POLICY:
        errors.append("environment diagnostic policy 不匹配，MANUAL_REQUIRED")
    manual = report.get("manual_spot_lighting")
    if (
        not isinstance(manual, Mapping)
        or manual.get("actor_labels") != list(MANUAL_SPOTLIGHT_ACTOR_LABELS)
        or manual.get("count") != len(MANUAL_SPOTLIGHT_ACTOR_LABELS)
        or manual.get("class") != "SpotLight"
        or manual.get("intensity_units") != "ev"
        or not isinstance(manual.get("intensity"), (int, float))
    ):
        errors.append("manual SpotLight 10 灯组诊断不完整，MANUAL_REQUIRED")
    fill = report.get("venue_fill_lighting")
    if fill is not None:
        if not isinstance(fill, Mapping):
            errors.append("venue fill lighting diagnostic 无效，MANUAL_REQUIRED")
        elif fill.get("active_actor_labels") != []:
            errors.append("存在未恢复的 venue fill light Actor，MANUAL_REQUIRED")
        elif not isinstance(fill.get("requested"), Mapping) or set(
            fill.get("requested") or {}
        ) != {
            "enabled", "intensity", "intensity_units", "attenuation_radius_cm",
            "source_width_cm", "source_height_cm", "light_color_rgba",
            "cast_shadows", "locations_cm",
        }:
            errors.append("venue fill lighting requested 参数不完整，MANUAL_REQUIRED")
    runtime = report.get("runtime_state")
    if not isinstance(runtime, Mapping):
        errors.append("environment runtime state 未知，MANUAL_REQUIRED")
    elif runtime.get("play_session_active") is not False or runtime.get("mrq_state") != "inactive":
        errors.append("PIE/MRQ runtime state 未知或忙碌，MANUAL_REQUIRED")
    if report.get("preflight_can_start") is not True:
        errors.extend(str(error) for error in (report.get("errors") or []))
        if not any("MANUAL_REQUIRED" in error for error in errors):
            errors.append("environment diagnostic preflight_can_start=false，MANUAL_REQUIRED")
    return list(dict.fromkeys(errors))


def preflight_episode_environment(manifest, profile) -> dict:
    """先读 Task 3 诊断，再校验 P1 输入；全程不访问写属性 API。"""
    try:
        diagnostic_report = _diagnostic(manifest, profile)
    except Exception as exc:
        diagnostic_report = None
        diagnostic_errors = [
            "environment diagnostic 执行失败，MANUAL_REQUIRED: {}: {}".format(
                type(exc).__name__, exc
            )
        ]
    else:
        diagnostic_errors = _diagnostic_gate_errors(diagnostic_report)
    input_errors = []
    try:
        meta = profile.get("_meta")
        frames_path = profile.get("_frames_path")
        if not isinstance(meta, Mapping) or frames_path is None:
            input_errors.append("environment meta/frames_path 缺失")
        else:
            input_errors = validate_environment_inputs(
                manifest, profile, meta, Path(frames_path)
            )
    except Exception as exc:
        input_errors = ["environment input validation 失败: {}".format(exc)]
    errors = diagnostic_errors + input_errors
    return {
        "ok": not errors,
        "operation": "preflight",
        "state": "not_started" if not errors else "preflight_failed",
        "status": None if not errors else "MANUAL_REQUIRED",
        "errors": errors,
        "warnings": [],
        "diagnostic": diagnostic_report,
    }


def _snapshot_paths(snapshot_path: Path, report_path: Path) -> None:
    if Path(snapshot_path).exists() or Path(report_path).exists():
        raise RuntimeError("environment 恢复证据已存在，拒绝覆盖并保留现有文件")


def _diagnostic_actors(diagnostic_report: Mapping) -> dict:
    records = diagnostic_report.get("actors")
    if not isinstance(records, Mapping):
        raise ValueError("environment diagnostic actors 无效")
    return {label: deepcopy(records[label]) for label in ACTOR_CLASSES}


def _diagnostic_baseline(actors: Mapping) -> dict:
    baseline = {}
    for label in ACTOR_CLASSES:
        record = actors[label]
        readbacks = record.get("baseline_readback")
        if not isinstance(readbacks, Mapping):
            raise ValueError("{} baseline_readback 缺失".format(label))
        baseline[label] = {}
        for field, evidence in readbacks.items():
            if not isinstance(evidence, Mapping) or "readback" not in evidence:
                raise ValueError("{} {} 缺少 baseline evidence".format(label, field))
            baseline[label][field] = deepcopy(evidence["readback"])
    return baseline


def _diagnostic_manual_spot_lighting(diagnostic_report: Mapping) -> dict:
    record = diagnostic_report.get("manual_spot_lighting")
    if not isinstance(record, Mapping):
        raise ValueError("manual SpotLight diagnostic 缺失")
    labels = record.get("actor_labels")
    if labels != list(MANUAL_SPOTLIGHT_ACTOR_LABELS):
        raise ValueError("manual SpotLight Actor identity 不完整")
    if record.get("intensity_units") != "ev":
        raise ValueError("manual SpotLight intensity_units 必须为 ev")
    intensity = record.get("intensity")
    if not isinstance(intensity, (int, float)):
        raise ValueError("manual SpotLight intensity 缺失")
    return {
        "actor_labels": list(labels),
        "baseline": {
            "intensity": float(intensity),
            "intensity_units": "ev",
        },
    }


def apply_episode_environment(manifest, profile, snapshot_path, report_path) -> dict:
    """在 read-only gates 全部通过后保存证据；未确认属性时停止于 mutation 前。"""
    gate = preflight_episode_environment(manifest, profile)
    if not gate.get("ok"):
        return {
            "ok": False,
            "operation": "apply",
            "state": "preflight_failed",
            "status": "MANUAL_REQUIRED",
            "errors": gate.get("errors") or ["MANUAL_REQUIRED: environment preflight failed"],
            "warnings": [],
            "snapshot_written": False,
            "property_writes": 0,
        }
    try:
        _snapshot_paths(snapshot_path, report_path)
    except Exception as exc:
        return {
            "ok": False, "operation": "apply", "state": "preflight_failed",
            "status": "MANUAL_REQUIRED", "errors": [str(exc)], "warnings": [],
            "snapshot_written": False, "property_writes": 0,
        }
    diagnostic_report = gate.get("diagnostic")
    plan_result = build_environment_property_write_plan(manifest, diagnostic_report)
    if not plan_result["ok"]:
        return {
            "ok": False,
            "operation": "apply",
            "state": "preflight_failed",
            "status": "MANUAL_REQUIRED",
            "errors": plan_result["errors"],
            "warnings": [],
            "snapshot_written": False,
            "property_writes": 0,
        }
    try:
        transport = UEEnvironmentTransport()
    except Exception as exc:
        return {
            "ok": False,
            "operation": "apply",
            "state": "preflight_failed",
            "status": "MANUAL_REQUIRED",
            "errors": [
                "MANUAL_REQUIRED: 无法初始化 UE environment transport: {}: {}".format(
                    type(exc).__name__, exc
                )
            ],
            "warnings": [],
            "snapshot_written": False,
            "property_writes": 0,
        }
    return _execute_apply_transaction(
        manifest,
        diagnostic_report,
        snapshot_path,
        report_path,
        transport,
    )


def restore_episode_environment(snapshot_path, report_path, reason) -> dict:
    """仅按首次 snapshot baseline 恢复；无已验证写入 adapter 时停止并保留证据。"""
    try:
        snapshot = _read_json(Path(snapshot_path))
        report = _read_json(Path(report_path))
        errors = validate_environment_snapshot(snapshot)
        errors.extend(
            validate_environment_application_report(
                report,
                {
                    "episode_id": report.get("episode_id"),
                    "fingerprint": report.get("environment_manifest_fingerprint"),
                    "catalog_fingerprint": report.get("catalog_fingerprint"),
                    "trajectory_sha256": report.get("trajectory_sha256"),
                    "preset": report.get("preset"),
                },
            )
        )
        if errors:
            raise ValueError("; ".join(errors))
    except Exception as exc:
        return {
            "ok": False,
            "operation": "restore",
            "state": "restore_failed",
            "status": "MANUAL_REQUIRED",
            "errors": ["environment recovery evidence 无效: {}".format(exc)],
            "warnings": [],
        }
    try:
        transport = UEEnvironmentTransport()
        identity = transport.current_identity()
        track_actors = getattr(transport, "track_environment_actors", None)
        if callable(track_actors):
            track_actors()
        result = restore_environment_snapshot(snapshot, identity, transport)
    except Exception as exc:
        result = {
            "ok": False,
            "state": "restore_failed",
            "status": "MANUAL_REQUIRED",
            "errors": [
                "MANUAL_REQUIRED: environment restore transport 失败: {}: {}".format(
                    type(exc).__name__, exc
                )
            ],
        }
    result["operation"] = "restore"
    result["warnings"] = []
    try:
        report["state"] = "restored" if result.get("ok") else "restore_failed"
        report["restore_status"] = "restored" if result.get("ok") else "failed"
        for actor_record in report.get("actors") or []:
            actor_record["restore_status"] = "restored" if result.get("ok") else "failed"
        if isinstance(report.get("venue_fill_lighting"), dict):
            report["venue_fill_lighting"]["enabled"] = False
            report["venue_fill_lighting"]["restore_status"] = (
                "restored" if result.get("ok") else "failed"
            )
            report["venue_fill_lighting"]["actor_labels"] = [] if result.get("ok") else list(
                (snapshot.get("transient_fill_lighting") or {}).get("actor_labels") or []
            )
        if isinstance(report.get("venue_spot_lighting"), dict):
            report["venue_spot_lighting"]["restore_status"] = (
                "restored" if result.get("ok") else "failed"
            )
        if not result.get("ok"):
            report["errors"] = list(dict.fromkeys(
                list(report.get("errors") or [])
                + list(result.get("errors") or ["environment restore failed"])
            ))
        _write_json(Path(report_path), report)
    except Exception as exc:
        result["ok"] = False
        result["state"] = "restore_failed"
        result["status"] = "MANUAL_REQUIRED"
        result["errors"] = list(result.get("errors") or []) + [
            "environment restore report 写入失败: {}: {}".format(type(exc).__name__, exc)
        ]
    return result


def restore_environment_snapshot(snapshot: Mapping, current_identity: Mapping, transport: Any) -> dict:
    """按 snapshot baseline 恢复并验证 Level、Actor identity 和全部字段。"""
    errors = validate_environment_snapshot(snapshot)
    if errors:
        return {
            "ok": False,
            "state": "restore_failed",
            "status": "MANUAL_REQUIRED",
            "errors": errors,
        }
    if current_identity.get("level") != snapshot["level"]:
        errors.append("restore Level identity drift")
    current_actors = current_identity.get("actors")
    try:
        actor_identity_ok = (
            isinstance(current_actors, Mapping)
            and not _validate_actor_identities(current_actors)
            and _actor_identity_fingerprint(current_actors)
            == snapshot["actor_identity_fingerprint"]
        )
    except (TypeError, ValueError, OverflowError):
        actor_identity_ok = False
    if not actor_identity_ok:
        errors.append("restore Actor identity drift")
    if errors:
        return {
            "ok": False,
            "state": "restore_failed",
            "status": "MANUAL_REQUIRED",
            "errors": errors,
        }
    plan = snapshot.get("write_plan") or {}
    if not plan:
        return {
            "ok": False,
            "state": "restore_failed",
            "status": "MANUAL_REQUIRED",
            "errors": ["snapshot write_plan 缺失，无法恢复"],
        }
    baseline = snapshot["baseline"]
    overrides = snapshot.get("overrides") or {}
    try:
        for label, fields in plan.items():
            for field, source in fields.items():
                entry = {
                    "field": field,
                    "target": source.get("target"),
                    "property_name": source.get("property_name"),
                    "evidence_source": source.get("evidence_source"),
                    "baseline": deepcopy(baseline[label][field]),
                    "override_value": bool(
                        (overrides.get(label, {}).get(field) or {}).get(
                            SETTINGS_OVERRIDE_FLAGS.get(field, ""), False
                        )
                    ),
                }
                if isinstance(source.get("overrides"), Mapping):
                    entry["overrides"] = deepcopy(source["overrides"])
                transport.write(label, entry, entry["baseline"])
        restore_manual = getattr(transport, "restore_manual_spot_lighting", None)
        if not callable(restore_manual):
            errors.append("manual SpotLight restore API 缺失")
        else:
            restore_manual(snapshot["manual_spot_lighting"])
            readbacks = transport.read_all(snapshot)
            mismatch = verify_environment_baseline_readback(snapshot, readbacks)
            if not mismatch["ok"]:
                errors.extend(mismatch["errors"])
        fill_state = snapshot.get("transient_fill_lighting") or {}
        fill_labels = list(fill_state.get("actor_labels") or [])
        release_fill = getattr(transport, "release_fill_lighting", None)
        if fill_labels:
            if not callable(release_fill):
                errors.append("venue fill lighting restore API 缺失")
            else:
                release_fill(fill_labels)
                snapshot["transient_fill_lighting"] = {
                    "enabled": False,
                    "actor_labels": [],
                }
        release = getattr(transport, "release_material", None)
        if callable(release):
            release()
        restore_package_baseline = getattr(
            transport, "restore_tracked_package_baseline", None
        )
        if callable(restore_package_baseline):
            restore_package_baseline()
        _commit_transport(transport)
        flush = getattr(transport, "flush_dirty", None)
        if callable(flush):
            failures = flush()
            if failures:
                errors.extend(
                    "restore dirty flag 清除失败: {}".format(package)
                    for package in failures
                )
        clean_state = transport.read_clean_state(snapshot)
        if clean_state != snapshot["clean_baseline"]:
            errors.append("restore package clean baseline mismatch")
        if fill_labels:
            snapshot["transient_fill_lighting"] = {
                "enabled": False,
                "actor_labels": [],
            }
    except Exception as exc:
        errors.append("restore transport failed: {}".format(exc))
    return {
        "ok": not errors,
        "state": "restored" if not errors else "restore_failed",
        "status": None if not errors else "MANUAL_REQUIRED",
        "errors": errors,
    }
