"""Episode environment v1.5 的纯 Python catalog 与确定性参数。"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from copy import deepcopy
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Optional, Tuple


ENVIRONMENT_POLICY = "futsalmot_environment_v1_5"
ENVIRONMENT_CATALOG_ID = "futsalmot_indoor_environment_v1_5"
ENVIRONMENT_CATALOG_VERSION = 2
ENVIRONMENT_PRESET_IDS = (
    "daylight_clear_v1",
    "overcast_soft_v1",
    "warm_evening_v1",
    "cool_evening_v1",
    "light_haze_v1",
    "cloudy_variation_v1",
    "dawn_cool_v1",
    "night_floodlit_v1",
)
ENVIRONMENT_PRESET_SELECTION_ALGORITHM = "sha256_weather_preset_v1"
_MAX_SEED = 0x7FFFFFFF
ENVIRONMENT_PARAMETER_ALGORITHM_ID = "sha256_namespace_micro_variation_v1"
ENVIRONMENT_PARAMETER_ALGORITHM_VERSION = 1
ENVIRONMENT_PARAMETER_RANGES = MappingProxyType(
    {
        "directional_rotation_deg_amplitude": 2.0,
        "directional_intensity_fraction": 0.04,
        "color_channel_amplitude": 0.008,
        "sky_intensity_fraction": 0.04,
        "cloud_scalar_amplitude": 0.025,
        "fog_density_fraction": 0.05,
        "fog_falloff_amplitude": 0.01,
        "post_exposure_amplitude": 0.025,
    }
)
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
_VENUE_FILL_LIGHTING = {
    "daylight_clear_v1": (False, 0.0),
    "overcast_soft_v1": (False, 0.0),
    "warm_evening_v1": (False, 0.0),
    "cool_evening_v1": (False, 0.0),
    "light_haze_v1": (False, 0.0),
    "cloudy_variation_v1": (False, 0.0),
    "dawn_cool_v1": (False, 0.0),
    "night_floodlit_v1": (False, 0.0),
}
_MANUAL_SPOT_LIGHTING = {
    "daylight_clear_v1": 5.0,
    "overcast_soft_v1": 8.0,
    "warm_evening_v1": 10.0,
    "cool_evening_v1": 10.0,
    "light_haze_v1": 5.0,
    "cloudy_variation_v1": 8.5,
    "dawn_cool_v1": 10.0,
    "night_floodlit_v1": 10.0,
}
ENVIRONMENT_MANIFEST_SCHEMA = "futsalmot_episode_environment"
ENVIRONMENT_MANIFEST_VERSION = 1
ENVIRONMENT_MANIFEST_FIELDS = frozenset(
    {
        "schema",
        "version",
        "episode_id",
        "environment_seed",
        "policy",
        "catalog_id",
        "preset",
        "namespace_seed",
        "parameters",
        "trajectory_sha256",
        "catalog_fingerprint",
        "fingerprint",
    }
)
MANUAL_SPOTLIGHT_FIELDS = frozenset(
    {"enabled", "intensity", "intensity_units", "actor_labels"}
)


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(child) for key, child in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(child) for child in value)
    return value


def _canonical_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _canonical_value(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return [_canonical_value(child) for child in value]
    return value


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        _canonical_value(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _require_seed(value: Any, field: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{field} 必须是非 bool 整数")
    if value < 0:
        raise ValueError(f"{field} 不能为负数")
    return value


def _validate_catalog_and_preset(catalog_id: str, preset: str) -> None:
    if catalog_id != ENVIRONMENT_CATALOG_ID:
        raise ValueError(f"environment catalog_id 非法: {catalog_id!r}")
    if preset not in ENVIRONMENT_PRESET_IDS:
        raise ValueError(f"environment preset 非法: {preset!r}")


def _unit_value(namespace_seed: int, label: str) -> float:
    payload = f"{ENVIRONMENT_POLICY}:{namespace_seed}:{label}".encode("utf-8")
    integer = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return integer / float(1 << 64)


def _vary(value: float, amplitude: float, namespace_seed: int, label: str) -> float:
    return value + (2.0 * _unit_value(namespace_seed, label) - 1.0) * amplitude


def _round_float(value: float) -> float:
    rounded = round(float(value), 6)
    if not math.isfinite(rounded):
        raise ValueError("environment 参数必须是有限浮点数")
    return rounded


def _clamp_rgba(value: Optional[list]) -> Optional[list]:
    if value is None:
        return None
    return [
        _round_float(max(0.0, min(1.0, component)))
        for component in value
    ]


def _color(red: float, green: float, blue: float, namespace_seed: int, label: str) -> list:
    amplitude = ENVIRONMENT_PARAMETER_RANGES["color_channel_amplitude"]
    return _clamp_rgba(
        [
            _vary(red, amplitude, namespace_seed, f"{label}.r"),
            _vary(green, amplitude, namespace_seed, f"{label}.g"),
            _vary(blue, amplitude, namespace_seed, f"{label}.b"),
            1.0,
        ]
    )


def _preset(
    *,
    rotation_deg: Tuple[float, float, float],
    intensity: float,
    directional_color: Tuple[float, float, float],
    sky_intensity: float,
    sky_color: Tuple[float, float, float],
    cloud_coverage: float,
    cloud_density: float,
    fog_density: float,
    height_falloff: float,
    exposure_compensation: float,
    color_contrast: Optional[list],
    color_saturation: Optional[list],
    temperature_enabled: bool = False,
    temperature_kelvin: Optional[float] = None,
    fog_height_falloff: Optional[float] = None,
    volumetric_fog_scattering_distribution: Optional[float] = None,
    volumetric_fog_albedo_rgba: Optional[list] = None,
    white_balance_temperature: Optional[float] = None,
    white_balance_tint: Optional[float] = None,
    venue_fill_intensity: float = 800.0,
    venue_fill_enabled: bool = False,
    venue_spot_intensity: float = 5.0,
) -> dict:
    return {
        "directional_light": {
            "rotation_deg": list(rotation_deg),
            "intensity": intensity,
            "light_color_rgba": [*directional_color, 1.0],
            "temperature_enabled": temperature_enabled,
            "temperature_kelvin": temperature_kelvin,
            "cast_shadows": True,
        },
        "sky_light": {
            "intensity": sky_intensity,
            "light_color_rgba": [*sky_color, 1.0],
            "source_type": "specified_cube_or_scene",
            "real_time_capture": False,
        },
        "sky_atmosphere": {
            "transform_rotation_deg": [0.0, 0.0, 0.0],
            "multi_scattering": 1.0,
        },
        "volumetric_cloud": {
            "visible": True,
            "layer_bottom_cm": None,
            "layer_height_cm": None,
            "cloud_coverage": cloud_coverage,
            "cloud_density": cloud_density,
        },
        "exponential_height_fog": {
            "visible": True,
            "fog_density": fog_density,
            "height_falloff": height_falloff,
            "fog_height_falloff": height_falloff,
            "volumetric_fog_scattering_distribution": volumetric_fog_scattering_distribution,
            "volumetric_fog_albedo_rgba": volumetric_fog_albedo_rgba,
        },
        "post_process": {
            "infinite_extent": True,
            "exposure_compensation": exposure_compensation,
            "white_balance_temperature": white_balance_temperature,
            "white_balance_tint": white_balance_tint,
            "color_contrast": color_contrast,
            "color_saturation": color_saturation,
        },
        "venue_fill_lighting": {
            "enabled": venue_fill_enabled,
            "intensity": venue_fill_intensity,
            "intensity_units": "lumens",
            "attenuation_radius_cm": 2600.0,
            "source_width_cm": 350.0,
            "source_height_cm": 350.0,
            "light_color_rgba": [1.0, 1.0, 1.0, 1.0],
            "cast_shadows": False,
            "locations_cm": ([
                [-3000.0, -2300.0, 900.0],
                [3000.0, -2300.0, 900.0],
                [-3000.0, 2300.0, 900.0],
                [3000.0, 2300.0, 900.0],
            ] if venue_fill_enabled else []),
        },
        "venue_spot_lighting": {
            "enabled": True,
            "intensity": venue_spot_intensity,
            "intensity_units": "ev",
            "actor_labels": list(MANUAL_SPOTLIGHT_ACTOR_LABELS),
        },
    }


_ENVIRONMENT_PRESETS = {
    "daylight_clear_v1": _preset(
        rotation_deg=(-32.0, -25.0, 0.0),
        intensity=8.0,
        directional_color=(1.0, 0.98, 0.92),
        sky_intensity=1.4,
        sky_color=(0.92, 0.96, 1.0),
        cloud_coverage=0.15,
        cloud_density=0.18,
        fog_density=0.002,
        height_falloff=0.20,
        exposure_compensation=0.0,
        color_contrast=None,
        color_saturation=None,
        venue_fill_enabled=_VENUE_FILL_LIGHTING["daylight_clear_v1"][0],
        venue_fill_intensity=_VENUE_FILL_LIGHTING["daylight_clear_v1"][1],
        venue_spot_intensity=_MANUAL_SPOT_LIGHTING["daylight_clear_v1"],
    ),
    "overcast_soft_v1": _preset(
        rotation_deg=(-38.0, -20.0, 0.0),
        intensity=1.5,
        directional_color=(0.92, 0.95, 1.0),
        sky_intensity=2.0,
        sky_color=(0.84, 0.89, 0.96),
        cloud_coverage=0.78,
        cloud_density=0.55,
        fog_density=0.003,
        height_falloff=0.18,
        exposure_compensation=0.15,
        color_contrast=[0.92, 0.92, 0.92, 1.0],
        color_saturation=None,
        venue_fill_intensity=_VENUE_FILL_LIGHTING["overcast_soft_v1"][1],
        venue_spot_intensity=_MANUAL_SPOT_LIGHTING["overcast_soft_v1"],
    ),
    "warm_evening_v1": _preset(
        rotation_deg=(-10.0, -55.0, 0.0),
        intensity=3.2,
        directional_color=(1.0, 0.72, 0.42),
        sky_intensity=0.0,
        sky_color=(1.0, 0.66, 0.46),
        cloud_coverage=0.28,
        cloud_density=0.24,
        fog_density=0.0025,
        height_falloff=0.20,
        exposure_compensation=0.05,
        color_contrast=None,
        color_saturation=[1.05, 1.02, 0.98, 1.0],
        temperature_enabled=True,
        temperature_kelvin=4200.0,
        white_balance_temperature=8000.0,
        white_balance_tint=0.0,
        venue_fill_intensity=_VENUE_FILL_LIGHTING["warm_evening_v1"][1],
        venue_spot_intensity=_MANUAL_SPOT_LIGHTING["warm_evening_v1"],
    ),
    "cool_evening_v1": _preset(
        rotation_deg=(-8.0, 35.0, 0.0),
        intensity=1.5,
        directional_color=(0.48, 0.68, 1.0),
        sky_intensity=0.0,
        sky_color=(0.48, 0.68, 1.0),
        cloud_coverage=0.32,
        cloud_density=0.28,
        fog_density=0.0025,
        height_falloff=0.20,
        exposure_compensation=0.10,
        color_contrast=None,
        color_saturation=[0.98, 1.0, 1.05, 1.0],
        temperature_enabled=True,
        temperature_kelvin=7000.0,
        white_balance_temperature=5000.0,
        white_balance_tint=0.0,
        venue_fill_intensity=_VENUE_FILL_LIGHTING["cool_evening_v1"][1],
        venue_spot_intensity=_MANUAL_SPOT_LIGHTING["cool_evening_v1"],
    ),
    "light_haze_v1": _preset(
        rotation_deg=(-30.0, -18.0, 0.0),
        intensity=6.8,
        directional_color=(0.98, 0.98, 0.94),
        sky_intensity=1.65,
        sky_color=(0.88, 0.92, 0.96),
        cloud_coverage=0.35,
        cloud_density=0.30,
        fog_density=0.012,
        height_falloff=0.12,
        exposure_compensation=0.10,
        color_contrast=[0.96, 0.96, 0.96, 1.0],
        color_saturation=None,
        volumetric_fog_scattering_distribution=0.25,
        volumetric_fog_albedo_rgba=[0.86, 0.90, 0.94, 1.0],
        venue_fill_enabled=_VENUE_FILL_LIGHTING["light_haze_v1"][0],
        venue_fill_intensity=_VENUE_FILL_LIGHTING["light_haze_v1"][1],
        venue_spot_intensity=_MANUAL_SPOT_LIGHTING["light_haze_v1"],
    ),
    "cloudy_variation_v1": _preset(
        rotation_deg=(-45.0, 8.0, 0.0),
        intensity=2.0,
        directional_color=(0.90, 0.94, 1.0),
        sky_intensity=1.8,
        sky_color=(0.80, 0.87, 0.96),
        cloud_coverage=0.88,
        cloud_density=0.68,
        fog_density=0.0035,
        height_falloff=0.17,
        exposure_compensation=0.12,
        color_contrast=[0.95, 0.95, 0.95, 1.0],
        color_saturation=None,
        venue_fill_intensity=_VENUE_FILL_LIGHTING["cloudy_variation_v1"][1],
        venue_spot_intensity=_MANUAL_SPOT_LIGHTING["cloudy_variation_v1"],
    ),
    "dawn_cool_v1": _preset(
        rotation_deg=(-6.0, 125.0, 0.0),
        intensity=2.2,
        directional_color=(0.75, 0.84, 1.0),
        sky_intensity=0.35,
        sky_color=(0.72, 0.82, 1.0),
        cloud_coverage=0.42,
        cloud_density=0.32,
        fog_density=0.008,
        height_falloff=0.14,
        exposure_compensation=0.10,
        color_contrast=[0.96, 0.96, 0.96, 1.0],
        color_saturation=None,
        white_balance_temperature=5600.0,
        white_balance_tint=0.0,
        venue_fill_intensity=0.0,
        venue_spot_intensity=_MANUAL_SPOT_LIGHTING["dawn_cool_v1"],
    ),
    "night_floodlit_v1": _preset(
        rotation_deg=(-20.0, 35.0, 0.0),
        intensity=0.0,
        directional_color=(0.65, 0.75, 1.0),
        sky_intensity=0.0,
        sky_color=(0.48, 0.58, 0.80),
        cloud_coverage=0.55,
        cloud_density=0.40,
        fog_density=0.002,
        height_falloff=0.20,
        exposure_compensation=0.10,
        color_contrast=None,
        color_saturation=None,
        white_balance_temperature=6500.0,
        white_balance_tint=0.0,
        venue_fill_intensity=0.0,
        venue_spot_intensity=_MANUAL_SPOT_LIGHTING["night_floodlit_v1"],
    ),
}

ENVIRONMENT_PRESETS = _freeze(_ENVIRONMENT_PRESETS)
ENVIRONMENT_CATALOG = _freeze(
    {
        "policy": ENVIRONMENT_POLICY,
        "catalog_id": ENVIRONMENT_CATALOG_ID,
        "version": ENVIRONMENT_CATALOG_VERSION,
        "presets": _ENVIRONMENT_PRESETS,
    }
)


def derive_environment_seed(explicit_seed: Optional[int], export_seed: int) -> int:
    """返回非负环境 root seed，显式 seed 优先于 export seed。"""
    if explicit_seed is not None:
        return _require_seed(explicit_seed, "environment_seed")
    return _require_seed(export_seed, "export_seed")


def resolve_environment_preset(environment_seed: int, preset: str) -> str:
    """显式档位直接使用；auto 用独立 SHA-256 命名空间确定天气，不消费全局 RNG。"""
    seed = _require_seed(environment_seed, "environment_seed")
    if preset != "auto":
        _validate_catalog_and_preset(ENVIRONMENT_CATALOG_ID, preset)
        return preset
    payload = f"{ENVIRONMENT_PRESET_SELECTION_ALGORITHM}:{seed}".encode("utf-8")
    index = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return ENVIRONMENT_PRESET_IDS[index % len(ENVIRONMENT_PRESET_IDS)]


def derive_environment_namespace_seed(environment_seed: int, catalog_id: str, preset: str) -> int:
    """按 v1.5 字节协议派生环境命名空间 seed。"""
    seed = _require_seed(environment_seed, "environment_seed")
    _validate_catalog_and_preset(catalog_id, preset)
    payload = f"{ENVIRONMENT_POLICY}:{seed}:{catalog_id}:{preset}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big") & _MAX_SEED


def _validate_finite_json(value: Any) -> None:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("environment 参数必须是有限浮点数")
    elif isinstance(value, Mapping):
        for child in value.values():
            _validate_finite_json(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _validate_finite_json(child)


def build_environment_parameters(
    environment_seed: int, catalog_id: str, preset: str
) -> Dict[str, Any]:
    """生成六类控制面使用的最终便携参数，不依赖 Unreal 或全局随机状态。"""
    seed = _require_seed(environment_seed, "environment_seed")
    namespace_seed = derive_environment_namespace_seed(seed, catalog_id, preset)
    base = _canonical_value(ENVIRONMENT_PRESETS[preset])

    directional = base["directional_light"]
    directional["rotation_deg"] = [
        _round_float(
            _vary(
                value,
                ENVIRONMENT_PARAMETER_RANGES["directional_rotation_deg_amplitude"],
                namespace_seed,
                f"directional.rotation.{index}",
            )
        )
        for index, value in enumerate(directional["rotation_deg"])
    ]
    directional["intensity"] = _round_float(
        _vary(
            directional["intensity"],
            directional["intensity"]
            * ENVIRONMENT_PARAMETER_RANGES["directional_intensity_fraction"],
            namespace_seed,
            "directional.intensity",
        )
    )
    directional["light_color_rgba"] = _color(
        *directional["light_color_rgba"][:3], namespace_seed, "directional.color"
    )

    sky = base["sky_light"]
    if sky["intensity"] > 0.0:
        sky["intensity"] = _round_float(
            _vary(
                sky["intensity"],
                sky["intensity"] * ENVIRONMENT_PARAMETER_RANGES["sky_intensity_fraction"],
                namespace_seed,
                "sky.intensity",
            )
        )
    sky["light_color_rgba"] = _color(
        *sky["light_color_rgba"][:3], namespace_seed, "sky.color"
    )

    cloud = base["volumetric_cloud"]
    cloud["cloud_coverage"] = _round_float(
        max(
            0.0,
            min(
                1.0,
                _vary(
                    cloud["cloud_coverage"],
                    ENVIRONMENT_PARAMETER_RANGES["cloud_scalar_amplitude"],
                    namespace_seed,
                    "cloud.coverage",
                ),
            ),
        )
    )
    cloud["cloud_density"] = _round_float(
        max(
            0.0,
            min(
                1.0,
                _vary(
                    cloud["cloud_density"],
                    ENVIRONMENT_PARAMETER_RANGES["cloud_scalar_amplitude"],
                    namespace_seed,
                    "cloud.density",
                ),
            ),
        )
    )

    fog = base["exponential_height_fog"]
    fog["fog_density"] = _round_float(
        max(
            0.0,
            _vary(
                fog["fog_density"],
                fog["fog_density"] * ENVIRONMENT_PARAMETER_RANGES["fog_density_fraction"],
                namespace_seed,
                "fog.density",
            ),
        )
    )
    fog["height_falloff"] = _round_float(
        max(
            0.0,
            _vary(
                fog["height_falloff"],
                ENVIRONMENT_PARAMETER_RANGES["fog_falloff_amplitude"],
                namespace_seed,
                "fog.falloff",
            ),
        )
    )
    # fog_height_falloff 是 height_falloff 的兼容别名；两者映射到同一个
    # UE 属性，随机化后必须保持同值，否则 apply 的 duplicate-target 检查会拒绝。
    fog["fog_height_falloff"] = fog["height_falloff"]
    fog["volumetric_fog_albedo_rgba"] = _clamp_rgba(
        fog["volumetric_fog_albedo_rgba"]
    )

    post = base["post_process"]
    post["exposure_compensation"] = _round_float(
        _vary(
            post["exposure_compensation"],
            ENVIRONMENT_PARAMETER_RANGES["post_exposure_amplitude"],
            namespace_seed,
            "post.exposure",
        )
    )

    _validate_finite_json(base)
    json.dumps(base, allow_nan=False, separators=(",", ":"))
    return base


def environment_catalog_fingerprint() -> str:
    """返回包含参数生成协议的环境 catalog canonical SHA-256 fingerprint。"""
    payload = {
        **_canonical_value(ENVIRONMENT_CATALOG),
        "parameter_generation": {
            "algorithm_id": ENVIRONMENT_PARAMETER_ALGORITHM_ID,
            "algorithm_version": ENVIRONMENT_PARAMETER_ALGORITHM_VERSION,
            "ranges": _canonical_value(ENVIRONMENT_PARAMETER_RANGES),
        },
        "preset_selection": {
            "algorithm": ENVIRONMENT_PRESET_SELECTION_ALGORITHM,
            "ordered_presets": list(ENVIRONMENT_PRESET_IDS),
        },
    }
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _manifest_without_fingerprint(manifest: Mapping[str, Any]) -> Dict[str, Any]:
    payload = deepcopy(dict(manifest))
    payload.pop("fingerprint", None)
    return payload


def _manifest_fingerprint(manifest: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        _canonical_json(_manifest_without_fingerprint(manifest))
    ).hexdigest()


def _environment_profile_values(profile: Mapping[str, Any]) -> Tuple[int, str, str, str]:
    if not isinstance(profile, Mapping):
        raise ValueError("environment profile 必须是对象")
    if profile.get("mode") != "seeded":
        raise ValueError("environment profile mode 必须为 seeded")
    policy = profile.get("policy")
    if policy != ENVIRONMENT_POLICY:
        raise ValueError("environment profile policy 非法")
    catalog_id = profile.get("catalog_id")
    preset = profile.get("preset")
    if not isinstance(catalog_id, str) or not isinstance(preset, str):
        raise ValueError("environment profile catalog_id/preset 非法")
    environment_seed = _require_seed(
        profile.get("environment_seed"), "environment_seed"
    )
    _validate_catalog_and_preset(catalog_id, preset)
    expected_catalog_fingerprint = environment_catalog_fingerprint()
    if profile.get("catalog_fingerprint") != expected_catalog_fingerprint:
        raise ValueError("environment profile catalog_fingerprint 非法")
    return environment_seed, policy, catalog_id, preset


def build_environment_manifest(
    meta: Mapping[str, Any], frames_path: Path, profile: Mapping[str, Any]
) -> Dict[str, Any]:
    """根据当前 episode 输入生成确定性的环境 provenance manifest。"""
    profile_copy = deepcopy(dict(profile))
    environment_seed, policy, catalog_id, preset = _environment_profile_values(
        profile_copy
    )
    if not frames_path.is_file():
        raise ValueError(f"frames.jsonl 不存在: {frames_path}")
    parameters = build_environment_parameters(environment_seed, catalog_id, preset)
    manifest: Dict[str, Any] = {
        "schema": ENVIRONMENT_MANIFEST_SCHEMA,
        "version": ENVIRONMENT_MANIFEST_VERSION,
        "episode_id": meta.get("episode_id", ""),
        "environment_seed": environment_seed,
        "policy": policy,
        "catalog_id": catalog_id,
        "preset": preset,
        "namespace_seed": derive_environment_namespace_seed(
            environment_seed, catalog_id, preset
        ),
        "parameters": parameters,
        "trajectory_sha256": hashlib.sha256(frames_path.read_bytes()).hexdigest(),
        "catalog_fingerprint": environment_catalog_fingerprint(),
    }
    manifest["fingerprint"] = _manifest_fingerprint(manifest)
    return manifest


def validate_environment_manifest(
    manifest: Mapping[str, Any],
    *,
    meta: Mapping[str, Any],
    frames_path: Path,
    profile: Mapping[str, Any],
) -> List[str]:
    """返回环境 manifest 与当前 meta、轨迹和 resolved profile 的不一致项。"""
    errors: List[str] = []
    if not isinstance(manifest, Mapping):
        return ["environment manifest 必须是对象"]
    unknown_fields = sorted(set(manifest) - ENVIRONMENT_MANIFEST_FIELDS)
    if unknown_fields:
        errors.append(
            "environment manifest 存在未知顶层字段: "
            + ", ".join(str(field) for field in unknown_fields)
        )
    try:
        expected = build_environment_manifest(meta, frames_path, profile)
    except (OSError, TypeError, ValueError, KeyError, OverflowError) as exc:
        return [f"environment 输入无效: {exc}"]

    if manifest.get("schema") != ENVIRONMENT_MANIFEST_SCHEMA:
        errors.append("environment manifest schema 无效")
    if (
        type(manifest.get("version")) is not int
        or manifest.get("version") != ENVIRONMENT_MANIFEST_VERSION
    ):
        errors.append("environment manifest version 无效")

    for key in (
        "episode_id",
        "environment_seed",
        "policy",
        "catalog_id",
        "preset",
        "namespace_seed",
        "trajectory_sha256",
        "catalog_fingerprint",
    ):
        if manifest.get(key) != expected[key]:
            if key == "trajectory_sha256":
                errors.append("trajectory_sha256 与当前 frames.jsonl 不一致")
            elif key == "catalog_fingerprint":
                errors.append("catalog_fingerprint 与当前 catalog 不一致")
            elif key == "environment_seed":
                errors.append("environment_seed 与当前 profile 不一致")
            elif key == "preset":
                errors.append("preset 与当前 profile 不一致")
            else:
                errors.append(f"manifest 字段 {key} 与当前输入不一致")

    parameters = manifest.get("parameters")
    if not isinstance(parameters, Mapping):
        errors.append("manifest parameters 必须是对象")
    else:
        try:
            _validate_finite_json(parameters)
            _canonical_json(parameters)
        except (TypeError, ValueError, OverflowError):
            errors.append("manifest parameters 必须是有限且可 JSON 序列化的对象")
    if parameters != expected["parameters"]:
        errors.append("manifest parameters 与当前 preset/seed 不一致")

    try:
        actual_fingerprint = _manifest_fingerprint(manifest)
        if manifest.get("fingerprint") != actual_fingerprint:
            errors.append("environment manifest fingerprint 无效")
    except (TypeError, ValueError, OverflowError):
        errors.append("environment manifest fingerprint 无法计算")
    return list(dict.fromkeys(errors))


def write_environment_manifest(episode_dir: Path, manifest: Mapping[str, Any]) -> Path:
    """原子写入 episode provenance/environment.json。"""
    output_dir = episode_dir / "provenance"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "environment.json"
    payload = json.dumps(
        manifest,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ).encode("utf-8") + b"\n"

    file_descriptor, temp_name = tempfile.mkstemp(
        prefix=".environment.", suffix=".tmp", dir=str(output_dir)
    )
    try:
        with os.fdopen(file_descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        Path(temp_name).replace(output_path)
    except Exception:
        try:
            Path(temp_name).unlink()
        except FileNotFoundError:
            pass
        raise
    return output_path
