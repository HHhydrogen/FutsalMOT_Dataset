"""环境参数版本一点五的契约测试。"""

from __future__ import annotations

import json
import hashlib
import math
from copy import deepcopy
from pathlib import Path

import pytest

import grf_ue_bridge.environment as environment
from grf_ue_bridge.environment import (
    ENVIRONMENT_CATALOG_ID,
    ENVIRONMENT_PRESETS,
    build_environment_parameters,
    derive_environment_namespace_seed,
    derive_environment_seed,
    environment_catalog_fingerprint,
)


def _environment_profile(seed=42, preset="daylight_clear_v1"):
    return {
        "mode": "seeded",
        "policy": environment.ENVIRONMENT_POLICY,
        "catalog_id": ENVIRONMENT_CATALOG_ID,
        "environment_seed": seed,
        "preset": preset,
        "catalog_fingerprint": environment_catalog_fingerprint(),
    }


def _environment_meta(episode_id="episode_environment"):
    return {"episode_id": episode_id}


def _manifest_functions():
    functions = tuple(
        getattr(environment, name, None)
        for name in (
            "build_environment_manifest",
            "validate_environment_manifest",
            "write_environment_manifest",
        )
    )
    assert all(callable(function) for function in functions), (
        "环境 manifest API 尚未实现"
    )
    return functions


def _assert_json_finite(value):
    if isinstance(value, float):
        assert math.isfinite(value)
    elif isinstance(value, dict):
        for child in value.values():
            _assert_json_finite(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _assert_json_finite(child)


def test_environment_seed_prefers_explicit_seed_and_rejects_invalid_values():
    assert derive_environment_seed(77, 42) == 77
    assert derive_environment_seed(None, 42) == 42

    with pytest.raises(ValueError):
        derive_environment_seed(True, 42)
    with pytest.raises(ValueError):
        derive_environment_seed(-1, 42)
    with pytest.raises(ValueError):
        derive_environment_seed(1.0, 42)


def test_environment_namespace_seed_follows_stable_sha256_protocol():
    first = derive_environment_namespace_seed(42, ENVIRONMENT_CATALOG_ID, "daylight_clear_v1")
    second = derive_environment_namespace_seed(42, ENVIRONMENT_CATALOG_ID, "daylight_clear_v1")

    assert first == second
    assert first == 1795257039
    assert first != derive_environment_namespace_seed(
        42, ENVIRONMENT_CATALOG_ID, "overcast_soft_v1"
    )


@pytest.mark.parametrize("preset", sorted(ENVIRONMENT_PRESETS))
def test_each_environment_preset_has_portable_finite_parameters(preset):
    parameters = build_environment_parameters(42, ENVIRONMENT_CATALOG_ID, preset)

    assert set(parameters) == set(_EXPECTED_PARAMETER_GROUPS)
    for group, fields in _EXPECTED_PARAMETER_GROUPS.items():
        assert set(parameters[group]) == fields

    json.dumps(parameters, allow_nan=False, separators=(",", ":"))
    _assert_json_finite(parameters)


@pytest.mark.parametrize("preset", sorted(ENVIRONMENT_PRESETS))
def test_environment_colors_are_clamped_for_many_seeds(preset):
    for seed in range(200):
        parameters = build_environment_parameters(seed, ENVIRONMENT_CATALOG_ID, preset)
        colors = [
            parameters["directional_light"]["light_color_rgba"],
            parameters["sky_light"]["light_color_rgba"],
        ]
        fog_color = parameters["exponential_height_fog"]["volumetric_fog_albedo_rgba"]
        if fog_color is not None:
            colors.append(fog_color)
        assert all(
            0.0 <= component <= 1.0
            for color in colors
            for component in color
        )


def test_volumetric_fog_albedo_is_clamped_when_catalog_injects_out_of_range_values(
    monkeypatch,
):
    patched_presets = json.loads(
        json.dumps(environment._canonical_value(environment.ENVIRONMENT_PRESETS))
    )
    patched_presets["light_haze_v1"]["exponential_height_fog"][
        "volumetric_fog_albedo_rgba"
    ] = [-0.5, 1.5, 0.5, 2.0]
    monkeypatch.setattr(environment, "ENVIRONMENT_PRESETS", patched_presets)

    parameters = build_environment_parameters(
        42, ENVIRONMENT_CATALOG_ID, "light_haze_v1"
    )

    assert parameters["exponential_height_fog"]["volumetric_fog_albedo_rgba"] == [
        0.0,
        1.0,
        0.5,
        1.0,
    ]


_EXPECTED_PARAMETER_GROUPS = {
    "directional_light": {
        "rotation_deg",
        "intensity",
        "light_color_rgba",
        "temperature_enabled",
        "temperature_kelvin",
        "cast_shadows",
    },
    "sky_light": {
        "intensity",
        "light_color_rgba",
        "source_type",
        "real_time_capture",
    },
    "sky_atmosphere": {"transform_rotation_deg", "multi_scattering"},
    "volumetric_cloud": {
        "visible",
        "layer_bottom_cm",
        "layer_height_cm",
        "cloud_coverage",
        "cloud_density",
    },
    "exponential_height_fog": {
        "visible",
        "fog_density",
        "height_falloff",
        "fog_height_falloff",
        "volumetric_fog_scattering_distribution",
        "volumetric_fog_albedo_rgba",
    },
    "post_process": {
        "infinite_extent",
        "exposure_compensation",
        "white_balance_temperature",
        "white_balance_tint",
        "color_contrast",
        "color_saturation",
    },
    "venue_fill_lighting": {
        "enabled",
        "intensity",
        "intensity_units",
        "attenuation_radius_cm",
        "source_width_cm",
        "source_height_cm",
        "light_color_rgba",
        "cast_shadows",
        "locations_cm",
    },
    "venue_spot_lighting": {
        "enabled",
        "intensity",
        "intensity_units",
        "actor_labels",
    },
}


def test_unused_environment_fields_are_explicitly_null():
    parameters = build_environment_parameters(42, ENVIRONMENT_CATALOG_ID, "daylight_clear_v1")

    assert parameters["directional_light"]["temperature_kelvin"] is None
    assert parameters["volumetric_cloud"]["layer_bottom_cm"] is None
    assert parameters["volumetric_cloud"]["layer_height_cm"] is None
    assert (
        parameters["exponential_height_fog"]["fog_height_falloff"]
        == parameters["exponential_height_fog"]["height_falloff"]
    )
    assert parameters["post_process"]["color_contrast"] is None
    assert parameters["post_process"]["color_saturation"] is None


def test_environment_parameters_are_stable_and_sensitive_to_seed_and_preset():
    first = build_environment_parameters(42, ENVIRONMENT_CATALOG_ID, "daylight_clear_v1")
    repeat = build_environment_parameters(42, ENVIRONMENT_CATALOG_ID, "daylight_clear_v1")
    other_seed = build_environment_parameters(43, ENVIRONMENT_CATALOG_ID, "daylight_clear_v1")
    other_preset = build_environment_parameters(42, ENVIRONMENT_CATALOG_ID, "overcast_soft_v1")

    assert first == repeat
    assert first != other_seed
    assert first != other_preset


def test_environment_parameters_reject_unknown_catalog_and_preset():
    with pytest.raises(ValueError):
        build_environment_parameters(42, "other_catalog", "daylight_clear_v1")
    with pytest.raises(ValueError):
        build_environment_parameters(42, ENVIRONMENT_CATALOG_ID, "unknown_v1")


def test_environment_catalog_fingerprint_is_stable_sha256():
    fingerprint = environment_catalog_fingerprint()

    assert fingerprint == environment_catalog_fingerprint()
    assert len(fingerprint) == 64
    assert fingerprint == fingerprint.lower()
    assert all(character in "0123456789abcdef" for character in fingerprint)


def test_environment_catalog_fingerprint_includes_generation_protocol(monkeypatch):
    original = environment_catalog_fingerprint()

    monkeypatch.setattr(
        environment, "ENVIRONMENT_PARAMETER_ALGORITHM_VERSION", 2, raising=False
    )

    assert environment_catalog_fingerprint() != original


@pytest.mark.parametrize(
    "attribute, replacement",
    [
        ("ENVIRONMENT_PARAMETER_ALGORITHM_ID", "other_algorithm"),
        ("ENVIRONMENT_PARAMETER_RANGES", {"changed": 1.0}),
    ],
)
def test_environment_catalog_fingerprint_changes_when_generation_payload_changes(
    monkeypatch, attribute, replacement
):
    original = environment_catalog_fingerprint()

    monkeypatch.setattr(environment, attribute, replacement)

    assert environment_catalog_fingerprint() != original


def test_build_environment_manifest_is_valid_and_does_not_mutate_profile(tmp_path):
    build_environment_manifest, validate_environment_manifest, _ = _manifest_functions()
    frames_path = tmp_path / "frames.jsonl"
    frames_path.write_bytes(b'{"step":0}\n')
    profile = _environment_profile()
    original_profile = deepcopy(profile)

    manifest = build_environment_manifest(
        _environment_meta(), frames_path, profile
    )

    assert manifest["schema"] == "futsalmot_episode_environment"
    assert manifest["version"] == 1
    assert len(manifest["fingerprint"]) == 64
    assert manifest["trajectory_sha256"] == hashlib.sha256(
        frames_path.read_bytes()
    ).hexdigest()
    assert validate_environment_manifest(
        manifest,
        meta=_environment_meta(),
        frames_path=frames_path,
        profile=profile,
    ) == []
    assert profile == original_profile


def test_environment_manifest_rejects_replaced_trajectory_and_catalog(tmp_path):
    build_environment_manifest, validate_environment_manifest, _ = _manifest_functions()
    frames_path = tmp_path / "frames.jsonl"
    frames_path.write_bytes(b'{"step":0}\n')
    profile = _environment_profile()
    manifest = build_environment_manifest(_environment_meta(), frames_path, profile)

    frames_path.write_bytes(b'{"step":1}\n')
    replaced_profile = _environment_profile(preset="overcast_soft_v1")

    errors = validate_environment_manifest(
        manifest,
        meta=_environment_meta(),
        frames_path=frames_path,
        profile=replaced_profile,
    )

    assert any("trajectory_sha256" in error for error in errors)
    assert any("preset" in error or "parameters" in error for error in errors)

    manifest["fingerprint"] = "0" * 64
    errors = validate_environment_manifest(
        manifest,
        meta=_environment_meta(),
        frames_path=frames_path,
        profile=replaced_profile,
    )
    assert any("fingerprint" in error for error in errors)


def test_environment_manifest_rejects_non_finite_parameters(tmp_path):
    build_environment_manifest, validate_environment_manifest, _ = _manifest_functions()
    frames_path = tmp_path / "frames.jsonl"
    frames_path.write_bytes(b'{"step":0}\n')
    profile = _environment_profile()
    manifest = build_environment_manifest(_environment_meta(), frames_path, profile)
    manifest["parameters"]["directional_light"]["intensity"] = float("nan")

    errors = validate_environment_manifest(
        manifest,
        meta=_environment_meta(),
        frames_path=frames_path,
        profile=profile,
    )

    assert any("parameters" in error for error in errors)


def test_environment_manifest_rejects_unknown_top_level_fields_even_with_valid_fingerprint(
    tmp_path,
):
    build_environment_manifest, validate_environment_manifest, _ = _manifest_functions()
    frames_path = tmp_path / "frames.jsonl"
    frames_path.write_bytes(b'{"step":0}\n')
    profile = _environment_profile()
    manifest = build_environment_manifest(_environment_meta(), frames_path, profile)
    manifest["unexpected_field"] = "must_fail"
    fingerprint_payload = dict(manifest)
    fingerprint_payload.pop("fingerprint")
    manifest["fingerprint"] = hashlib.sha256(
        environment._canonical_json(fingerprint_payload)
    ).hexdigest()

    errors = validate_environment_manifest(
        manifest,
        meta=_environment_meta(),
        frames_path=frames_path,
        profile=profile,
    )

    assert any("未知顶层字段" in error for error in errors)


def test_write_environment_manifest_uses_canonical_path_without_temp_artifact(tmp_path):
    build_environment_manifest, _, write_environment_manifest = _manifest_functions()
    frames_path = tmp_path / "frames.jsonl"
    frames_path.write_bytes(b'{"step":0}\n')
    manifest = build_environment_manifest(
        _environment_meta(), frames_path, _environment_profile()
    )

    output_path = write_environment_manifest(tmp_path, manifest)

    assert output_path == tmp_path / "provenance" / "environment.json"
    assert json.loads(output_path.read_text(encoding="utf-8")) == manifest
    assert list(output_path.parent.glob(".environment.*.tmp")) == []


def test_every_preset_directional_light_points_downward():
    """UE DirectionalLight 的 pitch 必须 <= 0，否则太阳朝上照射导致场景全黑。"""
    for preset in environment.ENVIRONMENT_PRESET_IDS:
        parameters = environment.build_environment_parameters(
            42, environment.ENVIRONMENT_CATALOG_ID, preset
        )
        rotation = parameters["directional_light"]["rotation_deg"]
        assert rotation[0] <= 0.0, preset
        assert -90.0 <= rotation[0] <= 0.0, preset

def test_preset_white_balance_stays_inside_ue_ranges():
    """UE PostProcessSettings.WhiteTint 的有效范围是 [-1, 1]，超出会产生异常色偏。"""
    for preset in environment.ENVIRONMENT_PRESET_IDS:
        post_process = environment.build_environment_parameters(
            42, environment.ENVIRONMENT_CATALOG_ID, preset
        )["post_process"]
        tint = post_process.get("white_balance_tint")
        temperature = post_process.get("white_balance_temperature")
        if tint is not None:
            assert -1.0 <= tint <= 1.0, preset
        if temperature is not None:
            assert 2000.0 <= temperature <= 20000.0, preset

def test_fog_height_falloff_is_an_alias_of_height_falloff():
    """两个 semantic 字段映射到同一 UE 属性，值必须一致否则 apply 会拒绝。"""
    for preset in environment.ENVIRONMENT_PRESET_IDS:
        fog = environment.build_environment_parameters(
            42, environment.ENVIRONMENT_CATALOG_ID, preset
        )["exponential_height_fog"]
        assert fog["fog_height_falloff"] == fog["height_falloff"], preset


def test_environment_presets_include_reversible_venue_fill_lighting():
    """旧的 transient RectLight 参数保持关闭，避免创建临时场景 Actor。"""
    for preset in environment.ENVIRONMENT_PRESET_IDS:
        parameters = environment.build_environment_parameters(
            42, environment.ENVIRONMENT_CATALOG_ID, preset
        )
        fill = parameters["venue_fill_lighting"]
        assert set(fill) == {
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
        assert fill["enabled"] is False
        assert fill["intensity"] == 0.0
        assert fill["locations_cm"] == []

        spot = parameters["venue_spot_lighting"]
        assert spot["enabled"] is True
        assert spot["intensity_units"] == "ev"
        assert spot["actor_labels"] == list(environment.MANUAL_SPOTLIGHT_ACTOR_LABELS)
        assert 5.0 <= spot["intensity"] <= 10.0


def test_venue_fill_lighting_is_subdued_for_approved_daylight_presets():
    """已通过验收的 daylight/haze 只保留很弱的补光，避免改变现有视觉。"""
    for preset in ("daylight_clear_v1", "light_haze_v1"):
        fill = environment.build_environment_parameters(
            42, environment.ENVIRONMENT_CATALOG_ID, preset
        )["venue_fill_lighting"]
        assert fill["enabled"] is False, preset


def test_approved_daylight_and_haze_solar_parameters_are_unchanged():
    for preset in ("daylight_clear_v1", "light_haze_v1"):
        parameters = environment.build_environment_parameters(
            42, environment.ENVIRONMENT_CATALOG_ID, preset
        )
        assert parameters["directional_light"]["rotation_deg"][0] <= -28.0
        assert parameters["sky_light"]["intensity"] > 0.0
    assert environment.ENVIRONMENT_PRESETS["daylight_clear_v1"]["directional_light"]["rotation_deg"] == (-32.0, -25.0, 0.0)
    assert environment.ENVIRONMENT_PRESETS["light_haze_v1"]["directional_light"]["rotation_deg"] == (-30.0, -18.0, 0.0)


def test_manual_spotlight_group_is_explicit_and_preserves_scene_layout():
    parameters = environment.build_environment_parameters(
        42, environment.ENVIRONMENT_CATALOG_ID, "daylight_clear_v1"
    )

    spotlights = parameters["venue_spot_lighting"]

    assert spotlights == {
        "enabled": True,
        "intensity": 5.0,
        "intensity_units": "ev",
        "actor_labels": [
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
        ],
    }


@pytest.mark.parametrize("preset", ["warm_evening_v1", "cool_evening_v1"])
def test_evening_preset_uses_ten_ev_manual_spotlights_and_turns_off_sky(preset):
    parameters = environment.build_environment_parameters(
        42, environment.ENVIRONMENT_CATALOG_ID, preset
    )

    assert parameters["venue_spot_lighting"]["intensity"] == 10.0
    assert parameters["venue_spot_lighting"]["intensity_units"] == "ev"
    assert parameters["sky_light"]["intensity"] == 0.0


@pytest.mark.parametrize(
    "preset, expected_ev",
    [
        ("daylight_clear_v1", 5.0),
        ("overcast_soft_v1", 8.0),
        ("light_haze_v1", 5.0),
        ("cloudy_variation_v1", 8.5),
    ],
)
def test_sky_lit_preset_uses_calibrated_manual_spotlights(preset, expected_ev):
    parameters = environment.build_environment_parameters(
        42, environment.ENVIRONMENT_CATALOG_ID, preset
    )

    assert parameters["venue_spot_lighting"]["intensity"] == expected_ev
    assert parameters["venue_spot_lighting"]["intensity_units"] == "ev"
    assert parameters["sky_light"]["intensity"] > 0.0


def test_evening_preset_keeps_sky_light_disabled_after_seed_variation():
    for preset in ("warm_evening_v1", "cool_evening_v1"):
        for seed in range(20):
            parameters = environment.build_environment_parameters(
                seed, environment.ENVIRONMENT_CATALOG_ID, preset
            )
            assert parameters["sky_light"]["intensity"] == 0.0


def test_auto_weather_selection_is_stable_independent_and_covers_the_catalog():
    import random

    before = random.getstate()
    selected = [environment.resolve_environment_preset(seed, "auto") for seed in range(256)]
    assert selected == [environment.resolve_environment_preset(seed, "auto") for seed in range(256)]
    assert set(selected) == set(environment.ENVIRONMENT_PRESET_IDS)
    assert selected[42] == "cloudy_variation_v1"
    assert selected[47] == "night_floodlit_v1"
    assert random.getstate() == before
    for preset in environment.ENVIRONMENT_PRESET_IDS:
        assert environment.resolve_environment_preset(42, preset) == preset
    with pytest.raises(ValueError):
        environment.resolve_environment_preset(True, "auto")
    with pytest.raises(ValueError):
        environment.resolve_environment_preset(42, "unknown")


@pytest.mark.parametrize("preset", ["warm_evening_v1", "cool_evening_v1", "dawn_cool_v1"])
def test_low_sun_weather_keeps_sun_above_horizon_and_venue_lights_on(preset):
    for seed in range(100):
        parameters = build_environment_parameters(seed, ENVIRONMENT_CATALOG_ID, preset)
        assert -15.0 <= parameters["directional_light"]["rotation_deg"][0] <= -3.0
        assert parameters["venue_spot_lighting"]["intensity"] == 10.0


def test_night_has_no_solar_or_sky_illumination_for_any_seed():
    for seed in range(100):
        parameters = build_environment_parameters(seed, ENVIRONMENT_CATALOG_ID, "night_floodlit_v1")
        assert parameters["directional_light"]["intensity"] == 0.0
        assert parameters["sky_light"]["intensity"] == 0.0
        assert parameters["venue_spot_lighting"]["intensity"] == 10.0
