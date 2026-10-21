"""Episode environment v1.5 只读诊断报告契约测试。"""

from __future__ import annotations

from copy import deepcopy

import pytest


def test_fresh_ue_diagnostic_reads_manifest_without_importing_p1(monkeypatch, tmp_path):
    """重启后的 UE 不安装 P1 包；诊断应只读 manifest 参数。"""
    import builtins
    import json
    import sys

    provenance = tmp_path / "provenance"
    provenance.mkdir()
    fill = {"enabled": False}
    (provenance / "environment.json").write_text(
        json.dumps({"parameters": {"venue_fill_lighting": fill}}), encoding="utf-8"
    )
    original_import = builtins.__import__

    def isolated_import(name, *args, **kwargs):
        if name.startswith("grf_ue_bridge"):
            raise ModuleNotFoundError("UE 中不安装 P1 包")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", isolated_import)
    monkeypatch.setitem(sys.modules, "unreal", object())
    monkeypatch.setattr(diagnostic, "collect_environment_diagnostic", lambda unreal, parameters: {"parameters": parameters})
    report = diagnostic.diagnose_environment(
        tmp_path, {"policy": "futsalmot_environment_v1_5", "preset": "night_floodlit_v1"}
    )
    assert report == {"parameters": fill}

import diagnose_episode_environment as diagnostic


REQUIRED_ACTORS = (
    ("DirectionalLight", "DirectionalLight"),
    ("SkyLight", "SkyLight"),
    ("SkyAtmosphere", "SkyAtmosphere"),
    ("VolumetricCloud", "VolumetricCloud"),
    ("ExponentialHeightFog", "ExponentialHeightFog"),
    ("PostProcessVolume", "PostProcessVolume"),
)


def _expected_discovery(label, field):
    if field in ("rotation_deg", "transform_rotation_deg"):
        return {
            "property_name": "get_actor_rotation",
            "operation": "read",
            "target": "actor",
            "evidence_source": "get_actor_rotation",
        }
    target, property_name = diagnostic._VERIFIED_PROPERTY_SPECS[label].get(
        field, (None, None)
    )
    if target is None:
        return {
            "property_name": None,
            "operation": "unverified",
            "target": None,
            "evidence_source": "unknown",
        }
    return {
        "property_name": property_name,
        "operation": (
            "get_scalar_parameter_value"
            if target == "material"
            else "get_editor_property"
        ),
        "target": target,
        "attempts": 1,
        "evidence_source": (
            "get_scalar_parameter_value"
            if target == "material"
            else "get_editor_property"
        ),
    }


def _valid_actor_records():
    records = {}
    for label, class_name in REQUIRED_ACTORS:
        discovered = {
            field: _expected_discovery(label, field)
            for field in diagnostic.ENVIRONMENT_PROPERTY_FIELDS[label]
        }
        records[label] = {
            "actor_label": label,
            "actor_path": "/Game/FutsalMOT/Maps/L_FutsalCourt.{}".format(label),
            "class": class_name,
            "package_path": "/Game/__ExternalActors__/{}".format(label),
            "match_count": 1,
            "discovered_properties": discovered,
            "baseline_readback": {
                field: {
                    "readback": "baseline-{}".format(field),
                    "evidence_source": _expected_discovery(label, field)[
                        "evidence_source"
                    ],
                }
                for field in diagnostic.ENVIRONMENT_PROPERTY_FIELDS[label]
            },
            "property_evidence_source": {
                field: _expected_discovery(label, field)["evidence_source"]
                for field in diagnostic.ENVIRONMENT_PROPERTY_FIELDS[label]
            },
        }
    return records


def _install_known_property_contract(monkeypatch):
    verified = {}
    for label, fields in diagnostic.ENVIRONMENT_PROPERTY_FIELDS.items():
        verified[label] = tuple(
            field
            for field in fields
            if field in diagnostic._VERIFIED_PROPERTY_SPECS.get(label, {})
            or field in ("rotation_deg", "transform_rotation_deg")
        )
    monkeypatch.setattr(diagnostic, "ENVIRONMENT_PROPERTY_FIELDS", verified)


def _build_report(
    records,
    *,
    current_level=None,
    package_dirty=None,
    errors=None,
    policy=None,
    **kwargs
):
    kwargs.setdefault("play_session_active", False)
    kwargs.setdefault("mrq_state", "inactive")
    kwargs.setdefault(
        "runtime_evidence",
        {
            "play_session_api": "EditorLevelLibrary.get_pie_worlds(False)",
            "pie_world_count": 0,
            "mrq_api": "MoviePipelineQueueSubsystem.is_rendering/get_active_executor",
            "mrq_subsystem_is_rendering": False,
            "active_executor_exists": False,
            "active_executor_is_rendering": None,
        },
    )
    kwargs.setdefault(
        "manual_spot_lighting",
        {
            "actor_labels": list(diagnostic.MANUAL_SPOTLIGHT_ACTOR_LABELS),
            "count": 10,
            "class": "SpotLight",
            "intensity_units": "ev",
            "intensity": 5.0,
        },
    )
    return diagnostic.build_environment_diagnostic_report(
        diagnostic.FORMAL_MAP if current_level is None else current_level,
        records,
        [] if package_dirty is None else package_dirty,
        [] if errors is None else errors,
        policy=diagnostic.ENVIRONMENT_POLICY if policy is None else policy,
        **kwargs
    )


def test_有效且全部已验证的环境诊断报告允许进入_preflight(monkeypatch):
    _install_known_property_contract(monkeypatch)
    report = _build_report(_valid_actor_records())

    assert report["schema"] == "futsalmot_episode_environment_diagnostic"
    assert report["version"] == 1
    assert report["environment_policy"] == diagnostic.ENVIRONMENT_POLICY
    assert report["preflight_can_start"] is True
    assert set(report["actors"]) == {
        label for label, _class_name in REQUIRED_ACTORS
    }


def test_collect_diagnostic_fails_closed_when_fill_actor_only_exists_in_registry(monkeypatch):
    class RegistryTransport:
        _active_fill_lights = {
            "FutsalMOTEnvironmentFill_abc_1": object(),
        }

    import episode_environment

    monkeypatch.setattr(
        episode_environment.UEEnvironmentTransport,
        "_active_fill_lights",
        RegistryTransport._active_fill_lights,
    )
    records = _valid_actor_records()
    report = diagnostic.build_environment_diagnostic_report(
        diagnostic.FORMAL_MAP,
        records,
        [],
        [],
        policy=diagnostic.ENVIRONMENT_POLICY,
        fill_parameters={
            "enabled": False,
            "intensity": 0.0,
            "intensity_units": "lumens",
            "attenuation_radius_cm": 2600.0,
            "source_width_cm": 350.0,
            "source_height_cm": 350.0,
            "light_color_rgba": [1.0, 1.0, 1.0, 1.0],
            "cast_shadows": False,
            "locations_cm": [],
        },
        active_fill_actor_labels=list(RegistryTransport._active_fill_lights),
        play_session_active=False,
        mrq_state="inactive",
    )

    assert report["preflight_can_start"] is False
    assert any("unrecovered" in error or "未恢复" in error for error in report["errors"])


def test_manual_spotlight_diagnostic_requires_exactly_ten_scene_lights():
    report = _build_report(
        _valid_actor_records(),
        manual_spot_lighting={
        "actor_labels": ["SpotLight"],
        "count": 1,
        "intensity_units": "ev",
        },
    )

    assert report["preflight_can_start"] is False
    assert any("manual" in error.lower() or "SpotLight" in error for error in report["errors"])


def test_manual_spotlight_diagnostic_normalizes_unreal_ev_enum_repr():
    class UnrealEnum:
        name = "EV"

        def __str__(self):
            return "<LightUnits.EV: 3>"

    assert diagnostic._normalize_light_unit(UnrealEnum()) == "ev"


def test_错误地图阻止_preflight并报告地图错误():
    report = _build_report(
        _valid_actor_records(), current_level="/Game/FutsalMOT/Maps/OtherMap"
    )

    assert report["preflight_can_start"] is False
    assert any("Level" in error for error in report["errors"])


def test_重复_actor_record阻止_preflight并报告重复():
    records = _valid_actor_records()
    records["DirectionalLight"]["match_count"] = 2

    report = _build_report(records)

    assert report["preflight_can_start"] is False
    assert any(
        "DirectionalLight" in error and "match_count" in error
        for error in report["errors"]
    )


def test_缺失或不完整_property_evidence阻止_preflight并标识_actor():
    records = _valid_actor_records()
    records["SkyLight"]["baseline_readback"].pop("intensity")

    report = _build_report(records)

    assert report["preflight_can_start"] is False
    assert any("property" in error and "SkyLight" in error for error in report["errors"])


def test_简单值和无效占位证据不能替代真实_readback_source():
    records = _valid_actor_records()
    records["PostProcessVolume"]["baseline_readback"]["infinite_extent"] = True

    report = _build_report(records)

    assert report["preflight_can_start"] is False
    assert any(
        "property" in error and "PostProcessVolume" in error
        for error in report["errors"]
    )


def test_不匹配策略阻止_preflight():
    report = _build_report(_valid_actor_records(), policy="other_policy")

    assert report["preflight_can_start"] is False
    assert any("policy" in error for error in report["errors"])


def test_dirty_package阻止_preflight():
    report = _build_report(
        _valid_actor_records(), package_dirty=["/Game/Dirty/SomePackage"]
    )

    assert report["preflight_can_start"] is False
    assert any("dirty" in error for error in report["errors"])


def test_错误输入阻止_preflight():
    report = _build_report(_valid_actor_records(), errors=["已有诊断错误"])

    assert report["preflight_can_start"] is False
    assert "已有诊断错误" in report["errors"]


def test_错误class阻止_preflight并保留actor记录():
    records = _valid_actor_records()
    records["DirectionalLight"]["class"] = "SkyLight"

    report = _build_report(records)

    assert report["preflight_can_start"] is False
    assert any("class" in error and "DirectionalLight" in error for error in report["errors"])
    assert report["actors"]["DirectionalLight"]["class"] == "SkyLight"


@pytest.mark.parametrize(
    ("field", "value", "error_fragment"),
    [
        ("actor_label", "WrongLabel", "label"),
        ("actor_path", None, "Actor path"),
        ("package_path", None, "package path"),
    ],
)
def test_actor_identity字段错误分别阻止_preflight(field, value, error_fragment):
    records = _valid_actor_records()
    records["DirectionalLight"][field] = value

    report = _build_report(records)

    assert report["preflight_can_start"] is False
    assert any("DirectionalLight" in error and error_fragment in error for error in report["errors"])


def test_缺少必需actor阻止_preflight():
    records = _valid_actor_records()
    records.pop("SkyLight")

    report = _build_report(records)

    assert report["preflight_can_start"] is False
    assert any("SkyLight" in error and "match_count" in error for error in report["errors"])


def test_重复actor_path阻止_preflight():
    records = _valid_actor_records()
    records["SkyLight"]["actor_path"] = records["DirectionalLight"]["actor_path"]

    report = _build_report(records)

    assert report["preflight_can_start"] is False
    assert any("Actor path 重复" in error for error in report["errors"])


def test_重复package_path阻止_preflight():
    records = _valid_actor_records()
    records["SkyLight"]["package_path"] = records["DirectionalLight"]["package_path"]

    report = _build_report(records)

    assert report["preflight_can_start"] is False
    assert any("package path 重复" in error for error in report["errors"])


def test_结构化_actor记录在报告中保留():
    records = _valid_actor_records()
    records["SkyAtmosphere"]["custom_read_only_note"] = "来自当前地图"
    original = deepcopy(records["SkyAtmosphere"])

    report = _build_report(records)

    assert report["actors"]["SkyAtmosphere"]["source_record"] == original


def test_伪造_property_source不能通过_evidence(monkeypatch):
    _install_known_property_contract(monkeypatch)
    records = _valid_actor_records()
    records["DirectionalLight"]["baseline_readback"]["intensity"][
        "evidence_source"
    ] = "fabricated_source"

    report = _build_report(records)

    assert report["preflight_can_start"] is False
    assert any("DirectionalLight" in error and "intensity" in error for error in report["errors"])


def test_未知_property_mapping不能通过_discovery(monkeypatch):
    _install_known_property_contract(monkeypatch)
    records = _valid_actor_records()
    records["SkyLight"]["discovered_properties"]["intensity"][
        "property_name"
    ] = "unknown_property"

    report = _build_report(records)

    assert report["preflight_can_start"] is False
    assert any("SkyLight" in error and "mapping" in error for error in report["errors"])


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("operation", "set_editor_property"),
        ("target", "actor"),
        ("evidence_source", "fabricated_source"),
        ("attempts", True),
    ],
)
def test_discovery_mapping各协议字段必须匹配(key, value, monkeypatch):
    _install_known_property_contract(monkeypatch)
    records = _valid_actor_records()
    records["SkyLight"]["discovered_properties"]["intensity"][key] = value

    report = _build_report(records)

    assert report["preflight_can_start"] is False
    assert any("SkyLight" in error and "intensity" in error for error in report["errors"])


def test_缺失_actor级_property_evidence_source不能通过(monkeypatch):
    _install_known_property_contract(monkeypatch)
    records = _valid_actor_records()
    records["PostProcessVolume"].pop("property_evidence_source")

    report = _build_report(records)

    assert report["preflight_can_start"] is False
    assert any(
        "PostProcessVolume" in error and "actor-level" in error
        for error in report["errors"]
    )


def test_actor级伪造_property_evidence_source不能通过(monkeypatch):
    _install_known_property_contract(monkeypatch)
    records = _valid_actor_records()
    records["PostProcessVolume"]["property_evidence_source"][
        "exposure_compensation"
    ] = "fabricated_source"

    report = _build_report(records)

    assert report["preflight_can_start"] is False
    assert any(
        "PostProcessVolume" in error and "exposure_compensation" in error
        for error in report["errors"]
    )


@pytest.mark.parametrize("match_count", [True, False, 1.0])
def test_match_count必须是严格整数_one(match_count, monkeypatch):
    _install_known_property_contract(monkeypatch)
    records = _valid_actor_records()
    records["DirectionalLight"]["match_count"] = match_count

    report = _build_report(records)

    assert report["preflight_can_start"] is False
    assert any("DirectionalLight" in error and "match_count" in error for error in report["errors"])


@pytest.mark.parametrize(
    ("play_session_active", "mrq_state", "fragment"),
    [
        (True, "inactive", "PIE/Simulate"),
        (False, "active", "MRQ"),
        (None, "inactive", "PIE/Simulate"),
        (False, None, "MRQ"),
    ],
)
def test_runtime_state任何激活或未知状态都阻止_preflight(
    play_session_active, mrq_state, fragment
):
    errors = diagnostic.validate_runtime_state(
        play_session_active=play_session_active,
        mrq_state=mrq_state,
    )

    assert any(fragment in error for error in errors)


def test_runtime_state全为明确空闲时允许继续():
    assert diagnostic.validate_runtime_state(
        play_session_active=False,
        mrq_state="inactive",
    ) == []


def test_UE_Array代理可迭代时识别PIE与MRQ状态():
    class UnrealArray:
        def __init__(self, values):
            self.values = values

        def __iter__(self):
            return iter(self.values)

    class Subsystem:
        def is_rendering(self):
            return False

        def get_active_executor(self):
            return None

    unreal_module = type(
        "FakeUnreal",
        (),
        {
            "EditorLevelLibrary": type(
                "EditorLevelLibrary",
                (),
                {"get_pie_worlds": staticmethod(lambda _include: UnrealArray([]))},
            ),
            "MoviePipelineQueueSubsystem": object,
            "get_editor_subsystem": staticmethod(lambda _cls: Subsystem()),
        },
    )

    state, errors = diagnostic._runtime_state(unreal_module)

    assert errors == []
    assert state["play_session_active"] is False
    assert state["mrq_state"] == "inactive"
    assert state["evidence"]["pie_world_count"] == 0


def test_UE_Array查询异常必须标记runtime未知():
    class Subsystem:
        def is_rendering(self):
            return False

        def get_active_executor(self):
            return None

    unreal_module = type(
        "FakeUnreal",
        (),
        {
            "EditorLevelLibrary": type(
                "EditorLevelLibrary",
                (),
                {
                    "get_pie_worlds": staticmethod(
                        lambda _include: (_ for _ in ()).throw(RuntimeError("query failed"))
                    )
                },
            ),
            "MoviePipelineQueueSubsystem": object,
            "get_editor_subsystem": staticmethod(lambda _cls: Subsystem()),
        },
    )

    state, errors = diagnostic._runtime_state(unreal_module)

    assert state["play_session_active"] is None
    assert state["mrq_state"] == "inactive"
    assert any("MANUAL_REQUIRED" in error and "PIE/Simulate" in error for error in errors)


@pytest.mark.parametrize(
    ("play_session_active", "mrq_state", "fragment"),
    [
        (True, "inactive", "PIE/Simulate"),
        (False, "active", "MRQ"),
        (None, "inactive", "PIE/Simulate"),
        (False, None, "MRQ"),
    ],
)
def test_runtime_state通过_builder同样关闭门禁(
    play_session_active, mrq_state, fragment, monkeypatch
):
    _install_known_property_contract(monkeypatch)
    report = _build_report(
        _valid_actor_records(),
        play_session_active=play_session_active,
        mrq_state=mrq_state,
    )

    assert report["preflight_can_start"] is False
    assert report["runtime_state"]["play_session_active"] is play_session_active
    assert report["runtime_state"]["mrq_state"] == mrq_state
    assert isinstance(report["runtime_state"]["evidence"], dict)
    assert any(fragment in error for error in report["errors"])


def test_ue58_environment_mapping_uses_available_cloud_and_fog_properties():
    assert diagnostic._VERIFIED_PROPERTY_SPECS["VolumetricCloud"][
        "cloud_coverage"
    ] == ("material", "Cloud_GlobalCoverage")
    assert diagnostic._VERIFIED_PROPERTY_SPECS["VolumetricCloud"][
        "cloud_density"
    ] == ("material", "Cloud_GlobalDensity")
    assert diagnostic._VERIFIED_PROPERTY_SPECS["ExponentialHeightFog"][
        "height_falloff"
    ] == ("component", "fog_height_falloff")


def test_cloud_material_parameter_evidence_is_required_for_preflight():
    records = _valid_actor_records()
    records["VolumetricCloud"]["baseline_readback"]["cloud_density"][
        "evidence_source"
    ] = "get_editor_property"

    report = _build_report(records)

    assert report["preflight_can_start"] is False
    assert any("cloud_density" in error for error in report["errors"])


def test_cloud_material_property_discovery_reads_scalar_from_material_target():
    class FakeMaterial:
        def get_scalar_parameter_value(self, parameter_name):
            return {"Cloud_GlobalCoverage": 0.31, "Cloud_GlobalDensity": 0.42}[parameter_name]

    class FakeCloudComponent:
        def get_editor_property(self, property_name):
            values = {
                "visible": True,
                "layer_bottom_altitude": 1200.0,
                "layer_height": 800.0,
                "material": FakeMaterial(),
            }
            return values[property_name]

    class FakeActor:
        def get_components_by_class(self, _component_class):
            return [FakeCloudComponent()]

        def get_actor_rotation(self):
            raise AssertionError("cloud actor rotation is not requested")

    fake_unreal = type("FakeUnreal", (), {"VolumetricCloudComponent": object})
    discovered, baseline, sources, errors = diagnostic._component_properties(
        fake_unreal, FakeActor(), "VolumetricCloud"
    )

    assert errors == []
    assert discovered["cloud_coverage"]["target"] == "material"
    assert baseline["cloud_coverage"]["readback"] == 0.31
    assert baseline["cloud_density"]["readback"] == 0.42
    assert sources["cloud_density"] == "get_scalar_parameter_value"
