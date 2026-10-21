"""Episode environment v1.5 snapshot、readback 和 report 契约测试。"""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import hashlib
import json

import diagnose_episode_environment as diagnostic
import episode_environment as environment
from grf_ue_bridge.environment import (
    ENVIRONMENT_CATALOG_ID,
    build_environment_parameters,
    derive_environment_namespace_seed,
    environment_catalog_fingerprint,
)


ACTOR_CLASSES = {
    "DirectionalLight": "DirectionalLight",
    "SkyLight": "SkyLight",
    "SkyAtmosphere": "SkyAtmosphere",
    "VolumetricCloud": "VolumetricCloud",
    "ExponentialHeightFog": "ExponentialHeightFog",
    "PostProcessVolume": "PostProcessVolume",
}


def _manifest():
    manifest = {
        "schema": "futsalmot_episode_environment",
        "version": 1,
        "episode_id": "episode_environment",
        "environment_seed": 42,
        "policy": environment.ENVIRONMENT_POLICY,
        "catalog_id": ENVIRONMENT_CATALOG_ID,
        "preset": "daylight_clear_v1",
        "namespace_seed": derive_environment_namespace_seed(
            42, ENVIRONMENT_CATALOG_ID, "daylight_clear_v1"
        ),
        "catalog_fingerprint": environment_catalog_fingerprint(),
        "trajectory_sha256": "a" * 64,
        "parameters": build_environment_parameters(
            42, ENVIRONMENT_CATALOG_ID, "daylight_clear_v1"
        ),
    }
    manifest["fingerprint"] = hashlib.sha256(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return manifest


def _profile():
    return {
        "mode": "seeded",
        "policy": environment.ENVIRONMENT_POLICY,
        "catalog_id": ENVIRONMENT_CATALOG_ID,
        "environment_seed": 42,
        "preset": "daylight_clear_v1",
        "catalog_fingerprint": environment_catalog_fingerprint(),
    }


def _actors():
    return {
        label: {
            "actor_label": label,
            "actor_path": "/Game/FutsalMOT/Maps/L_FutsalCourt.{}".format(label),
            "class": class_name,
            "package_path": "/Game/__ExternalActors__/{}".format(label),
            "match_count": 1,
        }
        for label, class_name in ACTOR_CLASSES.items()
    }


def _baseline():
    parameters = _manifest()["parameters"]
    baseline = {
        label: deepcopy(parameters[group])
        for label, group in (
            ("DirectionalLight", "directional_light"),
            ("SkyLight", "sky_light"),
            ("SkyAtmosphere", "sky_atmosphere"),
            ("VolumetricCloud", "volumetric_cloud"),
            ("ExponentialHeightFog", "exponential_height_fog"),
            ("PostProcessVolume", "post_process"),
        )
    }
    return baseline


def _level():
    return {
        "path": diagnostic.FORMAL_MAP,
        "package_path": diagnostic.FORMAL_MAP,
        "class": "World",
    }


def _snapshot():
    return environment.build_environment_snapshot(
        level=_level(),
        actors=_actors(),
        baseline=_baseline(),
        manifest=_manifest(),
        transaction_id="env_tx_001",
        manual_spot_lighting={
            "actor_labels": list(environment.MANUAL_SPOTLIGHT_ACTOR_LABELS),
            "baseline": {
                "intensity": _manifest()["parameters"]["venue_spot_lighting"]["intensity"],
                "intensity_units": "ev",
            },
        },
    )


def test_environment_snapshot_records_identity_baseline_and_clean_state():
    snapshot = _snapshot()

    assert snapshot["schema"] == environment.SNAPSHOT_SCHEMA
    assert snapshot["version"] == environment.SNAPSHOT_VERSION
    assert set(snapshot["actors"]) == set(ACTOR_CLASSES)
    assert snapshot["baseline"] == _baseline()
    assert snapshot["clean_baseline"] == {"level_clean": True, "packages": []}
    assert environment.validate_environment_snapshot(snapshot) == []


def test_environment_snapshot_records_transient_fill_lighting_state():
    snapshot = _snapshot()

    assert snapshot["transient_fill_lighting"] == {
        "enabled": False,
        "actor_labels": [],
    }
    assert snapshot["venue_fill_parameters"] == _manifest()["parameters"]["venue_fill_lighting"]


def test_environment_snapshot_records_manual_spotlight_baseline_and_plan():
    snapshot = _snapshot()

    assert snapshot["manual_spot_lighting"] == {
        "actor_labels": list(environment.MANUAL_SPOTLIGHT_ACTOR_LABELS),
        "baseline": {"intensity": 5.0, "intensity_units": "ev"},
    }


def test_environment_snapshot_rejects_manifest_or_actor_identity_drift():
    snapshot = _snapshot()

    drifted_manifest = deepcopy(snapshot)
    drifted_manifest["manifest_fingerprint"] = "wrong"
    assert environment.validate_environment_snapshot(drifted_manifest)

    drifted_actor = deepcopy(snapshot)
    drifted_actor["actors"]["SkyLight"]["actor_path"] = (
        "/Game/FutsalMOT/Maps/L_FutsalCourt.OtherSky"
    )
    assert environment.validate_environment_snapshot(drifted_actor)


def test_environment_snapshot_rejects_missing_baseline_dirty_state_and_absolute_path():
    missing_baseline = _snapshot()
    missing_baseline["baseline"].pop("SkyLight")
    assert environment.validate_environment_snapshot(missing_baseline)

    dirty = _snapshot()
    dirty["clean_baseline"]["packages"] = ["/Game/Dirty/Package"]
    assert environment.validate_environment_snapshot(dirty)

    absolute = _snapshot()
    absolute["actors"]["SkyLight"]["actor_path"] = r"C:\\project\\SkyLight"
    assert environment.validate_environment_snapshot(absolute)


def test_environment_application_report_records_portable_transaction_state():
    actors = [
        {
            "actor_label": label,
            "actor_path": record["actor_path"],
            "class": record["class"],
            "package_path": record["package_path"],
            "apply_status": "not_started",
            "restore_status": "not_started",
        }
        for label, record in _actors().items()
    ]
    report = environment.build_environment_application_report(
        _manifest(), "env_tx_001", "not_started", actors, [], []
    )

    assert report["schema"] == environment.APPLICATION_REPORT_SCHEMA
    assert report["environment_manifest_fingerprint"] == _manifest()["fingerprint"]
    assert report["state"] == "not_started"
    assert report["venue_fill_lighting"]["enabled"] is False
    assert report["venue_fill_lighting"]["light_count"] == 0
    assert environment.validate_environment_application_report(report, _manifest()) == []


def test_environment_application_report_rejects_transaction_fingerprint_and_path_drift():
    actors = [
        {
            "actor_label": label,
            "actor_path": record["actor_path"],
            "class": record["class"],
            "package_path": record["package_path"],
            "apply_status": "not_started",
            "restore_status": "not_started",
        }
        for label, record in _actors().items()
    ]
    report = environment.build_environment_application_report(
        _manifest(), "env_tx_001", "not_started", actors, [], []
    )

    wrong_fingerprint = deepcopy(report)
    wrong_fingerprint["environment_manifest_fingerprint"] = "wrong"
    assert environment.validate_environment_application_report(
        wrong_fingerprint, _manifest()
    )

    wrong_transaction = deepcopy(report)
    wrong_transaction["transaction_id"] = "other"
    assert environment.validate_environment_application_report(
        wrong_transaction, _manifest()
    )

    absolute = deepcopy(report)
    absolute["actors"][0]["actor_path"] = r"C:\\project\\Actor"
    assert environment.validate_environment_application_report(absolute, _manifest())


def test_environment_readback_plan_returns_per_field_evidence_for_matching_values():
    actual_values = deepcopy(_manifest()["parameters"])
    snapshot = _snapshot()

    result = environment.apply_environment_readback_plan(
        _manifest(), snapshot, actual_values
    )

    assert result["ok"] is True, result["errors"]
    assert result["state"] == "applied"
    assert result["evidence"]["DirectionalLight"]["intensity"]["actual"] == (
        _manifest()["parameters"]["directional_light"]["intensity"]
    )


def test_environment_readback_plan_fails_closed_on_missing_or_drifted_field():
    snapshot = _snapshot()
    actual_values = deepcopy(_manifest()["parameters"])
    actual_values["sky_light"]["intensity"] = 99.0
    drifted = environment.apply_environment_readback_plan(
        _manifest(), snapshot, actual_values
    )
    assert drifted["ok"] is False
    assert drifted["state"] == "restore_failed"

    missing_values = deepcopy(_manifest()["parameters"])
    missing_values["sky_light"].pop("intensity")
    missing = environment.apply_environment_readback_plan(
        _manifest(), snapshot, missing_values
    )
    assert missing["ok"] is False
    assert missing["state"] == "restore_failed"


def test_environment_readback_accepts_ue_float_precision_but_rejects_real_drift():
    snapshot = _snapshot()
    actual_values = deepcopy(_manifest()["parameters"])
    actual_values["directional_light"]["intensity"] += 0.0000002

    close = environment.apply_environment_readback_plan(
        _manifest(), snapshot, actual_values
    )

    assert close["ok"] is True, close["errors"]

    actual_values["directional_light"]["intensity"] += 0.001
    drifted = environment.apply_environment_readback_plan(
        _manifest(), snapshot, actual_values
    )
    assert drifted["ok"] is False


def test_environment_write_plan_rejects_write_only_property_evidence():
    manifest = _manifest()
    actor_records = {}
    for label, values in environment._expected_field_map(manifest).items():
        actor_records[label] = {
            "discovered_properties": {
                field: {
                    "property_name": field,
                    "operation": "set_editor_property",
                    "target": "component",
                    "evidence_source": "set_editor_property",
                }
                for field in values
            },
            "baseline_readback": {
                field: {
                    "readback": value,
                    "evidence_source": "set_editor_property",
                }
                for field, value in values.items()
            },
        }

    result = environment.build_environment_property_write_plan(
        manifest,
        {"actors": actor_records},
    )

    assert result["ok"] is False
    assert any("write mapping" in error for error in result["errors"])


def test_validate_environment_inputs_rejects_manifest_and_trajectory_drift(tmp_path):
    frames_path = tmp_path / "frames.jsonl"
    frames_path.write_bytes(b'{"step":0}\n')
    manifest = _manifest()
    manifest["trajectory_sha256"] = sha256(frames_path.read_bytes()).hexdigest()
    manifest.pop("fingerprint")
    manifest["fingerprint"] = hashlib.sha256(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    profile = _profile()

    assert environment.validate_environment_inputs(
        manifest,
        profile,
        {"episode_id": "episode_environment"},
        frames_path,
    ) == []

    frames_path.write_bytes(b'{"step":1}\n')
    errors = environment.validate_environment_inputs(
        manifest,
        profile,
        {"episode_id": "episode_environment"},
        frames_path,
    )
    assert any("trajectory_sha256" in error for error in errors)


def test_environment_preflight_returns_manual_required_before_any_mutation(
    monkeypatch, tmp_path
):
    calls = []

    monkeypatch.setattr(
        diagnostic,
        "diagnose_environment",
        lambda **_kwargs: {
            "schema": diagnostic.ENVIRONMENT_SCHEMA,
            "version": 1,
            "active": True,
            "preflight_can_start": False,
            "current_level": diagnostic.FORMAL_MAP,
            "environment_policy": environment.ENVIRONMENT_POLICY,
            "actors": {},
            "package_dirty": [],
            "errors": [
                "VolumetricCloud property cloud_coverage 未验证，MANUAL_REQUIRED"
            ],
        },
    )
    monkeypatch.setattr(
        environment,
        "_load_unreal",
        lambda: calls.append("unreal") or object(),
        raising=False,
    )

    profile = dict(_profile())
    profile.update(
        {
            "_episode_dir": tmp_path,
            "_frames_path": tmp_path / "frames.jsonl",
            "_meta": {"episode_id": "episode_environment"},
        }
    )
    result = environment.preflight_episode_environment(_manifest(), profile)

    assert result["ok"] is False
    assert result["state"] == "preflight_failed"
    assert result["status"] == "MANUAL_REQUIRED"
    assert calls == []


def test_environment_write_plan_accepts_verified_material_target():
    manifest = _manifest()
    actors = {}
    for label, values in environment._expected_field_map(manifest).items():
        actors[label] = {
            "discovered_properties": {
                field: {
                    "property_name": field,
                    "operation": "get_editor_property",
                    "target": "material" if label == "VolumetricCloud" and field in {"cloud_coverage", "cloud_density"} else "component",
                    "evidence_source": "get_editor_property",
                }
                for field in values
            },
            "baseline_readback": {
                field: {
                    "readback": value,
                    "evidence_source": "get_editor_property",
                }
                for field, value in values.items()
            },
        }

    result = environment.build_environment_property_write_plan(
        manifest, {"actors": actors}
    )

    assert result["ok"] is True, result["errors"]
    assert result["plan"]["VolumetricCloud"]["cloud_density"]["target"] == "material"


def test_environment_apply_transaction_writes_and_reads_back_all_verified_fields(tmp_path):
    class FakeTransport:
        def __init__(self):
            self.values = deepcopy(_manifest()["parameters"])

        def new_transaction_id(self):
            return "env_tx_fake"

        def track_environment_actors(self):
            return None

        def write(self, label, entry, value):
            self.values[environment.PARAMETER_GROUPS[label]][entry["field"]] = deepcopy(value)

        def read(self, label, entry):
            return deepcopy(self.values[environment.PARAMETER_GROUPS[label]][entry["field"]])

        def create_fill_lighting(self, config, transaction_id=None):
            self.fill_config = deepcopy(config)
            if not config.get("enabled"):
                return []
            return ["FutsalMOTEnvironmentFill_fake_{}".format(i + 1) for i in range(4)]

        def apply_manual_spot_lighting(self, config):
            self.values["venue_spot_lighting"].update(
                {key: deepcopy(config[key]) for key in ("intensity", "intensity_units")}
            )
            return list(environment.MANUAL_SPOTLIGHT_ACTOR_LABELS)

        def read_manual_spot_lighting(self, labels):
            spot = self.values["venue_spot_lighting"]
            return {
                "enabled": True,
                "intensity": spot["intensity"],
                "intensity_units": spot["intensity_units"],
                "actor_labels": list(labels),
            }

        def fill_lighting_actor_labels(self, transaction_id, count):
            return ["FutsalMOTEnvironmentFill_fake_{}".format(i + 1) for i in range(count)]

        def read_fill_lighting(self, labels):
            return deepcopy(self.fill_config)

        def assert_transient_fill_lighting(self, labels):
            return None

        def restore_tracked_package_baseline(self):
            return None

        def track_environment_actors(self):
            return None

        def read_all(self, snapshot):
            return deepcopy(self.values)

        def flush_dirty(self):
            return []

        def read_clean_state(self, snapshot):
            return deepcopy(snapshot["clean_baseline"])

        def release_fill_lighting(self, labels):
            self.fill_config["enabled"] = False
            self.fill_config["locations_cm"] = []

        def restore_manual_spot_lighting(self, snapshot):
            self.values["venue_spot_lighting"].update(snapshot["baseline"])

        def _restore_tracked_packages(self, labels):
            return None

    manifest = _manifest()
    actors = _actors()
    diagnostic_report = {
        "current_level": diagnostic.FORMAL_MAP,
        "package_dirty": [],
        "manual_spot_lighting": {
            "actor_labels": list(environment.MANUAL_SPOTLIGHT_ACTOR_LABELS),
            "count": 10,
            "class": "SpotLight",
            "intensity_units": "ev",
            "intensity": 5.0,
        },
        "actors": {
            label: {
                **record,
                "discovered_properties": {
                    field: {
                        "property_name": field,
                        "operation": "get_editor_property",
                        "target": "material" if label == "VolumetricCloud" and field in {"cloud_coverage", "cloud_density"} else "component",
                        "evidence_source": "get_editor_property",
                    }
                    for field in environment._expected_field_map(manifest)[label]
                },
                "baseline_readback": {
                    field: {"readback": value, "evidence_source": "get_editor_property"}
                    for field, value in _baseline()[label].items()
                },
            }
            for label, record in actors.items()
        },
    }
    for field in environment._expected_field_map(manifest)["VolumetricCloud"]:
        diagnostic_report["actors"]["VolumetricCloud"]["discovered_properties"][field]["property_name"] = field

    result = environment._execute_apply_transaction(
        manifest,
        diagnostic_report,
        tmp_path / "snapshot.json",
        tmp_path / "report.json",
        FakeTransport(),
    )

    assert result["ok"] is True, result["errors"]
    assert result["state"] == "applied"

    class BrokenFillReadback(FakeTransport):
        def read_fill_lighting(self, labels):
            raise RuntimeError("simulated fill readback failure")

    failing_manifest = deepcopy(manifest)
    failing_fill = failing_manifest["parameters"]["venue_fill_lighting"]
    failing_fill["enabled"] = True
    failing_fill["intensity"] = 9000.0
    failing_fill["locations_cm"] = [[0.0, 0.0, 900.0]] * 4
    failing_manifest["fingerprint"] = environment._manifest_fingerprint(
        failing_manifest
    )
    failed = environment._execute_apply_transaction(
        failing_manifest,
        diagnostic_report,
        tmp_path / "failed-snapshot.json",
        tmp_path / "failed-report.json",
        BrokenFillReadback(),
    )

    assert failed["ok"] is False
    assert failed["state"] == "restored", failed["errors"]
    assert failed["report"]["restore_status"] == "restored"
    assert failed["report"]["venue_fill_lighting"]["restore_status"] == "restored"


def test_environment_color_readback_accepts_eight_bit_quantization():
    actual_values = deepcopy(_manifest()["parameters"])
    quantized = [round(channel * 255.0) / 255.0 for channel in actual_values["directional_light"]["light_color_rgba"]]
    actual_values["directional_light"]["light_color_rgba"] = quantized
    result = environment.apply_environment_readback_plan(
        _manifest(), _snapshot(), actual_values
    )
    assert result["ok"] is True, result["errors"]

    actual_values["directional_light"]["light_color_rgba"] = [0.5, 0.5, 0.5, 1.0]
    drifted = environment.apply_environment_readback_plan(
        _manifest(), _snapshot(), actual_values
    )
    assert drifted["ok"] is False


def _diagnostic_with_mappings():
    manifest = _manifest()
    baseline = _baseline()
    actors = {}
    for label, record in _actors().items():
        fields = environment._expected_field_map(manifest)[label]
        discovered = {}
        readbacks = {}
        for field in fields:
            is_rotation = field in ("rotation_deg", "transform_rotation_deg")
            is_actor_property = label == "PostProcessVolume" and field == "infinite_extent"
            is_material = label == "VolumetricCloud" and field in {"cloud_coverage", "cloud_density"}
            is_settings = label == "PostProcessVolume" and field != "infinite_extent"
            if is_rotation:
                property_name = "get_actor_rotation"
                operation = "read"
                target = "actor"
                evidence_source = "get_actor_rotation"
            elif is_material:
                property_name = "Cloud_GlobalCoverage" if field == "cloud_coverage" else "Cloud_GlobalDensity"
                operation = "get_scalar_parameter_value"
                target = "material"
                evidence_source = "get_scalar_parameter_value"
            elif is_settings:
                property_name = {
                    "exposure_compensation": "auto_exposure_bias",
                    "white_balance_temperature": "white_temp",
                    "white_balance_tint": "white_tint",
                    "color_contrast": "color_contrast",
                    "color_saturation": "color_saturation",
                }[field]
                operation = "get_editor_property"
                target = "settings"
                evidence_source = "get_editor_property"
            elif is_actor_property:
                property_name = "unbound"
                operation = "get_editor_property"
                target = "actor"
                evidence_source = "get_editor_property"
            else:
                property_name = field
                operation = "get_editor_property"
                target = "component"
                evidence_source = "get_editor_property"
            discovered[field] = {
                "property_name": property_name,
                "operation": operation,
                "target": target,
                "evidence_source": evidence_source,
            }
            if operation != "read":
                discovered[field]["attempts"] = 1
            readbacks[field] = {
                "readback": deepcopy(baseline[label][field]),
                "evidence_source": evidence_source,
            }
            override_flag = environment.SETTINGS_OVERRIDE_FLAGS.get(field)
            if target == "settings" and override_flag:
                readbacks[field]["overrides"] = {override_flag: False}
        actors[label] = {
            **deepcopy(record),
            "discovered_properties": discovered,
            "baseline_readback": readbacks,
            "property_evidence_source": {
                field: readbacks[field]["evidence_source"] for field in fields
            },
        }
    return {
        "schema": diagnostic.ENVIRONMENT_SCHEMA,
        "current_level": diagnostic.FORMAL_MAP,
        "package_dirty": [],
        "actors": actors,
    }


def test_environment_write_plan_accepts_actor_rotation_and_settings_overrides():
    report = _diagnostic_with_mappings()
    result = environment.build_environment_property_write_plan(_manifest(), report)
    assert result["ok"] is True, result["errors"]
    assert result["plan"]["DirectionalLight"]["rotation_deg"]["target"] == "actor"
    settings_entry = result["plan"]["PostProcessVolume"]["exposure_compensation"]
    assert settings_entry["target"] == "settings"
    assert settings_entry["overrides"] == {"override_auto_exposure_bias": False}
    assert settings_entry["override_value"] is True


def test_environment_restore_snapshot_uses_write_plan_and_restores_overrides():
    class RecordingTransport:
        def __init__(self):
            self.writes = []

        def write(self, label, entry, value):
            self.writes.append((label, entry["field"], entry["property_name"], entry["target"], deepcopy(value), entry.get("override_value")))

        def read_all(self, snapshot):
            values = deepcopy(_manifest()["parameters"])
            values["venue_spot_lighting"] = {
                "enabled": True,
                "intensity": 5.0,
                "intensity_units": "ev",
                "actor_labels": list(environment.MANUAL_SPOTLIGHT_ACTOR_LABELS),
            }
            return values

        def restore_manual_spot_lighting(self, snapshot):
            self.manual_snapshot = deepcopy(snapshot)

        def release_material(self):
            pass

        def commit(self):
            pass

        def flush_dirty(self):
            return []

        def read_clean_state(self, snapshot):
            return deepcopy(snapshot["clean_baseline"])

    report = _diagnostic_with_mappings()
    plan = environment.build_environment_property_write_plan(_manifest(), report)
    overrides = {
        label: {field: entry["overrides"] for field, entry in fields.items() if entry.get("overrides")}
        for label, fields in plan["plan"].items()
    }
    snapshot = environment.build_environment_snapshot(
        level=_level(),
        actors=_actors(),
        baseline=_baseline(),
        manifest=_manifest(),
        transaction_id="env_tx_restore",
        write_plan=plan["plan"],
        overrides=overrides,
    )
    transport = RecordingTransport()

    result = environment.restore_environment_snapshot(
        snapshot, {"level": _level(), "actors": _actors()}, transport
    )

    assert result["ok"] is True, result["errors"]
    exposure = [write for write in transport.writes if write[1] == "exposure_compensation"][0]
    assert exposure[2] == "auto_exposure_bias"
    assert exposure[3] == "settings"
    assert exposure[5] is False

class _FakeEnumValue:
    def __init__(self, name):
        self._name = name

    def __str__(self):
        return "SkyLightSourceType." + self._name


class _FakeColor:
    # 与 UE 5.8 一致：位置参数顺序为 (b, g, r, a)，关键字参数按属性名赋值。
    def __init__(self, b=0, g=0, r=0, a=0):
        self.r, self.g, self.b, self.a = r, g, b, a


class _FakeRotator:
    def __init__(self, roll=0.0, pitch=0.0, yaw=0.0):
        self.roll, self.pitch, self.yaw = roll, pitch, yaw


class _FakeVector4:
    def __init__(self, x=0.0, y=0.0, z=0.0, w=0.0):
        self.x, self.y, self.z, self.w = x, y, z, w


class _FakeSettings:
    def __init__(self, values):
        self.values = dict(values)

    def get_editor_property(self, name):
        return self.values[name]

    def set_editor_property(self, name, value):
        self.values[name] = value


class _FakeComponent:
    def __init__(self, **properties):
        self.properties = dict(properties)
        self.recaptured = 0

    def get_editor_property(self, name):
        return self.properties[name]

    def set_editor_property(self, name, value):
        self.properties[name] = value

    def get_scalar_parameter_value(self, name):
        return self.properties[name]

    def recapture_sky(self):
        self.recaptured += 1


class _FakeClass:
    def __init__(self, name):
        self._name = name

    def get_name(self):
        return self._name


class _FakePackage:
    def __init__(self, name):
        self._name = name

    def get_name(self):
        return self._name


class _FakeActor:
    def __init__(self, label, class_name, component):
        self.label = label
        self.class_name = class_name
        self.component = component
        self.rotation = _FakeRotator()
        self.properties = {"unbound": True}
        self.package = _FakePackage("/Game/__ExternalActors__/" + label)

    def get_actor_label(self):
        return self.label

    def get_class(self):
        return _FakeClass(self.class_name)

    def get_path_name(self):
        return "/Game/FutsalMOT/Maps/L_FutsalCourt.{}".format(self.label)

    def get_components_by_class(self, component_class):
        expected = self.class_name + "Component"
        if getattr(component_class, "__name__", None) == expected:
            return [self.component]
        return []

    def get_actor_rotation(self):
        return self.rotation

    def set_actor_rotation(self, rotation, teleport_physics):
        self.rotation = rotation
        return True

    def get_editor_property(self, name):
        if name in self.properties:
            return self.properties[name]
        return self.component.get_editor_property(name)

    def set_editor_property(self, name, value):
        if name in self.properties:
            self.properties[name] = value
        else:
            self.component.set_editor_property(name, value)


class _FakeMID:
    def __init__(self, parent):
        self.parent = parent
        self.scalars = {}

    def get_class(self):
        return _FakeClass("MaterialInstanceDynamic")

    def get_editor_property(self, name):
        if name == "parent":
            return self.parent
        raise KeyError(name)

    def set_scalar_parameter_value(self, name, value):
        self.scalars[name] = value

    def get_scalar_parameter_value(self, name):
        return self.scalars[name]


class _FakeActorSubsystem:
    def __init__(self, actors):
        self.actors = actors

    def get_all_level_actors(self):
        return list(self.actors)

    def spawn_actor_from_class(self, actor_class, location, rotation, transient=False):
        actor = _FakeRectLightActor(self, location, rotation)
        actor.transient = transient
        self.actors.append(actor)
        return actor

    def destroy_actor(self, actor):
        self.actors.remove(actor)
        return True


class _FakeRectLightActor:
    def __init__(self, subsystem, location, rotation):
        self.subsystem = subsystem
        self.location = location
        self.rotation = rotation
        self.label = ""
        self.component = _FakeComponent(
            intensity=5000.0,
            intensity_units="unitless",
            attenuation_radius=1000.0,
            source_width=64.0,
            source_height=64.0,
            light_color=_FakeColor(255, 255, 255, 255),
            cast_shadows=True,
        )

    def get_components_by_class(self, component_class):
        return [self.component]

    def set_actor_label(self, label):
        self.label = label

    def get_actor_label(self):
        return self.label

    def get_path_name(self):
        return "/Engine/Transient." + self.label

    def get_class(self):
        return _FakeClass("RectLight")

    def get_actor_location(self):
        return self.location

    def get_path_name(self):
        return "TransientLevel." + self.label


class _FakeAssetSubsystem:
    def __init__(self):
        self.cleared = []

    def set_dirty_flag(self, obj, state):
        self.cleared.append((obj.get_actor_label(), state))
        return True


class _FakeWorld:
    def get_path_name(self):
        return diagnostic.FORMAL_MAP + ".L_FutsalCourt"


class _FakeFakeUnreal:
    """最小 UE 5.8 命名空间替身，只覆盖 environment transport 使用的 API。"""

    class EditorActorSubsystem:
        pass

    class EditorAssetSubsystem:
        pass

    def __init__(self):
        self.dirty_packages = []
        self.material_library = type("MaterialLibrary", (), {})
        self.material_library.create_dynamic_material_instance = self._create_mid
        self.created_mids = []
        self.spawned_lights = []
        self.destroyed_lights = []
        self.dirty_packages = []
        self.asset_subsystem = _FakeAssetSubsystem()
        self.actor_subsystem = None
        self.Color = _FakeColor
        self.Rotator = _FakeRotator
        self.Vector4 = _FakeVector4
        self.SkyLightSourceType = type(
            "SkyLightSourceType",
            (),
            {
                "SLS_CAPTURED_SCENE": _FakeEnumValue("SLS_CAPTURED_SCENE"),
                "SLS_SPECIFIED_CUBEMAP": _FakeEnumValue("SLS_SPECIFIED_CUBEMAP"),
            },
        )
        self.MaterialLibrary = self.material_library
        self.SystemLibrary = type(
            "SystemLibrary",
            (),
            {"get_outermost": staticmethod(lambda obj: getattr(obj, "package", obj))},
        )
        self.EditorLoadingAndSavingUtils = type(
            "EditorLoadingAndSavingUtils",
            (),
            {
                "get_dirty_map_packages": staticmethod(lambda: self.dirty_packages),
                "get_dirty_content_packages": staticmethod(lambda: []),
            },
        )
        self.EditorLevelLibrary = type(
            "EditorLevelLibrary",
            (),
            {"get_editor_world": staticmethod(lambda: _FakeWorld())},
        )
        for label in ACTOR_CLASSES:
            setattr(
                self,
                label + "Component",
                type(label + "Component", (), {}),
            )
        self.RectLight = type("RectLight", (), {})
        self.RectLightComponent = type("RectLightComponent", (), {})
        self.Vector = lambda x=0.0, y=0.0, z=0.0: type(
            "FakeVector", (), {"x": x, "y": y, "z": z}
        )()
        self.MathLibrary = type(
            "MathLibrary",
            (),
            {"find_look_at_rotation": staticmethod(lambda start, target: _FakeRotator())},
        )
        self.Rotator = _FakeRotator
        self.LightUnits = type("LightUnits", (), {"UNITLESS": "unitless", "LUMENS": "lumens"})

    def _create_mid(self, world, parent):
        mid = _FakeMID(parent)
        self.created_mids.append(mid)
        return mid

    def get_editor_subsystem(self, cls):
        if cls is _FakeFakeUnreal.EditorAssetSubsystem:
            return self.asset_subsystem
        return self.actor_subsystem

    def _spawn_rect_light(self, actor_class, location, rotation, transient=False):
        actor = _FakeRectLightActor(self.actor_subsystem, location, rotation)
        actor.transient = transient
        self.actor_subsystem.actors.append(actor)
        self.spawned_lights.append(actor)
        return actor


class _FakeMaterial:
    def __init__(self, class_name):
        self._class_name = class_name

    def get_class(self):
        return _FakeClass(self._class_name)


def _transport_fixture():
    fake = _FakeFakeUnreal()
    parent_material = _FakeMaterial("MaterialInstanceConstant")
    components = {}
    actors = []
    for label, class_name in ACTOR_CLASSES.items():
        if label == "PostProcessVolume":
            component = _FakeComponent(
                settings=_FakeSettings(
                    {
                        "auto_exposure_bias": 1.0,
                        "white_temp": 6500.0,
                        "white_tint": 0.0,
                        "color_contrast": _FakeVector4(1.0, 1.0, 1.0, 1.0),
                        "color_saturation": _FakeVector4(1.0, 1.0, 1.0, 1.0),
                        "override_auto_exposure_bias": False,
                        "override_white_temp": False,
                        "override_white_tint": False,
                        "override_color_contrast": False,
                        "override_color_saturation": False,
                    }
                )
            )
        elif label == "VolumetricCloud":
            component = _FakeComponent(
                material=parent_material,
                visible=True,
                layer_bottom_altitude=0.0,
                layer_height=0.0,
                Cloud_GlobalCoverage=0.0,
                Cloud_GlobalDensity=0.0,
            )
        elif label == "SkyLight":
            component = _FakeComponent(
                intensity=1.0,
                light_color=_FakeColor(255, 255, 255, 255),
                source_type=_FakeEnumValue("SLS_CAPTURED_SCENE"),
                real_time_capture=True,
            )
        elif label == "DirectionalLight":
            component = _FakeComponent(
                intensity=3.0,
                light_color=_FakeColor(255, 255, 255, 255),
                use_temperature=True,
                temperature=6500.0,
                cast_shadows=True,
            )
        else:
            component = _FakeComponent(
                multi_scattering_factor=1.0,
                visible=True,
                fog_density=0.02,
                fog_height_falloff=0.2,
                volumetric_fog_scattering_distribution=0.0,
                volumetric_fog_albedo=_FakeColor(255, 255, 255, 255),
            )
        components[label] = component
        actors.append(_FakeActor(label, class_name, component))
    fake.actor_subsystem = _FakeActorSubsystem(actors)
    fake.actor_subsystem.spawn_actor_from_class = fake._spawn_rect_light
    def destroy_actor(actor):
        fake.destroyed_lights.append(actor)
        if actor in fake.actor_subsystem.actors:
            fake.actor_subsystem.actors.remove(actor)
        return True

    fake.actor_subsystem.destroy_actor = destroy_actor
    return fake, components


def test_ue_environment_transport_writes_rotation_color_and_settings():
    fake, components = _transport_fixture()
    transport = environment.UEEnvironmentTransport(fake)

    transport.write(
        "DirectionalLight",
        {"field": "rotation_deg", "target": "actor", "property_name": "get_actor_rotation"},
        [32.0, -25.0, 0.0],
    )
    assert transport.read(
        "DirectionalLight",
        {"field": "rotation_deg", "target": "actor", "property_name": "get_actor_rotation"},
    ) == [32.0, -25.0, 0.0]

    transport.write(
        "DirectionalLight",
        {"field": "light_color_rgba", "target": "component", "property_name": "light_color"},
        [0.98, 0.5, 1.0, 1.0],
    )
    color = components["DirectionalLight"].get_editor_property("light_color")
    assert (color.r, color.g, color.b, color.a) == (250, 128, 255, 255)

    transport.write(
        "PostProcessVolume",
        {
            "field": "exposure_compensation",
            "target": "settings",
            "property_name": "auto_exposure_bias",
            "override_value": True,
        },
        0.5,
    )
    settings = components["PostProcessVolume"].get_editor_property("settings")
    assert settings.get_editor_property("auto_exposure_bias") == 0.5
    assert settings.get_editor_property("override_auto_exposure_bias") is True
    assert transport.read(
        "PostProcessVolume",
        {"field": "exposure_compensation", "target": "settings", "property_name": "auto_exposure_bias"},
    ) == 0.5


def test_ue_environment_transport_uses_transient_material_and_restores_parent():
    fake, components = _transport_fixture()
    transport = environment.UEEnvironmentTransport(fake)
    parent = components["VolumetricCloud"].get_editor_property("material")

    transport.write(
        "VolumetricCloud",
        {"field": "cloud_coverage", "target": "material", "property_name": "Cloud_GlobalCoverage"},
        0.15,
    )
    mid = components["VolumetricCloud"].get_editor_property("material")
    assert mid is fake.created_mids[0]
    assert mid is not parent
    assert transport.read(
        "VolumetricCloud",
        {"field": "cloud_coverage", "target": "material", "property_name": "Cloud_GlobalCoverage"},
    ) == 0.15

    transport.release_material()
    assert components["VolumetricCloud"].get_editor_property("material") is parent


def test_manual_spotlight_transport_changes_only_intensity_and_restores_baseline():
    fake, components = _transport_fixture()
    lights = []
    for label in environment.MANUAL_SPOTLIGHT_ACTOR_LABELS:
        component = _FakeComponent(
            intensity=5.0,
            intensity_units="ev",
            light_color="white",
            attenuation_radius=8000.0,
            outer_cone_angle=40.0,
            cast_shadows=True,
        )
        actor = _FakeActor(label, "SpotLight", component)
        lights.append(actor)
    fake.actor_subsystem.actors.extend(lights)
    fake.SpotLightComponent = type("SpotLightComponent", (), {})
    fake.LightUnits.EV = "ev"
    transport = environment.UEEnvironmentTransport(fake)

    applied = transport.apply_manual_spot_lighting(
        {
            "enabled": True,
            "intensity": 10.0,
            "intensity_units": "ev",
            "actor_labels": list(environment.MANUAL_SPOTLIGHT_ACTOR_LABELS),
        }
    )

    assert applied == list(environment.MANUAL_SPOTLIGHT_ACTOR_LABELS)
    assert transport.read_manual_spot_lighting(applied) == {
        "enabled": True,
        "intensity": 10.0,
        "intensity_units": "ev",
        "actor_labels": list(environment.MANUAL_SPOTLIGHT_ACTOR_LABELS),
    }
    assert all(light.component.properties["light_color"] == "white" for light in lights)
    assert all(light.component.properties["attenuation_radius"] == 8000.0 for light in lights)
    assert all(light.component.properties["outer_cone_angle"] == 40.0 for light in lights)
    assert all(light.component.properties["cast_shadows"] is True for light in lights)

    transport.restore_manual_spot_lighting(
        {
            "actor_labels": list(environment.MANUAL_SPOTLIGHT_ACTOR_LABELS),
            "baseline": {"intensity": 5.0, "intensity_units": "ev"},
        }
    )

    assert all(light.component.properties["intensity"] == 5.0 for light in lights)
    assert all(light.component.properties["intensity_units"] == "ev" for light in lights)


def test_manual_spotlight_transport_normalizes_unreal_enum_repr_with_numeric_suffix():
    class UnrealEnum:
        name = "EV"

        def __str__(self):
            return "<LightUnits.EV: 3>"

    fake, _components = _transport_fixture()
    lights = []
    for label in environment.MANUAL_SPOTLIGHT_ACTOR_LABELS:
        component = _FakeComponent(intensity=5.0, intensity_units=UnrealEnum())
        lights.append(_FakeActor(label, "SpotLight", component))
    fake.actor_subsystem.actors.extend(lights)
    fake.SpotLightComponent = type("SpotLightComponent", (), {})
    transport = environment.UEEnvironmentTransport(fake)

    result = transport.read_manual_spot_lighting()

    assert result["intensity_units"] == "ev"


def test_ue_environment_transport_reuses_attached_mid_for_later_restore():
    fake, components = _transport_fixture()
    entry = {
        "field": "cloud_coverage",
        "target": "material",
        "property_name": "Cloud_GlobalCoverage",
    }
    applied = environment.UEEnvironmentTransport(fake)
    parent = components["VolumetricCloud"].get_editor_property("material")
    applied.write("VolumetricCloud", entry, 0.15)
    mid = components["VolumetricCloud"].get_editor_property("material")

    restorer = environment.UEEnvironmentTransport(fake)
    assert restorer.read("VolumetricCloud", entry) == 0.15
    restorer.write("VolumetricCloud", entry, -0.2)
    assert components["VolumetricCloud"].get_editor_property("material") is mid
    assert len(fake.created_mids) == 1
    restorer.release_material()
    assert components["VolumetricCloud"].get_editor_property("material") is parent


def test_ue_environment_transport_commit_recapture_and_clean_state():
    fake, components = _transport_fixture()
    transport = environment.UEEnvironmentTransport(fake)
    transport.write(
        "SkyLight",
        {"field": "intensity", "target": "component", "property_name": "intensity"},
        1.4,
    )
    transport.commit()
    assert components["SkyLight"].recaptured == 1
    assert transport.flush_dirty() == []
    assert [entry[0] for entry in fake.asset_subsystem.cleared] == ["SkyLight"]
    assert transport.read_clean_state({}) == {"level_clean": True, "packages": []}


def test_ue_environment_transport_restores_transient_fill_lights():
    fake, components = _transport_fixture()
    transport = environment.UEEnvironmentTransport(fake)

    transport.create_fill_lighting(
        {
            "enabled": True,
            "intensity": 9000.0,
            "intensity_units": "lumens",
            "attenuation_radius_cm": 2600.0,
            "source_width_cm": 350.0,
            "source_height_cm": 350.0,
            "light_color_rgba": [1.0, 1.0, 1.0, 1.0],
            "cast_shadows": False,
            "locations_cm": [[-2500.0, -1800.0, 800.0], [2500.0, -1800.0, 800.0], [-2500.0, 1800.0, 800.0], [2500.0, 1800.0, 800.0]],
        },
        "env_tx_fake",
    )

    assert len(fake.spawned_lights) == 4
    labels = [actor.get_actor_label() for actor in fake.spawned_lights]
    actual = transport.read_fill_lighting(labels)
    assert actual["intensity"] == 9000.0
    assert actual["intensity_units"] == "lumens"
    transport.release_fill_lighting(labels)
    assert len(fake.destroyed_lights) == 4


def test_transient_fill_light_readback_uses_process_registry():
    fake, components = _transport_fixture()
    transport = environment.UEEnvironmentTransport(fake)
    labels = transport.create_fill_lighting(
        {
            "enabled": True,
            "intensity": 9000.0,
            "intensity_units": "lumens",
            "attenuation_radius_cm": 2600.0,
            "source_width_cm": 350.0,
            "source_height_cm": 350.0,
            "light_color_rgba": [1.0, 1.0, 1.0, 1.0],
            "cast_shadows": False,
            "locations_cm": [[0.0, 0.0, 900.0]],
        },
        "env_tx_registry",
    )
    fake.actor_subsystem.actors = [
        actor for actor in fake.actor_subsystem.actors
        if not isinstance(actor, _FakeRectLightActor)
    ]

    actual = transport.read_fill_lighting(labels)

    assert actual["enabled"] is True
    assert len(actual["locations_cm"]) == 1
    transport.release_fill_lighting(labels)


def test_read_clean_state_detects_registry_transient_actor_missing_from_level_list():
    fake, components = _transport_fixture()
    transport = environment.UEEnvironmentTransport(fake)
    labels = transport.create_fill_lighting(
        {
            "enabled": True,
            "intensity": 9000.0,
            "intensity_units": "lumens",
            "attenuation_radius_cm": 2600.0,
            "source_width_cm": 350.0,
            "source_height_cm": 350.0,
            "light_color_rgba": [1.0, 1.0, 1.0, 1.0],
            "cast_shadows": False,
            "locations_cm": [[0.0, 0.0, 900.0]],
        },
        "env_tx_clean_state",
    )
    fake.actor_subsystem.actors = [
        actor for actor in fake.actor_subsystem.actors
        if not isinstance(actor, _FakeRectLightActor)
    ]
    snapshot = _snapshot()
    snapshot["manual_spot_lighting"]["actor_labels"] = list(
        environment.MANUAL_SPOTLIGHT_ACTOR_LABELS
    )
    snapshot["transient_fill_lighting"] = {
        "enabled": True,
        "actor_labels": labels,
    }

    assert transport.read_clean_state(snapshot)["level_clean"] is False
    transport.release_fill_lighting(labels)
    assert transport.read_clean_state(snapshot)["level_clean"] is False


def test_restore_episode_environment_retracks_environment_packages(monkeypatch, tmp_path):
    snapshot_path = tmp_path / "rollback.json"
    report_path = tmp_path / "application.json"
    snapshot = _snapshot()
    report = environment.build_environment_application_report(
        _manifest(),
        "env_tx_retrack",
        "applied",
        [
            {
                "actor_label": label,
                "actor_path": record["actor_path"],
                "class": record["class"],
                "package_path": record["package_path"],
                "apply_status": "applied",
                "restore_status": "not_started",
            }
            for label, record in _actors().items()
        ],
        [],
        [],
    )
    environment._write_json(snapshot_path, snapshot)
    environment._write_json(report_path, report)
    transport = type(
        "TrackingTransport",
        (),
        {
            "current_identity": lambda self: {
                "level": _level(),
                "actors": _actors(),
            },
            "track_environment_actors": lambda self: setattr(self, "tracked", True),
        },
    )()
    monkeypatch.setattr(environment, "UEEnvironmentTransport", lambda: transport)
    monkeypatch.setattr(
        environment,
        "restore_environment_snapshot",
        lambda snapshot, identity, transport: {
            "ok": True,
            "state": "restored",
            "errors": [],
        },
    )

    result = environment.restore_episode_environment(
        snapshot_path, report_path, "test"
    )

    assert result["ok"] is True
    assert transport.tracked is True
    restored_report = environment._read_json(report_path)
    assert all(record["restore_status"] == "restored" for record in restored_report["actors"])


def test_restore_transport_releases_fill_lights_created_by_another_adapter():
    fake, components = _transport_fixture()
    creator = environment.UEEnvironmentTransport(fake)
    labels = creator.create_fill_lighting(
        {
            "enabled": True,
            "intensity": 9000.0,
            "intensity_units": "lumens",
            "attenuation_radius_cm": 2600.0,
            "source_width_cm": 350.0,
            "source_height_cm": 350.0,
            "light_color_rgba": [1.0, 1.0, 1.0, 1.0],
            "cast_shadows": False,
            "locations_cm": [[0.0, 0.0, 900.0]],
        },
        "env_tx_other_adapter",
    )
    restorer = environment.UEEnvironmentTransport(fake)

    restorer.release_fill_lighting(labels)

    assert len(fake.destroyed_lights) == 1
    assert labels[0] not in environment.UEEnvironmentTransport._active_fill_lights


def test_ue_environment_transport_maps_sky_light_source_type_semantically():
    fake, components = _transport_fixture()
    transport = environment.UEEnvironmentTransport(fake)
    transport.write(
        "SkyLight",
        {"field": "source_type", "target": "component", "property_name": "source_type"},
        "specified_cube_or_scene",
    )
    assert str(components["SkyLight"].get_editor_property("source_type")) == (
        "SkyLightSourceType.SLS_CAPTURED_SCENE"
    )
    assert transport.read(
        "SkyLight",
        {"field": "source_type", "target": "component", "property_name": "source_type"},
    ) == "specified_cube_or_scene"

def test_environment_baseline_readback_verifies_snapshot_not_manifest():
    report = _diagnostic_with_mappings()
    plan = environment.build_environment_property_write_plan(_manifest(), report)
    snapshot = environment.build_environment_snapshot(
        level=_level(),
        actors=_actors(),
        baseline=_baseline(),
        manifest=_manifest(),
        transaction_id="env_tx_baseline",
        write_plan=plan["plan"],
    )

    matching = environment.verify_environment_baseline_readback(
        snapshot, deepcopy(_manifest()["parameters"])
    )
    assert matching["ok"] is True, matching["errors"]

    drifted_values = deepcopy(_manifest()["parameters"])
    drifted_values["directional_light"]["intensity"] = 99.0
    drifted = environment.verify_environment_baseline_readback(snapshot, drifted_values)
    assert drifted["ok"] is False
    assert any("DirectionalLight intensity" in error for error in drifted["errors"])
