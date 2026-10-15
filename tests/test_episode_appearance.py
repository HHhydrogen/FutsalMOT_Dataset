"""P2 appearance manifest 和 application report 纯契约测试。"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import subprocess
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest

from grf_ue_bridge.appearance import build_appearance_manifest
import episode_appearance as appearance_module
import diagnose_episode_appearance as appearance_diagnostic

from episode_appearance import (
    APPLICATION_REPORT_SCHEMA,
    APPLICATION_REPORT_VERSION,
    ENTITY_IDS,
    apply_episode_appearance,
    create_application_report,
    preflight_episode_appearance,
    read_application_report,
    restore_episode_appearance,
    serialize_application_report,
    transition_application_report,
    validate_appearance_inputs,
    write_application_report,
)


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def appearance_catalog() -> dict:
    return {
        "mode": "seeded",
        "policy": "futsalmot_appearance_v1",
        "catalog_id": "mpfb_players_003_009_v1",
        "body_texture_parameter": "SkinTexture",
        "models": [
            {
                "player_id": player_id,
                "mesh": "/Game/FutsalMOT/Characters/MPFB/Players/Player_{:03d}/MPFB_Player_{:03d}".format(player_id, player_id),
                "anim_class": "/Game/FutsalMOT/Characters/MPFB/Core/Retarget/ABP_MPFB_RuntimeRetarget",
                "skeleton": "/Game/FutsalMOT/Characters/MPFB/Core/Skeleton/SKEL_MPFB_Base",
            }
            for player_id in range(3, 10)
        ],
        "skins": [
            {
                "skin_id": "skin-{:02d}".format(index),
                "texture": "/Game/FutsalMOT/Characters/MPFB/Textures/Skins/Male/T_Skin_{:02d}".format(index),
            }
            for index in range(1, 10)
        ],
        "kits": [
            {
                "set_id": "{:02d}".format(index),
                "jersey_material": "/Game/FutsalMOT/Characters/MPFB/Materials/M_MPFB_Jersey_Set_{:02d}".format(index),
                "shorts_material": "/Game/FutsalMOT/Characters/MPFB/Materials/M_MPFB_Shorts_Set_{:02d}".format(index),
            }
            for index in range(1, 6)
        ],
    }


def appearance_meta() -> dict:
    return {
        "schema": "grf_ue_episode",
        "version": 1,
        "episode_id": "episode-appearance-test",
        "source": {"seed": 42},
        "randomness": {"root_seed": 42, "ue_visual_seed": 1001},
        "entities": [
            {
                "id": entity_id,
                "team": "left" if entity_id.startswith("L") else "right",
                "is_goalkeeper": entity_id in {"L2", "R1"},
            }
            for entity_id in ENTITY_IDS
        ],
    }


def write_frames(tmp_path: Path, payload: bytes = b'{"step":0}\n') -> Path:
    frames_path = tmp_path / "frames.jsonl"
    frames_path.write_bytes(payload)
    return frames_path


def valid_manifest(meta: dict, catalog: dict, frames_path: Path) -> dict:
    return build_appearance_manifest(meta, frames_path, catalog)


@pytest.fixture
def appearance_inputs(tmp_path):
    catalog = appearance_catalog()
    meta = appearance_meta()
    frames_path = write_frames(tmp_path)
    manifest = valid_manifest(meta, catalog, frames_path)
    mapping = {entity_id: "Player_{}".format(entity_id) for entity_id in ENTITY_IDS}
    return manifest, catalog, meta, frames_path, mapping


def mark_all_applied(report: dict) -> dict:
    return transition_application_report(
        report,
        "applied",
        actor_updates={entity_id: {"apply_status": "applied"} for entity_id in ENTITY_IDS},
    )


def test_p2_validator_accepts_current_manifest_and_inputs(appearance_inputs):
    manifest, catalog, meta, frames_path, _ = appearance_inputs

    assert validate_appearance_inputs(manifest, catalog, meta, frames_path) == []


@pytest.mark.parametrize(
    ("field", "value", "error_fragment"),
    [
        ("root_seed", 42.0, "root_seed"),
        ("ue_visual_seed", 1001.0, "ue_visual_seed"),
    ],
)
def test_p2_validator_rejects_non_exact_integer_seed_metadata(
    appearance_inputs, field, value, error_fragment
):
    manifest, catalog, meta, frames_path, _ = appearance_inputs
    changed_meta = copy.deepcopy(meta)
    if field == "root_seed":
        changed_meta["randomness"][field] = value
    else:
        changed_meta["randomness"][field] = value

    errors = validate_appearance_inputs(manifest, catalog, changed_meta, frames_path)

    assert any(error_fragment in error for error in errors)


@pytest.mark.parametrize("field", ["root_seed", "ue_visual_seed"])
def test_p2_validator_rejects_bool_seed_even_when_equal_to_manifest_integer(
    appearance_inputs, field
):
    _, catalog, meta, frames_path, _ = appearance_inputs
    changed_meta = copy.deepcopy(meta)
    if field == "root_seed":
        changed_meta["source"]["seed"] = 1
        changed_meta["randomness"]["root_seed"] = 1
    else:
        changed_meta["randomness"]["ue_visual_seed"] = 1
    manifest = build_appearance_manifest(changed_meta, frames_path, catalog)
    changed_meta["randomness"][field] = True

    errors = validate_appearance_inputs(manifest, catalog, changed_meta, frames_path)

    assert any(field in error for error in errors)


@pytest.mark.parametrize("fallback_seed", [42.0, True])
def test_p2_validator_rejects_non_exact_integer_source_seed_fallback(
    appearance_inputs, fallback_seed
):
    manifest, catalog, meta, frames_path, _ = appearance_inputs
    changed_meta = copy.deepcopy(meta)
    changed_meta["randomness"].pop("root_seed")
    changed_meta["source"]["seed"] = fallback_seed

    errors = validate_appearance_inputs(manifest, catalog, changed_meta, frames_path)

    assert any("root_seed" in error for error in errors)


def test_p2_validator_rejects_valid_catalog_assignment_with_recomputed_fingerprint(
    appearance_inputs,
):
    manifest, catalog, meta, frames_path, _ = appearance_inputs
    changed_manifest = copy.deepcopy(manifest)
    entity = changed_manifest["entities"][0]
    alternate_model = next(
        model for model in catalog["models"] if model["player_id"] != entity["player_id"]
    )
    entity.update(
        {
            "player_id": alternate_model["player_id"],
            "mesh": alternate_model["mesh"],
            "anim_class": alternate_model["anim_class"],
            "skeleton": alternate_model["skeleton"],
        }
    )
    changed_manifest["fingerprint"] = _canonical_sha256(
        {key: value for key, value in changed_manifest.items() if key != "fingerprint"}
    )

    errors = validate_appearance_inputs(
        changed_manifest, catalog, meta, frames_path
    )

    assert any("deterministic" in error or "ue_visual_seed" in error for error in errors)


def test_p2_manifest_and_appearance_diagnostic_import_without_unreal_in_subprocess():
    code = (
        "import builtins, json, sys; "
        "original = builtins.__import__; "
        "builtins.__import__ = lambda name, *a, **k: "
        "(_ for _ in ()).throw(ImportError('unreal unavailable')) if name == 'unreal' "
        "else original(name, *a, **k); "
        "import episode_appearance, diagnose_episode_appearance; "
        "print(json.dumps({'loaded': 'unreal' in sys.modules}))"
    )
    process = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(Path(__file__).resolve().parent.parent / "ue"),
        capture_output=True,
        text=True,
    )

    assert process.returncode == 0, process.stderr
    assert json.loads(process.stdout) == {"loaded": False}


@pytest.mark.parametrize("manifest", [None, {}])
def test_p2_validator_rejects_missing_manifest(appearance_inputs, manifest):
    _, catalog, meta, frames_path, _ = appearance_inputs

    errors = validate_appearance_inputs(manifest, catalog, meta, frames_path)

    assert errors
    assert any("manifest" in error.lower() for error in errors)


def test_p2_validator_rejects_stale_trajectory_hash(appearance_inputs):
    manifest, catalog, meta, frames_path, _ = appearance_inputs
    frames_path.write_bytes(b'{"step":0,"changed":true}\n')

    errors = validate_appearance_inputs(manifest, catalog, meta, frames_path)

    assert any("trajectory_sha256" in error for error in errors)


def test_p2_validator_rejects_missing_frames_file(appearance_inputs, tmp_path):
    manifest, catalog, meta, _, _ = appearance_inputs

    errors = validate_appearance_inputs(
        manifest, catalog, meta, tmp_path / "missing-frames.jsonl"
    )

    assert any("frames.jsonl" in error for error in errors)


def test_p2_validator_rejects_catalog_mismatch(appearance_inputs):
    manifest, catalog, meta, frames_path, _ = appearance_inputs
    changed_catalog = copy.deepcopy(catalog)
    changed_catalog["models"][0]["mesh"] += "_different"

    errors = validate_appearance_inputs(manifest, changed_catalog, meta, frames_path)

    assert any("catalog_fingerprint" in error for error in errors)


def test_p2_validator_rejects_invalid_catalog_asset_path(appearance_inputs):
    manifest, catalog, meta, frames_path, _ = appearance_inputs
    changed_catalog = copy.deepcopy(catalog)
    changed_catalog["skins"][0]["texture"] = r"C:\Skins\skin.png"

    errors = validate_appearance_inputs(manifest, changed_catalog, meta, frames_path)

    assert any("/Game/" in error for error in errors)


def test_p2_validator_rejects_entity_set_mismatch(appearance_inputs):
    manifest, catalog, meta, frames_path, _ = appearance_inputs
    changed_manifest = copy.deepcopy(manifest)
    changed_manifest["entities"].pop()
    changed_manifest["fingerprint"] = _canonical_sha256(
        {key: value for key, value in changed_manifest.items() if key != "fingerprint"}
    )

    errors = validate_appearance_inputs(changed_manifest, catalog, meta, frames_path)

    assert any("entities" in error.lower() for error in errors)


def test_p2_validator_rejects_stale_entity_metadata(appearance_inputs):
    manifest, catalog, meta, frames_path, _ = appearance_inputs
    changed_meta = copy.deepcopy(meta)
    changed_meta["entities"][0]["is_goalkeeper"] = True

    errors = validate_appearance_inputs(manifest, catalog, changed_meta, frames_path)

    assert any("meta.entities" in error for error in errors)


def test_p2_validator_rejects_invalid_asset_paths(appearance_inputs):
    manifest, catalog, meta, frames_path, _ = appearance_inputs
    changed_manifest = copy.deepcopy(manifest)
    changed_manifest["entities"][0]["mesh"] = r"C:\MPFB\mesh.uasset"
    changed_manifest["fingerprint"] = _canonical_sha256(
        {key: value for key, value in changed_manifest.items() if key != "fingerprint"}
    )

    errors = validate_appearance_inputs(changed_manifest, catalog, meta, frames_path)

    assert any("/Game/" in error or "asset" in error.lower() for error in errors)


def test_preflight_requires_exact_unique_player_mapping(appearance_inputs):
    manifest, catalog, _, _, mapping = appearance_inputs
    invalid_mapping = dict(mapping)
    invalid_mapping["R4"] = invalid_mapping["L0"]

    result = preflight_episode_appearance(manifest, catalog, invalid_mapping)

    assert result["ok"] is False
    assert any("mapping" in error.lower() for error in result["errors"])


def test_preflight_returns_all_ten_resolved_editor_actors(appearance_inputs, monkeypatch):
    manifest, catalog, _, _, mapping = appearance_inputs
    actor_map = {
        entity_id: SimpleNamespace(
            get_actor_label=lambda label=mapping[entity_id]: label,
            get_path_name=lambda label=mapping[entity_id]: "/Game/__ExternalActors__/{}".format(label),
        )
        for entity_id in ENTITY_IDS
    }
    monkeypatch.setattr(appearance_module, "_load_unreal", lambda: object())
    monkeypatch.setattr(
        appearance_module,
        "_ue_preflight",
        lambda *_args: {
            "ok": True,
            "errors": [],
            "actors": actor_map,
            "dirty_paths": set(),
            "loaded": {},
        },
    )

    result = preflight_episode_appearance(manifest, catalog, mapping)

    assert result["ok"] is True
    assert len(result["actors"]) == 10
    assert {actor["entity_id"] for actor in result["actors"]} == set(ENTITY_IDS)
    assert all("actor_path" in actor for actor in result["actors"])


def test_preflight_stops_before_ue_for_invalid_pure_contract(appearance_inputs, monkeypatch):
    manifest, catalog, _, _, mapping = appearance_inputs
    invalid_mapping = dict(mapping)
    invalid_mapping["R4"] = invalid_mapping["L0"]
    monkeypatch.setattr(
        appearance_module,
        "_load_unreal",
        lambda: pytest.fail("UE must not load after pure preflight failure"),
    )

    result = preflight_episode_appearance(manifest, catalog, invalid_mapping)

    assert result["ok"] is False
    assert any("mapping" in error.lower() for error in result["errors"])


def test_appearance_safety_gate_requires_formal_map_and_exact_ten_player_roster():
    valid_roster = list(ENTITY_IDS)

    assert appearance_module._appearance_safety_gate(
        "/Game/FutsalMOT/Maps/L_FutsalCourt", valid_roster
    ) == []

    wrong_map_errors = appearance_module._appearance_safety_gate(
        "/Temp/Untitled_0", valid_roster
    )
    assert any("/Game/FutsalMOT/Maps/L_FutsalCourt" in error for error in wrong_map_errors)

    incomplete_errors = appearance_module._appearance_safety_gate(
        "/Game/FutsalMOT/Maps/L_FutsalCourt", valid_roster[:-1]
    )
    assert any("R4" in error and "roster" in error.lower() for error in incomplete_errors)

    duplicate_errors = appearance_module._appearance_safety_gate(
        "/Game/FutsalMOT/Maps/L_FutsalCourt", valid_roster + ["L0"]
    )
    assert any("L0" in error and "重复" in error for error in duplicate_errors)


def test_pre_mutation_roster_check_rejects_level_change_and_actor_identity_change(
    appearance_inputs, monkeypatch
):
    _, _, _, _, mapping = appearance_inputs
    old_actors = {
        entity_id: SimpleNamespace(get_path_name=lambda value=entity_id: "/Game/Old/{}".format(value))
        for entity_id in ENTITY_IDS
    }
    new_actors = {
        entity_id: SimpleNamespace(
            get_path_name=lambda value=entity_id: "/Game/New/{}".format(value)
        )
        for entity_id in ENTITY_IDS
    }
    monkeypatch.setattr(appearance_module, "_current_level_path", lambda _ue: "/Temp/Untitled_0")
    monkeypatch.setattr(appearance_module, "_actor_matches", lambda _ue, _mapping: (new_actors, []))

    _, errors = appearance_module._verify_pre_mutation_roster(
        object(), mapping, old_actors
    )

    assert any("正式地图" in error for error in errors)
    assert any("identity" in error for error in errors)


def test_restore_roster_safety_gate_stops_before_actor_lookup_or_asset_load(monkeypatch):
    monkeypatch.setattr(
        appearance_module,
        "_current_level_path",
        lambda _ue: "/Temp/Untitled_0",
    )
    monkeypatch.setattr(
        appearance_module,
        "_all_level_actors",
        lambda _ue: pytest.fail("不安全地图下不得查找或修改 restore Actor"),
    )

    errors = appearance_module._restore_roster(
        object(),
        [{"entity_id": entity_id} for entity_id in ENTITY_IDS],
        [],
    )

    assert any("正式地图" in error for error in errors)


def test_appearance_diagnostic_contract_requires_formal_map_roster_and_asset_checks():
    mapping = {entity_id: "Player_{}".format(entity_id) for entity_id in ENTITY_IDS}
    actors = {
        entity_id: {
            "expected_label": mapping[entity_id],
            "actor_label": mapping[entity_id],
            "actor_path": "/Game/Map.Player_{}".format(entity_id),
            "package_path": "/Game/__ExternalActors__/Player_{}".format(entity_id),
        }
        for entity_id in ENTITY_IDS
    }
    asset_checks = [
        {"path": "/Game/Asset", "expected_class": "Material", "exists": True, "actual_class": "Material"}
    ]

    report = appearance_diagnostic.build_diagnostic_report(
        "/Game/FutsalMOT/Maps/L_FutsalCourt", mapping, actors, [], asset_checks
    )

    assert report["formal_map_match"] is True
    assert report["roster"]["ok"] is True
    assert report["key_asset_checks_ok"] is True
    assert report["task7_preflight_can_start"] is True
    assert len(report["roster"]["actors"]) == 10

    actors.pop("R4")
    blocked = appearance_diagnostic.build_diagnostic_report(
        "/Temp/Untitled_0", mapping, actors, [], asset_checks
    )
    assert blocked["formal_map_match"] is False
    assert blocked["roster"]["ok"] is False
    assert blocked["task7_preflight_can_start"] is False

    duplicate_mapping = dict(mapping)
    duplicate_mapping["R4"] = duplicate_mapping["L0"]
    duplicate_actor_records = copy.deepcopy(actors)
    duplicate_actor_records["R4"] = copy.deepcopy(duplicate_actor_records["L0"])
    duplicate_mapping_report = appearance_diagnostic.build_diagnostic_report(
        "/Game/FutsalMOT/Maps/L_FutsalCourt",
        duplicate_mapping,
        duplicate_actor_records,
        [],
        asset_checks,
    )
    assert duplicate_mapping_report["roster"]["ok"] is False
    assert duplicate_mapping_report["task7_preflight_can_start"] is False


def test_appearance_diagnostic_collects_malformed_mapping_as_structured_failure():
    unreal = SimpleNamespace(
        EditorLevelLibrary=SimpleNamespace(
            get_editor_world=lambda: SimpleNamespace(
                get_path_name=lambda: "/Game/FutsalMOT/Maps/L_FutsalCourt.L_FutsalCourt"
            )
        ),
        EditorActorSubsystem=object,
        get_editor_subsystem=lambda _subsystem: SimpleNamespace(
            get_all_level_actors=lambda: []
        ),
        EditorLoadingAndSavingUtils=SimpleNamespace(
            get_dirty_map_packages=lambda: [], get_dirty_content_packages=lambda: []
        ),
        EditorAssetLibrary=SimpleNamespace(
            find_asset_data=lambda _path: SimpleNamespace(is_valid=lambda: False),
        ),
    )

    report = appearance_diagnostic.collect_diagnostic(unreal, None)

    assert report["task7_preflight_can_start"] is False
    assert any("actor mapping 必须" in error for error in report["errors"])


def test_appearance_asset_checks_use_asset_data_class_path_and_isolate_failures(
    monkeypatch,
):
    specs = (
        ("mesh", "valid", "/Game/Valid", "SkeletalMesh"),
        ("mesh", "invalid", "/Game/Invalid", "SkeletalMesh"),
        ("mesh", "exception", "/Game/Exception", "SkeletalMesh"),
    )

    class AssetData:
        def __init__(self, valid, asset_name):
            self.asset_class_path = SimpleNamespace(asset_name=asset_name)
            self._valid = valid

        def is_valid(self):
            return self._valid

    class AssetLibrary:
        @staticmethod
        def find_asset_data(path):
            if path == "/Game/Exception":
                raise RuntimeError("metadata lookup failed")
            if path == "/Game/Invalid":
                return AssetData(False, "")
            return AssetData(True, "SkeletalMesh")

    monkeypatch.setattr(appearance_diagnostic, "_ASSET_SPECS", specs)
    checks = appearance_diagnostic._asset_checks(
        SimpleNamespace(EditorAssetLibrary=AssetLibrary)
    )

    assert checks[0]["exists"] is True
    assert checks[0]["actual_class"] == "SkeletalMesh"
    assert checks[1]["exists"] is False
    assert checks[1]["actual_class"] is None
    assert "AssetData" in checks[1]["error"]
    assert "metadata lookup failed" in checks[2]["error"]


def test_appearance_diagnostic_collects_current_editor_state_without_world_loading():
    mapping = {entity_id: "Player_{}".format(entity_id) for entity_id in ENTITY_IDS}

    class Actor:
        def __init__(self, entity_id):
            self.entity_id = entity_id

        def get_actor_label(self):
            return mapping[self.entity_id]

        def get_path_name(self):
            return "/Game/Maps/L_FutsalCourt.L_FutsalCourt:PersistentLevel.{}".format(
                mapping[self.entity_id]
            )

        def get_outermost(self):
            return SimpleNamespace(
                get_path_name=lambda: "/Game/__ExternalActors__/{}".format(
                    mapping[self.entity_id]
                )
            )

    class AssetLibrary:
        @staticmethod
        def does_asset_exist(_path):
            return pytest.fail("资产检查应使用 find_asset_data")

        @staticmethod
        def get_asset_class(path):
            return pytest.fail("资产检查应读取 AssetData class path")

        @staticmethod
        def find_asset_data(path):
            if path == "/Game/FutsalMOT/Maps/L_FutsalCourt":
                asset_class = "World"
            elif "/Players/" in path:
                asset_class = "SkeletalMesh"
            elif "/Textures/" in path:
                asset_class = "Texture2D"
            elif "/Skeleton/" in path:
                asset_class = "Skeleton"
            elif "/Retarget/" in path:
                asset_class = "AnimBlueprint"
            else:
                asset_class = "Material"
            return SimpleNamespace(
                is_valid=lambda: True,
                asset_class_path=SimpleNamespace(asset_name=asset_class),
            )

    unreal = SimpleNamespace(
        EditorLevelLibrary=SimpleNamespace(
            get_editor_world=lambda: SimpleNamespace(
                get_path_name=lambda: "/Game/FutsalMOT/Maps/L_FutsalCourt.L_FutsalCourt"
            )
        ),
        EditorActorSubsystem=object,
        EditorAssetLibrary=AssetLibrary,
        EditorLoadingAndSavingUtils=SimpleNamespace(
            get_dirty_map_packages=lambda: [], get_dirty_content_packages=lambda: []
        ),
        get_editor_subsystem=lambda _subsystem: SimpleNamespace(
            get_all_level_actors=lambda: [Actor(entity_id) for entity_id in ENTITY_IDS]
        ),
        load_asset=lambda *_args: pytest.fail("诊断不得加载任何资产对象"),
        load_map=lambda *_args: pytest.fail("诊断不得切换地图"),
    )
    report = appearance_diagnostic.collect_diagnostic(unreal, mapping)

    assert report["task7_preflight_can_start"] is True
    assert report["current_level"] == "/Game/FutsalMOT/Maps/L_FutsalCourt"
    assert len(report["roster"]["actors"]) == 10


def _actor_snapshot(entity_id: str, override_materials: list) -> dict:
    return {
        "entity_id": entity_id,
        "actor_label": "Player_{}".format(entity_id),
        "actor_path": "/Game/Maps/L_FutsalCourt.L_FutsalCourt:PersistentLevel.BP_Player_{}".format(entity_id),
        "package_path": "/Game/__ExternalActors__/FutsalMOT/Maps/L_FutsalCourt/Player_{}".format(entity_id),
        "component_name": "MPFBVisual",
        "skeletal_mesh": "/Game/FutsalMOT/Characters/MPFB/Players/Player_003/MPFB_Player_003",
        "anim_class": "/Game/FutsalMOT/Characters/MPFB/Core/Retarget/ABP_MPFB_RuntimeRetarget.ABP_MPFB_RuntimeRetarget_C",
        "attach_parent": "/Game/Map.Player.CharacterMesh0",
        "actor_transform": {
            "location": [10.0, 20.0, 30.0],
            "rotation": [1.0, 2.0, 3.0],
            "scale": [1.0, 1.0, 1.0],
        },
        "relative_transform": {
            "location": [0.0, 0.0, 0.0],
            "rotation": [0.0, 0.0, 0.0],
            "scale": [1.0, 1.0, 1.0],
        },
        "override_materials": override_materials,
        "leader_pose_component": None,
        "visibility": {
            "visible": True,
            "render_in_main_pass": True,
            "render_in_depth_pass": True,
        },
        "package_dirty": False,
    }


def test_snapshot_preserves_exact_override_array_including_none_entries():
    overrides = [
        None,
        "/Game/FutsalMOT/Characters/MPFB/Players/Player_003/Materials/M_Player003_Hair",
        None,
    ]
    source_state = _actor_snapshot("L0", overrides)

    snapshot = appearance_module._build_actor_snapshot(source_state)

    assert snapshot["override_materials"] == overrides
    assert snapshot["override_materials"] is not overrides
    assert snapshot["override_materials"][0] is None
    assert snapshot["override_materials"][2] is None
    assert snapshot["package_dirty"] is False


def test_snapshot_roster_requires_every_player_once_and_complete_state():
    snapshots = [_actor_snapshot(entity_id, [None] * 7) for entity_id in ENTITY_IDS]

    assert appearance_module._validate_actor_snapshots(snapshots) == []

    incomplete = copy.deepcopy(snapshots)
    incomplete[0].pop("override_materials")
    errors = appearance_module._validate_actor_snapshots(incomplete)

    assert any("override_materials" in error for error in errors)

    missing_world_transform = copy.deepcopy(snapshots)
    missing_world_transform[0].pop("actor_transform")
    errors = appearance_module._validate_actor_snapshots(missing_world_transform)

    assert any("actor_transform" in error for error in errors)


def test_snapshot_roster_rejects_duplicate_or_missing_entity():
    snapshots = [_actor_snapshot(entity_id, [None] * 7) for entity_id in ENTITY_IDS]
    snapshots[-1]["entity_id"] = "L0"

    errors = appearance_module._validate_actor_snapshots(snapshots)

    assert any("entity" in error.lower() for error in errors)


def test_snapshot_roster_rejects_duplicate_actor_identity_and_dirty_baseline():
    snapshots = [_actor_snapshot(entity_id, [None] * 7) for entity_id in ENTITY_IDS]
    snapshots[1]["actor_path"] = snapshots[0]["actor_path"]
    snapshots[2]["package_dirty"] = True

    errors = appearance_module._validate_actor_snapshots(snapshots)

    assert any("actor_path" in error for error in errors)
    assert any("package_dirty" in error for error in errors)


def test_snapshot_keeps_actor_object_and_external_actor_package_paths_distinct():
    snapshot = _actor_snapshot("L0", [None] * 7)

    assert appearance_module._validate_actor_snapshots(
        [
            _actor_snapshot(entity_id, [None] * 7)
            for entity_id in ENTITY_IDS
        ]
    ) == []
    assert snapshot["actor_path"].startswith("/Game/Maps/")
    assert snapshot["package_path"].startswith("/Game/__ExternalActors__/")


@pytest.mark.parametrize("bad_path", [None, "/Game/Maps/L_FutsalCourt.L_FutsalCourt"])
def test_snapshot_roster_rejects_missing_or_non_external_package_path(bad_path):
    snapshots = [_actor_snapshot(entity_id, [None] * 7) for entity_id in ENTITY_IDS]
    if bad_path is None:
        snapshots[0].pop("package_path")
    else:
        snapshots[0]["package_path"] = bad_path

    errors = appearance_module._validate_actor_snapshots(snapshots)

    assert any("package_path" in error for error in errors)


def test_package_path_helper_uses_actor_outermost_package_identity():
    package = SimpleNamespace(
        get_path_name=lambda: "/Game/__ExternalActors__/FutsalMOT/Maps/L_FutsalCourt/Actor"
    )
    actor = SimpleNamespace(
        get_path_name=lambda: "/Game/Maps/L_FutsalCourt.L_FutsalCourt:PersistentLevel.Actor",
        get_outermost=lambda: package,
    )

    assert appearance_module._actor_package_path(actor) == package.get_path_name()


def test_actor_transform_preflight_requires_world_readback_and_restore_methods():
    methods = (
        "get_actor_location",
        "get_actor_rotation",
        "get_actor_scale3d",
        "set_actor_location_and_rotation",
        "set_actor_scale3d",
    )
    complete_actor_api = SimpleNamespace(**{name: lambda *args: None for name in methods})
    incomplete_actor_api = SimpleNamespace(
        **{name: lambda *args: None for name in methods if name != "set_actor_scale3d"}
    )

    assert appearance_module._actor_transform_api_errors(complete_actor_api) == []
    assert any(
        "set_actor_scale3d" in error
        for error in appearance_module._actor_transform_api_errors(incomplete_actor_api)
    )


def test_snapshot_roster_requires_all_captured_render_flags():
    snapshots = [_actor_snapshot(entity_id, [None] * 7) for entity_id in ENTITY_IDS]
    snapshots[0]["visibility"].pop("render_in_depth_pass")

    errors = appearance_module._validate_actor_snapshots(snapshots)

    assert any("render_in_depth_pass" in error for error in errors)


def test_snapshot_comparison_covers_override_array_transform_and_dirty_state():
    snapshot = _actor_snapshot("L0", [None, "/Game/Original.Hair", None])
    current = copy.deepcopy(snapshot)

    assert appearance_module._snapshot_matches_state(snapshot, current)

    current["override_materials"] = [None, None, None]
    assert not appearance_module._snapshot_matches_state(snapshot, current)

    current = copy.deepcopy(snapshot)
    current["relative_transform"]["location"][0] = 1.0
    assert not appearance_module._snapshot_matches_state(snapshot, current)

    current = copy.deepcopy(snapshot)
    current["actor_transform"]["location"][0] += 1.0
    assert not appearance_module._snapshot_matches_state(snapshot, current)

    current = copy.deepcopy(snapshot)
    current["package_dirty"] = True
    assert not appearance_module._snapshot_matches_state(snapshot, current)


def test_dirty_reset_candidates_exclude_failed_component_restore_or_readback():
    restored = {entity_id: True for entity_id in ENTITY_IDS}
    readback = {entity_id: True for entity_id in ENTITY_IDS}
    restored["L1"] = False
    readback["R2"] = False

    candidates = appearance_module._dirty_reset_candidates(
        ENTITY_IDS, restored, readback
    )

    assert candidates == {
        entity_id for entity_id in ENTITY_IDS if entity_id not in {"L1", "R2"}
    }


def test_asset_reload_candidates_are_clean_baseline_and_currently_dirty_intersection():
    baseline = [
        {"package_path": "/Game/Materials/CleanA", "package_dirty": False},
        {"package_path": "/Game/Materials/CleanB", "package_dirty": False},
        {"package_path": "/Game/Materials/WasDirty", "package_dirty": True},
    ]

    assert appearance_module._asset_package_reload_candidates(
        baseline,
        {
            "/Game/Materials/CleanA",
            "/Game/Materials/WasDirty",
            "/Game/Materials/Unrelated",
        },
    ) == {"/Game/Materials/CleanA"}


def test_asset_package_baseline_rejects_dirty_duplicate_and_invalid_entries():
    dirty_baseline = [
        {"package_path": "/Game/Materials/AlreadyDirty", "package_dirty": True}
    ]
    duplicate_baseline = [
        {"package_path": "/Game/Materials/Same", "package_dirty": False},
        {"package_path": "/Game/Materials/Same", "package_dirty": False},
    ]
    invalid_path_baseline = [
        {"package_path": "C:/Materials/Local", "package_dirty": False}
    ]

    assert any(
        "dirty" in error.lower()
        for error in appearance_module._validate_asset_package_baseline(dirty_baseline)
    )
    assert any(
        "duplicate" in error.lower()
        for error in appearance_module._validate_asset_package_baseline(duplicate_baseline)
    )
    assert appearance_module._validate_asset_package_baseline(invalid_path_baseline)


def test_asset_package_baseline_rejects_empty_section():
    assert appearance_module._validate_asset_package_baseline([])


def test_reloadable_asset_package_filter_excludes_external_actor_uobjects():
    content_asset = SimpleNamespace(
        get_outermost=lambda: SimpleNamespace(
            get_path_name=lambda: "/Game/FutsalMOT/Materials/M_MPFB_Body"
        )
    )
    attachment_component = SimpleNamespace(
        get_outermost=lambda: SimpleNamespace(
            get_path_name=lambda: "/Game/__ExternalActors__/FutsalMOT/Maps/L_FutsalCourt/C/C7/2XUU639P3OUDQ5AAYW0AE2"
        )
    )

    assert appearance_module._reloadable_asset_package_path(
        "/Game/FutsalMOT/Materials/M_MPFB_Body", content_asset
    ) == "/Game/FutsalMOT/Materials/M_MPFB_Body"
    assert appearance_module._reloadable_asset_package_path(
        "/Game/Maps/L_FutsalCourt.L_FutsalCourt:PersistentLevel.CharacterMesh0",
        attachment_component,
    ) is None


def test_selected_asset_paths_include_ten_player_assignments_and_body_only(
    appearance_inputs,
):
    manifest, catalog, _, _, _ = appearance_inputs
    paths = appearance_module._selected_appearance_asset_paths(manifest)

    expected = {
        entity[field]
        for entity in manifest["entities"]
        for field in appearance_module._ENTITY_ASSET_FIELDS
    }
    expected.add(appearance_module._BODY_MATERIAL)
    all_catalog_paths = {
        item[field]
        for group, fields in (
            (catalog["models"], ("mesh", "anim_class", "skeleton")),
            (catalog["skins"], ("texture",)),
            (catalog["kits"], ("jersey_material", "shorts_material")),
        )
        for item in group
        for field in fields
    }

    assert paths == expected
    assert paths < all_catalog_paths | {appearance_module._BODY_MATERIAL}


def test_local_rollback_payload_contains_asset_package_baseline(appearance_inputs):
    manifest, _, _, _, mapping = appearance_inputs
    snapshots = [_actor_snapshot(entity_id, [None] * 7) for entity_id in ENTITY_IDS]
    asset_packages = [
        {"package_path": "/Game/FutsalMOT/Materials/M_MPFB_Body", "package_dirty": False}
    ]

    payload = appearance_module._build_rollback_payload(
        manifest["fingerprint"],
        snapshots,
        asset_packages,
    )

    assert payload["asset_packages"] == asset_packages
    assert "asset_packages" not in json.loads(
        serialize_application_report(create_application_report(manifest, mapping))
    )


def test_transaction_id_binds_asset_package_baseline():
    snapshots = [_actor_snapshot(entity_id, [None] * 7) for entity_id in ENTITY_IDS]
    baseline = [{"package_path": "/Game/Materials/Jersey", "package_dirty": False}]

    first = appearance_module._snapshot_transaction_id("a" * 64, snapshots, baseline)
    changed = appearance_module._snapshot_transaction_id(
        "a" * 64,
        snapshots,
        [{"package_path": "/Game/Materials/Shorts", "package_dirty": False}],
    )

    assert first != changed


def test_rollback_snapshot_validation_requires_clean_asset_package_baseline(
    appearance_inputs,
):
    manifest, _, _, _, _ = appearance_inputs
    snapshots = [_actor_snapshot(entity_id, [None] * 7) for entity_id in ENTITY_IDS]
    payload = appearance_module._build_rollback_payload(
        manifest["fingerprint"],
        snapshots,
        [{"package_path": "/Game/Materials/Jersey", "package_dirty": False}],
    )

    assert appearance_module._validate_rollback_payload(payload) == []
    payload["asset_packages"][0]["package_dirty"] = True
    assert any(
        "dirty" in error.lower()
        for error in appearance_module._validate_rollback_payload(payload)
    )


def test_transaction_id_binds_manifest_and_exact_override_snapshot():
    snapshots = [_actor_snapshot(entity_id, [None] * 7) for entity_id in ENTITY_IDS]
    first = appearance_module._snapshot_transaction_id("a" * 64, snapshots)

    changed = copy.deepcopy(snapshots)
    changed[0]["override_materials"][3] = "/Game/Original/Material.MI_Original"
    second = appearance_module._snapshot_transaction_id("a" * 64, changed)
    changed_world_transform = copy.deepcopy(snapshots)
    changed_world_transform[0]["actor_transform"]["location"][0] += 1.0
    third = appearance_module._snapshot_transaction_id(
        "a" * 64, changed_world_transform
    )
    different_manifest = appearance_module._snapshot_transaction_id("b" * 64, snapshots)

    assert len(first) == 64
    assert first != second
    assert first != third
    assert first != different_manifest


def test_restore_component_applies_actor_world_transform_before_visual_state():
    events = []

    class Asset:
        def __init__(self, path):
            self.path = path

        def get_path_name(self):
            return self.path

    class Actor:
        def __init__(self):
            self.world = {
                "location": [0.0, 0.0, 0.0],
                "rotation": [0.0, 0.0, 0.0],
                "scale": [1.0, 1.0, 1.0],
            }

        def get_components_by_class(self, _component_class):
            return [component]

        def get_actor_label(self):
            return "Player_L0"

        def get_path_name(self):
            return snapshot["actor_path"]

        def get_outermost(self):
            return SimpleNamespace(get_path_name=lambda: snapshot["package_path"])

        def get_actor_location(self):
            return SimpleNamespace(**dict(zip(("x", "y", "z"), self.world["location"])))

        def get_actor_rotation(self):
            return SimpleNamespace(**dict(zip(("pitch", "yaw", "roll"), self.world["rotation"])))

        def get_actor_scale3d(self):
            return SimpleNamespace(**dict(zip(("x", "y", "z"), self.world["scale"])))

        def set_actor_location_and_rotation(self, location, rotation, _sweep, _teleport):
            events.append("actor_transform")
            self.world["location"] = [location.x, location.y, location.z]
            self.world["rotation"] = [rotation.pitch, rotation.yaw, rotation.roll]

        def set_actor_scale3d(self, scale):
            self.world["scale"] = [scale.x, scale.y, scale.z]

    class Component:
        def __init__(self):
            self.parent = object()
            self.mesh = None
            self.properties = {
                "anim_class": None,
                "relative_location": None,
                "relative_rotation": None,
                "relative_scale3d": None,
                "override_materials": [],
                "leader_pose_component": None,
                "visible": True,
                "render_in_main_pass": True,
                "render_in_depth_pass": True,
            }

        def get_name(self):
            return "MPFBVisual"

        def get_editor_property(self, name):
            return self.properties[name]

        def get_attach_parent(self):
            return self.parent

        def get_skeletal_mesh_asset(self):
            return self.mesh

        def set_skeletal_mesh_asset(self, value):
            events.append("visual_mesh")
            self.mesh = value

        def set_editor_property(self, name, value):
            events.append("visual_" + name)
            self.properties[name] = value

    snapshot = _actor_snapshot("L0", [None] * 7)
    mesh = Asset(snapshot["skeletal_mesh"] + ".MPFB_Player_003")
    anim_class = Asset(snapshot["anim_class"])
    parent = SimpleNamespace(get_path_name=lambda: snapshot_parent_path)
    component = Component()
    component.parent = parent
    actor = Actor()
    snapshot_parent_path = snapshot["attach_parent"]
    snapshot.update(
        {
            "skeletal_mesh": mesh.get_path_name(),
            "actor_transform": {
                "location": [101.0, 202.0, 303.0],
                "rotation": [4.0, 5.0, 6.0],
                "scale": [1.0, 1.0, 1.0],
            },
            "relative_transform": {
                "location": [7.0, 8.0, 9.0],
                "rotation": [10.0, 11.0, 12.0],
                "scale": [1.0, 1.0, 1.0],
            },
        }
    )
    assets = {snapshot["skeletal_mesh"]: mesh, snapshot["anim_class"]: anim_class}
    object_by_path = {snapshot["attach_parent"]: parent}
    unreal = SimpleNamespace(
        SkeletalMeshComponent=object,
        Vector=lambda x, y, z: SimpleNamespace(x=x, y=y, z=z),
        Rotator=lambda pitch, yaw, roll: SimpleNamespace(
            pitch=pitch, yaw=yaw, roll=roll
        ),
        AttachmentRule=SimpleNamespace(KEEP_RELATIVE=object()),
        DetachmentRule=SimpleNamespace(KEEP_RELATIVE=object()),
    )

    appearance_module._restore_component(
        unreal, actor, snapshot, assets, object_by_path
    )

    assert actor.world == snapshot["actor_transform"]
    assert events[0] == "actor_transform"
    assert events.index("actor_transform") < events.index("visual_mesh")
    assert component.properties["relative_location"].x == 7.0
    readback = appearance_module._component_state(actor, component, False)
    readback["entity_id"] = snapshot["entity_id"]
    assert readback["actor_transform"] == snapshot["actor_transform"]
    assert appearance_module._snapshot_matches_state(snapshot, readback)


def test_transform_restore_preserves_snapshot_pitch_yaw_roll_with_ue_rotator_order():
    class UEStyleRotator:
        def __init__(self, *args, **kwargs):
            if args:
                self.roll, self.pitch, self.yaw = args
            else:
                self.pitch = kwargs["pitch"]
                self.yaw = kwargs["yaw"]
                self.roll = kwargs["roll"]

    class Actor:
        def __init__(self):
            self.rotation = None

        def set_actor_location_and_rotation(self, _location, rotation, _sweep, _teleport):
            self.rotation = rotation

        def set_actor_scale3d(self, _scale):
            pass

    class Component:
        def __init__(self):
            self.rotation = None

        def set_editor_property(self, name, value):
            if name == "relative_rotation":
                self.rotation = value

    unreal = SimpleNamespace(
        Vector=lambda x, y, z: SimpleNamespace(x=x, y=y, z=z),
        Rotator=UEStyleRotator,
    )
    transform = {
        "location": [1.0, 2.0, 3.0],
        "rotation": [4.0, 5.0, 6.0],
        "scale": [1.0, 1.0, 1.0],
    }
    actor = Actor()
    component = Component()

    appearance_module._set_actor_transform(unreal, actor, transform)
    appearance_module._set_component_transform(unreal, component, transform)

    assert (actor.rotation.pitch, actor.rotation.yaw, actor.rotation.roll) == (4.0, 5.0, 6.0)
    assert (component.rotation.pitch, component.rotation.yaw, component.rotation.roll) == (
        4.0,
        5.0,
        6.0,
    )


def test_snapshot_asset_preload_paths_include_baseline_refs_but_skip_none_overrides():
    snapshot = _actor_snapshot(
        "L0",
        [None, "/Game/Original/M_PlayerHair.M_PlayerHair", None],
    )

    assert appearance_module._snapshot_asset_paths([snapshot]) == {
        snapshot["skeletal_mesh"],
        snapshot["anim_class"],
        "/Game/Map.Player.CharacterMesh0",
        "/Game/Original/M_PlayerHair.M_PlayerHair",
    }


def test_generated_anim_class_path_normalizes_to_blueprint_asset_path():
    assert appearance_module._anim_blueprint_asset_path(
        "/Game/FutsalMOT/Characters/MPFB/Core/Retarget/ABP_MPFB_RuntimeRetarget_C"
    ) == "/Game/FutsalMOT/Characters/MPFB/Core/Retarget/ABP_MPFB_RuntimeRetarget"


def test_candidate_compatibility_checks_shared_assets_and_semantic_slots():
    mesh = SimpleNamespace(
        skeleton=SimpleNamespace(
            get_path_name=lambda: "/Game/FutsalMOT/Characters/MPFB/Core/Skeleton/SKEL_MPFB_Base.SKEL_MPFB_Base"
        ),
        get_editor_property=lambda name: [
            SimpleNamespace(get_editor_property=lambda field, slot=slot: slot)
            for slot in (
                "Human_body",
                "Human_eyebrow001",
                "材质",
                "Human_futsalmot_shorts_male_final",
                "Human_low-poly",
                "Human_shoes06",
                "Human_short01",
            )
        ],
    )
    blueprint = SimpleNamespace(
        get_editor_property=lambda field: SimpleNamespace(
            get_path_name=lambda: "/Game/FutsalMOT/Characters/MPFB/Core/Skeleton/SKEL_MPFB_Base.SKEL_MPFB_Base"
        )
    )
    candidate = {
        "player_id": 3,
        "mesh": "/Game/FutsalMOT/Characters/MPFB/Players/Player_003/MPFB_Player_003",
        "anim_class": "/Game/FutsalMOT/Characters/MPFB/Core/Retarget/ABP_MPFB_RuntimeRetarget",
        "skeleton": "/Game/FutsalMOT/Characters/MPFB/Core/Skeleton/SKEL_MPFB_Base",
    }

    assert appearance_module._validate_candidate_compatibility(
        candidate, mesh, blueprint
    ) == []

    candidate["skeleton"] = "/Game/Other/Skeleton"
    errors = appearance_module._validate_candidate_compatibility(
        candidate, mesh, blueprint
    )
    assert any("shared Skeleton" in error for error in errors)


def test_baseline_component_compatibility_requires_skeletal_mesh_and_anim_class():
    def typed_object(name):
        return SimpleNamespace(
            get_class=lambda: SimpleNamespace(get_name=lambda: name)
        )

    assert appearance_module._baseline_component_errors(
        typed_object("SkeletalMesh"), typed_object("AnimBlueprintGeneratedClass")
    ) == []
    errors = appearance_module._baseline_component_errors(
        typed_object("StaticMesh"), typed_object("BlueprintGeneratedClass")
    )
    assert any("SkeletalMesh" in error for error in errors)
    assert any("AnimBlueprintGeneratedClass" in error for error in errors)


def test_baseline_material_layout_requires_semantic_slots_and_valid_override_length():
    slot_names = [
        "Human_body",
        "Human_eyebrow001",
        "材质",
        "Human_futsalmot_shorts_male_final",
        "Human_low-poly",
        "Human_shoes06",
        "Human_short01",
    ]
    mesh = SimpleNamespace(
        get_editor_property=lambda _field: [
            SimpleNamespace(get_editor_property=lambda _name, slot=slot: slot)
            for slot in slot_names
        ]
    )

    assert appearance_module._baseline_material_layout_errors(mesh, [None] * 7) == []
    errors = appearance_module._baseline_material_layout_errors(mesh, [None] * 8)
    assert any("override_materials" in error for error in errors)


def test_ue_actor_lookup_uses_exact_labels_and_ignores_substring_lookalikes(
    appearance_inputs,
):
    _, _, _, _, mapping = appearance_inputs
    actors = [
        SimpleNamespace(get_actor_label=lambda label=mapping[entity_id]: label)
        for entity_id in ENTITY_IDS
    ]
    actors.append(SimpleNamespace(get_actor_label=lambda: "Player_L0_Extra"))
    unreal = SimpleNamespace(
        EditorActorSubsystem=object,
        get_editor_subsystem=lambda subsystem: SimpleNamespace(
            get_all_level_actors=lambda: actors
        ),
    )

    found, errors = appearance_module._actor_matches(unreal, mapping)

    assert errors == []
    assert set(found) == set(ENTITY_IDS)
    assert found["L0"] is actors[0]


def test_ue_actor_lookup_rejects_duplicate_exact_label(appearance_inputs):
    _, _, _, _, mapping = appearance_inputs
    actors = [
        SimpleNamespace(get_actor_label=lambda label=mapping[entity_id]: label)
        for entity_id in ENTITY_IDS
    ]
    actors.append(SimpleNamespace(get_actor_label=lambda: mapping["L0"]))
    unreal = SimpleNamespace(
        EditorActorSubsystem=object,
        get_editor_subsystem=lambda subsystem: SimpleNamespace(
            get_all_level_actors=lambda: actors
        ),
    )

    found, errors = appearance_module._actor_matches(unreal, mapping)

    assert "L0" not in found
    assert any("Player_L0" in error and "实际 2" in error for error in errors)
    assert appearance_module._anim_blueprint_asset_path(
        "/Game/FutsalMOT/Characters/MPFB/Core/Retarget/ABP_MPFB_RuntimeRetarget.ABP_MPFB_RuntimeRetarget_C"
    ) == "/Game/FutsalMOT/Characters/MPFB/Core/Retarget/ABP_MPFB_RuntimeRetarget"


def test_semantic_slot_resolution_uses_exact_verified_names():
    names = [
        "Human_body",
        "Human_afro01",
        "Human_eyebrow001",
        "材质",
        "Human_futsalmot_shorts_male_v2_005",
        "Human_low-poly",
        "Human_shoes06",
    ]

    assert appearance_module._resolve_semantic_slots(names) == {
        "Body": 0,
        "Hair": 1,
        "Eyebrows": 2,
        "Jersey": 3,
        "Shorts": 4,
        "Eyes": 5,
        "Shoes": 6,
    }


def test_semantic_slot_resolution_rejects_missing_or_duplicate_target_slot():
    with pytest.raises(ValueError, match="Body"):
        appearance_module._resolve_semantic_slots(["材质", "Human_futsalmot_shorts_male_final"])

    with pytest.raises(ValueError, match="Body"):
        appearance_module._resolve_semantic_slots(
            ["Human_body", "Human_body", "材质", "Human_futsalmot_shorts_male_final"]
        )

    with pytest.raises(ValueError, match="unknown"):
        appearance_module._resolve_semantic_slots(
            [
                "Human_body",
                "Human_eyebrow001",
                "材质",
                "Human_futsalmot_shorts_male_final",
                "Human_low-poly",
                "Human_shoes06",
                "Human_short01",
                "UnknownSlot",
            ]
        )


def test_non_target_overrides_follow_semantic_slots_and_keep_none_entries():
    source_names = [
        "Human_body",
        "Human_eyebrow001",
        "材质",
        "Human_futsalmot_shorts_male_final",
        "Human_low-poly",
        "Human_shoes06",
        "Human_short01",
    ]
    target_names = [
        "Human_body",
        "Human_short04",
        "Human_eyebrow001",
        "材质",
        "Human_futsalmot_shorts_male_final",
        "Human_low-poly",
        "Human_shoes06",
    ]
    source_overrides = [
        "/Game/Old/BodyOverride",
        "/Game/Old/BrowOverride",
        "/Game/Old/JerseyOverride",
        "/Game/Old/ShortsOverride",
        None,
        "/Game/Old/ShoesOverride",
        "/Game/Old/HairOverride",
    ]

    remapped = appearance_module._remap_non_target_overrides(
        source_names, source_overrides, target_names
    )

    assert remapped == [
        None,
        "/Game/Old/HairOverride",
        "/Game/Old/BrowOverride",
        None,
        None,
        None,
        "/Game/Old/ShoesOverride",
    ]


def test_non_target_override_remap_rejects_array_longer_than_source_slots():
    with pytest.raises(ValueError, match="override_materials"):
        appearance_module._remap_non_target_overrides(
            ["Human_body"], [None, None], ["Human_body"]
        )


def test_non_target_override_readback_checks_only_verified_preserved_semantics():
    slot_names = [
        "Human_body",
        "Human_short04",
        "Human_eyebrow001",
        "材质",
        "Human_futsalmot_shorts_male_final",
        "Human_low-poly",
        "Human_shoes06",
    ]
    expected = [
        None,
        "/Game/Old/Hair.Hair",
        "/Game/Old/Brows.Brows",
        None,
        None,
        None,
        "/Game/Old/Shoes.Shoes",
    ]

    assert appearance_module._non_target_override_errors(
        slot_names, expected, expected
    ) == []
    actual = list(expected)
    actual[1] = None
    errors = appearance_module._non_target_override_errors(
        slot_names, expected, actual
    )
    assert any("Hair" in error for error in errors)


def test_report_state_machine_accepts_successful_render_and_restore(appearance_inputs):
    manifest, _, _, _, mapping = appearance_inputs
    report = create_application_report(manifest, mapping)
    assert report["state"] == "not_started"

    report = mark_all_applied(report)
    report = transition_application_report(report, "rendering")
    report = transition_application_report(
        report,
        "restored",
        actor_updates={entity_id: {"restore_status": "restored"} for entity_id in ENTITY_IDS},
        mrq_jobs=[{"camera_id": "C1", "status": "success"}],
        mrq_pending=False,
    )

    assert report["state"] == "restored"


def test_report_state_machine_accepts_failed_terminal_paths(appearance_inputs):
    manifest, _, _, _, mapping = appearance_inputs
    preflight_failed = transition_application_report(
        create_application_report(manifest, mapping), "preflight_failed", error="invalid"
    )
    restore_failed = transition_application_report(
        mark_all_applied(create_application_report(manifest, mapping)),
        "restore_failed",
        error="restore failed",
    )

    assert preflight_failed["state"] == "preflight_failed"
    assert restore_failed["state"] == "restore_failed"


def test_report_rejects_restored_until_every_actor_is_restored(appearance_inputs):
    manifest, _, _, _, mapping = appearance_inputs
    report = transition_application_report(
        mark_all_applied(create_application_report(manifest, mapping)),
        "rendering",
    )
    report["actors"][0]["restore_status"] = "restored"

    with pytest.raises(ValueError, match="actor"):
        transition_application_report(report, "restored", mrq_pending=False)


def test_report_rejects_restored_while_mrq_job_is_pending(appearance_inputs):
    manifest, _, _, _, mapping = appearance_inputs
    report = transition_application_report(
        mark_all_applied(create_application_report(manifest, mapping)),
        "rendering",
        actor_updates={entity_id: {"restore_status": "restored"} for entity_id in ENTITY_IDS},
        mrq_jobs=[{"camera_id": "C1", "status": "rendering"}],
        mrq_pending=True,
    )

    with pytest.raises(ValueError, match="MRQ"):
        transition_application_report(report, "restored")


def test_report_rejects_nonterminal_job_with_pending_false_and_all_actors_restored(
    appearance_inputs,
):
    manifest, _, _, _, mapping = appearance_inputs
    report = transition_application_report(
        mark_all_applied(create_application_report(manifest, mapping)),
        "rendering",
        actor_updates={
            entity_id: {"restore_status": "restored"} for entity_id in ENTITY_IDS
        },
        mrq_jobs=[{"camera_id": "C1", "status": "rendering"}],
        mrq_pending=False,
    )

    with pytest.raises(ValueError, match="terminal"):
        transition_application_report(report, "restored")


def _valid_restored_report(manifest: dict, mapping: dict) -> dict:
    return transition_application_report(
        transition_application_report(
            mark_all_applied(create_application_report(manifest, mapping)),
            "rendering",
        ),
        "restored",
        actor_updates={
            entity_id: {"restore_status": "restored"} for entity_id in ENTITY_IDS
        },
        mrq_jobs=[{"camera_id": "C1", "status": "success"}],
        mrq_pending=False,
    )


def _write_restore_failed_evidence(tmp_path, manifest, mapping, *, mismatch=False):
    snapshots = [_actor_snapshot(entity_id, [None] * 7) for entity_id in ENTITY_IDS]
    asset_packages = [
        {"package_path": "/Game/FutsalMOT/Materials/M_MPFB_Body", "package_dirty": False}
    ]
    rollback = appearance_module._build_rollback_payload(
        manifest["fingerprint"], snapshots, asset_packages
    )
    report = mark_all_applied(create_application_report(manifest, mapping))
    report["transaction_id"] = rollback["transaction_id"]
    report = transition_application_report(report, "rendering")
    report = transition_application_report(
        report,
        "restore_failed",
        error="initial dirty-state/readback failure",
        actor_updates={
            entity_id: {"restore_status": "failed"} for entity_id in ENTITY_IDS
        },
        mrq_jobs=[{"camera_id": "C1", "status": "success"}],
        mrq_pending=False,
        render_result={
            "status": "success",
            "jobs": [{"camera_id": "C1", "status": "success"}],
        },
    )
    if mismatch:
        report["transaction_id"] = "b" * 64
    rollback_path = tmp_path / "appearance-rollback.json"
    report_path = tmp_path / "appearance-application.json"
    rollback_path.write_text(json.dumps(rollback), encoding="utf-8")
    write_application_report(report_path, report)
    return rollback_path, report_path, rollback, report


def test_restore_failed_retry_restores_only_through_verified_snapshot_adapter(
    tmp_path, appearance_inputs, monkeypatch
):
    manifest, _, _, _, mapping = appearance_inputs
    rollback_path, report_path, rollback, _ = _write_restore_failed_evidence(
        tmp_path, manifest, mapping
    )
    restore_calls = []

    def restore_roster(_ue, snapshots, asset_packages):
        assert appearance_module._validate_actor_snapshots(snapshots) == []
        assert snapshots == rollback["actors"]
        assert asset_packages == rollback["asset_packages"]
        restore_calls.append(True)
        return []

    monkeypatch.setattr(appearance_module, "_load_unreal", lambda: object())
    monkeypatch.setattr(
        appearance_module,
        "_current_level_path",
        lambda _ue: "/Game/FutsalMOT/Maps/L_FutsalCourt",
    )
    monkeypatch.setattr(appearance_module, "_restore_roster", restore_roster)

    result = restore_episode_appearance(
        str(rollback_path), str(report_path), "success"
    )

    report = read_application_report(report_path)
    assert result["ok"] is True
    assert result["state"] == "restored"
    assert restore_calls == [True]
    assert report["state"] == "restored"
    assert report["render_result"]["status"] == "success"
    assert report["transaction_id"] == rollback["transaction_id"]
    assert all(actor["restore_status"] == "restored" for actor in report["actors"])


def test_local_recovery_evidence_validator_accepts_current_restore_failed_pair(
    tmp_path, appearance_inputs
):
    manifest, catalog, _, _, mapping = appearance_inputs
    _, _, rollback, report = _write_restore_failed_evidence(
        tmp_path, manifest, mapping
    )

    assert appearance_module.validate_local_recovery_evidence(
        rollback, report, manifest, catalog
    ) == []


def test_local_recovery_evidence_validator_accepts_verified_restored_pair(
    tmp_path, appearance_inputs
):
    manifest, catalog, _, _, mapping = appearance_inputs
    snapshots = [_actor_snapshot(entity_id, [None] * 7) for entity_id in ENTITY_IDS]
    packages = [
        {"package_path": "/Game/FutsalMOT/Materials/M_MPFB_Body", "package_dirty": False}
    ]
    rollback = appearance_module._build_rollback_payload(
        manifest["fingerprint"], snapshots, packages
    )
    report = create_application_report(manifest, mapping)
    report["transaction_id"] = rollback["transaction_id"]
    report = transition_application_report(
        report,
        "applied",
        actor_updates={
            entity_id: {"apply_status": "applied"} for entity_id in ENTITY_IDS
        },
    )
    report = transition_application_report(report, "rendering")
    report = transition_application_report(
        report,
        "restored",
        actor_updates={
            entity_id: {"restore_status": "restored"} for entity_id in ENTITY_IDS
        },
        mrq_jobs=[{"camera_id": "C1", "status": "success"}],
        mrq_pending=False,
    )

    assert appearance_module.validate_local_recovery_evidence(
        rollback, report, manifest, catalog
    ) == []


def test_local_recovery_evidence_validator_rejects_manifest_identity_mismatch(
    tmp_path, appearance_inputs
):
    manifest, catalog, _, _, mapping = appearance_inputs
    _, _, rollback, report = _write_restore_failed_evidence(
        tmp_path, manifest, mapping
    )
    changed_manifest = copy.deepcopy(manifest)
    changed_manifest["fingerprint"] = "f" * 64

    errors = appearance_module.validate_local_recovery_evidence(
        rollback, report, changed_manifest, catalog
    )

    assert any("manifest_fingerprint" in error for error in errors)


@pytest.mark.parametrize(
    ("target", "field", "value", "expected_error"),
    [
        ("snapshot", "schema", "wrong", "snapshot schema"),
        ("report", "transaction_id", "b" * 64, "transaction_id"),
        ("report", "catalog_fingerprint", "c" * 64, "catalog_fingerprint"),
    ],
)
def test_local_recovery_evidence_validator_rejects_forged_pair_identity(
    tmp_path, appearance_inputs, target, field, value, expected_error
):
    manifest, catalog, _, _, mapping = appearance_inputs
    _, _, rollback, report = _write_restore_failed_evidence(
        tmp_path, manifest, mapping
    )
    if target == "snapshot":
        rollback[field] = value
    else:
        report[field] = value

    errors = appearance_module.validate_local_recovery_evidence(
        rollback, report, manifest, catalog
    )

    assert any(expected_error in error for error in errors)


def test_restore_failed_retry_rejects_transaction_mismatch_before_restore_adapter(
    tmp_path, appearance_inputs, monkeypatch
):
    manifest, _, _, _, _ = appearance_inputs
    rollback_path, report_path, _, _ = _write_restore_failed_evidence(
        tmp_path, manifest, {entity_id: "Player_{}".format(entity_id) for entity_id in ENTITY_IDS},
        mismatch=True,
    )
    monkeypatch.setattr(appearance_module, "_load_unreal", lambda: object())
    monkeypatch.setattr(
        appearance_module,
        "_restore_roster",
        lambda *_args: pytest.fail("transaction mismatch must block UE restore"),
    )

    result = restore_episode_appearance(
        str(rollback_path), str(report_path), "success"
    )

    report = read_application_report(report_path)
    assert result["ok"] is False
    assert result["state"] == "restore_failed"
    assert any("transaction_id" in error for error in result["errors"])
    assert report["state"] == "restore_failed"


def test_restore_failed_retry_rejects_manifest_fingerprint_mismatch(
    tmp_path, appearance_inputs, monkeypatch
):
    manifest, _, _, _, mapping = appearance_inputs
    rollback_path, report_path, _, report = _write_restore_failed_evidence(
        tmp_path, manifest, mapping
    )
    report["manifest_fingerprint"] = "c" * 64
    write_application_report(report_path, report)
    monkeypatch.setattr(
        appearance_module,
        "_load_unreal",
        lambda: pytest.fail("fingerprint mismatch must block UE access"),
    )

    result = restore_episode_appearance(
        str(rollback_path), str(report_path), "success"
    )

    persisted = read_application_report(report_path)
    assert result["ok"] is False
    assert any("manifest_fingerprint" in error for error in result["errors"])
    assert persisted["state"] == "restore_failed"


@pytest.mark.parametrize(
    ("final_status", "render_result", "error_fragment"),
    [
        ("rendering", None, "final_status"),
        ("success", {"status": "rendering"}, "render_result"),
    ],
)
def test_restore_failed_retry_requires_terminal_status_and_valid_render_result(
    tmp_path, appearance_inputs, monkeypatch, final_status, render_result, error_fragment
):
    manifest, _, _, _, mapping = appearance_inputs
    rollback_path, report_path, _, _ = _write_restore_failed_evidence(
        tmp_path, manifest, mapping
    )
    monkeypatch.setattr(
        appearance_module,
        "_load_unreal",
        lambda: pytest.fail("invalid terminal evidence must block UE access"),
    )

    result = restore_episode_appearance(
        str(rollback_path),
        str(report_path),
        final_status,
        render_result=render_result,
    )

    persisted = read_application_report(report_path)
    assert result["ok"] is False
    assert result["state"] == "restore_failed"
    assert any(error_fragment in error for error in result["errors"])
    assert persisted["render_result"]["status"] == "success"


def test_restore_failed_retry_failure_keeps_state_and_appends_error(
    tmp_path, appearance_inputs, monkeypatch
):
    manifest, _, _, _, mapping = appearance_inputs
    rollback_path, report_path, _, _ = _write_restore_failed_evidence(
        tmp_path, manifest, mapping
    )
    monkeypatch.setattr(appearance_module, "_load_unreal", lambda: object())
    monkeypatch.setattr(
        appearance_module,
        "_current_level_path",
        lambda _ue: "/Game/FutsalMOT/Maps/L_FutsalCourt",
    )
    monkeypatch.setattr(
        appearance_module,
        "_restore_roster",
        lambda *_args: ["retry dirty-state cleanup failed"],
    )

    result = restore_episode_appearance(
        str(rollback_path), str(report_path), "success"
    )

    report = read_application_report(report_path)
    assert result["ok"] is False
    assert result["state"] == "restore_failed"
    assert report["state"] == "restore_failed"
    assert "initial dirty-state/readback failure" in report["errors"]
    assert any("retry dirty-state cleanup failed" in error for error in report["errors"])


def test_restore_failed_transition_cannot_be_fabricated_by_report_caller(
    tmp_path, appearance_inputs
):
    manifest, _, _, _, mapping = appearance_inputs
    _, _, _, report = _write_restore_failed_evidence(tmp_path, manifest, mapping)

    with pytest.raises(ValueError, match="transition"):
        transition_application_report(
            report,
            "restored",
            actor_updates={
                entity_id: {"restore_status": "restored"}
                for entity_id in ENTITY_IDS
            },
            mrq_pending=False,
            mrq_jobs=[{"camera_id": "C1", "status": "success"}],
        )


@pytest.mark.parametrize("entry_point", ["serialize", "read"])
@pytest.mark.parametrize(
    ("invariant", "message"),
    [
        ("actor", "actor"),
        ("pending", "mrq_pending"),
        ("job", "terminal"),
    ],
)
def test_every_report_entry_point_rejects_fabricated_restored_state(
    tmp_path, appearance_inputs, entry_point, invariant, message
):
    manifest, _, _, _, mapping = appearance_inputs
    report = _valid_restored_report(manifest, mapping)
    if invariant == "actor":
        report["actors"][0]["restore_status"] = "restoring"
    elif invariant == "pending":
        report["mrq_pending"] = True
    else:
        report["mrq_jobs"][0]["status"] = "rendering"

    with pytest.raises(ValueError, match=message):
        if entry_point == "serialize":
            serialize_application_report(report)
        else:
            report_path = tmp_path / "appearance_application.json"
            report_path.write_text(json.dumps(report), encoding="utf-8")
            read_application_report(report_path)


def test_report_rejects_invalid_state_transition(appearance_inputs):
    manifest, _, _, _, mapping = appearance_inputs
    report = create_application_report(manifest, mapping)

    with pytest.raises(ValueError, match="transition"):
        transition_application_report(report, "rendering")


def test_report_serialization_is_portable_and_atomic(tmp_path, appearance_inputs):
    manifest, _, _, _, mapping = appearance_inputs
    report = _valid_restored_report(manifest, mapping)
    local_rollback = str(tmp_path / "appearance-rollback.json")
    report["rollback_path"] = local_rollback
    report["render_result"] = {
        "status": "success",
        "jobs": [{"camera_id": "C1", "status": "success"}],
    }

    payload = serialize_application_report(report)
    target = tmp_path / "provenance" / "appearance_application.json"
    write_application_report(target, report)

    assert local_rollback not in payload
    assert "rollback_path" not in json.loads(payload)
    written = json.loads(target.read_text(encoding="utf-8"))
    assert written["schema"] == APPLICATION_REPORT_SCHEMA
    assert written["version"] == APPLICATION_REPORT_VERSION
    assert written["render_result"] == report["render_result"]
    assert written["actors"][0]["assets"]["mesh"].startswith("/Game/")
    assert written["actors"][0]["slots"]["body_texture_parameter"] == "SkinTexture"
    assert not target.with_name(target.name + ".tmp").exists()


def test_render_result_must_be_terminal_and_portable(appearance_inputs):
    manifest, _, _, _, mapping = appearance_inputs
    report = _valid_restored_report(manifest, mapping)
    report["render_result"] = {"status": "rendering"}

    with pytest.raises(ValueError, match="render_result"):
        serialize_application_report(report)


def test_render_result_rejects_python_values_that_json_would_coerce(appearance_inputs):
    manifest, _, _, _, mapping = appearance_inputs
    report = _valid_restored_report(manifest, mapping)
    report["render_result"] = {
        "status": "success",
        "jobs": ({"camera_id": "C1", "status": "success"},),
    }

    with pytest.raises(ValueError, match="render_result"):
        serialize_application_report(report)


def test_transition_can_record_portable_render_result(appearance_inputs):
    manifest, _, _, _, mapping = appearance_inputs
    report = transition_application_report(
        mark_all_applied(create_application_report(manifest, mapping)),
        "rendering",
    )

    restored = transition_application_report(
        report,
        "restored",
        actor_updates={
            entity_id: {"restore_status": "restored"} for entity_id in ENTITY_IDS
        },
        mrq_jobs=[{"camera_id": "C1", "status": "success"}],
        mrq_pending=False,
        render_result={
            "status": "success",
            "jobs": [{"camera_id": "C1", "status": "success"}],
        },
    )

    assert restored["render_result"]["status"] == "success"


def test_apply_and_restore_fail_closed_without_claiming_actor_changes(
    tmp_path, appearance_inputs
):
    manifest, catalog, _, _, mapping = appearance_inputs
    report_path = tmp_path / "provenance" / "appearance_application.json"
    rollback_path = tmp_path / ".futsalmot" / "runtime" / "task" / "appearance-rollback.json"

    applied = apply_episode_appearance(
        manifest, catalog, mapping, str(rollback_path), str(report_path)
    )
    restored = restore_episode_appearance(
        str(rollback_path), str(report_path), "success"
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert applied["ok"] is False
    assert applied["state"] == "preflight_failed"
    assert restored["ok"] is False
    assert restored["state"] == "preflight_failed"
    assert report["state"] == "preflight_failed"
    assert str(rollback_path) not in json.dumps(report)


def test_apply_does_not_overwrite_an_existing_report_or_rollback_snapshot(
    tmp_path, appearance_inputs
):
    manifest, catalog, _, _, mapping = appearance_inputs
    report_path = tmp_path / "appearance_application.json"
    rollback_path = tmp_path / "appearance_rollback.json"
    original_report = b"do-not-overwrite-existing-report\n"
    report_path.write_bytes(original_report)

    result = apply_episode_appearance(
        manifest, catalog, mapping, str(rollback_path), str(report_path)
    )

    assert result["ok"] is False
    assert report_path.read_bytes() == original_report
    assert not rollback_path.exists()


def test_module_import_does_not_load_unreal():
    module_name = "episode_appearance_import_probe"
    module_path = Path(__file__).resolve().parent.parent / "ue" / "episode_appearance.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    before = "unreal" in sys.modules

    spec.loader.exec_module(module)

    assert ("unreal" in sys.modules) is before
    assert callable(apply_episode_appearance)
    assert callable(restore_episode_appearance)
