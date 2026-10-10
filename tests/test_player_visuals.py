"""Player_N 视觉 Profile 的纯 Python 契约测试。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ue.player_visuals import (
    build_restore_payload,
    json_default,
    normalize_player_visuals,
    player_visuals_fingerprint,
    required_driver_visibility_tick_option,
    select_rollback_payload,
    sanitized_player_visuals_provenance,
    sanitize_snapshot,
    write_player_visuals_provenance,
)
from ue.apply_player_visuals import serialize_result


PLAYER003_PROFILE = {
    "mesh": "/Game/FutsalMOT/Characters/MPFB/Players/Player_003/MPFB_Player_003",
    "anim_class": "/Game/FutsalMOT/Characters/MPFB/Core/Retarget/ABP_MPFB_RuntimeRetarget",
    "skeleton": "/Game/FutsalMOT/Characters/MPFB/Core/Skeleton/SKEL_MPFB_Base",
}


def player003_config():
    return {
        "default_profile": "mpfb_player_003",
        "profiles": {"mpfb_player_003": PLAYER003_PROFILE.copy()},
        "entity_profiles": {},
    }


def test_player003_profile_expands_to_all_ten_entities():
    normalized = normalize_player_visuals(player003_config())

    assert set(normalized["entity_profiles"]) == {
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
    }
    assert all(
        value == "mpfb_player_003"
        for value in normalized["entity_profiles"].values()
    )


def test_fingerprint_is_stable_and_provenance_has_no_machine_paths():
    normalized = normalize_player_visuals(player003_config())

    first = player_visuals_fingerprint(normalized)
    second = player_visuals_fingerprint(json.loads(json.dumps(normalized)))
    provenance = sanitized_player_visuals_provenance(normalized)

    assert first == second
    assert "D:" not in json.dumps(provenance)
    assert provenance["fingerprint"] == first
    assert provenance["profiles"]["mpfb_player_003"]["mesh"] == PLAYER003_PROFILE["mesh"]
    assert provenance["profiles"]["mpfb_player_003"]["visual_relative_location_cm"] == [0.0, 0.0, 0.0]


def test_invalid_profile_reference_fails_fast():
    with pytest.raises(ValueError, match="不存在的 profile"):
        normalize_player_visuals(
            {"default_profile": "missing", "profiles": {}, "entity_profiles": {}}
        )


def test_missing_player_visuals_preserves_legacy_behavior():
    assert normalize_player_visuals(None) is None
    assert normalize_player_visuals({}) is None


def test_snapshot_and_restore_payload_have_narrow_field_scope():
    snapshot = {
        "actor_path": "/Game/FutsalMOT/Maps/L_FutsalCourt:PersistentLevel.Player_L0",
        "label": "Player_L0",
        "tags": ["PoseL0"],
        "transform": {"location": [1.0, 2.0, 3.0]},
        "component_name": "CharacterMesh0",
        "old_mesh": "/Game/OldMesh",
        "old_anim_class": "/Game/OldAnimBP",
        "old_skeleton": "/Game/OldSkeleton",
        "unexpected": "must not be retained",
    }

    sanitized = sanitize_snapshot(snapshot)
    restore = build_restore_payload(sanitized)

    assert "unexpected" not in sanitized
    assert set(restore) == {
        "actor_path", "component_name", "skeletal_mesh", "anim_class"
    }
    assert restore["skeletal_mesh"] == "/Game/OldMesh"
    assert restore["anim_class"] == "/Game/OldAnimBP"


def test_provenance_is_written_without_machine_rollback_paths(tmp_path):
    normalized = normalize_player_visuals(player003_config())

    output = write_player_visuals_provenance(tmp_path, normalized)
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert output == tmp_path / "provenance" / "player-visuals.json"
    assert payload["fingerprint"] == player_visuals_fingerprint(normalized)
    assert str(tmp_path) not in output.read_text(encoding="utf-8")


def test_no_profile_does_not_write_legacy_provenance(tmp_path):
    assert write_player_visuals_provenance(tmp_path, None) is None
    assert not (tmp_path / "provenance").exists()


def test_existing_rollback_snapshot_is_preserved_for_repeat_apply():
    original = {
        "schema": "futsalmot_player_visuals_rollback",
        "version": 1,
        "level": "/Game/FutsalMOT/Maps/L_FutsalCourt",
        "actors": [
            {
                "entity_id": entity_id,
                "actor_path": f"/Actors/{entity_id}",
                "old_mesh": "/Game/DefaultMesh",
                "old_anim_class": "/Game/DefaultAnim",
            }
            for entity_id in ("L0", "L1")
        ],
    }
    current = {
        **original,
        "actors": [
            {
                "entity_id": entity_id,
                "actor_path": f"/Actors/{entity_id}",
                "old_mesh": "/Game/MPFB",
                "old_anim_class": "/Game/MPFBAnim",
            }
            for entity_id in ("L0", "L1")
        ],
    }

    selected, preserved = select_rollback_payload(original, current)

    assert preserved is True
    assert selected == original


def test_invalid_existing_rollback_fails_closed():
    with pytest.raises(ValueError, match="rollback 快照无效"):
        select_rollback_payload({"schema": "wrong"}, {"schema": "new"})


def test_player003_profile_uses_a_separate_visible_component():
    config = player003_config()
    config["profiles"]["mpfb_player_003"].update(
        {
            "visual_component": "MPFBVisual",
            "visual_parent": "CharacterMesh0",
            "visual_relative_location_cm": [0.0, 0.0, 0.0],
            "visual_relative_rotation_deg": [0.0, 0.0, 0.0],
            "visual_relative_scale": [1.0, 1.0, 1.0],
        }
    )

    normalized = normalize_player_visuals(config)

    assert normalized["profiles"]["mpfb_player_003"]["visual_component"] == "MPFBVisual"
    assert normalized["profiles"]["mpfb_player_003"]["visual_parent"] == "CharacterMesh0"
    assert normalized["profiles"]["mpfb_player_003"]["visual_relative_location_cm"] == [0.0, 0.0, 0.0]


def test_default_visible_component_offset_is_zero_for_production():
    normalized = normalize_player_visuals(player003_config())
    profile = normalized["profiles"]["mpfb_player_003"]
    assert profile["visual_relative_location_cm"] == [0.0, 0.0, 0.0]


def test_generated_anim_class_path_is_not_treated_as_blueprint_asset():
    from ue.player_visuals import is_generated_class_path

    assert is_generated_class_path(
        "/Game/FutsalMOT/Characters/FutsalPlayerBase/Animation/ABP_FutsalPlayerBase.ABP_FutsalPlayerBase_C"
    )
    assert not is_generated_class_path(
        "/Game/FutsalMOT/Characters/FutsalPlayerBase/Animation/ABP_FutsalPlayerBase"
    )


def test_snapshot_restore_can_remove_component_created_by_profile():
    snapshot = {
        "entity_id": "L0",
        "actor_path": "/Actors/L0",
        "label": "Player_L0",
        "tags": ["PoseL0"],
        "transform": {},
        "component_name": "CharacterMesh0",
        "old_mesh": "/Game/OldMesh",
        "old_anim_class": "/Game/OldAnim",
        "old_skeleton": "/Game/OldSkeleton",
        "visual_component": None,
    }
    assert sanitize_snapshot(snapshot)["visual_component"] is None


def test_run_task_reloads_player_visuals_after_editor_module_cache():
    source = (
        Path(__file__).resolve().parents[1] / "ue" / "run_task.py"
    ).read_text(encoding="utf-8")
    module_block = source.split("_UE_MODULE_NAMES = (", 1)[1].split(")", 1)[0]
    assert '"player_visuals"' in module_block


def test_run_task_passes_export_playback_fps_to_annotation_config():
    source = (
        Path(__file__).resolve().parents[1] / "ue" / "run_task.py"
    ).read_text(encoding="utf-8")
    assert 'ann_cfg["playback_fps"]' in source


def test_player_visuals_never_uses_cross_skeleton_leader_pose():
    source = (
        Path(__file__).resolve().parents[1] / "ue" / "player_visuals.py"
    ).read_text(encoding="utf-8")
    assert 'visual.set_editor_property("leader_pose_component", None)' in source
    assert 'visual.set_editor_property("leader_pose_component", component)' not in source


def test_hidden_driver_mesh_refreshes_bones_for_runtime_retarget():
    assert required_driver_visibility_tick_option() == "ALWAYS_TICK_POSE_AND_REFRESH_BONES"


def test_apply_result_serializes_editor_values_as_json():
    class EditorText:
        def __str__(self):
            return "editor-value"

    payload = json.loads(serialize_result({"value": EditorText()}))
    assert payload == {"value": "editor-value"}


def test_unreal_array_like_values_are_serialized_as_arrays():
    class EditorArray:
        def __iter__(self):
            return iter(("L0", "L1"))

    payload = json.loads(serialize_result({"actors": EditorArray()}))
    assert payload == {"actors": ["L0", "L1"]}
    assert json_default(EditorArray()) == ["L0", "L1"]


def test_apply_source_reads_driver_before_configuring_visual_component():
    source = (
        Path(__file__).resolve().parents[1] / "ue" / "player_visuals.py"
    ).read_text(encoding="utf-8")
    driver_index = source.index('if c.get_name() == "CharacterMesh0"')
    visual_index = source.index("visual = _ensure_visual_component")
    assert driver_index < visual_index


def test_driver_tick_script_targets_all_formal_player_labels():
    source = (
        Path(__file__).resolve().parents[1] / "ue" / "set_player_driver_tick.py"
    ).read_text(encoding="utf-8")
    for label in ("Player_L0", "Player_L4", "Player_R0", "Player_R4"):
        assert f'"{label}"' in source
    assert '"ALWAYS_TICK_POSE_AND_REFRESH_BONES"' in source
