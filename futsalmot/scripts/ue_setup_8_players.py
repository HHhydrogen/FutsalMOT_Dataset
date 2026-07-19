# -*- coding: utf-8 -*-
"""
FutsalMOT one-time Unreal Editor setup for eight outfield players.

Version:
    UE_SETUP_4V4_OUTFIELD_V1

Run in the Unreal Editor Python console:
    py "D:/projects/FustalMOT_UEDataset/Content/FutsalMOT/code/23_ue_setup_8_players.py"

The script is idempotent.  It creates only missing Player_05..Player_08 actors,
using Player_01..Player_04 as templates.  It does not create goalkeepers and it
does not save the level automatically.  Review the new actors and save the level
manually after the report is PASS.
"""

import json
import os
import traceback

import unreal

SCRIPT_VERSION = "UE_SETUP_4V4_OUTFIELD_V1"
SCRIPT_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "_agent_test_outputs")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "ue_setup_8_players_report.json")

SOURCE_TO_TARGET = {
    "Player_01": "Player_05",
    "Player_02": "Player_06",
    "Player_03": "Player_07",
    "Player_04": "Player_08",
}

# Temporary editor positions only.  The sequence builder moves actors to the
# first trajectory frame before export.
STAGING_POSITIONS = {
    "Player_05": unreal.Vector(-250.0, -700.0, 90.0),
    "Player_06": unreal.Vector(-250.0, -250.0, 90.0),
    "Player_07": unreal.Vector(-250.0, 250.0, 90.0),
    "Player_08": unreal.Vector(-250.0, 700.0, 90.0),
}


def actor_label(actor):
    try:
        return actor.get_actor_label()
    except Exception:
        return ""


def safe_path(obj):
    try:
        return obj.get_path_name()
    except Exception:
        return None


def get_all_actors(subsystem):
    try:
        return list(subsystem.get_all_level_actors() or [])
    except Exception:
        return list(unreal.EditorLevelLibrary.get_all_level_actors() or [])


def find_by_label(actors, label):
    for actor in actors:
        if actor_label(actor) == label:
            return actor
    return None


def editor_world():
    try:
        editor = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem)
        if editor:
            world = editor.get_editor_world()
            if world:
                return world
    except Exception:
        pass
    try:
        return unreal.EditorLevelLibrary.get_editor_world()
    except Exception:
        return None


def copy_component_appearance(source, target):
    """Best-effort fallback copy when actor duplication is unavailable."""
    try:
        source_components = list(
            source.get_components_by_class(unreal.SkeletalMeshComponent) or []
        )
        target_components = list(
            target.get_components_by_class(unreal.SkeletalMeshComponent) or []
        )
    except Exception:
        return

    if not source_components or not target_components:
        return
    source_component = source_components[0]
    target_component = target_components[0]

    try:
        mesh = source_component.get_skeletal_mesh_asset()
        if mesh is not None:
            try:
                target_component.set_skeletal_mesh_asset(mesh)
            except Exception:
                target_component.set_editor_property("skeletal_mesh", mesh)
    except Exception:
        pass

    try:
        material_count = source_component.get_num_materials()
        for index in range(material_count):
            material = source_component.get_material(index)
            if material is not None:
                target_component.set_material(index, material)
    except Exception:
        pass


def duplicate_or_spawn(subsystem, source, target_label, location):
    world = editor_world()
    duplicated = None
    duplicate_errors = []

    if hasattr(subsystem, "duplicate_actor"):
        for args in (
            (source, world, unreal.Vector(0.0, 0.0, 0.0)),
            (source, world),
            (source,),
        ):
            try:
                duplicated = subsystem.duplicate_actor(*args)
                if duplicated is not None:
                    break
            except Exception as exc:
                duplicate_errors.append(str(exc))

    if duplicated is None:
        actor_class = source.get_class()
        rotation = source.get_actor_rotation()
        spawn_errors = []
        try:
            duplicated = unreal.EditorLevelLibrary.spawn_actor_from_class(
                actor_class, location, rotation
            )
        except Exception as exc:
            spawn_errors.append(str(exc))
        if duplicated is None and hasattr(subsystem, "spawn_actor_from_class"):
            try:
                duplicated = subsystem.spawn_actor_from_class(
                    actor_class, location, rotation, False
                )
            except Exception as exc:
                spawn_errors.append(str(exc))
        if duplicated is None:
            raise RuntimeError(
                "无法 duplicate/spawn {} -> {}；duplicate={} spawn={}".format(
                    actor_label(source), target_label, duplicate_errors, spawn_errors
                )
            )
        copy_component_appearance(source, duplicated)

    duplicated.set_actor_label(target_label, mark_dirty=True)
    duplicated.set_actor_location(location, False, False)
    try:
        duplicated.set_actor_rotation(source.get_actor_rotation(), False)
    except Exception:
        pass
    try:
        duplicated.set_actor_scale3d(source.get_actor_scale3d())
    except Exception:
        pass
    try:
        duplicated.tags = list(source.tags)
    except Exception:
        pass
    try:
        folder = source.get_folder_path()
        if folder:
            duplicated.set_folder_path(folder)
    except Exception:
        pass
    return duplicated


def actor_description(actor):
    entry = {
        "label": actor_label(actor),
        "class": None,
        "path": safe_path(actor),
        "location": None,
        "skeletal_mesh": None,
        "skeleton": None,
    }
    try:
        entry["class"] = actor.get_class().get_name()
    except Exception:
        pass
    try:
        value = actor.get_actor_location()
        entry["location"] = [value.x, value.y, value.z]
    except Exception:
        pass
    try:
        components = list(
            actor.get_components_by_class(unreal.SkeletalMeshComponent) or []
        )
        if components:
            mesh = components[0].get_skeletal_mesh_asset()
            entry["skeletal_mesh"] = safe_path(mesh)
            try:
                skeleton = getattr(mesh, "skeleton", None)
                entry["skeleton"] = safe_path(skeleton)
            except Exception:
                pass
    except Exception:
        pass
    return entry


def destroy_created(subsystem, actors):
    for actor in reversed(actors):
        try:
            subsystem.destroy_actor(actor)
        except Exception:
            try:
                unreal.EditorLevelLibrary.destroy_actor(actor)
            except Exception:
                pass


def write_report(report):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    unreal.log("[8P-SETUP] report={}".format(OUTPUT_PATH))


def main():
    subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    if subsystem is None:
        raise RuntimeError("EditorActorSubsystem 不可用")

    initial_actors = get_all_actors(subsystem)
    missing_sources = [
        source
        for source in SOURCE_TO_TARGET
        if find_by_label(initial_actors, source) is None
    ]
    if missing_sources:
        raise RuntimeError("缺少模板 Actor: {}".format(missing_sources))

    created = []
    skipped = []
    try:
        for source_label, target_label in SOURCE_TO_TARGET.items():
            actors = get_all_actors(subsystem)
            existing = find_by_label(actors, target_label)
            if existing is not None:
                skipped.append(actor_description(existing))
                unreal.log("[8P-SETUP] SKIP existing {}".format(target_label))
                continue
            source = find_by_label(actors, source_label)
            target = duplicate_or_spawn(
                subsystem,
                source,
                target_label,
                STAGING_POSITIONS[target_label],
            )
            created.append(target)
            unreal.log(
                "[8P-SETUP] CREATED {} from {}".format(
                    target_label, source_label
                )
            )

        final_actors = get_all_actors(subsystem)
        expected = ["Player_{:02d}".format(index) for index in range(1, 9)]
        missing_final = [
            label for label in expected if find_by_label(final_actors, label) is None
        ]
        if missing_final:
            raise RuntimeError("创建后仍缺少 Actor: {}".format(missing_final))

        descriptions = {
            label: actor_description(find_by_label(final_actors, label))
            for label in expected
        }
        skeletons = {
            entry.get("skeleton")
            for entry in descriptions.values()
            if entry.get("skeleton")
        }
        report = {
            "schema_version": "1.0",
            "tool_version": SCRIPT_VERSION,
            "status": "PASS",
            "created_count": len(created),
            "skipped_existing_count": len(skipped),
            "created": [actor_description(actor) for actor in created],
            "skipped_existing": skipped,
            "players": descriptions,
            "unique_skeletons": sorted(skeletons),
            "goalkeepers_created": 0,
            "level_saved": False,
            "next_action": "Review Player_05..Player_08, then save the current level manually.",
        }
        write_report(report)
        unreal.log("[8P-SETUP] PASS: 8 outfield players present; no goalkeepers")
        unreal.log("[8P-SETUP] Level was NOT saved automatically.")
        return report
    except Exception as exc:
        destroy_created(subsystem, created)
        report = {
            "schema_version": "1.0",
            "tool_version": SCRIPT_VERSION,
            "status": "ERROR",
            "error": "{}: {}".format(type(exc).__name__, exc),
            "traceback": traceback.format_exc(),
            "rolled_back_created_count": len(created),
            "level_saved": False,
        }
        write_report(report)
        unreal.log_error("[8P-SETUP] ERROR: {}".format(exc))
        raise


if __name__ == "__main__":
    main()
