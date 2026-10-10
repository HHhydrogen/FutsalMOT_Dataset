"""为正式地图的十个球员开启隐藏驱动网格的骨骼刷新。"""

from __future__ import annotations

import json

import unreal


LEVEL = "/Game/FutsalMOT/Maps/L_FutsalCourt"
PLAYER_LABELS = (
    "Player_L0",
    "Player_L1",
    "Player_L2",
    "Player_L3",
    "Player_L4",
    "Player_R0",
    "Player_R1",
    "Player_R2",
    "Player_R3",
    "Player_R4",
)
EXPECTED_TICK = "ALWAYS_TICK_POSE_AND_REFRESH_BONES"


def main() -> int:
    world = unreal.EditorLoadingAndSavingUtils.load_map(LEVEL)
    if world is None:
        print(json.dumps({"ok": False, "errors": [f"地图加载失败: {LEVEL}"]}, ensure_ascii=False))
        return 1

    enum_value = getattr(unreal.VisibilityBasedAnimTickOption, EXPECTED_TICK)
    editor = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    actors = [a for a in editor.get_all_level_actors() if a.get_actor_label() in PLAYER_LABELS]
    by_label = {a.get_actor_label(): a for a in actors}
    errors = []
    changed = []
    saved = []

    for label in PLAYER_LABELS:
        actor = by_label.get(label)
        if actor is None:
            errors.append(f"缺少 Actor: {label}")
            continue
        components = [
            c
            for c in actor.get_components_by_class(unreal.SkeletalMeshComponent)
            if c.get_name() == "CharacterMesh0"
        ]
        if len(components) != 1:
            errors.append(f"{label} 的 CharacterMesh0 数量非法: {len(components)}")
            continue
        component = components[0]
        component.set_editor_property("visibility_based_anim_tick_option", enum_value)
        changed.append(label)
        saved_ok = unreal.EditorAssetLibrary.save_loaded_asset(
            actor,
            only_if_is_dirty=True,
        )
        saved.append({"label": label, "saved": bool(saved_ok)})
        if not saved_ok:
            errors.append(f"External Actor 保存失败: {label}")

    result = {
        "ok": not errors and len(changed) == len(PLAYER_LABELS),
        "level": world.get_path_name(),
        "expected_tick": EXPECTED_TICK,
        "changed": changed,
        "saved": saved,
        "errors": errors,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
