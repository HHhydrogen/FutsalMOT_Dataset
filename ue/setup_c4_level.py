"""C4 关卡准备：把 BP_PoseRecorderC4_G0..G4 实例放入当前关卡（不保存关卡）。

只 spawn 实例，不 delete、不 rebuild、不保存关卡（避免 World Partition external actor 崩溃）。
若关卡已有同名实例则跳过。

用法（Unreal Editor Python，FutsalMOTTools.run_python_file）：
    py ".../setup_c4_level.py"
"""

from pathlib import Path

LOG = []
LOG_PATH = Path(r"D:\projects\FustalMOT_UEDataset\Content\FutsalMOT\code\.futsalmot\setup_c4_level.log")
GROUPS = [f"BP_PoseRecorderC4_G{i}" for i in range(5)]


def _log(msg):
    print(msg)
    LOG.append(str(msg))


def _flush():
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        LOG_PATH.write_text("\n".join(LOG), encoding="utf-8")
    except Exception as e:
        print(f"  写日志失败: {e}")


def cleanup_debug_actors(ed_sub=None):
    """销毁仅用于调试的 actor（当前 Editor/PIE 内存世界，不保存关卡）。

    严格匹配条件（只删调试项，不影响正常 TextRenderActor）：
      1. actor label == CAPTURE_DISPLAY
      2. TextRenderComponent.text 包含 "CAPTURE="（大小写不敏感）

    Args:
        ed_sub: EditorActorSubsystem，缺省自动获取。

    Returns:
        销毁的调试 actor 数量。
    """
    import unreal
    if ed_sub is None:
        ed_sub = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    removed = 0
    for a in list(ed_sub.get_all_level_actors()):
        try:
            label = str(a.get_actor_label())
        except Exception:
            continue
        try:
            is_text_render = a.get_class().get_name().startswith("TextRender")
        except Exception:
            is_text_render = False
        if label == "CAPTURE_DISPLAY":
            ed_sub.destroy_actor(a)
            removed += 1
            continue
        if is_text_render:
            hit = False
            try:
                comps = a.get_components_by_class(unreal.TextRenderComponent)
                for c in comps:
                    try:
                        text = str(c.get_editor_property("text"))
                    except Exception:
                        text = ""
                    if "CAPTURE=" in text.upper():
                        hit = True
                        break
            except Exception:
                hit = False
            if hit:
                ed_sub.destroy_actor(a)
                removed += 1
    return removed


def remaining_debug_actors(ed_sub=None):
    """返回当前仍存在的调试 actor label 列表（用于 cleanup 后验证）。"""
    import unreal
    if ed_sub is None:
        ed_sub = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    found = []
    for a in ed_sub.get_all_level_actors():
        try:
            label = str(a.get_actor_label())
        except Exception:
            continue
        try:
            is_text_render = a.get_class().get_name().startswith("TextRender")
        except Exception:
            is_text_render = False
        if label == "CAPTURE_DISPLAY":
            found.append(label)
            continue
        if is_text_render:
            try:
                comps = a.get_components_by_class(unreal.TextRenderComponent)
                for c in comps:
                    try:
                        text = str(c.get_editor_property("text"))
                    except Exception:
                        text = ""
                    if "CAPTURE=" in text.upper():
                        found.append(label)
                        break
            except Exception:
                pass
    return found


def preflight_debug_cleanup(ed_sub=None):
    """正式渲染前的 preflight：清理调试 actor 并验证 CAPTURE_DISPLAY count == 0。

    Returns:
        (ok, removed, remaining)：ok=False 时禁止启动 MRQ。
    """
    removed = cleanup_debug_actors(ed_sub)
    remaining = remaining_debug_actors(ed_sub)
    ok = len(remaining) == 0
    print(f"  [preflight] 调试 actor cleanup: removed={removed}, remaining={remaining}")
    return ok, removed, remaining


def main():
    import unreal
    _log("======== C4 关卡准备：放置 G0..G4 实例 ========")

    ed_sub = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    existing_labels = set()
    for a in ed_sub.get_all_level_actors():
        try:
            existing_labels.add(a.get_actor_label())
        except Exception:
            pass

    placed = 0
    for name in GROUPS:
        if name in existing_labels:
            _log(f"  {name} 已在关卡，跳过")
            continue
        bp = unreal.load_asset(f"/Game/FutsalMOT/Blueprints/{name}")
        if bp is None:
            _log(f"  {name} 资产加载失败")
            continue
        cls = unreal.BlueprintEditorLibrary.generated_class(bp)
        if cls is None:
            _log(f"  {name} 无 GeneratedClass")
            continue
        spawned = ed_sub.spawn_actor_from_class(cls, unreal.Vector(0, 0, 0), unreal.Rotator(0, 0, 0))
        if spawned is not None:
            spawned.set_actor_label(name)
            _log(f"  已放置 {name} -> {spawned.get_name()}")
            placed += 1
        else:
            _log(f"  {name} spawn 失败")

    _log(f"\n完成放置 {placed}/5，未保存关卡")
    _flush()


if __name__ == "__main__":
    main()