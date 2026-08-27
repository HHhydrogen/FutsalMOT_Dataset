"""Phase B：最小 Call Function Node Probe（WBP_C4_CallProbe_0N）。

只验证一件事：能否在 WBP 中创建调用 BP_PoseRecorderC4_G0.CaptureOutputFrame 的
Blueprint Call Function 节点。

两种候选方法（一次探测，不做多轮）：
  A. editor.add_call_function_node("<包>.<生成的_C 类>:CaptureOutputFrame")
     —— 对应 C++ FSoftObjectPath.LoadSynchronous() 路径，不依赖 action DB。
  B. editor.list_available_nodes() 找确切 "Category|MenuName" 字符串后
     editor.create_node_from_name(..., declaring_class=GeneratedClass)。

本会话 EditorAssetLibrary（does_asset_exist/delete_asset）不可靠，因此：
  用 unreal.load_asset 判断存在性；用递增唯一名避免 delete。

用法（Unreal Editor Python，FutsalMOTTools.run_python_file）：
    py ".../probe_c4_call.py"
"""

from pathlib import Path

LOG = []
LOG_PATH = Path(r"D:\projects\FustalMOT_UEDataset\Content\FutsalMOT\code\.futsalmot\probe_c4_call.log")
PROBE_BASE = "/Game/FutsalMOT/Blueprints/WBP_C4_CallProbe_"
G0_ASSET = "/Game/FutsalMOT/Blueprints/BP_PoseRecorderC4_G0"
G0_FUNC = f"{G0_ASSET}.BP_PoseRecorderC4_G0_C:CaptureOutputFrame"


def _log(msg):
    print(msg)
    LOG.append(str(msg))


def _flush():
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        LOG_PATH.write_text("\n".join(LOG), encoding="utf-8")
    except Exception as e:
        print(f"  写日志失败: {e}")


def main():
    import unreal
    _log("======== Phase B：最小 CallFunction Probe ========")

    import time
    probe_asset = f"{PROBE_BASE}{int(time.time() * 1000) % 100000:05d}"
    _log(f"  使用探针资产: {probe_asset}")

    bp = unreal.BlueprintEditorLibrary.create_blueprint_asset_with_parent(probe_asset, unreal.UserWidget)
    _log(f"[1] create probe WBP -> {bp}")
    if bp is None:
        _log("  ERROR: 创建失败")
        _flush()
        return

    eg = unreal.BlueprintEditorLibrary.find_event_graph(bp)
    editor = unreal.BlueprintGraphEditor.get_graph_editor(eg)
    evt = editor.add_custom_event_node("ProbeStart")
    _log(f"[2] custom event -> {evt}")

    node_a = None
    try:
        node_a = editor.add_call_function_node(G0_FUNC)
        _log(f"[3][A] add_call_function_node({G0_FUNC}) -> {node_a}")
    except Exception as e:
        _log(f"[3][A] add_call_function_node ERR: {type(e).__name__} {e}")

    node_b = None
    try:
        avail = editor.list_available_nodes([])
        hits = [s for s in avail if "captureoutputframe" in s.lower() or "bpposerecorderc4" in s.lower()]
        _log(f"[3][B] list_available_nodes hits ({len(hits)}): {hits}")
        gc = unreal.BlueprintEditorLibrary.generated_class(unreal.load_asset(G0_ASSET))
        for name in hits:
            try:
                node = editor.create_node_from_name(name, unreal.Vector2D(0, 0), [], gc)
                _log(f"[3][B] create_node_from_name({name!r}, declaring_class) -> {node}")
                if node is not None:
                    node_b = node
                    break
            except Exception as e:
                _log(f"[3][B] create_node_from_name({name!r}) ERR: {type(e).__name__} {e}")
    except Exception as e:
        _log(f"[3][B] list_available_nodes ERR: {type(e).__name__} {e}")

    created = node_a if node_a is not None else node_b
    if created is not None:
        try:
            pins = [str(p.get_pin_name()) for p in created.list_all_pins()]
            _log(f"[4] 成功节点: {created.get_class().get_name()} pins: {sorted(pins)}")
        except Exception as e:
            _log(f"[4] pins ERR: {type(e).__name__} {e}")
    else:
        _log("[4] 两种方法均失败 -> 判定 Python 程序化创建 BP 函数调用节点不可靠，转 fallback")

    try:
        ok = unreal.BlueprintEditorLibrary.compile_blueprint(bp)
        _log(f"[5] compile -> {ok}")
        saved = unreal.EditorAssetLibrary.save_asset(probe_asset, only_if_is_dirty=True)
        _log(f"[5] save -> {saved}")
    except Exception as e:
        _log(f"[5] compile/save ERR: {type(e).__name__} {e}")

    _flush()


if __name__ == "__main__":
    main()