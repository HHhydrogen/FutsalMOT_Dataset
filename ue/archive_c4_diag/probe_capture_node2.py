"""探测2：找到 BP_PoseRecorderC4_G0 CaptureOutputFrame 的调用节点创建方式。"""

from pathlib import Path

LOG = []
LOG_PATH = Path(r"D:\projects\FustalMOT_UEDataset\Content\FutsalMOT\code\.futsalmot\probe_capture_node2.log")


def _log(msg):
    print(msg)
    LOG.append(str(msg))


def _flush():
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        LOG_PATH.write_text("\n".join(LOG), encoding="utf-8")
    except Exception:
        pass


def main():
    import unreal
    _log("======== 探测2：CaptureOutputFrame 节点名 ========")

    bp_path = "/Game/FutsalMOT/Blueprints/BP_PoseRecorderC4_G0"
    bp = unreal.EditorAssetLibrary.load_asset(bp_path)
    _log(f"BP: {bp}")

    # 1. 列出蓝图所有图（函数图/事件图）
    try:
        graphs = bp.get_editor_property("function_graphs")
        _log(f"function_graphs: {graphs}")
        for g in graphs:
            _log(f"  graph: {g.get_name()} title={g.get_editor_property('graph_name') if hasattr(g, 'get_editor_property') else '?'}")
    except Exception as e:
        _log(f"  function_graphs ERR: {type(e).__name__} {e}")

    # 2. 尝试更多 create_node_from_name 变体
    try:
        eg = unreal.BlueprintEditorLibrary.find_event_graph(bp)
        ed = unreal.BlueprintGraphEditor.get_graph_editor(eg)
        cands = [
            "类|BP_PoseRecorderC4_G0_C|CaptureOutputFrame",
            "类|BP_PoseRecorderC4G0|CaptureOutputFrame",
            "类|BP_PoseRecorderC4_G0|CaptureOutputFrame",
            "BPPoseRecorderC4_G0|CaptureOutputFrame",
            "类|BPPoseRecorderC4_G0|Capture Output Frame",
            "类|BPPoseRecorderC4_G0|CaptureOutputFrame(_,_)",
        ]
        for name in cands:
            try:
                n = ed.create_node_from_name(name, unreal.Vector2D(0, 0), [])
                _log(f"  create_node_from_name({name!r}) -> {n}")
                if n is not None:
                    break
            except Exception as e:
                _log(f"  create_node_from_name({name!r}) ERR: {type(e).__name__}")
    except Exception as e:
        _log(f"  editor ERR: {type(e).__name__} {e}")

    # 3. 直接查 UFunction（通过 class 的 find_function_by_name）
    try:
        gc = unreal.BlueprintEditorLibrary.generated_class(bp)
        fn = gc.find_function_by_name("CaptureOutputFrame")
        _log(f"find_function_by_name('CaptureOutputFrame') -> {fn}")
        if fn is not None:
            _log(f"  fn: {fn} 类: {fn.get_outer()}")
    except Exception as e:
        _log(f"  find_function ERR: {type(e).__name__} {e}")

    _flush()


if __name__ == "__main__":
    main()
    print("\n结果写入 probe_capture_node2.log")