"""C5.2 生命周期审计（只读）：dump C4 Recorder / WBP / MRQ 回调面。"""

from pathlib import Path

LOG = []
LOG_PATH = Path(r"D:\projects\FustalMOT_UEDataset\Content\FutsalMOT\code\.futsalmot\diag_c5_lifecycle.log")


def _log(msg):
    print(msg)
    LOG.append(str(msg))


def _flush():
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        LOG_PATH.write_text("\n".join(LOG), encoding="utf-8")
    except Exception as e:
        print(f"  写日志失败: {e}")


def _node_title(n):
    try:
        return str(n.get_node_title())
    except Exception:
        try:
            return n.get_class().get_name()
        except Exception:
            return "?"


def _dump_blueprint(bp, label):
    import unreal
    _log(f"\n===== {label} =====")
    # 事件图
    eg = unreal.BlueprintEditorLibrary.find_event_graph(bp)
    ed = unreal.BlueprintGraphEditor.get_graph_editor(eg)
    nodes = ed.list_all_nodes()
    _log(f"  EventGraph nodes ({len(nodes)}):")
    for n in nodes:
        _log(f"    {_node_title(n)}  [{n.get_class().get_name()}]")
    # 函数图
    for fn_name in ("CaptureOutputFrame",):
        try:
            ed2 = unreal.BlueprintGraphEditor.get_graph_editor_by_name(bp, fn_name)
            nodes2 = ed2.list_all_nodes()
            _log(f"  Function '{fn_name}' nodes ({len(nodes2)}):")
            for n in nodes2:
                _log(f"    {_node_title(n)}  [{n.get_class().get_name()}]")
        except Exception as e:
            _log(f"  Function '{fn_name}' ERR: {type(e).__name__} {e}")


def _dump_burnin_wbp(bp):
    import unreal
    _log(f"\n===== WBP_PoseMRQBurnInC4 =====")
    eg = unreal.BlueprintEditorLibrary.find_event_graph(bp)
    ed = unreal.BlueprintGraphEditor.get_graph_editor(eg)
    nodes = ed.list_all_nodes()
    _log(f"  EventGraph nodes ({len(nodes)}):")
    call_funcs = 0
    for n in nodes:
        title = _node_title(n)
        cls = n.get_class().get_name()
        _log(f"    {title}  [{cls}]")
        if cls == "K2Node_CallFunction":
            call_funcs += 1
    _log(f"  CallFunction 节点数: {call_funcs}")


def main():
    import unreal
    _log("======== C5.2 生命周期审计 ========")

    # 1) C4 Recorder G0
    bp = unreal.load_asset("/Game/FutsalMOT/Blueprints/BP_PoseRecorderC4_G0")
    _dump_blueprint(bp, "BP_PoseRecorderC4_G0")

    # 2) WBP
    wbp = unreal.load_asset("/Game/FutsalMOT/Blueprints/WBP_PoseMRQBurnInC4")
    _dump_burnin_wbp(wbp)

    # 3) MRQ 回调面
    _log("\n===== MRQ 回调面 =====")
    for cls in (unreal.MoviePipeline, unreal.MoviePipelineExecutorBase, unreal.MoviePipelinePIEExecutor):
        if cls is None:
            _log(f"  {cls} None")
            continue
        _log(f"  --- {cls} ---")
        for m in dir(cls):
            ml = m.lower()
            if "finish" in ml or "complete" in ml or "output" in ml or "frame" in ml or "shot" in ml or "delegate" in ml:
                _log(f"    {m}")

    # 4) MoviePipelineBlueprintLibrary（类可能非 Python 属性，尝试反射）
    _log("\n===== MoviePipelineBlueprintLibrary（C++ 类，探测 Python 反射）=====")
    try:
        lib = getattr(unreal, "MoviePipelineBlueprintLibrary", None)
        if lib is not None:
            for m in dir(lib):
                if not m.startswith("__"):
                    _log(f"    {m}")
        else:
            _log("    unreal.MoviePipelineBlueprintLibrary 非 Python 属性（仅 C++/BP 可调用）")
    except Exception as e:
        _log(f"    ERR {e}")

    _flush()


if __name__ == "__main__":
    main()