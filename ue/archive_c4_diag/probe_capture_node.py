"""探测：如何为 BP_PoseRecorderC4_G0 创建 CaptureOutputFrame 调用节点。"""

from pathlib import Path

LOG = []
LOG_PATH = Path(r"D:\projects\FustalMOT_UEDataset\Content\FutsalMOT\code\.futsalmot\probe_capture_node.log")


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
    _log("======== 探测 CaptureOutputFrame 调用节点 ========")

    # 检查 BP 是否存在 + 函数图
    bp_path = "/Game/FutsalMOT/Blueprints/BP_PoseRecorderC4_G0"
    bp = unreal.EditorAssetLibrary.load_asset(bp_path)
    _log(f"BP: {bp}")
    if bp is None:
        _log("  BP 不存在！")
        _flush()
        return

    gc = unreal.BlueprintEditorLibrary.generated_class(bp)
    _log(f"generated_class: {gc}")
    funcs = [m for m in dir(gc) if "capture" in m.lower() or "sample" in m.lower()]
    _log(f"含 capture/sample 方法: {funcs}")

    # 尝试 add_call_function_node（在事件图编辑器）
    try:
        eg = unreal.BlueprintEditorLibrary.find_event_graph(bp)
        ed = unreal.BlueprintGraphEditor.get_graph_editor(eg)
        paths = [
            "/Game/FutsalMOT/Blueprints/BP_PoseRecorderC4_G0.CaptureOutputFrame",
            "/Script/Engine.BlueprintGeneratedClass.CaptureOutputFrame",
            "BP_PoseRecorderC4_G0.CaptureOutputFrame",
        ]
        for p in paths:
            try:
                n = ed.add_call_function_node(p)
                _log(f"  add_call_function_node({p!r}) -> {n}")
            except Exception as e:
                _log(f"  add_call_function_node({p!r}) ERR: {type(e).__name__}")
    except Exception as e:
        _log(f"  editor ERR: {type(e).__name__} {e}")

    # 尝试 create_node_from_name 各种名称
    try:
        eg2 = unreal.BlueprintEditorLibrary.find_event_graph(bp)
        ed2 = unreal.BlueprintGraphEditor.get_graph_editor(eg2)
        cands = [
            "类|BPPoseRecorderC4_G0|CaptureOutputFrame",
            "类|BPPoseRecorderC4G0|CaptureOutputFrame",
            "Class|BPPoseRecorderC4_G0|CaptureOutputFrame",
            "类|BPPoseRecorder|CaptureOutputFrame",
            "函数|BPPoseRecorderC4_G0|CaptureOutputFrame",
            "BPPoseRecorderC4_G0|CaptureOutputFrame",
        ]
        for name in cands:
            try:
                n = ed2.create_node_from_name(name, unreal.Vector2D(0, 0), [])
                _log(f"  create_node_from_name({name!r}) -> {n}")
                if n is not None:
                    break
            except Exception as e:
                _log(f"  create_node_from_name({name!r}) ERR: {type(e).__name__}")
    except Exception as e:
        _log(f"  editor2 ERR: {type(e).__name__} {e}")

    _flush()


if __name__ == "__main__":
    main()
    print("\n结果写入 probe_capture_node.log")