"""探测3：蓝图名复杂度 vs create_node_from_name 调用函数。

建一个简单名 BP（BP_DiagCall）和当前 C4 BP，分别尝试 create_node_from_name。
验证「复杂蓝图名导致函数调用节点解析失败」假设。

用法（Unreal Editor Python Console）：
    py "D:/projects/FustalMOT_UEDataset/Content/FutsalMOT/code/ue/probe_call_func.py"
"""

from pathlib import Path

LOG = []
LOG_PATH = Path(r"D:\projects\FustalMOT_UEDataset\Content\FutsalMOT\code\.futsalmot\probe_call_func.log")


def _log(msg):
    print(msg)
    LOG.append(str(msg))


def _flush():
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        LOG_PATH.write_text("\n".join(LOG), encoding="utf-8")
    except Exception:
        pass


def _try_create(editor, names):
    for name in names:
        try:
            n = editor.create_node_from_name(name, unreal.Vector2D(0, 0), [])
            _log(f"    create_node_from_name({name!r}) -> {n}")
            if n is not None:
                return n
        except Exception as e:
            _log(f"    create_node_from_name({name!r}) ERR: {type(e).__name__}")
    return None


def main():
    import unreal
    global unreal
    _log("======== 探测：蓝图名复杂度 vs 函数调用节点 ========")

    # 1. 建简单名 BP + 函数
    simple_path = "/Game/FutsalMOT/Blueprints/BP_DiagCall"
    if unreal.EditorAssetLibrary.does_asset_exist(simple_path):
        unreal.EditorAssetLibrary.delete_asset(simple_path)
    bp = unreal.BlueprintEditorLibrary.create_blueprint_asset_with_parent(simple_path, unreal.Actor)
    _log(f"[1] 简单名 BP: {bp}")
    bl = unreal.BlueprintEditorLibrary
    int_type = bl.get_basic_type_by_name("int")
    fn_ed = unreal.BlueprintGraphEditor.create_and_edit_function_graph(bp, "MyFunc")
    fn_ed.add_graph_input_parameter("N", int_type)
    unreal.BlueprintEditorLibrary.compile_blueprint(bp)
    _log("  简单名 BP compile OK")

    # 2. 尝试对简单名 BP 创建 MyFunc 调用节点
    _log("[2] 简单名 BP 的 MyFunc 调用:")
    eg = unreal.BlueprintEditorLibrary.find_event_graph(bp)
    ed = unreal.BlueprintGraphEditor.get_graph_editor(eg)
    _try_create(ed, ["类|BPDiagCall|MyFunc", "类|BP_DiagCall|MyFunc", "Class|BPDiagCall|MyFunc"])

    # 3. 尝试对 C4 BP 创建 CaptureOutputFrame 调用节点
    _log("[3] C4 BP 的 CaptureOutputFrame 调用:")
    c4 = unreal.EditorAssetLibrary.load_asset("/Game/FutsalMOT/Blueprints/BP_PoseRecorderC4_G0")
    _log(f"  C4 BP: {c4}")
    if c4 is not None:
        eg2 = unreal.BlueprintEditorLibrary.find_event_graph(c4)
        ed2 = unreal.BlueprintGraphEditor.get_graph_editor(eg2)
        _try_create(ed2, [
            "类|BPPoseRecorderC4_G0|CaptureOutputFrame",
            "类|BPPoseRecorderC4G0|CaptureOutputFrame",
            "类|BP_PoseRecorderC4_G0|CaptureOutputFrame",
        ])

    # 清理简单名 BP
    try:
        unreal.EditorAssetLibrary.delete_asset(simple_path)
        _log("[4] 清理简单名 BP OK")
    except Exception as e:
        _log(f"[4] 清理 ERR: {type(e).__name__}")

    _flush()


if __name__ == "__main__":
    main()
    print("\n结果写入 probe_call_func.log")