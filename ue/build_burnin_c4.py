"""构建 WBP_PoseMRQBurnInC4：OnOutputFrameStarted 调 5 个 BP_PoseRecorderC4_G* 的 CaptureOutputFrame。

可靠方法（Phase B 已验证）：
    editor.add_call_function_node("<包>.<生成的_C 类>:CaptureOutputFrame")
    对应 C++ FSoftObjectPath.LoadSynchronous()，不依赖 FBlueprintActionDatabase。

每个 Group BP 负责 2 actor 采样，写独立 slot。WBP 在每输出帧对 5 个 Group 各调一次。
已存在资产优先复用（清空 EventGraph 节点后重建），不 delete_asset。

用法（Unreal Editor Python，FutsalMOTTools.run_python_file）：
    py "D:/projects/FustalMOT_UEDataset/Content/FutsalMOT/code/ue/build_burnin_c4.py"
"""

import sys
from pathlib import Path

LOG = []
LOG_PATH = Path(r"D:\projects\FustalMOT_UEDataset\Content\FutsalMOT\code\.futsalmot\build_burnin_c4.log")
ASSET_PATH = "/Game/FutsalMOT/Blueprints/WBP_PoseMRQBurnInC4"

GROUP_CLASSES = [
    "/Game/FutsalMOT/Blueprints/BP_PoseRecorderC4_G0.BP_PoseRecorderC4_G0_C",
    "/Game/FutsalMOT/Blueprints/BP_PoseRecorderC4_G1.BP_PoseRecorderC4_G1_C",
    "/Game/FutsalMOT/Blueprints/BP_PoseRecorderC4_G2.BP_PoseRecorderC4_G2_C",
    "/Game/FutsalMOT/Blueprints/BP_PoseRecorderC4_G3.BP_PoseRecorderC4_G3_C",
    "/Game/FutsalMOT/Blueprints/BP_PoseRecorderC4_G4.BP_PoseRecorderC4_G4_C",
]
GET_ALL = "/Script/Engine.GameplayStatics.GetAllActorsOfClass"
ARRAY_GET = "/Script/Engine.KismetArrayLibrary.Array_Get"
GET_ROOT = "/Script/MovieRenderPipelineCore.MoviePipelineBlueprintLibrary.GetRootFrameNumber"
GET_SHOT = "/Script/MovieRenderPipelineCore.MoviePipelineBlueprintLibrary.GetCurrentShotFrameNumber"
CONV = "/Script/TimeManagement.TimeManagementBlueprintLibrary.Conv_FrameNumberToInteger"


def _log(msg):
    print(msg)
    LOG.append(str(msg))


def _flush():
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        LOG_PATH.write_text("\n".join(LOG), encoding="utf-8")
    except Exception as e:
        print(f"  写日志失败: {e}")


def _pins(node):
    result = {}
    try:
        for p in node.list_all_pins():
            try:
                result[str(p.get_pin_name()).lower()] = p
            except Exception:
                pass
    except Exception:
        pass
    return result


def _set_value(node_pins, name, value, label):
    p = node_pins.get(name)
    if p is None:
        _log(f"  [set] {label}: 无 {name} pin")
        return
    try:
        ok = p.set_pin_value(value)
        if not ok:
            _log(f"  [set] {label}.{name}={value}: False")
    except Exception as e:
        _log(f"  [set] {label}.{name} ERR: {type(e).__name__} {e}")


def _connect(a_pins, a_name, b_pins, b_name, label):
    a = a_pins.get(a_name)
    b = b_pins.get(b_name)
    if a is None or b is None:
        _log(f"  [conn] {label}: 缺 pin ({a_name}={a is not None}, {b_name}={b is not None})")
        return False
    try:
        ok = a.try_create_connection(b)
        if not ok:
            _log(f"  [conn] {label}: {a_name}->{b_name} = False")
        return ok
    except Exception as e:
        _log(f"  [conn] {label} ERR: {type(e).__name__} {e}")
        return False


def _call(editor, path, label):
    try:
        n = editor.add_call_function_node(path)
        _log(f"[3] {label} -> {n}")
        return n
    except Exception as e:
        _log(f"[3] {label} ERR: {type(e).__name__} {e}")
        return None


def main():
    import unreal
    _log("======== 构建 WBP_PoseMRQBurnInC4（可靠方法：软路径调用节点）========")

    existing = unreal.load_asset(ASSET_PATH)
    if existing is not None:
        _log(f"  已存在，复用 {ASSET_PATH}")
        bp = existing
        event_graph = unreal.BlueprintEditorLibrary.find_event_graph(bp)
        editor = unreal.BlueprintGraphEditor.get_graph_editor(event_graph)
        try:
            nodes = editor.list_all_nodes()
            _log(f"  清空 {len(nodes)} 个旧节点")
            editor.remove_nodes(nodes)
        except Exception as e:
            _log(f"  清空旧节点 ERR: {type(e).__name__} {e}")
    else:
        bp = unreal.BlueprintEditorLibrary.create_blueprint_asset_with_parent(ASSET_PATH, unreal.MoviePipelineBurnInWidget)
        _log(f"[1] create WBP -> {bp}")
        if bp is None:
            _log("  ERROR: 创建失败")
            _flush()
            return
        event_graph = unreal.BlueprintEditorLibrary.find_event_graph(bp)
        editor = unreal.BlueprintGraphEditor.get_graph_editor(event_graph)

    onframe = unreal.BlueprintEditorLibrary.add_event_override(bp, "OnOutputFrameStarted", unreal.IntPoint(0, 0))
    _log(f"[2] OnOutputFrameStarted -> {onframe}")

    # 共享：root/shot 帧号
    root = _call(editor, GET_ROOT, "GetRootFrameNumber")
    shot = _call(editor, GET_SHOT, "GetCurrentShotFrameNumber")
    convroot = _call(editor, CONV, "ConvRoot")
    convshot = _call(editor, CONV, "ConvShot")

    groups = {}
    for g, cls in zip(["G0", "G1", "G2", "G3", "G4"], GROUP_CLASSES):
        getall = _call(editor, GET_ALL, f"GetAll[{g}]")
        get = _call(editor, ARRAY_GET, f"ArrayGet[{g}]")
        capture = _call(editor, f"{cls}:CaptureOutputFrame", f"Capture[{g}]")
        groups[g] = {"cls": cls, "getall": getall, "get": get, "capture": capture}

    if any(v["getall"] is None or v["get"] is None or v["capture"] is None for v in groups.values()):
        _log("  ERROR: 节点创建失败")
        _flush()
        return

    pins = {"onframe": _pins(onframe), "root": _pins(root), "shot": _pins(shot),
            "convroot": _pins(convroot), "convshot": _pins(convshot)}
    for g in groups:
        pins[f"getall_{g}"] = _pins(groups[g]["getall"])
        pins[f"get_{g}"] = _pins(groups[g]["get"])
        pins[f"capture_{g}"] = _pins(groups[g]["capture"])

    # 常量
    for g in groups:
        _set_value(pins[f"getall_{g}"], "actorclass", groups[g]["cls"], f"GetAll[{g}]")
        _set_value(pins[f"get_{g}"], "index", "0", f"ArrayGet[{g}]")

    # 数据连接（GetAll→Get 共享；conv 结果接每个 capture 的 RootFrame/ShotFrame）
    for g in groups:
        _connect(pins["onframe"], "forpipeline", pins["root"], "inmoviepipeline", f"Pipeline→Root[{g}]")
        _connect(pins["onframe"], "forpipeline", pins["shot"], "inmoviepipeline", f"Pipeline→Shot[{g}]")
        _connect(pins["root"], "returnvalue", pins["convroot"], "inframenumber", f"Root→Conv[{g}]")
        _connect(pins["shot"], "returnvalue", pins["convshot"], "inframenumber", f"Shot→Conv[{g}]")
        _connect(pins[f"getall_{g}"], "outactors", pins[f"get_{g}"], "targetarray", f"Actors→Get[{g}]")

    # 目标连接：get.item → capture.self（GetAll 的 ActorClass 已设为具体类，outactors 应已重定型）
    for g in groups:
        ok = _connect(pins[f"get_{g}"], "item", pins[f"capture_{g}"], "self", f"Actor→Capture[{g}].Self")
        if not ok:
            _log(f"  [{g}] 直接连接失败，尝试 Cast 节点")
            try:
                avail = editor.list_available_nodes([])
                cast_name = next((s for s in avail if f"casttobpposerecorderc4g{g.lower()}" in s.lower()), None)
                _log(f"  [{g}] cast 名: {cast_name}")
                if cast_name:
                    cast_node = editor.create_node_from_name(cast_name, unreal.Vector2D(0, 0), [])
                    _log(f"  [{g}] cast -> {cast_node}")
                    if cast_node is not None:
                        cast_pins = _pins(cast_node)
                        _connect(pins[f"get_{g}"], "item", cast_pins, "object", f"Actor→Cast[{g}].Object")
                        cast_out = next((c for c in cast_pins if c not in ("object", "execute", "then", "castfailed")), None)
                        _connect(cast_pins, cast_out, pins[f"capture_{g}"], "self", f"Cast→Capture[{g}].Self")
            except Exception as e:
                _log(f"  [{g}] cast ERR: {type(e).__name__} {e}")
        _connect(pins["convroot"], "returnvalue", pins[f"capture_{g}"], "rootframe", f"Root→Capture[{g}].Root")
        _connect(pins["convshot"], "returnvalue", pins[f"capture_{g}"], "shotframe", f"Shot→Capture[{g}].Shot")

    # exec 链：OnFrame→GetAllG0→CaptureG0→GetAllG1→...
    prev_pin = pins["onframe"]
    prev_name = "then"
    for g in groups:
        _connect(prev_pin, prev_name, pins[f"getall_{g}"], "execute", f"exec→GetAll[{g}]")
        _connect(pins[f"getall_{g}"], "then", pins[f"capture_{g}"], "execute", f"exec→Capture[{g}]")
        prev_pin, prev_name = pins[f"capture_{g}"], "then"

    # 编译保存
    try:
        ok = unreal.BlueprintEditorLibrary.compile_blueprint(bp)
        _log(f"[4] compile -> {ok}")
        saved = unreal.EditorAssetLibrary.save_asset(ASSET_PATH, only_if_is_dirty=True)
        _log(f"[5] save -> {saved}")
    except Exception as e:
        _log(f"[4] compile/save ERR: {type(e).__name__} {e}")

    _flush()


if __name__ == "__main__":
    main()