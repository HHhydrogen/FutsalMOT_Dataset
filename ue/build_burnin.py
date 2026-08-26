"""构建 WBP_PoseMRQBurnIn 完整回调：OnOutputFrameStarted → 打印帧信息。

用法（Unreal Editor Python Console）：
    py "D:/projects/FustalMOT_UEDataset/Content/FutsalMOT/code/ue/build_burnin.py"

OnOutputFrameStarted(ForPipeline)
  → GetRootFrameNumber(ForPipeline) → Conv_FrameNumberToInteger → root
  → GetCurrentShotFrameNumber(ForPipeline) → Conv → shot
  → GetGameTimeInSeconds → gameTime
  → 打印 "OUTPUT CALLBACK | root=... | shot=... | gameTime=..."
"""

import sys
from pathlib import Path

LOG = []
LOG_PATH = Path(r"D:\projects\FustalMOT_UEDataset\Content\FutsalMOT\code\.futsalmot\build_burnin.log")
ASSET_PATH = "/Game/FutsalMOT/Blueprints/WBP_PoseMRQBurnIn"


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
    except Exception as e:
        _log(f"  [pins] ERR: {type(e).__name__}")
    return result


def _set_value(node_pins, name, value, label):
    p = node_pins.get(name)
    if p is None:
        _log(f"  [set] {label}: 无 {name} pin")
        return
    try:
        ok = p.set_pin_value(value)
        _log(f"  [set] {label}.{name}={value}: {ok}")
    except Exception as e:
        _log(f"  [set] {label}.{name} ERR: {type(e).__name__} {e}")


def _connect(a_pins, a_name, b_pins, b_name, label):
    a = a_pins.get(a_name)
    b = b_pins.get(b_name)
    if a is None or b is None:
        _log(f"  [conn] {label}: 缺 pin ({a_name}={a is not None}, {b_name}={b is not None})")
        return
    try:
        ok = a.try_create_connection(b)
        _log(f"  [conn] {label}: {a_name}->{b_name} = {ok}")
    except Exception as e:
        _log(f"  [conn] {label} ERR: {type(e).__name__} {e}")


def main():
    import unreal
    _log("======== 构建 WBP_PoseMRQBurnIn 完整回调 ========")

    if unreal.EditorAssetLibrary.does_asset_exist(ASSET_PATH):
        unreal.EditorAssetLibrary.delete_asset(ASSET_PATH)
    bp = unreal.BlueprintEditorLibrary.create_blueprint_asset_with_parent(ASSET_PATH, unreal.MoviePipelineBurnInWidget)
    _log(f"[1] create WBP -> {bp}")

    event_graph = unreal.BlueprintEditorLibrary.find_event_graph(bp)
    editor = unreal.BlueprintGraphEditor.get_graph_editor(event_graph)

    onframe = unreal.BlueprintEditorLibrary.add_event_override(bp, "OnOutputFrameStarted", unreal.IntPoint(0, 0))
    _log(f"[2] OnOutputFrameStarted -> {onframe}")

    def _call(path, label):
        try:
            n = editor.add_call_function_node(path)
            _log(f"[3] {label} -> {n}")
            return n
        except Exception as e:
            _log(f"[3] {label} ERR: {type(e).__name__} {e}")
            return None

    nodes = {}
    nodes["root"] = _call("/Script/MovieRenderPipelineCore.MoviePipelineBlueprintLibrary.GetRootFrameNumber", "GetRootFrameNumber")
    nodes["shot"] = _call("/Script/MovieRenderPipelineCore.MoviePipelineBlueprintLibrary.GetCurrentShotFrameNumber", "GetCurrentShotFrameNumber")
    nodes["convroot"] = _call("/Script/TimeManagement.TimeManagementBlueprintLibrary.Conv_FrameNumberToInteger", "ConvRoot")
    nodes["convshot"] = _call("/Script/TimeManagement.TimeManagementBlueprintLibrary.Conv_FrameNumberToInteger", "ConvShot")
    nodes["time"] = _call("/Script/Engine.KismetSystemLibrary.GetGameTimeInSeconds", "GetGameTimeInSeconds")
    nodes["rstr"] = _call("/Script/Engine.KismetStringLibrary.Conv_IntToString", "RootToString")
    nodes["sstr"] = _call("/Script/Engine.KismetStringLibrary.Conv_IntToString", "ShotToString")
    nodes["tstr"] = _call("/Script/Engine.KismetStringLibrary.Conv_DoubleToString", "TimeToString")
    nodes["c1"] = _call("/Script/Engine.KismetStringLibrary.Concat_StrStr", "Concat1")
    nodes["c2"] = _call("/Script/Engine.KismetStringLibrary.Concat_StrStr", "Concat2")
    nodes["c3"] = _call("/Script/Engine.KismetStringLibrary.Concat_StrStr", "Concat3")
    nodes["c4"] = _call("/Script/Engine.KismetStringLibrary.Concat_StrStr", "Concat4")
    nodes["print"] = _call("/Script/Engine.KismetSystemLibrary.PrintString", "PrintString")

    if None in nodes.values():
        _log("  ERROR: 节点创建失败")
        _flush()
        return

    pins = {k: _pins(v) for k, v in nodes.items()}
    pins["onframe"] = _pins(onframe)
    for k, p in pins.items():
        _log(f"  {k} pins: {sorted(p)}")

    # 参数
    _set_value(pins["c1"], "a", "OUTPUT CALLBACK | root=", "Concat1.a")
    _set_value(pins["c2"], "a", " | shot=", "Concat2.a")
    _set_value(pins["c3"], "a", " | gameTime=", "Concat3.a")
    _set_value(pins["print"], "bprinttolog", "True", "PrintString")

    # 数据链
    _connect(pins["onframe"], "forpipeline", pins["root"], "inmoviepipeline", "Pipeline→Root")
    _connect(pins["onframe"], "forpipeline", pins["shot"], "inmoviepipeline", "Pipeline→Shot")
    _connect(pins["root"], "returnvalue", pins["convroot"], "inframenumber", "RootFrame→Conv")
    _connect(pins["shot"], "returnvalue", pins["convshot"], "inframenumber", "ShotFrame→Conv")
    _connect(pins["convroot"], "returnvalue", pins["rstr"], "inint", "RootInt→Str")
    _connect(pins["convshot"], "returnvalue", pins["sstr"], "inint", "ShotInt→Str")
    _connect(pins["time"], "returnvalue", pins["tstr"], "indouble", "Time→Str")
    _connect(pins["rstr"], "returnvalue", pins["c1"], "b", "RootStr→Concat1.B")
    _connect(pins["sstr"], "returnvalue", pins["c2"], "b", "ShotStr→Concat2.B")
    _connect(pins["tstr"], "returnvalue", pins["c3"], "b", "TimeStr→Concat3.B")
    _connect(pins["c1"], "returnvalue", pins["c2"], "a", "C1→C2.A")
    _connect(pins["c2"], "returnvalue", pins["c3"], "a", "C2→C3.A")
    _connect(pins["c3"], "returnvalue", pins["c4"], "a", "C3→C4.A")
    _connect(pins["c3"], "returnvalue", pins["print"], "instring", "C3→PrintString")

    # exec 链
    _connect(pins["onframe"], "then", pins["print"], "execute", "OnOutputFrameStarted→Print(exec)")

    # 编译保存
    try:
        ok = unreal.BlueprintEditorLibrary.compile_blueprint(bp)
        _log(f"[4] compile -> {ok}")
        unreal.EditorAssetLibrary.save_asset(ASSET_PATH, only_if_is_dirty=True)
        _log("[5] save OK")
    except Exception as e:
        _log(f"[4] compile/save ERR: {type(e).__name__} {e}")

    _flush()


if __name__ == "__main__":
    main()
    print("\n脚本已执行，结果写入 build_burnin.log")