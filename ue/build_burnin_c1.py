"""构建 WBP_PoseMRQBurnIn C1：OnOutputFrameStarted 调用 BP_PoseRecorder.CaptureOutputFrame。

用法（Unreal Editor Python Console）：
    py "D:/projects/FustalMOT_UEDataset/Content/FutsalMOT/code/ue/build_burnin_c1.py"

OnOutputFrameStarted(ForPipeline)
  → root = Conv(GetRootFrameNumber(ForPipeline))
  → shot = Conv(GetCurrentShotFrameNumber(ForPipeline))
  → BP = GetAllActorsOfClass(BP_PoseRecorder) → Get(0)
  → Call CaptureOutputFrame(BP, Root, Shot)
  → 打印 root/shot（调试）
"""

import sys
from pathlib import Path

LOG = []
LOG_PATH = Path(r"D:\projects\FustalMOT_UEDataset\Content\FutsalMOT\code\.futsalmot\build_burnin_c1.log")
ASSET_PATH = "/Game/FutsalMOT/Blueprints/WBP_PoseMRQBurnIn"
RECORDER_CLASS = "/Game/FutsalMOT/Blueprints/BP_PoseRecorder.BP_PoseRecorder_C"
RECORDER_FUNC = "/Game/FutsalMOT/Blueprints/BP_PoseRecorder.CaptureOutputFrame"


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
    _log("======== 构建 WBP_PoseMRQBurnIn（C1）========")

    # 先保存关卡（World Partition 一致性，防止 delete_asset 触发 umap 丢失）
    try:
        lvl = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
        lvl.save_current_level()
        _log("  关卡已保存")
    except Exception as e:
        _log(f"  保存关卡 ERR: {type(e).__name__}")

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
    nodes["getall"] = _call("/Script/Engine.GameplayStatics.GetAllActorsOfClass", "GetAllActorsOfClass")
    nodes["get"] = _call("/Script/Engine.KismetArrayLibrary.Array_Get", "Array_Get")
    # CaptureOutputFrame 调用节点 + Cast（本地化节点名）
    capture_node = None
    for name in ("类|BPPoseRecorder|CaptureOutputFrame", "Class|BPPoseRecorder|CaptureOutputFrame"):
        try:
            capture_node = editor.create_node_from_name(name, unreal.Vector2D(0, 0), [])
            if capture_node is not None:
                _log(f"[3] CaptureOutputFrame 节点 -> {capture_node}")
                break
        except Exception as e:
            _log(f"[3] Capture({name}) ERR: {type(e).__name__}")
    nodes["capture"] = capture_node

    cast_node = None
    for name in ("工具|Casting|CastToBP_PoseRecorder", "Utilities|Casting|CastToBP_PoseRecorder"):
        try:
            cast_node = editor.create_node_from_name(name, unreal.Vector2D(0, 0), [])
            if cast_node is not None:
                _log(f"[3] CastToBP_PoseRecorder -> {cast_node}")
                break
        except Exception as e:
            _log(f"[3] Cast({name}) ERR: {type(e).__name__}")
    nodes["cast"] = cast_node

    if None in nodes.values():
        _log("  ERROR: 节点创建失败")
        _flush()
        return

    pins = {k: _pins(v) for k, v in nodes.items()}
    pins["onframe"] = _pins(onframe)
    for k, p in pins.items():
        _log(f"  {k} pins: {sorted(p)}")

    # 参数
    _set_value(pins["getall"], "actorclass", RECORDER_CLASS, "GetAllActorsOfClass")
    _set_value(pins["get"], "index", "0", "Array_Get")

    # 数据链
    _connect(pins["onframe"], "forpipeline", pins["root"], "inmoviepipeline", "Pipeline→Root")
    _connect(pins["onframe"], "forpipeline", pins["shot"], "inmoviepipeline", "Pipeline→Shot")
    _connect(pins["root"], "returnvalue", pins["convroot"], "inframenumber", "Root→Conv")
    _connect(pins["shot"], "returnvalue", pins["convshot"], "inframenumber", "Shot→Conv")
    _connect(pins["getall"], "outactors", pins["get"], "targetarray", "Actors→Get")
    # Cast to BP_PoseRecorder
    cast_out = next((c for c in pins["cast"] if c not in ("object", "execute", "then", "castfailed")), None)
    _log(f"  cast 输出: {cast_out}")
    _connect(pins["get"], "item", pins["cast"], "object", "Actor0→Cast")
    # 调用 CaptureOutputFrame：Target=cast 输出(DBP_PoseRecorder), RootFrame, ShotFrame
    capture_pins = pins["capture"]
    target_pin = next((c for c in capture_pins if c in ("self", "target", "bpposerecorder")), None)
    root_in = next((c for c in capture_pins if "root" in c), None)
    shot_in = next((c for c in capture_pins if "shot" in c), None)
    _log(f"  capture target={target_pin} root_in={root_in} shot_in={shot_in}")
    _connect(pins["cast"], cast_out, pins["capture"], target_pin, "BP→Capture.Target")
    _connect(pins["convroot"], "returnvalue", pins["capture"], root_in, "Root→Capture.Root")
    _connect(pins["convshot"], "returnvalue", pins["capture"], shot_in, "Shot→Capture.Shot")

    # exec 链
    _connect(pins["onframe"], "then", pins["getall"], "execute", "OnFrame→GetAll(exec)")
    _connect(pins["getall"], "then", pins["cast"], "execute", "GetAll→Cast(exec)")
    _connect(pins["cast"], "then", pins["capture"], "execute", "Cast→Capture(exec)")

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
    print("\n脚本已执行，结果写入 build_burnin_c1.log")