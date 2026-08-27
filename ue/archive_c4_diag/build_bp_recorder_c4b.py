"""C4 脚本2：构建 BP_PoseRecorder 的 CaptureOutputFrame（调用 SampleActor ×10 + 计时）。

前置：先运行 build_bp_recorder_c4a.py（建 SampleActor 函数并 compile）。

CaptureOutputFrame(RootFrame, ShotFrame)：
  - 10 组：GetAllActorsWithTag(PoseXX) → Array_Get(0) → Call SampleActor(actor, Root, Shot, "XX")
  - 计时：头 GetGameTime→StartCaptureTime，尾→EndCaptureTime，差值 append CaptureDurations

用法（Unreal Editor Python Console）：
    py "D:/projects/FustalMOT_UEDataset/Content/FutsalMOT/code/ue/build_bp_recorder_c4b.py"
"""

import sys
from pathlib import Path

LOG = []
LOG_PATH = Path(r"D:\projects\FustalMOT_UEDataset\Content\FutsalMOT\code\.futsalmot\build_bp_recorder_c4b.log")
ASSET_PATH = "/Game/FutsalMOT/Blueprints/BP_PoseRecorder"

ACTORS = [
    {"tag": "PoseL0", "actor_id": "L0"},
    {"tag": "PoseL1", "actor_id": "L1"},
    {"tag": "PoseL2", "actor_id": "L2"},
    {"tag": "PoseL3", "actor_id": "L3"},
    {"tag": "PoseL4", "actor_id": "L4"},
    {"tag": "PoseR0", "actor_id": "R0"},
    {"tag": "PoseR1", "actor_id": "R1"},
    {"tag": "PoseR2", "actor_id": "R2"},
    {"tag": "PoseR3", "actor_id": "R3"},
    {"tag": "PoseR4", "actor_id": "R4"},
]
GET_ALL_TAG = "/Script/Engine.GameplayStatics.GetAllActorsWithTag"
ARRAY_GET = "/Script/Engine.KismetArrayLibrary.Array_Get"
GET_TIME = "/Script/Engine.KismetSystemLibrary.GetGameTimeInSeconds"
SUB_FLOAT = "/Script/Engine.KismetMathLibrary.Subtract_FloatFloat"
ARRAY_ADD = "/Script/Engine.KismetArrayLibrary.Array_Add"
MAKE_LITERAL = "/Script/Engine.KismetSystemLibrary.MakeLiteralString"


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
        return editor.add_call_function_node(path)
    except Exception as e:
        _log(f"[3] {label} ERR: {type(e).__name__} {e}")
        return None


def main():
    import unreal
    _log("======== C4b：CaptureOutputFrame（调用 SampleActor ×10）========")

    bp = unreal.EditorAssetLibrary.load_asset(ASSET_PATH)
    _log(f"[1] load BP -> {bp}")
    if bp is None:
        _log("  ERROR: BP 不存在，请先跑 build_bp_recorder_c4a.py")
        _flush()
        return

    bl = unreal.BlueprintEditorLibrary
    int_type = bl.get_basic_type_by_name("int")

    fn_editor = unreal.BlueprintGraphEditor.create_and_edit_function_graph(bp, "CaptureOutputFrame")
    _log(f"[2] CaptureOutputFrame 函数 -> {fn_editor}")
    root_param = fn_editor.add_graph_input_parameter("RootFrame", int_type)
    shot_param = fn_editor.add_graph_input_parameter("ShotFrame", int_type)

    fn = {}
    fn["time"] = _call(fn_editor, GET_TIME, "Fn_GetGameTime")
    fn["setstart"] = fn_editor.add_set_member_variable_node("StartCaptureTime", "")
    fn["setend"] = fn_editor.add_set_member_variable_node("EndCaptureTime", "")
    fn["getstart"] = fn_editor.add_get_member_variable_node("StartCaptureTime", "")
    fn["getend"] = fn_editor.add_get_member_variable_node("EndCaptureTime", "")
    fn["getdurs"] = fn_editor.add_get_member_variable_node("CaptureDurations", "")
    fn["sub"] = _call(fn_editor, SUB_FLOAT, "Fn_SubFloat")
    fn["adddur"] = _call(fn_editor, ARRAY_ADD, "Fn_AddDur")

    fn["gettag"] = []
    fn["get"] = []
    fn["call"] = []
    fn["litactor"] = []
    for tgt in ACTORS:
        fn["gettag"].append(_call(fn_editor, GET_ALL_TAG, f"GetAllTag[{tgt['tag']}]"))
        fn["get"].append(_call(fn_editor, ARRAY_GET, f"ArrayGet[{tgt['tag']}]"))
        call_node = None
        for name in ("类|BPPoseRecorder|SampleActor", "Class|BPPoseRecorder|SampleActor"):
            try:
                call_node = fn_editor.create_node_from_name(name, unreal.Vector2D(0, 0), [])
                if call_node is not None:
                    _log(f"  SampleActor 调用节点[{tgt['actor_id']}] -> {call_node}")
                    break
            except Exception as e:
                _log(f"  SampleActor({name}) ERR: {type(e).__name__}")
        fn["call"].append(call_node)
        fn["litactor"].append(_call(fn_editor, MAKE_LITERAL, f"LitActor[{tgt['actor_id']}]"))

    fn_pins = {}
    for k, v in fn.items():
        if isinstance(v, list):
            fn_pins[k] = [_pins(x) if x is not None else None for x in v]
        elif v is not None:
            fn_pins[k] = _pins(v)

    # 常量
    for i, tgt in enumerate(ACTORS):
        _set_value(fn_pins["gettag"][i], "tag", tgt["tag"], f"GetAllTag[{tgt['tag']}]")
        _set_value(fn_pins["get"][i], "index", "0", f"ArrayGet[{tgt['tag']}]")
        _set_value(fn_pins["litactor"][i], "value", tgt["actor_id"], f"LitActor[{tgt['actor_id']}]")

    # 数据连接
    for i, tgt in enumerate(ACTORS):
        _connect(fn_pins["gettag"][i], "outactors", fn_pins["get"][i], "targetarray", f"Tag→Get[{tgt['tag']}]")
        # SampleActor 调用的 TargetActor 参数（pin 名可能是 targetactor / self / actor）
        target_pin = None
        for cand in ("targetactor", "self", "actor"):
            if cand in (fn_pins["call"][i] or {}):
                target_pin = cand
                break
        if target_pin:
            _connect(fn_pins["get"][i], "item", fn_pins["call"][i], target_pin, f"Get→Sample[{tgt['tag']}]")
        else:
            _log(f"  [conn] Get→Sample[{tgt['tag']}]: 无 target pin（call pins: {sorted(fn_pins['call'][i] or {})}）")
        try:
            root_param.try_create_connection(fn_pins["call"][i]["rootframe"])
        except Exception:
            pass
        try:
            shot_param.try_create_connection(fn_pins["call"][i]["shotframe"])
        except Exception:
            pass
        _connect(fn_pins["litactor"][i], "returnvalue", fn_pins["call"][i], "actorid", f"LitActor→Sample[{tgt['tag']}]")

    # 计时数据
    _connect(fn_pins["time"], "returnvalue", fn_pins["setstart"], "startcapturetime", "Time→Start")
    _connect(fn_pins["time"], "returnvalue", fn_pins["setend"], "endcapturetime", "Time→End")
    _connect(fn_pins["getstart"], "startcapturetime", fn_pins["sub"], "a", "Start→Sub.A")
    _connect(fn_pins["getend"], "endcapturetime", fn_pins["sub"], "b", "End→Sub.B")
    _connect(fn_pins["sub"], "returnvalue", fn_pins["adddur"]["newitem"], "Sub→AddDur")
    _connect(fn_pins["getdurs"], "capturedurations", fn_pins["adddur"], "targetarray", "Durs→AddDur")

    # exec 链：Entry→SetStart→GetTag0→Get0→Sample0→...→GetTag9→Get9→Sample9→SetEnd→Sub→AddDur
    entry_pin = fn_editor.find_graph_entry_pin()
    try:
        entry_pin.try_create_connection(fn_pins["setstart"]["execute"])
        _log("  [conn] Entry→SetStart(exec): True")
    except Exception as e:
        _log(f"  [conn] Entry→SetStart ERR: {type(e).__name__}")

    prev, prev_pin = fn_pins["setstart"], "then"
    for i, tgt in enumerate(ACTORS):
        _connect(prev, prev_pin, fn_pins["gettag"][i], "execute", f"exec→GetTag[{tgt['tag']}]")
        _connect(fn_pins["gettag"][i], "then", fn_pins["get"][i], "execute", f"exec→Get[{tgt['tag']}]")
        # get 是纯函数（Array_Get 无 exec）？实际 Array_Get 有 execute。检查
        # Array_Get 有 execute/then（impure）。连接：
        _connect(fn_pins["get"][i], "then", fn_pins["call"][i], "execute", f"exec→Sample[{tgt['actor_id']}]")
        prev, prev_pin = fn_pins["call"][i], "then"

    _connect(prev, prev_pin, fn_pins["setend"], "execute", "exec→SetEnd")
    _connect(fn_pins["setend"], "then", fn_pins["sub"], "execute", "SetEnd→Sub(exec)")
    _connect(fn_pins["sub"], "then", fn_pins["adddur"], "execute", "Sub→AddDur(exec)")

    # 编译保存 + spawn
    try:
        ok = unreal.BlueprintEditorLibrary.compile_blueprint(bp)
        _log(f"[3] compile -> {ok}")
        unreal.EditorAssetLibrary.save_asset(ASSET_PATH, only_if_is_dirty=True)
        _log("[3] save OK")
    except Exception as e:
        _log(f"[3] compile/save ERR: {type(e).__name__} {e}")

    try:
        cls = unreal.BlueprintEditorLibrary.generated_class(bp)
        ed_sub = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
        for a in ed_sub.get_all_level_actors():
            try:
                if a.get_actor_label() == "BP_PoseRecorder":
                    ed_sub.destroy_actor(a)
            except Exception:
                pass
        spawned = ed_sub.spawn_actor_from_class(cls, unreal.Vector(0, 0, 0), unreal.Rotator(0, 0, 0))
        if spawned is not None:
            spawned.set_actor_label("BP_PoseRecorder")
            _log("[4] 新 BP 已放入关卡（不保存关卡）")
    except Exception as e:
        _log(f"[4] spawn ERR: {type(e).__name__} {e}")

    _flush()


if __name__ == "__main__":
    main()
    print("\n脚本已执行，结果写入 build_bp_recorder_c4b.log")