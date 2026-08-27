"""C4 脚本1：构建 BP_PoseRecorder 的 SampleActor 函数 + EndPlay（单 actor 13 骨）。

架构（函数化，避免 10×13 内联 1820 节点压垮 Undo 事务）：
  - SampleActor(TargetActor, RootFrame, ShotFrame, ActorId)：对单 actor 采 13 骨 → 7 数组
  - EndPlay：填 SG（含 capture_durations）
  - 本脚本 compile 后，脚本2 用 create_node_from_name 调用 SampleActor ×10

8 平行数组 + CaptureDurations(每帧1条) + Start/EndCaptureTime。
用法（Unreal Editor Python Console）：
    py "D:/projects/FustalMOT_UEDataset/Content/FutsalMOT/code/ue/build_bp_recorder_c4a.py"
"""

import sys
from pathlib import Path

LOG = []
LOG_PATH = Path(r"D:\projects\FustalMOT_UEDataset\Content\FutsalMOT\code\.futsalmot\build_bp_recorder_c4a.log")
ASSET_PATH = "/Game/FutsalMOT/Blueprints/BP_PoseRecorder"
SG_PATH = "/Game/FutsalMOT/Blueprints/SG_PoseCapture.SG_PoseCapture_C"
SKELETAL_COMP = "/Script/Engine.SkeletalMeshComponent"
SLOT_NAME = "PoseCapture"

BONES = [
    "head",
    "upperarm_l", "upperarm_r",
    "lowerarm_l", "lowerarm_r",
    "hand_l", "hand_r",
    "thigh_l", "thigh_r",
    "calf_l", "calf_r",
    "foot_l", "foot_r",
]
GET_SOCKET_LOC = "/Script/Engine.SceneComponent.GetSocketLocation"
GET_SOCKET_ROT = "/Script/Engine.SceneComponent.GetSocketRotation"
ARRAY_ADD = "/Script/Engine.KismetArrayLibrary.Array_Add"
MAKE_LITERAL = "/Script/Engine.KismetSystemLibrary.MakeLiteralString"
GET_COMP = "/Script/Engine.Actor.GetComponentByClass"
GET_TIME = "/Script/Engine.KismetSystemLibrary.GetGameTimeInSeconds"


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
    _log("======== C4a：SampleActor 函数 + EndPlay ========")

    # 先销毁关卡中的旧 BP_PoseRecorder 实例（避免 delete_asset 后悬空引用触发事务崩溃）
    try:
        ed_sub = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
        for a in list(ed_sub.get_all_level_actors()):
            try:
                if a.get_actor_label() == "BP_PoseRecorder":
                    ed_sub.destroy_actor(a)
                    _log("  销毁旧 BP_PoseRecorder 实例")
            except Exception:
                pass
    except Exception as e:
        _log(f"  初始化 ERR: {type(e).__name__}")

    if unreal.EditorAssetLibrary.does_asset_exist(ASSET_PATH):
        unreal.EditorAssetLibrary.delete_asset(ASSET_PATH)
    bp = unreal.BlueprintEditorLibrary.create_blueprint_asset_with_parent(ASSET_PATH, unreal.Actor)
    _log(f"[1] 创建蓝图 -> {bp}")

    bl = unreal.BlueprintEditorLibrary
    int_type = bl.get_basic_type_by_name("int")
    real_type = bl.get_basic_type_by_name("real")
    arr_int = bl.get_array_type(int_type)
    arr_real = bl.get_array_type(real_type)
    vec_type = bl.get_struct_type(unreal.Vector.static_struct())
    arr_vec = bl.get_array_type(vec_type)
    rot_type = bl.get_struct_type(unreal.Rotator.static_struct())
    arr_rot = bl.get_array_type(rot_type)
    str_type = bl.get_basic_type_by_name("string")
    arr_str = bl.get_array_type(str_type)
    savegame_ref = bl.get_object_reference_type(unreal.SaveGame)
    for name, pt, label in [
        ("CaptureIndices", arr_int, "int[]"), ("ShotFrames", arr_int, "int[]"),
        ("GameTimes", arr_real, "float[]"), ("ActorIds", arr_str, "string[]"),
        ("BoneNames", arr_str, "string[]"), ("WorldLocations", arr_vec, "Vector[]"),
        ("WorldRotations", arr_rot, "Rotator[]"), ("SaveGameRef", savegame_ref, "SaveGame"),
        ("CaptureDurations", arr_real, "float[]"), ("StartCaptureTime", real_type, "float"),
        ("EndCaptureTime", real_type, "float"),
    ]:
        _log(f"  add {name}: {bl.add_member_variable(bp, name, pt)}")

    # ---- SampleActor 函数 ----
    fn_editor = unreal.BlueprintGraphEditor.create_and_edit_function_graph(bp, "SampleActor")
    _log(f"[2] SampleActor 函数 -> {fn_editor}")
    target_param = fn_editor.add_graph_input_parameter("TargetActor", bl.get_object_reference_type(unreal.Actor))
    root_param = fn_editor.add_graph_input_parameter("RootFrame", int_type)
    shot_param = fn_editor.add_graph_input_parameter("ShotFrame", int_type)
    actorid_param = fn_editor.add_graph_input_parameter("ActorId", str_type)

    fn = {}
    fn["comp"] = _call(fn_editor, GET_COMP, "Fn_GetComponentByClass")
    fn["time"] = _call(fn_editor, GET_TIME, "Fn_GetGameTime")
    fn["getcap"] = fn_editor.add_get_member_variable_node("CaptureIndices", "")
    fn["getshot"] = fn_editor.add_get_member_variable_node("ShotFrames", "")
    fn["gettime"] = fn_editor.add_get_member_variable_node("GameTimes", "")
    fn["getactor"] = fn_editor.add_get_member_variable_node("ActorIds", "")
    fn["getbone"] = fn_editor.add_get_member_variable_node("BoneNames", "")
    fn["getworld"] = fn_editor.add_get_member_variable_node("WorldLocations", "")
    fn["getrot"] = fn_editor.add_get_member_variable_node("WorldRotations", "")
    fn["litbone"] = []
    fn["socket"] = []
    fn["rot"] = []
    fn["addcap"] = []
    fn["addshot"] = []
    fn["addtime"] = []
    fn["addactor"] = []
    fn["addbone"] = []
    fn["addloc"] = []
    fn["addrot"] = []
    for bone in BONES:
        fn["litbone"].append(_call(fn_editor, MAKE_LITERAL, f"LitBone[{bone}]"))
        fn["socket"].append(_call(fn_editor, GET_SOCKET_LOC, f"Socket[{bone}]"))
        fn["rot"].append(_call(fn_editor, GET_SOCKET_ROT, f"Rot[{bone}]"))
        fn["addcap"].append(_call(fn_editor, ARRAY_ADD, f"AddCap[{bone}]"))
        fn["addshot"].append(_call(fn_editor, ARRAY_ADD, f"AddShot[{bone}]"))
        fn["addtime"].append(_call(fn_editor, ARRAY_ADD, f"AddTime[{bone}]"))
        fn["addactor"].append(_call(fn_editor, ARRAY_ADD, f"AddActor[{bone}]"))
        fn["addbone"].append(_call(fn_editor, ARRAY_ADD, f"AddBone[{bone}]"))
        fn["addloc"].append(_call(fn_editor, ARRAY_ADD, f"AddLoc[{bone}]"))
        fn["addrot"].append(_call(fn_editor, ARRAY_ADD, f"AddRot[{bone}]"))

    fn_pins = {}
    for k, v in fn.items():
        if isinstance(v, list):
            fn_pins[k] = [_pins(x) if x is not None else None for x in v]
        elif v is not None:
            fn_pins[k] = _pins(v)

    _set_value(fn_pins["comp"], "componentclass", SKELETAL_COMP, "Comp")
    for j, bone in enumerate(BONES):
        _set_value(fn_pins["socket"][j], "insocketname", bone, f"Socket[{bone}]")
        _set_value(fn_pins["rot"][j], "insocketname", bone, f"Rot[{bone}]")
        _set_value(fn_pins["litbone"][j], "value", bone, f"LitBone[{bone}]")

    # TargetActor → comp.self
    try:
        target_param.try_create_connection(fn_pins["comp"]["self"])
        _log("  [conn] TargetActor→Comp.Self: True")
    except Exception as e:
        _log(f"  [conn] TargetActor→Comp.Self ERR: {type(e).__name__}")

    for j, bone in enumerate(BONES):
        _connect(fn_pins["comp"], "returnvalue", fn_pins["socket"][j], "self", f"Comp→Socket")
        _connect(fn_pins["comp"], "returnvalue", fn_pins["rot"][j], "self", f"Comp→Rot")
        _connect(fn_pins["getcap"], "captureindices", fn_pins["addcap"][j], "targetarray", "Cap→AddCap")
        _connect(fn_pins["getshot"], "shotframes", fn_pins["addshot"][j], "targetarray", "Shot→AddShot")
        _connect(fn_pins["gettime"], "gametimes", fn_pins["addtime"][j], "targetarray", "Time→AddTime")
        _connect(fn_pins["getactor"], "actorids", fn_pins["addactor"][j], "targetarray", "Actor→AddActor")
        _connect(fn_pins["getbone"], "bonenames", fn_pins["addbone"][j], "targetarray", "Bone→AddBone")
        _connect(fn_pins["getworld"], "worldlocations", fn_pins["addloc"][j], "targetarray", "World→AddLoc")
        _connect(fn_pins["getrot"], "worldrotations", fn_pins["addrot"][j], "targetarray", "Rot→AddRot")
        _connect(fn_pins["socket"][j], "returnvalue", fn_pins["addloc"][j], "newitem", "Loc→AddLoc")
        _connect(fn_pins["rot"][j], "returnvalue", fn_pins["addrot"][j], "newitem", "Rot→AddRot")
        _connect(fn_pins["time"], "returnvalue", fn_pins["addtime"][j], "newitem", "Time→AddTime")
        _connect(fn_pins["litbone"][j], "returnvalue", fn_pins["addbone"][j], "newitem", "LitBone→AddBone")
        try:
            actorid_param.try_create_connection(fn_pins["addactor"][j]["newitem"])
        except Exception:
            pass
        try:
            root_param.try_create_connection(fn_pins["addcap"][j]["newitem"])
        except Exception:
            pass
        try:
            shot_param.try_create_connection(fn_pins["addshot"][j]["newitem"])
        except Exception:
            pass

    # 入口 exec → addcap[0]
    entry_pin = fn_editor.find_graph_entry_pin()
    try:
        ok = entry_pin.try_create_connection(fn_pins["addcap"][0]["execute"])
        _log(f"  [conn] Entry→AddCap0(exec): {ok}")
    except Exception as e:
        _log(f"  [conn] Entry→AddCap0(exec) ERR: {type(e).__name__} {e}")

    prev, prev_pin = fn_pins["addcap"][0], "then"
    for j in range(len(BONES)):
        group = [fn_pins["addcap"][j], fn_pins["addshot"][j], fn_pins["addtime"][j],
                 fn_pins["addactor"][j], fn_pins["addbone"][j], fn_pins["addloc"][j],
                 fn_pins["addrot"][j]]
        for k, node_pins in enumerate(group):
            if j == 0 and k == 0:
                continue
            _connect(prev, prev_pin, node_pins, "execute", f"exec→{BONES[j]}.{k}")
            prev, prev_pin = node_pins, "then"

    # ---- EndPlay SaveGame ----
    event_graph = unreal.BlueprintEditorLibrary.find_event_graph(bp)
    ed_editor = unreal.BlueprintGraphEditor.get_graph_editor(event_graph)
    endplay_node = unreal.BlueprintEditorLibrary.add_event_override(bp, "ReceiveEndPlay", unreal.IntPoint(0, 0))
    _log(f"[3] EndPlay -> {endplay_node}")

    nodes = {}
    nodes["create"] = _call(ed_editor, "/Script/Engine.GameplayStatics.CreateSaveGameObject", "Create")
    nodes["setref"] = ed_editor.add_set_member_variable_node("SaveGameRef", "")
    nodes["getref"] = ed_editor.add_get_member_variable_node("SaveGameRef", "")
    nodes["getcap"] = ed_editor.add_get_member_variable_node("CaptureIndices", "")
    nodes["getshot"] = ed_editor.add_get_member_variable_node("ShotFrames", "")
    nodes["gettime"] = ed_editor.add_get_member_variable_node("GameTimes", "")
    nodes["getactor"] = ed_editor.add_get_member_variable_node("ActorIds", "")
    nodes["getbone"] = ed_editor.add_get_member_variable_node("BoneNames", "")
    nodes["getworld"] = ed_editor.add_get_member_variable_node("WorldLocations", "")
    nodes["getrot"] = ed_editor.add_get_member_variable_node("WorldRotations", "")
    nodes["getdurs"] = ed_editor.add_get_member_variable_node("CaptureDurations", "")
    nodes["setcap"] = ed_editor.add_set_member_variable_node("capture_indices", SG_PATH)
    nodes["setshot"] = ed_editor.add_set_member_variable_node("shot_frames", SG_PATH)
    nodes["settime"] = ed_editor.add_set_member_variable_node("game_times", SG_PATH)
    nodes["setactor"] = ed_editor.add_set_member_variable_node("actor_ids", SG_PATH)
    nodes["setbone"] = ed_editor.add_set_member_variable_node("bone_names", SG_PATH)
    nodes["setworld"] = ed_editor.add_set_member_variable_node("world_locations", SG_PATH)
    nodes["setrot"] = ed_editor.add_set_member_variable_node("world_rotations", SG_PATH)
    nodes["setdurs"] = ed_editor.add_set_member_variable_node("capture_durations", SG_PATH)
    nodes["arrlen"] = _call(ed_editor, "/Script/Engine.KismetArrayLibrary.Array_Length", "ArrayLen")
    nodes["settotal"] = ed_editor.add_set_member_variable_node("total_samples", SG_PATH)
    nodes["save"] = _call(ed_editor, "/Script/Engine.GameplayStatics.SaveGameToSlot", "SaveToSlot")

    if None in nodes.values():
        _log("  ERROR: EndPlay 节点创建失败")
        _flush()
        return

    pins = {k: _pins(v) for k, v in nodes.items()}
    pins["endplay"] = _pins(endplay_node)

    _set_value(pins["create"], "savegameclass", SG_PATH, "Create")
    _set_value(pins["save"], "slotname", SLOT_NAME, "SaveToSlot")
    _set_value(pins["save"], "userindex", "0", "SaveToSlot")

    _connect(pins["create"], "returnvalue", pins["setref"], "savegameref", "Create→SetRef")
    _connect(pins["getcap"], "captureindices", pins["setcap"], "capture_indices", "Cap→SG.cap")
    _connect(pins["getshot"], "shotframes", pins["setshot"], "shot_frames", "Shot→SG.shot")
    _connect(pins["gettime"], "gametimes", pins["settime"], "game_times", "Time→SG.time")
    _connect(pins["getactor"], "actorids", pins["setactor"], "actor_ids", "Actor→SG.actor")
    _connect(pins["getbone"], "bonenames", pins["setbone"], "bone_names", "Bone→SG.bone")
    _connect(pins["getworld"], "worldlocations", pins["setworld"], "world_locations", "World→SG.world")
    _connect(pins["getrot"], "worldrotations", pins["setrot"], "world_rotations", "Rot→SG.rot")
    _connect(pins["getdurs"], "capturedurations", pins["setdurs"], "capture_durations", "Durs→SG.durs")
    _connect(pins["getcap"], "captureindices", pins["arrlen"], "targetarray", "Cap→Len")
    _connect(pins["arrlen"], "returnvalue", pins["settotal"], "total_samples", "Len→SG.total")
    for k in ("setcap", "setshot", "settime", "setactor", "setbone", "setworld", "setrot", "setdurs", "settotal"):
        _connect(pins["create"], "returnvalue", pins[k], "self", f"Create→{k}.self")
    _connect(pins["getref"], "savegameref", pins["save"], "savegameobject", "Ref→Save")

    _connect(pins["endplay"], "then", pins["create"], "execute", "EndPlay→Create(exec)")
    _connect(pins["create"], "then", pins["setref"], "execute", "Create→SetRef(exec)")
    _connect(pins["setref"], "then", pins["setcap"], "execute", "SetRef→SetCap(exec)")
    _connect(pins["setcap"], "then", pins["setshot"], "execute", "SetCap→SetShot(exec)")
    _connect(pins["setshot"], "then", pins["settime"], "execute", "SetShot→SetTime(exec)")
    _connect(pins["settime"], "then", pins["setactor"], "execute", "SetTime→SetActor(exec)")
    _connect(pins["setactor"], "then", pins["setbone"], "execute", "SetActor→SetBone(exec)")
    _connect(pins["setbone"], "then", pins["setworld"], "execute", "SetBone→SetWorld(exec)")
    _connect(pins["setworld"], "then", pins["setrot"], "execute", "SetWorld→SetRot(exec)")
    _connect(pins["setrot"], "then", pins["setdurs"], "execute", "SetRot→SetDurs(exec)")
    _connect(pins["setdurs"], "then", pins["settotal"], "execute", "SetDurs→SetTotal(exec)")
    _connect(pins["settotal"], "then", pins["save"], "execute", "SetTotal→Save(exec)")

    # 编译保存（让 SampleActor 可被脚本2 create_node_from_name 找到）
    try:
        ok = unreal.BlueprintEditorLibrary.compile_blueprint(bp)
        _log(f"[4] compile -> {ok}")
        unreal.EditorAssetLibrary.save_asset(ASSET_PATH, only_if_is_dirty=True)
        _log("[4] save OK")
    except Exception as e:
        _log(f"[4] compile/save ERR: {type(e).__name__} {e}")

    _flush()


if __name__ == "__main__":
    main()
    print("\n脚本已执行，结果写入 build_bp_recorder_c4a.log")