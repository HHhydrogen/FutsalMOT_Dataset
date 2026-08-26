"""构建 BP_PoseRecorder 阶段 C2：CaptureOutputFrame 采样 5 骨骼 + SaveGame。

在已验证的 C1（OnOutputFrameStarted 同帧采样）基础上，把单骨骼 hand_l 扩展为 5 骨骼：
hand_l / lowerarm_l / thigh_l / calf_l / foot_l。

每帧 5 条记录（同一 root_frame），存入 6 个平行数组：
  CaptureIndices(int[]) / ShotFrames(int[]) / GameTimes(float[])
  ActorIds(string[]) / BoneNames(string[]) / WorldLocations(Vector[])
总记录数 = 90 帧 × 5 骨骼 = 450。

要点（相对 C1 的修复）：
  - 函数入口 exec pin 直接 try_create_connection（不经过 _pins，避免 pin 对象无 list_all_pins）
  - bone_name / actor_id 用字符串成员变量默认值 → get 节点 → 连接 Array_Add.NewItem
    （K2Node_CallArrayFunction 的 newitem pin 用 set_pin_value 会失败）

用法（Unreal Editor Python Console）：
    py "D:/projects/FustalMOT_UEDataset/Content/FutsalMOT/code/ue/build_bp_recorder_c2.py"
"""

import sys
from pathlib import Path

LOG = []
LOG_PATH = Path(r"D:\projects\FustalMOT_UEDataset\Content\FutsalMOT\code\.futsalmot\build_bp_recorder_c2.log")
ASSET_PATH = "/Game/FutsalMOT/Blueprints/BP_PoseRecorder"
SG_PATH = "/Game/FutsalMOT/Blueprints/SG_PoseCapture.SG_PoseCapture_C"
PLAYER_CLASS = "/Game/ThirdPerson/Blueprints/BP_ThirdPersonCharacter.BP_ThirdPersonCharacter_C"
SKELETAL_COMP = "/Script/Engine.SkeletalMeshComponent"
SLOT_NAME = "PoseCapture"
ACTOR_ID = "Player_L0"
BONES = ["hand_l", "lowerarm_l", "thigh_l", "calf_l", "foot_l"]


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
    _log("======== 构建 BP_PoseRecorder（阶段 C2：5 骨骼）========")

    # 先销毁关卡中的旧 BP_PoseRecorder 实例 + 保存关卡，避免 delete_asset 触发 World Partition 清理
    try:
        ed_sub = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
        for a in list(ed_sub.get_all_level_actors()):
            try:
                if a.get_actor_label() == "BP_PoseRecorder":
                    ed_sub.destroy_actor(a)
                    _log("  销毁旧 BP_PoseRecorder 实例")
            except Exception:
                pass
        try:
            lvl = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
            lvl.save_current_level()
            _log("  关卡已保存（World Partition 一致性）")
        except Exception:
            pass
    except Exception as e:
        _log(f"  销毁实例/保存关卡 ERR: {type(e).__name__}")

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
    str_type = bl.get_basic_type_by_name("string")
    arr_str = bl.get_array_type(str_type)
    savegame_ref = bl.get_object_reference_type(unreal.SaveGame)
    for name, pt, label in [
        ("CaptureIndices", arr_int, "int[]"), ("ShotFrames", arr_int, "int[]"),
        ("GameTimes", arr_real, "float[]"), ("ActorIds", arr_str, "string[]"),
        ("BoneNames", arr_str, "string[]"), ("WorldLocations", arr_vec, "Vector[]"),
        ("SaveGameRef", savegame_ref, "SaveGame"),
    ]:
        _log(f"  add {name}: {bl.add_member_variable(bp, name, pt)}")

    # ---- CaptureOutputFrame 函数 ----
    fn_editor = unreal.BlueprintGraphEditor.create_and_edit_function_graph(bp, "CaptureOutputFrame")
    _log(f"[2] CaptureOutputFrame 函数 -> {fn_editor}")
    root_param = fn_editor.add_graph_input_parameter("RootFrame", int_type)
    shot_param = fn_editor.add_graph_input_parameter("ShotFrame", int_type)
    _log(f"  RootFrame param -> {root_param}  ShotFrame param -> {shot_param}")

    def _call(editor, path, label):
        try:
            n = editor.add_call_function_node(path)
            _log(f"[3] {label} -> {n}")
            return n
        except Exception as e:
            _log(f"[3] {label} ERR: {type(e).__name__} {e}")
            return None

    fn = {}
    fn["actor"] = _call(fn_editor, "/Script/Engine.GameplayStatics.GetActorOfClass", "Fn_GetActorOfClass")
    fn["comp"] = _call(fn_editor, "/Script/Engine.Actor.GetComponentByClass", "Fn_GetComponentByClass")
    fn["time"] = _call(fn_editor, "/Script/Engine.KismetSystemLibrary.GetGameTimeInSeconds", "Fn_GetGameTime")
    fn["getcap"] = fn_editor.add_get_member_variable_node("CaptureIndices", "")
    fn["getshot"] = fn_editor.add_get_member_variable_node("ShotFrames", "")
    fn["gettime"] = fn_editor.add_get_member_variable_node("GameTimes", "")
    fn["getactor"] = fn_editor.add_get_member_variable_node("ActorIds", "")
    fn["getbone"] = fn_editor.add_get_member_variable_node("BoneNames", "")
    fn["getworld"] = fn_editor.add_get_member_variable_node("WorldLocations", "")
    # MakeLiteralString 常量节点：1 个 actor id + 5 个骨骼名
    fn["litactor"] = _call(fn_editor, "/Script/Engine.KismetSystemLibrary.MakeLiteralString", "Fn_LitActor")
    fn["litbone"] = []
    for i, bone in enumerate(BONES):
        fn["litbone"].append(_call(fn_editor, "/Script/Engine.KismetSystemLibrary.MakeLiteralString", f"Fn_LitBone[{bone}]"))

    # 5 骨骼：每个 = GetSocketLocation + 6 个 Array_Add
    fn["socket"] = []
    fn["addcap"] = []
    fn["addshot"] = []
    fn["addtime"] = []
    fn["addactor"] = []
    fn["addbone"] = []
    fn["addloc"] = []
    for i, bone in enumerate(BONES):
        fn["socket"].append(_call(fn_editor, "/Script/Engine.SceneComponent.GetSocketLocation", f"Fn_Socket[{bone}]"))
        fn["addcap"].append(_call(fn_editor, "/Script/Engine.KismetArrayLibrary.Array_Add", f"Fn_AddCap[{bone}]"))
        fn["addshot"].append(_call(fn_editor, "/Script/Engine.KismetArrayLibrary.Array_Add", f"Fn_AddShot[{bone}]"))
        fn["addtime"].append(_call(fn_editor, "/Script/Engine.KismetArrayLibrary.Array_Add", f"Fn_AddTime[{bone}]"))
        fn["addactor"].append(_call(fn_editor, "/Script/Engine.KismetArrayLibrary.Array_Add", f"Fn_AddActor[{bone}]"))
        fn["addbone"].append(_call(fn_editor, "/Script/Engine.KismetArrayLibrary.Array_Add", f"Fn_AddBone[{bone}]"))
        fn["addloc"].append(_call(fn_editor, "/Script/Engine.KismetArrayLibrary.Array_Add", f"Fn_AddLoc[{bone}]"))

    fn_pins = {}
    for k, v in fn.items():
        if isinstance(v, list):
            fn_pins[k] = [_pins(x) if x is not None else None for x in v]
        elif v is not None:
            fn_pins[k] = _pins(v)
    for k, v in fn_pins.items():
        if isinstance(v, list):
            _log(f"  fn {k}: [{len(v)} 组]")
            for idx, p in enumerate(v):
                _log(f"    [{idx}] {sorted(p) if p else None}")
        else:
            _log(f"  fn {k}: {sorted(v) if v else None}")

    # 常量
    _set_value(fn_pins["actor"], "actorclass", PLAYER_CLASS, "Fn_GetActorOfClass")
    _set_value(fn_pins["comp"], "componentclass", SKELETAL_COMP, "Fn_GetComponentByClass")
    for i, bone in enumerate(BONES):
        _set_value(fn_pins["socket"][i], "insocketname", bone, f"Fn_Socket[{bone}]")
        _set_value(fn_pins["litbone"][i], "value", bone, f"Fn_LitBone[{bone}]")
    _set_value(fn_pins["litactor"], "value", ACTOR_ID, "Fn_LitActor")

    # 数据连接：共享源
    _connect(fn_pins["actor"], "returnvalue", fn_pins["comp"], "self", "Actor→Comp")
    for i, bone in enumerate(BONES):
        _connect(fn_pins["comp"], "returnvalue", fn_pins["socket"][i], "self", f"Comp→Socket[{bone}]")
        _connect(fn_pins["getcap"], "captureindices", fn_pins["addcap"][i], "targetarray", f"Cap→AddCap[{bone}]")
        _connect(fn_pins["getshot"], "shotframes", fn_pins["addshot"][i], "targetarray", f"Shot→AddShot[{bone}]")
        _connect(fn_pins["gettime"], "gametimes", fn_pins["addtime"][i], "targetarray", f"Time→AddTime[{bone}]")
        _connect(fn_pins["getactor"], "actorids", fn_pins["addactor"][i], "targetarray", f"Actor→AddActor[{bone}]")
        _connect(fn_pins["getbone"], "bonenames", fn_pins["addbone"][i], "targetarray", f"Bone→AddBone[{bone}]")
        _connect(fn_pins["getworld"], "worldlocations", fn_pins["addloc"][i], "targetarray", f"World→AddLoc[{bone}]")
        _connect(fn_pins["socket"][i], "returnvalue", fn_pins["addloc"][i], "newitem", f"Loc→AddLoc[{bone}].NewItem")
        _connect(fn_pins["time"], "returnvalue", fn_pins["addtime"][i], "newitem", f"Time→AddTime[{bone}].NewItem")
        # actor_id / bone_name 来自 MakeLiteralString 常量节点
        _connect(fn_pins["litactor"], "returnvalue", fn_pins["addactor"][i], "newitem", f"LitActor→AddActor[{bone}].NewItem")
        _connect(fn_pins["litbone"][i], "returnvalue", fn_pins["addbone"][i], "newitem", f"LitBone→AddBone[{bone}].NewItem")
        try:
            root_param.try_create_connection(fn_pins["addcap"][i]["newitem"])
            _log(f"  [conn] RootFrame→AddCap[{bone}].NewItem: True")
        except Exception as e:
            _log(f"  [conn] RootFrame→AddCap[{bone}] ERR: {type(e).__name__}")
        try:
            shot_param.try_create_connection(fn_pins["addshot"][i]["newitem"])
            _log(f"  [conn] ShotFrame→AddShot[{bone}].NewItem: True")
        except Exception as e:
            _log(f"  [conn] ShotFrame→AddShot[{bone}] ERR: {type(e).__name__}")

    # 函数入口 exec → Actor（直接连 pin 对象，不经 _pins）
    entry_pin = fn_editor.find_graph_entry_pin()
    try:
        ok = entry_pin.try_create_connection(fn_pins["actor"]["execute"])
        _log(f"  [conn] Entry→Actor(exec): {ok}")
    except Exception as e:
        _log(f"  [conn] Entry→Actor(exec) ERR: {type(e).__name__} {e}")

    # exec 链：Actor→(AddCap0→AddShot0→AddTime0→AddActor0→AddBone0→AddLoc0)→(AddCap1→...)
    prev = fn_pins["actor"]
    prev_pin = "then"
    for i in range(len(BONES)):
        group = [fn_pins["addcap"][i], fn_pins["addshot"][i], fn_pins["addtime"][i],
                 fn_pins["addactor"][i], fn_pins["addbone"][i], fn_pins["addloc"][i]]
        for j, node_pins in enumerate(group):
            _connect(prev, prev_pin, node_pins, "execute", f"exec→{BONES[i]}.{j}")
            prev, prev_pin = node_pins, "then"

    # ---- EndPlay SaveGame ----
    event_graph = unreal.BlueprintEditorLibrary.find_event_graph(bp)
    ed_editor = unreal.BlueprintGraphEditor.get_graph_editor(event_graph)
    endplay_node = unreal.BlueprintEditorLibrary.add_event_override(bp, "ReceiveEndPlay", unreal.IntPoint(0, 0))
    _log(f"[4] EndPlay -> {endplay_node}")

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
    nodes["setcap"] = ed_editor.add_set_member_variable_node("capture_indices", SG_PATH)
    nodes["setshot"] = ed_editor.add_set_member_variable_node("shot_frames", SG_PATH)
    nodes["settime"] = ed_editor.add_set_member_variable_node("game_times", SG_PATH)
    nodes["setactor"] = ed_editor.add_set_member_variable_node("actor_ids", SG_PATH)
    nodes["setbone"] = ed_editor.add_set_member_variable_node("bone_names", SG_PATH)
    nodes["setworld"] = ed_editor.add_set_member_variable_node("world_locations", SG_PATH)
    nodes["arrlen"] = _call(ed_editor, "/Script/Engine.KismetArrayLibrary.Array_Length", "ArrayLen")
    nodes["settotal"] = ed_editor.add_set_member_variable_node("total_samples", SG_PATH)
    nodes["save"] = _call(ed_editor, "/Script/Engine.GameplayStatics.SaveGameToSlot", "SaveToSlot")

    if None in nodes.values():
        _log("  ERROR: EndPlay 节点创建失败")
        _flush()
        return

    pins = {k: _pins(v) for k, v in nodes.items()}
    pins["endplay"] = _pins(endplay_node)
    for k, p in pins.items():
        _log(f"  {k}: {sorted(p)}")

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
    _connect(pins["getcap"], "captureindices", pins["arrlen"], "targetarray", "Cap→Len")
    _connect(pins["arrlen"], "returnvalue", pins["settotal"], "total_samples", "Len→SG.total")
    for k in ("setcap", "setshot", "settime", "setactor", "setbone", "setworld", "settotal"):
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
    _connect(pins["setworld"], "then", pins["settotal"], "execute", "SetWorld→SetTotal(exec)")
    _connect(pins["settotal"], "then", pins["save"], "execute", "SetTotal→Save(exec)")

    # 编译保存
    try:
        ok = unreal.BlueprintEditorLibrary.compile_blueprint(bp)
        _log(f"[5] compile -> {ok}")
        unreal.EditorAssetLibrary.save_asset(ASSET_PATH, only_if_is_dirty=True)
        _log("[5] save OK")
    except Exception as e:
        _log(f"[5] compile/save ERR: {type(e).__name__} {e}")

    # 放进关卡
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
            _log("[6] 新 BP 已放入关卡")
        try:
            lvl = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
            lvl.save_current_level()
            _log("[6] 关卡已保存")
        except Exception as e:
            _log(f"[6] save level ERR: {type(e).__name__}")
    except Exception as e:
        _log(f"[6] spawn ERR: {type(e).__name__} {e}")

    _flush()


if __name__ == "__main__":
    main()
    print("\n脚本已执行，结果写入 build_bp_recorder_c2.log")