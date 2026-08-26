"""构建 BP_PoseRecorder 阶段 C3b：双 actor（Player_L0 + Player_R0）COCO17 source。

在 C3（13 骨采样）基础上扩展为两个 actor，每帧 26 条记录：
  - 构建时给 Player_L0 打 Tag "PoseL0"，Player_R0 打 "PoseR0"
  - CaptureOutputFrame(Root, Shot) 内联两个 actor 各 13 骨采样（不依赖自定义函数调用）

7 平行数组（每帧 26 条）：CaptureIndices/ShotFrames/GameTimes/ActorIds/BoneNames/
WorldLocations/WorldRotations。总记录数 = 90 帧 × 2 actor × 13 骨 = 2340。

只使用 add_call_function_node（已验证可靠），不创建自定义函数调用节点。

用法（Unreal Editor Python Console）：
    py "D:/projects/FustalMOT_UEDataset/Content/FutsalMOT/code/ue/build_bp_recorder_c3.py"
"""

import sys
from pathlib import Path

LOG = []
LOG_PATH = Path(r"D:\projects\FustalMOT_UEDataset\Content\FutsalMOT\code\.futsalmot\build_bp_recorder_c3.log")
ASSET_PATH = "/Game/FutsalMOT/Blueprints/BP_PoseRecorder"
SG_PATH = "/Game/FutsalMOT/Blueprints/SG_PoseCapture.SG_PoseCapture_C"
SKELETAL_COMP = "/Script/Engine.SkeletalMeshComponent"
SLOT_NAME = "PoseCapture"
ACTORS = [
    {"tag": "PoseL0", "actor_id": "Player_L0"},
    {"tag": "PoseR0", "actor_id": "Player_R0"},
]
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
GET_ALL_TAG = "/Script/Engine.GameplayStatics.GetAllActorsWithTag"
ARRAY_GET = "/Script/Engine.KismetArrayLibrary.Array_Get"
GET_COMP = "/Script/Engine.Actor.GetComponentByClass"


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
    _log("======== 构建 BP_PoseRecorder（C3b：双 actor 内联 13 骨）========")

    # 给 Player_L0/R0 打 Tag + 销毁旧 BP_PoseRecorder + 保存关卡
    try:
        ed_sub = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
        for a in list(ed_sub.get_all_level_actors()):
            try:
                label = a.get_actor_label()
                if label == "BP_PoseRecorder":
                    ed_sub.destroy_actor(a)
                    _log("  销毁旧 BP_PoseRecorder 实例")
                for tgt in ACTORS:
                    if label == tgt["actor_id"]:
                        tags = list(a.get_editor_property("tags") or [])
                        if tgt["tag"] not in tags:
                            tags.append(tgt["tag"])
                            a.set_editor_property("tags", tags)
                            _log(f"  {label} 已加 Tag {tgt['tag']}")
                        else:
                            _log(f"  {label} 已有 Tag {tgt['tag']}")
            except Exception as e:
                _log(f"  tag/setup ERR: {type(e).__name__}")
        try:
            lvl = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
            lvl.save_current_level()
            _log("  关卡已保存")
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
    ]:
        _log(f"  add {name}: {bl.add_member_variable(bp, name, pt)}")

    # ---- CaptureOutputFrame 函数：内联两个 actor 各 13 骨 ----
    fn_editor = unreal.BlueprintGraphEditor.create_and_edit_function_graph(bp, "CaptureOutputFrame")
    _log(f"[2] CaptureOutputFrame 函数 -> {fn_editor}")
    root_param = fn_editor.add_graph_input_parameter("RootFrame", int_type)
    shot_param = fn_editor.add_graph_input_parameter("ShotFrame", int_type)
    _log(f"  RootFrame/ShotFrame params")

    # 共享：GetGameTime + 7 个 get 数组节点 + GetComponentByClass(每个 actor 各自)
    fn = {}
    fn["time"] = _call(fn_editor, "/Script/Engine.KismetSystemLibrary.GetGameTimeInSeconds", "Fn_GetGameTime")
    fn["getcap"] = fn_editor.add_get_member_variable_node("CaptureIndices", "")
    fn["getshot"] = fn_editor.add_get_member_variable_node("ShotFrames", "")
    fn["gettime"] = fn_editor.add_get_member_variable_node("GameTimes", "")
    fn["getactor"] = fn_editor.add_get_member_variable_node("ActorIds", "")
    fn["getbone"] = fn_editor.add_get_member_variable_node("BoneNames", "")
    fn["getworld"] = fn_editor.add_get_member_variable_node("WorldLocations", "")
    fn["getrot"] = fn_editor.add_get_member_variable_node("WorldRotations", "")
    fn["litactor"] = []
    fn["litbone"] = []
    fn["gettag"] = []
    fn["get"] = []
    fn["comp"] = []
    fn["socket"] = []
    fn["rot"] = []
    fn["addcap"] = []
    fn["addshot"] = []
    fn["addtime"] = []
    fn["addactor"] = []
    fn["addbone"] = []
    fn["addloc"] = []
    fn["addrot"] = []
    for tgt in ACTORS:
        fn["litactor"].append(_call(fn_editor, MAKE_LITERAL, f"Fn_LitActor[{tgt['actor_id']}]"))
        fn["gettag"].append(_call(fn_editor, GET_ALL_TAG, f"Fn_GetAllTag[{tgt['tag']}]"))
        fn["get"].append(_call(fn_editor, ARRAY_GET, f"Fn_ArrayGet[{tgt['tag']}]"))
        fn["comp"].append(_call(fn_editor, GET_COMP, f"Fn_Comp[{tgt['actor_id']}]"))
        for bone in BONES:
            fn["litbone"].append(_call(fn_editor, MAKE_LITERAL, f"Fn_LitBone[{tgt['actor_id']}|{bone}]"))
            fn["socket"].append(_call(fn_editor, GET_SOCKET_LOC, f"Fn_Socket[{tgt['actor_id']}|{bone}]"))
            fn["rot"].append(_call(fn_editor, GET_SOCKET_ROT, f"Fn_Rot[{tgt['actor_id']}|{bone}]"))
            fn["addcap"].append(_call(fn_editor, ARRAY_ADD, f"Fn_AddCap[{tgt['actor_id']}|{bone}]"))
            fn["addshot"].append(_call(fn_editor, ARRAY_ADD, f"Fn_AddShot[{tgt['actor_id']}|{bone}]"))
            fn["addtime"].append(_call(fn_editor, ARRAY_ADD, f"Fn_AddTime[{tgt['actor_id']}|{bone}]"))
            fn["addactor"].append(_call(fn_editor, ARRAY_ADD, f"Fn_AddActor[{tgt['actor_id']}|{bone}]"))
            fn["addbone"].append(_call(fn_editor, ARRAY_ADD, f"Fn_AddBone[{tgt['actor_id']}|{bone}]"))
            fn["addloc"].append(_call(fn_editor, ARRAY_ADD, f"Fn_AddLoc[{tgt['actor_id']}|{bone}]"))
            fn["addrot"].append(_call(fn_editor, ARRAY_ADD, f"Fn_AddRot[{tgt['actor_id']}|{bone}]"))

    fn_pins = {}
    for k, v in fn.items():
        if isinstance(v, list):
            fn_pins[k] = [_pins(x) if x is not None else None for x in v]
        elif v is not None:
            fn_pins[k] = _pins(v)
    for k, v in fn_pins.items():
        if isinstance(v, list):
            _log(f"  fn {k}: [{len(v)} 组]")
        else:
            _log(f"  fn {k}: {sorted(v) if v else None}")

    # 常量
    for i, tgt in enumerate(ACTORS):
        _set_value(fn_pins["gettag"][i], "tag", tgt["tag"], f"GetAllTag[{tgt['tag']}]")
        _set_value(fn_pins["get"][i], "index", "0", f"ArrayGet[{tgt['tag']}]")
        _set_value(fn_pins["comp"][i], "componentclass", SKELETAL_COMP, f"Comp[{tgt['actor_id']}]")
        _set_value(fn_pins["litactor"][i], "value", tgt["actor_id"], f"LitActor[{tgt['actor_id']}]")
        for j, bone in enumerate(BONES):
            idx = i * len(BONES) + j
            _set_value(fn_pins["socket"][idx], "insocketname", bone, f"Socket[{tgt['actor_id']}|{bone}]")
            _set_value(fn_pins["rot"][idx], "insocketname", bone, f"Rot[{tgt['actor_id']}|{bone}]")
            _set_value(fn_pins["litbone"][idx], "value", bone, f"LitBone[{tgt['actor_id']}|{bone}]")

    # 数据连接
    for i, tgt in enumerate(ACTORS):
        _connect(fn_pins["gettag"][i], "outactors", fn_pins["get"][i], "targetarray", f"Tag→Get[{tgt['tag']}]")
        _connect(fn_pins["get"][i], "item", fn_pins["comp"][i], "self", f"Get→Comp[{tgt['actor_id']}].Self")
        for j, bone in enumerate(BONES):
            idx = i * len(BONES) + j
            _connect(fn_pins["comp"][i], "returnvalue", fn_pins["socket"][idx], "self", f"Comp→Socket[{tgt['actor_id']}|{bone}]")
            _connect(fn_pins["comp"][i], "returnvalue", fn_pins["rot"][idx], "self", f"Comp→Rot[{tgt['actor_id']}|{bone}]")
            _connect(fn_pins["getcap"], "captureindices", fn_pins["addcap"][idx], "targetarray", f"Cap→AddCap[{tgt['actor_id']}|{bone}]")
            _connect(fn_pins["getshot"], "shotframes", fn_pins["addshot"][idx], "targetarray", f"Shot→AddShot[{tgt['actor_id']}|{bone}]")
            _connect(fn_pins["gettime"], "gametimes", fn_pins["addtime"][idx], "targetarray", f"Time→AddTime[{tgt['actor_id']}|{bone}]")
            _connect(fn_pins["getactor"], "actorids", fn_pins["addactor"][idx], "targetarray", f"Actor→AddActor[{tgt['actor_id']}|{bone}]")
            _connect(fn_pins["getbone"], "bonenames", fn_pins["addbone"][idx], "targetarray", f"Bone→AddBone[{tgt['actor_id']}|{bone}]")
            _connect(fn_pins["getworld"], "worldlocations", fn_pins["addloc"][idx], "targetarray", f"World→AddLoc[{tgt['actor_id']}|{bone}]")
            _connect(fn_pins["getrot"], "worldrotations", fn_pins["addrot"][idx], "targetarray", f"Rot→AddRot[{tgt['actor_id']}|{bone}]")
            _connect(fn_pins["socket"][idx], "returnvalue", fn_pins["addloc"][idx], "newitem", f"Loc→AddLoc[{tgt['actor_id']}|{bone}].NewItem")
            _connect(fn_pins["rot"][idx], "returnvalue", fn_pins["addrot"][idx], "newitem", f"Rot→AddRot[{tgt['actor_id']}|{bone}].NewItem")
            _connect(fn_pins["time"], "returnvalue", fn_pins["addtime"][idx], "newitem", f"Time→AddTime[{tgt['actor_id']}|{bone}].NewItem")
            _connect(fn_pins["litactor"][i], "returnvalue", fn_pins["addactor"][idx], "newitem", f"LitActor→AddActor[{tgt['actor_id']}|{bone}].NewItem")
            _connect(fn_pins["litbone"][idx], "returnvalue", fn_pins["addbone"][idx], "newitem", f"LitBone→AddBone[{tgt['actor_id']}|{bone}].NewItem")
            try:
                root_param.try_create_connection(fn_pins["addcap"][idx]["newitem"])
                _log(f"  [conn] RootFrame→AddCap[{tgt['actor_id']}|{bone}].NewItem: True")
            except Exception as e:
                _log(f"  [conn] RootFrame→AddCap[{tgt['actor_id']}|{bone}] ERR: {type(e).__name__}")
            try:
                shot_param.try_create_connection(fn_pins["addshot"][idx]["newitem"])
                _log(f"  [conn] ShotFrame→AddShot[{tgt['actor_id']}|{bone}].NewItem: True")
            except Exception as e:
                _log(f"  [conn] ShotFrame→AddShot[{tgt['actor_id']}|{bone}] ERR: {type(e).__name__}")

    # exec 链：Entry → GetTagL0 → (AddCapL0..AddRotL0) → GetTagR0 → (AddCapR0..AddRotR0)
    entry_pin = fn_editor.find_graph_entry_pin()
    try:
        ok = entry_pin.try_create_connection(fn_pins["gettag"][0]["execute"])
        _log(f"  [conn] Entry→GetTagL0(exec): {ok}")
    except Exception as e:
        _log(f"  [conn] Entry→GetTagL0(exec) ERR: {type(e).__name__} {e}")

    prev, prev_pin = fn_pins["gettag"][0], "then"
    for i, tgt in enumerate(ACTORS):
        base = i * len(BONES)
        group = []
        for j in range(len(BONES)):
            idx = base + j
            group.append(fn_pins["addcap"][idx])
            group.append(fn_pins["addshot"][idx])
            group.append(fn_pins["addtime"][idx])
            group.append(fn_pins["addactor"][idx])
            group.append(fn_pins["addbone"][idx])
            group.append(fn_pins["addloc"][idx])
            group.append(fn_pins["addrot"][idx])
        for k, node_pins in enumerate(group):
            _connect(prev, prev_pin, node_pins, "execute", f"exec→{tgt['actor_id']}.{k}")
            prev, prev_pin = node_pins, "then"
        if i + 1 < len(ACTORS):
            _connect(prev, prev_pin, fn_pins["gettag"][i + 1], "execute", f"exec→GetTag{i+1}")
            prev, prev_pin = fn_pins["gettag"][i + 1], "then"

    # ---- EndPlay SaveGame（同 C3）----
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
    nodes["getrot"] = ed_editor.add_get_member_variable_node("WorldRotations", "")
    nodes["setcap"] = ed_editor.add_set_member_variable_node("capture_indices", SG_PATH)
    nodes["setshot"] = ed_editor.add_set_member_variable_node("shot_frames", SG_PATH)
    nodes["settime"] = ed_editor.add_set_member_variable_node("game_times", SG_PATH)
    nodes["setactor"] = ed_editor.add_set_member_variable_node("actor_ids", SG_PATH)
    nodes["setbone"] = ed_editor.add_set_member_variable_node("bone_names", SG_PATH)
    nodes["setworld"] = ed_editor.add_set_member_variable_node("world_locations", SG_PATH)
    nodes["setrot"] = ed_editor.add_set_member_variable_node("world_rotations", SG_PATH)
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
    _connect(pins["getrot"], "worldrotations", pins["setrot"], "world_rotations", "Rot→SG.rot")
    _connect(pins["getcap"], "captureindices", pins["arrlen"], "targetarray", "Cap→Len")
    _connect(pins["arrlen"], "returnvalue", pins["settotal"], "total_samples", "Len→SG.total")
    for k in ("setcap", "setshot", "settime", "setactor", "setbone", "setworld", "setrot", "settotal"):
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
    _connect(pins["setrot"], "then", pins["settotal"], "execute", "SetRot→SetTotal(exec)")
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
    print("\n脚本已执行，结果写入 build_bp_recorder_c3.log")