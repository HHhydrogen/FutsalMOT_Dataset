"""构建 BP_PoseRecorder 阶段 C1：CaptureOutputFrame(Root, Shot) 采样 hand_l + SaveGame。

用法（Unreal Editor Python Console）：
    py "D:/projects/FustalMOT_UEDataset/Content/FutsalMOT/code/ue/build_bp_recorder_c1.py"

结构：
  - 成员变量：CaptureIndices(int[]), ShotFrames(int[]), GameTimes(float[]),
             HandLocations(Vector[]), ActorIds(string[]), BoneNames(string[]),
             SaveGameRef(SaveGame)
  - 函数 CaptureOutputFrame(RootFrame:int, ShotFrame:int)：
      找 Player_L0 → GetSocketLocation(hand_l) → 记录 root/shot/gameTime/hand 到数组
  - EndPlay：CreateSaveGameObject → 填 SG → SaveGameToSlot
"""

import sys
from pathlib import Path

LOG = []
LOG_PATH = Path(r"D:\projects\FustalMOT_UEDataset\Content\FutsalMOT\code\.futsalmot\build_bp_recorder_c1.log")
ASSET_PATH = "/Game/FutsalMOT/Blueprints/BP_PoseRecorder"
SG_PATH = "/Game/FutsalMOT/Blueprints/SG_PoseCapture.SG_PoseCapture_C"
PLAYER_CLASS = "/Game/ThirdPerson/Blueprints/BP_ThirdPersonCharacter.BP_ThirdPersonCharacter_C"
SKELETAL_COMP = "/Script/Engine.SkeletalMeshComponent"
SLOT_NAME = "PoseCapture"


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
    _log("======== 构建 BP_PoseRecorder（阶段 C1）========")

    # 先销毁关卡中的旧 BP_PoseRecorder 实例 + 保存关卡，避免 delete_asset 触发 World Partition 清理
    try:
        ed_sub = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
        for a in list(ed_sub.get_all_level_actors()):
            try:
                if a.get_actor_label() == "BP_PoseRecorder":
                    ed_sub.destroy_actor(a)
                    _log(f"  销毁旧 BP_PoseRecorder 实例")
            except Exception:
                pass
        # 保存关卡（World Partition 一致性）
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

    # 成员变量
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
        ("GameTimes", arr_real, "float[]"), ("HandLocations", arr_vec, "Vector[]"),
        ("ActorIds", arr_str, "string[]"), ("BoneNames", arr_str, "string[]"),
        ("SaveGameRef", savegame_ref, "SaveGame"),
    ]:
        _log(f"  add {name}: {bl.add_member_variable(bp, name, pt)}")

    # EndPlay 事件
    event_graph = unreal.BlueprintEditorLibrary.find_event_graph(bp)
    ed_editor = unreal.BlueprintGraphEditor.get_graph_editor(event_graph)
    endplay_node = unreal.BlueprintEditorLibrary.add_event_override(bp, "ReceiveEndPlay", unreal.IntPoint(0, 0))
    _log(f"[2] EndPlay -> {endplay_node}")

    def _call(editor, path, label):
        try:
            n = editor.add_call_function_node(path)
            _log(f"[3] {label} -> {n}")
            return n
        except Exception as e:
            _log(f"[3] {label} ERR: {type(e).__name__} {e}")
            return None

    # ---- CaptureOutputFrame 函数 ----
    fn_editor = unreal.BlueprintGraphEditor.create_and_edit_function_graph(bp, "CaptureOutputFrame")
    _log(f"[4] CaptureOutputFrame 函数 -> {fn_editor}")
    bl.add_member_variable(bp, "_fn_param_RootFrame", int_type) if False else None
    root_param = fn_editor.add_graph_input_parameter("RootFrame", int_type)
    shot_param = fn_editor.add_graph_input_parameter("ShotFrame", int_type)
    _log(f"  RootFrame param -> {root_param}  ShotFrame param -> {shot_param}")

    fn = {}
    fn["entry"] = fn_editor.find_graph_entry_pin()
    fn["actor"] = _call(fn_editor, "/Script/Engine.GameplayStatics.GetActorOfClass", "Fn_GetActorOfClass")
    fn["comp"] = _call(fn_editor, "/Script/Engine.Actor.GetComponentByClass", "Fn_GetComponentByClass")
    fn["socket"] = _call(fn_editor, "/Script/Engine.SceneComponent.GetSocketLocation", "Fn_GetSocketLocation")
    fn["time"] = _call(fn_editor, "/Script/Engine.KismetSystemLibrary.GetGameTimeInSeconds", "Fn_GetGameTime")
    fn["getcap"] = fn_editor.add_get_member_variable_node("CaptureIndices", "")
    fn["getshot"] = fn_editor.add_get_member_variable_node("ShotFrames", "")
    fn["gettime"] = fn_editor.add_get_member_variable_node("GameTimes", "")
    fn["gethand"] = fn_editor.add_get_member_variable_node("HandLocations", "")
    fn["arrcap"] = _call(fn_editor, "/Script/Engine.KismetArrayLibrary.Array_Add", "Fn_AddCap")
    fn["arrshot"] = _call(fn_editor, "/Script/Engine.KismetArrayLibrary.Array_Add", "Fn_AddShot")
    fn["arrtime"] = _call(fn_editor, "/Script/Engine.KismetArrayLibrary.Array_Add", "Fn_AddTime")
    fn["arrhand"] = _call(fn_editor, "/Script/Engine.KismetArrayLibrary.Array_Add", "Fn_AddHand")

    # 获取函数参数 pin（RootFrame/ShotFrame）
    fn_pins = {k: _pins(v) if not hasattr(v, "get_pin_name") else {str(v.get_pin_name()).lower(): v} for k, v in fn.items() if v is not None}
    for k, p in fn_pins.items():
        _log(f"  fn {k}: {sorted(p)}")

    # 连接函数体
    _set_value(fn_pins["actor"], "actorclass", PLAYER_CLASS, "Fn_GetActorOfClass")
    _set_value(fn_pins["comp"], "componentclass", SKELETAL_COMP, "Fn_GetComponentByClass")
    _set_value(fn_pins["socket"], "insocketname", "hand_l", "Fn_GetSocketLocation")

    _connect(fn_pins["actor"], "returnvalue", fn_pins["comp"], "self", "Fn Actor→Comp")
    _connect(fn_pins["comp"], "returnvalue", fn_pins["socket"], "self", "Fn Comp→Socket")
    _connect(fn_pins["getcap"], "captureindices", fn_pins["arrcap"], "targetarray", "Cap→AddCap")
    _connect(fn_pins["getshot"], "shotframes", fn_pins["arrshot"], "targetarray", "Shot→AddShot")
    _connect(fn_pins["gettime"], "gametimes", fn_pins["arrtime"], "targetarray", "Time→AddTime")
    _connect(fn_pins["gethand"], "handlocations", fn_pins["arrhand"], "targetarray", "Hand→AddHand")
    _connect(fn_pins["socket"], "returnvalue", fn_pins["arrhand"], "newitem", "Loc→AddHand.NewItem")
    _connect(fn_pins["time"], "returnvalue", fn_pins["arrtime"], "newitem", "Time→AddTime.NewItem")
    # 参数 RootFrame/ShotFrame → 数组 NewItem
    try:
        ok = root_param.try_create_connection(fn_pins["arrcap"]["newitem"])
        _log(f"  [conn] RootFrame→AddCap.NewItem: {ok}")
    except Exception as e:
        _log(f"  [conn] RootFrame→AddCap ERR: {type(e).__name__}")
    try:
        ok = shot_param.try_create_connection(fn_pins["arrshot"]["newitem"])
        _log(f"  [conn] ShotFrame→AddShot.NewItem: {ok}")
    except Exception as e:
        _log(f"  [conn] ShotFrame→AddShot ERR: {type(e).__name__}")
    _connect(fn_pins["entry"], "then", fn_pins["actor"], "execute", "Entry→Actor(exec)")
    _connect(fn_pins["actor"], "then", fn_pins["arrcap"], "execute", "Actor→AddCap(exec)")
    _connect(fn_pins["arrcap"], "then", fn_pins["arrshot"], "execute", "AddCap→AddShot(exec)")
    _connect(fn_pins["arrshot"], "then", fn_pins["arrtime"], "execute", "AddShot→AddTime(exec)")
    _connect(fn_pins["arrtime"], "then", fn_pins["arrhand"], "execute", "AddTime→AddHand(exec)")

    # ---- EndPlay SaveGame ----
    nodes = {}
    nodes["create"] = _call(ed_editor, "/Script/Engine.GameplayStatics.CreateSaveGameObject", "Create")
    nodes["setref"] = ed_editor.add_set_member_variable_node("SaveGameRef", "")
    nodes["getref"] = ed_editor.add_get_member_variable_node("SaveGameRef", "")
    nodes["getcap"] = ed_editor.add_get_member_variable_node("CaptureIndices", "")
    nodes["getshot"] = ed_editor.add_get_member_variable_node("ShotFrames", "")
    nodes["gettime"] = ed_editor.add_get_member_variable_node("GameTimes", "")
    nodes["gethand"] = ed_editor.add_get_member_variable_node("HandLocations", "")
    nodes["setcap"] = ed_editor.add_set_member_variable_node("capture_indices", SG_PATH)
    nodes["setshot"] = ed_editor.add_set_member_variable_node("shot_frames", SG_PATH)
    nodes["settime"] = ed_editor.add_set_member_variable_node("game_times", SG_PATH)
    nodes["sethand"] = ed_editor.add_set_member_variable_node("world_locations", SG_PATH)
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
    _connect(pins["gethand"], "handlocations", pins["sethand"], "world_locations", "Hand→SG.world_loc")
    _connect(pins["getcap"], "captureindices", pins["arrlen"], "targetarray", "Cap→Len")
    _connect(pins["arrlen"], "returnvalue", pins["settotal"], "total_samples", "Len→SG.total")
    for k in ("setcap", "setshot", "settime", "sethand", "settotal"):
        _connect(pins["create"], "returnvalue", pins[k], "self", f"Create→{k}.self")
    _connect(pins["getref"], "savegameref", pins["save"], "savegameobject", "Ref→Save")

    _connect(pins["endplay"], "then", pins["create"], "execute", "EndPlay→Create(exec)")
    _connect(pins["create"], "then", pins["setref"], "execute", "Create→SetRef(exec)")
    _connect(pins["setref"], "then", pins["setcap"], "execute", "SetRef→SetCap(exec)")
    _connect(pins["setcap"], "then", pins["setshot"], "execute", "SetCap→SetShot(exec)")
    _connect(pins["setshot"], "then", pins["settime"], "execute", "SetShot→SetTime(exec)")
    _connect(pins["settime"], "then", pins["sethand"], "execute", "SetTime→SetHand(exec)")
    _connect(pins["sethand"], "then", pins["settotal"], "execute", "SetHand→SetTotal(exec)")
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
        # 保存关卡（World Partition 一致性，防止 umap 丢失）
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
    print("\n脚本已执行，结果写入 build_bp_recorder_c1.log")