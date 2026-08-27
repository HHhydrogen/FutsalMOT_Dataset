"""构建 BP_PoseRecorderC4_G0..G4（5 个 BP，每组 2 actor，13 骨采样）。

背景：UE 5.8 Python 对「单 BP 内联 900+ 节点」不稳定（BlueprintEditorLibrary 崩溃）。
因此把 10 actor 拆成 5 个 BP，每组 2 actor（182 节点，同 C3 验证规模），
每 BP 写独立 SaveGame slot：PoseCaptureG0..PoseCaptureG4。P1 合并 5 slot。

组划分：
  G0: L0, L1     G1: L2, L3     G2: L4, R0     G3: R1, R2     G4: R3, R4

每 BP：
  - 成员：7 平行数组 + SaveGameRef + CaptureDurations + Start/EndCaptureTime
  - CaptureOutputFrame(Root, Shot)：内联 2 actor 13 骨 + 计时（每 actor 段末尾 append duration）
  - EndPlay：CreateSaveGameObject → 填 8 数组 → SaveGameToSlot(PoseCaptureG{N})

前置：先跑 tag_players_c4.py（给 Player 打 Tag）。

用法（Unreal Editor Python Console）：
    py "D:/projects/FustalMOT_UEDataset/Content/FutsalMOT/code/ue/build_bp_recorder_c4.py"
"""

import sys
from pathlib import Path

LOG = []
LOG_PATH = Path(r"D:\projects\FustalMOT_UEDataset\Content\FutsalMOT\code\.futsalmot\build_bp_recorder_c4.log")
ASSET_BASE = "/Game/FutsalMOT/Blueprints/Pose/Recorder/BP_PoseRecorderC4_G"
SG_PATH = "/Game/FutsalMOT/Blueprints/Pose/SaveGame/SG_PoseCapture.SG_PoseCapture_C"
SKELETAL_COMP = "/Script/Engine.SkeletalMeshComponent"

# 每组 2 actor（tag, actor_id）
GROUPS = [
    {"idx": 0, "slot": "PoseCaptureG0", "actors": [
        {"tag": "PoseL0", "actor_id": "L0"}, {"tag": "PoseL1", "actor_id": "L1"}]},
    {"idx": 1, "slot": "PoseCaptureG1", "actors": [
        {"tag": "PoseL2", "actor_id": "L2"}, {"tag": "PoseL3", "actor_id": "L3"}]},
    {"idx": 2, "slot": "PoseCaptureG2", "actors": [
        {"tag": "PoseL4", "actor_id": "L4"}, {"tag": "PoseR0", "actor_id": "R0"}]},
    {"idx": 3, "slot": "PoseCaptureG3", "actors": [
        {"tag": "PoseR1", "actor_id": "R1"}, {"tag": "PoseR2", "actor_id": "R2"}]},
    {"idx": 4, "slot": "PoseCaptureG4", "actors": [
        {"tag": "PoseR3", "actor_id": "R3"}, {"tag": "PoseR4", "actor_id": "R4"}]},
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
GET_TIME = "/Script/Engine.KismetSystemLibrary.GetGameTimeInSeconds"
SUB_FLOAT = "/Script/Engine.KismetMathLibrary.Subtract_FloatFloat"


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
        return editor.add_call_function_node(path)
    except Exception as e:
        _log(f"[3] {label} ERR: {type(e).__name__} {e}")
        return None


def _build_group(group):
    import unreal
    idx = group["idx"]
    asset_path = f"{ASSET_BASE}{idx}"
    slot = group["slot"]
    actors = group["actors"]

    _log(f"\n===== 构建 BP_PoseRecorderC4_G{idx}（{len(actors)} actor, slot={slot}）=====")
    if unreal.EditorAssetLibrary.does_asset_exist(asset_path):
        _log(f"  ERROR: {asset_path} 已存在，请手动删除后重跑。")
        return False

    bp = unreal.BlueprintEditorLibrary.create_blueprint_asset_with_parent(asset_path, unreal.Actor)
    _log(f"  create -> {bp}")

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
    bool_type = bl.get_basic_type_by_name("bool")
    savegame_ref = bl.get_object_reference_type(unreal.SaveGame)
    for name, pt, label in [
        ("CaptureIndices", arr_int, "int[]"), ("ShotFrames", arr_int, "int[]"),
        ("GameTimes", arr_real, "float[]"), ("ActorIds", arr_str, "string[]"),
        ("BoneNames", arr_str, "string[]"), ("WorldLocations", arr_vec, "Vector[]"),
        ("WorldRotations", arr_rot, "Rotator[]"), ("SaveGameRef", savegame_ref, "SaveGame"),
        ("CaptureDurations", arr_real, "float[]"), ("StartCaptureTime", real_type, "float"),
        ("EndCaptureTime", real_type, "float"),
        # C5.2 崩溃安全生命周期（v1）：active 标记 + 动态 slot + session_id
        ("CaptureSessionActive", bool_type, "bool"),
        ("SaveSlotName", str_type, "string"),
        ("SessionId", str_type, "string"),
    ]:
        _log(f"  add {name}: {bl.add_member_variable(bp, name, pt)}")

    # ---- CaptureOutputFrame：内联 N actor 13 骨 + 计时 ----
    fn_editor = unreal.BlueprintGraphEditor.create_and_edit_function_graph(bp, "CaptureOutputFrame")
    _log(f"  CaptureOutputFrame 函数 -> {fn_editor}")
    root_param = fn_editor.add_graph_input_parameter("RootFrame", int_type)
    shot_param = fn_editor.add_graph_input_parameter("ShotFrame", int_type)

    fn = {}
    fn["time"] = _call(fn_editor, GET_TIME, "Fn_GetGameTime")
    fn["getcap"] = fn_editor.add_get_member_variable_node("CaptureIndices", "")
    fn["getshot"] = fn_editor.add_get_member_variable_node("ShotFrames", "")
    fn["gettime"] = fn_editor.add_get_member_variable_node("GameTimes", "")
    fn["getactor"] = fn_editor.add_get_member_variable_node("ActorIds", "")
    fn["getbone"] = fn_editor.add_get_member_variable_node("BoneNames", "")
    fn["getworld"] = fn_editor.add_get_member_variable_node("WorldLocations", "")
    fn["getrot"] = fn_editor.add_get_member_variable_node("WorldRotations", "")
    fn["getdurs"] = fn_editor.add_get_member_variable_node("CaptureDurations", "")
    fn["sub"] = _call(fn_editor, SUB_FLOAT, "Fn_SubFloat")

    fn["litactor"] = []
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
    fn["setstart"] = []
    fn["setend"] = []
    fn["adddur"] = []
    for act in actors:
        fn["litactor"].append(_call(fn_editor, MAKE_LITERAL, f"LitActor[{act['actor_id']}]"))
        fn["gettag"].append(_call(fn_editor, GET_ALL_TAG, f"GetAllTag[{act['tag']}]"))
        fn["get"].append(_call(fn_editor, ARRAY_GET, f"ArrayGet[{act['tag']}]"))
        fn["comp"].append(_call(fn_editor, GET_COMP, f"Comp[{act['actor_id']}]"))
        for bone in BONES:
            fn["socket"].append(_call(fn_editor, GET_SOCKET_LOC, f"Socket[{act['actor_id']}|{bone}]"))
            fn["rot"].append(_call(fn_editor, GET_SOCKET_ROT, f"Rot[{act['actor_id']}|{bone}]"))
            fn["addcap"].append(_call(fn_editor, ARRAY_ADD, f"AddCap[{act['actor_id']}|{bone}]"))
            fn["addshot"].append(_call(fn_editor, ARRAY_ADD, f"AddShot[{act['actor_id']}|{bone}]"))
            fn["addtime"].append(_call(fn_editor, ARRAY_ADD, f"AddTime[{act['actor_id']}|{bone}]"))
            fn["addactor"].append(_call(fn_editor, ARRAY_ADD, f"AddActor[{act['actor_id']}|{bone}]"))
            fn["addbone"].append(_call(fn_editor, ARRAY_ADD, f"AddBone[{act['actor_id']}|{bone}]"))
            fn["addloc"].append(_call(fn_editor, ARRAY_ADD, f"AddLoc[{act['actor_id']}|{bone}]"))
            fn["addrot"].append(_call(fn_editor, ARRAY_ADD, f"AddRot[{act['actor_id']}|{bone}]"))
        # 每 actor 计时：SetStart 在该 actor 段前、SetEnd 在段后
        fn["setstart"].append(fn_editor.add_set_member_variable_node("StartCaptureTime", ""))
        fn["setend"].append(fn_editor.add_set_member_variable_node("EndCaptureTime", ""))
        fn["adddur"].append(_call(fn_editor, ARRAY_ADD, f"AddDur[{act['actor_id']}]"))

    fn_pins = {}
    for k, v in fn.items():
        if isinstance(v, list):
            fn_pins[k] = [_pins(x) if x is not None else None for x in v]
        elif v is not None:
            fn_pins[k] = _pins(v)

    # 常量
    for i, act in enumerate(actors):
        _set_value(fn_pins["gettag"][i], "tag", act["tag"], f"GetAllTag[{act['tag']}]")
        _set_value(fn_pins["get"][i], "index", "0", f"ArrayGet[{act['tag']}]")
        _set_value(fn_pins["comp"][i], "componentclass", SKELETAL_COMP, f"Comp[{act['actor_id']}]")
        _set_value(fn_pins["litactor"][i], "value", act["actor_id"], f"LitActor[{act['actor_id']}]")
        for j, bone in enumerate(BONES):
            idx = i * len(BONES) + j
            _set_value(fn_pins["socket"][idx], "insocketname", bone, f"Socket[{act['actor_id']}|{bone}]")
            _set_value(fn_pins["rot"][idx], "insocketname", bone, f"Rot[{act['actor_id']}|{bone}]")

    # 数据连接
    for i, act in enumerate(actors):
        base = i * len(BONES)
        _connect(fn_pins["gettag"][i], "outactors", fn_pins["get"][i], "targetarray", f"Tag→Get[{act['tag']}]")
        _connect(fn_pins["get"][i], "item", fn_pins["comp"][i], "self", f"Get→Comp[{act['actor_id']}]")
        for j, bone in enumerate(BONES):
            idx = base + j
            _connect(fn_pins["comp"][i], "returnvalue", fn_pins["socket"][idx], "self", f"Comp→Socket")
            _connect(fn_pins["comp"][i], "returnvalue", fn_pins["rot"][idx], "self", f"Comp→Rot")
            _connect(fn_pins["getcap"], "captureindices", fn_pins["addcap"][idx], "targetarray", "Cap→AddCap")
            _connect(fn_pins["getshot"], "shotframes", fn_pins["addshot"][idx], "targetarray", "Shot→AddShot")
            _connect(fn_pins["gettime"], "gametimes", fn_pins["addtime"][idx], "targetarray", "Time→AddTime")
            _connect(fn_pins["getactor"], "actorids", fn_pins["addactor"][idx], "targetarray", "Actor→AddActor")
            _connect(fn_pins["getbone"], "bonenames", fn_pins["addbone"][idx], "targetarray", "Bone→AddBone")
            _connect(fn_pins["getworld"], "worldlocations", fn_pins["addloc"][idx], "targetarray", "World→AddLoc")
            _connect(fn_pins["getrot"], "worldrotations", fn_pins["addrot"][idx], "targetarray", "Rot→AddRot")
            _connect(fn_pins["socket"][idx], "returnvalue", fn_pins["addloc"][idx], "newitem", "Loc→AddLoc")
            _connect(fn_pins["rot"][idx], "returnvalue", fn_pins["addrot"][idx], "newitem", "Rot→AddRot")
            _connect(fn_pins["time"], "returnvalue", fn_pins["addtime"][idx], "newitem", "Time→AddTime")
            _connect(fn_pins["litactor"][i], "returnvalue", fn_pins["addactor"][idx], "newitem", "LitActor→AddActor")
            # bone 名：用 MakeLiteralString（骨骼名）
            litbone = _call(fn_editor, MAKE_LITERAL, f"LitBone[{act['actor_id']}|{bone}]")
            _set_value(_pins(litbone), "value", bone, f"LitBone[{act['actor_id']}|{bone}]")
            _connect(_pins(litbone), "returnvalue", fn_pins["addbone"][idx], "newitem", "LitBone→AddBone")
            try:
                root_param.try_create_connection(fn_pins["addcap"][idx]["newitem"])
            except Exception:
                pass
            try:
                shot_param.try_create_connection(fn_pins["addshot"][idx]["newitem"])
            except Exception:
                pass
        # 计时数据：SetStart/SetEnd/Sub/AddDur
        _connect(fn_pins["time"], "returnvalue", fn_pins["setstart"][i], "startcapturetime", f"Time→Start[{act['actor_id']}]")
        _connect(fn_pins["time"], "returnvalue", fn_pins["setend"][i], "endcapturetime", f"Time→End[{act['actor_id']}]")
        _connect(fn_pins["getdurs"], "capturedurations", fn_pins["adddur"][i], "targetarray", f"Durs→AddDur[{act['actor_id']}]")
        _connect(fn_pins["sub"], "returnvalue", fn_pins["adddur"][i], "newitem", f"Sub→AddDur[{act['actor_id']}].NewItem")

    _log("  常量/数据连接完成")

    # exec 链：Entry→SetActive(C5)→SetStart0→(L0 段)→SetEnd0→Sub→AddDur0→SetStart1→(L1 段)→SetEnd1→Sub→AddDur1
    # Sub 是纯函数（无 exec），exec 在 AddDur 之间
    setact_c5 = fn_editor.add_set_member_variable_node("CaptureSessionActive", "")
    _set_value(_pins(setact_c5), "capturesessionactive", "True", "C5_SetActive")
    entry_pin = fn_editor.find_graph_entry_pin()
    prev, prev_pin = None, None
    try:
        entry_pin.try_create_connection(_pins(setact_c5)["execute"])
        _connect(_pins(setact_c5), "then", fn_pins["setstart"][0], "execute", "SetActive→SetStart0")
        _log("  [conn] Entry→SetActive(C5)→SetStart0: True")
        prev, prev_pin = fn_pins["setstart"][0], "then"
    except Exception as e:
        _log(f"  [conn] Entry→SetActive(C5) ERR: {type(e).__name__}")

    for i, act in enumerate(actors):
        # GetTag→Get→Comp（数据已连）→ exec：prev → gettag[i] → group[0]
        # Get 是纯函数（Array_Get 无 exec/then），exec 从 gettag.then 直接到 AddCap0
        _connect(prev, prev_pin, fn_pins["gettag"][i], "execute", f"exec→GetTag[{act['tag']}]")
        base = i * len(BONES)
        group = []
        for j in range(len(BONES)):
            idx = base + j
            group += [fn_pins["addcap"][idx], fn_pins["addshot"][idx], fn_pins["addtime"][idx],
                      fn_pins["addactor"][idx], fn_pins["addbone"][idx], fn_pins["addloc"][idx],
                      fn_pins["addrot"][idx]]
        _connect(fn_pins["gettag"][i], "then", group[0], "execute", f"exec→{act['actor_id']}.0")
        prev, prev_pin = group[0], "then"
        for k in range(1, len(group)):
            _connect(prev, prev_pin, group[k], "execute", f"exec→{act['actor_id']}.{k}")
            prev, prev_pin = group[k], "then"
        # 段末：AddRot.then → SetEnd[i] → AddDur[i]
        _connect(prev, prev_pin, fn_pins["setend"][i], "execute", f"exec→SetEnd[{act['actor_id']}]")
        _connect(fn_pins["setend"][i], "then", fn_pins["adddur"][i], "execute", f"exec→AddDur[{act['actor_id']}]")
        if i + 1 < len(actors):
            _connect(fn_pins["adddur"][i], "then", fn_pins["setstart"][i + 1], "execute", f"exec→SetStart{i+1}")
            prev, prev_pin = fn_pins["setstart"][i + 1], "then"

    # ---- EndPlay SaveGame ----
    event_graph = unreal.BlueprintEditorLibrary.find_event_graph(bp)
    ed_editor = unreal.BlueprintGraphEditor.get_graph_editor(event_graph)
    endplay_node = unreal.BlueprintEditorLibrary.add_event_override(bp, "ReceiveEndPlay", unreal.IntPoint(0, 0))
    _log(f"  EndPlay -> {endplay_node}")

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
    # C5.2 崩溃安全生命周期：guard + 动态 slot + capture_complete/session_id
    nodes["getact"] = ed_editor.add_get_member_variable_node("CaptureSessionActive", "")
    nodes["guard"] = ed_editor.add_branch_node()
    nodes["getslot"] = ed_editor.add_get_member_variable_node("SaveSlotName", "")
    nodes["getsess"] = ed_editor.add_get_member_variable_node("SessionId", "")
    nodes["setcomplete"] = ed_editor.add_set_member_variable_node("capture_complete", SG_PATH)
    nodes["setsess"] = ed_editor.add_set_member_variable_node("session_id", SG_PATH)

    if None in nodes.values():
        _log("  ERROR: EndPlay 节点创建失败")
        return False

    pins = {k: _pins(v) for k, v in nodes.items()}
    pins["endplay"] = _pins(endplay_node)

    _set_value(pins["create"], "savegameclass", SG_PATH, "Create")
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
    for k in ("setcap", "setshot", "settime", "setactor", "setbone", "setworld", "setrot", "setdurs", "settotal",
              "setcomplete", "setsess"):
        _connect(pins["create"], "returnvalue", pins[k], "self", f"Create→{k}.self")
    _connect(pins["getref"], "savegameref", pins["save"], "savegameobject", "Ref→Save")
    # C5.2：动态 slot + capture_complete(=active 粗标记) + session_id
    _connect(pins["getslot"], "saveslotname", pins["save"], "slotname", "SaveSlotName→Save.slot")
    _connect(pins["getact"], "capturesessionactive", pins["setcomplete"], "capture_complete", "Active→SG.complete")
    _connect(pins["getsess"], "sessionid", pins["setsess"], "session_id", "SessId→SG.sess")

    # exec 链：EndPlay→Guard→(True)→Create→...→SetTotal→SetComplete→SetSess→Save；False→NO-OP
    _connect(pins["endplay"], "then", pins["guard"], "execute", "EndPlay→Guard")
    _connect(pins["guard"], "condition", pins["getact"], "capturesessionactive", "Active→Guard")
    _connect(pins["guard"], "then", pins["create"], "execute", "GuardT→Create")
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
    _connect(pins["settotal"], "then", pins["setcomplete"], "execute", "SetTotal→SetComplete(exec)")
    _connect(pins["setcomplete"], "then", pins["setsess"], "execute", "SetComplete→SetSess(exec)")
    _connect(pins["setsess"], "then", pins["save"], "execute", "SetSess→Save(exec)")

    # 编译保存
    try:
        ok = unreal.BlueprintEditorLibrary.compile_blueprint(bp)
        _log(f"  compile -> {ok}")
        unreal.EditorAssetLibrary.save_asset(asset_path, only_if_is_dirty=True)
        _log("  save OK")
        return True
    except Exception as e:
        _log(f"  compile/save ERR: {type(e).__name__} {e}")
        return False


def main():
    import unreal
    import os
    sel = os.environ.get("C5_BUILD_GROUP")
    groups = GROUPS if sel is None else [g for g in GROUPS if g["idx"] == int(sel)]
    _log(f"======== 构建 BP_PoseRecorderC4（groups={[g['idx'] for g in groups]}）========")

    ok_count = 0
    for group in groups:
        try:
            if _build_group(group):
                ok_count += 1
        except Exception as e:
            _log(f"  G{group['idx']} 构建异常: {type(e).__name__} {e}")

    _log(f"\n完成 {ok_count}/{len(groups)} 个 BP")
    _flush()


if __name__ == "__main__":
    main()
    print("\n脚本已执行，结果写入 build_bp_recorder_c4.log")