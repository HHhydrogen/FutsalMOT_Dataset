"""C5.2 Finalization：崩溃安全 v1 生命周期升级（对已恢复的 C4 基线资产）。

对每个 BP_PoseRecorderC4_G*：
  1. 成员：CaptureSessionActive(bool) / SaveSlotName(string) / SessionId(string)
  2. CaptureOutputFrame 首行：Set CaptureSessionActive=True（1 个 set 节点）
  3. EndPlay：
     - guard：Branch(active)；false→NO-OP；true→原保存链
     - 动态 slot：SaveSlotName → SaveGameToSlot.slotname
     - 写 capture_complete(=active，粗标记) + session_id
  完整度（captured==expected）由 Python 计算并回写，BP 内不做复杂数学。

env: C5_UPGRADE_GROUP（0..4，缺省全部）

用法（Unreal Editor Python，FutsalMOTTools.run_python_file）：
    py ".../upgrade_recorder_c5.py"
"""

import os
import sys
from pathlib import Path

LOG = []
LOG_PATH = Path(r"D:\projects\FustalMOT_UEDataset\Content\FutsalMOT\code\.futsalmot\upgrade_recorder_c5.log")
ASSET_BASE = "/Game/FutsalMOT/Blueprints/Pose/Recorder/BP_PoseRecorderC4_G"
SG_PATH = "/Game/FutsalMOT/Blueprints/Pose/SaveGame/SG_PoseCapture.SG_PoseCapture_C"


def _log(msg):
    print(msg)
    LOG.append(str(msg))


def _flush():
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        LOG_PATH.write_text("\n".join(LOG), encoding="utf-8")
    except Exception as e:
        print(f"  写日志失败: {e}")


def _title(n):
    try:
        return str(n.get_node_title())
    except Exception:
        return "?"


def _find(ed, sub):
    return [n for n in ed.list_all_nodes() if sub.lower() in _title(n).lower()]


def _pin(node, name):
    try:
        for p in node.list_all_pins():
            if str(p.get_pin_name()).lower() == name:
                return p
    except Exception:
        pass
    return None


def _conn(a_pin, b_pin, label):
    if a_pin is None or b_pin is None:
        _log(f"  [conn] {label}: 缺 pin ({a_pin is not None}, {b_pin is not None})")
        return
    try:
        a_pin.try_create_connection(b_pin)
    except Exception as e:
        _log(f"  [conn] {label} ERR: {type(e).__name__} {e}")


def _break(pin):
    try:
        pin.break_pin_links()
    except Exception:
        pass


def _linked(pin):
    try:
        return pin.list_connected_pins()
    except Exception:
        return []


def _set_default(pin, value, label):
    if pin is None:
        _log(f"  [set] {label}: 无 pin")
        return
    try:
        pin.set_pin_value(value)
    except Exception as e:
        _log(f"  [set] {label} ERR: {type(e).__name__} {e}")


def _upgrade(idx):
    import unreal
    asset_path = f"{ASSET_BASE}{idx}"
    _log(f"\n===== v1 升级 {asset_path} =====")
    if not unreal.EditorAssetLibrary.does_asset_exist(asset_path):
        _log("  资产不存在")
        return False
    bp = unreal.load_asset(asset_path)
    bl = unreal.BlueprintEditorLibrary
    for name, pt in [
        ("CaptureSessionActive", bl.get_basic_type_by_name("bool")),
        ("SaveSlotName", bl.get_basic_type_by_name("string")),
        ("SessionId", bl.get_basic_type_by_name("string")),
    ]:
        try:
            bl.add_member_variable(bp, name, pt)
            _log(f"  [var] {name} ok")
        except Exception as e:
            _log(f"  [var] {name} ERR: {type(e).__name__}")

    # CaptureOutputFrame：入口后插 Set CaptureSessionActive=True
    fn_ed = unreal.BlueprintGraphEditor.get_graph_editor_by_name(bp, "CaptureOutputFrame")
    entries = [n for n in _find(fn_ed, "CaptureOutputFrame") if n.get_class().get_name() == "K2Node_FunctionEntry"]
    if not entries:
        _log("  ERROR: 无 FunctionEntry")
        return False
    entry = entries[0]
    entry_then = _pin(entry, "then")
    targets = _linked(entry_then)
    setact = fn_ed.add_set_member_variable_node("CaptureSessionActive", "")
    _set_default(_pin(setact, "capturesessionactive"), "True", "SetActive")
    _break(entry_then)
    _conn(entry_then, _pin(setact, "execute"), "Entry→SetActive")
    if targets:
        _conn(_pin(setact, "then"), targets[0], "SetActive→next")
    _log("  CaptureOutputFrame SetActive ok")

    # EndPlay：guard + 动态 slot + capture_complete(=active) + session_id
    eg = unreal.BlueprintEditorLibrary.find_event_graph(bp)
    ed = unreal.BlueprintGraphEditor.get_graph_editor(eg)
    endplays = [n for n in _find(ed, "end play") if n.get_class().get_name() == "K2Node_Event"]
    if not endplays:
        _log("  ERROR: 无 EndPlay 事件")
        return False
    endplay = endplays[0]
    endplay_then = _pin(endplay, "then")
    create_targets = _linked(endplay_then)
    getact = ed.add_get_member_variable_node("CaptureSessionActive", "")
    guard = ed.add_branch_node()
    _break(endplay_then)
    _conn(endplay_then, _pin(guard, "execute"), "EndPlay→Guard")
    _conn(_pin(guard, "condition"), _pin(getact, "capturesessionactive"), "Active→Guard")
    if create_targets:
        _conn(_pin(guard, "then"), create_targets[0], "GuardT→Create")

    saves = _find(ed, "SaveGameToSlot")
    creates = _find(ed, "CreateSaveGameObject")
    if not (saves and creates):
        _log("  ERROR: 缺 Save/Create 节点")
        return False
    save, create = saves[0], creates[0]
    getslot = ed.add_get_member_variable_node("SaveSlotName", "")
    _break(_pin(save, "slotname"))
    _conn(_pin(getslot, "saveslotname"), _pin(save, "slotname"), "SaveSlotName→Save.slot")
    getsess = ed.add_get_member_variable_node("SessionId", "")
    setcomplete = ed.add_set_member_variable_node("capture_complete", SG_PATH)
    setsess = ed.add_set_member_variable_node("session_id", SG_PATH)
    _conn(_pin(create, "returnvalue"), _pin(setcomplete, "self"), "Create→Complete.self")
    _conn(_pin(create, "returnvalue"), _pin(setsess, "self"), "Create→Sess.self")
    _conn(_pin(getact, "capturesessionactive"), _pin(setcomplete, "capture_complete"), "Active→SG.complete")
    _conn(_pin(getsess, "sessionid"), _pin(setsess, "session_id"), "SessId→SG.sess")
    # exec：setcomplete → setsess → save
    save_exec = _pin(save, "execute")
    feeders = _linked(save_exec)
    if feeders:
        _break(save_exec)
        _conn(feeders[0], _pin(setcomplete, "execute"), "Feed→SetComplete")
        _conn(_pin(setcomplete, "then"), _pin(setsess, "execute"), "SetComplete→SetSess")
        _conn(_pin(setsess, "then"), save_exec, "SetSess→Save")
    _log("  EndPlay guard/slot/meta ok")

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
    sel = os.environ.get("C5_UPGRADE_GROUP")
    idxs = [int(sel)] if sel is not None else list(range(5))
    _log(f"======== v1 升级 BP_PoseRecorderC4（group={idxs}）========")
    ok = 0
    for i in idxs:
        try:
            if _upgrade(i):
                ok += 1
        except Exception as e:
            import traceback
            _log(f"  G{i} 异常: {type(e).__name__} {e}")
            _log(traceback.format_exc())
    _log(f"完成 {ok}/{len(idxs)}")
    _flush()


if __name__ == "__main__":
    main()