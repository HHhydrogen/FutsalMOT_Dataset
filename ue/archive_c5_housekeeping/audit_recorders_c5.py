"""C5.2 最终收尾：审计重建的 G0..G4（只读，不修改）。

逐组检查：
- compile=True
- 生命周期变量在位（CaptureSessionActive/SaveSlotName/SessionId），无 v3 变量
- CaptureOutputFrame：首行 SetActive=True、GetSocketLocation 数量=26、无 v3 数学节点
- EndPlay：Branch guard、SaveSlotName→动态 slot、SessionId→SG、SG 新路径、无 v3 数学节点
"""

from pathlib import Path

LOG = []
LOG_PATH = Path(r"D:\projects\FustalMOT_UEDataset\Content\FutsalMOT\code\.futsalmot\audit_recorders_c5.log")
NEW = [f"/Game/FutsalMOT/Blueprints/Pose/Recorder/BP_PoseRecorderC4_G{i}" for i in range(5)]
EXPECT_BONES = ["head", "upperarm_l", "upperarm_r", "lowerarm_l", "lowerarm_r",
                "hand_l", "hand_r", "thigh_l", "thigh_r", "calf_l", "calf_r", "foot_l", "foot_r"]
LIFECYCLE = ["CaptureSessionActive", "SaveSlotName", "SessionId"]
V3_VARS = ["CapturedFrameCount", "FirstRootFrame", "LastRootFrame", "ExpectedFrameCount"]
MATH_TITLES = ("equal", "add (int)", "and (boolean)", "greater", "make literal")
SG_NEW = "/Game/FutsalMOT/Blueprints/Pose/SaveGame/SG_PoseCapture"


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


def _cls(n):
    try:
        return n.get_class().get_name()
    except Exception:
        return "?"


def _pins(n):
    d = {}
    try:
        for p in n.list_all_pins():
            d[str(p.get_pin_name()).lower()] = p
    except Exception:
        pass
    return d


def _pin_default(node, name):
    try:
        for p in node.list_all_pins():
            if str(p.get_pin_name()).lower() == name:
                return str(p.get_default_value())
    except Exception:
        pass
    return None


def _audit(idx):
    import unreal
    p = NEW[idx]
    bp = unreal.load_asset(p)
    out = []
    out.append(f"\n===== G{idx} 审计 =====")
    ok_all = True

    # compile
    ok = unreal.BlueprintEditorLibrary.compile_blueprint(bp)
    out.append(f"  compile = {ok}")
    ok_all = ok_all and bool(ok)

    # 变量
    names = unreal.BlueprintEditorLibrary.list_member_variable_names(bp)
    lc = [v for v in LIFECYCLE if v in names]
    v3v = [v for v in V3_VARS if v in names]
    out.append(f"  lifecycle_vars = {lc}")
    out.append(f"  v3_vars = {v3v if v3v else 'none'}")
    ok_all = ok_all and len(lc) == 3 and not v3v

    # fn 图
    fn_ed = unreal.BlueprintGraphEditor.get_graph_editor_by_name(bp, "CaptureOutputFrame")
    setact = 0
    sock = {}
    math_fn = []
    for n in fn_ed.list_all_nodes():
        c = _cls(n)
        t = _title(n)
        if c == "K2Node_VariableSet" and "capturesessionactive" in _pins(n):
            setact += 1
        if c == "K2Node_CallFunction" and "insocketname" in _pins(n):
            # socket 名来自连接的 MakeLiteralString.Value
            name = ""
            try:
                for p in n.list_all_pins():
                    if str(p.get_pin_name()).lower() == "insocketname":
                        for lp in p.list_connected_pins():
                            owner = lp.get_owner()
                            if "MakeLiteralString" in str(owner.get_node_title()):
                                for pp in owner.list_all_pins():
                                    if str(pp.get_pin_name()).lower() == "value":
                                        name = str(pp.get_default_value())
            except Exception:
                pass
            sock[name] = sock.get(name, 0) + 1
        if c == "K2Node_CallFunction" and any(k in t.lower() for k in MATH_TITLES):
            math_fn.append(t)
    out.append(f"  fn: SetActive_count={setact} (期望 1)")
    out.append(f"  fn: GetSocketLocation 数量={len(sock)} (期望 13 骨)")
    bones_ok = sorted(sock.keys()) == sorted(EXPECT_BONES)
    out.append(f"  fn: socket 名完整={bones_ok} ({sorted(sock.keys())})")
    out.append(f"  fn: v3 数学节点={math_fn if math_fn else 'none'}")
    ok_all = ok_all and setact == 1 and bones_ok and not math_fn

    # ep 图
    eg = unreal.BlueprintEditorLibrary.find_event_graph(bp)
    ed = unreal.BlueprintGraphEditor.get_graph_editor(eg)
    branch = 0
    math_ep = []
    getslot = 0
    getsess = 0
    setcomplete = 0
    setsess = 0
    sg_paths = set()
    for n in ed.list_all_nodes():
        c = _cls(n)
        t = _title(n)
        pins = _pins(n)
        if c == "K2Node_IfThenElse":
            branch += 1
        if c == "K2Node_VariableGet" and "saveslotname" in pins:
            getslot += 1
        if c == "K2Node_VariableGet" and "sessionid" in pins:
            getsess += 1
        if c == "K2Node_VariableSet" and "capture_complete" in pins:
            setcomplete += 1
        if c == "K2Node_VariableSet" and "session_id" in pins:
            setsess += 1
        if c == "K2Node_CallFunction" and any(k in t.lower() for k in MATH_TITLES):
            math_ep.append(t)
        if c == "K2Node_CallFunction" and "savegameclass" in pins:
            v = _pin_default(n, "savegameclass")
            if v and "SG_PoseCapture" in v:
                sg_paths.add(v)
        if c == "K2Node_VariableSet" and any(k in pins for k in ("capture_complete", "session_id", "total_samples")):
            for pin in n.list_all_pins():
                try:
                    linked = pin.list_connected_pins()
                    for lp in linked:
                        s = str(lp.get_owner().get_path_name())
                        if "SG_PoseCapture" in s:
                            sg_paths.add(s)
                except Exception:
                    pass
    out.append(f"  ep: Branch(guard)={branch} (期望 1)")
    out.append(f"  ep: Get SaveSlotName={getslot} (期望 1, 动态 slot)")
    out.append(f"  ep: Get SessionId={getsess} (期望 1)")
    out.append(f"  ep: Set capture_complete={setcomplete} (期望 1)")
    out.append(f"  ep: Set session_id={setsess} (期望 1)")
    out.append(f"  ep: SG 路径={sg_paths}")
    out.append(f"  ep: v3 数学节点={math_ep if math_ep else 'none'}")
    sg_ok = any(SG_NEW in s for s in sg_paths)
    ok_all = ok_all and branch == 1 and getslot == 1 and getsess == 1 and setcomplete == 1 and setsess == 1 and sg_ok and not math_ep

    out.append(f"  ==> G{idx} {'PASS' if ok_all else 'FAIL'}")
    return out, ok_all


def main():
    all_ok = True
    for i in range(5):
        try:
            lines, ok = _audit(i)
            for l in lines:
                _log(l)
            all_ok = all_ok and ok
        except Exception as e:
            import traceback
            _log(f"G{i} 审计异常: {type(e).__name__} {e}")
            _log(traceback.format_exc())
            all_ok = False
    _log(f"\n======== 总结果: {'ALL PASS' if all_ok else 'HAS FAIL'} ========")
    _flush()


if __name__ == "__main__":
    main()