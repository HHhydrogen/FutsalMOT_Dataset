"""外科修复：给现有 BP_PoseRecorderC4_G0..G4 的 GetSocketLocation/Rotation 节点补 InSocketName。

背景：build_bp_recorder_c4.py 漏设 insocketname → 全部骨骼返回组件位置（退化数据）。
本脚本就地修改既有 BP（不 delete、不 rebuild），按创建顺序把 13 骨名写到 socket 节点。

节点顺序（已验证）：list_all_nodes 顺序 = 创建顺序；每 actor 13 骨，loc/rot 交替；
每 BP 共 2 actor → 26 个 GetSocketLocation + 26 个 GetSocketRotation。
第 k 个 socket 节点（0 基）→ actor=k//13, bone=BONES[k%13]。

用法（Unreal Editor Python，FutsalMOTTools.run_python_file）：
    py ".../fix_socket_pins_c4.py"
"""

from pathlib import Path

LOG = []
LOG_PATH = Path(r"D:\projects\FustalMOT_UEDataset\Content\FutsalMOT\code\.futsalmot\fix_socket_pins_c4.log")
BONES = [
    "head",
    "upperarm_l", "upperarm_r", "lowerarm_l", "lowerarm_r",
    "hand_l", "hand_r", "thigh_l", "thigh_r",
    "calf_l", "calf_r", "foot_l", "foot_r",
]
GROUPS = [f"BP_PoseRecorderC4_G{i}" for i in range(5)]


def _log(msg):
    print(msg)
    LOG.append(str(msg))


def _flush():
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        LOG_PATH.write_text("\n".join(LOG), encoding="utf-8")
    except Exception as e:
        print(f"  写日志失败: {e}")


def _socket_nodes(ed):
    """返回 (loc_nodes, rot_nodes)，按 list_all_nodes 顺序。"""
    loc_nodes, rot_nodes = [], []
    for n in ed.list_all_nodes():
        if n.get_class().get_name() != "K2Node_CallFunction":
            continue
        title = "?"
        try:
            title = str(n.get_node_title())
        except Exception:
            pass
        if title == "GetSocketLocation":
            loc_nodes.append(n)
        elif title == "GetSocketRotation":
            rot_nodes.append(n)
    return loc_nodes, rot_nodes


def _set_socket_names(ed, nodes, kind, bp_name):
    ok = 0
    for k, n in enumerate(nodes):
        bone = BONES[k % len(BONES)]
        pin = None
        for p in n.list_all_pins():
            if str(p.get_pin_name()).lower() == "insocketname":
                pin = p
                break
        if pin is None:
            _log(f"  [{bp_name}] {kind}[{k}] 无 InSocketName pin!")
            continue
        try:
            pin.set_pin_value(bone)
            ok += 1
        except Exception as e:
            _log(f"  [{bp_name}] {kind}[{k}] set {bone} ERR: {type(e).__name__} {e}")
    return ok


def _verify(ed, nodes, bp_name):
    bad = []
    for k, n in enumerate(nodes):
        for p in n.list_all_pins():
            if str(p.get_pin_name()).lower() == "insocketname":
                try:
                    val = p.get_default_value()
                except Exception:
                    val = "?"
                if not val or val == "None":
                    bad.append(k)
                break
    return bad


def main():
    import unreal
    _log("======== 修复 G0..G4 socket pin ========")

    for name in GROUPS:
        bp = unreal.load_asset(f"/Game/FutsalMOT/Blueprints/{name}")
        if bp is None:
            _log(f"  {name} 加载失败")
            continue
        ed = unreal.BlueprintGraphEditor.get_graph_editor_by_name(bp, "CaptureOutputFrame")
        if ed is None:
            _log(f"  {name} 无 CaptureOutputFrame 图")
            continue
        loc_nodes, rot_nodes = _socket_nodes(ed)
        _log(f"[{name}] loc={len(loc_nodes)} rot={len(rot_nodes)}")
        ok_loc = _set_socket_names(ed, loc_nodes, "loc", name)
        ok_rot = _set_socket_names(ed, rot_nodes, "rot", name)
        bad_loc = _verify(ed, loc_nodes, name)
        bad_rot = _verify(ed, rot_nodes, name)
        _log(f"[{name}] 设置 loc={ok_loc}/26 rot={ok_rot}/26 校验空值={len(bad_loc)+len(bad_rot)}")
        try:
            comp = unreal.BlueprintEditorLibrary.compile_blueprint(bp)
            saved = unreal.EditorAssetLibrary.save_asset(f"/Game/FutsalMOT/Blueprints/{name}", only_if_is_dirty=True)
            _log(f"[{name}] compile={comp} save={saved}")
        except Exception as e:
            _log(f"[{name}] compile/save ERR: {type(e).__name__} {e}")

    _flush()


if __name__ == "__main__":
    main()