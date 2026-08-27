"""诊断：列出 G0 CaptureOutputFrame 图中所有带 insocketname 的节点及顺序。"""

from pathlib import Path

LOG = []
LOG_PATH = Path(r"D:\projects\FustalMOT_UEDataset\Content\FutsalMOT\code\.futsalmot\diag_socket_nodes.log")


def _log(msg):
    print(msg)
    LOG.append(str(msg))


def _flush():
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        LOG_PATH.write_text("\n".join(LOG), encoding="utf-8")
    except Exception as e:
        print(f"  写日志失败: {e}")


def main():
    import unreal
    _log("======== 诊断 G0 socket 节点 ========")
    bp = unreal.load_asset("/Game/FutsalMOT/Blueprints/BP_PoseRecorderC4_G0")
    ed = unreal.BlueprintGraphEditor.get_graph_editor_by_name(bp, "CaptureOutputFrame")
    nodes = ed.list_all_nodes()
    _log(f"total nodes = {len(nodes)}")

    sock = []
    for i, n in enumerate(nodes):
        cls = n.get_class().get_name()
        if cls != "K2Node_CallFunction":
            continue
        pins = {str(p.get_pin_name()).lower(): p for p in n.list_all_pins()}
        if "insocketname" in pins:
            title = "?"
            try:
                title = str(n.get_node_title())
            except Exception:
                pass
            pos = n.get_node_pos()
            sock.append((i, title, pos.x, pos.y))
    _log(f"insocketname 节点数 = {len(sock)}")
    for s in sock[:56]:
        _log(f"  idx={s[0]} title={s[1]} pos=({int(s[2])},{int(s[3])})")

    _flush()


if __name__ == "__main__":
    main()