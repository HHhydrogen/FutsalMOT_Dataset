"""诊断2：定位 C4 崩溃点——函数图节点规模 vs compile。

分级测试：创建函数图 → 加 N 个 Array_Add → 连 exec 链 → compile。
从 N=50 开始递增，打印每步结果，找到崩溃临界点。

用法（Unreal Editor Python Console）：
    py "D:/projects/FustalMOT_UEDataset/Content/FutsalMOT/code/ue/diag_scale_bp.py"
"""

import sys
from pathlib import Path

LOG = []
LOG_PATH = Path(r"D:\projects\FustalMOT_UEDataset\Content\FutsalMOT\code\.futsalmot\diag_scale_bp.log")
ARRAY_ADD = "/Script/Engine.KismetArrayLibrary.Array_Add"


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


def _connect(a_pins, a_name, b_pins, b_name):
    a = a_pins.get(a_name)
    b = b_pins.get(b_name)
    if a is None or b is None:
        return False
    try:
        return bool(a.try_create_connection(b))
    except Exception:
        return False


def main():
    import unreal
    _log("======== 分级诊断：函数图节点规模 ========")

    test_path = "/Game/FutsalMOT/Blueprints/BP_DiagScale_TMP"
    if unreal.EditorAssetLibrary.does_asset_exist(test_path):
        unreal.EditorAssetLibrary.delete_asset(test_path)
    bp = unreal.BlueprintEditorLibrary.create_blueprint_asset_with_parent(test_path, unreal.Actor)
    _log(f"[1] 创建 BP: {bp}")

    bl = unreal.BlueprintEditorLibrary
    int_type = bl.get_basic_type_by_name("int")
    arr_int = bl.get_array_type(int_type)
    bl.add_member_variable(bp, "DiagArray", arr_int)

    # 建函数图
    fn_editor = unreal.BlueprintGraphEditor.create_and_edit_function_graph(bp, "DiagFunc")
    _log(f"[2] 函数图: {fn_editor}")
    root_param = fn_editor.add_graph_input_parameter("N", int_type)
    fn_editor.add_graph_input_parameter("V", int_type)

    # 建数组 get 节点
    getarr = fn_editor.add_get_member_variable_node("DiagArray", "")
    getarr_p = _pins(getarr)
    _log(f"[3] get 节点 pins: {sorted(getarr_p)}")

    # 逐级加 Array_Add
    for n in [50, 100, 150, 200, 250, 300]:
        _log(f"[4] 测试 N={n} ...")
        # 建 n 个 Array_Add
        adds = []
        for i in range(n):
            nd = fn_editor.add_call_function_node(ARRAY_ADD)
            adds.append(nd)
        # 连 targetarray
        ok_target = 0
        for nd in adds:
            p = _pins(nd)
            if _connect(getarr_p, "diagarray", p, "targetarray"):
                ok_target += 1
        # 连 exec 链 + newitem
        prev_p = None
        prev_name = None
        entry_pin = fn_editor.find_graph_entry_pin()
        ok_exec = 0
        for i, nd in enumerate(adds):
            p = _pins(nd)
            if i == 0:
                if prev_p is None:
                    try:
                        entry_pin.try_create_connection(p["execute"])
                        prev_p, prev_name = p, "then"
                        ok_exec += 1
                    except Exception:
                        pass
            else:
                if _connect(prev_p, prev_name, p, "execute"):
                    ok_exec += 1
                    prev_p, prev_name = p, "then"
        _log(f"  N={n}: target 连 {ok_target}/{n}, exec 连 {ok_exec}/{n}")

        # 编译（每级都编译，测 compile 稳定性）
        try:
            ok = unreal.BlueprintEditorLibrary.compile_blueprint(bp)
            _log(f"  N={n}: compile -> {ok}")
        except Exception as e:
            _log(f"  N={n}: compile ERR {type(e).__name__}")
        _log(f"  N={n}: 完成（未崩）")

    # 清理
    try:
        unreal.EditorAssetLibrary.delete_asset(test_path)
        _log("[5] 清理 OK")
    except Exception as e:
        _log(f"[5] 清理 ERR: {type(e).__name__}")

    _log("======== 诊断完成（到 N=300 未崩，说明节点/连接不是主因）========")
    _flush()


if __name__ == "__main__":
    main()
    print("\n脚本已执行，结果写入 diag_scale_bp.log")