"""最小诊断：验证 UE Python 能否做基本蓝图操作（create + 变量 + compile）。

用于排查 C4 构建闪退是「编辑器会话/项目级问题」还是「脚本规模问题」。

用法（Unreal Editor Python Console）：
    py "D:/projects/FustalMOT_UEDataset/Content/FutsalMOT/code/ue/diag_min_bp.py"
"""

import sys
from pathlib import Path

LOG = []
LOG_PATH = Path(r"D:\projects\FustalMOT_UEDataset\Content\FutsalMOT\code\.futsalmot\diag_min_bp.log")


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
    _log("======== 最小蓝图操作诊断 ========")

    # 1. 基本库调用
    _log("[1] import unreal OK")
    try:
        ed = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
        _log(f"[2] get_editor_subsystem OK: {ed}")
    except Exception as e:
        _log(f"[2] get_editor_subsystem ERR: {type(e).__name__} {e}")

    # 2. 创建临时测试蓝图
    test_path = "/Game/FutsalMOT/Blueprints/BP_DiagTest_TMP"
    try:
        if unreal.EditorAssetLibrary.does_asset_exist(test_path):
            unreal.EditorAssetLibrary.delete_asset(test_path)
        bp = unreal.BlueprintEditorLibrary.create_blueprint_asset_with_parent(test_path, unreal.Actor)
        _log(f"[3] create_blueprint OK: {bp}")
    except Exception as e:
        _log(f"[3] create_blueprint ERR: {type(e).__name__} {e}")

    # 3. 加一个变量
    try:
        bl = unreal.BlueprintEditorLibrary
        int_type = bl.get_basic_type_by_name("int")
        ok = bl.add_member_variable(bp, "DiagInt", int_type)
        _log(f"[4] add_member_variable: {ok}")
    except Exception as e:
        _log(f"[4] add_member_variable ERR: {type(e).__name__} {e}")

    # 4. 编译 + 保存
    try:
        ok = unreal.BlueprintEditorLibrary.compile_blueprint(bp)
        _log(f"[5] compile: {ok}")
    except Exception as e:
        _log(f"[5] compile ERR: {type(e).__name__} {e}")

    # 5. 清理测试蓝图
    try:
        unreal.EditorAssetLibrary.delete_asset(test_path)
        _log("[6] 清理测试蓝图 OK")
    except Exception as e:
        _log(f"[6] 清理 ERR: {type(e).__name__} {e}")

    _log("======== 诊断完成 ========")
    _flush()


if __name__ == "__main__":
    main()
    print("\n脚本已执行，结果写入 diag_min_bp.log")