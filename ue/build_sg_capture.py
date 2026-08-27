"""增量更新 SG_PoseCapture：添加缺失变量，不删除（避免引用崩溃）。

用法（Unreal Editor Python Console）：
    py "D:/projects/FustalMOT_UEDataset/Content/FutsalMOT/code/ue/build_sg_capture.py"
"""

import sys
from pathlib import Path

LOG = []
LOG_PATH = Path(r"D:\projects\FustalMOT_UEDataset\Content\FutsalMOT\code\.futsalmot\build_sg_capture.log")
ASSET_PATH = "/Game/FutsalMOT/Blueprints/SG_PoseCapture"


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
    _log("======== 增量更新 SG_PoseCapture ========")

    # 加载（不删除）
    sg = unreal.EditorAssetLibrary.load_asset(ASSET_PATH)
    _log(f"[1] load SG -> {sg}")
    if sg is None:
        _log("  ERROR: 无法加载，尝试创建")
        sg = unreal.BlueprintEditorLibrary.create_blueprint_asset_with_parent(ASSET_PATH, unreal.SaveGame)
        _log(f"  create -> {sg}")
    if sg is None:
        _flush()
        return

    bl = unreal.BlueprintEditorLibrary
    int_type = bl.get_basic_type_by_name("int")
    real_type = bl.get_basic_type_by_name("real")
    arr_int = bl.get_array_type(int_type)
    arr_float = bl.get_array_type(real_type)
    str_type = bl.get_basic_type_by_name("string")
    arr_str = bl.get_array_type(str_type)
    vec_struct = unreal.Vector.static_struct()
    vec_type = bl.get_struct_type(vec_struct)
    arr_vec = bl.get_array_type(vec_type)
    rot_struct = unreal.Rotator.static_struct()
    rot_type = bl.get_struct_type(rot_struct)
    arr_rot = bl.get_array_type(rot_type)

    # 添加缺失变量（已存在会失败，打印即可，不崩溃）
    want = [
        ("fps", real_type, "float"),
        ("first_capture_game_time", real_type, "float"),
        ("total_samples", int_type, "int"),
        ("capture_indices", arr_int, "int[]"),
        ("shot_frames", arr_int, "int[]"),
        ("game_times", arr_float, "float[]"),
        ("actor_ids", arr_str, "string[]"),
        ("bone_names", arr_str, "string[]"),
        ("world_locations", arr_vec, "Vector[]"),
        ("world_rotations", arr_rot, "Rotator[]"),
        ("capture_durations", arr_float, "float[]"),
    ]
    for name, pt, label in want:
        try:
            ok = bl.add_member_variable(sg, name, pt)
            _log(f"  [var] {label}.{name} -> {ok} (False=已存在)")
        except Exception as e:
            _log(f"  [var] {label}.{name} ERR: {type(e).__name__}")

    # 编译保存
    try:
        ok = bl.compile_blueprint(sg)
        _log(f"[2] compile -> {ok}")
        unreal.EditorAssetLibrary.save_asset(ASSET_PATH, only_if_is_dirty=True)
        _log("[3] save OK")
    except Exception as e:
        _log(f"[3] compile/save ERR: {type(e).__name__} {e}")

    _flush()


if __name__ == "__main__":
    main()
    print("\n脚本已执行，结果写入 build_sg_capture.log")