"""Housekeeping：删除实验资产 BP_PoseRecorder_Proto + 验证移动结果。

- 删除（已确认 0 引用）
- 验证新路径存在、旧路径为 redirector
- 验证关卡中 G0..G4 Actor 类路径已指向新位置
- 编译所有移动的 BP
"""

from pathlib import Path

LOG = []
LOG_PATH = Path(r"D:\projects\FustalMOT_UEDataset\Content\FutsalMOT\code\.futsalmot\housekeeping_verify.log")

NEW = [f"/Game/FutsalMOT/Blueprints/Pose/Recorder/BP_PoseRecorderC4_G{i}" for i in range(5)] + [
    "/Game/FutsalMOT/Blueprints/Pose/MRQ/WBP_PoseMRQBurnInC4",
    "/Game/FutsalMOT/Blueprints/Pose/MRQ/WBP_PoseMRQBurnIn_Legacy",
    "/Game/FutsalMOT/Blueprints/Pose/SaveGame/SG_PoseCapture",
    "/Game/FutsalMOT/Blueprints/Pose/Recorder/BP_PoseRecorder_Legacy",
]
OLD = [f"/Game/FutsalMOT/Blueprints/BP_PoseRecorderC4_G{i}" for i in range(5)] + [
    "/Game/FutsalMOT/Blueprints/WBP_PoseMRQBurnInC4",
    "/Game/FutsalMOT/Blueprints/WBP_PoseMRQBurnIn",
    "/Game/FutsalMOT/Blueprints/SG_PoseCapture",
    "/Game/FutsalMOT/Blueprints/BP_PoseRecorder",
]


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
    _log("======== Housekeeping 验证 ========")

    # 删除实验资产
    proto = "/Game/FutsalMOT/Blueprints/BP_PoseRecorder_Proto"
    try:
        ok = unreal.EditorAssetLibrary.delete_asset(proto)
        _log(f"  删除 {proto}: {ok}")
    except Exception as e:
        _log(f"  删除 {proto} ERR: {type(e).__name__} {e}")

    # 新路径存在性
    for p in NEW:
        _log(f"  NEW exists={unreal.EditorAssetLibrary.does_asset_exist(p)}  {p}")

    # 旧路径状态（应无资产或为 redirector）
    for p in OLD:
        exists = unreal.EditorAssetLibrary.does_asset_exist(p)
        if exists:
            cls = unreal.EditorAssetLibrary.get_asset_class(p)
            _log(f"  OLD exists={exists} class={cls}  {p}")
        else:
            _log(f"  OLD exists={exists}  {p}")

    # 关卡中 G0..G4 Actor 类路径
    ed = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    for a in ed.get_all_level_actors():
        label = a.get_actor_label()
        if label.startswith("BP_PoseRecorderC4_G"):
            try:
                cls = a.get_class().get_path_name()
            except Exception:
                cls = "?"
            _log(f"  关卡Actor {label}: class={cls}")

    # 编译移动的 BP
    for p in NEW:
        try:
            bp = unreal.load_asset(p)
            if bp is None:
                _log(f"  compile skip (None) {p}")
                continue
            ok = unreal.BlueprintEditorLibrary.compile_blueprint(bp)
            unreal.EditorAssetLibrary.save_asset(p, only_if_is_dirty=True)
            _log(f"  compile {p}: {ok}")
        except Exception as e:
            _log(f"  compile {p} ERR: {type(e).__name__} {e}")
    _flush()


if __name__ == "__main__":
    main()