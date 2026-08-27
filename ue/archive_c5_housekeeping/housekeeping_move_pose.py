"""Housekeeping：整理正式 Pose 资产到统一目录（仅移动/重命名，不改 BP 逻辑）。

移动（AssetTools.rename_asset，创建 redirector + 修复已加载引用）：
  正式：
    Blueprints/BP_PoseRecorderC4_G0..G4  -> Blueprints/Pose/Recorder/
    Blueprints/WBP_PoseMRQBurnInC4        -> Blueprints/Pose/MRQ/
    Blueprints/SG_PoseCapture             -> Blueprints/Pose/SaveGame/
  Legacy（标记 _Legacy）：
    Blueprints/BP_PoseRecorder            -> Blueprints/Pose/Recorder/BP_PoseRecorder_Legacy
    Blueprints/WBP_PoseMRQBurnIn          -> Blueprints/Pose/MRQ/WBP_PoseMRQBurnIn_Legacy
"""

from pathlib import Path

LOG = []
LOG_PATH = Path(r"D:\projects\FustalMOT_UEDataset\Content\FutsalMOT\code\.futsalmot\housekeeping_move_pose.log")

MOVES = []
for i in range(5):
    MOVES.append((f"/Game/FutsalMOT/Blueprints/BP_PoseRecorderC4_G{i}",
                  f"/Game/FutsalMOT/Blueprints/Pose/Recorder/BP_PoseRecorderC4_G{i}"))
MOVES += [
    ("/Game/FutsalMOT/Blueprints/WBP_PoseMRQBurnInC4", "/Game/FutsalMOT/Blueprints/Pose/MRQ/WBP_PoseMRQBurnInC4"),
    ("/Game/FutsalMOT/Blueprints/SG_PoseCapture", "/Game/FutsalMOT/Blueprints/Pose/SaveGame/SG_PoseCapture"),
    ("/Game/FutsalMOT/Blueprints/BP_PoseRecorder", "/Game/FutsalMOT/Blueprints/Pose/Recorder/BP_PoseRecorder_Legacy"),
    ("/Game/FutsalMOT/Blueprints/WBP_PoseMRQBurnIn", "/Game/FutsalMOT/Blueprints/Pose/MRQ/WBP_PoseMRQBurnIn_Legacy"),
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
    _log("======== Housekeeping：移动正式 Pose 资产 ========")
    ok_all = True
    for src, dst in MOVES:
        try:
            ok = unreal.EditorAssetLibrary.rename_asset(src, dst)
            _log(f"  {src} -> {dst}: {ok}")
            ok_all = ok_all and bool(ok)
        except Exception as e:
            _log(f"  {src} -> {dst} ERR: {type(e).__name__} {e}")
            ok_all = False
    _log(f"移动完成 ok_all={ok_all}")
    _flush()


if __name__ == "__main__":
    main()