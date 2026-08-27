"""C5.2：把 SG_PoseCapture 完整性元数据字段标记为 InstanceEditable。

Python 需要能在 SaveGame 实例上 set_editor_property（pose_capture_export 回写完整性）。
Recorder 的 BP Set 节点不受此限制；但 Python set_editor_property 在实例上要求 InstanceEditable。

用法（Unreal Editor Python，FutsalMOTTools.run_python_file）：
    py ".../mark_sg_instance_editable.py"
"""

from pathlib import Path

LOG = []
LOG_PATH = Path(r"D:\projects\FustalMOT_UEDataset\Content\FutsalMOT\code\.futsalmot\mark_sg_instance_editable.log")
SG_ASSET = "/Game/FutsalMOT/Blueprints/Pose/SaveGame/SG_PoseCapture"
FIELDS = [
    "capture_complete", "captured_frame_count", "expected_frame_count",
    "first_root_frame", "last_root_frame", "session_id",
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
    _log("======== 标记 SG_PoseCapture 元数据字段 InstanceEditable ========")
    bp = unreal.load_asset(SG_ASSET)
    for f in FIELDS:
        try:
            unreal.BlueprintEditorLibrary.set_blueprint_variable_instance_editable(bp, f, True)
            _log(f"  {f}: instance_editable=True")
        except Exception as e:
            _log(f"  {f} ERR: {type(e).__name__} {e}")
    try:
        ok = unreal.BlueprintEditorLibrary.compile_blueprint(bp)
        _log(f"  compile -> {ok}")
        unreal.EditorAssetLibrary.save_asset(SG_ASSET, only_if_is_dirty=True)
        _log("  save OK")
    except Exception as e:
        _log(f"  compile/save ERR: {type(e).__name__} {e}")
    _flush()


if __name__ == "__main__":
    main()