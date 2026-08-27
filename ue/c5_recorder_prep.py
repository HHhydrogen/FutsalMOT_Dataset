"""C5.2 测试 prep：放置 C4 Recorder（已升级）实例 + 设置动态 SaveSlotName/SessionId。

只 spawn + set 成员，不 delete、不保存关卡。

用法（Unreal Editor Python，FutsalMOTTools.run_python_file）：
    py ".../c5_recorder_prep.py"
"""

from pathlib import Path

LOG = []
LOG_PATH = Path(r"D:\projects\FustalMOT_UEDataset\Content\FutsalMOT\code\.futsalmot\c5_recorder_prep.log")
EPISODE = "c5test"
CAMERA = "CineCam_01"
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


def main():
    import unreal
    import os
    expected = int(os.environ.get("C5_EXPECTED_FRAMES", "3"))
    _log(f"======== C5.2 prep：设 CDO 默认 + 重放 Recorder（{EPISODE}/{CAMERA}）expected={expected} ========")

    # C5.2 Finalization：preflight assert（不依赖上次 CDO 残留）
    session_id = f"{EPISODE}_{CAMERA}"
    slot_names = [f"PoseCapture_{EPISODE}_{CAMERA}_G{i}" for i in range(5)]
    problems = []
    if not session_id:
        problems.append("session_id 为空")
    if expected <= 0:
        problems.append(f"expected_frame_count({expected}) <= 0")
    if len(set(slot_names)) != 5:
        problems.append("slot names 不唯一")
    for s in slot_names:
        if EPISODE not in s or CAMERA not in s:
            problems.append(f"slot {s} 未包含 episode/camera")
    if problems:
        _log(f"  FAIL preflight: {'; '.join(problems)}")
        _flush()
        return
    _log(f"  preflight PASS: session={session_id} expected={expected}")

    # 1) 每个 G BP 的 CDO 默认值（PIE 实例从 CDO 复制）
    for i, name in enumerate(GROUPS):
        bp = unreal.load_asset(f"/Game/FutsalMOT/Blueprints/{name}")
        gc = unreal.BlueprintEditorLibrary.generated_class(bp)
        cdo = unreal.get_default_object(gc)
        cdo.set_editor_property("saveslotname", slot_names[i])
        cdo.set_editor_property("sessionid", session_id)
        unreal.EditorAssetLibrary.save_asset(f"/Game/FutsalMOT/Blueprints/{name}", only_if_is_dirty=True)
        _log(f"  CDO {name}: slot={cdo.get_editor_property('saveslotname')} sess={cdo.get_editor_property('sessionid')}")

    # 2) 销毁旧（升级前/陈旧）实例，重放新类
    ed = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    removed = 0
    for a in list(ed.get_all_level_actors()):
        try:
            if a.get_actor_label().startswith("BP_PoseRecorderC4_G"):
                ed.destroy_actor(a)
                removed += 1
        except Exception:
            pass
    _log(f"  销毁旧实例 {removed}")

    for i, name in enumerate(GROUPS):
        bp = unreal.load_asset(f"/Game/FutsalMOT/Blueprints/{name}")
        cls = unreal.BlueprintEditorLibrary.generated_class(bp)
        a = ed.spawn_actor_from_class(cls, unreal.Vector(0, 0, 0), unreal.Rotator(0, 0, 0))
        a.set_actor_label(name)
        slot = a.get_editor_property("saveslotname")
        _log(f"  放置 {name}: slot={slot}")
    _flush()


if __name__ == "__main__":
    main()