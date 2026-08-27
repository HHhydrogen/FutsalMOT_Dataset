"""C5.2-B 正式 Pose 渲染入口：读取 resolved task → prep Recorder → 提交 RGB+BurnIn 渲染。

不再需要手工串 mrq_run_c4.py。分辨率来自 resolved task（render_rgb，C5.1 唯一源）。
每帧同步 Pose 捕获（BurnIn → WBP_PoseMRQBurnInC4 → 5×CaptureOutputFrame）。

env:
  C5_POSE_TASK    resolved task JSON 路径（必须）
  C5_POSE_FRAMES  覆盖帧数（可选；缺省用 3）
  C5_POSE_DIR     输出目录（可选）

用法（Unreal Editor Python，FutsalMOTTools.run_python_file）：
    py ".../pose_render.py"
"""

import json
import os
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from render_preset import resolve_output_resolution  # noqa: E402

LOG = []
LOG_PATH = Path(r"D:\projects\FustalMOT_UEDataset\Content\FutsalMOT\code\.futsalmot\pose_render.log")
SEQ_ASSET = "/Game/FutsalMOT/Sequences/LS_Cam_01.LS_Cam_01"
SEQ_LOAD = "/Game/FutsalMOT/Sequences/LS_Cam_01"
MAP = "/Game/FutsalMOT/Maps/L_FutsalCourt"
BURN_IN_CLASS = "/Game/FutsalMOT/Blueprints/WBP_PoseMRQBurnInC4.WBP_PoseMRQBurnInC4_C"
RECORDER_BPS = [f"/Game/FutsalMOT/Blueprints/BP_PoseRecorderC4_G{i}" for i in range(5)]


def _log(msg):
    print(msg)
    LOG.append(str(msg))


def _flush():
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        LOG_PATH.write_text("\n".join(LOG), encoding="utf-8")
    except Exception as e:
        print(f"  写日志失败: {e}")


def _prep_recorders(episode_id, camera):
    """设置 5 个 C4 Recorder 的 CDO 默认值（动态 slot）+ 重放实例。"""
    import unreal
    ed = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    for a in list(ed.get_all_level_actors()):
        try:
            if a.get_actor_label().startswith("BP_PoseRecorderC4_G"):
                ed.destroy_actor(a)
        except Exception:
            pass
    for i, bp_path in enumerate(RECORDER_BPS):
        bp = unreal.load_asset(bp_path)
        gc = unreal.BlueprintEditorLibrary.generated_class(bp)
        cdo = unreal.get_default_object(gc)
        cdo.set_editor_property("saveslotname", f"PoseCapture_{episode_id}_{camera}_G{i}")
        cdo.set_editor_property("sessionid", f"{episode_id}_{camera}")
        unreal.EditorAssetLibrary.save_asset(bp_path, only_if_is_dirty=True)
        a = ed.spawn_actor_from_class(gc, unreal.Vector(0, 0, 0), unreal.Rotator(0, 0, 0))
        a.set_actor_label(f"BP_PoseRecorderC4_G{i}")
    _log(f"  recorder 已 prep：slots PoseCapture_{episode_id}_{camera}_G0..G4")


def main():
    import unreal
    rt_path = os.environ.get("C5_POSE_TASK")
    if not rt_path:
        _log("  ERROR: C5_POSE_TASK 未设置")
        _flush()
        return
    rt = json.loads(Path(rt_path).read_text(encoding="utf-8"))
    ann_cfg = rt["ue_profile"].get("annotation_export") or {}
    w, h = resolve_output_resolution(ann_cfg)
    episode_id = rt.get("episode_name") or "episode"
    camera = (ann_cfg.get("cameras") or ["CineCam_01"])[0]
    frames = int(os.environ.get("C5_POSE_FRAMES", "3"))
    out_dir = Path(os.environ.get("C5_POSE_DIR", rf"D:\projects\FustalMOT_UEDataset\Content\FutsalMOT\code\.futsalmot\pose_render_{episode_id}"))
    _log(f"======== C5.2-B pose_render：{episode_id} {w}x{h} frames={frames} camera={camera} ========")

    # C5.2 Finalization：preflight assert（不允许依赖上次 CDO 残留）
    session_id = f"{episode_id}_{camera}"
    slot_names = [f"PoseCapture_{episode_id}_{camera}_G{i}" for i in range(5)]
    problems = []
    if not session_id:
        problems.append("session_id 为空")
    if frames <= 0:
        problems.append(f"expected_frame_count({frames}) <= 0")
    if len(set(slot_names)) != 5:
        problems.append("slot names 不唯一")
    for s in slot_names:
        if episode_id not in s or camera not in s:
            problems.append(f"slot {s} 未包含 episode/camera")
    if problems:
        _log(f"  FAIL preflight: {'; '.join(problems)} —— 禁止提交 Pose MRQ")
        _flush()
        return
    _log(f"  preflight PASS: session={session_id} expected={frames} slots={slot_names}")

    _prep_recorders(episode_id, camera)

    # 清空动态 slot
    for i in range(5):
        try:
            unreal.GameplayStatics.delete_game_in_slot(f"PoseCapture_{episode_id}_{camera}_G{i}", 0)
        except Exception:
            pass
    _log("  已清空动态 slot")

    import shutil
    if out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    seq = unreal.load_asset(SEQ_LOAD)
    seq.set_playback_start(0)
    seq.set_playback_end(frames)

    sub = None
    for cls_name in ("MoviePipelineQueueSubsystem", "MovieRenderQueueSubsystem"):
        cls = getattr(unreal, cls_name, None)
        if cls is not None:
            try:
                sub = unreal.get_editor_subsystem(cls)
                break
            except Exception:
                continue
    if sub is None:
        _log("  ERROR: 无 MRQ 子系统")
        _flush()
        return
    queue = sub.get_queue()
    try:
        queue.delete_all_jobs()
    except Exception:
        pass
    job = queue.allocate_new_job(unreal.MoviePipelineExecutorJob)
    job.sequence = unreal.SoftObjectPath(SEQ_ASSET)
    job.map = unreal.SoftObjectPath(MAP)
    try:
        config = job.get_configuration()
    except Exception:
        config = None
    if config is None:
        config = unreal.MoviePipelinePrimaryConfig()
    output = config.find_or_add_setting_by_class(unreal.MoviePipelineOutputSetting)
    output.output_directory = unreal.DirectoryPath(str(out_dir))
    output.file_name_format = "{frame_number}"
    output.zero_pad_frame_numbers = 4
    output.output_resolution = unreal.IntPoint(w, h)
    try:
        output.frame_rate = unreal.FrameRate(30, 1)
    except Exception:
        pass
    config.find_or_add_setting_by_class(unreal.MoviePipelineImageSequenceOutput_PNG)
    config.find_or_add_setting_by_class(unreal.MoviePipelineDeferredPassBase)
    try:
        b = config.find_or_add_setting_by_class(unreal.MoviePipelineBurnInSetting)
        b.set_editor_property("burn_in_class", unreal.SoftClassPath(BURN_IN_CLASS))
        b.set_editor_property("composite_onto_final_image", False)
    except Exception as e:
        _log(f"  BurnIn ERR: {type(e).__name__} {e}")
    try:
        job.set_configuration(config)
    except Exception:
        try:
            job.configuration = config
        except Exception:
            pass

    executor = None
    for cls_name in ("MoviePipelinePIEExecutor", "MoviePipelineEditorExecutor"):
        cls = getattr(unreal, cls_name, None)
        if cls is None:
            continue
        try:
            executor = cls()
            break
        except Exception:
            continue
    if executor is None:
        _log("  ERROR: 无 executor")
        _flush()
        return
    submitted = False
    for m_name in ("render_queue_with_executor_instance", "render_queue_with_executor"):
        m = getattr(sub, m_name, None)
        if m is None:
            continue
        try:
            m(executor)
            submitted = True
            _log(f"  已提交（{m_name}）")
            break
        except Exception as e:
            _log(f"  {m_name} ERR: {type(e).__name__} {e}")
    if not submitted:
        _log("  ERROR: 提交失败")
    _log(f"  异步渲染 {frames} 帧（Pose RGB）...")
    _flush()


if __name__ == "__main__":
    main()